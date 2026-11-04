# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Adapted from torchtitan-npu's graph_trainer ACLGraph implementation.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# (c) Meta Platforms, Inc. and affiliates.
# The original BSD-3-Clause license is retained in
# Third_Party_Open_Source_Software_Notice.
"""Partition a joint FX graph into eager host work and NPU capture regions."""

from __future__ import annotations

from collections import Counter, defaultdict
import logging
import operator
from typing import Any, Iterable

import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.split_module import split_module
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_leaves, tree_map

from ..pass_config import PassConfig
from .base import GraphPass
from .graph_metadata import (
    _native_overload,
    is_host_node as _host_boundary,
    mutated_input_indices as _mutated_input_indices,
    propagate_fake_values,
    tensor_storages as _storages,
    written_nodes,
)
from .parallel.fsdp_pass import _free_tensor_storage
from .aclgraph_runtime import (
    ACLGraphWrapper,
    _PartitionInterpreter,
    _RecaptureCoordinator,
    prepare_aclgraph_runtime,
)

logger = logging.getLogger(__name__)


def _is_npu_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "npu"


def _same_layout(value, first):
    if value.dtype != first.dtype or len(value.shape) != len(first.shape):
        return False
    layout = (*value.shape, *value.stride(), value.storage_offset())
    other = (*first.shape, *first.stride(), first.storage_offset())
    return all(statically_known_true(left == right) for left, right in zip(layout, other, strict=True))


def _staging_aliases(placeholders, input_storages, staged_input_indices):
    storage_inputs = defaultdict(list)
    for i, storages in enumerate(input_storages):
        for storage in storages:
            storage_inputs[storage].append(i)
    input_aliases = {}
    for indices in storage_inputs.values():
        if len(indices) < 2:
            continue
        first = placeholders[indices[0]].meta["val"]
        if all(i in staged_input_indices and _same_layout(placeholders[i].meta["val"], first) for i in indices):
            input_aliases.update((i, indices[0]) for i in indices[1:])
        else:
            staged_input_indices.difference_update(indices)
    return input_aliases


def _collect_inputs(gm, static_input_indices):
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    input_storages = [_storages(node) for node in placeholders]
    static_storages = set().union(*(input_storages[i] for i in static_input_indices))
    mutated_storages = set().union(*(input_storages[i] for i in _mutated_input_indices(gm)))
    staged_input_indices = {
        i
        for i, node in enumerate(placeholders)
        if i not in static_input_indices
        and isinstance(value := node.meta.get("val"), torch.Tensor)
        and _is_npu_tensor(value)
        and value.is_contiguous()
        and len(input_storages[i]) == 1
        and not input_storages[i] & (mutated_storages | static_storages)
    }
    input_aliases = _staging_aliases(placeholders, input_storages, staged_input_indices)
    static_storages.update(set().union(*(input_storages[i] for i in staged_input_indices)))
    return placeholders, static_storages, staged_input_indices, input_aliases


def _partition_hosts(node_hosts):
    partitions, host_partitions = {}, set()
    partition_id, previous_host = -1, None
    for node, host in node_hosts.items():
        if host != previous_host:
            partition_id += 1
            previous_host = host
        partitions[node] = partition_id
        if host:
            host_partitions.add(partition_id)
    return partitions, host_partitions


def _storage_free_node(node):
    return (
        node.op == "call_function" and node.target is _free_tensor_storage
        and node.meta.get("comm_type") in ("fsdp_reshard_free", "fsdp_reshard_free_backward")
    )


def _known_free_metadata(value):
    """Exclude subclass/container contexts that could hide storage aliases."""
    if type(value) in (torch.Tensor, FakeTensor) or isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return True
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return True
    if type(value) in (tuple, list, immutable_list):
        return all(_known_free_metadata(item) for item in value)
    if type(value) in (dict, immutable_dict):
        return all(type(key) in (str, int) and _known_free_metadata(item) for key, item in value.items())
    return False


def _known_free_node_metadata(node):
    # FakeTensorProp omits val for None; this exact helper never returns storage.
    if _storage_free_node(node):
        return node.meta.get("val") is None
    return "val" in node.meta and _known_free_metadata(node.meta["val"])


def _storage_alias_forwarder(node):
    overload = _native_overload(node.target)
    return (
        node.target is operator.getitem
        or overload in (torch.ops._c10d_functional.wait_tensor.default, torch.ops.aten._unsafe_view.default)
        or (overload is not None and any(result.alias_info is not None for result in overload._schema.returns))
    )


def _free_storage_aliases(nodes, storages):
    """Include logical wait/view aliases even when their FakeTensor storages differ."""
    parents = {node: node for node in nodes}

    def _find(node):
        while parents[node] is not node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def _join(left, right):
        parents[_find(left)] = _find(right)

    producers = {}
    for node in nodes:
        for storage in storages[node]:
            _join(node, producers.setdefault(storage, node))
        if _storage_alias_forwarder(node):
            for source in node.all_input_nodes:
                if storages[source] or "val" not in source.meta:
                    _join(node, source)
    groups = defaultdict(set)
    for node in nodes:
        groups[_find(node)].add(node)
    return {node: groups[_find(node)] for node in nodes}


def _private_region_values(nodes, storages):
    """Prove allocation ownership from exact roots, never from Fake storage differences."""
    owned = set()
    roots = (torch.ops.aten.clone.default, torch.ops._c10d_functional.all_gather_into_tensor.default)
    for node in nodes:
        overload = _native_overload(node.target)
        if overload in roots:
            owned.add(node)
        elif _storage_alias_forwarder(node) and (overload is None or not overload._schema.is_mutable):
            inputs = [source for source in node.all_input_nodes if storages[source]]
            if inputs and all(source in owned for source in inputs):
                owned.add(node)
    return owned


def _internal_free_reason(node, region, owned, aliases, order):
    if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], fx.Node):
        return "unknown_free_signature"
    target = node.args[0]
    if not isinstance(target.meta.get("val"), torch.Tensor) or target not in owned:
        return "allocation_not_internal"
    storage_aliases = aliases[target]
    if not storage_aliases <= region:
        return "external_storage_alias"
    for alias in storage_aliases:
        for reader in alias.users:
            if reader not in region:
                return "escaping_storage_alias"
            if order[reader] > order[node]:
                return "alias_read_after_free"
            if reader is not node and not _storage_free_node(reader) and written_nodes(reader):
                return "storage_alias_mutation"
            overload = _native_overload(reader.target)
            if overload is not None and not overload._schema.name.startswith(("aten::", "_c10d_functional::")):
                return "unknown_alias_consumer"
    return None


def _admit_internal_storage_frees(gm, node_hosts):
    helpers = [node for node in node_hosts if _storage_free_node(node)]
    if not helpers:
        return node_hosts, {}, 0, {}
    nodes = list(gm.graph.nodes)
    order = {node: index for index, node in enumerate(nodes)}
    metadata_known = {node: _known_free_node_metadata(node) for node in nodes}
    storages = {node: _storages(node) if metadata_known[node] else set() for node in nodes}
    aliases = _free_storage_aliases(nodes, storages)
    active, rejected, iterations = set(helpers), {}, 0
    metadata_failures = set()
    value_only_writes = (
        torch.ops.aten.copy_.default, torch.ops.aten.add_.Tensor, torch.ops.aten.logical_and_.default,
    )
    node_hosts = {node: False if node in active else host for node, host in node_hosts.items()}
    while active:
        iterations += 1
        partitions, _ = _partition_hosts(node_hosts)
        groups = defaultdict(list)
        for node, partition in partitions.items():
            groups[partition].append(node)
        failures = {}
        for members in groups.values():
            frees = active.intersection(members)
            if not frees:
                continue
            region = set(members)
            inputs = {source for node in members for source in node.all_input_nodes if source not in region}
            known = all(metadata_known[node] for node in region | inputs)
            if not known:
                metadata_failures.update(node for node in region | inputs if not metadata_known[node])
            inline_tensor = any(isinstance(value, torch.Tensor) for node in members
                                for value in tree_leaves((node.args, node.kwargs)))
            boundary_write = any(
                not aliases[written] <= region
                for node in members
                if not _storage_free_node(node) and node.target not in value_only_writes
                for written in written_nodes(node)
            )
            owned = _private_region_values(members, storages)
            for node in frees:
                reason = ("missing_or_unknown_metadata" if not known else "inline_tensor_argument" if inline_tensor
                          else "boundary_input_mutation" if boundary_write
                          else _internal_free_reason(node, region, owned, aliases, order))
                if reason is not None:
                    failures[node] = reason
        if not failures:
            break
        for node, reason in failures.items():
            active.remove(node)
            rejected[node.name] = reason
            node_hosts[node] = True
    missing = [node for node in nodes if node in metadata_failures]
    diagnostics = {
        "missing_metadata_node_count": len(missing),
        "missing_metadata_target_counts": dict(Counter(str(node.target) for node in missing).most_common(16)),
        "missing_metadata_examples": [
            {"name": node.name, "op": node.op, "target": str(node.target),
             "kind": "missing_val" if "val" not in node.meta else "unknown_value_type",
             "value_type": type(node.meta.get("val")).__name__}
            for node in missing[:16]
        ],
    }
    return node_hosts, rejected, iterations, diagnostics


def _partitions(gm, max_capture_regions=0):
    node_hosts = {node: _host_boundary(node) for node in gm.graph.nodes
                  if node.op not in ("placeholder", "get_attr", "output")}
    partitions, host_partitions = _partition_hosts(node_hosts)
    strict_regions = len(set(partitions.values()) - host_partitions)
    helpers = [node for node in node_hosts if _storage_free_node(node)]
    node_hosts, rejected, iterations, diagnostics = _admit_internal_storage_frees(gm, node_hosts)
    partitions, host_partitions = _partition_hosts(node_hosts)
    sizes = Counter(partition for partition in partitions.values() if partition not in host_partitions)
    admitted = [node for node in helpers if not node_hosts[node]]
    if max_capture_regions and len(sizes) > max_capture_regions:
        selected = set(sorted(sizes, key=lambda partition: (-sizes[partition], partition))[:max_capture_regions])
        logger.info("ACLGraph capture budget kept %d of %d device regions", len(selected), len(sizes))
        partitions, host_partitions = _partition_hosts(
            {node: partition not in selected for node, partition in partitions.items()}
        )
    captured = len(set(partitions.values()) - host_partitions)
    gm.meta["aclgraph_partition_stats"] = {
        "strict_device_regions": strict_regions, "device_regions_after_free_admission": len(sizes),
        "max_capture_regions": max_capture_regions, "capture_regions": captured, "eager_regions": len(host_partitions),
        "budget_dropped_device_regions": len(sizes) - captured, "storage_free_candidates": len(helpers),
        "storage_free_admitted": len(admitted), "storage_free_rejected": len(rejected),
        "admitted_helper_names": [node.name for node in admitted], "rejected_helper_reasons": rejected,
        "rejected_helper_names": [node.name for node in helpers if node.name in rejected],
        "captured_helper_names": [node.name for node in admitted if partitions[node] not in host_partitions],
        "fixed_point_iterations": iterations, **diagnostics,
    }
    return partitions, host_partitions


def _output_clone_indices(output, escaping_storages, unknown_returns):
    nodes = [node for node in tree_leaves(output.args[0]) if isinstance(node, fx.Node)]
    if unknown_returns or any("val" not in node.meta for node in nodes):
        return None
    values = tree_map(lambda value: value.meta["val"] if isinstance(value, fx.Node) else value, output.args[0])
    return tuple(
        i for i, value in enumerate(tree_leaves(values))
        if isinstance(value, torch.Tensor) and StorageWeakRef(value.untyped_storage()) in escaping_storages
    )


def _capture_ownership(result, host_partitions, escaping_storages, unknown_returns, placeholders):
    captured_storages = set()
    output_clones, cloned_nodes = {}, set()
    for name, region in result.named_children():
        if not isinstance(region, torch.fx.GraphModule):
            continue
        if int(name.rsplit("_", 1)[1]) not in host_partitions:
            output = next(node for node in region.graph.nodes if node.op == "output")
            output_nodes = [node for node in tree_leaves(output.args) if isinstance(node, torch.fx.Node)]
            indices = output_clones[name] = _output_clone_indices(output, escaping_storages, unknown_returns)
            cloned_nodes.update(node.name for node in output_nodes
                                if indices is None or _storages(node) & escaping_storages)
            region_input_storages = set().union(
                *(_storages(node) for node in region.graph.nodes if node.op == "placeholder")
            )
            for node in tree_leaves(output.args):
                if isinstance(node, torch.fx.Node):
                    # Views of eager inputs inherit their producer's changing
                    # addresses; only allocations made here are capture-owned.
                    captured_storages.update(_storages(node) - region_input_storages)
    for node in placeholders:
        captured_storages.difference_update(_storages(node))
    return captured_storages, output_clones, cloned_nodes


def _region_inputs(region, producers, eager_producers, cloned_nodes, static_storages, captured_storages):
    region_inputs = [node for node in region.graph.nodes if node.op == "placeholder"]
    # SAR can reuse forward FakeTensor storage metadata on eager
    # recomputations. Trust the actual FX producer, not storage alone.
    captured_inputs = {
        i
        for i, node in enumerate(region_inputs)
        if (producer := producers.get(node.name)) is not None and producer not in eager_producers
        and node.name not in cloned_nodes
    }
    mutated = _mutated_input_indices(region)
    written_storages = set().union(*(_storages(region_inputs[i]) for i in mutated))
    static_indices = (
        mutated
        | captured_inputs
        | {i for i, node in enumerate(region_inputs) if _storages(node) & (static_storages | written_storages)}
    )
    # A mixed container may also own eager/state buffers. Borrow it only
    # when every tensor storage comes from a captured producer.
    borrowed_indices = {
        i
        for i, node in enumerate(region_inputs)
        if i in captured_inputs and (storages := _storages(node)) and storages <= captured_storages
    }
    return static_indices, borrowed_indices


def _read_only_producers(gm, escaping_storages, unknown_returns):
    """Prove producer storage is read-only across the complete scheduled graph."""
    if unknown_returns:
        return set()
    written_storages = set()
    for node in gm.graph.nodes:
        for written in written_nodes(node):
            if "val" not in written.meta:
                return set()
            written_storages.update(_storages(written))
    excluded = written_storages | escaping_storages
    native_producers, shareable = set(), set()
    for node in gm.graph.nodes:
        native = node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)
        getitem = (node.op == "call_function" and node.target is operator.getitem
                   and node.args and isinstance(node.args[0], fx.Node) and node.args[0] in native_producers)
        if not (native or getitem):
            continue
        native_producers.add(node)
        storages = _storages(node)
        if isinstance(node.meta.get("val"), torch.Tensor) and storages and not storages & excluded:
            shareable.add(node.name)
    return shareable


def _wrap_regions(result, partitions, host_partitions, ownership, static_storages, coordinator, read_only_producers):
    captured_storages, output_clones, cloned_nodes = ownership
    producers = {node.name: partition for node, partition in partitions.items()}
    summaries, input_plans = [], {}
    shared_producers = Counter()
    for name, region in result.named_children():
        if not isinstance(region, torch.fx.GraphModule) or int(name.rsplit("_", 1)[1]) in host_partitions:
            continue
        static_indices, borrowed_indices = _region_inputs(
            region, producers, host_partitions, cloned_nodes, static_storages, captured_storages
        )
        shared_keys = {
            index: node.name
            for index, node in enumerate(region.graph.find_nodes(op="placeholder"))
            if index not in static_indices and index not in borrowed_indices and node.name in read_only_producers
        }
        shared_producers.update(shared_keys.values())
        input_plans[name] = static_indices, borrowed_indices, shared_keys
    for name, region in result.named_children():
        if not isinstance(region, torch.fx.GraphModule):
            continue
        host = int(name.rsplit("_", 1)[1]) in host_partitions
        summaries.append(
            {
                "name": name,
                "host": host,
                "capture": not host,
            }
        )
        if host:
            continue
        clone_indices = output_clones[name]
        static_indices, borrowed_indices, shared_keys = input_plans[name]
        wrapper = ACLGraphWrapper(
            region.forward,
            static_input_indices=static_indices,
            capture_runnable=torch.fx.Interpreter(region).boxed_run,
            label=name,
            output_clone_indices=clone_indices,
            recapture_coordinator=coordinator,
            borrowed_input_indices=borrowed_indices,
            capture_boxed=True,
            shared_input_keys={index: key for index, key in shared_keys.items() if shared_producers[key] > 1},
        )
        object.__setattr__(region, "_aclgraph_wrapper", wrapper)
        region.forward = wrapper
        coordinator.wrappers.append(wrapper)
    return summaries


def _partition_graph(gm, *, static_input_indices, coordinate_recapture, max_capture_regions=0):
    """Capture contiguous device work, retaining host boundaries in eager."""
    placeholders, static_storages, staged, aliases = _collect_inputs(gm, static_input_indices)
    returned = next(node for node in gm.graph.nodes if node.op == "output")
    returned_nodes = [node for node in tree_leaves(returned.args) if isinstance(node, fx.Node)]
    escaping_storages = set().union(*(_storages(node) for node in returned_nodes))
    unknown_returns = any("val" not in node.meta for node in returned_nodes)
    read_only_producers = _read_only_producers(gm, escaping_storages, unknown_returns)
    partitions, host_partitions = _partitions(gm, max_capture_regions)
    result = split_module(gm, gm, partitions.__getitem__, keep_original_order=True, keep_original_node_name=True)
    ownership = _capture_ownership(
        result, host_partitions, escaping_storages, unknown_returns, placeholders
    )
    coordinator = _RecaptureCoordinator(coordinate_recapture=coordinate_recapture)
    summaries = _wrap_regions(
        result, partitions, host_partitions, ownership, static_storages, coordinator, read_only_producers
    )
    result.meta.update(gm.meta)
    result.meta["aclgraph_partitions"] = summaries
    object.__setattr__(result, "_aclgraph_wrappers", coordinator.wrappers)
    object.__setattr__(result, "_aclgraph_recapture_coordinator", coordinator)
    result.forward = _PartitionInterpreter(result, staged_input_indices=staged, input_aliases=aliases).run
    logger.info("Applied ACLGraph pass: %d capture regions, %d eager regions", len(coordinator.wrappers),
                len(host_partitions))
    return result


def _has_npu_work(graph_module):
    return any(_is_npu_tensor(value) for node in graph_module.graph.nodes
               for value in tree_leaves(node.meta.get("val")) if isinstance(value, torch.Tensor))


def _validate_static_indices(indices, count):
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count for index in indices):
        raise ValueError("ACLGraph static_input_indices must index graph placeholders")


class ACLGraphPass(GraphPass):
    """Capture device regions after FSDP, bucketing and communication scheduling.

    Parameters, buffers and mutated inputs retain their original storage.
    Host reads stay eager. Static mode rejects changed replay signatures;
    dynamic mode coordinates recapture across ranks using a CPU process group.
    """

    name = "aclgraph"

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        example_inputs: tuple[Any, ...] | None = None,
        static_input_indices: Iterable[int] | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Install NPU capture wrappers, returning the partitioned graph.

        Args:
            graph_module: Traced joint forward/backward graph.
            pass_config: Selects off, static, or dynamic ACLGraph replay.
            example_inputs: Fake inputs used to fill metadata on inserted nodes.
            static_input_indices: Lifted state indices. Defaults to the tracer's
                ``num_state_inputs`` leading placeholders.
            model: Live model, whose parameters may already have been sharded.
            **kwargs: Additional pipeline context, unused here.

        Returns:
            A graph whose device regions warm up once before capture and replay.
            CPU-only graphs and disabled passes are returned unchanged.
        """
        if pass_config.aclgraph_mode == "off":
            return graph_module
        pass_config.validate()
        if graph_module.meta.get("inductor_compiled") in ("regional", "full"):
            raise ValueError("ACLGraphPass cannot capture a graph already compiled with Inductor")
        placeholders = list(graph_module.graph.find_nodes(op="placeholder"))
        if example_inputs is not None or model is not None:
            propagate_fake_values(graph_module, example_inputs, model)
        if not _has_npu_work(graph_module):
            return graph_module
        if static_input_indices is None:
            static_input_indices = range(getattr(graph_module, "num_state_inputs", 0))
        static_input_indices = set(static_input_indices)
        _validate_static_indices(static_input_indices, len(placeholders))
        prepare_aclgraph_runtime()
        result = _partition_graph(
            graph_module,
            static_input_indices=static_input_indices,
            coordinate_recapture=pass_config.aclgraph_mode == "dynamic",
            max_capture_regions=pass_config.aclgraph_max_capture_regions,
        )
        # Tracing and physical FSDP layout contracts live outside ``meta``.
        # Gradient clipping must still distinguish shards from replicas.
        for name in ("state_fqns", "state_is_param", "num_state_inputs", "user_inputs_spec", "loss_dict_keys",
                     "fsdp_sharded_param_names"):
            if hasattr(graph_module, name):
                setattr(result, name, getattr(graph_module, name))
        return result
