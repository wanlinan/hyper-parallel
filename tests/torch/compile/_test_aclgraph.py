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
"""Native NPU recording/replay and HCCL numerical acceptance workers."""

import copy
import operator
import os

import pytest
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.immutable_collections import immutable_list

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.compile.passes.aclgraph_pass import ACLGraphPass
from hyper_parallel.compile.passes.aclgraph_runtime import ACLGraphWrapper, teardown_aclgraph
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage


def _loss(model, value, target):
    return (model(value) - target).square().mean()


def test_aclgraph_internal_collective_storage_free():
    """Record a private all-gather/free and replay changing inputs without stale outputs."""
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    count = 16 * world_size
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device(f"npu:{rank}")
    mode = FakeTensorMode()

    def fake(shape, dtype=torch.float32):
        return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), device)

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = fake((4, 4))
    state = graph.placeholder("state")
    state.meta["val"] = fake(())
    accumulated = graph.placeholder("accumulated")
    accumulated.meta["val"] = fake(())
    mask = graph.placeholder("mask")
    mask.meta["val"] = fake((), torch.bool)
    gathered = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        (value, world_size, dist.group.WORLD.group_name),
    )
    gathered.meta["val"] = fake((4 * world_size, 4))
    waited = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (gathered,))
    waited.meta["val"] = fake((4 * world_size, 4))
    pieces = graph.call_function(torch.ops.aten.split_with_sizes.default, (waited, [4 * world_size], 0))
    pieces.meta["val"] = immutable_list([fake((4 * world_size, 4))])
    viewed = graph.call_function(operator.getitem, (pieces, 0))
    viewed.meta["val"] = pieces.meta["val"][0]
    doubled = graph.call_function(torch.ops.aten.mul.Tensor, (viewed, 2.0))
    doubled.meta["val"] = fake((4 * world_size, 4))
    release = graph.call_function(_free_tensor_storage, (waited,))
    release.meta["comm_type"] = "fsdp_reshard_free"
    total = graph.call_function(torch.ops.aten.sum.default, (doubled,))
    total.meta["val"] = fake(())
    copied = graph.call_function(torch.ops.aten.copy_.default, (state, total))
    copied.meta["val"] = state.meta["val"]
    added = graph.call_function(torch.ops.aten.add_.Tensor, (accumulated, total))
    added.meta["val"] = accumulated.meta["val"]
    condition = graph.call_function(torch.ops.aten.lt.Scalar, (total, float(count * (count - 1) + 5 * count)))
    condition.meta["val"] = fake((), torch.bool)
    masked = graph.call_function(torch.ops.aten.logical_and_.default, (mask, condition))
    masked.meta["val"] = mask.meta["val"]
    graph.output(total)
    module = None
    retained = []
    graphs = None
    current = None
    try:
        module = ACLGraphPass().run(
            torch.fx.GraphModule({}, graph), PassConfig(fsdp_enabled=False, aclgraph_mode="static"),
        )
        stats = module.meta["aclgraph_partition_stats"]
        assert stats["storage_free_admitted"] == 1, f"Expected the private free to be admitted: {stats}"
        assert stats["capture_regions"] == 1, f"Expected one complete capture: {stats}"
        state_value = torch.zeros((), device=device)
        accumulated_value = torch.zeros((), device=device)
        mask_value = torch.ones((), device=device, dtype=torch.bool)
        expected_accumulated = 0.0
        for step in range(6):
            data = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4) + rank * 16 + step
            actual = module(data, state_value, accumulated_value, mask_value)
            expected = float(count * (count - 1) + 2 * count * step)
            expected_accumulated += expected
            torch.testing.assert_close(state_value, torch.tensor(expected, device=device), rtol=0, atol=0)
            torch.testing.assert_close(
                accumulated_value, torch.tensor(expected_accumulated, device=device), rtol=0, atol=0,
            )
            assert mask_value.item() is (step < 3), f"Expected the live mask update at step {step}"
            retained.append((actual, expected))
            for output, wanted in retained:
                torch.testing.assert_close(output, torch.tensor(wanted, device=device), rtol=0, atol=0)
            if step:
                current = [wrapper.graph for wrapper in module._aclgraph_wrappers]
                if graphs is not None:
                    assert all(left is right for left, right in zip(current, graphs, strict=True))
                graphs = current
        print(f"Private storage-free acceptance: rank={rank}, regions=1, calls=6")
    finally:
        graphs = None
        current = None
        if module is not None:
            teardown_aclgraph(module)
        if dist.is_initialized():
            dist.destroy_process_group()


def test_aclgraph_mutated_state():
    """Native capture must preserve live buffer identity and retained outputs."""
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu")
    state = torch.zeros((), device=device)

    def _step(value, accumulator):
        accumulator.add_(value.sum())
        return accumulator, value * 2

    wrapper = ACLGraphWrapper(_step, static_input_indices=(1,))
    retained = []
    try:
        for index in range(1, 5):
            output_state, gradient = wrapper(torch.full((4,), float(index), device=device), state)
            assert output_state is state, f"Captured state lost its identity on call {index}"
            torch.testing.assert_close(state, torch.tensor(2 * index * (index + 1), device=device).float())
            retained.append((gradient, torch.full((4,), float(2 * index), device=device)))
            for actual, expected in retained:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch_npu.npu.synchronize()
        wrapper.teardown()


def _compare_gradients(model, reference, rank, world_size, step):
    for (name, parameter), expected_parameter in zip(model.named_parameters(), reference.parameters(), strict=True):
        expected = expected_parameter.grad
        if world_size > 1:
            dist.all_reduce(expected)
            expected.div_(world_size)
            expected = expected.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-6,
                                   msg=lambda message: f"step={step}, rank={rank}, parameter={name}: {message}")


def _compare_parameters(model, reference, rank, world_size):
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        expected = expected_parameter.detach()
        if world_size > 1:
            expected = expected.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)


def _check_captures(wrappers, previous, mode, rank, step):
    current = [wrapper.graph for wrapper in wrappers]
    assert current and all(graph is not None for graph in current), (
        f"Expected captured regions on rank {rank}, got {current}"
    )
    if previous is not None:
        same = [graph is prior for graph, prior in zip(current, previous, strict=True)]
        if mode == "dynamic" and step == 2:
            assert not all(same), f"Rank {rank} did not coordinate changed inputs: {same}"
        else:
            assert all(same), f"Stable inputs unexpectedly re-recorded on rank {rank}, step={step}: {same}"
    return current


def _check_fsdp_capture(compiler):
    gm = compiler._joint_graph.graph_module
    assert gm.fsdp_sharded_param_names, "Capture lost physical FSDP layout metadata used by gradient clipping"
    captured = {entry["name"] for entry in gm.meta["aclgraph_partitions"] if entry["capture"]}
    communication = [(name, str(node.target)) for name, region in gm.named_children()
                     if isinstance(region, torch.fx.GraphModule) for node in region.graph.nodes
                     if str(node.target).startswith("_c10d_functional.")]
    assert communication, f"Expected FSDP communication, got {gm.graph}"
    outside = [(name, target) for name, target in communication if name not in captured]
    assert not outside, f"Expected all native collectives inside capture, got {outside}"


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_aclgraph_training(mode):
    """Capture joint forward/backward, replay with live weights, then release."""
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device(f"npu:{rank}")
    torch.manual_seed(2026)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).to(device)
    reference = copy.deepcopy(model)
    symbolic_shapes = mode == "dynamic"
    compiler = GraphCompiler(
        model, _loss, device=device,
        pass_config=PassConfig(fsdp_enabled=world_size > 1, aclgraph_mode=mode),
        dynamic_arg_dims={"value": [0], "target": [0]} if symbolic_shapes else None,
    )
    optimizer = None
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    retained = []
    graphs = None
    try:
        # One rank's batch-shape change must coordinate recapture on every rank.
        for step in range(4):
            reference_optimizer.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            for microbatch in range(2):
                size = 6 if symbolic_shapes and step >= 2 and rank == 0 else 4
                generator = torch.Generator().manual_seed(100 + rank + 10 * step + microbatch)
                value = torch.randn(size, 16, generator=generator).to(device)
                target = torch.randn(size, 8, generator=generator).to(device)
                expected = _loss(reference, value, target)
                expected.backward()
                actual, _ = compiler.forward_backward(value=value, target=target)
                interpreter = compiler._joint_graph.graph_module.forward.__self__
                assert not interpreter.env, f"Expected released root outputs, got env keys={tuple(interpreter.env)}"
                if optimizer is None:
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
                torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-6)
                for previous, snapshot in retained:
                    torch.testing.assert_close(previous, snapshot, rtol=0, atol=0)
                retained.append((actual, actual.clone()))
            _compare_gradients(model, reference, rank, world_size, step)
            optimizer.step()
            reference_optimizer.step()
            _compare_parameters(model, reference, rank, world_size)
            wrappers = compiler._joint_graph.graph_module._aclgraph_wrappers
            graphs = _check_captures(wrappers, graphs, mode, rank, step)
        if world_size > 1:
            _check_fsdp_capture(compiler)
        print(f"ACLGraph acceptance: rank={rank}, mode={mode}, "
              f"world_size={world_size}, regions={len(wrappers)}")
    finally:
        # Drop test-only graph references before releasing HCCL resources.
        graphs = None
        compiler.close()
        dist.destroy_process_group()
