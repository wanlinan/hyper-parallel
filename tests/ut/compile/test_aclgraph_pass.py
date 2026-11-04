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
"""CPU contracts for ACLGraph partitioning, replay guards and compiler wiring."""

import copy
import gc
import operator
import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.traceback import preserve_node_meta
from torch.utils._pytree import tree_leaves

from hyper_parallel.compile import GraphCompiler, GraphTrainer, PassConfig
from hyper_parallel.compile.passes import aclgraph_pass as passes
from hyper_parallel.compile.passes import aclgraph_runtime as runtime
from hyper_parallel.compile.passes import graph_metadata
from hyper_parallel.compile.passes.parallel.fsdp_pass import _free_tensor_storage
from hyper_parallel.compile.passes.pipeline import PassPipeline
from hyper_parallel.distributed.expert_parallel._tracing import expert_parallel_region
from hyper_parallel.models.build_options import CompileConfig


class _Capture:
    """Simulate only native graph recording; Python/FX still execute on CPU."""

    def __init__(self) -> None:
        """Track mock recording boundaries and captured callable state."""
        self.current = None
        self.recordings = []
        self.npu = SimpleNamespace(
            NPUGraph=self._new_graph, graph=self._record, graph_pool_handle=object,
            synchronize=Mock(), get_task_queue_enable=lambda: 1,
        )

    @staticmethod
    def _new_graph():
        return SimpleNamespace(replay=lambda: None)

    @contextmanager
    def _record(self, graph, *, pool):
        del pool
        self.current = graph
        self.recordings.append(graph)
        try:
            yield
        finally:
            self.current = None

    def _run(self, function, inputs):
        """Use captured buffers when replaying an FX region on CPU."""
        saved = function(*inputs)
        if self.current is not None:
            # CPU executes during recording; native NPU recording runs on replay.
            initial = True

            def _replay():
                nonlocal initial
                if initial:
                    initial = False
                    return
                current = function(*inputs)
                for destination, value in zip(tree_leaves(saved), tree_leaves(current), strict=True):
                    if isinstance(destination, torch.Tensor):
                        destination.copy_(value)

            self.current.replay = _replay
        return saved


class TestACLGraphRuntime(unittest.TestCase):
    """Exercise guards and ownership without importing the optional backend."""

    def setUp(self) -> None:
        """Install a CPU simulation of the optional native backend."""
        self.capture = _Capture()
        self.patch = patch.object(runtime, "_get_torch_npu", return_value=SimpleNamespace(npu=self.capture.npu))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def _wrap(self, function, *, static=(), mode="static", output_clone_indices=None,
              coordinator=None, shared_input_keys=None):
        """Construct a wrapper whose captured callable uses CPU replay."""
        if coordinator is None:
            coordinator = runtime._RecaptureCoordinator(coordinate_recapture=mode == "dynamic")
        wrapper = runtime.ACLGraphWrapper(
            function, static_input_indices=static,
            capture_runnable=lambda *args: self.capture._run(function, args),
            recapture_coordinator=coordinator,
            output_clone_indices=output_clone_indices,
            shared_input_keys=shared_input_keys,
        )
        coordinator.wrappers.append(wrapper)
        self.addCleanup(coordinator.teardown)
        self.addCleanup(wrapper.teardown)
        return wrapper

    def test_replay_preserves_state_and_previous_gradients(self):
        """Live state is updated in place; old returned gradients remain valid."""
        state = torch.zeros(())

        def _step(value, accumulator):
            accumulator.add_(value.sum())
            return {"state": accumulator, "gradient": value * 2}

        wrapper = self._wrap(_step, static=(1,))
        retained = []
        with patch.object(wrapper.recapture_coordinator, "_coordinate") as coordinate:
            for index in range(1, 5):
                result = wrapper(torch.full((4,), float(index)), state)
                self.assertIs(result["state"], state)
                retained.append((result["gradient"], torch.full((4,), float(2 * index))))
                for actual, expected in retained:
                    torch.testing.assert_close(actual, expected)
                self.assertEqual(state.item(), 4 * index * (index + 1) / 2)
            coordinate.assert_not_called()
        self.assertEqual(len(self.capture.recordings), 1)
        wrapper.teardown()
        self.assertIsNone(wrapper.graph)
        self.assertIsNone(wrapper.args)

    def test_signature_changes_are_rejected_before_replay(self):
        """Shape, stride, dtype, scalar type/value and static address are guards."""
        cases = (
            (torch.ones(3, 2), 1, "shape"),
            (torch.ones(2, 2).t(), 1, "stride"),
            (torch.ones(2, 2, dtype=torch.float64), 1, "dtype"),
            (torch.ones(2, 2), 2, "scalar"),
            (torch.ones(2, 2), True, "scalar"),
        )
        for value, scale, reason in cases:
            with self.subTest(reason=reason, scale=scale):
                wrapper = self._wrap(lambda x, multiplier: x * multiplier)
                wrapper(torch.ones(2, 2), 1)
                wrapper(torch.ones(2, 2), 1)
                with self.assertRaisesRegex(ValueError, reason):
                    wrapper(value, scale)
        wrapper = self._wrap(lambda x: x + 1, static=(0,))
        value = torch.ones(2)
        wrapper(value)
        wrapper(value)
        with self.assertRaisesRegex(ValueError, "static address"):
            wrapper(value.clone())

    def test_first_capture_returns_owning_internal_results(self):
        """Mixed output cloning must retain owners for downstream recordings."""
        recorded = []

        def _step(value):
            output = {"internal": value * 2, "loss": [value.sum()]}
            recorded.append(output)
            return output

        wrapper = self._wrap(_step, output_clone_indices=(1,))
        wrapper(torch.ones(4))
        # Distinct aliases stand in for native non-owning replay handles.
        with patch.object(wrapper, "_replay_handle_factory", return_value=lambda tensor: tensor.detach()):
            actual = wrapper(torch.full((4,), 2.0))
        self.assertIs(actual["internal"], recorded[-1]["internal"])
        self.assertIsNot(actual["internal"], wrapper.output["internal"])
        self.assertNotEqual(actual["loss"][0].data_ptr(), recorded[-1]["loss"][0].data_ptr())
        torch.testing.assert_close(actual["internal"], torch.full((4,), 4.0))
        torch.testing.assert_close(actual["loss"][0], torch.tensor(8.0))
        current = wrapper(torch.full((4,), 3.0))
        self.assertIs(current["internal"], wrapper.output["internal"])
        torch.testing.assert_close(current["internal"], torch.full((4,), 6.0))
        torch.testing.assert_close(current["loss"][0], torch.tensor(12.0))
        torch.testing.assert_close(actual["loss"][0], torch.tensor(8.0))

    def test_dynamic_mode_recaptures_changed_layout_and_state(self):
        """Shared staging follows recording generations, then releases with graphs."""
        wrapper = self._wrap(lambda value, state: value + state, static=(1,), mode="dynamic",
                             shared_input_keys={0: "producer"})
        coordinator = wrapper.recapture_coordinator
        other = self._wrap(lambda value, state: value * state, static=(1,), mode="dynamic",
                           coordinator=coordinator, shared_input_keys={0: "producer"})
        state = torch.tensor(1.0)
        previous_size, previous_pointer = 0, 0
        for size in (2, 2, 2, 4, 4, 4):
            value = torch.full((size,), float(size))
            torch.testing.assert_close(wrapper(value, state), value + 1)
            torch.testing.assert_close(other(value, state), value)
            if wrapper.args is not None:
                self.assertEqual(wrapper.args[0].data_ptr(), other.args[0].data_ptr())
                if previous_size and previous_size != size:
                    self.assertNotEqual(wrapper.args[0].data_ptr(), previous_pointer)
                previous_size, previous_pointer = size, wrapper.args[0].data_ptr()
        self.assertEqual(len(self.capture.recordings), 4)
        prior_pointer = wrapper.args[0].data_ptr()
        other_state = state
        state = state.clone()
        torch.testing.assert_close(wrapper(torch.ones(4), state), torch.full((4,), 2.0))
        torch.testing.assert_close(other(torch.ones(4), other_state), torch.ones(4))
        self.assertEqual(len(self.capture.recordings), 5)
        self.assertNotEqual(wrapper.pool_id, other.pool_id)
        self.assertNotEqual(wrapper.args[0].data_ptr(), other.args[0].data_ptr())
        for _ in range(2):
            torch.testing.assert_close(wrapper(torch.ones(4), state), torch.full((4,), 2.0))
            torch.testing.assert_close(other(torch.ones(4), other_state), torch.ones(4))
        self.assertEqual(len(self.capture.recordings), 7)
        self.assertEqual(wrapper.pool_id, other.pool_id)
        self.assertNotEqual(wrapper.args[0].data_ptr(), prior_pointer)
        self.assertEqual(wrapper.args[0].data_ptr(), other.args[0].data_ptr())
        saved = weakref.ref(wrapper.args[0])
        wrapper.teardown()
        other.teardown()
        coordinator.teardown()
        self.capture.recordings.clear()
        gc.collect()
        self.assertIsNone(saved())

    def test_interpreter_releases_completed_and_failed_environments(self) -> None:
        """Caller-owned gradients survive; completed and exceptional roots do not."""
        module = make_fx(lambda value: (value.sum(), value * 2))(torch.ones(3))
        interpreter = runtime._PartitionInterpreter(module)
        result = interpreter.run(torch.tensor([1., 2., 3.]))
        torch.testing.assert_close(result[0], torch.tensor(6.))
        torch.testing.assert_close(result[1], torch.tensor([2., 4., 6.]))
        self.assertEqual(interpreter.env, {})
        gradient = weakref.ref(result[1])
        consumed = result[1].clone()
        del result
        gc.collect()
        self.assertIsNone(gradient())
        torch.testing.assert_close(consumed, torch.tensor([2., 4., 6.]))

        placeholder = next(iter(module.graph.find_nodes(op="placeholder")))
        sentinel = object()
        initial_env = {placeholder: torch.ones(3), sentinel: "external owner"}
        result = interpreter.run(initial_env=initial_env)
        torch.testing.assert_close(result[1], torch.full((3,), 2.))
        self.assertEqual(initial_env[sentinel], "external owner")
        self.assertEqual(interpreter.env, {})

        error = RuntimeError("host failure")
        graph = torch.fx.Graph()
        value = graph.placeholder("value")

        def _fail(tensor):
            del tensor
            raise error

        failed = graph.call_function(_fail, (value,))
        graph.output(failed)
        interpreter = runtime._PartitionInterpreter(torch.fx.GraphModule({}, graph))
        with self.assertRaisesRegex(RuntimeError, "host failure") as raised:
            interpreter.run(torch.ones(3))
        self.assertIs(raised.exception, error)
        self.assertEqual(interpreter.env, {})

    def test_remote_input_change_coordinates_recapture(self):
        """All ranks enter recording even when only another rank changes input."""
        wrapper = self._wrap(lambda value: value * 3, mode="dynamic")
        coordinator = wrapper.recapture_coordinator
        coordinator.initialized = True
        coordinator.group = object()
        with patch.object(runtime.torch.distributed, "all_reduce") as reduce:
            wrapper(torch.ones(2))
            wrapper(torch.ones(2))
            first_graph = wrapper.graph
            reduce.side_effect = lambda flags, **kwargs: flags.__setitem__(0, 1)
            wrapper(torch.ones(2))
            self.assertIsNot(wrapper.graph, first_graph)
            self.assertEqual(len(self.capture.recordings), 2)
            reduce.side_effect = None
            wrapper(torch.ones(2))
            self.assertEqual(len(self.capture.recordings), 2)
            coordinator.group = None


class TestACLGraphPass(unittest.TestCase):
    """Run the actual partition pass and FX interpreter against CPU tensors."""

    def setUp(self) -> None:
        """Install a CPU simulation of the optional native backend."""
        self.capture = _Capture()
        self.addCleanup(patch.stopall)
        patch.object(runtime, "_get_torch_npu", return_value=SimpleNamespace(npu=self.capture.npu)).start()
        patch.object(passes, "prepare_aclgraph_runtime").start()
        # CUDA FakeTensors are available in CPU PyTorch builds, including those
        # with no registered private NPU backend. Only metadata uses this label.
        patch.object(passes, "_is_npu_tensor", side_effect=lambda value: value.device.type == "cuda").start()
        original = torch.fx.Interpreter.boxed_run

        def _boxed_run(interpreter, args):
            values = list(args)
            return self.capture._run(lambda *inputs: original(interpreter, list(inputs)), values)

        patch.object(torch.fx.Interpreter, "boxed_run", _boxed_run).start()

    @staticmethod
    def _graph():
        """Create device work separated by an explicit CPU scalar read."""
        mode = FakeTensorMode()
        def _fake(shape):
            return FakeTensor(mode, torch.empty(shape, device="meta"), torch.device("cuda:0"))
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        state = graph.placeholder("state")
        value.meta["val"] = _fake((2, 2))
        state.meta["val"] = _fake((2, 2))
        product = graph.call_function(torch.ops.aten.mm.default, (value, state))
        product.meta["val"] = _fake((2, 2))
        total = graph.call_function(torch.ops.aten.sum.default, (product,))
        total.meta["val"] = _fake(())
        host = graph.call_function(torch.ops.aten._to_copy.default, (total,), {"device": torch.device("cpu")})
        host.meta["val"] = torch.empty(())
        scalar = graph.call_function(torch.ops.aten._local_scalar_dense.default, (host,))
        scalar.meta["val"] = 1.0
        result = graph.call_function(torch.ops.aten.mul.Scalar, (product, scalar))
        result.meta["val"] = _fake((2, 2))
        graph.output(result)
        module = torch.fx.GraphModule({}, graph)
        module.loss_dict_keys = ["loss"]
        return module

    def test_host_partition_replay_and_escaping_outputs(self):
        """Host reads run each step; captured outputs track changing batch values."""
        source = self._graph()
        source.fsdp_sharded_param_names = frozenset({"weight"})
        gm = passes.ACLGraphPass().run(source, PassConfig(aclgraph_mode="static"), static_input_indices=(1,))
        self.addCleanup(runtime.teardown_aclgraph, gm)
        self.assertEqual([entry["host"] for entry in gm.meta["aclgraph_partitions"]], [False, True, False])
        self.assertEqual(gm.loss_dict_keys, ["loss"])
        self.assertEqual(gm.fsdp_sharded_param_names, {"weight"})
        state = torch.eye(2)
        retained = []
        for data in ([[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]], [[3., 1.], [2., 4.]]):
            source = torch.tensor(data)
            actual = gm(source, state)
            retained.append((actual, source * source.sum()))
            for output, expected in retained:
                torch.testing.assert_close(output, expected)
        self.assertEqual(len(self.capture.recordings), 2)
        runtime.teardown_aclgraph(gm)
        self.assertTrue(all(wrapper.graph is None for wrapper in gm._aclgraph_wrappers))

    @staticmethod
    def _shared_eager_graph(*, return_alias=False, write_alias=False):
        """Make two capture consumers separated by a host read or alias write."""
        def _read(total):
            return total.sum().item()

        def _write(value, total):
            value.view(-1).add_(1)
            return total.item()

        template = next(iter(TestACLGraphPass._graph().graph.nodes)).meta["val"]
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        value.meta["val"] = template
        prepared = graph.call_function(torch.ops.aten.mul.Scalar, (value, 2))
        prepared.name = "prepared"
        prepared.meta["val"] = template.clone()
        host_input = graph.call_function(
            torch.ops.aten._to_copy.default, (prepared,), {"device": torch.device("cpu")},
        )
        host_input.meta["val"] = torch.empty(2, 2)
        scale = graph.call_function(_read, (host_input,))
        scale.meta["val"] = 12.0
        scaled = graph.call_function(torch.ops.aten.mul.Scalar, (prepared, scale))
        scaled.meta["val"] = template.clone()
        first = graph.call_function(torch.ops.aten.square.default, (scaled,))
        first.meta["val"] = template.clone()
        output = first
        if return_alias:
            output = graph.call_function(torch.ops.aten.view.default, (prepared, (4,)))
            output.meta["val"] = prepared.meta["val"].view(4)
        total = graph.call_function(torch.ops.aten.sum.default, (first,))
        total.meta["val"] = template.sum()
        host = graph.call_function(torch.ops.aten._to_copy.default, (total,), {"device": torch.device("cpu")})
        host.meta["val"] = torch.empty(())
        scalar = graph.call_function(_write, (prepared, host)) if write_alias else graph.call_function(_read, (host,))
        scalar.meta["val"] = 1.0
        second = graph.call_function(torch.ops.aten.mul.Scalar, (prepared, scalar))
        second.meta["val"] = template.clone()
        positive = graph.call_function(torch.ops.aten.relu.default, (second,))
        positive.meta["val"] = template.clone()
        graph.output((output, positive))
        return torch.fx.GraphModule({}, graph)

    def test_shared_eager_inputs_respect_writes_and_retained_outputs(self) -> None:
        """Only globally read-only producers share staging across capture consumers."""
        for return_alias, write_alias in ((False, False), (True, False), (False, True)):
            with self.subTest(return_alias=return_alias, write_alias=write_alias):
                source = self._shared_eager_graph(return_alias=return_alias, write_alias=write_alias)
                gm = passes._partition_graph(
                    source, static_input_indices=set(), coordinate_recapture=False, max_capture_regions=2,
                )
                self.addCleanup(runtime.teardown_aclgraph, gm)
                retained = []
                for index in range(4):
                    value = torch.arange(4.).roll(index).reshape(2, 2)
                    prepared = value * 2
                    squared = (prepared * prepared.sum().item()).square()
                    first = prepared.view(4) if return_alias else squared
                    second = (prepared + int(write_alias)) * squared.sum().item()
                    actual = gm(value)
                    retained.append((actual, (first, second)))
                    for output, expected in retained:
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                buffers = []
                for region in gm.children():
                    wrapper = getattr(region, "_aclgraph_wrapper", None)
                    if wrapper is None:
                        continue
                    names = [node.name for node in region.graph.find_nodes(op="placeholder")]
                    position = names.index("prepared")
                    self.assertEqual(position in wrapper.shared_input_keys, not (return_alias or write_alias))
                    buffers.append(wrapper.args[position])
                self.assertEqual(len(buffers), 2)
                if return_alias or write_alias:
                    self.assertNotEqual(buffers[0].data_ptr(), buffers[1].data_ptr())
                else:
                    self.assertEqual(buffers[0].data_ptr(), buffers[1].data_ptr())

    def test_capture_budget_prefers_larger_regions(self):
        """A later larger region wins over an earlier smaller candidate."""
        source = self._graph()
        output = next(node for node in source.graph.nodes if node.op == "output")
        result = output.args[0]
        with source.graph.inserting_before(output):
            positive = source.graph.call_function(torch.ops.aten.relu.default, (result,))
            positive.meta["val"] = result.meta["val"].clone()
            squared = source.graph.call_function(torch.ops.aten.square.default, (positive,))
            squared.meta["val"] = result.meta["val"].clone()
        output.args = (squared,)
        source.recompile()
        gm = passes.ACLGraphPass().run(
            source, PassConfig(aclgraph_mode="static", aclgraph_max_capture_regions=1), static_input_indices=(1,),
        )
        self.addCleanup(runtime.teardown_aclgraph, gm)
        self.assertEqual([entry["host"] for entry in gm.meta["aclgraph_partitions"]], [True, False])
        state = torch.eye(2)
        for values in ([[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]], [[3., 1.], [2., 4.]]):
            value = torch.tensor(values)
            torch.testing.assert_close(gm(value, state), (value * value.sum()).relu().square())
        self.assertEqual(len(self.capture.recordings), 1)

    @staticmethod
    def _storage_free_graph(case="internal"):
        """Build exact FSDP releases with separate fake and runtime CPU storage."""
        template = next(iter(TestACLGraphPass._graph().graph.nodes)).meta["val"]
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        value.meta["val"] = template
        clone_input = value
        if case == "immutable_dict_input":
            value.meta["val"] = immutable_dict({"value": template})
            clone_input = graph.call_function(operator.getitem, (value, "value"))
            clone_input.meta["val"] = template
        state = None
        if case in ("external_copy", "external_add", "external_logical", "private_alias_copy"):
            state = graph.placeholder("state")
            state.meta["val"] = template.to(torch.bool) if case == "external_logical" else template.clone()
        target = graph.call_function(
            torch.clone if case == "host_producer" else torch.ops.aten.clone.default, (clone_input,),
        )
        target.meta["val"] = template.clone()
        if case == "external_input":
            target = value
        if case == "internal_all_gather":
            target.target = torch.ops._c10d_functional.all_gather_into_tensor.default
            target.args = (value, 2, "unused")
            target.meta["val"] = template.repeat(2, 1)
            waited = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (target,))
            waited.meta["val"] = target.meta["val"].clone()
            target = waited
        if case == "external_wait":
            target.target = torch.ops._c10d_functional.all_gather_into_tensor.default
            target.args = (value, 1, "unused")
            barrier = graph.call_function(float, (1,))
            barrier.meta["val"] = 1.0
            target = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (target,))
            target.meta["val"] = template.clone()
        escaped = target
        if case == "wait_escape":
            waited = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (target,))
            waited.meta["val"] = template.clone()
            escaped = graph.call_function(torch.ops.aten.view.default, (waited, (4,)))
            escaped.meta["val"] = waited.meta["val"].view(4)
        if case == "unsafe_view_escape":
            escaped = graph.call_function(torch.ops.aten._unsafe_view.default, (target, (4,)))
            escaped.meta["val"] = template.clone().view(4)
        read_target = target
        immutable_cases = ("immutable_list", "immutable_alias_escape", "immutable_alias_mutation",
                           "unknown_immutable_element")
        if case in immutable_cases:
            parts = graph.call_function(torch.ops.aten.split_with_sizes.default, (target, [1, 1], 0))
            views = target.meta["val"].split((1, 1), dim=0)
            parts.meta["val"] = immutable_list(views)
            if case == "unknown_immutable_element":
                parts.meta["val"] = immutable_list([views[0], SimpleNamespace(hidden=views[1])])
            read_target = graph.call_function(operator.getitem, (parts, 0))
            read_target.meta["val"] = views[0]
            if case == "immutable_alias_escape":
                escaped = read_target
            if case == "immutable_alias_mutation":
                modified = graph.call_function(torch.ops.aten.add_.Scalar, (read_target, 1))
                modified.meta["val"] = views[0]
        computed = graph.call_function(torch.ops.aten.sin.default, (read_target,))
        computed_value = read_target.meta["val"].clone()
        computed.meta["val"] = computed_value
        if state is not None:
            source = target if case == "private_alias_copy" else computed
            operation = torch.ops.aten.copy_.default
            if case == "external_add":
                operation = torch.ops.aten.add_.Tensor
            elif case == "external_logical":
                source = graph.call_function(torch.ops.aten.gt.Scalar, (computed, 0))
                source.meta["val"] = computed_value > 0
                operation = torch.ops.aten.logical_and_.default
            copied = graph.call_function(
                operation, (state, source),
            )
            copied.meta["val"] = state.meta["val"]
        if case in ("private_alias_add", "private_alias_logical"):
            operation = (torch.ops.aten.add_.Tensor if case == "private_alias_add"
                         else torch.ops.aten.logical_and_.default)
            modified = graph.call_function(operation, (target, computed))
            modified.meta["val"] = target.meta["val"]
        if case == "missing_metadata":
            del computed.meta["val"]
        if case == "unknown_metadata_context":
            computed.meta["val"] = SimpleNamespace(value=template.clone())
        if case == "storage_rebind":
            rebound = graph.call_function(torch.ops.aten.set_.source_Tensor, (value, target))
            rebound.meta["val"] = value.meta["val"]
        if case == "fixed_point":
            bad_free = graph.call_function(_free_tensor_storage, (value,))
            bad_free.meta["comm_type"] = "fsdp_reshard_free"
        released = graph.call_function(_free_tensor_storage, (target,))
        released.meta["comm_type"] = "fsdp_reshard_free_backward"
        if case == "wrong_tag":
            released.meta["comm_type"] = "unrecognized_free"
        result = graph.call_function(torch.ops.aten.mul.Scalar, (target if case == "later_alias_read" else computed, 2))
        result.meta["val"] = computed_value.clone()
        graph.output((result, escaped) if case in (
            "escaping_output", "wait_escape", "unsafe_view_escape", "immutable_alias_escape",
        ) else result)
        return torch.fx.GraphModule({}, graph), released

    def test_private_storage_free_stays_inside_one_capture_across_replays(self) -> None:
        """A private clone is released after its last read without splitting the region."""
        for case in ("internal", "immutable_list", "immutable_dict_input",
                     "external_copy", "external_add", "external_logical"):
            with self.subTest(case=case):
                source, released = self._storage_free_graph(case)
                for node in source.graph.nodes:
                    if type(node.meta.get("val")) in (immutable_list, immutable_dict):
                        self.assertTrue(graph_metadata.tensor_storages(node))
                recordings = len(self.capture.recordings)
                gm = passes._partition_graph(source, static_input_indices=set(), coordinate_recapture=False)
                self.addCleanup(runtime.teardown_aclgraph, gm)
                stats = gm.meta["aclgraph_partition_stats"]
                self.assertEqual((stats["strict_device_regions"], stats["device_regions_after_free_admission"]), (2, 1))
                self.assertEqual(stats["admitted_helper_names"], [released.name])
                self.assertEqual(len(gm._aclgraph_wrappers), 1)
                state = torch.ones(2, 2, dtype=torch.bool) if case == "external_logical" else torch.zeros(2, 2)
                state_expected = state.clone()
                retained = []
                for index in range(4):
                    value = torch.arange(4.).reshape(2, 2) + index
                    inputs = ({"value": value},) if case == "immutable_dict_input" else (value,)
                    if case in ("external_copy", "external_add", "external_logical"):
                        inputs = (*inputs, state)
                    expected = (value[:1] if case == "immutable_list" else value).sin() * 2
                    retained.append((gm(*inputs), expected))
                    for actual, expected in retained:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if case == "external_copy":
                        state_expected = value.sin()
                    elif case == "external_add":
                        state_expected = state_expected + value.sin()
                    elif case == "external_logical":
                        state_expected = state_expected & (value.sin() > 0)
                    torch.testing.assert_close(state, state_expected, rtol=0, atol=0)
                self.assertEqual(len(self.capture.recordings), recordings + 1)

    def test_internal_all_gather_free_accepts_absent_none_result_metadata(self) -> None:
        """Functional AG owns wait/view storage even when FakeTensorProp omits free val."""
        source, released = self._storage_free_graph("internal_all_gather")
        self.assertNotIn("val", released.meta)
        partitions, hosts = passes._partitions(source)
        self.assertNotIn(partitions[released], hosts)
        self.assertEqual(source.meta["aclgraph_partition_stats"]["storage_free_admitted"], 1)
        self.assertEqual(source.meta["aclgraph_partition_stats"]["device_regions_after_free_admission"], 1)

    def test_storage_free_rejects_external_escaping_and_unproven_aliases(self) -> None:
        """Frees cannot borrow external storage or hide wait aliases behind fake allocations."""
        for case in ("external_input", "escaping_output", "missing_metadata", "host_producer",
                     "external_wait", "wait_escape", "unsafe_view_escape", "later_alias_read",
                     "storage_rebind", "unknown_metadata_context", "wrong_tag", "immutable_alias_escape",
                     "immutable_alias_mutation", "unknown_immutable_element", "private_alias_copy",
                     "private_alias_add", "private_alias_logical"):
            with self.subTest(case=case):
                source, released = self._storage_free_graph(case)
                partitions, hosts = passes._partitions(source)
                self.assertIn(partitions[released], hosts)
                stats = source.meta["aclgraph_partition_stats"]
                self.assertEqual(stats["admitted_helper_names"], [])
                self.assertEqual(stats["rejected_helper_names"], [] if case == "wrong_tag" else [released.name])
                if case == "missing_metadata":
                    self.assertGreater(stats["missing_metadata_node_count"], 0)
                    self.assertIn("aten.sin.default", stats["missing_metadata_target_counts"])

    def test_failed_storage_free_repartitions_until_ownership_is_stable(self) -> None:
        """Rejecting one helper must invalidate another whose allocation moved outside."""
        source, released = self._storage_free_graph("fixed_point")
        partitions, hosts = passes._partitions(source, max_capture_regions=1)
        stats = source.meta["aclgraph_partition_stats"]
        self.assertEqual(stats["storage_free_candidates"], 2)
        self.assertEqual(stats["storage_free_admitted"], 0)
        self.assertEqual(stats["storage_free_rejected"], 2)
        self.assertGreaterEqual(stats["fixed_point_iterations"], 2)
        self.assertIn(partitions[released], hosts)

    def test_dynamic_routed_forward_and_backward_replay(self):
        """Changing routed row counts must preserve loss and gradient across replay."""
        @expert_parallel_region()
        def _experts(value):
            return value.masked_select(value > 0).square().sum()

        def _joint(value):
            loss = _experts(value * 2).square()
            return loss, torch.autograd.grad(loss, value)[0]

        with preserve_node_meta():
            source = make_fx(_joint, tracing_mode="symbolic")(torch.tensor([1., -1., 2., -2.], requires_grad=True))
        mode = FakeTensorMode()
        for node in source.graph.nodes:
            if isinstance(value := node.meta.get("val"), torch.Tensor):
                node.meta["val"] = FakeTensor(mode, torch.empty_like(value, device="meta"), torch.device("cuda"))
        gm = passes._partition_graph(source, static_input_indices=set(), coordinate_recapture=False)
        self.addCleanup(runtime.teardown_aclgraph, gm)
        self.assertTrue(gm._aclgraph_wrappers)
        for region in gm.children():
            if hasattr(region, "_aclgraph_wrapper"):
                self.assertFalse(any(node.meta.get("custom", {}).get("expert_parallel")
                                     for node in region.graph.nodes if node.op != "placeholder"))
        retained = []
        for values in ([1., -1., 2., -2.], [1., 2., 3., 4.], [-1., -2., 3., -4.],
                       [-1., -2., -3., -4.], [2., 1., -1., 3.]):
            positive_square_sum = sum(value * value for value in values if value > 0)
            expected = (torch.tensor(16.0 * positive_square_sum ** 2), torch.tensor([
                64 * positive_square_sum * value if value > 0 else 0. for value in values
            ]))
            retained.append((gm(torch.tensor(values)), expected))
            for actual, reference in retained:
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertEqual(len(self.capture.recordings), len(gm._aclgraph_wrappers))

    def test_metadata_uses_live_shards_instead_of_full_trace_examples(self):
        """Post-FSDP propagation must size collectives from physical shards."""
        model = torch.nn.Linear(4, 4, bias=False)
        gm = make_fx(lambda weight, value: value.mm(weight.t()), tracing_mode="fake")(
            torch.ones(4, 4), torch.ones(3, 4)
        )
        placeholders = list(gm.graph.find_nodes(op="placeholder"))
        examples = tuple(node.meta["val"] for node in placeholders)
        gm.state_fqns = ["weight"]
        weight = placeholders[0]
        users = list(weight.users)
        with gm.graph.inserting_after(weight):
            gathered = gm.graph.call_function(torch.ops.aten.repeat.default, (weight, (2, 1)))
        for user in users:
            user.replace_input_with(weight, gathered)
        gm.recompile()
        model.weight.data = torch.ones(2, 4)
        graph_metadata.propagate_fake_values(gm, examples, model)
        self.assertEqual(tuple(weight.meta["val"].shape), (2, 4))
        self.assertEqual(tuple(gathered.meta["val"].shape), (4, 4))
        output = next(node for node in gm.graph.nodes if node.op == "output")
        self.assertEqual(tuple(output.args[0].meta["val"].shape), (3, 4))

    def test_python_helpers_mutate_live_inputs_on_every_call(self):
        """An unknown host helper runs once per call and keeps the caller's storage."""
        calls = []

        def _update(value):
            calls.append(1)
            return value.add_(1)

        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        update = graph.call_function(_update, (value,))
        result = graph.call_function(torch.ops.aten.sin.default, (update,))
        graph.output(result)
        metadata = next(iter(self._graph().graph.nodes)).meta["val"]
        for node in (value, update, result):
            node.meta["val"] = metadata
        gm = passes.ACLGraphPass().run(torch.fx.GraphModule({}, graph), PassConfig(aclgraph_mode="static"))
        self.addCleanup(runtime.teardown_aclgraph, gm)
        value = torch.zeros(2, 2)
        for index in range(1, 5):
            torch.testing.assert_close(gm(value), torch.full((2, 2), float(index)).sin())
            torch.testing.assert_close(value, torch.full((2, 2), float(index)))
        self.assertEqual(len(calls), 4)

    def test_cpu_and_disabled_graphs_do_not_load_backend(self):
        """CPU use and defaults retain normal FX execution."""
        gm = make_fx(lambda value: value.sin())(torch.ones(2))
        with patch.object(passes, "prepare_aclgraph_runtime") as prepare:
            self.assertIs(passes.ACLGraphPass().run(gm, PassConfig(aclgraph_mode="static")), gm)
            device_graph = self._graph()
            self.assertIs(passes.ACLGraphPass().run(device_graph, PassConfig()), device_graph)
            prepare.assert_not_called()


class TestACLGraphIntegration(unittest.TestCase):
    """Check configuration projection, replacement graphs and cleanup owners."""

    def test_config_validation_and_projection(self):
        """Capture is opt-in, YAML modes are validated, and projection preserves it."""
        self.assertEqual(PassConfig().aclgraph_mode, "off")
        self.assertEqual(PassConfig().aclgraph_max_capture_regions, 0)
        self.assertNotIn("aclgraph", [entry.name for entry in PassPipeline.from_config(PassConfig()).passes])
        for count in (-1, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "aclgraph_max_capture_regions"):
                PassConfig(aclgraph_max_capture_regions=count)
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "aclgraph_max_capture_regions"):
                CompileConfig(aclgraph_max_capture_regions=count)
        with self.assertRaises(ValueError):
            PassConfig(aclgraph_mode="unknown")
        for mode in ("static", "dynamic"):
            for options in ({"pp_enabled": True}, {"inductor_mode": "regional"}, {"inductor_mode": "full"}):
                with self.subTest(mode=mode, options=options), self.assertRaisesRegex(
                    ValueError, "ACLGraph currently requires"
                ):
                    PassConfig(aclgraph_mode=mode, **options)
            graph = torch.fx.symbolic_trace(torch.nn.Identity())
            for compiled_mode in ("regional", "full"):
                graph.meta["inductor_compiled"] = compiled_mode
                with self.subTest(mode=mode, compiled_mode=compiled_mode), self.assertRaisesRegex(
                    ValueError, "already compiled with Inductor"
                ):
                    passes.ACLGraphPass().run(graph, PassConfig(aclgraph_mode=mode))
        with self.assertRaisesRegex(ValueError, "use_joint_graph"):
            CompileConfig(aclgraph_mode="static")
        config = SimpleNamespace(
            compile=CompileConfig(enabled=True, use_joint_graph=True, aclgraph_mode="dynamic",
                                  aclgraph_max_capture_regions=512),
            fsdp_config=SimpleNamespace(dp_shard_size=1, edp_shard_size=1),
        )
        compiler = GraphCompiler(torch.nn.Linear(2, 2), lambda model, value: model(value).sum(),
                                 trainer_config=config, device=torch.device("cpu"))
        self.assertEqual(compiler.pass_config.aclgraph_mode, "dynamic")
        self.assertEqual(compiler.pass_config.aclgraph_max_capture_regions, 512)

    def test_compiler_executes_replacement_graph(self):
        """The compiler must use the graph returned by a partitioning pass."""
        model = torch.nn.Linear(2, 2)
        compiler = GraphCompiler(model, lambda module, value: module(value).sum(),
                                 pass_config=PassConfig(fsdp_enabled=False), device=torch.device("cpu"))
        transformed = []

        def _replace(graph, **kwargs):
            del kwargs
            result = copy.deepcopy(graph)
            result.meta["replacement"] = True
            transformed.append(result)
            return result

        with patch("hyper_parallel.compile.compiler.PassPipeline.from_config") as factory:
            factory.return_value.run.side_effect = _replace
            compiler.forward_backward(value=torch.ones(2, 2))
        self.assertIs(compiler._joint_graph.graph_module, transformed[0])
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
        with patch("hyper_parallel.compile.compiler.teardown_aclgraph") as teardown:
            compiler.close()
            teardown.assert_called_once_with(transformed[0])
        trainer = GraphTrainer.__new__(GraphTrainer)
        trainer._compiler = Mock()
        trainer.close()
        trainer._compiler.close.assert_called_once_with()
