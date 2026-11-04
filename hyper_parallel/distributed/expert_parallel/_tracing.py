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
"""FX scope shared by dynamically routed expert and embedding exchanges."""

from contextlib import contextmanager
from typing import Iterator

from torch.fx.traceback import get_current_meta


@contextmanager
def expert_parallel_region() -> Iterator[None]:
    """Annotate dynamic EP work without requiring newer Torch annotation APIs."""
    metadata = get_current_meta()
    previous = metadata.get("custom")
    metadata["custom"] = {**(previous or {}), "expert_parallel": True}
    try:
        yield
    finally:
        if previous is None:
            metadata.pop("custom", None)
        else:
            metadata["custom"] = previous
