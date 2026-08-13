#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TaskConfig:
    scenario: Dict[str, Any] = field(default_factory=dict)
    reward: Dict[str, Any] = field(default_factory=dict)
    include_graph_specs: bool = True
    group_name: str = "leo"
    graph_max_edges: Optional[int] = None
