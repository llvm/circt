#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import aig
from ._aig_ops_gen import *
from .._mlir_libs._circt._aig import _LongestPathAnalysis, _LongestPathCollection

import json
from dataclasses import dataclass
from typing import Any, Dict, List

# ============================================================================
# Core Data Structures for AIG Path Analysis
# ============================================================================


@dataclass
class InstancePathElement:
  """
    Represents a single element in a hierarchical instance path.
    In hardware design, modules are instantiated hierarchically. This class
    represents one level in that hierarchy, containing both the module type
    and the specific instance name.
    Attributes:
        instance_name: The name of this specific instance
        module_name: The type/name of the module being instantiated
    """

  instance_name: str
  module_name: str

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "InstancePathElement":
    """Create an InstancePathElement from a dictionary representation."""
    return cls(instance_name=data["instance_name"],
               module_name=data["module_name"])


@dataclass
class Object:
  """
    Represents a signal or port object in the dataflow graph.
    This class encapsulates a specific signal within the hardware hierarchy,
    including its location in the instance hierarchy, signal name, and bit position
    for multi-bit signals.
    Attributes:
        instance_path: Hierarchical path to the module containing this object
        name: The signal/port name within the module
        bit_pos: Bit position for multi-bit signals (0 for single-bit)
    """

  instance_path: List[InstancePathElement]
  name: str
  bit_pos: int

  # TODO: Associate with an MLIR value/op

  def __str__(self) -> str:
    """
        Generate a human-readable string representation of this object.
        Format: "module1:instance1/module2:instance2 signal_name[bit_pos]"
        """
    path = "/".join(f"{elem.module_name}:{elem.instance_name}"
                    for elem in self.instance_path)
    return f"{path} {self.name}[{self.bit_pos}]"

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Object":
    """Create an Object from a dictionary representation."""
    instance_path = [
        InstancePathElement.from_dict(elem) for elem in data["instance_path"]
    ]
    return cls(instance_path=instance_path,
               name=data["name"],
               bit_pos=data["bit_pos"])


@dataclass
class DebugPoint:
  """
    Represents a debug point in the timing path history.
    Debug points are intermediate points along a timing path that provide
    insight into the delay accumulation and signal propagation through
    the circuit. Each point captures the state at a specific location.
    Attributes:
        object: The signal/object at this debug point
        delay: Accumulated delay up to this point (in timing units)
        comment: Optional descriptive comment about this point
    """

  object: Object
  delay: int
  comment: str

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "DebugPoint":
    """Create a DebugPoint from a dictionary representation."""
    return cls(
        object=Object.from_dict(data["object"]),
        delay=data["delay"],
        comment=data["comment"],
    )


@dataclass
class OpenPath:
  """
    Represents an open timing path with detailed history.
    An open path represents a timing path that hasn't reached its final
    destination yet. It contains the current fan-in point, accumulated delay,
    and a history of debug points showing how the signal propagated.
    Attributes:
        fan_in: The input signal/object where this path segment begins
        delay: Total accumulated delay for this path segment
        history: Chronological list of debug points along the path
    """

  fan_in: Object
  delay: int
  history: List[DebugPoint]

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "OpenPath":
    """Create an OpenPath from a dictionary representation."""
    history = [DebugPoint.from_dict(point) for point in data["history"]]
    return cls(
        fan_in=Object.from_dict(data["fan_in"]),
        delay=data["delay"],
        history=history,
    )


@dataclass
class DataflowPath:
  """
    Represents a complete dataflow path from fan-out to fan-in.
    A dataflow path captures the complete timing path through a circuit,
    from an output point (fan-out) back to an input point (fan-in), including
    all intermediate debug points and the total delay.
    Attributes:
        fan_out: The output signal/object where this path ends
        path: The OpenPath containing the detailed path information
        root: The root module name for this analysis
    """

  fan_out: Object  # Output endpoint of the path
  path: OpenPath  # Detailed path information with history
  root: str  # Root module name

  # ========================================================================
  # Factory Methods for Object Creation
  # ========================================================================

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "DataflowPath":
    """Create a DataflowPath from a dictionary representation."""
    return cls(
        fan_out=Object.from_dict(data["fan_out"]),
        path=OpenPath.from_dict(data["path"]),
        root=data["root"],
    )

  @classmethod
  def from_json_string(cls, json_str: str) -> "DataflowPath":
    """Create a DataflowPath from a JSON string representation."""
    data = json.loads(json_str)
    return cls.from_dict(data)

  @property
  def delay(self) -> int:
    """Get the total delay of this path in timing units."""
    return self.path.delay

  @property
  def fan_in(self) -> "DataflowPath":
    """Get the input signal/object where this path begins."""
    return self.path.fan_in

  @property
  def history(self) -> List[DebugPoint]:
    """Get the history of debug points along this path."""
    return self.path.history

  # ========================================================================
  # Visualization and Analysis Methods
  # ========================================================================

  def to_flamegraph(self) -> str:
    """
        Convert this path to FlameGraph format for visualization.
        FlameGraphs are a visualization technique that shows call stacks or
        in this case, timing paths through the circuit hierarchy. Each line
        represents a segment of the path with its associated delay.
        The format is: "hierarchy_path delay_increment"
        where hierarchy_path uses semicolons to separate hierarchy levels.
        Returns:
            String in FlameGraph format showing the timing path progression
        """
    trace = []
    prefix = f"top:{self.root}"

    # Build hierarchy strings for start and end points
    fan_in_hierarchy = self._build_hierarchy_string(self.fan_in, prefix)
    fan_out_hierarchy = self._build_hierarchy_string(self.fan_out, prefix)

    # Track current position and delay for incremental output
    current_hierarchy = fan_in_hierarchy
    current_delay = 0

    # Process debug history points in reverse order (from input to output)
    for debug_point in self.history[::-1]:
      history_hierarchy = self._build_hierarchy_string(debug_point.object,
                                                       prefix)
      if history_hierarchy:
        # Add segment from current position to this debug point
        delay_increment = debug_point.delay - current_delay
        trace.append(f"{current_hierarchy} {delay_increment}")

        # Update current position
        current_hierarchy = history_hierarchy
        current_delay = debug_point.delay

    # Add final segment to fan-out if there's remaining delay
    if current_delay != self.delay:
      final_delay = self.delay - current_delay
      trace.append(f"{fan_out_hierarchy} {final_delay}")

    return "\n".join(trace)

  def _build_hierarchy_string(self, obj: Object, prefix: str = "") -> str:
    """
        Build a hierarchical string representation of an Object for FlameGraph format.
        This method constructs a semicolon-separated hierarchy string that represents
        the full path from the top-level module down to the specific signal. This
        format is compatible with FlameGraph visualization tools.
        Args:
            obj: Object to represent in the hierarchy
            prefix: Top-level prefix (typically "top:module_name")
        Returns:
            Hierarchical string in format: "top:root;module1:inst1;module2:inst2;signal[bit]"
        """
    parts = [prefix]

    # Add each level of the instance hierarchy
    for elem in obj.instance_path:
      parts.append(f"{elem.module_name}:{elem.instance_name}")

    # Add the signal name with bit position if applicable
    signal_part = obj.name
    if obj.bit_pos > 0:
      signal_part += f"[{obj.bit_pos}]"
    parts.append(signal_part)

    return ";".join(parts)


# ============================================================================
# Analysis Collection Classes
# ============================================================================


class LongestPathCollection:
  """
    A collection of timing paths sorted by delay (longest first).
    This class provides a Python wrapper around the C++ LongestPathCollection,
    offering convenient access to timing paths with caching for performance.
    The paths are pre-sorted by delay in descending order.
    Attributes:
        collection: The underlying C++ collection object
        length: Number of paths in the collection
        cache: Cache for parsed DataflowPath objects to avoid re-parsing JSON
    """

  def __init__(self, collection):
    """
        Initialize the collection wrapper.
        Args:
            collection: The underlying C++ LongestPathCollection object
        """
    self.collection = collection
    self.length = self.collection.get_size()
    self.cache = [None for _ in range(self.length)]

  # ========================================================================
  # Collection Interface Methods
  # ========================================================================

  def __len__(self) -> int:
    """Get the number of paths in the collection."""
    return self.length

  def __getitem__(self, index):
    """
        Get a specific path from the collection by index.
        Supports both integer and slice indexing. Integer indices can be negative.
        Results are cached to avoid expensive JSON parsing on repeated access.

        Args:
            index: Integer index or slice object to access paths

        Returns:
            DataflowPath or list of DataflowPaths for slice access

        Raises:
            IndexError: If index is out of range
        """
    if isinstance(index, slice):
      return [self[i] for i in range(*index.indices(len(self)))]

    # Handle negative indexing
    if index < 0:
      index += self.length
    if index < 0 or index >= self.length:
      raise IndexError("Index out of range")

    # Use cache to avoid expensive JSON parsing
    if self.cache[index] is not None:
      return self.cache[index]

    # Parse JSON and cache the result
    json_str = self.collection.get_path(index)
    self.cache[index] = DataflowPath.from_json_string(json_str)
    return self.cache[index]

  # ========================================================================
  # Analysis and Query Methods
  # ========================================================================

  @property
  def longest_path(self) -> DataflowPath:
    """Get the path with the maximum delay (first element since sorted)."""
    return self[0]

  def get_by_delay_ratio(self, ratio: float) -> DataflowPath:
    """
        Get the path at the specified position in the delay-sorted collection.
        Since paths are sorted by delay in descending order, higher ratios
        correspond to paths with higher delays (closer to the critical path).
        Args:
            ratio: Position ratio between 0.0 and 1.0
                  (e.g., 1.0 = longest delay path, 0.0 = shortest delay path,
                   0.95 = path among the top 5% slowest paths)
        Returns:
            DataflowPath at the specified position ratio
        """
    assert ratio >= 0.0 and ratio <= 1.0, "Ratio must be between 0.0 and 1.0"
    index = int(len(self) * (1 - ratio))
    return self[index]

  def print_summary(self) -> None:
    """Print a statistical summary of path delays in the collection."""
    print(f"Total paths: {len(self)}")
    print(f"Max delay: {self.longest_path.delay}")
    print(f"Min delay: {self[-1].delay}")
    print(f"50th percentile delay: {self.get_by_delay_ratio(0.5).delay}")
    print(f"90th percentile delay: {self.get_by_delay_ratio(0.9).delay}")
    print(f"95th percentile delay: {self.get_by_delay_ratio(0.95).delay}")
    print(f"99th percentile delay: {self.get_by_delay_ratio(0.99).delay}")
    print(f"99.9th percentile delay: {self.get_by_delay_ratio(0.999).delay}")


# ============================================================================
# Main Analysis Interface
# ============================================================================


class LongestPathAnalysis:
  """
    Main interface for performing longest path analysis on AIG circuits.
    This class provides a Python wrapper around the C++ LongestPathAnalysis,
    enabling timing analysis of AIG (And-Inverter Graph) circuits. It can
    identify critical timing paths and provide detailed path information.
    Attributes:
        analysis: The underlying C++ analysis object
    """

  def __init__(self, module, trace_debug_points: bool = True):
    """
        Initialize the longest path analysis for a given module.
        Args:
            module: The MLIR module to analyze
            trace_debug_points: Whether to include debug points in the analysis.
                              The debug points provide additional information about the path,
                              but increase the analysis time and memory usage.
        """
    self.analysis = aig._LongestPathAnalysis(module, trace_debug_points)

  def get_all_paths(self, module_name: str) -> LongestPathCollection:
    """
        Perform longest path analysis and return all timing paths.
        This method analyzes the specified module and returns a collection
        of all timing paths, sorted by delay in descending order.
        Args:
            module_name: Name of the module to analyze
        Returns:
            LongestPathCollection containing all paths sorted by delay
        """
    return LongestPathCollection(self.analysis.get_all_paths(module_name, True))
