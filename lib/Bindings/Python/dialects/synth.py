#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import synth, hw
from ._synth_ops_gen import *
from .._mlir_libs._circt._synth import _LongestPathAnalysis, _LongestPathCollection, _LongestPathDataflowPath, _LongestPathHistory, _LongestPathObject

from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ============================================================================
# Core Data Structures for Synth Longest Path Analysis
# ============================================================================


@dataclass
class Instance:
  """
    Represents a single element in a hierarchical instance path.
    In hardware design, modules are instantiated hierarchically. This class
    represents one level in that hierarchy, containing both the module type
    and the specific instance name.
    Attributes:
        instance_name: The name of this specific instance
        module_name: The type/name of the module being instantiated
    """
  _instance: hw.InstanceOp

  def __init__(self, instance: hw.InstanceOp):
    self._instance = instance

  @property
  def instance_name(self) -> str:
    return self._instance.attributes["instanceName"].value

  @property
  def module_name(self) -> str:
    return self._instance.attributes["moduleName"].value


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

  _object: _LongestPathObject

  # TODO: Associate with an MLIR value/op

  def __str__(self) -> str:
    """
        Generate a human-readable string representation of this object.
        Format: "module1:instance1/module2:instance2 signal_name[bit_pos]"
        """
    path = "/".join(f"{elem.module_name}:{elem.instance_name}"
                    for elem in self.instance_path)
    return f"{path} {self.name}[{self.bit_pos}]"

  def __repr__(self) -> str:
    return f"Object({self.instance_path}, {self.name}, {self.bit_pos})"

  @property
  def instance_path(self) -> List[Instance]:
    """Get the hierarchical instance path to this object."""
    operations = self._object.instance_path

    return [Instance(op) for op in operations]

  @property
  def name(self) -> str:
    """Get the name of this signal/port."""
    return self._object.name

  @property
  def bit_pos(self) -> int:
    """Get the bit position for multi-bit signals."""
    return self._object.bit_pos


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
class DataflowPath:
  """
    Represents a complete dataflow path from end point to start point.
    A dataflow path captures the complete timing path through a circuit,
    from an output point (end-point) back to an input point (start-point), including
    all intermediate debug points and the total delay.
    Attributes:
        end_point: The output signal/object where this path ends
        path: The OpenPath containing the detailed path information
        root: The root module name for this analysis
    """

  _path: _LongestPathDataflowPath

  @property
  def delay(self) -> int:
    """Get the total delay of this path in timing units."""
    return self._path.delay

  @property
  def start_point(self) -> Object:
    """Get the input signal/object where this path begins."""
    return Object(self._path.start_point)

  @property
  def end_point(self) -> Object:
    """Get the output signal/object where this path ends."""
    return Object(self._path.end_point)

  @property
  def history(self) -> List[DebugPoint]:
    """Get the history of debug points along this path."""
    return [i for i in LongestPathHistory(self._path.history)]

  @property
  def root(self) -> str:
    """Get the root module name for this analysis."""
    return self._path.root.attributes["sym_name"].value

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
    start_point_hierarchy = self._build_hierarchy_string(
        self.start_point, prefix)
    end_point_hierarchy = self._build_hierarchy_string(self.end_point, prefix)

    # Track current position and delay for incremental output
    current_hierarchy = start_point_hierarchy
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

    # Add final segment to end-point if there's remaining delay
    if current_delay != self.delay:
      final_delay = self.delay - current_delay
      trace.append(f"{end_point_hierarchy} {final_delay}")

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
      parts.append(f"{elem.instance_name}:{elem.module_name}")

    # Add the signal name with bit position if applicable
    signal_part = obj.name
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
    """

  def __init__(self, collection):
    """
        Initialize the collection wrapper.
        Args:
            collection: The underlying C++ LongestPathCollection object
        """
    self.collection = collection
    self.length = self.collection.get_size()

  # ========================================================================
  # Collection Interface Methods
  # ========================================================================

  def __len__(self) -> int:
    """Get the number of paths in the collection."""
    return self.length

  def __getitem__(
      self, index: Union[slice,
                         int]) -> Union[DataflowPath, List[DataflowPath]]:
    """
        Get a specific path from the collection by index.
        Supports both integer and slice indexing. Integer indices can be negative.

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

    return DataflowPath(self.collection.get_path(index))

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

  def merge(self, src: "LongestPathCollection"):
    """
        Merge another collection into this one.
        Args:
            src: The collection to merge into this one
    """
    self.collection.merge(src.collection)
    # Re-initialize to reflect the merged collection
    self.__init__(self.collection)


# ============================================================================
# Main Analysis Interface
# ============================================================================


class LongestPathAnalysis:
  """
    Main interface for performing longest path analysis on Synth dialect.
    This class provides a Python wrapper around the C++ LongestPathAnalysis,
    enabling timing analysis of Synth dialect. It can identify critical timing
    paths and provide detailed path information.
    Attributes:
        analysis: The underlying C++ analysis object
    """

  def __init__(self,
               module,
               collect_debug_info: bool = True,
               keep_only_max_delay_paths: bool = False,
               lazy_computation: bool = False,
               top_module_name: str = ""):
    """
        Initialize the longest path analysis for a given module.
        Args:
            module: The MLIR module to analyze
            collect_debug_info: Whether to include debug points in the analysis.
                                Debug points provide additional information about the path,
                                but increase analysis time and memory usage.
            keep_only_max_delay_paths: Keep only maximum-delay paths in collections.
            lazy_computation: Enable lazy (on-demand) computation.
        """
    self.analysis = synth._LongestPathAnalysis(module, collect_debug_info,
                                               keep_only_max_delay_paths,
                                               lazy_computation,
                                               top_module_name)

  def get_paths(self,
                value,
                bit_pos: int,
                elaborate_paths: bool = True) -> LongestPathCollection:
    """
        Perform longest path analysis and return all timing paths to the
        specified value and bit position.
        Args:
            value: The value to analyze
            bit_pos: The bit position to analyze
            elaborate_paths: Whether to elaborate the paths with detailed information
        Returns:
            LongestPathCollection containing all paths sorted by delay
      """
    return LongestPathCollection(
        self.analysis.get_paths(value, bit_pos, elaborate_paths))

  def get_internal_paths(self,
                         module_name: str,
                         elaborate_paths: bool = True) -> LongestPathCollection:
    """
        Get internal paths within the module.

        Internal paths are paths that start and end at sequential elements
        (registers/flip-flops), forming complete paths through combinational
        logic. These paths may cross module boundaries but both endpoints are
        sequential elements, not ports.

        Args:
            module_name: Name of the module to analyze
            elaborate_paths: Whether to include full hierarchical instance information

        Returns:
            LongestPathCollection containing all internal paths sorted by delay
        """
    return LongestPathCollection(
        self.analysis.get_internal_paths(module_name, elaborate_paths))

  def get_paths_from_input_ports_to_internal(
      self, module_name: str) -> LongestPathCollection:
    """
        Get external paths from module input ports to internal sequential elements.

        These are input-to-register paths that start at module input ports and end
        at internal sequential elements (registers/flip-flops).

        Args:
            module_name: Name of the module to analyze

        Returns:
            LongestPathCollection containing input-to-internal paths sorted by delay
        """
    return LongestPathCollection(
        self.analysis.get_paths_from_input_ports_to_internal(module_name))

  def get_paths_from_internal_to_output_ports(
      self, module_name: str) -> LongestPathCollection:
    """
        Get external paths from internal sequential elements to module output ports.

        These are register-to-output paths that start at internal sequential elements
        (registers/flip-flops) and end at module output ports.

        Args:
            module_name: Name of the module to analyze

        Returns:
            LongestPathCollection containing internal-to-output paths sorted by delay
        """
    return LongestPathCollection(
        self.analysis.get_paths_from_internal_to_output_ports(module_name))

  def get_all_paths(self,
                    module_name: str,
                    elaborate_paths: bool = True) -> LongestPathCollection:
    """
        Get all paths in the module (internal and external paths combined).

        Args:
            module_name: Name of the module to analyze
            elaborate_paths: Whether to include full hierarchical instance information
                           (only applies to internal paths)

        Returns:
            LongestPathCollection containing all paths sorted by delay
        """
    collection = self.get_internal_paths(module_name, elaborate_paths)
    collection.merge(self.get_paths_from_input_ports_to_internal(module_name))
    collection.merge(self.get_paths_from_internal_to_output_ports(module_name))
    return collection


@dataclass
class LongestPathHistory:
  """
    Represents the history of a timing path, including intermediate debug points.
    This class provides a Python wrapper around the C++ LongestPathHistory,
    enabling iteration over the path's history and access to debug points.
    Attributes:
        history: The underlying C++ history object
    """
  history: _LongestPathHistory

  def __iter__(self):
    """Iterate over the debug points in the history."""
    while not self.history.empty:
      object, delay, comment = self.history.head
      yield DebugPoint(Object(object), delay, comment)
      self.history = self.history.tail
