#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.ir as ir

from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import List

_current_backedge_builder = ContextVar("current_bb")


class ConnectionError(RuntimeError):
  pass


class UnconnectedSignalError(ConnectionError):

  def __init__(self, module: str, port_names: List[str]):
    super().__init__(
        f"Ports {port_names} unconnected in design module {module}.")


def get_value(obj) -> ir.Value:
  """Resolve a Value from a few supported types."""

  if isinstance(obj, ir.Value):
    return obj
  if isinstance(obj, ir.Operation):
    return obj.result
  if isinstance(obj, ir.OpView):
    return obj.result
  if isinstance(obj, OpOperand):
    return obj.value
  return None


def connect(destination, source):
  """A convenient way to use BackedgeBuilder."""
  if not isinstance(destination, OpOperand):
    raise TypeError(
        f"cannot connect to destination of type {type(destination)}")
  value = get_value(source)
  if value is None:
    raise TypeError(f"cannot connect from source of type {type(source)}")

  index = destination.index
  destination.operation.operands[index] = value
  if isinstance(destination, BuilderValue) and \
     index in destination.builder.backedges:
    destination.builder.backedges[index].erase()


def var_to_attribute(obj, none_on_fail: bool = False) -> ir.Attribute:
  """Create an MLIR attribute from a Python object for a few common cases."""
  if isinstance(obj, ir.Attribute):
    return obj
  if isinstance(obj, int):
    attrTy = ir.IntegerType.get_signless(64)
    return ir.IntegerAttr.get(attrTy, obj)
  if isinstance(obj, str):
    return ir.StringAttr.get(obj)
  if isinstance(obj, list):
    return ir.ArrayAttr.get([var_to_attribute(x) for x in obj])
  if none_on_fail:
    return None
  raise TypeError(f"Cannot convert type '{type(obj)}' to MLIR attribute")


class BackedgeBuilder(AbstractContextManager):

  class Edge:

    def __init__(self, creator, type: ir.Type, port_name: str, op_view,
                 instance_of: ir.Operation):
      self.creator: BackedgeBuilder = creator
      self.dummy_op = ir.Operation.create("TemporaryBackedge", [type])
      self.instance_of = instance_of
      self.op_view = op_view
      self.port_name = port_name
      self.erased = False

    @property
    def result(self):
      return self.dummy_op.result

    def erase(self):
      if self.erased:
        return
      if self in self.creator.edges:
        self.creator.edges.remove(self)
        self.dummy_op.operation.erase()

  def __init__(self):
    self.edges = set()

  @staticmethod
  def current():
    bb = _current_backedge_builder.get(None)
    if bb is None:
      raise RuntimeError("No backedge builder found in context!")
    return bb

  @staticmethod
  def create(*args, **kwargs):
    return BackedgeBuilder.current()._create(*args, **kwargs)

  def _create(self,
              type: ir.Type,
              port_name: str,
              op_view,
              instance_of: ir.Operation = None):
    edge = BackedgeBuilder.Edge(self, type, port_name, op_view, instance_of)
    self.edges.add(edge)
    return edge

  def __enter__(self):
    self.old_bb_token = _current_backedge_builder.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_backedge_builder.reset(self.old_bb_token)
    errors = []
    for edge in list(self.edges):
      # TODO: Make this use `UnconnectedSignalError`.
      msg = "Port:       " + edge.port_name + "\n"
      if edge.instance_of is not None:
        msg += "InstanceOf: " + str(edge.instance_of).split(" {")[0] + "\n"
      if edge.op_view is not None:
        op = edge.op_view.operation
        msg += "Instance:   " + str(op)
        op.erase()
      edge.erase()

      # Clean up the IR and Python references.
      errors.append(msg)

    if errors:
      errors.insert(0, "Uninitialized ports remain in circuit!")
      raise RuntimeError("\n".join(errors))


class OpOperand:
  __slots__ = ["index", "operation", "value"]

  def __init__(self, operation, index, value):
    self.index = index
    self.operation = operation
    self.value = value


# Are there any situations in which this needs to be used to index into results?
class BuilderValue(OpOperand):
  """Class that holds a value, as well as builder and index of this value in
     the operand or result list. This can represent an OpOperand and index into
     OpOperandList or a OpResult and index into an OpResultList"""

  def __init__(self, value, builder, index):
    super().__init__(builder.operation, index, value)
    self.builder = builder


class NamedValueOpView:
  """Helper class to incrementally construct an instance of an operation that
     names its operands and results"""

  def __init__(self,
               cls,
               data_type,
               input_port_mapping={},
               pre_args=[],
               post_args=[],
               **kwargs):
    # Set result_indices to name each result.
    result_names = self.result_names()
    result_indices = {}
    for i in range(len(result_names)):
      result_indices[result_names[i]] = i

    # Set operand_indices to name each operand. Give them an initial value,
    # either from input_port_mapping or a default value.
    backedges = {}
    operand_indices = {}
    operand_values = []
    operand_names = self.operand_names()
    for i in range(len(operand_names)):
      arg_name = operand_names[i]
      operand_indices[arg_name] = i
      if arg_name in input_port_mapping:
        value = get_value(input_port_mapping[arg_name])
        operand = value
      else:
        backedge = self.create_default_value(i, data_type, arg_name)
        backedges[i] = backedge
        operand = backedge.result
      operand_values.append(operand)

    # Some ops take a list of operand values rather than splatting them out.
    if isinstance(data_type, list):
      operand_values = [operand_values]

    self.opview = cls(data_type, *pre_args, *operand_values, *post_args,
                      **kwargs)
    self.operand_indices = operand_indices
    self.result_indices = result_indices
    self.backedges = backedges

  def __getattr__(self, name):
    # Check for the attribute in the arg name set.
    if name in self.operand_indices:
      index = self.operand_indices[name]
      value = self.opview.operands[index]
      return BuilderValue(value, self, index)

    # Check for the attribute in the result name set.
    if name in self.result_indices:
      index = self.result_indices[name]
      value = self.opview.results[index]
      return BuilderValue(value, self, index)

    # If we fell through to here, the name isn't a result.
    raise AttributeError(f"unknown port name {name}")

  def create_default_value(self, index, data_type, arg_name):
    return BackedgeBuilder.create(data_type, arg_name, self)

  @property
  def operation(self):
    """Get the operation associated with this builder."""
    return self.opview.operation
