#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.ir as ir

from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import List


_current_backedge_builder = ContextVar("current_bb")


class UnconnectedSignalError(RuntimeError):
  def __init__(self, module: str, port_names: List[str]):
    super().__init__(
        f"Ports {port_names} unconnected in design module {module}.")


class BackedgeBuilder(AbstractContextManager):

  class Edge:
    def __init__(self, creator, type: ir.Type, port_name: str,
                 op_view, instance_of: ir.Operation):
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

  def _create(self, type: ir.Type, port_name: str,
              op_view, instance_of: ir.Operation = None):
    edge = BackedgeBuilder.Edge(self, type, port_name, op_view, instance_of)
    self.edges.add(edge)
    return edge

  def __enter__(self):
    self.old_bb_token = _current_backedge_builder.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
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


class BuilderValue:
  """Class that holds a value, as well as builder and index of this value in
     the operand or result list. This can represent an OpOperand and index into
     OpOperandList or a OpResult and index into an OpResultList"""

  def __init__(self, value, builder, index):
    self.value = value
    self.builder = builder
    self.index = index


class NamedValueBuilder:
  """Helper class to incrementally construct an instance of an operation that
     names its operands and results"""

  def __init__(self, opview, operand_indices, result_indices, backedges):
    self.opview = opview
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

  @property
  def operation(self):
    """Get the operation associated with this builder."""
    return self.opview.operation
