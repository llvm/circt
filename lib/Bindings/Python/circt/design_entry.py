#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from circt.support import BuilderValue, BackedgeBuilder, OpOperand
import circt

import mlir.ir

import atexit

# Push a default context onto the context stack at import time.
DefaultContext = mlir.ir.Context()
DefaultContext.__enter__()
circt.register_dialects(DefaultContext)
DefaultContext.allow_unregistered_dialects = True


@atexit.register
def __exit_ctxt():
  DefaultContext.__exit__(None, None, None)


# Until we get source location based on Python stack traces, default to unknown
# locations.
DefaultLocation = mlir.ir.Location.unknown()
DefaultLocation.__enter__()


@atexit.register
def __exit_loc():
  DefaultLocation.__exit__(None, None, None)


class ModuleDecl:
  """Represents an input or output port on a design module."""

  __slots__ = [
      "name",
      "type"
  ]

  def __init__(self, type: mlir.ir.Type, name: str = None):
    self.name: str = name
    self.type: mlir.ir.Type = type


class Output(ModuleDecl):
  pass


class Input(ModuleDecl):
  pass


def module(cls):
  """The CIRCT design entry module class decorator."""

  class __Module(cls, mlir.ir.OpView):
    OPERATION_NAME = "circt.design_entry." + cls.__name__
    _ODS_REGIONS = (0, True)

    # Default mappings to operand/result numbers.
    input_ports = dict[str, int]()
    output_ports = dict[str, int]()

    def __init__(self, *args, **kwargs):
      """Scan the class and eventually instance for Input/Output members and
      treat the inputs as operands and outputs as results."""
      cls.__init__(self, *args, **kwargs)

      # The OpView attributes cannot be touched before OpView is constructed.
      # Get a list and don't touch them.
      dont_touch = set([x for x in dir(mlir.ir.OpView)])

      # After the wrapped class' construct, all the IO should be known.
      input_ports = list[Input]()
      output_ports = list[Output]()
      # Scan for them.
      for attr_name in dir(self):
        if attr_name in dont_touch:
          continue
        attr = self.__getattribute__(attr_name)
        if isinstance(attr, Input):
          attr.name = attr_name
          input_ports.append(attr)
        if isinstance(attr, Output):
          attr.name = attr_name
          output_ports.append(attr)

      # Build a list of operand values for the operation we're gonna create.
      input_ports_values = list[mlir.ir.Value]()
      for input in input_ports:
        if input.name in kwargs:
          value = kwargs[input.name]
          if isinstance(value, mlir.ir.OpView):
            value = value.operation.result
          elif isinstance(value, mlir.ir.Operation):
            value = value.result
          assert isinstance(value, mlir.ir.Value)
        else:
          value = BackedgeBuilder.current().create(
              input.type, input.name, self).result
        input_ports_values.append(value)

      # Init the OpView, which creates the operation.
      mlir.ir.OpView.__init__(self, self.build_generic(
          attributes={},
          results=[x.type for x in output_ports],
          operands=[x for x in input_ports_values]
      ))

      # Build the mappings for __getattribute__.
      self.input_ports = {port.name: i for i, port in enumerate(input_ports)}
      self.output_ports = {port.name: i for i, port in enumerate(output_ports)}

    def __getattribute__(self, name: str):
      # Base case.
      if name == "input_ports" or name == "output_ports" or \
         name == "operands" or name == "results":
        return super().__getattribute__(name)

      # To emulate OpView, if 'name' is either an input or output port,
      # redirect.
      if name in self.input_ports:
        op_num = self.input_ports[name]
        operand = self.operands[op_num]
        return OpOperand(self, op_num, operand)
      if name in self.output_ports:
        op_num = self.output_ports[name]
        return self.results[op_num]
      return super().__getattribute__(name)

  return __Module


def connect(destination, source):
  if not isinstance(destination, OpOperand):
    raise TypeError(
        f"cannot connect to destination of type {type(destination)}")
  if not isinstance(source, OpOperand) and not isinstance(
      source, mlir.ir.Value) and not (isinstance(source, mlir.ir.Operation) and
                                      hasattr(source, "result")):
    raise TypeError(f"cannot connect from source of type {type(source)}")
  index = destination.index
  if isinstance(source, OpOperand):
    value = source.value
  elif isinstance(source, mlir.ir.Value):
    value = source
  elif isinstance(source, mlir.ir.OpView):
    value = source.result

  destination.operation.operands[index] = value
  if isinstance(destination, BuilderValue) and \
     index in destination.builder.backedges:
    destination.builder.backedges[index].erase()
