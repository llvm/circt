#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from circt.support import BuilderValue, BackedgeBuilder
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


class Output:
  """Represents an output port on a design module. Call the 'set' method to set
  the output value during implementation."""

  __slots__ = [
      "name",
      "type",
      "value"
  ]

  def __init__(self, type: mlir.ir.Type, name: str = None):
    self.name: str = name
    self.type: mlir.ir.Type = type
    self.value: mlir.ir.Value = None

  def set(self, val):
    """Sets the final output signal. Should only be called by the implementation
    code."""
    if isinstance(val, mlir.ir.Value):
      self.value = val
    else:
      self.value = val.result


class Input:
  __slots__ = [
      "name",
      "type",
      "value"
  ]

  def __init__(self, type: mlir.ir.Type, name: str = None):
    self.name = name
    self.type = type
    self.value = None


def module(cls):
  """The CIRCT design entry module class decorator."""

  class __Module(cls, mlir.ir.OpView):
    OPERATION_NAME = "circt.design_entry." + cls.__name__
    _ODS_REGIONS = (0, True)

    def __init__(self, *args, **kwargs):

      # Copy the classmember Input/Output declarations to be instance members.
      for attr_name in dir(cls):
        attr = self.__getattribute__(attr_name)
        if isinstance(attr, Input):
          self.__setattr__(attr_name, Input(attr.type, attr_name))
        if isinstance(attr, Output):
          self.__setattr__(attr_name, Output(attr.type, attr_name))

      cls.__init__(self, *args, **kwargs)

      # After the wrapped class' construct, all the IO should be known.
      input_ports = list[Input]()
      output_ports = list[Output]()
      dont_touch = set([x for x in dir(mlir.ir.OpView)])
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

      for input in input_ports:
        if input.name in kwargs:
          input.value = kwargs[input.name]
          if isinstance(input.value, mlir.ir.OpView):
            input.value = input.value.operation.result
          elif isinstance(input.value, mlir.ir.Operation):
            input.value = input.value.result
          assert isinstance(input.value, mlir.ir.Value)
        else:
          input.value = BackedgeBuilder.current().create(
              input.type, input.name, self).result

      mlir.ir.OpView.__init__(self, self.build_generic(
          attributes={},
          results=[x.type for x in output_ports],
          operands=[x.value for x in input_ports]
      ))

      # # This function sets up the inputs to construct, then connects the
      # # outputs.
      # def body_build(mod):
      #   inputs = dict()
      #   for index, (name, _) in enumerate(input_ports):
      #     inputs[name] = mod.entry_block.arguments[index]

      #   self.construct(**inputs)

      #   outputs = []
      #   unconnected_ports = []
      #   for (name, output) in output_ports:
      #     if output.value is None:
      #       unconnected_ports.append(name)
      #     outputs.append(output.value)
      #   if len(unconnected_ports) > 0:
      #     raise UnconnectedSignalError(cls.__name__, unconnected_ports)
      #   hw.OutputOp(outputs)

      # # Construct things as HWModules.
      # self.module = hw.HWModuleOp(
      #     name=cls.__name__,
      #     input_ports=[(name, port.type) for (name, port) in input_ports],
      #     output_ports=[(name, port.type) for (name, port) in output_ports],
      #     body_builder=body_build)

  return __Module


def connect(destination, source):
  if not isinstance(destination, BuilderValue):
    raise TypeError(
        f"cannot connect to destination of type {type(destination)}")
  if not isinstance(source, BuilderValue) and not isinstance(
      source, mlir.ir.Value) and not (isinstance(source, mlir.ir.OpView) and
                                      hasattr(source, "result")):
    raise TypeError(f"cannot connect from source of type {type(source)}")
  builder = destination.builder
  index = destination.index
  if isinstance(source, BuilderValue):
    value = source.value
  elif isinstance(source, mlir.ir.Value):
    value = source
  elif isinstance(source, mlir.ir.OpView):
    value = source.result

  builder.operation.operands[index] = value
  if index in builder.backedges:
    builder.backedges[index].erase()
