#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from circt.dialects import hw
from .support import UnconnectedSignalError

import mlir.ir


class Output:
  """Represents an output port on a design module. Call the 'set' method to set
  the output value during implementation."""

  __slots__ = [
      "type",
      "value"
  ]

  def __init__(self, type: mlir.ir.Type):
    self.type = type
    self.value = None

  def set(self, val):
    """Sets the final output signal. Should only be called by the implementation
    code."""
    if type(val) is mlir.ir.OpResult:
      self.value = val
    else:
      self.value = val.result


class Input:
  __slots__ = [
      "type"
  ]

  def __init__(self, type: mlir.ir.Type):
    self.type = type


def module(cls):
  """The CIRCT design entry module class decorator."""

  class __Module(cls):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      # After the wrapped class' construct, all the IO should be known.
      input_ports = []
      output_ports = []
      for attr_name in dir(self):
        attr = self.__getattribute__(attr_name)
        if type(attr) is Input:
          input_ports.append((attr_name, attr))
        if type(attr) is Output:
          output_ports.append((attr_name, attr))

      # This function sets up the inputs to construct, then connects the
      # outputs.
      def body_build(mod):
        inputs = dict()
        for index, (name, _) in enumerate(input_ports):
          inputs[name] = mod.entry_block.arguments[index]

        self.construct(**inputs)

        outputs = []
        unconnected_ports = []
        for (name, output) in output_ports:
          if output.value is None:
            unconnected_ports.append(name)
          outputs.append(output.value)
        if len(unconnected_ports) > 0:
          raise UnconnectedSignalError(cls.__name__, unconnected_ports)
        hw.OutputOp(outputs)

      # Construct things as HWModules.
      self.module = hw.HWModuleOp(
          name=cls.__name__,
          input_ports=[(name, port.type) for (name, port) in input_ports],
          output_ports=[(name, port.type) for (name, port) in output_ports],
          body_builder=body_build)

  return __Module
