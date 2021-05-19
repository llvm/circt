#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from circt.dialects import hw

import mlir.ir


class Output:
  __slots__ = [
      "type",
      "value"
  ]

  def __init__(self, type: mlir.ir.Type):
    self.type = type
    self.value = None

  def set(self, val):
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


class UnconnectedOutputError(Exception):
  def __init__(self, module: str, port_name: str):
    super().__init__(f"Port {port_name} unconnected in design module {module}.")


def module(cls):
  class Module(cls):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

      input_ports = []
      output_ports = []
      for attr_name in dir(self):
        attr = self.__getattribute__(attr_name)
        if type(attr) is Input:
          input_ports.append((attr_name, attr))
        if type(attr) is Output:
          output_ports.append((attr_name, attr))

      def body_build(mod):
        inputs = dict()
        for index, (name, _) in enumerate(input_ports):
          inputs[name] = mod.entry_block.arguments[index]

        self.construct(**inputs)

        outputs = []
        for (name, output) in output_ports:
          if output.value is None:
            raise UnconnectedOutputError(cls.__name__, name)
          outputs.append(output.value)
        hw.OutputOp(outputs)

      self.module = hw.HWModuleOp(
          name=cls.__name__,
          input_ports=[(name, port.type) for (name, port) in input_ports],
          output_ports=[(name, port.type) for (name, port) in output_ports],
          body_builder=body_build)

    def __setattr__(self, name: str, value) -> None:
        return super().__setattr__(name, value)

  return Module
