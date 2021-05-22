#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from circt.dialects import hw
from .support import BuilderValue, UnconnectedSignalError
import circt

import mlir.ir

import atexit

# Push a default context onto the context stack at import time.
DefaultContext = mlir.ir.Context()
DefaultContext.__enter__()
circt.register_dialects(DefaultContext)


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

  __slots__ = ["type", "value"]

  def __init__(self, type: mlir.ir.Type):
    self.type = type
    self.value = None

  def set(self, val):
    """Sets the final output signal. Should only be called by the implementation
    code."""
    if isinstance(val, mlir.ir.Value):
      self.value = val
    else:
      self.value = val.result


class Input:
  __slots__ = ["type"]

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

      # Construct things as HWModules.
      self.module = hw.HWModuleOp(
          name=cls.__name__,
          input_ports=[(name, port.type) for (name, port) in input_ports],
          output_ports=[(name, port.type) for (name, port) in output_ports],
          body_builder=self.construct)

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
