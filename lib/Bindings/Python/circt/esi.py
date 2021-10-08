#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from mlir.passmanager import PassManager
import mlir.ir

from mlir._mlir_libs._circt._esi import *
import circt
from circt.dialects import hw
from circt.dialects.esi import *

import typing
import sys


class System(CppSystem):

  mod = None
  passes = [
      "lower-esi-ports", "lower-esi-to-physical", "lower-esi-to-hw",
      "hw.module(hw-cleanup)"
  ]
  passed = False

  def __init__(self):
    self.ctxt = mlir.ir.Context()
    with self.ctxt, mlir.ir.Location.unknown():
      circt.register_dialects(self.ctxt)
      self.mod = mlir.ir.Module.create()
      super().__init__(self.mod)
      self.get_types()

      with mlir.ir.InsertionPoint(self.body):
        self.declare_externs()
        self.top_module()

  def declare_externs(self):
    pass

  def top_module(self):
    return hw.HWModuleOp(name='top',
                         input_ports=[('clk', self.i1), ('rstn', self.i1)],
                         output_ports=[],
                         body_builder=self.build_top)

  def build_top(self, topMod):
    self.build(topMod)
    hw.OutputOp([])

  def get_types(self):
    self.i1 = mlir.ir.IntegerType.get_signless(1)
    self.byte = mlir.ir.IntegerType.get_signless(8)

  @property
  def body(self):
    return self.mod.body

  def print(self):
    self.mod.operation.print()

  def run_passes(self):
    if self.passed:
      return
    with self.ctxt:
      pm = PassManager.parse(",".join(self.passes))
      pm.run(self.mod)
      self.passed = True

  def print_verilog(self, out_stream: typing.TextIO = sys.stdout):
    self.run_passes()
    circt.export_verilog(self.mod, out_stream)

  def cosim(self, name, id, clk, rstn, recv_type=None, send=None):
    if recv_type is None:
      recv_type = mlir.ir.IntegerType.get_signless(1)
    recv_type = ChannelType.get(recv_type)
    if send is None:
      send = NullSourceOp(ChannelType.get(
          mlir.ir.IntegerType.get_signless(1))).out
    ep = CosimEndpoint(recv_type, clk, rstn, send,
                       mlir.ir.Attribute.parse(str(id)))
    ep.operation.attributes["name"] = mlir.ir.StringAttr.get(name)
    return ep


class Types:
  """Python syntactic sugar to get types"""

  @staticmethod
  def __getattr__(name: str) -> mlir.ir.Type:
    return mlir.ir.Type.parse(name)

  @staticmethod
  def array(inner: mlir.ir.Type, size: int) -> hw.ArrayType:
    return hw.ArrayType.get(inner, size)

  @staticmethod
  def chan(inner: mlir.ir.Type) -> mlir.ir.Type:
    return ChannelType.get(inner)


types = Types()
