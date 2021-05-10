# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt import msft
from circt.dialects import rtl, seq

from mlir.ir import *
import sys

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  i32 = IntegerType.get_signless(32)
  i1 = IntegerType.get_signless(1)

  m = Module.create()
  with InsertionPoint(m.body):
    # CHECK: rtl.module @MyWidget()
    # CHECK:   rtl.output
    op = rtl.RTLModuleOp(name='MyWidget',
                         input_ports=[],
                         output_ports=[],
                         body_builder=lambda module: rtl.OutputOp([]))
    top = rtl.RTLModuleOp(name='top',
                          input_ports=[],
                          output_ports=[],
                          body_builder=lambda module: rtl.OutputOp([]))

  with InsertionPoint.at_block_terminator(top.body.blocks[0]):
    inst = op.create(m, "inst1", {})
    msft.locate(inst.operation, "mem", devtype=msft.M20K, x=50, y=100, num=1)
    # CHECK: rtl.instance "inst1" @MyWidget() {"loc:mem" = #msft.physloc<M20K, 50, 100, 1>, parameters = {}} : () -> ()

    val = rtl.ConstantOp(i32, IntegerAttr.get(i32, 14)).result
    clk = rtl.ConstantOp(i1, IntegerAttr.get(i1, 0)).result
    reg = seq.reg(val, clk, name="MyLocatableRegister")
    msft.locate(reg.owner, "mem", devtype=msft.M20K, x=25, y=25, num=1)
    # CHECK: seq.compreg {{.+}} {"loc:mem" = #msft.physloc<M20K, 25, 25, 1>, name = "MyLocatableRegister"}

  m.operation.print()

  # CHECK-LABEL: === tcl ===
  print("=== tcl ===")

  # CHECK: proc top_config { parent } {
  # CHECK:   set_location_assignment M20K_X50_Y100_N1 -to $parent|inst1|mem
  # CHECK:   set_location_assignment M20K_X25_Y25_N1 -to $parent|MyLocatableRegister|mem
  msft.export_tcl(m, sys.stdout)
