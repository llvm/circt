# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt import msft
from circt.dialects import rtl

from mlir.ir import *
import sys

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

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

  m.operation.print()

  # CHECK-LABEL: === tcl ===
  print("=== tcl ===")

  # CHECK: proc top_config { parent } {
  # CHECK:   set_location_assignment M20K_X50_Y100_N1 -to $parent|inst1|mem
  msft.export_tcl(m, sys.stdout)
