# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt import msft
from circt.dialects import hw, seq

import mlir.ir as ir
import sys

with ir.Context() as ctx, ir.Location.unknown():
  circt.register_dialects(ctx)
  i32 = ir.IntegerType.get_signless(32)
  i1 = ir.IntegerType.get_signless(1)

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):
    extmod = hw.HWModuleExternOp(name='MyExternMod',
                                 input_ports=[],
                                 output_ports=[])

    op = hw.HWModuleOp(name='MyWidget',
                       input_ports=[],
                       output_ports=[],
                       body_builder=lambda module: hw.OutputOp([]))
    top = hw.HWModuleOp(name='top',
                        input_ports=[],
                        output_ports=[],
                        body_builder=lambda module: hw.OutputOp([]))

  with ir.InsertionPoint.at_block_terminator(op.body.blocks[0]):
    ext_inst = extmod.create("ext1")

  with ir.InsertionPoint.at_block_terminator(top.body.blocks[0]):
    inst = op.create("inst1")

  # CHECK: #msft.physloc<M20K, 2, 6, 1>
  physAttr = msft.PhysLocationAttr.get(msft.M20K, x=2, y=6, num=1)
  print(physAttr)

  inst = msft.RootedInstancePathAttr.get(
      ir.Attribute.parse("@top"),
      [ir.StringAttr.get("inst1"),
       ir.StringAttr.get("ext1")])
  # CHECK-NEXT: #msft.switch.inst<@top["inst1","ext1"]=#msft.physloc<M20K, 2, 6, 1>>
  instSwitch = msft.SwitchInstanceAttr.get([(inst, physAttr)])
  print(instSwitch)

  resolved_inst = msft.get_instance(top.operation,
                                    ir.Attribute.parse("@inst1::@ext1"))
  assert (resolved_inst == ext_inst.operation)
  resolved_inst.attributes["loc:subpath"] = instSwitch

  not_found_inst = msft.get_instance(top.operation,
                                     ir.Attribute.parse("@inst_none::@ext1"))
  assert (not_found_inst is None)

  # CHECK: hw.module @MyWidget()
  # CHECK:   hw.output
  m.operation.print()

  # CHECK-LABEL: === tcl ===
  print("=== tcl ===")

  # CHECK: proc top_config { parent } {
  # CHECK:   set_location_assignment M20K_X2_Y6_N1 -to $parent|inst1|ext1|ext1|subpath
  msft.export_tcl(top.operation, sys.stdout)
