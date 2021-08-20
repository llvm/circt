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
    # CHECK: hw.module @MyWidget()
    # CHECK:   hw.output
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
    msft.locate(inst.operation, "mem", devtype=msft.M20K, x=50, y=100, num=1)
    # CHECK: hw.instance "inst1" @MyWidget() {"loc:mem" = #msft.physloc<M20K, 50, 100, 1>, parameters = {}} : () -> ()

    val = hw.ConstantOp.create(i32, 14).result
    clk = hw.ConstantOp.create(i1, 0).result
    reg = seq.reg(val, clk, name="MyLocatableRegister")
    msft.locate(reg.owner, "mem", devtype=msft.M20K, x=25, y=25, num=1)
    # CHECK: seq.compreg {{.+}} {"loc:mem" = #msft.physloc<M20K, 25, 25, 1>, name = "MyLocatableRegister"}

  m.operation.print()

  resolved_inst = msft.get_instance(top.operation,
                                    ir.Attribute.parse("@inst1::@ext1"))
  assert(resolved_inst == ext_inst.operation)

  not_found_inst = msft.get_instance(top.operation,
                                     ir.Attribute.parse("@inst_none::@ext1"))
  assert (not_found_inst is None)

  # CHECK: #msft.physloc<M20K, 2, 6, 1>
  physAttr = msft.PhysLocationAttr.get(msft.M20K, x=2, y=6, num=1)
  print(physAttr)

  inst = ir.Attribute.parse("@foo::@bar")
  # CHECK-NEXT: #msft.switch.inst<@foo::@bar=#msft.physloc<M20K, 2, 6, 1>>
  instSwitch = msft.SwitchInstance.get([(inst, physAttr)])
  print(instSwitch)

  # CHECK-LABEL: === tcl ===
  print("=== tcl ===")

  # CHECK: proc top_config { parent } {
  # CHECK:   set_location_assignment M20K_X50_Y100_N1 -to $parent|inst1|mem
  # CHECK:   set_location_assignment M20K_X25_Y25_N1 -to $parent|MyLocatableRegister|mem
  msft.export_tcl(m, sys.stdout)
