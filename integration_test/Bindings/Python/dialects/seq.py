# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import sys

import circt
from circt.support import connect
from circt.dialects import hw, seq, sv

from circt.ir import *
from circt.passmanager import PassManager

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  clk = seq.ClockType.get(ctx)
  i1 = IntegerType.get_signless(1)
  i32 = IntegerType.get_signless(32)

  # CHECK-LABEL: === MLIR ===
  m = Module.create()
  with InsertionPoint(m.body):

    def top(module):
      # CHECK: %[[RESET_VAL:.+]] = hw.constant 0
      reg_reset = hw.ConstantOp.create(i32, 0).result
      poweron_value = hw.ConstantOp.create(i32, 42).result
      # CHECK: %[[INPUT_VAL:.+]] = hw.constant 45
      reg_input = hw.ConstantOp.create(i32, 45).result
      # CHECK-NEXT: %[[POWERON_VAL:.+]] = seq.initial() {
      # CHECK-NEXT:   %[[C42:.+]] = hw.constant 42 : i32
      # CHECK-NEXT:   seq.yield %[[C42]] : i32
      # CHECK-NEXT: } : () -> !seq.immutable<i32>
      # CHECK: %[[DATA_VAL:.+]] = seq.compreg %[[INPUT_VAL]], %clk reset %rst, %[[RESET_VAL]] initial %[[POWERON_VAL]]
      reg = seq.CompRegOp(i32,
                          reg_input,
                          module.clk,
                          reset=module.rst,
                          reset_value=reg_reset,
                          power_on_value=poweron_value,
                          name="my_reg")

      # CHECK: seq.compreg %[[INPUT_VAL]], %clk
      seq.reg(reg_input, module.clk)
      # CHECK: seq.compreg %[[INPUT_VAL]], %clk reset %rst, %{{.+}}
      seq.reg(reg_input, module.clk, reset=module.rst)
      # CHECK: %[[RESET_VALUE:.+]] = hw.constant 123
      # CHECK: seq.compreg %[[INPUT_VAL]], %clk reset %rst, %[[RESET_VALUE]]
      custom_reset = hw.ConstantOp.create(i32, 123).result
      seq.reg(reg_input, module.clk, reset=module.rst, reset_value=custom_reset)
      # CHECK: %FuBar = seq.compreg {{.+}}
      seq.reg(reg_input, module.clk, name="FuBar")
      # CHECK: seq.compreg sym @FuBar
      seq.reg(reg_input, module.clk, sym_name="FuBar")

      # CHECK: %reg1 = seq.compreg %[[INPUT_VAL]], %clk {sv.attributes = [#sv.attribute<"no_merge">]} : i32
      sv_attr = sv.SVAttributeAttr.get("no_merge")
      reg1 = seq.CompRegOp.create(i32, clk=module.clk, name="reg1")

      reg1.attributes["sv.attributes"] = ArrayAttr.get([sv_attr])
      connect(reg1.input, reg_input)

      # CHECK: %reg2 = seq.compreg %[[INPUT_VAL]], %clk
      reg2 = seq.CompRegOp.create(i32, name="reg2")
      connect(reg2.input, reg_input)
      connect(reg2.clk, module.clk)

      # CHECK: seq.compreg sym @reg1
      seq.CompRegOp.create(i32,
                           input=reg_input,
                           clk=module.clk,
                           sym_name="reg1")

      # CHECK: hw.output %[[DATA_VAL]]
      hw.OutputOp([reg.data])

    hw.HWModuleOp(name="top",
                  input_ports=[("clk", clk), ("rst", i1)],
                  output_ports=[("result", i32)],
                  body_builder=top)

  print("=== MLIR ===")
  print(m)

  # CHECK-LABEL: === Verilog ===
  print("=== Verilog ===")

  pm = PassManager.parse("builtin.module(lower-seq-to-sv,canonicalize)")
  pm.run(m.operation)
  # CHECK: always @(posedge clk)
  # CHECK: my_reg <= {{.+}}
  # CHECK: (* no_merge *)
  # CHECK: reg [31:0] reg1;
  circt.export_verilog(m, sys.stdout)
