# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import sys

import circt
from circt.support import connect
from circt.dialects import hw, seq

from mlir.ir import *
from mlir.passmanager import PassManager

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i1 = IntegerType.get_signless(1)
  i32 = IntegerType.get_signless(32)

  # CHECK-LABEL: === MLIR ===
  m = Module.create()
  with InsertionPoint(m.body):

    def top(module):
      # CHECK: %[[RESET_VAL:.+]] = hw.constant 0
      reg_reset = hw.ConstantOp.create(i32, 0).result
      # CHECK: %[[INPUT_VAL:.+]] = hw.constant 45
      reg_input = hw.ConstantOp.create(i32, 45).result
      # CHECK: %[[DATA_VAL:.+]] = seq.compreg %[[INPUT_VAL]], %clk, %rstn, %[[RESET_VAL]]
      reg = seq.CompRegOp(i32,
                          reg_input,
                          module.clk,
                          reset=module.rstn,
                          reset_value=reg_reset,
                          name="my_reg")

      # CHECK: seq.compreg %[[INPUT_VAL]], %clk
      seq.reg(reg_input, module.clk)
      # CHECK: seq.compreg %[[INPUT_VAL]], %clk, %rstn, %{{.+}}
      seq.reg(reg_input, module.clk, reset=module.rstn)
      # CHECK: %[[RESET_VALUE:.+]] = hw.constant 123
      # CHECK: seq.compreg %[[INPUT_VAL]], %clk, %rstn, %[[RESET_VALUE]]
      custom_reset = hw.ConstantOp.create(i32, 123).result
      seq.reg(reg_input,
              module.clk,
              reset=module.rstn,
              reset_value=custom_reset)
      # CHECK: %FuBar = seq.compreg {{.+}}
      seq.reg(reg_input, module.clk, name="FuBar")

      # CHECK: %reg1 = seq.compreg %[[INPUT_VAL]], %clk
      reg1 = seq.CompRegOp.create(i32, clk=module.clk, name="reg1")
      connect(reg1.input, reg_input)

      # CHECK: %reg2 = seq.compreg %[[INPUT_VAL]], %clk
      reg2 = seq.CompRegOp.create(i32, name="reg2")
      connect(reg2.input, reg_input)
      connect(reg2.clk, module.clk)

      # CHECK: hw.output %[[DATA_VAL]]
      hw.OutputOp([reg.data])

    hw.HWModuleOp(name="top",
                  input_ports=[("clk", i1), ("rstn", i1)],
                  output_ports=[("result", i32)],
                  body_builder=top)

  print("=== MLIR ===")
  print(m)

  # CHECK-LABEL: === Verilog ===
  print("=== Verilog ===")

  pm = PassManager.parse("lower-seq-to-sv")
  pm.run(m)
  # CHECK: always @(posedge clk)
  # CHECK: my_reg <= {{.+}}
  circt.export_verilog(m, sys.stdout)
