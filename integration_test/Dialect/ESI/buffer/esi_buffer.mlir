// REQUIRES: iverilog,cocotb

// Functionally test the `esi.buffer` lowering (feed-forward register chain plus
// run-out FIFO) end-to-end in simulation. Lower the abstract buffer all the way
// down to SystemVerilog and exercise it with cocotb.

// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --canonicalize --prettify-verilog --export-verilog -o %t.hw.mlir > %t.sv
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=esi_buffer --pythonFolder="%S,%S/../widgets" %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  // Default single-stage buffer (stages = 1, default slack).
  hw.module @buf_s1(
      in %clk: !seq.clock, in %rst: i1,
      in %in: !esi.channel<i32>,
      out out: !esi.channel<i32>) {
    %b = esi.buffer %clk, %rst, %in : !esi.channel<i32> -> !esi.channel<i32>
    hw.output %b : !esi.channel<i32>
  }

  // Multi-stage buffer with explicit slack, exercising the register chain and a
  // deeper run-out FIFO.
  hw.module @buf_s4(
      in %clk: !seq.clock, in %rst: i1,
      in %in: !esi.channel<i32>,
      out out: !esi.channel<i32>) {
    %b = esi.buffer %clk, %rst, %in {stages = 4 : i64, slack = 3 : i64} : !esi.channel<i32> -> !esi.channel<i32>
    hw.output %b : !esi.channel<i32>
  }

  hw.module @top(in %clk: !seq.clock, in %rst: i1,
      in %s1_in: !esi.channel<i32>, out s1_out: !esi.channel<i32>,
      in %s4_in: !esi.channel<i32>, out s4_out: !esi.channel<i32>) {
    %s1 = hw.instance "s1" @buf_s1(
      clk: %clk: !seq.clock, rst: %rst: i1, in: %s1_in: !esi.channel<i32>) -> (out: !esi.channel<i32>)
    %s4 = hw.instance "s4" @buf_s4(
      clk: %clk: !seq.clock, rst: %rst: i1, in: %s4_in: !esi.channel<i32>) -> (out: !esi.channel<i32>)
    hw.output %s1, %s4 : !esi.channel<i32>, !esi.channel<i32>
  }
}
