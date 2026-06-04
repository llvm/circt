// REQUIRES: iverilog,cocotb

// Regression test for the `seq.fifo` lowering with a non-zero read latency
// (registered read). Exercises both the default (inline) lowering and the
// outlined-module lowering.

// RUN: circt-opt %s --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --sv-trace-iverilog --export-verilog -o %t.mlir > %t.sv
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=fifo_rdlatency \
// RUN:     --pythonModule=fifo_rdlatency --pythonFolder="%S" %t.sv 2>&1 | FileCheck %s

// RUN: circt-opt %s --lower-seq-fifo=outline-modules=true --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --sv-trace-iverilog --export-verilog -o %t.outline.mlir > %t.outline.sv
// RUN: rm -rf %t.outline.dir && mkdir %t.outline.dir
// RUN: circt-cocotb-driver.py --objdir=%t.outline.dir --topLevel=fifo_rdlatency \
// RUN:     --pythonModule=fifo_rdlatency --pythonFolder="%S" %t.outline.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @fifo_rdlatency(in %clk : !seq.clock, in %rst : i1, in %inp : i32, in %rdEn : i1, in %wrEn : i1, out out: i32, out empty: i1, out full: i1, out almost_empty : i1, out almost_full : i1) {
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 6 rd_latency 1 almost_full 2 almost_empty 1 in %inp rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out, %empty, %full, %almostEmpty, %almostFull : i32, i1, i1, i1, i1
}
