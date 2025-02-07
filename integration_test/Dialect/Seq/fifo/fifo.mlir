// REQUIRES: iverilog,cocotb

// RUN: circt-opt %s --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --sv-trace-iverilog --export-verilog -o %t.mlir > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=fifo \
// RUN:     --pythonModule=fifo --pythonFolder="%S,%S/.." %t.sv 2>&1

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @fifo(in %clk : !seq.clock, in %rst : i1, in %inp : i32, in %rdEn : i1, in %wrEn : i1, out out: i32, out empty: i1, out full: i1, out almost_empty : i1, out almost_full : i1) {
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 6 almost_full 2 almost_empty 1 in %inp rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out, %empty, %full, %almostEmpty, %almostFull : i32, i1, i1, i1, i1
}
