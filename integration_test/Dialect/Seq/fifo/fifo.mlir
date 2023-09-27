// REQUIRES: iverilog,cocotb

// RUN: circt-opt %s --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --sv-trace-iverilog --export-verilog -o %t.mlir > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=fifo \
// RUN:     --pythonModule=fifo --pythonFolder="%S,%S/.." %t.sv 2>&1

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @fifo(input %clk : !seq.clock, input %rst : i1, input %inp : i32, input %rdEn : i1, input %wrEn : i1, output out: i32, empty: i1, full: i1, almost_empty : i1, almost_full : i1) {
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 4 almost_full 2 almost_empty 1 in %inp rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out, %empty, %full, %almostEmpty, %almostFull : i32, i1, i1, i1, i1
}
