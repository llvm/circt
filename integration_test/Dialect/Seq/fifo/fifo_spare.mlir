// REQUIRES: iverilog,cocotb

// Test a FIFO whose depth is one short of a power of two, so its memory has a
// spare slot and is written every cycle, without a write enable.

// RUN: circt-opt %s --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --sv-trace-iverilog --export-verilog -o %t.mlir > %t.sv
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=fifo_spare \
// RUN:     --pythonModule=fifo_spare --pythonFolder="%S,%S/.." %t.sv 2>&1

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @fifo_spare(in %clk : !seq.clock, in %rst : i1, in %inp : i16, in %rdEn : i1, in %wrEn : i1, out out: i16, out empty: i1, out full: i1) {
  %out, %full, %empty = seq.fifo depth 7 rd_latency 1 in %inp rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i16
  hw.output %out, %empty, %full : i16, i1, i1
}
