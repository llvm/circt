// REQUIRES: iverilog,cocotb

// Test a FIFO whose read latency is large enough that a popped slot could be
// overwritten by a later push before the memory read has completed.

// RUN: circt-opt %s --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --sv-trace-iverilog --export-verilog -o %t.mlir > %t.sv
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=fifo_latency \
// RUN:     --pythonModule=fifo_latency --pythonFolder="%S,%S/.." %t.sv 2>&1

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @fifo_latency(in %clk : !seq.clock, in %rst : i1, in %inp : i16, in %rdEn : i1, in %wrEn : i1, out out: i16, out empty: i1, out full: i1) {
  %out, %full, %empty = seq.fifo depth 4 rd_latency 3 in %inp rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i16
  hw.output %out, %empty, %full : i16, i1, i1
}
