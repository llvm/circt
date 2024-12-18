// REQUIRES: iverilog,cocotb

// Test the original HandshakeToHW flow.

// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --sv-trace-iverilog --prettify-verilog --export-verilog -o %t.hw.mlir > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=fifo1 --pythonModule=esi_widgets --pythonFolder="%S" %t.sv %esi_prims

module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  hw.module @fifo1(in %clk: !seq.clock, in %rst: i1, in %in: !esi.channel<i32, FIFO>, out out: !esi.channel<i32, FIFO(2)>) {
    %fifo = esi.fifo in %in clk %clk rst %rst depth 12 : !esi.channel<i32, FIFO> -> !esi.channel<i32, FIFO(2)>
    hw.output %fifo : !esi.channel<i32, FIFO(2)>
  }
}
