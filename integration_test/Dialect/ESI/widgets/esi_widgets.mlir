// REQUIRES: iverilog,cocotb

// Test the original HandshakeToHW flow.

// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --canonicalize --prettify-verilog --export-verilog -o %t.hw.mlir > %t.sv
// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=top --pythonModule=esi_widgets --pythonFolder="%S" %t.sv %esi_prims

module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  hw.module @fifo1(
      in %clk: !seq.clock, in %rst: i1,
      in %in: !esi.channel<i32, FIFO>,
      out out: !esi.channel<i32, FIFO(2)>) {
    %fifo = esi.fifo in %in clk %clk rst %rst depth 12 : !esi.channel<i32, FIFO> -> !esi.channel<i32, FIFO(2)>
    hw.output %fifo : !esi.channel<i32, FIFO(2)>
  }

  hw.module @fifoValidReadyInput(
      in %clk: !seq.clock, in %rst: i1,
      in %in: !esi.channel<i32, ValidReady>,
      out out: !esi.channel<i32, FIFO(2)>) {
    %fifo = esi.fifo in %in clk %clk rst %rst depth 12 : !esi.channel<i32, ValidReady> -> !esi.channel<i32, FIFO(2)>
    hw.output %fifo : !esi.channel<i32, FIFO(2)>
  }

  hw.module @fifoValidReadyOutput(
      in %clk: !seq.clock, in %rst: i1,
      in %in: !esi.channel<i32, FIFO>,
      out out: !esi.channel<i32, ValidReady>) {
    %fifo = esi.fifo in %in clk %clk rst %rst depth 12 : !esi.channel<i32, FIFO> -> !esi.channel<i32, ValidReady>
    hw.output %fifo : !esi.channel<i32, ValidReady>
  }


  hw.module @top(in %clk: !seq.clock, in %rst: i1,
      in %fifo1_in: !esi.channel<i32, FIFO>, out fifo1_out: !esi.channel<i32, FIFO(2)>,
      in %fifoValidReadyInput_in: !esi.channel<i32, ValidReady>, out fifoValidReadyInput_out: !esi.channel<i32, FIFO(2)>,
      in %fifoValidReadyOutput_in: !esi.channel<i32, FIFO>, out fifoValidReadyOutput_out: !esi.channel<i32, ValidReady>
  ) {
    %fifo1 = hw.instance "fifo1" @fifo1(
      clk: %clk: !seq.clock, rst: %rst: i1, in: %fifo1_in: !esi.channel<i32, FIFO>) -> (out: !esi.channel<i32, FIFO(2)>)
    %fifoValidReadyInput = hw.instance "fifoValidReadyInput" @fifoValidReadyInput(
      clk: %clk: !seq.clock, rst: %rst: i1, in: %fifoValidReadyInput_in: !esi.channel<i32, ValidReady>) -> (out: !esi.channel<i32, FIFO(2)>)
    %fifoValidReadyOutput = hw.instance "fifoValidReadyOutput" @fifoValidReadyOutput(
      clk: %clk: !seq.clock, rst: %rst: i1, in: %fifoValidReadyOutput_in: !esi.channel<i32, FIFO>) -> (out: !esi.channel<i32, ValidReady>)
    hw.output %fifo1, %fifoValidReadyInput, %fifoValidReadyOutput : !esi.channel<i32, FIFO(2)>, !esi.channel<i32, FIFO(2)>, !esi.channel<i32, ValidReady>
  }
}
