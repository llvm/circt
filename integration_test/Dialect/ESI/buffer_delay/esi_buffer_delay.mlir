// REQUIRES: iverilog,cocotb

// Test `esi.buffer` with channels which have a non-zero data delay, including
// buffers which change the data delay.

// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --lower-seq-fifo --lower-seq-hlmem --lower-seq-to-sv --lower-verif-to-sv --canonicalize --prettify-verilog --export-verilog -o %t.hw.mlir > %t.sv
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-cocotb-driver.py --objdir=%t.dir --topLevel=top --pythonModule=esi_buffer_delay --pythonFolder="%S" %t.sv %esi_prims

module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  // A `seq.fifo` with read latency L wrapped as a FIFO(L) channel, then
  // buffered with S stages. Named `FifoSrc<L><S>`.
  hw.module @FifoSrc11(in %clk: !seq.clock, in %rst: i1, in %din: i16,
                       in %push: i1, out full: i1,
                       out out: !esi.channel<i16, ValidReady(1)>) {
    %d, %full, %empty = seq.fifo depth 32 rd_latency 1 in %din rdEn %rden wrEn %push clk %clk rst %rst : i16
    %chan, %rden = esi.wrap.fifo %d, %empty : !esi.channel<i16, FIFO(1)>
    %b = esi.buffer %clk, %rst, %chan {stages = 1 : i64} : !esi.channel<i16, FIFO(1)> -> !esi.channel<i16, ValidReady(1)>
    hw.output %full, %b : i1, !esi.channel<i16, ValidReady(1)>
  }
  hw.module @FifoSrc12(in %clk: !seq.clock, in %rst: i1, in %din: i16,
                       in %push: i1, out full: i1,
                       out out: !esi.channel<i16, ValidReady(1)>) {
    %d, %full, %empty = seq.fifo depth 32 rd_latency 1 in %din rdEn %rden wrEn %push clk %clk rst %rst : i16
    %chan, %rden = esi.wrap.fifo %d, %empty : !esi.channel<i16, FIFO(1)>
    %b = esi.buffer %clk, %rst, %chan {stages = 2 : i64} : !esi.channel<i16, FIFO(1)> -> !esi.channel<i16, ValidReady(1)>
    hw.output %full, %b : i1, !esi.channel<i16, ValidReady(1)>
  }
  // FIFO output.
  hw.module @FifoSrc21(in %clk: !seq.clock, in %rst: i1, in %din: i16,
                       in %push: i1, out full: i1,
                       out out: !esi.channel<i16, FIFO(2)>) {
    %d, %full, %empty = seq.fifo depth 32 rd_latency 2 in %din rdEn %rden wrEn %push clk %clk rst %rst : i16
    %chan, %rden = esi.wrap.fifo %d, %empty : !esi.channel<i16, FIFO(2)>
    %b = esi.buffer %clk, %rst, %chan {stages = 1 : i64} : !esi.channel<i16, FIFO(2)> -> !esi.channel<i16, FIFO(2)>
    hw.output %full, %b : i1, !esi.channel<i16, FIFO(2)>
  }
  hw.module @FifoSrc22(in %clk: !seq.clock, in %rst: i1, in %din: i16,
                       in %push: i1, out full: i1,
                       out out: !esi.channel<i16, ValidReady(2)>) {
    %d, %full, %empty = seq.fifo depth 32 rd_latency 2 in %din rdEn %rden wrEn %push clk %clk rst %rst : i16
    %chan, %rden = esi.wrap.fifo %d, %empty : !esi.channel<i16, FIFO(2)>
    %b = esi.buffer %clk, %rst, %chan {stages = 2 : i64} : !esi.channel<i16, FIFO(2)> -> !esi.channel<i16, ValidReady(2)>
    hw.output %full, %b : i1, !esi.channel<i16, ValidReady(2)>
  }

  // Buffers which change the data delay. Named `FifoSrc<L>To<output>`.
  hw.module @FifoSrc1ToVR3(in %clk: !seq.clock, in %rst: i1, in %din: i16,
                           in %push: i1, out full: i1,
                           out out: !esi.channel<i16, ValidReady(3)>) {
    %d, %full, %empty = seq.fifo depth 32 rd_latency 1 in %din rdEn %rden wrEn %push clk %clk rst %rst : i16
    %chan, %rden = esi.wrap.fifo %d, %empty : !esi.channel<i16, FIFO(1)>
    %b = esi.buffer %clk, %rst, %chan {stages = 1 : i64} : !esi.channel<i16, FIFO(1)> -> !esi.channel<i16, ValidReady(3)>
    hw.output %full, %b : i1, !esi.channel<i16, ValidReady(3)>
  }

  hw.module @top(in %clk: !seq.clock, in %rst: i1,
      in %fifo11_din: i16, in %fifo11_push: i1,
      out fifo11_full: i1, out fifo11_out: !esi.channel<i16, ValidReady(1)>,
      in %fifo12_din: i16, in %fifo12_push: i1,
      out fifo12_full: i1, out fifo12_out: !esi.channel<i16, ValidReady(1)>,
      in %fifo21_din: i16, in %fifo21_push: i1,
      out fifo21_full: i1, out fifo21_out: !esi.channel<i16, FIFO(2)>,
      in %fifo22_din: i16, in %fifo22_push: i1,
      out fifo22_full: i1, out fifo22_out: !esi.channel<i16, ValidReady(2)>,
      in %fifo1vr3_din: i16, in %fifo1vr3_push: i1,
      out fifo1vr3_full: i1, out fifo1vr3_out: !esi.channel<i16, ValidReady(3)>,
      in %vr1_in: !esi.channel<i16, ValidReady(1)>,
      out vr1_out: !esi.channel<i16, ValidReady(1)>,
      in %vr2_in: !esi.channel<i16, ValidReady(2)>,
      out vr2_out: !esi.channel<i16, ValidReady(2)>,
      in %vr3_in: !esi.channel<i16, ValidReady(3)>,
      out vr3_out: !esi.channel<i16, ValidReady(3)>,
      in %vr0vr2_in: !esi.channel<i16, ValidReady>,
      out vr0vr2_out: !esi.channel<i16, ValidReady(2)>,
      in %vr1fifo3_in: !esi.channel<i16, ValidReady(1)>,
      out vr1fifo3_out: !esi.channel<i16, FIFO(3)>,
      in %vr0vo2_in: !esi.channel<i16, ValidReady>,
      out vr0vo2_out: !esi.channel<i16, ValidOnly(2)>,
      in %vo1_in: !esi.channel<i16, ValidOnly(1)>,
      out vo1_out: !esi.channel<i16, ValidReady(1)>,
      in %vo0vr1_in: !esi.channel<i16, ValidOnly>,
      out vo0vr1_out: !esi.channel<i16, ValidReady(1)>) {
    %f11Full, %f11Out = hw.instance "fifo11" @FifoSrc11(
        clk: %clk: !seq.clock, rst: %rst: i1, din: %fifo11_din: i16,
        push: %fifo11_push: i1) -> (full: i1, out: !esi.channel<i16, ValidReady(1)>)
    %f12Full, %f12Out = hw.instance "fifo12" @FifoSrc12(
        clk: %clk: !seq.clock, rst: %rst: i1, din: %fifo12_din: i16,
        push: %fifo12_push: i1) -> (full: i1, out: !esi.channel<i16, ValidReady(1)>)
    %f21Full, %f21Out = hw.instance "fifo21" @FifoSrc21(
        clk: %clk: !seq.clock, rst: %rst: i1, din: %fifo21_din: i16,
        push: %fifo21_push: i1) -> (full: i1, out: !esi.channel<i16, FIFO(2)>)
    %f22Full, %f22Out = hw.instance "fifo22" @FifoSrc22(
        clk: %clk: !seq.clock, rst: %rst: i1, din: %fifo22_din: i16,
        push: %fifo22_push: i1) -> (full: i1, out: !esi.channel<i16, ValidReady(2)>)
    %f1vr3Full, %f1vr3Out = hw.instance "fifo1vr3" @FifoSrc1ToVR3(
        clk: %clk: !seq.clock, rst: %rst: i1, din: %fifo1vr3_din: i16,
        push: %fifo1vr3_push: i1) -> (full: i1, out: !esi.channel<i16, ValidReady(3)>)

    %vr1 = esi.buffer %clk, %rst, %vr1_in {stages = 1 : i64} : !esi.channel<i16, ValidReady(1)> -> !esi.channel<i16, ValidReady(1)>
    %vr2 = esi.buffer %clk, %rst, %vr2_in {stages = 2 : i64} : !esi.channel<i16, ValidReady(2)> -> !esi.channel<i16, ValidReady(2)>
    %vr3 = esi.buffer %clk, %rst, %vr3_in {stages = 1 : i64} : !esi.channel<i16, ValidReady(3)> -> !esi.channel<i16, ValidReady(3)>
    %vr0vr2 = esi.buffer %clk, %rst, %vr0vr2_in {stages = 2 : i64} : !esi.channel<i16, ValidReady> -> !esi.channel<i16, ValidReady(2)>
    %vr1fifo3 = esi.buffer %clk, %rst, %vr1fifo3_in {stages = 1 : i64} : !esi.channel<i16, ValidReady(1)> -> !esi.channel<i16, FIFO(3)>
    %vr0vo2 = esi.buffer %clk, %rst, %vr0vo2_in {stages = 1 : i64} : !esi.channel<i16, ValidReady> -> !esi.channel<i16, ValidOnly(2)>
    %vo1 = esi.buffer %clk, %rst, %vo1_in : !esi.channel<i16, ValidOnly(1)> -> !esi.channel<i16, ValidReady(1)>
    %vo0vr1 = esi.buffer %clk, %rst, %vo0vr1_in : !esi.channel<i16, ValidOnly> -> !esi.channel<i16, ValidReady(1)>

    hw.output %f11Full, %f11Out, %f12Full, %f12Out, %f21Full, %f21Out,
              %f22Full, %f22Out, %f1vr3Full, %f1vr3Out,
              %vr1, %vr2, %vr3, %vr0vr2, %vr1fifo3, %vr0vo2, %vo1, %vo0vr1
      : i1, !esi.channel<i16, ValidReady(1)>, i1, !esi.channel<i16, ValidReady(1)>,
        i1, !esi.channel<i16, FIFO(2)>, i1, !esi.channel<i16, ValidReady(2)>,
        i1, !esi.channel<i16, ValidReady(3)>,
        !esi.channel<i16, ValidReady(1)>, !esi.channel<i16, ValidReady(2)>,
        !esi.channel<i16, ValidReady(3)>, !esi.channel<i16, ValidReady(2)>,
        !esi.channel<i16, FIFO(3)>, !esi.channel<i16, ValidOnly(2)>,
        !esi.channel<i16, ValidReady(1)>, !esi.channel<i16, ValidReady(1)>
  }
}
