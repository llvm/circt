// RUN: circt-opt %s --verify-roundtrip

hw.module @unscheduled1(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.unscheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = pipeline.latency 2 -> (i32) {
      %1 = comb.add %a0, %a1 : i32
      pipeline.latency.return %1 : i32
    }
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}


hw.module @scheduled1(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

   ^bb1(%s1_enable : i1):
    %bb1_0 = pipeline.src %0 : i32
    pipeline.return %bb1_0 : i32
  }
  hw.output %0 : i32
}

hw.module @scheduled_with_latency(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1
   ^bb1(%s1_enable : i1):
    %bb1_0 = pipeline.src %0 : i32
    %1 = pipeline.latency 1 -> (i32) {
      %2 = comb.add %bb1_0, %bb1_0 : i32
      pipeline.latency.return %2 : i32
    }
    pipeline.stage ^bb2
  ^bb2(%s2_enable : i1):
    %bb2_1 = pipeline.src %1 : i32
    pipeline.return %bb2_1 : i32
  }
  hw.output %0 : i32
}



hw.module @scheduled2(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1 regs(%0 : i32)

   ^bb1(%s0_0 : i32, %s1_enable : i1):
    pipeline.return %s0_0 : i32
  }
  hw.output %0 : i32
}


hw.module @scheduledWithPassthrough(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:3 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out0: i32, out1: i32) {
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1 regs(%0 : i32) pass(%a1 : i32)

   ^bb1(%s0_0 : i32, %s0_pass_a1 : i32, %s1_enable : i1):
    pipeline.return %s0_0, %s0_pass_a1 : i32, i32
  }
  hw.output %0#0 : i32
}


hw.module @withStall(in %arg0 : i32, in %stall : i1, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) stall(%stall) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    pipeline.return %a0 : i32
  }
  hw.output %0 : i32
}


hw.module @withMultipleRegs(in %arg0 : i32, in %stall : i1, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) stall(%stall) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    pipeline.stage ^bb1 regs(%a0 : i32, %a0 : i32)

   ^bb1(%0 : i32, %1 : i32, %s1_enable : i1):
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}


hw.module @withClockGates(in %arg0 : i32, in %stall : i1, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) stall(%stall) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    %true1 = hw.constant true
    %true2 = hw.constant true
    %true3 = hw.constant true
    pipeline.stage ^bb1 regs(%a0 : i32 gated by [%true1], %a0 : i32, %a0 : i32 gated by [%true2, %true3])

   ^bb1(%0 : i32, %1 : i32, %2 : i32, %s1_enable : i1):
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}


hw.module @withNames(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled "MyPipeline"(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1 regs("myAdd" = %0 : i32, %0 : i32, "myOtherAdd" = %0 : i32)

   ^bb1(%r1 : i32, %r2 : i32, %r3 : i32, %s1_enable : i1):
    pipeline.return %r1 : i32
  }
  hw.output %0 : i32
}

hw.module @withStallability(in %arg0 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, in %stall : i1, out out: i32) {
  %0:2 = pipeline.scheduled "MyPipeline"(%a0 : i32 = %arg0) stall(%stall) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable)
    {stallability = [true, false, true]}
   -> (out: i32) {
    pipeline.stage ^bb1
   ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2
   ^bb2(%s2_enable : i1):
    pipeline.stage ^bb3
   ^bb3(%s3_enable : i1):
    %bb3_0 = pipeline.src %a0 : i32
    pipeline.return %bb3_0 : i32
  }
  hw.output %0 : i32
}


hw.module @withoutReset(in %arg0 : i32, in %stall : i1, in %go : i1, in %clk : !seq.clock, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) go(%go) entryEn(%s0_enable) -> (out: i32) {
    pipeline.stage ^bb1
   ^bb1(%s1_enable : i1):
    %bb1_0 = pipeline.src %a0 : i32
    pipeline.return %bb1_0 : i32
  }

  %1:2 = pipeline.unscheduled (%a0 : i32 = %arg0) stall (%stall) clock (%clk) go (%go) entryEn (%s0_enable) -> (out: i32) {
    pipeline.return %a0 : i32
  }
  hw.output %0 : i32
}
