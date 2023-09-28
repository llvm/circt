// RUN: circt-opt -split-input-file -verify-diagnostics %s


hw.module @res_argn(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.unscheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    // expected-error @+1 {{'pipeline.return' op expected 1 return values, got 0.}}
    pipeline.return
  }
  hw.output %0#0 : i32
}

// -----

hw.module @res_argtype(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i31) {
  %0:2 = pipeline.unscheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i31) {
    // expected-error @+1 {{'pipeline.return' op expected return value of type 'i31', got 'i32'.}}
    pipeline.return %a0 : i32
  }
  hw.output %0#0 : i31
}

// -----

hw.module @unterminated(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op all blocks must be terminated with a `pipeline.stage` or `pipeline.return` op.}}
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32

  ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2 regs(%0 : i32)

  ^bb2(%s2_s0 : i32, %s2_enable : i1):
    pipeline.return %s2_s0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @mixed_stages(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

  ^bb1(%s1_enable : i1):
  // expected-error @+1 {{'pipeline.stage' op Pipeline is in register materialized mode - operand 0 is defined in a different stage, which is illegal.}}
    pipeline.stage ^bb2 regs(%0: i32)

  ^bb2(%s2_s0 : i32, %s2_enable : i1):
    pipeline.return %s2_s0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @cycle_pipeline1(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op pipeline contains a cycle.}}
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

  ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2

  ^bb2(%s2_enable : i1):
    pipeline.stage ^bb1

  ^bb3(%s3_enable : i1):
    pipeline.return %0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @cycle_pipeline2(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op pipeline contains a cycle.}}
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

  ^bb1(%s1_enable : i1):
    pipeline.stage ^bb1

  ^bb3(%s3_enable : i1):
    pipeline.return %0 : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @earlyAccess(in %arg0: i32, in %arg1: i32, in %go: i1, in %clk : !seq.clock, in %rst: i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %a0, %a0 : i32
      pipeline.latency.return %6 : i32
    }
    pipeline.stage ^bb1
  ^bb1(%s1_enable : i1):
    // expected-note@+1 {{use was operand 0. The result is available 1 stages later than this use.}}
    pipeline.return %1 : i32
  }
  hw.output %0#0 : i32
}

// -----

// Test which verifies that the values referenced within the body of a
// latency operation also adhere to the latency constraints.
hw.module @earlyAccess2(in %arg0: i32, in %arg1: i32, in %go: i1, in %clk : !seq.clock, in %rst: i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %res = comb.add %a0, %a0 : i32
      pipeline.latency.return %res : i32
    }
    pipeline.stage ^bb1

  ^bb1(%s1_enable : i1):
    %2 = pipeline.latency 2 -> (i32) {
      %c1_i32 = hw.constant 1 : i32
      // expected-note@+1 {{use was operand 0. The result is available 1 stages later than this use.}}
      %res2 = comb.add %1, %c1_i32 : i32
      pipeline.latency.return %res2 : i32
    }
    pipeline.stage ^bb2

  ^bb2(%s2_enable : i1):
    pipeline.stage ^bb3

  ^bb3(%s3_enable : i1):
    pipeline.return %2 : i32
  }
  hw.output %0#0 : i32
}

// -----


hw.module @registeredPass(in %arg0: i32, in %arg1: i32, in %go: i1, in %clk : !seq.clock, in %rst: i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %a0, %a0 : i32
      pipeline.latency.return %6 : i32
    }
    // expected-note@+1 {{use was operand 0. The result is available 2 stages later than this use.}}
    pipeline.stage ^bb1 regs(%1 : i32)
  ^bb1(%v : i32, %s1_enable : i1):
    pipeline.return %v : i32
  }
  hw.output %0#0 : i32
}

// -----

hw.module @missing_valid_entry3(in %arg : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1) {
  // expected-error @+1 {{'pipeline.scheduled' op block 1 must have an i1 argument as the last block argument (stage valid signal).}}
  %done = pipeline.scheduled(%a0 : i32 = %arg) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> () {
     pipeline.stage ^bb1
   ^bb1:
      pipeline.return
  }
  hw.output
}

// -----

hw.module @invalid_clock_gate(in %arg : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1) {
  %done = pipeline.scheduled(%a0 : i32 = %arg) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> () {
     // expected-note@+1 {{prior use here}}
     %c0_i2 = hw.constant 0 : i2
     // expected-error @+1 {{use of value '%c0_i2' expects different type than prior uses: 'i1' vs 'i2'}}
     pipeline.stage ^bb1 regs(%a0 : i32 gated by [%c0_i2])
   ^bb1(%0 : i32, %s1_enable : i1):
      pipeline.return
  }
  hw.output
}

// -----

hw.module @noStallSignalWithStallability(in %arg0 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op cannot specify stallability without a stall signal.}}
  %0:2 = pipeline.scheduled "MyPipeline"(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable)
    {stallability = [true, false, true]}
   -> (out: i32) {
    pipeline.stage ^bb1
   ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2
   ^bb2(%s2_enable : i1):
    pipeline.stage ^bb3
   ^bb3(%s3_enable : i1):
    pipeline.return %a0 : i32
  }
  hw.output %0 : i32
}

// -----

hw.module @incorrectStallabilitySize(in %arg0 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, in %stall : i1, out out: i32) {
  // expected-error @+1 {{'pipeline.scheduled' op stallability array must be the same length as the number of stages. Pipeline has 3 stages but array had 2 elements.}}
  %0:2 = pipeline.scheduled "MyPipeline"(%a0 : i32 = %arg0) stall(%stall) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable)
    {stallability = [true, false]}
   -> (out: i32) {
    pipeline.stage ^bb1
   ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2
   ^bb2(%s2_enable : i1):
    pipeline.stage ^bb3
   ^bb3(%s3_enable : i1):
    pipeline.return %a0 : i32
  }
  hw.output %0 : i32
}
