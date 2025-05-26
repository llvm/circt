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
    %bb1_0 = pipeline.src %0 : i32
    pipeline.stage ^bb2 regs(%bb1_0 : i32)

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
  // expected-error @+1 {{'pipeline.stage' op operand 0 is defined in a different stage. Value should have been passed through block arguments}}
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
    // expected-note@below {{use was operand 0. The result is available 1 stages later than this use.}}
    %bb1_1 = pipeline.src %1 : i32
    pipeline.return %bb1_1 : i32
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
    %bb3_a0 = pipeline.src %a0 : i32
    pipeline.return %bb3_a0 : i32
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
    %a0_bb3 = pipeline.src %a0 : i32
    pipeline.return %a0_bb3 : i32
  }
  hw.output %0 : i32
}

// -----

hw.module @unmaterialized_latency_with_missing_src(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1
   ^bb1(%s1_enable : i1):
    %1 = pipeline.latency 1 -> (i32) {
      // expected-error @below {{'comb.add' op operand 0 is defined in a different stage. Value should have been passed through a `pipeline.src` op}}
      %2 = comb.add %0, %0 : i32
      pipeline.latency.return %2 : i32
    }
    pipeline.stage ^bb2
  ^bb2(%s2_enable : i1):
    %bb2_1 = pipeline.src %1 : i32
    pipeline.return %bb2_1 : i32
  }
  hw.output %0 : i32
}
