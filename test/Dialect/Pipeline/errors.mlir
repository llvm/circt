// RUN: circt-opt %s -split-input-file -verify-diagnostics

// -----

hw.module @body_argn(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.unscheduled' op expected 2 arguments in the pipeline body block, got 1.}}
  %0 = pipeline.unscheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32):
    %valid = hw.constant 1 : i1
    pipeline.return %a0 : i32
  }
  hw.output %0 : i32
}

// -----

hw.module @res_argn(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.unscheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %valid = hw.constant 1 : i1
    // expected-error @+1 {{'pipeline.return' op expected 1 return values, got 0.}}
    pipeline.return
  }
  hw.output %0 : i32
}

// -----

hw.module @body_argtype(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.unscheduled' op expected body block argument 0 to have type 'i32', got 'i31'.}}
  %0 = pipeline.unscheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i31, %a1 : i32):
    %valid = hw.constant 1 : i1
    pipeline.return %a1 : i32
  }
  hw.output %0 : i32
}

// -----

hw.module @res_argtype(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i31) {
  %0 = pipeline.unscheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i31) {
   ^bb0(%a0 : i32, %a1 : i32):
    %valid = hw.constant 1 : i1
    // expected-error @+1 {{'pipeline.return' op expected return value of type 'i31', got 'i32'.}}
    pipeline.return %a0 : i32
  }
  hw.output %0 : i31
}

// -----

hw.module @mixed_stages(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %c1_i1 = hw.constant true
    %0 = comb.add %a0, %a1 : i32
  
  ^bb1:
  // expected-error @+1 {{'pipeline.stage' op Pipeline is in register materialized mode - operand 0 is defined in a different stage, which is illegal.}}
    pipeline.stage ^bb2 regs(%0, %c1_i1 : i32, i1) enable %c1_i1
  
  ^bb2(%s2_s0 : i32, %s2_valid : i1):
    pipeline.return %s2_s0 : i32
  }
  hw.output %0 : i32
}

// -----

hw.module @earlyAccess(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.scheduled(%arg0) clock %clk reset %rst : (i32) -> i32 {
  ^bb0(%arg0_0: i32):
    %true = hw.constant true
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %arg0_0, %arg0_0 : i32
      pipeline.latency.return %6 : i32
    }
    pipeline.stage ^bb1 enable %true
  ^bb1:
    // expected-note@+1 {{use was operand 0. The result is available 1 stages later than this use.}}
    pipeline.return %1 : i32
  }
  hw.output %0 : i32
}

// -----


hw.module @registeredPass(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.scheduled(%arg0) clock %clk reset %rst : (i32) -> i32 {
  ^bb0(%arg0_0: i32):
    %true = hw.constant true
    // expected-error @+1 {{'pipeline.latency' op result 0 is used before it is available.}}
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %arg0_0, %arg0_0 : i32
      pipeline.latency.return %6 : i32
    }
    // expected-note@+1 {{use was operand 0. The result is available 2 stages later than this use.}}
    pipeline.stage ^bb1 regs(%1 : i32) enable %true
  ^bb1(%v : i32):
    pipeline.return %v : i32
  }
  hw.output %0 : i32
}
