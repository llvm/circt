// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @mixed_ports(%arg0 : !esi.channel<i32>, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.pipeline' op if any port of this pipeline is an ESI channel, all ports must be ESI channels.}}
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (!esi.channel<i32>, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %valid = hw.constant 1 : i1
    pipeline.return %a0 valid %valid : i32
  }
  hw.output %0 : i32
}


// -----

hw.module @body_argn(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.pipeline' op expected 2 arguments in the pipeline body block, got 1.}}
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32):
    %valid = hw.constant 1 : i1
    pipeline.return %a0 valid %valid : i32
  }
  hw.output %0 : i32
}

// -----

hw.module @res_argn(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %valid = hw.constant 1 : i1
    // expected-error @+1 {{'pipeline.return' op expected 1 return values, got 0.}}
    pipeline.return valid %valid
  }
  hw.output %0 : i32
}

// -----

hw.module @body_argtype(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.pipeline' op expected body block argument 0 to have type 'i32', got 'i31'.}}
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i31, %a1 : i32):
    %valid = hw.constant 1 : i1
    pipeline.return %a1 valid %valid : i32
  }
  hw.output %0 : i32
}

// -----

hw.module @res_argtype(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i31) {
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i31) {
   ^bb0(%a0 : i32, %a1 : i32):
    %valid = hw.constant 1 : i1
    // expected-error @+1 {{'pipeline.return' op expected argument 0 to have type 'i31', got 'i32'.}}
    pipeline.return %a0 valid %valid  : i32
  }
  hw.output %0 : i31
}

// -----

hw.module @mixed_stages(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  // expected-error @+1 {{'pipeline.pipeline' op mixing `pipeline.stage` and `pipeline.stage.register` ops is illegal.}}
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = comb.add %a0, %a1 : i32
    %c1_i1 = hw.constant 1 : i1
    %r_0, %s0_valid = pipeline.stage.register when %c1_i1 regs %0 : i32
    %s1_valid = pipeline.stage when %s0_valid
    pipeline.return %0 valid %s0_valid : i32
  }
  hw.output %0 : i32
}
