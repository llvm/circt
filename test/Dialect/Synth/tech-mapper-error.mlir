// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper)' %s --split-input-file --verify-diagnostics

hw.module @do_nothing(in %a : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1.0, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
    hw.output %a : i1
}

hw.module @test(in %a : i1, in %b : i1, out result : i1) {
    // expected-error-re@+1 {{No matching cut found for value: {{.*}}}}
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// -----

// expected-error@+1 {{Modules with multiple outputs are not supported yet}}
hw.module @multi_output(in %a : i1, out result1 : i1, out result2 : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result1", "a", 1.0, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
    hw.output %a, %a : i1, i1
}

hw.module @test(in %a : i1, out result : i1) {
    hw.output %a : i1
}


// -----

// expected-error@+1 {{All input ports must be single bit}}
hw.module @multibit(in %a : i2, in %b: i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1.0, 0.0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 2.0, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
  hw.output %b: i1
}

// -----


// expected-error@+1 {{All output ports must be single bit}}
hw.module @multibit(in %a : i1, in %b: i1, out result : i2) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1.0, 0.0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 2.0, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = comb.concat %a, %b : i1, i1
  hw.output %0: i2
}

// -----


hw.module @multibit(in %a : i1, in %b: i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1.0, 0.0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 2.0, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
  // expected-error@+1 {{Unsupported operation for truth table simulation}}
  %0 = hw.constant 1 : i2
  %1 = comb.extract %0 from 1 : (i2) -> i1
  hw.output %1: i1
}

// -----


hw.module @multibit(in %a : i1, in %b: i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1.0, 0.0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "b", 2.0, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
  // expected-error@+1 {{Unsupported operation for truth table simulation}}
  %0 = comb.add %a, %b : i1
  hw.output %0: i1
}

// -----

// expected-error@+1 {{Too many inputs for truth table generation}}
hw.module @too_many_input_bits(in %a0 : i1, in %a1 : i1, in %a2 : i1, in %a3 : i1, in %a4 : i1,
                               in %a5 : i1, in %a6 : i1, in %a7 : i1, in %a8 : i1, in %a9 : i1,
                               in %a10 : i1, in %a11 : i1, in %a12 : i1, in %a13 : i1, in %a14 : i1,
                               in %a15 : i1, in %a16 : i1,  out result : i1)
  attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
  #synth.linear_timing_arc<"result", "a0", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a1", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a2", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a3", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a4", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a5", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a6", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a7", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a8", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a9", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a10", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a11", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a12", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a13", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a14", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a15", 1.0, 0.0, #synth.polarity<positive>>,
  #synth.linear_timing_arc<"result", "a16", 1.0, 0.0, #synth.polarity<positive>>
  ], input_caps = {}>} {
  %0 = synth.aig.and_inv %a0, %a1 : i1
  hw.output %0: i1
}

// -----

// expected-error@+1 {{duplicate mapping cost arc for input 'a'}}
hw.module @duplicate_arc(in %a : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 1.0, 0.0, #synth.polarity<positive>>, #synth.linear_timing_arc<"result", "a", 2.0, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
  hw.output %a : i1
}

// -----

// expected-error@+1 {{expected integral intrinsic delay for input 'a' until TechMapper supports fractional delays}}
hw.module @fractional_delay(in %a : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"result", "a", 0.5, 0.0, #synth.polarity<positive>>], input_caps = {}>} {
  hw.output %a : i1
}
