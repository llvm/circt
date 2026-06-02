// RUN: circt-opt %s --verify-diagnostics --split-input-file

hw.module @test(out result : i1) {
    // expected-error @+1 {{'synth.choice' op expected 1 or more operands, but found 0}}
    %0 = synth.choice : i1
    hw.output %0 : i1
}

// -----

// expected-error @below {{argument type must be i1, but got 'i2'}}
synth.cut_rewrite_pattern (%a: i2) -> i1 attributes {cost = #synth.mapping_cost<area = 1.0 : f64>} {
  %0 = comb.extract %a from 0 : (i2) -> i1
  synth.yield %0 : i1
}

// -----

// expected-error @below {{result type must be i1, but got 'i2'}}
synth.cut_rewrite_pattern (%a: i1) -> i2 attributes {cost = #synth.mapping_cost<area = 1.0 : f64>} {
  %0 = hw.constant 0 : i2
  synth.yield %0 : i2
}

// -----

// expected-error @below {{requires exactly one result}}
synth.cut_rewrite_pattern (%a: i1) -> (i1, i1) attributes {cost = #synth.mapping_cost<area = 1.0 : f64>} {
  synth.yield %a, %a : i1, i1
}

// -----

// expected-error @below {{result type doesn't match with the terminator}}
synth.cut_rewrite_pattern (%a: i1) -> i1 attributes {cost = #synth.mapping_cost<area = 1.0 : f64>} {
  "synth.yield"() : () -> ()
}

// -----

// expected-error @below {{'i1' is expected but got 'i2'}}
synth.cut_rewrite_pattern (%a: i1) -> i1 attributes {cost = #synth.mapping_cost<area = 1.0 : f64>} {
  %0 = hw.constant 0 : i2
  synth.yield %0 : i2
}

// -----

// expected-error @below {{mapping cost arcs for cut rewrite patterns must use synth.positional_linear_timing_arc}}
synth.cut_rewrite_pattern (%a: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
    #synth.linear_timing_arc<"result", "a", 1, 0, #synth.polarity<positive>>
  ]>
} {
  synth.yield %a : i1
}

// -----

// expected-error @below {{mapping cost arc input index exceeds number of arguments}}
synth.cut_rewrite_pattern (%a: i1) -> i1 attributes {
  cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
    #synth.positional_linear_timing_arc<1, 1, 0, #synth.polarity<positive>>
  ]>
} {
  synth.yield %a : i1
}
