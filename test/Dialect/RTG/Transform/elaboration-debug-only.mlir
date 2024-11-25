// RUN: circt-opt --rtg-elaborate=debug=true --split-input-file --verify-diagnostics %s

rtg.sequence @seq0 {
  %2 = hw.constant 2 : i32
}

// Test that the elaborator value interning works as intended and exercise 'set_select_random' error messages.
rtg.test @setOperations : !rtg.dict<> {
  %0 = rtg.sequence_closure @seq0
  %1 = rtg.sequence_closure @seq0
  %set = rtg.set_create %0, %1 : !rtg.sequence
  // expected-warning @below {{set contained 1 duplicate value(s), the value at index 2 might not be the intended one}}
  // expected-error @below {{'rtg.elaboration' attribute value out of bounds, must be between 0 (incl.) and 2 (excl.)}}
  %seq = rtg.set_select_random %set : !rtg.set<!rtg.sequence> {rtg.elaboration = 2}
  rtg.invoke_sequence %seq
}

// -----

rtg.sequence @seq0 {
  %2 = hw.constant 2 : i32
}

rtg.test @test : !rtg.dict<> {
  %0 = rtg.sequence_closure @seq0
  %set = rtg.set_create %0, %0 : !rtg.sequence
  // expected-warning @below {{set contained 1 duplicate value(s), the value at index 0 might not be the intended one}}
  %seq = rtg.set_select_random %set : !rtg.set<!rtg.sequence> {rtg.elaboration = 0}
  rtg.invoke_sequence %seq
}
