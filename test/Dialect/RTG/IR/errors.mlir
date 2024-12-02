// RUN: circt-opt %s --split-input-file --verify-diagnostics


func.func @seq0() {
  return
}

// expected-error @below {{'seq0' does not reference a valid 'rtg.sequence' operation}}
rtg.sequence_closure @seq0

// -----

rtg.sequence @seq0 {
^bb0(%arg0: i32):
}

// expected-error @below {{referenced 'rtg.sequence' op's argument types must match 'args' types}}
rtg.sequence_closure @seq0

// -----

// expected-error @below {{terminator operand types must match dict entry types}}
rtg.target @target : !rtg.dict<a: i32> {
  rtg.yield
}

// -----

// expected-error @below {{argument types must match dict entry types}}
rtg.test @test : !rtg.dict<a: i32> {
}

// -----

// expected-error @below {{dictionary must be sorted by names and contain no duplicates, first violation at entry 'a'}}
rtg.test @test : !rtg.dict<a: i32, a: i32> {
^bb0(%arg0: i32, %arg1: i32):
}

// -----

// expected-error @below {{dictionary must be sorted by names and contain no duplicates, first violation at entry 'a'}}
rtg.test @test : !rtg.dict<b: i32, a: i32> {
^bb0(%arg0: i32, %arg1: i32):
}

// -----

// expected-error @below {{empty strings not allowed as entry names}}
rtg.test @test : !rtg.dict<"": i32> {
^bb0(%arg0: i32):
}
