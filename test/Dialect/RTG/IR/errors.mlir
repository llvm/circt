// RUN: circt-opt %s --split-input-file --verify-diagnostics


func.func @seq0() {
  return
}

// expected-error @below {{'seq0' does not reference a valid 'rtg.sequence' operation}}
rtg.sequence_closure @seq0

// -----

rtg.sequence @seq0(%arg0: i32) { }

// expected-error @below {{referenced 'rtg.sequence' op's argument types must match 'args' types}}
rtg.sequence_closure @seq0

// -----

// expected-error @below {{sequence type does not match block argument types}}
"rtg.sequence"()<{sym_name="seq0", sequenceType=!rtg.sequence<i32>}>({^bb0:}) : () -> ()

// -----

// expected-note @below {{prior use here}}
rtg.sequence @seq0(%arg0: !rtg.sequence) {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!rtg.randomized_sequence' vs '!rtg.sequence'}}
  rtg.embed_sequence %arg0
}

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

// -----

rtg.sequence @seq(%arg0: i32, %arg1: i64, %arg2: index) {
  // expected-error @below {{types of all elements must match}}
  "rtg.bag_create"(%arg0, %arg1, %arg2, %arg2){} : (i32, i64, index, index) -> !rtg.bag<i32>
}

// -----

rtg.sequence @seq(%arg0: i64, %arg1: i64, %arg2: index) {
  // expected-error @below {{operand types must match bag element type}}
  "rtg.bag_create"(%arg0, %arg1, %arg2, %arg2){} : (i64, i64, index, index) -> !rtg.bag<i32>
}

// -----

rtg.sequence @seq() {
  // expected-error @below {{expected 1 or more operands, but found 0}}
  rtg.set_union : !rtg.set<i32>
}

// -----

rtg.sequence @seq() {
  // expected-error @below {{expected 1 or more operands, but found 0}}
  rtg.bag_union : !rtg.bag<i32>
}
