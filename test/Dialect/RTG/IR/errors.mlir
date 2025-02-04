// RUN: circt-opt %s --split-input-file --verify-diagnostics


func.func @seq0() {
  return
}

// expected-error @below {{'seq0' does not reference a valid 'rtg.sequence' operation}}
rtg.get_sequence @seq0 : !rtg.sequence

// -----

rtg.sequence @seq0(%arg0: index) { }

// expected-error @below {{referenced 'rtg.sequence' op's type does not match}}
"rtg.get_sequence"() <{sequence="seq0"}> : () -> !rtg.sequence

// -----

rtg.sequence @seq0(%arg0: index) { }

%0 = rtg.get_sequence @seq0 : !rtg.sequence<index>
// expected-error @below {{must at least have one replacement value}}
rtg.substitute_sequence %0() : !rtg.sequence<index>

// -----

rtg.sequence @seq0(%arg0: index) { }

%c = index.constant 0
%0 = rtg.get_sequence @seq0 : !rtg.sequence<index>
// expected-error @below {{number of operands and types do not match: got 2 operands and 1 types}}
rtg.substitute_sequence %0(%c, %c) : !rtg.sequence<index>

// -----

rtg.sequence @seq0(%arg0: index) { }

// expected-note @below {{prior use here}}
%c = index.bool.constant true
%0 = rtg.get_sequence @seq0 : !rtg.sequence<index>
// expected-error @below {{use of value '%c' expects different type than prior uses: 'index' vs 'i1'}}
rtg.substitute_sequence %0(%c) : !rtg.sequence<index>

// -----

rtg.sequence @seq0(%arg0: index) { }

%c = index.constant 0
%0 = rtg.get_sequence @seq0 : !rtg.sequence<index>
// expected-error @below {{must not have more replacement values than sequence arguments}}
"rtg.substitute_sequence"(%0, %c, %c) : (!rtg.sequence<index>, index, index) -> !rtg.sequence

// -----

rtg.sequence @seq0(%arg0: index) { }

%c = index.bool.constant true
%0 = rtg.get_sequence @seq0 : !rtg.sequence<index>
// expected-error @below {{replacement types must match the same number of sequence argument types from the front}}
"rtg.substitute_sequence"(%0, %c) : (!rtg.sequence<index>, i1) -> !rtg.sequence

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
"rtg.test"() <{sym_name="test", target=!rtg.dict<a: i32>}> ({^bb0(%b: i8):}) : () -> ()

// -----

// expected-error @below {{dictionary must be sorted by names and contain no duplicates, first violation at entry 'a'}}
rtg.test @test(a = %a: i32, a = %a: i32) {
}

// -----

// expected-error @below {{dictionary must be sorted by names and contain no duplicates, first violation at entry 'a'}}
rtg.test @test(b = %b: i32, a = %a: i32) {
}

// -----

// expected-error @below {{empty strings not allowed as entry names}}
rtg.test @test(dict = %dict: !rtg.dict<"": i32>) { }

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

// -----

rtg.sequence @seq() {}

rtg.target @target : !rtg.dict<> {
  %0 = rtg.get_sequence @seq : !rtg.sequence
  // expected-error @below {{sequence type must have exactly 3 element types}}
  rtg.context_switch #rtgtest.cpu<0> -> #rtgtest.cpu<1>, %0 : !rtg.sequence
}

// -----

rtg.sequence @seq(%arg0: !rtg.sequence, %arg1: !rtg.sequence, %arg2: !rtg.sequence) {}

rtg.target @target : !rtg.dict<> {
  %0 = rtg.get_sequence @seq : !rtg.sequence<!rtg.sequence, !rtg.sequence, !rtg.sequence>
  // expected-error @below {{first sequence element type must match 'from' attribute type}}
  rtg.context_switch #rtgtest.cpu<0> -> #rtgtest.cpu<1>, %0 : !rtg.sequence<!rtg.sequence, !rtg.sequence, !rtg.sequence>
}

// -----

rtg.sequence @seq(%arg0: !rtgtest.cpu, %arg1: !rtg.sequence, %arg2: !rtg.sequence) {}

rtg.target @target : !rtg.dict<> {
  %0 = rtg.get_sequence @seq : !rtg.sequence<!rtgtest.cpu, !rtg.sequence, !rtg.sequence>
  // expected-error @below {{second sequence element type must match 'to' attribute type}}
  rtg.context_switch #rtgtest.cpu<0> -> #rtgtest.cpu<1>, %0 : !rtg.sequence<!rtgtest.cpu, !rtg.sequence, !rtg.sequence>
}

// -----

rtg.sequence @seq(%arg0: !rtgtest.cpu, %arg1: !rtgtest.cpu, %arg2: !rtgtest.cpu) {}

rtg.target @target : !rtg.dict<> {
  %0 = rtg.get_sequence @seq : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtgtest.cpu>
  // expected-error @below {{third sequence element type must be a fully substituted sequence}}
  rtg.context_switch #rtgtest.cpu<0> -> #rtgtest.cpu<1>, %0 : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtgtest.cpu>
}
