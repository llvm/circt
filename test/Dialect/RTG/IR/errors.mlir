// RUN: circt-opt %s --split-input-file --verify-diagnostics

rtg.test @constantTooBig() {
  // expected-error @below {{integer value out-of-range for bit-width 2}}
  rtg.constant #rtg.isa.immediate<2, 4>
}

// -----

rtg.test @immediateWidthMismatch() {
  // expected-error @below {{explicit immediate type bit-width does not match attribute bit-width, 1 vs 2}}
  rtg.constant #rtg.isa.immediate<2, 1> : !rtg.isa.immediate<1>
}

// -----

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
"rtg.test"() <{sym_name="test", templateName="test", targetType=!rtg.dict<a: i32>}> ({^bb0(%b: i8):}) : () -> ()

// -----

// expected-error @below {{template name must not be empty}}
"rtg.test"() <{sym_name="test", templateName="", targetType=!rtg.dict<>}> ({^bb0:}) : () -> ()

// -----

// expected-error @below {{'target' does not reference a valid 'rtg.target' operation}}
rtg.test @test(a = %a: i32) target @target {
}

// -----

rtg.target @target : !rtg.dict<> { }

// expected-error @below {{referenced 'rtg.target' op's type is invalid: missing entry called 'a' of type 'i32'}}
rtg.test @test(a = %a: i32) target @target {
}

// -----

rtg.target @target : !rtg.dict<a: index> {
  %0 = index.constant 0
  rtg.yield %0 : index
}

// expected-error @below {{referenced 'rtg.target' op's type is invalid: missing entry called 'a' of type 'i32'}}
rtg.test @test(a = %a: i32) target @target {
}

// -----

// expected-error @below {{template name must not be empty}}
rtg.test @test() template "" {}

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

rtg.sequence @setCartesianProduct() {
  // expected-error @below {{at least one set must be provided}}
  // expected-error @below {{failed to infer returned types}}
  %0 = "rtg.set_cartesian_product"() : () -> (!rtg.set<!rtg.tuple<index>>)
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

// -----

rtg.test @test() {
  // expected-error @below {{must have at least one sequence in the list}}
  %0 = rtg.interleave_sequences
}

// -----

// expected-note @below {{prior use here}}
rtg.test @test(a = %a: i32, b = %b: index) {
  // expected-error @below {{use of value '%a' expects different type than prior uses: 'index' vs 'i32'}}
  rtg.array_create %a, %b : index
}

// -----

rtg.test @test(a = %a: i32, b = %b: index) {
  // expected-error @below {{requires all operands to have the same type}}
  "rtg.array_create"(%a, %b) : (i32, index) -> (!rtg.array<index>)
}

// -----

rtg.test @incorrect_tuple_type(tup = %tup : tuple<index, i1>) {
  // expected-error @below {{only RTG tuples are supported}}
  rtg.tuple_extract %tup at 2 : tuple<index, i1>
}

// -----

rtg.test @tupleExtractOOB(tup = %tup : !rtg.tuple<index, i1>) {
  // expected-error @below {{index (2) must be smaller than number of elements in tuple (2)}}
  rtg.tuple_extract %tup at 2 : !rtg.tuple<index, i1>
}

// -----

rtg.target @memoryBlockAddressDoesNotFit : !rtg.dict<> {
  // expected-error @below {{address out of range for memory block with address width 2}}
  rtg.isa.memory_block_declare [0x0 - 0x8] : !rtg.isa.memory_block<2>
}

// -----

rtg.target @memoryBlockBaseAddrWidthMismatch : !rtg.dict<> {
  // expected-error @below {{base address width must match memory block address width}}
  "rtg.isa.memory_block_declare"() <{baseAddress=0x0 : i32, endAddress=0x8 : i64}> : () -> !rtg.isa.memory_block<64>
}

// -----

rtg.target @memoryBlockEndAddrWidthMismatch : !rtg.dict<> {
  // expected-error @below {{end address width must match memory block address width}}
  "rtg.isa.memory_block_declare"() <{baseAddress=0x0 : i64, endAddress=0x8 : i32}> : () -> !rtg.isa.memory_block<64>
}

// -----

rtg.target @memoryBlockBaseAddressLargerThanEndAddress : !rtg.dict<> {
  // expected-error @below {{base address must be smaller than or equal to the end address}}
  rtg.isa.memory_block_declare [0x9 - 0x8] : !rtg.isa.memory_block<64>
}

// -----

rtg.test @validate() {
  %0 = rtg.fixed_reg #rtgtest.t0
  // expected-error @below {{result type must be a valid content type for the ref value}}
  %2 = rtg.validate %0, %0 : !rtgtest.ireg -> !rtgtest.ireg
}
