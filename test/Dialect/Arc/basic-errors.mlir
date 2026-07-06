// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{body contains non-pure operation}}
arc.define @Foo(%arg0: !seq.clock) {
  // expected-note @+1 {{first non-pure operation here:}}
  arc.state @Bar() clock %arg0 latency 1 : () -> ()
  arc.output
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(in %clock: !seq.clock) {
  // expected-error @+1 {{'arc.state' op requires a clock}}
  arc.state @Bar() latency 1 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(in %clock: !seq.clock) {
  // expected-error @+1 {{'arc.state' op latency must be a positive integer}}
  arc.state @Bar() clock %clock latency 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

// expected-error @+1 {{body contains non-pure operation}}
arc.define @SupportRecursiveMemoryEffects(%arg0: i1, %arg1: !seq.clock) {
  // expected-note @+1 {{first non-pure operation here:}}
  scf.if %arg0 {
    arc.state @Bar() clock %arg1 latency 1 : () -> ()
  }
  arc.output
}
arc.define @Bar() {
  arc.output
}

// -----

// expected-error @below {{op must have exactly one argument}}
arc.model @MissingArg io !hw.modty<> {
^bb0:
}

// -----

// expected-error @below {{op must have exactly one argument}}
arc.model @TooManyArgs io !hw.modty<> {
^bb0(%arg0: !arc.storage, %arg1: !arc.storage):
}

// -----

// expected-error @below {{op argument must be of storage type}}
arc.model @WrongArgType io !hw.modty<> {
^bb0(%arg0: i32):
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{`Bar` does not reference a valid `arc.define`}}
  arc.call @Bar() : () -> ()
  arc.output
}
func.func @Bar() {
  return
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{incorrect number of operands: expected 1, but got 0}}
  arc.call @Bar() : () -> ()
  arc.output
}
arc.define @Bar(%arg0: i1) {
  arc.output
}

// -----

arc.define @Foo() {
  // expected-error @+1 {{incorrect number of results: expected 1, but got 0}}
  arc.call @Bar() : () -> ()
  arc.output
}
arc.define @Bar() -> i1 {
  %false = hw.constant false
  arc.output %false : i1
}

// -----

arc.define @Foo(%arg0: i1, %arg1: i32) {
  // expected-error @+3 {{operand type mismatch: operand #1}}
  // expected-note @+2 {{expected type: 'i42'}}
  // expected-note @+1 {{actual type: 'i32'}}
  arc.call @Bar(%arg0, %arg1) : (i1, i32) -> ()
  arc.output
}
arc.define @Bar(%arg0: i1, %arg1: i42) {
  arc.output
}

// -----

arc.define @Foo(%arg0: i1, %arg1: i32) {
  // expected-error @+3 {{result type mismatch: result #1}}
  // expected-note @+2 {{expected type: 'i42'}}
  // expected-note @+1 {{actual type: 'i32'}}
  %0, %1 = arc.call @Bar() : () -> (i1, i32)
  arc.output
}
arc.define @Bar() -> (i1, i42) {
  %false = hw.constant false
  %c0_i42 = hw.constant 0 : i42
  arc.output %false, %c0_i42 : i1, i42
}

// -----

arc.define @lut () -> () {
  // expected-error @+1 {{requires one result}}
  arc.lut () : () -> () {
    arc.output
  }
  arc.output
}

// -----

arc.define @lut () -> () {
  %0 = arc.lut () : () -> i32 {
    // expected-error @+1 {{incorrect number of outputs: expected 1, but got 0}}
    arc.output
  }
  arc.output
}

// -----

arc.define @lut () -> () {
  %0 = arc.lut () : () -> i32 {
    %1 = hw.constant 0 : i16
    // expected-error @+3 {{output type mismatch: output #0}}
    // expected-note @+2 {{expected type: 'i32'}}
    // expected-note @+1 {{actual type: 'i16'}}
    arc.output %1 : i16
  }
  arc.output
}

// -----

arc.define @lut (%arg0: i32, %arg1: i8) -> () {
  // expected-note @+1 {{required by region isolation constraints}}
  %1 = arc.lut (%arg1, %arg0) : (i8, i32) -> i32 {
    ^bb0(%arg2: i8, %arg3: i32):
      // expected-error @+1 {{using value defined outside the region}}
      arc.output %arg0 : i32
  }
  arc.output
}

// -----

arc.define @lutSideEffects () -> i32 {
  // expected-error @+1 {{no operations with side-effects allowed inside a LUT}}
  %0 = arc.lut () : () -> i32 {
    %true = hw.constant true
    // expected-note @+1 {{first operation with side-effects here}}
    %1 = arc.memory !arc.memory<20 x i32, i1>
    %2 = arc.memory_read_port %1[%true] : !arc.memory<20 x i32, i1>
    arc.output %2 : i32
  }
  arc.output %0 : i32
}

// -----

hw.module @memoryWritePortOpNoClock(in %clock: !seq.clock, in %en: i1) {
  %mem = arc.memory <4 x i32, i32>
  %c0_i32 = hw.constant 0 : i32
  // expected-error @+1 {{requires a clock}}
  arc.memory_write_port %mem, @identity(%c0_i32, %c0_i32, %en) latency 1 : !arc.memory<4 x i32, i32>, i32, i32, i1
}
arc.define @identity(%addr: i32, %data: i32, %enable: i1) -> (i32, i32, i1) {
  arc.output %addr, %data, %enable : i32, i32, i1
}

// -----

hw.module @memoryWritePortOpLatZero(in %clock: !seq.clock, in %en: i1) {
  %mem = arc.memory <4 x i32, i32>
  %c0_i32 = hw.constant 0 : i32
  // expected-error @+1 {{latency must be at least 1}}
  arc.memory_write_port %mem, @identity(%c0_i32, %c0_i32, %en) latency 0 : !arc.memory<4 x i32, i32>, i32, i32, i1
}
arc.define @identity(%addr: i32, %data: i32, %enable: i1) -> (i32, i32, i1) {
  arc.output %addr, %data, %enable : i32, i32, i1
}

// -----

arc.define @outputOpVerifier () -> i32 {
  // expected-error @+1 {{incorrect number of outputs: expected 1, but got 0}}
  arc.output
}

// -----

arc.define @outputOpVerifier () -> i32 {
  %0 = hw.constant 0 : i16
  // expected-error @+3 {{output type mismatch: output #0}}
  // expected-note @+2 {{expected type: 'i32'}}
  // expected-note @+1 {{actual type: 'i16'}}
  arc.output %0 : i16
}

// -----

hw.module @operand_type_mismatch(in %in0: i2, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{all input vector lane types must match}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i2, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @number_results_does_not_match_way(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1) {
  // expected-error @below {{number results must match input vector size}}
  %0 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> i1 {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0 : i1
}

// -----

hw.module @result_type_mismatch(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i2) {
  // expected-error @below {{all result types must match}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i2) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1 : i1, i2
}

// -----

hw.module @vectorized_block_arg_type_mismatch(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{if terminator type matches result type the argument types must match the input types}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i2, %arg1: i1):
    %0 = comb.extract %arg0 from 0 : (i2) -> i1
    %1 = comb.and %0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @number_vectorized_block_args_mismatch(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{number of block arguments must match number of input vectors}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1):
    arc.vectorize.return %arg0 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @only_one_block_allowed(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{region #0 ('body') failed to verify constraint: region with 1 blocks}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    cf.br ^bb1(%1 : i1)
  ^bb1(%arg2: i1):
    arc.vectorize.return %arg2 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @only_one_block_allowed(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{region #0 ('body') failed to verify constraint: region with 1 blocks}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {}
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @input_operand_list_not_empty(out out0: i1, out out1: i1) {
  // expected-error @below {{there has to be at least one input vector}}
  %0:2 = arc.vectorize : () -> (i1, i1) {
  ^bb0:
    %1 = arith.constant false
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @input_vector_sizes_must_match(in %in0: i1, in %in1: i1, in %in2: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{all input vectors must have the same size}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2) : (i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @input_vector_not_empty(out out0: i1, out out1: i1) {
  // expected-error @below {{input vector must have at least one element}}
  %0:2 = arc.vectorize () : () -> (i1, i1) {
  ^bb0:
    %1 = arith.constant false
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @at_least_one_result(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1) {
  // expected-error @below {{op must have at least one result}}
  arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> () {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output
}

// -----

hw.module @not_divisible_width(in %in0: i4, in %in1: i4, out out0: i4) {
  %0 = arc.vectorize (%in0), (%in1) : (i4, i4) -> i4 {
  ^bb0(%arg0: i3, %arg1: i3):
    %1 = comb.and %arg0, %arg1 : i3
    // expected-error @below {{operand type must match parent op's result value or be a vectorized or non-vectorized variant of it}}
    arc.vectorize.return %1 : i3
  }
  hw.output %0 : i4
}

// -----

hw.module @body_vector_size_must_match_vector_operand_number(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{when boundary not vectorized the number of vector element operands must match the width of the vectorized body}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i3, %arg1: i3):
    %1 = comb.and %arg0, %arg1 : i3
    arc.vectorize.return %1 : i3
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @vectorize(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{input and output vector width must match}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.concat %arg0, %arg1 : i1, i1
    arc.vectorize.return %1 : i2
  }
  hw.output %0#0, %0#1 : i1, i1
}

// -----

hw.module @vectorize(in %in0: i2, in %in1: i2, out out0: i2) {
  // expected-error @below {{block argument must be a vectorized variant of the operand}}
  %0 = arc.vectorize (%in0), (%in1) : (i2, i2) -> (i2) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.concat %arg0, %arg1, %arg1, %arg0 : i1, i1, i1, i1
    arc.vectorize.return %1 : i4
  }
  hw.output %0 : i2
}

// -----

hw.module @vectorize(in %in0: i2, in %in1: i2, out out0: i2) {
  // expected-error @below {{input and output vector width must match}}
  %0 = arc.vectorize (%in0), (%in1) : (i2, i2) -> (i2) {
  ^bb0(%arg0: i2, %arg1: i2):
    %1 = arith.constant false
    arc.vectorize.return %1 : i1
  }
  hw.output %0 : i2
}

// -----

hw.module @vectorize(in %in0: i4, in %in1: i4, out out0: i4) {
  // expected-error @below {{block argument must be a scalar variant of the vectorized operand}}
  %0 = arc.vectorize (%in0), (%in1) : (i4, i4) -> (i4) {
  ^bb0(%arg0: i8, %arg1: i8):
    %1 = arith.constant 0 : i2
    arc.vectorize.return %1 : i2
  }
  hw.output %0 : i4
}

// -----

// expected-error @below {{state type must have a known bit width}}
func.func @InvalidStateType(%arg0: !arc.state<index>)

// -----

// expected-error @below {{initializer 'Bar' does not reference a valid function}}
arc.model @Foo io !hw.modty<> initializer @Bar {
^bb0(%arg0: !arc.storage):
}

// -----

// expected-error @below {{finalizer 'Bar' does not reference a valid function}}
arc.model @Foo io !hw.modty<> storageBytes 42 finalizer @Bar {
^bb0(%arg0: !arc.storage):
}

// -----

// expected-error @below {{initializer 'Bar' does not reference a valid function}}
arc.model @Foo io !hw.modty<> storageBytes 42 initializer @Bar {
^bb0(%arg0: !arc.storage):
}
hw.module @Bar() {
}

// -----

// expected-error @below {{finalizer 'Bar' does not reference a valid function}}
arc.model @Foo io !hw.modty<> storageBytes 42 finalizer @Bar {
^bb0(%arg0: !arc.storage):
}
hw.module @Bar() {
}

// -----

// expected-error @below {{initializer 'Bar' arguments must match arguments of model}}
arc.model @Foo io !hw.modty<> storageBytes 42 initializer @Bar {
^bb0(%arg0: !arc.storage):
}

// expected-note @below {{initializer declared here:}}
func.func @Bar(!arc.state<i1>) {
^bb0(%arg0: !arc.state<i1>):
  return
}

// -----

// expected-error @below {{finalizer 'Bar' arguments must match arguments of model}}
arc.model @Foo io !hw.modty<> storageBytes 42 finalizer @Bar {
^bb0(%arg0: !arc.storage):
}

// expected-note @below {{finalizer declared here:}}
func.func @Bar(!arc.state<i1>) {
^bb0(%arg0: !arc.state<i1>):
  return
}

// -----

hw.module @InvalidInitType(in %clock: !seq.clock, in %input: i7) {
  %cst = hw.constant 0 : i8
  // expected-error @below {{failed to verify that types of initial arguments match result types}}
  %res = arc.state @Bar(%input) clock %clock initial (%cst: i8) latency 1 : (i7) -> i7
}

// -----

// expected-error @below {{region with at least 1 blocks}}
arc.execute {
}

// -----

%0 = hw.constant 0 : i42
// expected-error @below {{input type mismatch: input #0}}
// expected-note @below {{expected type: 'i42'}}
// expected-note @below {{actual type: 'i19'}}
arc.execute (%0 : i42) {
^bb0(%arg0: i19):
  arc.output
}

// -----

arc.execute -> (i42) {
  // expected-error @below {{incorrect number of outputs: expected 1, but got 0}}
  arc.output
}

// -----

arc.execute -> (i42) {
  %0 = hw.constant 0 : i19
  // expected-error @below {{output type mismatch: output #0}}
  // expected-note @below {{expected type: 'i42'}}
  // expected-note @below {{actual type: 'i19'}}
  arc.output %0 : i19
}

// -----

func.func @Foo(%arg0: !arc.coroutine_state<@NotACoroutine>, %arg1: !arc.coroutine_pc<@NotACoroutine>) {
  // expected-error @below {{`NotACoroutine` does not reference a valid `arc.coroutine.define`}}
  arc.coroutine.call @NotACoroutine(%arg0, %arg1) : (!arc.coroutine_state<@NotACoroutine>, !arc.coroutine_pc<@NotACoroutine>) -> (!arc.coroutine_state<@NotACoroutine>, !arc.coroutine_pc<@NotACoroutine>)
  return
}
func.func @NotACoroutine() {
  return
}

// -----

hw.module @Foo() {
  // expected-error @below {{`NotACoroutine` does not reference a valid `arc.coroutine.define`}}
  arc.coroutine.instance @NotACoroutine() : () -> ()
}
func.func @NotACoroutine() {
  return
}

// -----

func.func @Foo(%arg0: !arc.coroutine_state<@NeedsI42>, %arg1: !arc.coroutine_pc<@NeedsI42>, %arg2: i9001) {
  // expected-error @below {{operand type mismatch: operand #0}}
  // expected-note @below {{expected type: 'i42'}}
  // expected-note @below {{actual type: 'i9001'}}
  arc.coroutine.call @NeedsI42(%arg0, %arg1, %arg2) : (!arc.coroutine_state<@NeedsI42>, !arc.coroutine_pc<@NeedsI42>, i9001) -> (!arc.coroutine_state<@NeedsI42>, !arc.coroutine_pc<@NeedsI42>)
  return
}
arc.coroutine.define @NeedsI42(%arg0: i42) {
  arc.coroutine.return
}

// -----

hw.module @Foo(in %a: i9001) {
  // expected-error @below {{operand type mismatch: operand #0}}
  // expected-note @below {{expected type: 'i42'}}
  // expected-note @below {{actual type: 'i9001'}}
  arc.coroutine.instance @NeedsI42(%a) : (i9001) -> ()
}
arc.coroutine.define @NeedsI42(%arg0: i42) -> (i1, i64) {
  %c0_i1 = hw.constant 0 : i1
  %c0_i64 = hw.constant 0 : i64
  arc.coroutine.return %c0_i1, %c0_i64 : i1, i64
}

// -----

func.func @Foo(%arg0: !arc.coroutine_state<@BarB>, %arg1: !arc.coroutine_pc<@BarB>) {
  // expected-error @below {{bound to the op's callee symbol}}
  arc.coroutine.call @BarA(%arg0, %arg1) : (!arc.coroutine_state<@BarB>, !arc.coroutine_pc<@BarB>) -> (!arc.coroutine_state<@BarB>, !arc.coroutine_pc<@BarB>)
  return
}
arc.coroutine.define @BarA() {
  arc.coroutine.return
}
arc.coroutine.define @BarB() {
  arc.coroutine.return
}

// -----

hw.module @Foo() {
  // expected-error @below {{`DoesNotExist` does not reference a valid `arc.coroutine.define`}}
  arc.coroutine.instance @DoesNotExist() : () -> ()
}

// -----

func.func @Foo(%arg0: !arc.coroutine_state<@DoesNotExist>, %arg1: !arc.coroutine_pc<@DoesNotExist>) {
  // expected-error @below {{`DoesNotExist` does not reference a valid `arc.coroutine.define`}}
  arc.coroutine.call @DoesNotExist(%arg0, %arg1) : (!arc.coroutine_state<@DoesNotExist>, !arc.coroutine_pc<@DoesNotExist>) -> (!arc.coroutine_state<@DoesNotExist>, !arc.coroutine_pc<@DoesNotExist>)
  return
}

// -----

hw.module @Foo() {
  // expected-error @below {{referenced coroutine `Bar` must produce an `i64` wakeup time as its last result}}
  arc.coroutine.instance @Bar() : () -> ()
}
arc.coroutine.define @Bar() {
  arc.coroutine.return
}

// -----

hw.module @Foo() {
  // expected-error @below {{referenced coroutine `Bar` must produce an `i64` wakeup time as its last result}}
  arc.coroutine.instance @Bar() : () -> i42
}
arc.coroutine.define @Bar() -> i42 {
  %c0_i42 = hw.constant 0 : i42
  arc.coroutine.return %c0_i42 : i42
}

// -----

hw.module @Foo(in %a: i42) {
  // expected-error @below {{referenced coroutine `Bar` must produce an observe bitmask with one bit per argument (`i1`) as its second-to-last result}}
  arc.coroutine.instance @Bar(%a) : (i42) -> ()
}
// One argument, so the bitmask must be `i1`, but here it is `i8`.
arc.coroutine.define @Bar(%arg0: i42) -> (i8, i64) {
  %c0_i8 = hw.constant 0 : i8
  %c0_i64 = hw.constant 0 : i64
  arc.coroutine.return %c0_i8, %c0_i64 : i8, i64
}

// -----

arc.coroutine.define @Foo() -> i42 {
  // expected-error @below {{incorrect number of yielded values: expected 1, but got 0}}
  arc.coroutine.return
}

// -----

arc.coroutine.define @Foo() -> i42 {
  %c0_i9001 = hw.constant 0 : i9001
  // expected-error @below {{yielded value type mismatch: yielded value #0}}
  // expected-note @below {{expected type: 'i42'}}
  // expected-note @below {{actual type: 'i9001'}}
  arc.coroutine.return %c0_i9001 : i9001
}

// -----

arc.coroutine.define @Foo() -> i42 {
  // expected-error @below {{incorrect number of yielded values: expected 1, but got 0}}
  arc.coroutine.halt
}

// -----

arc.coroutine.define @Foo() -> i42 {
  %c0_i9001 = hw.constant 0 : i9001
  // expected-error @below {{yielded value type mismatch: yielded value #0}}
  // expected-note @below {{expected type: 'i42'}}
  // expected-note @below {{actual type: 'i9001'}}
  arc.coroutine.halt %c0_i9001 : i9001
}

// -----

arc.coroutine.define @Foo() -> i42 {
  // expected-error @below {{incorrect number of yielded values: expected 1, but got 0}}
  arc.coroutine.yield ^bb1
^bb1:
  %c0_i42 = hw.constant 0 : i42
  arc.coroutine.halt %c0_i42 : i42
}

// -----

arc.coroutine.define @Foo() -> i42 {
  %c0_i9001 = hw.constant 0 : i9001
  // expected-error @below {{yielded value type mismatch: yielded value #0}}
  // expected-note @below {{expected type: 'i42'}}
  // expected-note @below {{actual type: 'i9001'}}
  arc.coroutine.yield (%c0_i9001 : i9001), ^bb1
^bb1:
  %c0_i42 = hw.constant 0 : i42
  arc.coroutine.halt %c0_i42 : i42
}

// -----

arc.coroutine.define @Foo(%arg0: i42) {
  // expected-error @below {{branch has 1 operands for successor #0, but target block has 0}}
  arc.coroutine.yield ^bb1
^bb1:
  arc.coroutine.halt
}

// -----

arc.coroutine.define @Foo(%arg0: i42) {
  %c0_i42 = hw.constant 0 : i42
  // expected-error @below {{branch has 2 operands for successor #0, but target block has 1}}
  arc.coroutine.yield ^bb1(%c0_i42 : i42)
^bb1(%arg1: i42):
  arc.coroutine.halt
}

// -----

arc.coroutine.define @Foo(%arg0: i42) {
  // expected-error @below {{destination resume argument type mismatch: destination resume argument #0}}
  // expected-note @below {{expected type: 'i42'}}
  // expected-note @below {{actual type: 'i9001'}}
  arc.coroutine.yield ^bb1
^bb1(%arg1: i9001):
  arc.coroutine.halt
}

// -----

arc.coroutine.define @Foo(%arg0: i42) {
  %c0_i42 = hw.constant 0 : i42
  // expected-error @below {{type mismatch for bb argument #1 of successor #0}}
  arc.coroutine.yield ^bb1(%c0_i42 : i42)
^bb1(%arg1: i42, %arg2: i9001):
  arc.coroutine.halt
}
