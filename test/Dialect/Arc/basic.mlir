// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | circt-opt | FileCheck %s

// CHECK-LABEL: arc.define @Foo
arc.define @Foo(%arg0: i42, %arg1: i9) -> (i42, i9) {
  %c-1_i42 = hw.constant -1 : i42

  // CHECK: arc.output %c-1_i42, %arg1 : i42, i9
  arc.output %c-1_i42, %arg1 : i42, i9
}

arc.define @Bar(%arg0: i42) -> i42 {
  arc.output %arg0 : i42
}

// CHECK-LABEL: hw.module @Module
hw.module @Module(in %clock: !seq.clock, in %enable: i1, in %a: i42, in %b: i9) {
  // CHECK: arc.state @Foo(%a, %b) clock %clock latency 1 : (i42, i9) -> (i42, i9)
  arc.state @Foo(%a, %b) clock %clock latency 1 : (i42, i9) -> (i42, i9)

  // CHECK: arc.state @Foo(%a, %b) clock %clock enable %enable latency 1 : (i42, i9) -> (i42, i9)
  arc.state @Foo(%a, %b) clock %clock enable %enable latency 1 : (i42, i9) -> (i42, i9)
}

// CHECK-LABEL: arc.define @SupportRecurisveMemoryEffects
arc.define @SupportRecurisveMemoryEffects(%arg0: i42, %arg1: i1) {
  %0 = scf.if %arg1 -> i42 {
    %1 = comb.and %arg0, %arg0 : i42
    scf.yield %1 : i42
  } else {
    scf.yield %arg0 : i42
  }
  arc.output
}

// CHECK-LABEL: @LookupTable(%arg0: i32, %arg1: i8)
arc.define @LookupTable(%arg0: i32, %arg1: i8) -> () {
  // CHECK-NEXT: %{{.+}} = arc.lut() : () -> i32 {
  // CHECK-NEXT:   %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:   arc.output %c0_i32 : i32
  // CHECK-NEXT: }
  %0 = arc.lut () : () -> i32 {
    ^bb0():
      %0 = hw.constant 0 : i32
      arc.output %0 : i32
  }
  // CHECK-NEXT: %{{.+}} = arc.lut(%arg1, %arg0) : (i8, i32) -> i32 {
  // CHECK-NEXT: ^bb0(%arg2: i8, %arg3: i32):
  // CHECK-NEXT:   arc.output %arg3 : i32
  // CHECK-NEXT: }
  %1 = arc.lut (%arg1, %arg0) : (i8, i32) -> i32 {
    ^bb0(%arg2: i8, %arg3: i32):
      arc.output %arg3 : i32
  }
  arc.output
}

// CHECK-LABEL: func.func @StorageAccess
func.func @StorageAccess(%arg0: !arc.storage) {
  // CHECK-NEXT: arc.storage.get %arg0[42] : !arc.storage -> !arc.state<i9>
  // CHECK-NEXT: arc.storage.get %arg0[1337] : !arc.storage -> !arc.memory<4 x i19, i32>
  // CHECK-NEXT: arc.storage.get %arg0[9001] : !arc.storage -> !arc.storage
  // CHECK-NEXT: arc.storage.get %arg0[9009] : !arc.storage -> !arc.state<!llvm.ptr>
  %0 = arc.storage.get %arg0[42] : !arc.storage -> !arc.state<i9>
  %1 = arc.storage.get %arg0[1337] : !arc.storage -> !arc.memory<4 x i19, i32>
  %2 = arc.storage.get %arg0[9001] : !arc.storage -> !arc.storage
  %3 = arc.storage.get %arg0[9009] : !arc.storage -> !arc.state<!llvm.ptr>
  return
}

// CHECK-LABEL: func.func @zeroCount
func.func @zeroCount(%arg0 : i32) {
  // CHECK-NEXT: {{%.+}} = arc.zero_count leading %arg0  : i32
  %0 = arc.zero_count leading %arg0  : i32
  // CHECK-NEXT: {{%.+}} = arc.zero_count trailing %arg0  : i32
  %1 = arc.zero_count trailing %arg0  : i32
  return
}

// CHECK-LABEL: @testCallOp
arc.define @testCallOp(%arg0: i1, %arg1: i32) {
  // CHECK-NEXT: {{.*}} = arc.call @dummyCallee1(%arg0, %arg1) : (i1, i32) -> i32
  %0 = arc.call @dummyCallee1(%arg0, %arg1) : (i1, i32) -> i32
  // CHECK-NEXT: arc.call @dummyCallee2()
  arc.call @dummyCallee2() : () -> ()
  arc.output
}
arc.define @dummyCallee1(%arg0: i1, %arg1: i32) -> i32 {
  arc.output %arg1 : i32
}
arc.define @dummyCallee2() {
  arc.output
}

// CHECK-LABEL: hw.module @memoryOps
hw.module @memoryOps(in %clk: !seq.clock, in %en: i1, in %mask: i32, in %arg: i1) {
  %c0_i32 = hw.constant 0 : i32
  // CHECK: [[MEM:%.+]] = arc.memory <4 x i32, i32>
  %mem = arc.memory <4 x i32, i32>

  // CHECK-NEXT: %{{.+}} = arc.memory_read_port [[MEM]][%c0_i32] : <4 x i32, i32>
  %0 = arc.memory_read_port %mem[%c0_i32] : <4 x i32, i32>
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @identity1(%c0_i32, %c0_i32, %en) clock %clk enable latency 1 : <4 x i32, i32>, i32, i32, i1
  arc.memory_write_port %mem, @identity1(%c0_i32, %c0_i32, %en) clock %clk enable latency 1 : <4 x i32, i32>, i32, i32, i1
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @identity2(%c0_i32, %c0_i32, %en, %mask) clock %clk enable mask latency 2 : <4 x i32, i32>, i32, i32, i1, i32
  arc.memory_write_port %mem, @identity2(%c0_i32, %c0_i32, %en, %mask) clock %clk enable mask latency 2 : <4 x i32, i32>, i32, i32, i1, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @identity3(%c0_i32, %c0_i32, %mask) clock %clk mask latency 3 : <4 x i32, i32>, i32, i32, i32
  arc.memory_write_port %mem, @identity3(%c0_i32, %c0_i32, %mask) clock %clk mask latency 3 : <4 x i32, i32>, i32, i32, i32
  // CHECK-NEXT: arc.memory_write_port [[MEM]], @identity(%c0_i32, %c0_i32) clock %clk latency 4 : <4 x i32, i32>, i32, i32
  arc.memory_write_port %mem, @identity(%c0_i32, %c0_i32) clock %clk latency 4 : <4 x i32, i32>, i32, i32

  // CHECK: %{{.+}} = arc.memory_read [[MEM]][%c0_i32] : <4 x i32, i32>
  %2 = arc.memory_read %mem[%c0_i32] : <4 x i32, i32>

  // CHECK-NEXT: arc.memory_write [[MEM]][%c0_i32], %c0_i32 : <4 x i32, i32>
  arc.memory_write %mem[%c0_i32], %c0_i32 : <4 x i32, i32>
}
arc.define @identity(%arg0: i32, %arg1: i32) -> (i32, i32) {
  arc.output %arg0, %arg1 : i32, i32
}
arc.define @identity1(%arg0: i32, %arg1: i32, %arg2: i1) -> (i32, i32, i1) {
  arc.output %arg0, %arg1, %arg2 : i32, i32, i1
}
arc.define @identity2(%arg0: i32, %arg1: i32, %arg2: i1, %arg3: i32) -> (i32, i32, i1, i32) {
  arc.output %arg0, %arg1, %arg2, %arg3 : i32, i32, i1, i32
}
arc.define @identity3(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32, i32) {
  arc.output %arg0, %arg1, %arg2 : i32, i32, i32
}

hw.module @vectorize(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1, out out2: i1) {
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  %1 = arc.vectorize (%in0), (%in2) : (i1, i1) -> i1 {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1, %1 : i1, i1, i1
}

// CHECK-LABEL: hw.module @vectorize
//       CHECK: [[V0:%.+]]:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
//       CHECK: ^bb0([[A:%.+]]: i1, [[B:%.+]]: i1):
//       CHECK:   [[V1:%.+]] = comb.and [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V1]] : i1
//       CHECK: }
//       CHECK: [[V2:%.+]] = arc.vectorize (%in0), (%in2) : (i1, i1) -> i1 {
//       CHECK: ^bb0([[A:%.+]]: i1, [[B:%.+]]: i1):
//       CHECK:   [[V3:%.+]] = comb.and [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V3]] : i1
//       CHECK: }
//       CHECK: hw.output [[V0]]#0, [[V0]]#1, [[V2]] :

hw.module @vectorize_body_lowered(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1, out out2: i1, out out3: i1) {
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i2, %arg1: i2):
    %1 = arith.andi %arg0, %arg1 : i2
    arc.vectorize.return %1 : i2
  }

  %1:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: vector<2xi1>, %arg1: vector<2xi1>):
    %1 = arith.andi %arg0, %arg1 : vector<2xi1>
    arc.vectorize.return %1 : vector<2xi1>
  }

  hw.output %0#0, %0#1, %1#0, %1#1 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @vectorize_body_lowered
//  CHECK-SAME: (in [[IN0:%.+]] : i1, in [[IN1:%.+]] : i1, in [[IN2:%.+]] : i1, in [[IN3:%.+]] : i1,
//       CHECK: [[V0:%.+]]:2 = arc.vectorize ([[IN0]], [[IN1]]), ([[IN2]], [[IN2]]) : (i1, i1, i1, i1) -> (i1, i1) {
//       CHECK: ^bb0([[A:%.+]]: i2, [[B:%.+]]: i2):
//       CHECK:   [[V1:%.+]] = arith.andi [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V1]] : i2
//       CHECK: }
//       CHECK: [[V2:%.+]]:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
//       CHECK: ^bb0([[A:%.+]]: vector<2xi1>, [[B:%.+]]: vector<2xi1>):
//       CHECK:   [[V3:%.+]] = arith.andi [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V3]] : vector<2xi1>
//       CHECK: }
//       CHECK: hw.output [[V0]]#0, [[V0]]#1, [[V2]]#0, [[V2]]#1 :

hw.module @vectorize_boundary_lowered(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1, out out2: i1, out out3: i1) {
  %0 = comb.concat %in0, %in1 : i1, i1
  %1 = comb.replicate %in2 : (i1) -> i2
  %2 = arc.vectorize (%0), (%1) : (i2, i2) -> i2 {
  ^bb0(%arg0: i1, %arg1: i1):
    %3 = arith.andi %arg0, %arg1 : i1
    arc.vectorize.return %3 : i1
  }
  %3 = comb.extract %2 from 1 : (i2) -> i1
  %4 = comb.extract %2 from 0 : (i2) -> i1

  %cst = arith.constant dense<0> : vector<2xi1>
  %5 = vector.insert %in0, %cst[0] : i1 into vector<2xi1>
  %6 = vector.insert %in1, %5[1] : i1 into vector<2xi1>
  %7 = vector.broadcast %in2 : i1 to vector<2xi1>
  %8 = arc.vectorize (%6), (%7) : (vector<2xi1>, vector<2xi1>) -> vector<2xi1> {
  ^bb0(%arg0: i1, %arg1: i1):
    %9 = arith.andi %arg0, %arg1 : i1
    arc.vectorize.return %9 : i1
  }
  %9 = vector.extract %8[0] : i1 from vector<2xi1>
  %10 = vector.extract %8[1] : i1 from vector<2xi1>

  hw.output %3, %4, %9, %10 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @vectorize_boundary_lowered
//       CHECK: [[V0:%.+]] = comb.concat
//       CHECK: [[V1:%.+]] = comb.replicate
//       CHECK: [[V2:%.+]] = arc.vectorize ([[V0]]), ([[V1]]) : (i2, i2) -> i2 {
//       CHECK: ^bb0([[A:%.+]]: i1, [[B:%.+]]: i1):
//       CHECK:   [[V3:%.+]] = arith.andi [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V3]] : i1
//       CHECK: }
//       CHECK: [[V3:%.+]] = comb.extract [[V2]] from 1
//       CHECK: [[V4:%.+]] = comb.extract [[V2]] from 0
//       CHECK: vector.insert
//       CHECK: [[V5:%.+]] = vector.insert
//       CHECK: [[V6:%.+]] = vector.broadcast
//       CHECK: [[V7:%.+]] = arc.vectorize ([[V5]]), ([[V6]]) : (vector<2xi1>, vector<2xi1>) -> vector<2xi1> {
//       CHECK: ^bb0([[A:%.+]]: i1, [[B:%.+]]: i1):
//       CHECK:   [[V8:%.+]] = arith.andi [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V8]] : i1
//       CHECK: }
//       CHECK: [[V8:%.+]] = vector.extract [[V7]][0]
//       CHECK: [[V9:%.+]] = vector.extract [[V7]][1]
//       CHECK: hw.output [[V3]], [[V4]], [[V8]], [[V9]] :

hw.module @vectorize_both_sides_lowered(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1, out out2: i1, out out3: i1) {
  %0 = comb.concat %in0, %in1 : i1, i1
  %1 = comb.replicate %in2 : (i1) -> i2
  %2 = arc.vectorize (%0), (%1) : (i2, i2) -> i2 {
  ^bb0(%arg0: i2, %arg1: i2):
    %3 = arith.andi %arg0, %arg1 : i2
    arc.vectorize.return %3 : i2
  }
  %3 = comb.extract %2 from 1 : (i2) -> i1
  %4 = comb.extract %2 from 0 : (i2) -> i1

  %cst = arith.constant dense<0> : vector<2xi1>
  %5 = vector.insert %in0, %cst[0] : i1 into vector<2xi1>
  %6 = vector.insert %in1, %5[1] : i1 into vector<2xi1>
  %7 = vector.broadcast %in2 : i1 to vector<2xi1>
  %8 = arc.vectorize (%6), (%7) : (vector<2xi1>, vector<2xi1>) -> vector<2xi1> {
  ^bb0(%arg0: vector<2xi1>, %arg1: vector<2xi1>):
    %9 = arith.andi %arg0, %arg1 : vector<2xi1>
    arc.vectorize.return %9 : vector<2xi1>
  }
  %9 = vector.extract %8[0] : i1 from vector<2xi1>
  %10 = vector.extract %8[1] : i1 from vector<2xi1>

  hw.output %3, %4, %9, %10 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @vectorize_both_sides_lowered
//       CHECK: [[V0:%.+]] = comb.concat
//       CHECK: [[V1:%.+]] = comb.replicate
//       CHECK: [[V2:%.+]] = arc.vectorize ([[V0]]), ([[V1]]) : (i2, i2) -> i2 {
//       CHECK: ^bb0([[A:%.+]]: i2, [[B:%.+]]: i2):
//       CHECK:   [[V3:%.+]] = arith.andi [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V3]] : i2
//       CHECK: }
//       CHECK: [[V3:%.+]] = comb.extract [[V2]] from 1
//       CHECK: [[V4:%.+]] = comb.extract [[V2]] from 0
//       CHECK: vector.insert
//       CHECK: [[V5:%.+]] = vector.insert
//       CHECK: [[V6:%.+]] = vector.broadcast
//       CHECK: [[V7:%.+]] = arc.vectorize ([[V5]]), ([[V6]]) : (vector<2xi1>, vector<2xi1>) -> vector<2xi1> {
//       CHECK: ^bb0([[A:%.+]]: vector<2xi1>, [[B:%.+]]: vector<2xi1>):
//       CHECK:   [[V8:%.+]] = arith.andi [[A]], [[B]]
//       CHECK:   arc.vectorize.return [[V8]] : vector<2xi1>
//       CHECK: }
//       CHECK: [[V8:%.+]] = vector.extract [[V7]][0]
//       CHECK: [[V9:%.+]] = vector.extract [[V7]][1]
//       CHECK: hw.output [[V3]], [[V4]], [[V8]], [[V9]] :

// CHECK-LABEL: hw.module @sim_test
hw.module @sim_test(in %a : i8, out b : i8) {
  hw.output %a : i8
}

// CHECK-LABEL: func.func @no_attr
func.func @no_attr() {
  // CHECK: arc.sim.instantiate @sim_test as %{{.*}} {
  arc.sim.instantiate @sim_test as %model {}
  return
}

// CHECK-LABEL: func.func @with_attr
func.func @with_attr() {
  // CHECK: arc.sim.instantiate @sim_test as %{{.*}} attributes {foo = "foo"} {
  arc.sim.instantiate @sim_test as %model attributes {foo = "foo"} {}
  return
}

arc.runtime.model @rtFooModel "FooModel" numStateBytes 123

// CHECK-LABEL: func.func @with_rt
func.func @with_rt() {
  // CHECK: arc.sim.instantiate @sim_test as %{{.*}} runtime @rtFooModel("args") {
  // CHECK: arc.sim.instantiate @sim_test as %{{.*}} runtime @rtFooModel("args") attributes {foo = "foo"} {
  // CHECK: arc.sim.instantiate @sim_test as %{{.*}} runtime @rtFooModel() {
  // CHECK: arc.sim.instantiate @sim_test as %{{.*}} runtime ("args") {
  arc.sim.instantiate @sim_test as %model runtime @rtFooModel("args") {}
  arc.sim.instantiate @sim_test as %model runtime @rtFooModel("args") attributes {foo = "foo"} {}
  arc.sim.instantiate @sim_test as %model runtime @rtFooModel() {}
  arc.sim.instantiate @sim_test as %model runtime ("args") {}
  return
}

// CHECK-LABEL: func.func @ReadsWrites(
// CHECK-SAME: %arg0: !arc.state<i42>
// CHECK-SAME: %arg1: i42
// CHECK-SAME: %arg2: i1
func.func @ReadsWrites(%arg0: !arc.state<i42>, %arg1: i42, %arg2: i1) {
  // CHECK: arc.state_read %arg0 : <i42>
  arc.state_read %arg0 : <i42>
  // CHECK: arc.state_write %arg0 = %arg1 : <i42>
  arc.state_write %arg0 = %arg1 : <i42>
  return
}

func.func @Execute(%arg0: i42) {
  // CHECK: arc.execute {
  arc.execute {
    arc.output
  }
  // CHECK: arc.execute (%arg0 : i42) {
  arc.execute (%arg0 : i42) {
  ^bb0(%0: i42):
    arc.output
  }
  // CHECK: arc.execute -> (i42) {
  arc.execute -> (i42) {
    %0 = hw.constant 1337 : i42
    arc.output %0 : i42
  }
  // CHECK: arc.execute (%arg0 : i42) -> (i42) {
  arc.execute (%arg0 : i42) -> (i42) {
  ^bb0(%0: i42):
    arc.output %0 : i42
  }
  return
}

// CHECK-LABEL: func.func @CurrentTime
func.func @CurrentTime(%arg0: !arc.storage) {
  // CHECK-NEXT: arc.current_time %arg0 : !arc.storage
  %0 = arc.current_time %arg0 : !arc.storage
  return
}

// CHECK-LABEL: func.func @NextWakeup
func.func @NextWakeup(%arg0: !arc.storage, %arg1: i64) {
  // CHECK-NEXT: arc.get_next_wakeup %arg0 : !arc.storage
  %0 = arc.get_next_wakeup %arg0 : !arc.storage
  // CHECK-NEXT: arc.set_next_wakeup %arg0, %arg1 : !arc.storage
  arc.set_next_wakeup %arg0, %arg1 : !arc.storage
  return
}

// CHECK-LABEL: func.func @SimGetSetTime
func.func @SimGetSetTime() {
  arc.sim.instantiate @TimeTestModule as %model {
    // CHECK: arc.sim.get_time %{{.*}} : !arc.sim.instance<@TimeTestModule>
    %0 = arc.sim.get_time %model : !arc.sim.instance<@TimeTestModule>
    // CHECK: arc.sim.set_time %{{.*}}, %{{.*}} : !arc.sim.instance<@TimeTestModule>
    arc.sim.set_time %model, %0 : !arc.sim.instance<@TimeTestModule>
    // CHECK: arc.sim.get_next_wakeup %{{.*}} : !arc.sim.instance<@TimeTestModule>
    %1 = arc.sim.get_next_wakeup %model : !arc.sim.instance<@TimeTestModule>
    // CHECK: arc.sim.step %{{.*}} by %{{.*}} : !arc.sim.instance<@TimeTestModule>
    %tstep = arith.constant 123 : i64
    arc.sim.step %model by %tstep : !arc.sim.instance<@TimeTestModule>
  }
  return
}
hw.module @TimeTestModule() {}


// CHECK-LABEL: arc.coroutine.define @CoroutineEmpty
arc.coroutine.define @CoroutineEmpty() {
  arc.coroutine.return
}

// CHECK-LABEL: arc.coroutine.define @CoroutineNoResults
arc.coroutine.define @CoroutineNoResults(%arg0: i42) {
  // CHECK: arc.coroutine.yield ^bb1
  arc.coroutine.yield ^bb1
^bb1(%arg1: i42):
  // CHECK: arc.coroutine.yield ^bb2(%arg0 : i42)
  arc.coroutine.yield ^bb2(%arg0 : i42)
^bb2(%arg2: i42, %arg3: i42):
  // CHECK: arc.coroutine.return
  arc.coroutine.return
^bb3:
  // CHECK: arc.coroutine.halt
  arc.coroutine.halt
}

// CHECK-LABEL: arc.coroutine.define @CoroutineWithResults
arc.coroutine.define @CoroutineWithResults(%arg0: i42, %arg1: i9001) -> (i42, i9001) {
  // CHECK: arc.coroutine.yield (%arg0, %arg1 : i42, i9001), ^bb1
  arc.coroutine.yield (%arg0, %arg1 : i42, i9001), ^bb1
^bb1(%arg2: i42, %arg3: i9001):
  // CHECK: arc.coroutine.return %arg0, %arg1 : i42, i9001
  arc.coroutine.return %arg0, %arg1 : i42, i9001
^bb3:
  // CHECK: arc.coroutine.halt %arg0, %arg1 : i42, i9001
  arc.coroutine.halt %arg0, %arg1 : i42, i9001
}

// CHECK-LABEL: func.func @CoroutineCallEmpty
func.func @CoroutineCallEmpty(%arg0: !arc.coroutine_state<@CoroutineEmpty>, %arg1: !arc.coroutine_pc<@CoroutineEmpty>) {
  // CHECK: arc.coroutine.call @CoroutineEmpty(%arg0, %arg1)
  // CHECK-SAME: : (!arc.coroutine_state<@CoroutineEmpty>, !arc.coroutine_pc<@CoroutineEmpty>)
  // CHECK-SAME: -> (!arc.coroutine_state<@CoroutineEmpty>, !arc.coroutine_pc<@CoroutineEmpty>)
  %0, %1 = arc.coroutine.call @CoroutineEmpty(%arg0, %arg1) : (!arc.coroutine_state<@CoroutineEmpty>, !arc.coroutine_pc<@CoroutineEmpty>) -> (!arc.coroutine_state<@CoroutineEmpty>, !arc.coroutine_pc<@CoroutineEmpty>)
  return
}

// CHECK-LABEL: func.func @CoroutineCallWithResults
func.func @CoroutineCallWithResults(
  %arg0: !arc.coroutine_state<@CoroutineWithResults>,
  %arg1: !arc.coroutine_pc<@CoroutineWithResults>,
  %arg2: i42,
  %arg3: i9001
) {
  // CHECK: arc.coroutine.call @CoroutineWithResults(%arg0, %arg1, %arg2, %arg3)
  // CHECK-SAME: : (!arc.coroutine_state<@CoroutineWithResults>, !arc.coroutine_pc<@CoroutineWithResults>, i42, i9001)
  // CHECK-SAME: -> (!arc.coroutine_state<@CoroutineWithResults>, !arc.coroutine_pc<@CoroutineWithResults>, i42, i9001)
  %0, %1, %2:2 = arc.coroutine.call @CoroutineWithResults(%arg0, %arg1, %arg2, %arg3) : (!arc.coroutine_state<@CoroutineWithResults>, !arc.coroutine_pc<@CoroutineWithResults>, i42, i9001) -> (!arc.coroutine_state<@CoroutineWithResults>, !arc.coroutine_pc<@CoroutineWithResults>, i42, i9001)
  return
}

// CHECK: arc.coroutine.undefined_state : <@CoroutineEmpty>
arc.coroutine.undefined_state : !arc.coroutine_state<@CoroutineEmpty>
// CHECK: arc.coroutine.start_pc : <@CoroutineEmpty>
arc.coroutine.start_pc : !arc.coroutine_pc<@CoroutineEmpty>

// CHECK-LABEL: hw.module @CoroutineInstanceA
hw.module @CoroutineInstanceA(in %a: i42, out z: i9001) {
  // CHECK: arc.coroutine.instance @CoroutineInstanceB(%a) sensitive [false] : (i42) -> i9001
  %0 = arc.coroutine.instance @CoroutineInstanceB(%a) sensitive [false] : (i42) -> i9001
  hw.output %0 : i9001
}
// The coroutine produces its result, then an observe bitmask (one bit per
// argument), then the wakeup time; the instance exposes only the result.
arc.coroutine.define @CoroutineInstanceB(%arg0: i42) -> (i9001, i1, i64) {
  %c0_i9001 = hw.constant 0 : i9001
  %c0_i1 = hw.constant 0 : i1
  %c0_i64 = hw.constant 0 : i64
  arc.coroutine.halt %c0_i9001, %c0_i1, %c0_i64 : i9001, i1, i64
}
