// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @QueueSignature
// CHECK-SAME: (%arg0: !sim.queue<i3, 4>) -> !sim.queue<i3, 4>
func.func @QueueSignature(%queue: !moore.queue<i3, 4>) -> !moore.queue<i3, 4> {
  // CHECK: return %arg0 : !sim.queue<i3, 4>
  return %queue : !moore.queue<i3, 4>
}

// CHECK-LABEL: func.func @QueueSizeResult
// CHECK-SAME: (%arg0: !sim.queue<i8, 4>) -> i32
func.func @QueueSizeResult(%queue: !moore.queue<i8, 4>) -> !moore.i32 {
  // CHECK: %[[SIZE:.*]] = sim.queue.size %arg0 : <i8, 4>
  %size = moore.builtin.size %queue : <i8, 4>
  // CHECK: %[[SIZE_I32:.*]] = builtin.unrealized_conversion_cast %[[SIZE]] : i64 to i32
  // CHECK: return %[[SIZE_I32]] : i32
  return %size : !moore.i32
}

// CHECK-LABEL: func.func @QueueAggregateSignature
// CHECK-SAME: (%arg0: !sim.queue<!hw.struct<a: i3, b: i2>, 4>) -> !sim.queue<!hw.struct<a: i3, b: i2>, 4>
func.func @QueueAggregateSignature(%queue: !moore.queue<ustruct<{a: i3, b: i2}>, 4>) -> !moore.queue<ustruct<{a: i3, b: i2}>, 4> {
  // CHECK: return %arg0 : !sim.queue<!hw.struct<a: i3, b: i2>, 4>
  return %queue : !moore.queue<ustruct<{a: i3, b: i2}>, 4>
}

moore.class.classdecl @QueueClass {
}

// CHECK-LABEL: func.func @QueueClassBoundary
// CHECK-SAME: (%arg0: !sim.queue<!llvm.ptr, 4>) -> !sim.queue<!llvm.ptr, 4>
func.func @QueueClassBoundary(%queue: !moore.queue<class<@QueueClass>, 4>) -> !moore.queue<class<@QueueClass>, 4> {
  // CHECK: return %arg0 : !sim.queue<!llvm.ptr, 4>
  return %queue : !moore.queue<class<@QueueClass>, 4>
}

// CHECK-LABEL: func.func @QueueClassConditional
// CHECK-SAME: (%arg0: i1, %arg1: !sim.queue<!llvm.ptr, 4>, %arg2: !sim.queue<!llvm.ptr, 4>) -> !sim.queue<!llvm.ptr, 4>
func.func @QueueClassConditional(%cond: !moore.i1, %lhs: !moore.queue<class<@QueueClass>, 4>, %rhs: !moore.queue<class<@QueueClass>, 4>) -> !moore.queue<class<@QueueClass>, 4> {
  // CHECK: comb.mux %arg0, %arg1, %arg2 : !sim.queue<!llvm.ptr, 4>
  %out = moore.conditional %cond : i1 -> queue<class<@QueueClass>, 4> {
    moore.yield %lhs : queue<class<@QueueClass>, 4>
  } {
    moore.yield %rhs : queue<class<@QueueClass>, 4>
  }
  return %out : !moore.queue<class<@QueueClass>, 4>
}

// CHECK-LABEL: func.func @QueueRealBoundary
// CHECK-SAME: (%arg0: !sim.queue<f64, 4>) -> !sim.queue<f64, 4>
func.func @QueueRealBoundary(%queue: !moore.queue<f64, 4>) -> !moore.queue<f64, 4> {
  // CHECK: return %arg0 : !sim.queue<f64, 4>
  return %queue : !moore.queue<f64, 4>
}

// CHECK-LABEL: func.func @QueueTimeBoundary
// CHECK-SAME: (%arg0: !sim.queue<!llhd.time, 4>) -> !sim.queue<!llhd.time, 4>
func.func @QueueTimeBoundary(%queue: !moore.queue<time, 4>) -> !moore.queue<time, 4> {
  // CHECK: return %arg0 : !sim.queue<!llhd.time, 4>
  return %queue : !moore.queue<time, 4>
}

// CHECK-LABEL: func.func @QueueRealConditional
// CHECK-SAME: (%arg0: i1, %arg1: !sim.queue<f64, 4>, %arg2: !sim.queue<f64, 4>) -> !sim.queue<f64, 4>
func.func @QueueRealConditional(%cond: !moore.i1, %lhs: !moore.queue<f64, 4>, %rhs: !moore.queue<f64, 4>) -> !moore.queue<f64, 4> {
  // CHECK: comb.mux %arg0, %arg1, %arg2 : !sim.queue<f64, 4>
  %out = moore.conditional %cond : i1 -> queue<f64, 4> {
    moore.yield %lhs : queue<f64, 4>
  } {
    moore.yield %rhs : queue<f64, 4>
  }
  return %out : !moore.queue<f64, 4>
}

// CHECK-LABEL: func.func @QueueTimeConditional
// CHECK-SAME: (%arg0: i1, %arg1: !sim.queue<!llhd.time, 4>, %arg2: !sim.queue<!llhd.time, 4>) -> !sim.queue<!llhd.time, 4>
func.func @QueueTimeConditional(%cond: !moore.i1, %lhs: !moore.queue<time, 4>, %rhs: !moore.queue<time, 4>) -> !moore.queue<time, 4> {
  // CHECK: comb.mux %arg0, %arg1, %arg2 : !sim.queue<!llhd.time, 4>
  %out = moore.conditional %cond : i1 -> queue<time, 4> {
    moore.yield %lhs : queue<time, 4>
  } {
    moore.yield %rhs : queue<time, 4>
  }
  return %out : !moore.queue<time, 4>
}

// CHECK-LABEL: func.func @UArrayQueueSignature
// CHECK-SAME: (%arg0: !hw.array<2x!sim.queue<i3, 4>>) -> !hw.array<2x!sim.queue<i3, 4>>
func.func @UArrayQueueSignature(%arg0: !moore.uarray<2 x queue<i3, 4>>) -> !moore.uarray<2 x queue<i3, 4>> {
  // CHECK: return %arg0 : !hw.array<2x!sim.queue<i3, 4>>
  return %arg0 : !moore.uarray<2 x queue<i3, 4>>
}

// CHECK-LABEL: func.func @UStructQueueSignature
// CHECK-SAME: (%arg0: !hw.struct<q: !sim.queue<i3, 4>, bits: i2>) -> !hw.struct<q: !sim.queue<i3, 4>, bits: i2>
func.func @UStructQueueSignature(%arg0: !moore.ustruct<{q: queue<i3, 4>, bits: i2}>) -> !moore.ustruct<{q: queue<i3, 4>, bits: i2}> {
  // CHECK: return %arg0 : !hw.struct<q: !sim.queue<i3, 4>, bits: i2>
  return %arg0 : !moore.ustruct<{q: queue<i3, 4>, bits: i2}>
}

// CHECK-LABEL: func.func @UUnionQueueSignature
// CHECK-SAME: (%arg0: !hw.union<q: !sim.queue<i3, 4>, bits: i3>) -> !hw.union<q: !sim.queue<i3, 4>, bits: i3>
func.func @UUnionQueueSignature(%arg0: !moore.uunion<{q: queue<i3, 4>, bits: i3}>) -> !moore.uunion<{q: queue<i3, 4>, bits: i3}> {
  // CHECK: return %arg0 : !hw.union<q: !sim.queue<i3, 4>, bits: i3>
  return %arg0 : !moore.uunion<{q: queue<i3, 4>, bits: i3}>
}

// CHECK-LABEL: func.func @QueueRefSignature
// CHECK-SAME: (%arg0: !llhd.ref<!sim.queue<i3, 4>>) -> !llhd.ref<!sim.queue<i3, 4>>
func.func @QueueRefSignature(%queue: !moore.ref<queue<i3, 4>>) -> !moore.ref<queue<i3, 4>> {
  // CHECK: return %arg0 : !llhd.ref<!sim.queue<i3, 4>>
  return %queue : !moore.ref<queue<i3, 4>>
}

// CHECK-LABEL: func.func @QueueConditional
func.func @QueueConditional(%cond: !moore.i1) {
  // CHECK: [[A_EMPTY:%.+]] = sim.queue.empty : <i32, 4>
  // CHECK: [[A_REF:%.+]] = llhd.sig [[A_EMPTY]] : !sim.queue<i32, 4>
  %a_ref = moore.variable : <!moore.queue<i32, 4>>
  // CHECK: [[B_EMPTY:%.+]] = sim.queue.empty : <i32, 4>
  // CHECK: [[B_REF:%.+]] = llhd.sig [[B_EMPTY]] : !sim.queue<i32, 4>
  %b_ref = moore.variable : <!moore.queue<i32, 4>>
  // CHECK: [[A:%.+]] = llhd.prb [[A_REF]] : !sim.queue<i32, 4>
  %a = moore.read %a_ref : <!moore.queue<i32, 4>>
  // CHECK: [[B:%.+]] = llhd.prb [[B_REF]] : !sim.queue<i32, 4>
  %b = moore.read %b_ref : <!moore.queue<i32, 4>>
  // CHECK: comb.mux %arg0, [[A]], [[B]] : !sim.queue<i32, 4>
  %sel = moore.conditional %cond : i1 -> queue<i32, 4> {
    moore.yield %a : queue<i32, 4>
  } {
    moore.yield %b : queue<i32, 4>
  }
  return
}

// CHECK-LABEL: func.func @QueueUnionConditional
// CHECK-SAME: (%arg0: i1, %arg1: !hw.union<q: !sim.queue<i3, 4>, bits: i3>, %arg2: !hw.union<q: !sim.queue<i3, 4>, bits: i3>)
func.func @QueueUnionConditional(%cond: !moore.i1, %lhs: !moore.uunion<{q: queue<i3, 4>, bits: i3}>, %rhs: !moore.uunion<{q: queue<i3, 4>, bits: i3}>) -> !moore.uunion<{q: queue<i3, 4>, bits: i3}> {
  // CHECK: comb.mux %arg0, %arg1, %arg2 : !hw.union<q: !sim.queue<i3, 4>, bits: i3>
  %result = moore.conditional %cond : i1 -> uunion<{q: queue<i3, 4>, bits: i3}> {
    moore.yield %lhs : uunion<{q: queue<i3, 4>, bits: i3}>
  } {
    moore.yield %rhs : uunion<{q: queue<i3, 4>, bits: i3}>
  }
  return %result : !moore.uunion<{q: queue<i3, 4>, bits: i3}>
}

// CHECK-LABEL: hw.module @QueueModulePorts
// CHECK-SAME: in %queue_in : !sim.queue<i3, 4>
// CHECK-SAME: out queue_out : !sim.queue<i3, 4>
moore.module @QueueModulePorts(in %queue_in : !moore.queue<i3, 4>, out queue_out : !moore.queue<i3, 4>) {
  // CHECK: hw.output %queue_in : !sim.queue<i3, 4>
  moore.output %queue_in : !moore.queue<i3, 4>
}

// CHECK-LABEL: hw.module @QueueClassModulePorts
// CHECK-SAME: in %queue_in : !sim.queue<!llvm.ptr, 4>
// CHECK-SAME: out queue_out : !sim.queue<!llvm.ptr, 4>
moore.module @QueueClassModulePorts(in %queue_in : !moore.queue<class<@QueueClass>, 4>, out queue_out : !moore.queue<class<@QueueClass>, 4>) {
  // CHECK: hw.output %queue_in : !sim.queue<!llvm.ptr, 4>
  moore.output %queue_in : !moore.queue<class<@QueueClass>, 4>
}

// CHECK-LABEL: hw.module @QueueRealModulePorts
// CHECK-SAME: in %queue_in : !sim.queue<f64, 4>
// CHECK-SAME: out queue_out : !sim.queue<f64, 4>
moore.module @QueueRealModulePorts(in %queue_in : !moore.queue<f64, 4>, out queue_out : !moore.queue<f64, 4>) {
  // CHECK: hw.output %queue_in : !sim.queue<f64, 4>
  moore.output %queue_in : !moore.queue<f64, 4>
}

// CHECK-LABEL: hw.module @QueueTimeModulePorts
// CHECK-SAME: in %queue_in : !sim.queue<!llhd.time, 4>
// CHECK-SAME: out queue_out : !sim.queue<!llhd.time, 4>
moore.module @QueueTimeModulePorts(in %queue_in : !moore.queue<time, 4>, out queue_out : !moore.queue<time, 4>) {
  // CHECK: hw.output %queue_in : !sim.queue<!llhd.time, 4>
  moore.output %queue_in : !moore.queue<time, 4>
}

// CHECK-LABEL: hw.module @QueueInstTop
// CHECK-SAME: in %queue_in : !sim.queue<i3, 4>
// CHECK-SAME: out queue_out : !sim.queue<i3, 4>
moore.module @QueueInstTop(in %queue_in : !moore.queue<i3, 4>, out queue_out : !moore.queue<i3, 4>) {
  // CHECK: hw.instance "child" @QueueInstChild(queue_in: %queue_in: !sim.queue<i3, 4>) -> (queue_out: !sim.queue<i3, 4>)
  %child.queue_out = moore.instance "child" @QueueInstChild(queue_in: %queue_in : !moore.queue<i3, 4>) -> (queue_out: !moore.queue<i3, 4>)
  // CHECK: hw.output %child.queue_out : !sim.queue<i3, 4>
  moore.output %child.queue_out : !moore.queue<i3, 4>
}

// CHECK-LABEL: hw.module private @QueueInstChild
moore.module private @QueueInstChild(in %queue_in : !moore.queue<i3, 4>, out queue_out : !moore.queue<i3, 4>) {
  // CHECK: hw.output %queue_in : !sim.queue<i3, 4>
  moore.output %queue_in : !moore.queue<i3, 4>
}

// CHECK-LABEL: hw.module @QueueClassInstTop
// CHECK-SAME: in %queue_in : !sim.queue<!llvm.ptr, 4>
// CHECK-SAME: out queue_out : !sim.queue<!llvm.ptr, 4>
moore.module @QueueClassInstTop(in %queue_in : !moore.queue<class<@QueueClass>, 4>, out queue_out : !moore.queue<class<@QueueClass>, 4>) {
  // CHECK: hw.instance "child" @QueueClassInstChild(queue_in: %queue_in: !sim.queue<!llvm.ptr, 4>) -> (queue_out: !sim.queue<!llvm.ptr, 4>)
  %child.queue_out = moore.instance "child" @QueueClassInstChild(queue_in: %queue_in : !moore.queue<class<@QueueClass>, 4>) -> (queue_out: !moore.queue<class<@QueueClass>, 4>)
  // CHECK: hw.output %child.queue_out : !sim.queue<!llvm.ptr, 4>
  moore.output %child.queue_out : !moore.queue<class<@QueueClass>, 4>
}

// CHECK-LABEL: hw.module private @QueueClassInstChild
moore.module private @QueueClassInstChild(in %queue_in : !moore.queue<class<@QueueClass>, 4>, out queue_out : !moore.queue<class<@QueueClass>, 4>) {
  // CHECK: hw.output %queue_in : !sim.queue<!llvm.ptr, 4>
  moore.output %queue_in : !moore.queue<class<@QueueClass>, 4>
}

// CHECK-LABEL: hw.module @QueueRefModulePorts
// CHECK-SAME: in %queue_ref : !llhd.ref<!sim.queue<i3, 4>>
moore.module @QueueRefModulePorts(in %queue_ref : !moore.ref<queue<i3, 4>>) {
  // CHECK: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @QueueClassRefModulePorts
// CHECK-SAME: in %queue_ref : !llhd.ref<!sim.queue<!llvm.ptr, 4>>
moore.module @QueueClassRefModulePorts(in %queue_ref : !moore.ref<queue<class<@QueueClass>, 4>>) {
  // CHECK: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @QueueRealRefModulePorts
// CHECK-SAME: in %queue_ref : !llhd.ref<!sim.queue<f64, 4>>
moore.module @QueueRealRefModulePorts(in %queue_ref : !moore.ref<queue<f64, 4>>) {
  // CHECK: hw.output
  moore.output
}

// CHECK-LABEL: hw.module @QueueTimeRefModulePorts
// CHECK-SAME: in %queue_ref : !llhd.ref<!sim.queue<!llhd.time, 4>>
moore.module @QueueTimeRefModulePorts(in %queue_ref : !moore.ref<queue<time, 4>>) {
  // CHECK: hw.output
  moore.output
}
