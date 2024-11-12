// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: rtg.sequence @seq
// CHECK-SAME: attributes {rtg.some_attr} {
rtg.sequence @seq0 attributes {rtg.some_attr} {
  %arg = arith.constant 1 : i32
  // CHECK: [[LBL:%.*]] = rtg.label.decl "label_string_{0}_{1}", %{{.*}}, %{{.*}} : i32, i32 -> i32
  %0 = rtg.label.decl "label_string_{0}_{1}", %arg, %arg : i32, i32 -> i32
  // CHECK: rtg.label.decl "label_string" -> i32
  %1 = rtg.label.decl "label_string" -> i32
  // CHECK: rtg.label [[LBL]] : i32
  rtg.label %0 : i32
}

// CHECK-LABEL: rtg.sequence @seq1
rtg.sequence @seq1 {
  // CHECK: [[SEQ:%.+]] = rtg.sequence_closure @seq0{{$}}
  %sequence = rtg.sequence_closure @seq0
  // CHECK: [[RATIO:%.+]] = arith.constant 1 : i32
  %ratio = arith.constant 1 : i32
  // CHECK: rtg.select_random [[[SEQ]]] (() : ()), [[[RATIO]]] : !rtg.sequence
  rtg.select_random [%sequence](() : ()), [%ratio] : !rtg.sequence
  // CHECK: rtg.select_random [[[SEQ]], [[SEQ]]] ((), () : (), ()), [[[RATIO]], [[RATIO]]] : !rtg.sequence, !rtg.sequence
  rtg.select_random [%sequence, %sequence]((), () : (), ()), [%ratio, %ratio] : !rtg.sequence, !rtg.sequence
}

// CHECK-LABEL: rtg.sequence @seq2
// CHECK: ^bb0(%arg0: i32, %arg1: i64):
rtg.sequence @seq2 {
^bb0(%arg0: i32, %arg1: i64):
}

// CHECK-LABEL: @types
// CHECK-SAME: !rtg.sequence
// CHECK-SAME: !rtg.set<!rtg.context_resource>
// CHECK-SAME: !rtg.target<user: !rtg.mode, machine: !rtg.mode>
func.func @types(%arg1: !rtg.sequence, %arg2: !rtg.set<!rtg.context_resource>, %arg3: !rtg.target<user: !rtg.mode, machine: !rtg.mode>) {
  return
}

// CHECK-LABEL: @sets
func.func @sets(%arg0: i32, %arg1: i32) {
  // CHECK: [[SET:%.+]] = rtg.set_create %arg0, %arg1 : i32
  // CHECK: [[R:%.+]] = rtg.set_select_random [[SET]] : !rtg.set<i32>
  // CHECK: [[EMPTY:%.+]] = rtg.set_create : i32
  // CHECK: rtg.set_difference [[SET]], [[EMPTY]] : !rtg.set<i32>
  %set = rtg.set_create %arg0, %arg1 : i32
  %r = rtg.set_select_random %set : !rtg.set<i32>
  %empty = rtg.set_create : i32
  %diff = rtg.set_difference %set, %empty : !rtg.set<i32>

  return
}

// CHECK-LABEL: @contexts
func.func @contexts(%arg0: !rtg.context_resource, %arg1: !rtg.set<!rtg.context_resource>) {
  // CHECK: rtg.on_context %arg0 : !rtg.context_resource {
  // CHECK:   rtg.invoke %{{.*}} : !rtg.sequence
  // CHECK: }
  // CHECK: rtg.on_context %arg1 : !rtg.set<!rtg.context_resource> {
  // CHECK:   rtg.invoke %{{.*}} : !rtg.sequence
  // CHECK: }
  %seq = rtg.sequence_closure @seq0
  rtg.on_context %arg0 : !rtg.context_resource {
    rtg.invoke %seq : !rtg.sequence
  }
  rtg.on_context %arg1 : !rtg.set<!rtg.context_resource> {
    rtg.invoke %seq : !rtg.sequence
  }

  // CHECK: rtg.rendered_context [0, 1] {
  // CHECK:   rtg.invoke %{{.*}} : !rtg.sequence
  // CHECK: }, {
  // CHECK:   rtg.invoke %{{.*}} : !rtg.sequence
  // CHECK: }
  rtg.rendered_context [0, 1] {
    rtg.invoke %seq : !rtg.sequence
  }, {
    rtg.invoke %seq : !rtg.sequence
  }

  return
}

// CHECK-LABEL: rtg.target @empty_target : !rtg.target<> {
// CHECK-NOT: rtg.yield
rtg.target @empty_target : !rtg.target<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @empty_test : !rtg.target<> {
rtg.test @empty_test : !rtg.target<> { }

// CHECK-LABEL: rtg.target @target : !rtg.target<num_cpus: i32, num_modes: i32> {
// CHECK:   rtg.yield %{{.*}}, %{{.*}} : i32, i32
// CHECK: }
rtg.target @target : !rtg.target<num_cpus: i32, num_modes: i32> {
  %1 = arith.constant 4 : i32
  rtg.yield %1, %1 : i32, i32
}

// CHECK-LABEL: rtg.test @test : !rtg.target<num_cpus: i32, num_modes: i32> {
// CHECK: ^bb0(%arg0: i32, %arg1: i32):
// CHECK: }
rtg.test @test : !rtg.target<num_cpus: i32, num_modes: i32> {
^bb0(%arg0: i32, %arg1: i32):
}


// A requirement for the sequence/sequence to run
//rtg.requires()

// A static check that the sequence/sequence should ever have been picked
//rtg.static_assert()

// resource declarations
//??

// Start/end of time
//rtg.world_begin
//rtg.world_end

// start/end of this test
// how to deal with multiple contexts
//rtg.prerun
//rtg.postrun

// a context holder.  might be a cpu
// need to allow a specific (parameter), a set, all, and remainder
//rtg.context(id)

// body of test
// how to deal with multiple contexts
//rtg.body

// check a test
//rtg.check


// How to setup and specify exception handlers
// Should these be part of machine state requirements?
//??


// No suitable resource type in this dialect
//func.func @checkOnContext(
//    %arg1 : !rtg.context_resource_set<!rtg.context_resource>, 
//    %arg2: !rtg.context_resource_set<!rtg.context_resource>
//  ) {
//  rtg.label "a"
//  rtg.on_context %arg1 : !rtg.context_resource_set<!rtg.context_resource> {
//    rtg.label "b"
//    rtg.on_context %arg2 : !rtg.context_resource_set<!rtg.context_resource> {
//      rtg.label "c"
//    }
//    rtg.label "d"
//  }
//  rtg.on_context %arg2 : !rtg.context_resource_set<!rtg.context_resource> {
//    rtg.label "e"
//  }
//  return
//}
