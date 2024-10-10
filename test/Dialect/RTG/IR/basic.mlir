// RUN: circt-opt %s | FileCheck %s

// CHECK: [[SNIPPET:%.+]] = rtg.sequence attributes {rtg.some_attr} {
%snippet = rtg.sequence attributes {rtg.some_attr} {
  %arg = arith.constant 1 : i32
  // CHECK: [[LBL:%.*]] = rtg.label.decl "label_string_{0}_{1}", %{{.*}}, %{{.*}} : i32, i32 -> i32
  %0 = rtg.label.decl "label_string_{0}_{1}", %arg, %arg : i32, i32 -> i32
  // CHECK: rtg.label.decl "label_string" -> i32
  %1 = rtg.label.decl "label_string" -> i32
  // CHECK: rtg.label [[LBL]] : i32
  rtg.label %0 : i32
} -> !rtg.sequence

// CHECK: rtg.sequence
rtg.sequence {
  // CHECK: [[RATIO:%.+]] = arith.constant 1 : i32
  %ratio = arith.constant 1 : i32
  // CHECK: rtg.select_random [[[SNIPPET]]] (() : ()), [[[RATIO]]] : !rtg.sequence
  rtg.select_random [%snippet](() : ()), [%ratio] : !rtg.sequence
  // CHECK: rtg.select_random [[[SNIPPET]], [[SNIPPET]]] ((), () : (), ()), [[[RATIO]], [[RATIO]]] : !rtg.sequence, !rtg.sequence
  rtg.select_random [%snippet, %snippet]((), () : (), ()), [%ratio, %ratio] : !rtg.sequence, !rtg.sequence
} -> !rtg.sequence

%0 = rtg.sequence {
^bb0(%arg0: i32, %arg1: i64):
} -> !rtg.sequence<i32, i64>

// CHECK-LABEL: @types
// CHECK-SAME: !rtg.sequence
// CHECK-SAME: !rtg.resource
func.func @types(%arg1: !rtg.sequence, %arg2: !rtg.resource) {
  return
}





// A requirement for the snippet/sequence to run
//rtg.requires()

// A static check that the snippet/sequence should ever have been picked
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
