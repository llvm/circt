// RUN: circt-opt %s | FileCheck %s

// CHECK: [[SNIPPET:%.+]] = rtg.snippet attributes {rtg.some_attr} {
%snippet = rtg.snippet attributes {rtg.some_attr} {
  %arg = arith.constant 1 : i32
  // CHECK: rtg.label "label_string_{0}_{1}", %{{.*}}, %{{.*}} : i32, i32
  rtg.label "label_string_{0}_{1}", %arg, %arg : i32, i32
  // CHECK: rtg.label "label_string"
  rtg.label "label_string"
}

// CHECK: rtg.snippet
rtg.snippet {
  // CHECK: [[RATIO:%.+]] = arith.constant 1 : i32
  %ratio = arith.constant 1 : i32
  // CHECK: rtg.select_random [[[SNIPPET]]], [[[RATIO]]]
  rtg.select_random [%snippet], [%ratio]
  // CHECK: rtg.select_random [[[SNIPPET]], [[SNIPPET]]], [[[RATIO]], [[RATIO]]]
  rtg.select_random [%snippet, %snippet], [%ratio, %ratio]
}

// CHECK-LABEL: @types
// CHECK-SAME: !rtg.snippet
// CHECK-SAME: !rtg.resource
func.func @types(%arg1: !rtg.snippet, %arg2: !rtg.resource) {
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
