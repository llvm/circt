// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: rtg.sequence @seq
// CHECK-SAME: attributes {rtg.some_attr} {
rtg.sequence @seq0 attributes {rtg.some_attr} {
}

// CHECK-LABEL: rtg.sequence @seq1
// CHECK: ^bb0(%arg0: i32, %arg1: !rtg.sequence):
rtg.sequence @seq1 {
^bb0(%arg0: i32, %arg1: !rtg.sequence):
}

// CHECK-LABEL: rtg.sequence @invocations
rtg.sequence @invocations {
  // CHECK: [[V0:%.+]] = rtg.sequence_closure @seq0
  // CHECK: [[C0:%.+]] = arith.constant 0 : i32
  // CHECK: [[V1:%.+]] = rtg.sequence_closure @seq1([[C0]], [[V0]] : i32, !rtg.sequence)
  // CHECK: rtg.invoke_sequence [[V0]]
  // CHECK: rtg.invoke_sequence [[V1]]
  %0 = rtg.sequence_closure @seq0
  %c0_i32 = arith.constant 0 : i32
  %1 = rtg.sequence_closure @seq1(%c0_i32, %0 : i32, !rtg.sequence)
  rtg.invoke_sequence %0
  rtg.invoke_sequence %1
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

// CHECK-LABEL: @bags
rtg.sequence @bags {
^bb0(%arg0: i32, %arg1: i32, %arg2: index):
  // CHECK: [[BAG:%.+]] = rtg.bag_create (%arg2 x %arg0, %arg2 x %arg1) : i32 {rtg.some_attr}
  // CHECK: [[R:%.+]] = rtg.bag_select_random [[BAG]] : !rtg.bag<i32> {rtg.some_attr}
  // CHECK: [[EMPTY:%.+]] = rtg.bag_create : i32
  // CHECK: rtg.bag_difference [[BAG]], [[EMPTY]] : !rtg.bag<i32> {rtg.some_attr}
  // CHECK: rtg.bag_difference [[BAG]], [[EMPTY]] inf : !rtg.bag<i32>
  %bag = rtg.bag_create (%arg2 x %arg0, %arg2 x %arg1) : i32 {rtg.some_attr}
  %r = rtg.bag_select_random %bag : !rtg.bag<i32> {rtg.some_attr}
  %empty = rtg.bag_create : i32
  %diff = rtg.bag_difference %bag, %empty : !rtg.bag<i32> {rtg.some_attr}
  %diff2 = rtg.bag_difference %bag, %empty inf : !rtg.bag<i32>
}

// CHECK-LABEL: rtg.target @empty_target : !rtg.dict<> {
// CHECK-NOT: rtg.yield
rtg.target @empty_target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @empty_test : !rtg.dict<> {
rtg.test @empty_test : !rtg.dict<> { }

// CHECK-LABEL: rtg.target @target : !rtg.dict<num_cpus: i32, num_modes: i32> {
// CHECK:   rtg.yield %{{.*}}, %{{.*}} : i32, i32
// CHECK: }
rtg.target @target : !rtg.dict<num_cpus: i32, num_modes: i32> {
  %1 = arith.constant 4 : i32
  rtg.yield %1, %1 : i32, i32
}

// CHECK-LABEL: rtg.test @test : !rtg.dict<num_cpus: i32, num_modes: i32> {
// CHECK: ^bb0(%arg0: i32, %arg1: i32):
// CHECK: }
rtg.test @test : !rtg.dict<num_cpus: i32, num_modes: i32> {
^bb0(%arg0: i32, %arg1: i32):
}
