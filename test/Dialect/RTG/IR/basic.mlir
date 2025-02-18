// RUN: circt-opt %s --verify-roundtrip | FileCheck %s

// CHECK-LABEL: rtg.sequence @ranomizedSequenceType
// CHECK-SAME: (%{{.*}}: !rtg.randomized_sequence)
rtg.sequence @ranomizedSequenceType(%seq: !rtg.randomized_sequence) {}

// CHECK-LABEL: rtg.sequence @seq
rtg.sequence @seq0() {
  %arg = arith.constant 1 : index
  // CHECK: [[LBL0:%.*]] = rtg.label_decl "label_string_{0}_{1}", %{{.*}}, %{{.*}}
  %0 = rtg.label_decl "label_string_{0}_{1}", %arg, %arg
  // CHECK: [[LBL1:%.+]] = rtg.label_unique_decl "label_string"
  %1 = rtg.label_unique_decl "label_string"
  // CHECK: rtg.label local [[LBL0]]
  rtg.label local %0
  // CHECK: rtg.label global [[LBL1]]
  rtg.label global %1
  // CHECK: rtg.label external [[LBL0]]
  rtg.label external %0
}

// CHECK-LABEL: rtg.sequence @seqAttrsAndTypeElements
// CHECK-SAME: (%arg0: !rtg.sequence<!rtg.sequence<!rtg.label, !rtg.set<index>>>) attributes {rtg.some_attr} {
rtg.sequence @seqAttrsAndTypeElements(%arg0: !rtg.sequence<!rtg.sequence<!rtg.label, !rtg.set<index>>>) attributes {rtg.some_attr} {}

// CHECK-LABEL: rtg.sequence @seq1
// CHECK-SAME: (%arg0: i32, %arg1: !rtg.sequence)
rtg.sequence @seq1(%arg0: i32, %arg1: !rtg.sequence) { }

// CHECK-LABEL: rtg.sequence @seqRandomizationAndEmbedding
rtg.sequence @seqRandomizationAndEmbedding() {
  // CHECK: [[V0:%.+]] = rtg.get_sequence @seq0
  // CHECK: [[C0:%.+]] = arith.constant 0 : i32
  // CHECK: [[V1:%.+]] = rtg.get_sequence @seq1
  // CHECK: [[V2:%.+]] = rtg.substitute_sequence [[V1]]([[C0]], [[V0]]) : !rtg.sequence<i32, !rtg.sequence>
  // CHECK: [[V3:%.+]] = rtg.randomize_sequence [[V0]]
  // CHECK: [[V4:%.+]] = rtg.randomize_sequence [[V2]]
  // CHECK: rtg.embed_sequence [[V3]]
  // CHECK: rtg.embed_sequence [[V4]]
  %0 = rtg.get_sequence @seq0 : !rtg.sequence
  %c0_i32 = arith.constant 0 : i32
  %1 = rtg.get_sequence @seq1 : !rtg.sequence<i32, !rtg.sequence>
  %2 = rtg.substitute_sequence %1(%c0_i32, %0) : !rtg.sequence<i32, !rtg.sequence>
  %3 = rtg.randomize_sequence %0
  %4 = rtg.randomize_sequence %2
  rtg.embed_sequence %3
  rtg.embed_sequence %4
}

// CHECK-LABEL: @sets
func.func @sets(%arg0: i32, %arg1: i32) {
  // CHECK: [[SET:%.+]] = rtg.set_create %arg0, %arg1 : i32
  // CHECK: [[R:%.+]] = rtg.set_select_random [[SET]] : !rtg.set<i32>
  // CHECK: [[EMPTY:%.+]] = rtg.set_create : i32
  // CHECK: [[DIFF:%.+]] = rtg.set_difference [[SET]], [[EMPTY]] : !rtg.set<i32>
  // CHECK: rtg.set_union [[SET]], [[DIFF]] : !rtg.set<i32>
  // CHECK: rtg.set_size [[SET]] : !rtg.set<i32>
  %set = rtg.set_create %arg0, %arg1 : i32
  %r = rtg.set_select_random %set : !rtg.set<i32>
  %empty = rtg.set_create : i32
  %diff = rtg.set_difference %set, %empty : !rtg.set<i32>
  %union = rtg.set_union %set, %diff : !rtg.set<i32>
  %size = rtg.set_size %set : !rtg.set<i32>

  return
}

// CHECK-LABEL: @bags
rtg.sequence @bags(%arg0: i32, %arg1: i32, %arg2: index) {
  // CHECK: [[BAG:%.+]] = rtg.bag_create (%arg2 x %arg0, %arg2 x %arg1) : i32 {rtg.some_attr}
  // CHECK: [[R:%.+]] = rtg.bag_select_random [[BAG]] : !rtg.bag<i32> {rtg.some_attr}
  // CHECK: [[EMPTY:%.+]] = rtg.bag_create : i32
  // CHECK: [[DIFF:%.+]] = rtg.bag_difference [[BAG]], [[EMPTY]] : !rtg.bag<i32> {rtg.some_attr}
  // CHECK: rtg.bag_difference [[BAG]], [[EMPTY]] inf : !rtg.bag<i32>
  // CHECK: rtg.bag_union [[BAG]], [[EMPTY]], [[DIFF]] : !rtg.bag<i32>
  // CHECK: rtg.bag_unique_size [[BAG]] : !rtg.bag<i32>
  %bag = rtg.bag_create (%arg2 x %arg0, %arg2 x %arg1) : i32 {rtg.some_attr}
  %r = rtg.bag_select_random %bag : !rtg.bag<i32> {rtg.some_attr}
  %empty = rtg.bag_create : i32
  %diff = rtg.bag_difference %bag, %empty : !rtg.bag<i32> {rtg.some_attr}
  %diff2 = rtg.bag_difference %bag, %empty inf : !rtg.bag<i32>
  %union = rtg.bag_union %bag, %empty, %diff : !rtg.bag<i32>
  %size = rtg.bag_unique_size %bag : !rtg.bag<i32>
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

// CHECK-LABEL: rtg.sequence @switch_seq
// CHECK-SAME: (%{{.*}}: !rtgtest.cpu, %{{.*}}: !rtgtest.cpu, %{{.*}}: !rtg.sequence)
rtg.sequence @switch_seq(%from: !rtgtest.cpu, %to: !rtgtest.cpu, %seq: !rtg.sequence) { }

// CHECK-LABEL: rtg.target @context_switch
rtg.target @context_switch : !rtg.dict<> {
  // CHECK: [[V0:%.+]] = rtg.get_sequence
  // CHECK: rtg.context_switch #rtg.default : !rtgtest.cpu -> #rtgtest.cpu<1> : !rtgtest.cpu, [[V0]] : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  %0 = rtg.get_sequence @switch_seq : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
  rtg.context_switch #rtg.default : !rtgtest.cpu -> #rtgtest.cpu<1>, %0 : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>

  rtg.yield
}

// CHECK-LABEL: @contexts
rtg.test @contexts : !rtg.dict<ctxt0: !rtgtest.cpu> {
^bb0(%arg0: !rtgtest.cpu):
  // CHECK: rtg.on_context {{%.+}}, {{%.+}} : !rtgtest.cpu
  %seq = rtg.get_sequence @seq0 : !rtg.sequence
  rtg.on_context %arg0, %seq : !rtgtest.cpu
}

// CHECK-LABEL: rtg.test @test : !rtg.dict<num_cpus: i32, num_modes: i32> {
// CHECK: ^bb0(%arg0: i32, %arg1: i32):
// CHECK: }
rtg.test @test : !rtg.dict<num_cpus: i32, num_modes: i32> {
^bb0(%arg0: i32, %arg1: i32):
}

// CHECK-LABEL: rtg.sequence @integerHandlingOps
rtg.sequence @integerHandlingOps(%arg0: index, %arg1: index) {
  // CHECK: rtg.random_number_in_range [%arg0, %arg1)
  rtg.random_number_in_range [%arg0, %arg1)
}
