// RUN: circt-opt %s --verify-roundtrip | FileCheck %s

// CHECK-LABEL: @constants
rtg.test @constants() {
  // CHECK-NEXT: rtg.constant #rtg.isa.immediate<2, -1> : !rtg.isa.immediate<2>
  %0 = rtg.constant #rtg.isa.immediate<2, -1>

  // CHECK-NEXT: [[V0:%.+]] = index.constant 5
  // CHECK-NEXT: rtg.isa.int_to_immediate [[V0]] : !rtg.isa.immediate<32>
  %1 = index.constant 5
  %2 = rtg.isa.int_to_immediate %1 : !rtg.isa.immediate<32>

  // CHECK-NEXT: rtg.comment "this is a comment"
  rtg.comment "this is a comment"
}

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
// CHECK-SAME: (%arg0: !rtg.sequence<!rtg.sequence<!rtg.isa.label, !rtg.set<index>>>) attributes {rtg.some_attr} {
rtg.sequence @seqAttrsAndTypeElements(%arg0: !rtg.sequence<!rtg.sequence<!rtg.isa.label, !rtg.set<index>>>) attributes {rtg.some_attr} {}

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
func.func @sets(%arg0: i32, %arg1: i32) -> !rtg.set<tuple<i32, i32>> {
  // CHECK: [[SET:%.+]] = rtg.set_create %arg0, %arg1 : i32
  // CHECK: [[R:%.+]] = rtg.set_select_random [[SET]] : !rtg.set<i32>
  // CHECK: [[EMPTY:%.+]] = rtg.set_create : i32
  // CHECK: [[DIFF:%.+]] = rtg.set_difference [[SET]], [[EMPTY]] : !rtg.set<i32>
  // CHECK: rtg.set_union [[SET]], [[DIFF]] : !rtg.set<i32>
  // CHECK: rtg.set_size [[SET]] : !rtg.set<i32>
  // CHECK: rtg.set_cartesian_product [[SET]], [[SET]] : !rtg.set<i32>, !rtg.set<i32>
  // CHECK: rtg.set_convert_to_bag [[SET]] : !rtg.set<i32>
  %set = rtg.set_create %arg0, %arg1 : i32
  %r = rtg.set_select_random %set : !rtg.set<i32>
  %empty = rtg.set_create : i32
  %diff = rtg.set_difference %set, %empty : !rtg.set<i32>
  %union = rtg.set_union %set, %diff : !rtg.set<i32>
  %size = rtg.set_size %set : !rtg.set<i32>
  %prod = rtg.set_cartesian_product %set, %set : !rtg.set<i32>, !rtg.set<i32>
  %bag = rtg.set_convert_to_bag %set : !rtg.set<i32>

  return %prod : !rtg.set<tuple<i32, i32>>
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
  // CHECK: rtg.bag_convert_to_set [[BAG]] : !rtg.bag<i32>
  %bag = rtg.bag_create (%arg2 x %arg0, %arg2 x %arg1) : i32 {rtg.some_attr}
  %r = rtg.bag_select_random %bag : !rtg.bag<i32> {rtg.some_attr}
  %empty = rtg.bag_create : i32
  %diff = rtg.bag_difference %bag, %empty : !rtg.bag<i32> {rtg.some_attr}
  %diff2 = rtg.bag_difference %bag, %empty inf : !rtg.bag<i32>
  %union = rtg.bag_union %bag, %empty, %diff : !rtg.bag<i32>
  %size = rtg.bag_unique_size %bag : !rtg.bag<i32>
  %set = rtg.bag_convert_to_set %bag : !rtg.bag<i32>
}

// CHECK-LABEL: rtg.target @empty_target : !rtg.dict<> {
// CHECK-NOT: rtg.yield
rtg.target @empty_target : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @empty_test() {
rtg.test @empty_test() { }

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
rtg.test @contexts(ctxt0 = %ctxt0: !rtgtest.cpu) {
  // CHECK: rtg.on_context {{%.+}}, {{%.+}} : !rtgtest.cpu
  %seq = rtg.get_sequence @seq0 : !rtg.sequence
  rtg.on_context %ctxt0, %seq : !rtgtest.cpu

  // CHECK: rtg.constant #rtg.any_context : !rtgtest.cpu
  rtg.constant #rtg.any_context : !rtgtest.cpu
}

// CHECK-LABEL: rtg.test @test0
// CHECK-SAME: (num_cpus = %num_cpus: i32, num_modes = %num_modes: i32) {
rtg.test @test0(num_cpus = %num_cpus: i32, num_modes = %num_modes: i32) { }

// CHECK-LABEL: rtg.test @test1
// CHECK-SAME: (num_cpus = %num_cpus: i32, num_modes = %num_modes: i32) {
rtg.test @test1(num_cpus = %a: i32, num_modes = %b: i32) { }

// CHECK-LABEL: rtg.sequence @integerHandlingOps
rtg.sequence @integerHandlingOps(%arg0: index, %arg1: index) {
  // CHECK: rtg.random_number_in_range [%arg0, %arg1)
  rtg.random_number_in_range [%arg0, %arg1)
}

// CHECK-LABEL: rtg.test @interleaveSequences
rtg.test @interleaveSequences(seq0 = %seq0: !rtg.randomized_sequence, seq1 = %seq1: !rtg.randomized_sequence) {
  // CHECK: rtg.interleave_sequences %seq0 {rtg.some_attr}
  rtg.interleave_sequences %seq0 {rtg.some_attr}
  // CHECK: rtg.interleave_sequences %seq0, %seq1 batch 4 {rtg.some_attr}
  rtg.interleave_sequences %seq0, %seq1 batch 4 {rtg.some_attr}
}

// CHECK-LABEL: rtg.test @arrays
// CHECK-SAME: (arr = [[ARR:%.+]]: !rtg.array<index>)
rtg.test @arrays(arr = %arr: !rtg.array<index>) {
  // CHECK-NEXT: [[IDX1:%.+]] = index.constant 1
  // CHECK-NEXT: [[V0:%.+]] = rtg.array_create [[ARR]], [[ARR]] : !rtg.array<index>
  // CHECK-NEXT: [[V1:%.+]] = rtg.array_create : index
  // CHECK-NEXT: [[V2:%.+]] = rtg.array_extract [[V0]][[[IDX1]]] : !rtg.array<!rtg.array<index>>
  // CHECK-NEXT: [[V3:%.+]] = rtg.array_inject [[V2]][[[IDX1]]], [[IDX1]] : !rtg.array<index>
  // CHECK-NEXT: rtg.array_size [[V3]] : !rtg.array<index>
  %idx1 = index.constant 1
  %0 = rtg.array_create %arr, %arr : !rtg.array<index>
  %1 = rtg.array_create : index
  %2 = rtg.array_extract %0[%idx1] : !rtg.array<!rtg.array<index>>
  %3 = rtg.array_inject %2[%idx1], %idx1 : !rtg.array<index>
  %4 = rtg.array_size %3 : !rtg.array<index>
}

// CHECK-LABEL: rtg.test @tuples
rtg.test @tuples() {
  // CHECK-NEXT: [[IDX0:%.+]] = index.constant 0
  // CHECK-NEXT: [[TRUE:%.+]] = index.bool.constant true
  // CHECK-NEXT: [[TUPLE:%.+]] = rtg.tuple_create [[IDX0]], [[TRUE]] : index, i1
  // CHECK-NEXT: rtg.tuple_extract [[TUPLE]] at 1 : tuple<index, i1>
  %idx0 = index.constant 0
  %true = index.bool.constant true
  %0 = rtg.tuple_create %idx0, %true : index, i1
  %1 = rtg.tuple_extract %0 at 1 : tuple<index, i1>
}

// CHECK-LABEL: @memoryBlocks : !rtg.dict<mem_base_address: !rtg.isa.immediate<32>, mem_block: !rtg.isa.memory_block<32>, mem_size: index>
rtg.target @memoryBlocks : !rtg.dict<mem_base_address: !rtg.isa.immediate<32>, mem_block: !rtg.isa.memory_block<32>, mem_size: index> {
  // CHECK: rtg.isa.memory_block_declare [0x0 - 0x8] : !rtg.isa.memory_block<32>
  %0 = rtg.isa.memory_block_declare [0x0 - 0x8] : !rtg.isa.memory_block<32>

  // CHECK: [[IDX8:%.+]] = index.constant 8
  // CHECK: [[V1:%.+]] = rtg.isa.memory_alloc %0, [[IDX8]], [[IDX8]] : !rtg.isa.memory_block<32>
  // CHECK: [[V2:%.+]] = rtg.isa.memory_base_address [[V1]] : !rtg.isa.memory<32>
  // CHECK: [[V3:%.+]] = rtg.isa.memory_size [[V1]] : !rtg.isa.memory<32>
  %idx8 = index.constant 8
  %1 = rtg.isa.memory_alloc %0, %idx8, %idx8 : !rtg.isa.memory_block<32>
  %2 = rtg.isa.memory_base_address %1 : !rtg.isa.memory<32>
  %3 = rtg.isa.memory_size %1 : !rtg.isa.memory<32>
  
  rtg.yield %2, %0, %3 : !rtg.isa.immediate<32>, !rtg.isa.memory_block<32>, index
}
