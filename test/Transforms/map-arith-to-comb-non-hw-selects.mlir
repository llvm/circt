// RUN: circt-opt --split-input-file --verify-diagnostics --map-arith-to-comb=enable-best-effort-lowering %s | FileCheck %s

// CHECK-LABEL: func @allow_dynamic_strings
func.func @allow_dynamic_strings(%arg0: !sim.dstring, %arg1: !sim.dstring, %arg2: i1) -> !sim.dstring {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !sim.dstring
  %0 = arith.select %arg2, %arg0, %arg1 : !sim.dstring
  return %0 : !sim.dstring
}

// CHECK-LABEL: func @allow_format_strings
func.func @allow_format_strings(%arg0: !sim.fstring, %arg1: !sim.fstring, %arg2: i1) -> !sim.fstring {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !sim.fstring
  %0 = arith.select %arg2, %arg0, %arg1 : !sim.fstring
  return %0 : !sim.fstring
}

// CHECK-LABEL: func @allow_real_values
func.func @allow_real_values(%arg0: f64, %arg1: f64, %arg2: i1) -> f64 {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : f64
  %0 = arith.select %arg2, %arg0, %arg1 : f64
  return %0 : f64
}

// CHECK-LABEL: func @allow_time_values
func.func @allow_time_values(%arg0: !llhd.time, %arg1: !llhd.time, %arg2: i1) -> !llhd.time {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !llhd.time
  %0 = arith.select %arg2, %arg0, %arg1 : !llhd.time
  return %0 : !llhd.time
}

// CHECK-LABEL: func @allow_llvm_pointers
func.func @allow_llvm_pointers(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i1) -> !llvm.ptr {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !llvm.ptr
  %0 = arith.select %arg2, %arg0, %arg1 : !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: func @allow_sim_queues
func.func @allow_sim_queues(%arg0: !sim.queue<i32, 3>, %arg1: !sim.queue<i32, 3>, %arg2: i1) -> !sim.queue<i32, 3> {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !sim.queue<i32, 3>
  %0 = arith.select %arg2, %arg0, %arg1 : !sim.queue<i32, 3>
  return %0 : !sim.queue<i32, 3>
}

// CHECK-LABEL: func @allow_structs_with_non_hw_leaves
func.func @allow_structs_with_non_hw_leaves(%arg0: !hw.struct<s: !sim.dstring, t: !llhd.time>, %arg1: !hw.struct<s: !sim.dstring, t: !llhd.time>, %arg2: i1) -> !hw.struct<s: !sim.dstring, t: !llhd.time> {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.struct<s: !sim.dstring, t: !llhd.time>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.struct<s: !sim.dstring, t: !llhd.time>
  return %0 : !hw.struct<s: !sim.dstring, t: !llhd.time>
}

// CHECK-LABEL: func @allow_arrays_with_non_hw_leaves
func.func @allow_arrays_with_non_hw_leaves(%arg0: !hw.array<2x!llvm.ptr>, %arg1: !hw.array<2x!llvm.ptr>, %arg2: i1) -> !hw.array<2x!llvm.ptr> {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.array<2x!llvm.ptr>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.array<2x!llvm.ptr>
  return %0 : !hw.array<2x!llvm.ptr>
}

// CHECK-LABEL: func @allow_unions_with_non_hw_leaves
func.func @allow_unions_with_non_hw_leaves(%arg0: !hw.union<s: !sim.dstring, r: f64>, %arg1: !hw.union<s: !sim.dstring, r: f64>, %arg2: i1) -> !hw.union<s: !sim.dstring, r: f64> {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.union<s: !sim.dstring, r: f64>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.union<s: !sim.dstring, r: f64>
  return %0 : !hw.union<s: !sim.dstring, r: f64>
}

// CHECK-LABEL: func @allow_structs_with_queue_leaves
func.func @allow_structs_with_queue_leaves(%arg0: !hw.struct<q: !sim.queue<i32, 4>>, %arg1: !hw.struct<q: !sim.queue<i32, 4>>, %arg2: i1) -> !hw.struct<q: !sim.queue<i32, 4>> {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.struct<q: !sim.queue<i32, 4>>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.struct<q: !sim.queue<i32, 4>>
  return %0 : !hw.struct<q: !sim.queue<i32, 4>>
}

// CHECK-LABEL: func @allow_nested_queue_aggregates
func.func @allow_nested_queue_aggregates(%arg0: !hw.array<2xstruct<q: !sim.queue<i32, 4>>>, %arg1: !hw.array<2xstruct<q: !sim.queue<i32, 4>>>, %arg2: i1) -> !hw.array<2xstruct<q: !sim.queue<i32, 4>>> {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.array<2xstruct<q: !sim.queue<i32, 4>>>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.array<2xstruct<q: !sim.queue<i32, 4>>>
  return %0 : !hw.array<2xstruct<q: !sim.queue<i32, 4>>>
}

// CHECK-LABEL: func @allow_mixed_queue_and_non_hw_aggregates
func.func @allow_mixed_queue_and_non_hw_aggregates(%arg0: !hw.struct<q: !sim.queue<i32, 4>, s: !sim.dstring, t: !llhd.time>, %arg1: !hw.struct<q: !sim.queue<i32, 4>, s: !sim.dstring, t: !llhd.time>, %arg2: i1) -> !hw.struct<q: !sim.queue<i32, 4>, s: !sim.dstring, t: !llhd.time> {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.struct<q: !sim.queue<i32, 4>, s: !sim.dstring, t: !llhd.time>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.struct<q: !sim.queue<i32, 4>, s: !sim.dstring, t: !llhd.time>
  return %0 : !hw.struct<q: !sim.queue<i32, 4>, s: !sim.dstring, t: !llhd.time>
}
