// RUN: circt-opt %s -arc-lower-arrays -o - | FileCheck %s

// CHECK-LABEL: @test_func_arg
// CHECK-SAME: %arg0: !arc.arrayref<2xi32>
func.func @test_func_arg(%a: !hw.array<2xi32>, %b: i32) -> i32 {
  return %b : i32
}

// CHECK-LABEL: @test_func_result
// CHECK-SAME: %arg0: !arc.arrayref<2xi32>, %arg1: !arc.arrayref<2xi32>) -> !arc.arrayref<2xi32>
// CHECK: %[[R:.*]] = arc.arrayref.copy %arg0 = %arg1
// CHECK-NEXT: return %[[R]]
func.func @test_func_result(%a: !hw.array<2xi32>) -> !hw.array<2xi32> {
  return %a : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_func_call
// CHECK-SAME: (%[[ARG0:.*]]: !arc.arrayref<2xi32>, %[[ARG1:.*]]: !arc.arrayref<2xi32>) -> !arc.arrayref<2xi32>
// CHECK-NEXT: %[[CALL:.*]] = call @test_func_result(%[[ARG0]], %[[ARG1]]) : (!arc.arrayref<2xi32>, !arc.arrayref<2xi32>) -> !arc.arrayref<2xi32>
// CHECK-NEXT: return %[[CALL]] : !arc.arrayref<2xi32>
func.func @test_func_call(%a: !hw.array<2xi32>) -> !hw.array<2xi32> {
  %0 = func.call @test_func_result(%a) : (!hw.array<2xi32>) -> !hw.array<2xi32>
  return %0 : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_aggregate_constant
// CHECK-SAME: (%[[ARG0:.*]]: !arc.arrayref<2xi32>) -> !arc.arrayref<2xi32>
// CHECK-NEXT: %[[ALLOC:.*]] = arc.arrayref.alloc init [0 : i32, 1 : i32] <2xi32>
// CHECK-NEXT: %[[COPY:.*]] = arc.arrayref.copy %[[ARG0]] = %[[ALLOC]] : <2xi32>
// CHECK-NEXT: return %[[COPY]] : !arc.arrayref<2xi32>
func.func @test_aggregate_constant() -> !hw.array<2xi32> {
  %0 = hw.aggregate_constant [0 : i32, 1 : i32] : !hw.array<2xi32>
  return %0 : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_array_get
// CHECK-SAME: (%[[ARG0:.*]]: !arc.arrayref<2xi32>, %[[ARG1:.*]]: i1) -> i32
// CHECK-NEXT: %[[IDX:.*]] = arith.index_castui %[[ARG1]] : i1 to index
// CHECK-NEXT: %[[GET:.*]] = arc.arrayref.get %[[ARG0]][%[[IDX]]] : <2xi32> -> i32
// CHECK-NEXT: return %[[GET]] : i32
func.func @test_array_get(%a: !hw.array<2xi32>, %b: i1) -> i32 {
  %0 = hw.array_get %a[%b] : !hw.array<2xi32>, i1
  return %0 : i32
}

// CHECK-LABEL: func.func @test_array_inject
// CHECK-SAME: (%[[ARG0:.*]]: !arc.arrayref<2xi32>, %[[ARG1:.*]]: !arc.arrayref<2xi32>, %[[ARG2:.*]]: i1, %[[ARG3:.*]]: i32) -> !arc.arrayref<2xi32>
// CHECK-NEXT: %[[IDX:.*]] = arith.index_castui %[[ARG2]] : i1 to index
// CHECK-NEXT: %[[COPY:.*]] = arc.arrayref.copy %[[ARG0]] = %[[ARG1]] : <2xi32>
// CHECK-NEXT: %[[INJ:.*]] = arc.arrayref.inject %[[COPY]][%[[IDX]]], %[[ARG3]] : <2xi32>, i32 -> <2xi32>
// CHECK-NEXT: return %[[INJ]] : !arc.arrayref<2xi32>
func.func @test_array_inject(%a: !hw.array<2xi32>, %b: i1, %c: i32) -> !hw.array<2xi32> {
  %0 = hw.array_inject %a[%b], %c : !hw.array<2xi32>, i1
  return %0 : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_array_slice
// CHECK-SAME: (%[[ARG0:.*]]: !arc.arrayref<2xi32>, %[[ARG1:.*]]: !arc.arrayref<4xi32>, %[[ARG2:.*]]: i2) -> !arc.arrayref<2xi32>
// CHECK-NEXT: %[[ALLOC:.*]] = arc.arrayref.alloc <4xi32>
// CHECK-NEXT: %[[COPY1:.*]] = arc.arrayref.copy %[[ALLOC]] = %[[ARG1]] : <4xi32>
// CHECK-NEXT: %[[IDX:.*]] = arith.index_castui %[[ARG2]] : i2 to index
// CHECK-NEXT: %[[SLICE:.*]] = arc.arrayref.slice %[[COPY1]][%[[IDX]]] : (!arc.arrayref<4xi32>) -> !arc.arrayref<2xi32>
// CHECK-NEXT: %[[COPY2:.*]] = arc.arrayref.copy %[[ARG0]] = %[[SLICE]] : <2xi32>
// CHECK-NEXT: return %[[COPY2]] : !arc.arrayref<2xi32>
func.func @test_array_slice(%a: !hw.array<4xi32>, %b: i2) -> !hw.array<2xi32> {
  %0 = hw.array_slice %a[%b] : (!hw.array<4xi32>) -> !hw.array<2xi32>
  return %0 : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_array_concat
// CHECK-SAME: (%[[ARG0:.*]]: !arc.arrayref<4xi32>, %[[ARG1:.*]]: !arc.arrayref<2xi32>, %[[ARG2:.*]]: !arc.arrayref<2xi32>) -> !arc.arrayref<4xi32>
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[S2:.*]] = arc.arrayref.slice %[[ARG0]][%[[C2]]] : (!arc.arrayref<4xi32>) -> !arc.arrayref<2xi32>
// CHECK-NEXT: arc.arrayref.copy %[[S2]] = %[[ARG1]] : <2xi32>
// CHECK-NEXT: %[[S0:.*]] = arc.arrayref.slice %[[ARG0]][%[[C0]]] : (!arc.arrayref<4xi32>) -> !arc.arrayref<2xi32>
// CHECK-NEXT: arc.arrayref.copy %[[S0]] = %[[ARG2]] : <2xi32>
// CHECK-NEXT: return %[[ARG0]] : !arc.arrayref<4xi32>
func.func @test_array_concat(%a: !hw.array<2xi32>, %b: !hw.array<2xi32>) -> !hw.array<4xi32> {
  %0 = hw.array_concat %a, %b : !hw.array<2xi32>, !hw.array<2xi32>
  return %0 : !hw.array<4xi32>
}

// CHECK-LABEL: func.func @test_array_create
// CHECK-SAME: (%[[ARG0:.*]]: !arc.arrayref<2xi32>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) -> !arc.arrayref<2xi32>
// CHECK-NEXT: %[[ALLOC:.*]] = arc.arrayref.alloc <2xi32>
// CHECK-NEXT: %[[CREATE:.*]] = arc.arrayref.create %[[ALLOC]] = %[[ARG1]], %[[ARG2]] : <2xi32>
// CHECK-NEXT: %[[COPY:.*]] = arc.arrayref.copy %[[ARG0]] = %[[CREATE]] : <2xi32>
// CHECK-NEXT: return %[[COPY]] : !arc.arrayref<2xi32>
func.func @test_array_create(%a: i32, %b: i32) -> !hw.array<2xi32> {
  %0 = hw.array_create %a, %b : i32
  return %0 : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_storage_get
// CHECK-SAME: -> !arc.state<!arc.arrayref<2xi32>>
// CHECK-NEXT: %0 = arc.storage.get %arg0[0] : !arc.storage<8> -> !arc.state<!arc.arrayref<2xi32>>
// CHECK-NEXT: return %0 : !arc.state<!arc.arrayref<2xi32>>
func.func @test_storage_get(%a: !arc.storage<8>) -> !arc.state<!hw.array<2xi32>> {
  %0 = arc.storage.get %a[0] : !arc.storage<8> -> !arc.state<!hw.array<2xi32>>
  return %0 : !arc.state<!hw.array<2xi32>>
}

// CHECK-LABEL: func.func @test_mux
// CHECK-NEXT: %[[S:.*]] = arith.select %arg3, %arg1, %arg2 : !arc.arrayref<2xi32>
// CHECK-NEXT: %[[R:.*]] = arc.arrayref.copy %arg0 = %[[S]]
// CHECK-NEXT: return %[[R]]
func.func @test_mux(%a: !hw.array<2xi32>, %b: !hw.array<2xi32>, %c: i1) -> !hw.array<2xi32> {
  %r = comb.mux %c, %a, %b : !hw.array<2xi32>
  return %r : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_alloc_state
// CHECK: arc.alloc_state %arg0 : (!arc.storage<8>) -> !arc.state<!arc.arrayref<2xi32>>
func.func @test_alloc_state(%arg0: !arc.storage<8>) -> !arc.state<!hw.array<2xi32>> {
  %0 = arc.alloc_state %arg0 : (!arc.storage<8>) -> !arc.state<!hw.array<2xi32>>
  return %0 : !arc.state<!hw.array<2xi32>>
}

// CHECK-LABEL: func.func @test_state_read
// CHECK: %[[X:.*]] = arc.state_read %arg1 : <!arc.arrayref<2xi32>>
// CHECK: %[[R:.*]] = arc.arrayref.copy %arg0 = %[[X]] : <2xi32>
// CHECK: return %[[R]]
func.func @test_state_read(%arg0: !arc.state<!hw.array<2xi32>>) -> !hw.array<2xi32> {
  %0 = arc.state_read %arg0 : <!hw.array<2xi32>>
  return %0 : !hw.array<2xi32>
}

// CHECK-LABEL: func.func @test_state_write
// CHECK: arc.state_write %arg0 = %arg1 : <!arc.arrayref<2xi32>>
func.func @test_state_write(%arg0: !arc.state<!hw.array<2xi32>>, %arg1: !hw.array<2xi32>) {
  arc.state_write %arg0 = %arg1 : <!hw.array<2xi32>>
  return
}
