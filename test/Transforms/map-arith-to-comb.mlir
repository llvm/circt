// RUN: circt-opt --split-input-file --verify-diagnostics --map-arith-to-comb=enable-best-effort-lowering %s | FileCheck %s

// CHECK-LABEL: func @basics
func.func @basics(%arg0: i32, %arg1: i32, %arg2: i1) {
  // CHECK: comb.add %arg0, %arg1 : i32
  arith.addi %arg0, %arg1 : i32
  // CHECK: comb.sub %arg0, %arg1 : i32
  arith.subi %arg0, %arg1 : i32
  // CHECK: comb.mul %arg0, %arg1 : i32
  arith.muli %arg0, %arg1 : i32
  // CHECK: comb.divs %arg0, %arg1 : i32
  arith.divsi %arg0, %arg1 : i32
  // CHECK: comb.divu %arg0, %arg1 : i32
  arith.divui %arg0, %arg1 : i32
  // CHECK: comb.mods %arg0, %arg1 : i32
  arith.remsi %arg0, %arg1 : i32
  // CHECK: comb.modu %arg0, %arg1 : i32
  arith.remui %arg0, %arg1 : i32
  // CHECK: comb.and %arg0, %arg1 : i32
  arith.andi %arg0, %arg1 : i32
  // CHECK: comb.or %arg0, %arg1 : i32
  arith.ori %arg0, %arg1 : i32
  // CHECK: comb.xor %arg0, %arg1 : i32
  arith.xori %arg0, %arg1 : i32
  // CHECK: comb.shl %arg0, %arg1 : i32
  arith.shli %arg0, %arg1 : i32
  // CHECK: comb.shrs %arg0, %arg1 : i32
  arith.shrsi %arg0, %arg1 : i32
  // CHECK: comb.shru %arg0, %arg1 : i32
  arith.shrui %arg0, %arg1 : i32
  // CHECK: comb.mux %arg2, %arg0, %arg1 : i32
  arith.select %arg2, %arg0, %arg1 : i32
  // CHECK: [[TMP1:%.*]] = comb.extract %arg1 from 31 : (i32) -> i1
  // CHECK: [[TMP2:%.*]] = comb.replicate [[TMP1]] : (i1) -> i10
  // CHECK: comb.concat [[TMP2]], %arg1 : i10, i32
  arith.extsi %arg1 : i32 to i42
  // CHECK: [[TMP:%.*]] = hw.constant 0 : i10
  // CHECK: comb.concat [[TMP]], %arg1 : i10, i32
  arith.extui %arg1 : i32 to i42
  // CHECK: comb.extract %arg1 from 0 : (i32) -> i16
  arith.trunci %arg1 : i32 to i16
  // CHECK: comb.icmp slt %arg0, %arg1 : i32
  arith.cmpi slt, %arg0, %arg1 : i32
  // CHECK: hw.constant 0 : i32
  arith.constant 0 : i32
  // CHECK: return
  return
}

// CHECK-LABEL: func @allow_hw_arrays
func.func @allow_hw_arrays(%arg0: !hw.array<9xi42>, %arg1: !hw.array<9xi42>, %arg2: i1) {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.array<9xi42>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.array<9xi42>
  return
}

// CHECK-LABEL: func @allow_hw_structs
func.func @allow_hw_structs(%arg0: !hw.struct<a: i42, b: i1337>, %arg1: !hw.struct<a: i42, b: i1337>, %arg2: i1) {
  // CHECK: comb.mux %arg2, %arg0, %arg1 : !hw.struct<a: i42, b: i1337>
  %0 = arith.select %arg2, %arg0, %arg1 : !hw.struct<a: i42, b: i1337>
  return
}

// -----

hw.module @invalidVector(in %arg0 : vector<4xi32>) {
  // expected-error @+1 {{failed to legalize operation 'arith.extsi' that was explicitly marked illegal}}
    %0 = arith.extsi %arg0 : vector<4xi32> to vector<4xi33>
}
