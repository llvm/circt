// RUN: circt-opt %s --convert-comb-to-aig | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %arg0: i32, in %arg1: i32, in %arg2: i32, in %arg3: i32, out out0: i32, out out1: i32) {
  // CHECK-NEXT: %[[OR_TMP:.+]] = aig.and_inv not %arg0, not %arg1, not %arg2, not %arg3 : i32
  // CHECK-NEXT: %[[OR:.+]] = aig.and_inv not %0 : i32
  // CHECK-NEXT: %[[AND:.+]] = aig.and_inv %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: hw.output %[[OR]], %[[AND]] : i32, i32
  %0 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  %1 = comb.and %arg0, %arg1, %arg2, %arg3 : i32
  hw.output %0, %1 : i32, i32
}

// CHECK-LABEL: @xor
hw.module @xor(in %arg0: i32, in %arg1: i32, in %arg2: i32, out out0: i32) {
  // CHECK-NEXT: %[[RHS_NOT_AND:.+]] = aig.and_inv not %arg1, not %arg2 : i32
  // CHECK-NEXT: %[[RHS_AND:.+]] = aig.and_inv %arg1, %arg2 : i32
  // CHECK-NEXT: %[[RHS_XOR:.+]] = aig.and_inv not %[[RHS_NOT_AND]], not %[[RHS_AND]] : i32
  // CHECK-NEXT: %[[NOT_AND:.+]] = aig.and_inv not %arg0, not %[[RHS_XOR]] : i32
  // CHECK-NEXT: %[[AND:.+]] = aig.and_inv %arg0, %[[RHS_XOR]] : i32
  // CHECK-NEXT: %[[RESULT:.+]] = aig.and_inv not %[[NOT_AND]], not %[[AND]] : i32
  // CHECK-NEXT: hw.output %[[RESULT]]
  %0 = comb.xor %arg0, %arg1, %arg2 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @pass
hw.module @pass(in %arg0: i32, in %arg1: i1, out out: i2) {
  // CHECK-NEXT: %[[EXTRACT:.*]] = comb.extract %arg0 from 2 : (i32) -> i2
  // CHECK-NEXT: %[[REPLICATE:.*]] = comb.replicate %arg1 : (i1) -> i2
  // CHECK-NEXT: %[[CONCAT:.*]] = comb.concat %arg1, %arg1 : i1, i1
  // CHECK-NEXT: %[[AND3:.*]] = aig.and_inv %[[EXTRACT]], %[[REPLICATE]], %[[CONCAT]] : i2
  // CHECK-NEXT: hw.output %[[AND3]] : i2
  %0 = comb.extract %arg0 from 2 : (i32) -> i2
  %1 = comb.replicate %arg1 : (i1) -> i2
  %2 = comb.concat %arg1, %arg1 : i1, i1
  %3 = comb.and %0, %1, %2 : i2
  hw.output %3 : i2
}

// CHECK-LABEL: @mux
hw.module @mux(in %cond: i1, in %high: !hw.array<2xi4>, in %low: !hw.array<2xi4>, out out: !hw.array<2xi4>) {
  // CHECK-NEXT: %[[HIGH:.+]] = hw.bitcast %high : (!hw.array<2xi4>) -> i8
  // CHECK-NEXT: %[[LOW:.+]] = hw.bitcast %low : (!hw.array<2xi4>) -> i8
  // CHECK-NEXT: %[[COND:.+]] = comb.replicate %cond : (i1) -> i8
  // CHECK-NEXT: %[[LHS:.+]] = aig.and_inv %[[COND]], %[[HIGH]] : i8
  // CHECK-NEXT: %[[RHS:.+]] = aig.and_inv not %[[COND]], %[[LOW]] : i8
  // CHECK-NEXT: %[[NAND:.+]] = aig.and_inv not %[[LHS]], not %[[RHS]] : i8
  // CHECK-NEXT: %[[NOT:.+]] = aig.and_inv not %[[NAND]] : i8
  // CHECK-NEXT: %[[RESULT:.+]] = hw.bitcast %[[NOT]] : (i8) -> !hw.array<2xi4>
  // CHECK-NEXT: hw.output %[[RESULT]] : !hw.array<2xi4>
  %0 = comb.mux %cond, %high, %low : !hw.array<2xi4>
  hw.output %0 : !hw.array<2xi4>
}

// CHECK-LABEL: @mux
hw.module @mux(in %cond: i1, in %high: i32, in %low: i32, out out: i32) {
  // CHECK-NEXT: %[[REPLICATE0:.*]] = comb.replicate %cond : (i1) -> i32
  // CHECK-NEXT: %[[AND1:.*]] = aig.and_inv %[[REPLICATE0]], %high : i32
  // CHECK-NEXT: %[[AND2:.*]] = aig.and_inv not %[[REPLICATE0]], %low : i32
  // CHECK-NEXT: %[[AND3:.*]] = aig.and_inv not %[[AND1]], not %[[AND2]] : i32
  // CHECK-NEXT: %[[AND4:.*]] = aig.and_inv not %[[AND3]] : i32
  // CHECK-NEXT: hw.output %[[AND4]] : i32
  %0 = comb.mux %cond, %high, %low : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @datapass
hw.module @datapass(in %arg0: i32, in %arg1: i1, out out: i2) {
  // CHECK-NEXT: %[[EXTRACT0:.*]] = comb.extract %arg0 from 2 : (i32) -> i2
  // CHECK-NEXT: %[[REPLICATE1:.*]] = comb.replicate %arg1 : (i1) -> i2
  // CHECK-NEXT: %[[CONCAT2:.*]] = comb.concat %arg1, %arg1 : i1, i1
  // CHECK-NEXT: %[[AND3:.*]] = aig.and_inv %[[EXTRACT0]], %[[REPLICATE1]], %[[CONCAT2]] : i2
  // CHECK-NEXT: hw.output %[[AND3]] : i2
  %0 = comb.extract %arg0 from 2 : (i32) -> i2
  %1 = comb.replicate %arg1 : (i1) -> i2
  %2 = comb.concat %arg1, %arg1 : i1, i1
  %3 = comb.and %0, %1, %2 : i2
  hw.output %3 : i2
}