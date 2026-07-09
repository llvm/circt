// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// $clog2 lowers to a priority-encode mux chain over (x - 1), with a defined
// result of 0 for a zero input (IEEE 1800-2017 § 20.8.1).

// CHECK-LABEL: func.func @Clog2
func.func @Clog2(%arg0: !moore.i4) -> !moore.i4 {
  // CHECK-DAG: [[ONE:%.+]] = hw.constant 1 : i4
  // CHECK-DAG: [[ZERO:%.+]] = hw.constant 0 : i4
  // CHECK: [[XM1:%.+]] = comb.sub %arg0, [[ONE]]
  // CHECK: [[B0:%.+]] = comb.extract [[XM1]] from 0 : (i4) -> i1
  // CHECK: [[CNT1:%.+]] = hw.constant 1 : i4
  // CHECK: [[M0:%.+]] = comb.mux [[B0]], [[CNT1]], [[ZERO]]
  // CHECK: [[B1:%.+]] = comb.extract [[XM1]] from 1 : (i4) -> i1
  // CHECK: [[CNT2:%.+]] = hw.constant 2 : i4
  // CHECK: [[M1:%.+]] = comb.mux [[B1]], [[CNT2]], [[M0]]
  // CHECK: [[B2:%.+]] = comb.extract [[XM1]] from 2 : (i4) -> i1
  // CHECK: [[CNT3:%.+]] = hw.constant 3 : i4
  // CHECK: [[M2:%.+]] = comb.mux [[B2]], [[CNT3]], [[M1]]
  // CHECK: [[B3:%.+]] = comb.extract [[XM1]] from 3 : (i4) -> i1
  // CHECK: [[CNT4:%.+]] = hw.constant 4 : i4
  // CHECK: [[M3:%.+]] = comb.mux [[B3]], [[CNT4]], [[M2]]
  // CHECK: [[ISZERO:%.+]] = comb.icmp eq %arg0, [[ZERO]]
  // CHECK: [[RESULT:%.+]] = comb.mux [[ISZERO]], [[ZERO]], [[M3]]
  %0 = moore.builtin.clog2 %arg0 : i4
  // CHECK: return [[RESULT]]
  return %0 : !moore.i4
}
