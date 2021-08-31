// RUN: circt-opt %s -mlir-print-op-generic | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_bitwise
// CHECK-SAME: %[[A:.*]]: i64
// CHECK-SAME: %[[C:.*]]: i8
// CHECK-SAME: %[[SIG1:.*]]: !llhd.sig<i32>
// CHECK-SAME: %[[SIG2:.*]]: !llhd.sig<i4>
// CHECK-SAME: %[[SIGARRAY4:.*]]: !llhd.sig<!hw.array<4xi8>>
// CHECK-SAME: %[[SIGARRAY2:.*]]: !llhd.sig<!hw.array<2xi8>>
// CHECK-SAME: %[[ARRAY4:.*]]: !hw.array<4xi8>
// CHECK-SAME: %[[ARRAY2:.*]]: !hw.array<2xi8>
func @check_bitwise(%a : i64, %c : i8,
    %sig1 : !llhd.sig<i32>, %sig2 : !llhd.sig<i4>,
    %sigarray4: !llhd.sig<!hw.array<4xi8>>, %sigarray2: !llhd.sig<!hw.array<2xi8>>,
    %array4: !hw.array<4xi8>, %array2: !hw.array<2xi8>) {

  // CHECK-NEXT: %{{.*}} = comb.and %[[A]], %[[A]] : i64
  %1 = comb.and %a, %a : i64

  // CHECK-NEXT: %{{.*}} = comb.or %[[A]], %[[A]] : i64
  %2 = comb.or %a, %a : i64

  // CHECK-NEXT: %{{.*}} = comb.xor %[[A]], %[[A]] : i64
  %3 = comb.xor %a, %a : i64

  // CHECK-NEXT: %{{.*}} = llhd.shl %[[A]], %[[A]], %[[C]] : (i64, i64, i8) -> i64
  %4 = llhd.shl %a, %a, %c : (i64, i64, i8) -> i64
  // CHECK-NEXT: %{{.*}} = llhd.shl %[[SIG1]], %[[SIG2]], %[[C]] : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
  %5 = llhd.shl %sig1, %sig2, %c : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
  // CHECK-NEXT: %{{.*}} = llhd.shl %[[SIGARRAY4]], %[[SIGARRAY2]], %[[C]] : (!llhd.sig<!hw.array<4xi8>>, !llhd.sig<!hw.array<2xi8>>, i8) -> !llhd.sig<!hw.array<4xi8>>
  %6 = llhd.shl %sigarray4, %sigarray2, %c : (!llhd.sig<!hw.array<4xi8>>, !llhd.sig<!hw.array<2xi8>>, i8) -> !llhd.sig<!hw.array<4xi8>>
  // CHECK-NEXT: %{{.*}} = llhd.shl %[[ARRAY4]], %[[ARRAY2]], %[[C]] : (!hw.array<4xi8>, !hw.array<2xi8>, i8) -> !hw.array<4xi8>
  %7 = llhd.shl %array4, %array2, %c : (!hw.array<4xi8>, !hw.array<2xi8>, i8) -> !hw.array<4xi8>

  // CHECK-NEXT: %{{.*}} = llhd.shr %[[A]], %[[A]], %[[C]] : (i64, i64, i8) -> i64
  %8 = llhd.shr %a, %a, %c : (i64, i64, i8) -> i64
  // CHECK-NEXT: %{{.*}} = llhd.shr %[[SIG1]], %[[SIG2]], %[[C]] : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
  %9 = llhd.shr %sig1, %sig2, %c : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
  // CHECK-NEXT: %{{.*}} = llhd.shr %[[SIGARRAY4]], %[[SIGARRAY2]], %[[C]] : (!llhd.sig<!hw.array<4xi8>>, !llhd.sig<!hw.array<2xi8>>, i8) -> !llhd.sig<!hw.array<4xi8>>
  %10 = llhd.shr %sigarray4, %sigarray2, %c : (!llhd.sig<!hw.array<4xi8>>, !llhd.sig<!hw.array<2xi8>>, i8) -> !llhd.sig<!hw.array<4xi8>>
  // CHECK-NEXT: %{{.*}} = llhd.shr %[[ARRAY4]], %[[ARRAY2]], %[[C]] : (!hw.array<4xi8>, !hw.array<2xi8>, i8) -> !hw.array<4xi8>
  %11 = llhd.shr %array4, %array2, %c : (!hw.array<4xi8>, !hw.array<2xi8>, i8) -> !hw.array<4xi8>

  return
}
