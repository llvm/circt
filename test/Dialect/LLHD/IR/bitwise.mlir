//RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: func @check_bitwise(%[[A:.*]]: i64, %[[B:.*]]: i64, %[[C:.*]]: i8, %[[SIG1:.*]]: !llhd.sig<i32>, %[[SIG2:.*]]: !llhd.sig<i4>) {
func @check_bitwise(%a : i64, %b : i64, %c : i8, %sig1 : !llhd.sig<i32>, %sig2 : !llhd.sig<i4>) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.not %[[A]] : i64
    %0 = "llhd.not"(%a) : (i64) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.and %[[A]], %[[B]] : i64
    %1 = "llhd.and"(%a, %b) : (i64, i64) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.or %[[A]], %[[B]] : i64
    %2 = "llhd.or"(%a, %b) : (i64, i64) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.xor %[[A]], %[[B]] : i64
    %3 = "llhd.xor"(%a, %b) : (i64, i64) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.shl %[[A]], %[[B]], %[[C]] : (i64, i64, i8) -> i64
    %4 = "llhd.shl"(%a, %b, %c) : (i64, i64, i8) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.shl %[[SIG1]], %[[SIG2]], %[[C]] : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %5 = "llhd.shl"(%sig1, %sig2, %c) : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>

    // CHECK-NEXT: %{{.*}} = llhd.shr %[[A]], %[[B]], %[[C]] : (i64, i64, i8) -> i64
    %6 = "llhd.shr"(%a, %b, %c) : (i64, i64, i8) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.shr %[[SIG1]], %[[SIG2]], %[[C]] : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %7 = "llhd.shr"(%sig1, %sig2, %c) : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>

    return
}
