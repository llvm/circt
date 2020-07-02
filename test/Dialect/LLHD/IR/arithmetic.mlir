//RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: func @check_arithmetic(%[[A:.*]]: i64, %[[B:.*]]: i64) {
func @check_arithmetic(%a : i64, %b : i64) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.neg %[[A]] : i64
    %0 = "llhd.neg"(%a) : (i64) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.smod %[[A]], %[[B]] : i64
    %2 = "llhd.smod"(%a, %b) : (i64, i64) -> i64

    return
}
