// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @check_bitwise
// CHECK-SAME: %[[A:.*]]: i64
// CHECK-SAME: %[[C:.*]]: i8
// CHECK-SAME: %[[SIG1:.*]]: !llhd.sig<i32>
// CHECK-SAME: %[[SIG2:.*]]: !llhd.sig<i4>
// CHECK-SAME: %[[SIGVEC4:.*]]: !llhd.sig<vector<4xi8>>
// CHECK-SAME: %[[SIGVEC2:.*]]: !llhd.sig<vector<2xi8>>
// CHECK-SAME: %[[VEC4:.*]]: vector<4xi8>
// CHECK-SAME: %[[VEC2:.*]]: vector<2xi8>
func @check_bitwise(%a : i64, %c : i8,
        %sig1 : !llhd.sig<i32>, %sig2 : !llhd.sig<i4>,
        %sigvec4: !llhd.sig<vector<4xi8>>, %sigvec2: !llhd.sig<vector<2xi8>>,
        %vec4: vector<4xi8>, %vec2: vector<2xi8>) {

    // CHECK-NEXT: %{{.*}} = llhd.not %[[A]] : i64
    %0 = llhd.not %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.and %[[A]], %[[A]] : i64
    %1 = llhd.and %a, %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.or %[[A]], %[[A]] : i64
    %2 = llhd.or %a, %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.xor %[[A]], %[[A]] : i64
    %3 = llhd.xor %a, %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.shl %[[A]], %[[A]], %[[C]] : (i64, i64, i8) -> i64
    %4 = llhd.shl %a, %a, %c : (i64, i64, i8) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.shl %[[SIG1]], %[[SIG2]], %[[C]] : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %5 = llhd.shl %sig1, %sig2, %c : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    // CHECK-NEXT: %{{.*}} = llhd.shl %[[SIGVEC4]], %[[SIGVEC2]], %[[C]] : (!llhd.sig<vector<4xi8>>, !llhd.sig<vector<2xi8>>, i8) -> !llhd.sig<vector<4xi8>>
    %6 = llhd.shl %sigvec4, %sigvec2, %c : (!llhd.sig<vector<4xi8>>, !llhd.sig<vector<2xi8>>, i8) -> !llhd.sig<vector<4xi8>>
    // CHECK-NEXT: %{{.*}} = llhd.shl %[[VEC4]], %[[VEC2]], %[[C]] : (vector<4xi8>, vector<2xi8>, i8) -> vector<4xi8>
    %7 = llhd.shl %vec4, %vec2, %c : (vector<4xi8>, vector<2xi8>, i8) -> vector<4xi8>

    // CHECK-NEXT: %{{.*}} = llhd.shr %[[A]], %[[A]], %[[C]] : (i64, i64, i8) -> i64
    %8 = llhd.shr %a, %a, %c : (i64, i64, i8) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.shr %[[SIG1]], %[[SIG2]], %[[C]] : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %9 = llhd.shr %sig1, %sig2, %c : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    // CHECK-NEXT: %{{.*}} = llhd.shr %[[SIGVEC4]], %[[SIGVEC2]], %[[C]] : (!llhd.sig<vector<4xi8>>, !llhd.sig<vector<2xi8>>, i8) -> !llhd.sig<vector<4xi8>>
    %10 = llhd.shr %sigvec4, %sigvec2, %c : (!llhd.sig<vector<4xi8>>, !llhd.sig<vector<2xi8>>, i8) -> !llhd.sig<vector<4xi8>>
    // CHECK-NEXT: %{{.*}} = llhd.shr %[[VEC4]], %[[VEC2]], %[[C]] : (vector<4xi8>, vector<2xi8>, i8) -> vector<4xi8>
    %11 = llhd.shr %vec4, %vec2, %c : (vector<4xi8>, vector<2xi8>, i8) -> vector<4xi8>

    return
}
