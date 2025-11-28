// RUN: circt-opt -split-input-file --switch-to-if --convert-index-to-uint --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @simple_cmp
// CHECK-NOT: arith.index_cast
// CHECK-NOT: : index
// CHECK: %[[C2:.*]] = arith.constant 2 : i4
// CHECK: %[[CMP0:.*]] = arith.cmpi ult, %arg0, %[[C2]] : i4
// CHECK: %[[CMP1:.*]] = arith.cmpi eq, %arg1, %[[C2]] : i4
// CHECK: %[[RES:.*]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK: return %[[RES]] : i1
module {
  func.func @simple_cmp(%arg0: i4, %arg1: i4) -> i1 {
    %a = arith.index_cast %arg0 : i4 to index
    %c2 = arith.constant 2 : index
    %cmp0 = arith.cmpi ult, %a, %c2 : index
    %b = arith.index_cast %arg1 : i4 to index
    %cmp1 = arith.cmpi eq, %b, %c2 : index
    %res = arith.andi %cmp0, %cmp1 : i1
    return %res : i1
  }
}

// -----

// CHECK-LABEL: func.func @single_case
// CHECK-NOT: arith.index_cast
// CHECK: %[[C5:.*]] = arith.constant 5 : i8
// CHECK: %[[CMP:.*]] = arith.cmpi eq, %arg0, %[[C5]] : i8
// CHECK: return %[[CMP]] : i1
module {
  func.func @single_case(%cond: i8) -> i1 {
    %switch_val = arith.index_cast %cond : i8 to index
    %0 = scf.index_switch %switch_val -> i1
      case 5 {
        %t = arith.constant true
        scf.yield %t : i1
      }
      default {
        %f = arith.constant false
        scf.yield %f : i1
      }
    return %0 : i1
  }
}

// -----

// CHECK-LABEL: func.func @multi_case
// CHECK-NOT: arith.index_cast
// CHECK-NOT: : index
// CHECK-DAG: %[[CNEG:.*]] = arith.constant -3 : i3
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i3
// CHECK: %[[MASK:.*]] = arith.trunci %{{.*}} : i16 to i3
// CHECK: %[[CMP0:.*]] = arith.cmpi eq, %[[MASK]], %[[ZERO]] : i3
// CHECK: %[[RES:.*]] = scf.if %[[CMP0]] -> (i1) {
// CHECK: } else {
// CHECK:   %[[CMP1:.*]] = arith.cmpi eq, %[[MASK]], %[[CNEG]] : i3
// CHECK:   scf.yield %[[CMP1]] : i1
// CHECK: }
// CHECK: return %[[RES]] : i1
module {
  func.func @multi_case(%arg0: i16) -> i1 {
    %c5_i16 = arith.constant 5 : i16
    %cst_true = arith.constant true
    %cst_false = arith.constant false
    %shr = arith.shrui %arg0, %c5_i16 : i16
    %mask = arith.trunci %shr : i16 to i3
    %switch_val = arith.index_cast %mask : i3 to index
    %0 = scf.index_switch %switch_val -> i1
      case 0 {
        scf.yield %cst_true : i1
      }
      case 5 {
        scf.yield %cst_true : i1
      }
      default {
        scf.yield %cst_false : i1
      }
    return %0 : i1
  }
}
