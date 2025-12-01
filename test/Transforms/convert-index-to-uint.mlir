// RUN: circt-opt -split-input-file --convert-index-to-uint %s | FileCheck %s --check-prefixes=SIMPLE,SINGLE,MULTI

// SIMPLE-LABEL: func.func @simple_cmp
// SIMPLE-NOT: arith.index_cast
// SIMPLE-NOT: : index
// SIMPLE: %[[C2:.*]] = arith.constant 2 : i4
// SIMPLE: %[[CMP0:.*]] = arith.cmpi ult, %arg0, %[[C2]] : i4
// SIMPLE: %[[CMP1:.*]] = arith.cmpi eq, %arg1, %[[C2]] : i4
// SIMPLE: %[[RES:.*]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// SIMPLE: return %[[RES]] : i1
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

// SINGLE-LABEL: func.func @single_case
// SINGLE-NOT: arith.index_cast
// SINGLE: %[[C5:.*]] = arith.constant 5 : i8
// SINGLE: %[[CMP:.*]] = arith.cmpi eq, %arg0, %[[C5]] : i8
// SINGLE: return %[[CMP]] : i1
module {
  func.func @single_case(%cond: i8) -> i1 {
    %switch_val = arith.index_cast %cond : i8 to index
    %c5 = arith.constant 5 : index
    %cmp = arith.cmpi eq, %switch_val, %c5 : index
    return %cmp : i1
  }
}

// -----

// MULTI-LABEL: func.func @multi_case
// MULTI-NOT: arith.index_cast
// MULTI-NOT: : index
// MULTI-DAG: %[[CNEG:.*]] = arith.constant -3 : i3
// MULTI-DAG: %[[ZERO:.*]] = arith.constant 0 : i3
// MULTI-DAG: %[[TRUE:.*]] = arith.constant true
// MULTI: %[[MASK:.*]] = arith.trunci %{{.*}} : i16 to i3
// MULTI: %[[CMP0:.*]] = arith.cmpi eq, %[[MASK]], %[[ZERO]] : i3
// MULTI: %[[RES:.*]] = scf.if %[[CMP0]] -> (i1) {
// MULTI:   scf.yield %[[TRUE]] : i1
// MULTI: } else {
// MULTI:   %[[CMP1:.*]] = arith.cmpi eq, %[[MASK]], %[[CNEG]] : i3
// MULTI:   scf.yield %[[CMP1]] : i1
// MULTI: }
// MULTI: return %[[RES]] : i1
module {
  func.func @multi_case(%arg0: i16) -> i1 {
    %c5_i16 = arith.constant 5 : i16
    %cst_true = arith.constant true
    %shr = arith.shrui %arg0, %c5_i16 : i16
    %mask = arith.trunci %shr : i16 to i3
    %switch_val = arith.index_cast %mask : i3 to index
    %c0 = arith.constant 0 : index
    %cmp0 = arith.cmpi eq, %switch_val, %c0 : index
    %0 = scf.if %cmp0 -> (i1) {
      scf.yield %cst_true : i1
    } else {
      %c5 = arith.constant 5 : index
      %cmp1 = arith.cmpi eq, %switch_val, %c5 : index
      scf.yield %cmp1 : i1
    }
    return %0 : i1
  }
}
