// RUN: circt-opt --insert-merge-blocks %s --split-input-file | FileCheck %s


// CHECK-LABEL: func.func @noWorkNeeded(
// CHECK-SAME:    %[[ARG0:.*]]: i64, 
// CHECK-SAME:    %[[ARG1:.*]]: i1) -> i64 {
// CHECK-NEXT:   cf.cond_br %[[ARG1]], ^bb1(%[[ARG0]] : i64), ^bb2(%[[ARG0]] : i64)
// CHECK-NEXT: ^bb1(%[[VAL0:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb3(%[[VAL0]] : i64)
// CHECK-NEXT: ^bb2(%[[VAL1:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb3(%[[VAL1]] : i64)
// CHECK-NEXT: ^bb3(%[[VAL2:.*]]: i64):
// CHECK-NEXT:   return %[[VAL2]] : i64
// CHECK-NEXT: }

func.func @noWorkNeeded(%val : i64, %cond: i1) -> i64 {
  cf.cond_br %cond, ^1(%val: i64), ^2(%val: i64)
^1(%i: i64):
  cf.br ^3(%i: i64)
^2(%j: i64):
  cf.br ^3(%j: i64)
^3(%res: i64):
  return %res: i64
}

// -----

// CHECK-LABEL: func.func @blockWith3Preds(
// CHECK-SAME:    %[[ARG0:.*]]: i64,
// CHECK-SAME:    %[[ARG1:.*]]: i1,
// CHECK-SAME:    %[[ARG2:.*]]: i1) -> i64 {
// CHECK-NEXT:   cf.cond_br %[[ARG1]], ^bb1(%[[ARG0]] : i64), ^bb4(%[[ARG0]] : i64)
// CHECK-NEXT: ^bb1(%[[VAL0:.*]]: i64):
// CHECK-NEXT:   cf.cond_br %[[ARG2]], ^bb2(%[[VAL0]] : i64), ^bb3(%[[VAL0]] : i64)
// CHECK-NEXT: ^bb2(%[[VAL1:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb5(%[[VAL1]] : i64)
// CHECK-NEXT: ^bb3(%[[VAL2:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb5(%[[VAL2]] : i64)
// CHECK-NEXT: ^bb4(%[[VAL3:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb6(%[[VAL3]] : i64)
// CHECK-NEXT: ^bb5(%[[VAL4:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb6(%[[VAL4]] : i64)
// CHECK-NEXT: ^bb6(%[[VAL5:.*]]: i64):
// CHECK-NEXT:   return %[[VAL5]] : i64
// CHECK-NEXT: }

func.func @blockWith3Preds(%val : i64, %cond0: i1, %cond1: i1) -> i64 {
  cf.cond_br %cond0, ^1(%val: i64), ^4(%val: i64)
^1(%val1: i64):
  cf.cond_br %cond1, ^2(%val1: i64), ^3(%val1: i64)
^2(%val2: i64):
  cf.br ^end(%val2: i64)
^3(%val3: i64):
  cf.br ^end(%val3: i64)
^4(%val4: i64):
  cf.br ^end(%val4: i64)
^end(%res: i64):
  return %res: i64
}

// -----

// CHECK-LABEL: func.func @splitToDirectMerge(
// CHECK-SAME:    %[[ARG0:.*]]: i64,
// CHECK-SAME:    %[[ARG1:.*]]: i1) -> i64 {
// CHECK-NEXT:   cf.cond_br %[[ARG1]], ^bb1(%[[ARG0]] : i64), ^bb2(%[[ARG0]] : i64)
// CHECK-NEXT: ^bb1(%[[VAL0:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb2(%[[VAL0]] : i64)
// CHECK-NEXT: ^bb2(%[[VAL1:.*]]: i64):
// CHECK-NEXT:   return %[[VAL1]] : i64
// CHECK-NEXT: }

func.func @splitToDirectMerge(%val : i64, %cond0: i1) -> i64 {
  cf.cond_br %cond0, ^1(%val: i64), ^2(%val: i64)
^1(%val1: i64):
cf.br ^2(%val1: i64)
^2(%val2: i64):
  return %val2: i64
}

// -----

// CHECK-LABEL: func.func @splitTo3Merge(
// CHECK-SAME:    %[[ARG0:.*]]: i64,
// CHECK-SAME:    %[[ARG1:.*]]: i1,
// CHECK-SAME:    %[[ARG2:.*]]: i1) -> i64 {
// CHECK-NEXT:   cf.cond_br %[[ARG1]], ^bb1(%[[ARG0]] : i64), ^bb4(%[[ARG0]] : i64)
// CHECK-NEXT: ^bb1(%[[VAL0:.*]]: i64):
// CHECK-NEXT:   cf.cond_br %[[ARG2]], ^bb2(%[[VAL0]] : i64), ^bb3(%[[VAL0]] : i64)
// CHECK-NEXT: ^bb2(%[[VAL1:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb3(%[[VAL1]] : i64)
// CHECK-NEXT: ^bb3(%[[VAL2:.*]]: i64):
// CHECK-NEXT:   cf.br ^bb4(%[[VAL2]] : i64)
// CHECK-NEXT: ^bb4(%[[VAL3:.*]]: i64):
// CHECK-NEXT:   return %[[VAL3]] : i64
// CHECK-NEXT: }

func.func @splitTo3Merge(%val : i64, %cond0: i1, %cond1: i1) -> i64 {
  cf.cond_br %cond0, ^1(%val: i64), ^3(%val: i64)
^1(%val1: i64):
cf.cond_br %cond1, ^2(%val1: i64), ^3(%val1: i64)
^2(%val2: i64):
cf.br ^3(%val2: i64)
^3(%val3: i64):
  return %val3: i64
}
