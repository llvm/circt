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

// -----

// CHECK-LABEL: func.func @simple_loop(%{{.*}}: i64) {
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i64
// CHECK-NEXT:    cf.br ^[[BB1:.*]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB1]](%{{.*}}: i64):  // 2 preds: ^{{.*}}, ^[[BB2:.*]]
// CHECK-NEXT:    %1 = arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    cf.cond_br %{{.*}}, ^[[BB3:.*]], ^[[BB2]]
// CHECK-NEXT:  ^[[BB2]]:  // pred: ^[[BB1]]
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i64
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    cf.br ^[[BB1]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB3]]:  // pred: ^[[BB1]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @simple_loop(%n: i64) {
  %c0 = arith.constant 1 : i64
  cf.br ^0(%c0 : i64)
^0(%i: i64):
  %cond = arith.cmpi eq, %i, %n : i64
  cf.cond_br %cond, ^2, ^1
^1:
  %c1 = arith.constant 1 : i64
  %ni = arith.addi %i, %c1 : i64
  cf.br ^0(%ni: i64)
^2:
  return
}

// -----

// CHECK-LABEL:  func.func @blockWith3PredsAndLoop(%arg0: i64, %arg1: i1, %arg2: i1) -> i64 {
// CHECK-NEXT:     cf.cond_br %arg1, ^bb1(%arg0 : i64), ^bb4(%arg0 : i64)
// CHECK-NEXT:   ^bb1(%0: i64):  // pred: ^bb0
// CHECK-NEXT:     cf.cond_br %arg2, ^bb2(%0 : i64), ^bb3(%0 : i64)
// CHECK-NEXT:   ^bb2(%1: i64):  // pred: ^bb1
// CHECK-NEXT:     cf.br ^bb7(%1 : i64)
// CHECK-NEXT:   ^bb3(%2: i64):  // pred: ^bb1
// CHECK-NEXT:     cf.br ^bb7(%2 : i64)
// CHECK-NEXT:   ^bb4(%3: i64):  // pred: ^bb0
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     cf.br ^bb5(%c1_i64 : i64)
// CHECK-NEXT:   ^bb5(%4: i64):  // 2 preds: ^bb4, ^bb6
// CHECK-NEXT:     %5 = arith.cmpi eq, %4, %3 : i64
// CHECK-NEXT:     cf.cond_br %5, ^bb8(%4 : i64), ^bb6
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %c1_i64_0 = arith.constant 1 : i64
// CHECK-NEXT:     %6 = arith.addi %4, %c1_i64_0 : i64
// CHECK-NEXT:     cf.br ^bb5(%6 : i64)
// CHECK-NEXT:   ^bb7(%7: i64):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:     cf.br ^bb8(%7 : i64)
// CHECK-NEXT:   ^bb8(%8: i64):  // 2 preds: ^bb5, ^bb7
// CHECK-NEXT:     return %8 : i64
// CHECK-NEXT:   }

func.func @blockWith3PredsAndLoop(%val : i64, %cond0: i1, %cond1: i1) -> i64 {
  cf.cond_br %cond0, ^1(%val: i64), ^4(%val: i64)
^1(%val1: i64):
  cf.cond_br %cond1, ^2(%val1: i64), ^3(%val1: i64)
^2(%val2: i64):
  cf.br ^end(%val2: i64)
^3(%val3: i64):
  cf.br ^end(%val3: i64)
^4(%val4: i64):
  %c0 = arith.constant 1 : i64
  cf.br ^5(%c0 : i64)
^5(%i: i64):
  %cond = arith.cmpi eq, %i, %val4 : i64
  cf.cond_br %cond, ^end(%i: i64), ^6
^6:
  %c1 = arith.constant 1 : i64
  %ni = arith.addi %i, %c1 : i64
  cf.br ^5(%ni: i64)
^end(%res: i64):
  return %res: i64
}

// -----


// CHECK-LABEL: func.func @otherBlockOrder(%{{.*}}: i64, %{{.*}}: i1, %{{.*}}: i1) -> i64 {
  // CHECK-NEXT:    cf.cond_br %{{.*}}, ^[[BB1:.*]](%{{.*}} : i64), ^[[BB4:.*]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB1]](%{{.*}}: i64):  // pred: ^{{.*}}
// CHECK-NEXT:    cf.cond_br %{{.*}}, ^[[BB2:.*]](%{{.*}} : i64), ^[[BB3:.*]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB2]](%{{.*}}: i64):  // pred: ^[[BB1]]
// CHECK-NEXT:    cf.br ^[[BB7:.*]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB3]](%{{.*}}: i64):  // pred: ^[[BB1]]
// CHECK-NEXT:    cf.br ^[[BB7]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB4]](%{{.*}}: i64):  // pred: ^{{.*}}
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i64
// CHECK-NEXT:    cf.br ^[[BB5:.*]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB5]](%{{.*}}: i64):  // 2 preds: ^[[BB4]], ^[[BB6:.*]]
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i64
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    cf.br ^[[BB6]]
// CHECK-NEXT:  ^[[BB6]]:  // pred: ^[[BB5]]
// CHECK-NEXT:    %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    cf.cond_br %{{.*}}, ^[[BB8:.*]](%{{.*}} : i64), ^[[BB5]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB7]](%{{.*}}: i64):  // 2 preds: ^[[BB2]], ^[[BB3]]
// CHECK-NEXT:    cf.br ^[[BB8]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB8]](%{{.*}}: i64):  // 2 preds: ^[[BB6]], ^[[BB7]]
// CHECK-NEXT:    return %{{.*}} : i64
// CHECK-NEXT:  }

func.func @otherBlockOrder(%val : i64, %cond0: i1, %cond1: i1) -> i64 {
  cf.cond_br %cond0, ^1(%val: i64), ^4(%val: i64)
^1(%val1: i64):
  cf.cond_br %cond1, ^2(%val1: i64), ^3(%val1: i64)
^2(%val2: i64):
  cf.br ^end(%val2: i64)
^3(%val3: i64):
  cf.br ^end(%val3: i64)
^4(%val4: i64):
  %c0 = arith.constant 1 : i64
  cf.br ^5(%c0 : i64)
^5(%i: i64):
  %c1 = arith.constant 1 : i64
  %ni = arith.addi %i, %c1 : i64
  cf.br ^6
^6:
  %cond = arith.cmpi eq, %ni, %val4 : i64
  cf.cond_br %cond, ^end(%ni: i64), ^5(%ni: i64)
^end(%res: i64):
  return %res: i64
}
