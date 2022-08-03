// RUN: circt-opt -lower-std-to-handshake %s --canonicalize --split-input-file | FileCheck %s

module {
  func.func @simpleDiamond(%arg0: i1, %arg1: i64) {
    cf.cond_br %arg0, ^bb1(%arg1: i64), ^bb2(%arg1: i64)
  ^bb1(%v1: i64):  // pred: ^bb0
    cf.br ^bb3(%v1: i64)
  ^bb2(%v2: i64):  // pred: ^bb0
    cf.br ^bb3(%v2: i64)
  ^bb3(%v3: i64):  // 2 preds: ^bb1, ^bb2
    return
  }
}


// -----
module {
  func.func @nestedDiamond(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    cf.cond_br %arg0, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    cf.br ^bb5
  ^bb3:  // pred: ^bb1
    cf.br ^bb5
  ^bb4:  // pred: ^bb0
    cf.br ^bb6
  ^bb5:  // 2 preds: ^bb2, ^bb3
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    return
  }
}


// -----
module {
  func.func @triangle(%arg0: i1, %val0: i64) {
  cf.cond_br %arg0, ^bb1(%val0: i64), ^bb2(%val0: i64)
  ^bb1(%val1: i64):  // pred: ^bb0
    cf.br ^bb2(%val1: i64)
  ^bb2(%val2: i64):  // 2 preds: ^bb0, ^bb1
    return
  }
}


// -----
module {
  func.func @nestedTriangle(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    cf.cond_br %arg0, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb0, ^bb3
    return
  }
}


// -----
module {
  func.func @multiple_blocks_needed(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    cf.cond_br %arg0, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb0, ^bb3
    cf.cond_br %arg0, ^bb5, ^bb8
  ^bb5:  // pred: ^bb4
    cf.cond_br %arg0, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb4, ^bb7
    return
  }
}

// -----

func.func @sameSuccessor(%cond: i1) {
  cf.cond_br %cond, ^1, ^1
^1:
  return
}


// -----
module {
  func.func @simple_loop(%arg0: i64) {
    %c1_i64 = arith.constant 1 : i64
    cf.br ^bb1(%c1_i64 : i64)
  ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi eq, %0, %arg0 : i64
    cf.cond_br %1, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %c1_i64_0 = arith.constant 1 : i64
    %2 = arith.addi %0, %c1_i64_0 : i64
    cf.br ^bb1(%2 : i64)
  ^bb3:  // pred: ^bb1
    return
  }
}


// -----
module {
  func.func @blockWith3PredsAndLoop(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    cf.cond_br %arg0, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    cf.br ^bb7
  ^bb3:  // pred: ^bb1
    cf.br ^bb7
  ^bb4:  // pred: ^bb0
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb6
    cf.cond_br %arg0, ^bb8, ^bb6
  ^bb6:  // pred: ^bb5
    cf.br ^bb5
  ^bb7:  // 2 preds: ^bb2, ^bb3
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb5, ^bb7
    return
  }
}


// -----

module {
  func.func @otherBlockOrder(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    cf.cond_br %arg0, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    cf.br ^bb7
  ^bb3:  // pred: ^bb1
    cf.br ^bb7
  ^bb4:  // pred: ^bb0
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb6
    cf.br ^bb6
  ^bb6:  // pred: ^bb5
    cf.cond_br %arg0, ^bb8, ^bb5
  ^bb7:  // 2 preds: ^bb2, ^bb3
    cf.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    return
  }
}

// -----

module {
  func.func @multiple_block_args(%arg0: i1, %arg1: i64) {
    cf.cond_br %arg0, ^bb1(%arg1 : i64), ^bb4(%arg1, %arg1 : i64, i64)
  ^bb1(%0: i64):  // pred: ^bb0
    cf.cond_br %arg0, ^bb2(%0 : i64), ^bb3(%0, %0 : i64, i64)
  ^bb2(%1: i64):  // pred: ^bb1
    cf.br ^bb5
  ^bb3(%2: i64, %3: i64):  // pred: ^bb1
    cf.br ^bb5
  ^bb4(%4: i64, %5: i64):  // pred: ^bb0
    cf.br ^bb6
  ^bb5:  // 2 preds: ^bb2, ^bb3
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    return
  }
}

// -----

// TODO check that
module {
  func.func @mergeBlockAsLoopHeader(%arg0: i1) {
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    cf.br ^bb3
  ^bb3:  // pred: ^bb1, ^bb2, ^bb4
    cf.cond_br %arg0, ^bb5, ^bb4
  ^bb4:  // pred: ^bb1, ^bb2
    cf.br ^bb3
  ^bb5:  // 2 preds: ^bb3
    return
  }
}
