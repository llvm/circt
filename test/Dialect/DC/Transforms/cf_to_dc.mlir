// RUN: circt-opt --dc-test-cf-to-dc %s | FileCheck %s


func.func @scf.if.yield(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb3(%arg1 : i32)
^bb2:  // pred: ^bb0
  cf.br ^bb3(%arg2 : i32)
^bb3(%0: i32):  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // pred: ^bb3
  return %0 : i32
}

func.func @scf.while(%arg0: index, %arg1: index, %arg2: index) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  cf.br ^bb1(%arg0, %c1_i32 : index, i32)
^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %0, %arg1 : index
  cf.cond_br %2, ^bb2(%0, %1 : index, i32), ^bb3
^bb2(%3: index, %4: i32):  // pred: ^bb1
  %5 = arith.addi %3, %arg2 : index
  %6 = arith.addi %4, %4 : i32
  cf.br ^bb1(%5, %6 : index, i32)
^bb3:  // pred: ^bb1
  return %1 : i32
}

func.func @scf.while.nested(%arg0: index, %arg1: index, %arg2: index) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  cf.br ^bb1(%arg0, %c1_i32 : index, i32)
^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb5
  %2 = arith.cmpi slt, %0, %arg1 : index
  cf.cond_br %2, ^bb2(%0, %1 : index, i32), ^bb6
^bb2(%3: index, %4: i32):  // pred: ^bb1
  %5 = arith.addi %3, %arg2 : index
  cf.br ^bb3(%arg0, %4 : index, i32)
^bb3(%6: index, %7: i32):  // 2 preds: ^bb2, ^bb4
  %8 = arith.cmpi slt, %6, %arg1 : index
  cf.cond_br %8, ^bb4(%6, %7 : index, i32), ^bb5
^bb4(%9: index, %10: i32):  // pred: ^bb3
  %11 = arith.addi %9, %arg2 : index
  %12 = arith.addi %10, %10 : i32
  cf.br ^bb3(%11, %12 : index, i32)
^bb5:  // pred: ^bb3
  cf.br ^bb1(%5, %7 : index, i32)
^bb6:  // pred: ^bb1
  return %1 : i32
}