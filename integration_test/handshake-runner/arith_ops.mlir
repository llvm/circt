// RUN: handshake-runner %s 6 3 | FileCheck %s
// RUN: circt-opt -lower-cf-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner - 6 3 | FileCheck %s
// CHECK: 74

module {
  func.func @main(%a: i32, %b: i32) -> i32 {
    %and = arith.andi %a, %b : i32
    %or = arith.ori %a, %b : i32
    %shl = arith.shli %a, %b : i32
    %shrs = arith.shrsi %a, %b : i32
    %shru = arith.shrui %a, %b : i32
    %rems = arith.remsi %a, %b : i32
    %remu = arith.remui %a, %b : i32
    %max = arith.maxsi %a, %b : i32
    %min = arith.minsi %a, %b : i32
    %gt = arith.cmpi sgt, %a, %b : i32
    %sel = arith.select %gt, %and, %or : i32
    %tr = arith.trunci %a : i32 to i8
    %ex = arith.extsi %tr : i8 to i32

    %0 = arith.addi %and, %or : i32
    %1 = arith.addi %0, %shl : i32
    %2 = arith.addi %1, %shrs : i32
    %3 = arith.addi %2, %shru : i32
    %4 = arith.addi %3, %rems : i32
    %5 = arith.addi %4, %remu : i32
    %6 = arith.addi %5, %max : i32
    %7 = arith.addi %6, %min : i32
    %8 = arith.addi %7, %sel : i32
    %9 = arith.addi %8, %ex : i32
    return %9 : i32
  }
}
