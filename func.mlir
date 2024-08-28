func.func @entry() {
  smt.solver () : () -> () {
    %t01 = smt.declare_fun "t01" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t12 = smt.declare_fun "t12" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t21 = smt.declare_fun "t21" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t22 = smt.declare_fun "t22" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %4 = smt.apply_func %t12(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %5 = smt.apply_func %t21(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c5 = smt.int.constant 5
      %6 = smt.eq %arg0, %c5 : !smt.int
      %c1 = smt.int.constant 1
      %7 = smt.eq %arg0, %c1 : !smt.int
      %8 = smt.and %7, %6
      %9 = smt.and %4, %8
      %10 = smt.implies %9, %5
      smt.yield %10 : !smt.bool
    }
    %1 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %4 = smt.apply_func %t12(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %5 = smt.apply_func %t22(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c0 = smt.int.constant 0
      %6 = smt.eq %arg0, %c0 : !smt.int
      %c1 = smt.int.constant 1
      %7 = smt.eq %arg0, %c1 : !smt.int
      %8 = smt.and %7, %6
      %9 = smt.and %4, %8
      %10 = smt.implies %9, %5
      smt.yield %10 : !smt.bool
    }
    %2 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %4 = smt.apply_func %t21(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %5 = smt.apply_func %t12(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %6 = smt.eq %arg0, %c1 : !smt.int
      %c5 = smt.int.constant 5
      %7 = smt.eq %arg0, %c5 : !smt.int
      %8 = smt.and %7, %6
      %9 = smt.and %4, %8
      %10 = smt.implies %9, %5
      smt.yield %10 : !smt.bool
    }
    %3 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %4 = smt.apply_func %t22(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %5 = smt.apply_func %t21(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c5 = smt.int.constant 5
      %6 = smt.eq %arg0, %c5 : !smt.int
      %c0 = smt.int.constant 0
      %7 = smt.eq %arg0, %c0 : !smt.int
      %8 = smt.and %7, %6
      %9 = smt.and %4, %8
      %10 = smt.implies %9, %5
      smt.yield %10 : !smt.bool
    }
  }
  return
}