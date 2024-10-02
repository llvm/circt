
module {
  smt.solver() : () -> () {
    %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.int) !smt.bool>
    %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.int) !smt.bool>
    %F_C = smt.declare_fun "F_C" : !smt.func<(!smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c0 = smt.int.constant 0
      %6 = smt.eq %arg1, %c0 : !smt.int
      %7 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %8 = smt.implies %6, %7
      smt.yield %8 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %6 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %7 = smt.int.add %arg1, %c1
      %8 = smt.apply_func %F_B(%7) : !smt.func<(!smt.int) !smt.bool>
      %true = smt.constant true
      %9 = smt.and %6, %true
      %10 = smt.implies %9, %8
      smt.yield %10 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %6 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %7 = smt.int.add %arg1, %c1
      %8 = smt.apply_func %F_C(%7) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %9 = smt.eq %arg0, %c1_0 : !smt.int
      %10 = smt.and %6, %9
      %11 = smt.implies %10, %8
      smt.yield %11 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %6 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %7 = smt.int.add %arg1, %c1
      %8 = smt.apply_func %F_B(%7) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %9 = smt.distinct %arg0, %c1_0 : !smt.int
      %10 = smt.and %6, %9
      %11 = smt.implies %10, %8
      smt.yield %11 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %6 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %7 = smt.int.add %arg1, %c1
      %8 = smt.apply_func %F_A(%7) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %9 = smt.eq %arg0, %c1_0 : !smt.int
      %10 = smt.and %6, %9
      %11 = smt.implies %10, %8
      smt.yield %11 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %6 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %7 = smt.int.add %arg1, %c1
      %8 = smt.apply_func %F_B(%7) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %9 = smt.distinct %arg0, %c1_0 : !smt.int
      %10 = smt.and %6, %9
      %11 = smt.implies %10, %8
      smt.yield %11 : !smt.bool
    }
    smt.assert %5
  }
}

