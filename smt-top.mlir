module {
  smt.solver() : () -> () {
    %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.int) !smt.bool>
    %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.int) !smt.bool>
    %F_C = smt.declare_fun "F_C" : !smt.func<(!smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c0 = smt.int.constant 0
      %9 = smt.eq %arg1, %c0 : !smt.int
      %10 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %11 = smt.implies %9, %10
      smt.yield %11 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %9 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %10 = smt.int.add %arg1, %c1
      %11 = smt.apply_func %F_B(%10) : !smt.func<(!smt.int) !smt.bool>
      %true = smt.constant true
      %12 = smt.and %9, %true
      %13 = smt.implies %12, %11
      smt.yield %13 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %9 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %10 = smt.int.add %arg1, %c1
      %11 = smt.apply_func %F_C(%10) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %12 = smt.eq %arg0, %c1_0 : !smt.int
      %13 = smt.and %9, %12
      %14 = smt.implies %13, %11
      smt.yield %14 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %9 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %10 = smt.int.add %arg1, %c1
      %11 = smt.apply_func %F_B(%10) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %12 = smt.distinct %arg0, %c1_0 : !smt.int
      %13 = smt.and %9, %12
      %14 = smt.implies %13, %11
      smt.yield %14 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %9 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %10 = smt.int.add %arg1, %c1
      %11 = smt.apply_func %F_A(%10) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %12 = smt.eq %arg0, %c1_0 : !smt.int
      %13 = smt.and %9, %12
      %14 = smt.implies %13, %11
      smt.yield %14 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %9 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %10 = smt.int.add %arg1, %c1
      %11 = smt.apply_func %F_B(%10) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %12 = smt.distinct %arg0, %c1_0 : !smt.int
      %13 = smt.and %9, %12
      %14 = smt.implies %13, %11
      smt.yield %14 : !smt.bool
    }
    smt.assert %5
    %6 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %true = smt.constant true
      %9 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %10 = smt.not %9
      %11 = smt.and %true, %10
      %12 = smt.apply_func %F_C(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %13 = smt.not %12
      %14 = smt.and %11, %13
      %15 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %16 = smt.implies %15, %14
      smt.yield %16 : !smt.bool
    }
    smt.assert %6
    %7 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %true = smt.constant true
      %9 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %10 = smt.not %9
      %11 = smt.and %true, %10
      %12 = smt.apply_func %F_C(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %13 = smt.not %12
      %14 = smt.and %11, %13
      %15 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %16 = smt.implies %15, %14
      smt.yield %16 : !smt.bool
    }
    smt.assert %7
    %8 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %true = smt.constant true
      %9 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %10 = smt.not %9
      %11 = smt.and %true, %10
      %12 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %13 = smt.not %12
      %14 = smt.and %11, %13
      %15 = smt.apply_func %F_C(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %16 = smt.implies %15, %14
      smt.yield %16 : !smt.bool
    }
    smt.assert %8
  }
}

