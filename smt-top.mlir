module {
  smt.solver() : () -> () {
    %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.int) !smt.bool>
    %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.int) !smt.bool>
    %F_C = smt.declare_fun "F_C" : !smt.func<(!smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c0 = smt.int.constant 0
      %11 = smt.eq %arg1, %c0 : !smt.int
      %12 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %13 = smt.implies %11, %12
      smt.yield %13 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %11 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %12 = smt.int.add %arg1, %c1
      %13 = smt.apply_func %F_B(%12) : !smt.func<(!smt.int) !smt.bool>
      %true = smt.constant true
      %14 = smt.and %11, %true
      %15 = smt.implies %14, %13
      smt.yield %15 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %11 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %12 = smt.int.add %arg1, %c1
      %13 = smt.apply_func %F_C(%12) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %14 = smt.eq %arg0, %c1_0 : !smt.int
      %15 = smt.and %11, %14
      %16 = smt.implies %15, %13
      smt.yield %16 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %11 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %12 = smt.int.add %arg1, %c1
      %13 = smt.apply_func %F_B(%12) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %14 = smt.distinct %arg0, %c1_0 : !smt.int
      %15 = smt.and %11, %14
      %16 = smt.implies %15, %13
      smt.yield %16 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %11 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %12 = smt.int.add %arg1, %c1
      %13 = smt.apply_func %F_A(%12) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %14 = smt.eq %arg0, %c1_0 : !smt.int
      %15 = smt.and %11, %14
      %16 = smt.implies %15, %13
      smt.yield %16 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %11 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %12 = smt.int.add %arg1, %c1
      %13 = smt.apply_func %F_B(%12) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %14 = smt.distinct %arg0, %c1_0 : !smt.int
      %15 = smt.and %11, %14
      %16 = smt.implies %15, %13
      smt.yield %16 : !smt.bool
    }
    smt.assert %5
    %6 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c1 = smt.int.constant 1
      %11 = smt.int.add %arg1, %c1
      %12 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %13 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %14 = smt.eq %arg0, %c1_0 : !smt.int
      %15 = smt.not %14
      %16 = smt.and %12, %15
      %17 = smt.implies %16, %13
      smt.yield %17 : !smt.bool
    }
    smt.assert %6
    %7 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c1 = smt.int.constant 1
      %11 = smt.int.add %arg1, %c1
      %12 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %13 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %14 = smt.eq %arg0, %c1_0 : !smt.int
      %15 = smt.not %14
      %16 = smt.and %12, %15
      %17 = smt.implies %16, %13
      smt.yield %17 : !smt.bool
    }
    smt.assert %7
    %8 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c1 = smt.int.constant 1
      %11 = smt.int.add %arg1, %c1
      %c1_0 = smt.int.constant 1
      %12 = smt.distinct %arg0, %c1_0 : !smt.int
      %13 = smt.not %12
      %c1_1 = smt.int.constant 1
      %14 = smt.distinct %arg0, %c1_1 : !smt.int
      %15 = smt.not %14
      %16 = smt.and %15, %13
      %17 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %18 = smt.apply_func %F_C(%11) : !smt.func<(!smt.int) !smt.bool>
      %19 = smt.and %17, %16
      %20 = smt.implies %19, %18
      smt.yield %20 : !smt.bool
    }
    smt.assert %8
    %9 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c1 = smt.int.constant 1
      %11 = smt.int.add %arg1, %c1
      %12 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %13 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %14 = smt.distinct %arg0, %c1_0 : !smt.int
      %15 = smt.not %14
      %16 = smt.and %12, %15
      %17 = smt.implies %16, %13
      smt.yield %17 : !smt.bool
    }
    smt.assert %9
    %10 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c1 = smt.int.constant 1
      %11 = smt.int.add %arg1, %c1
      %c1_0 = smt.int.constant 1
      %12 = smt.eq %arg0, %c1_0 : !smt.int
      %13 = smt.not %12
      %c1_1 = smt.int.constant 1
      %14 = smt.eq %arg0, %c1_1 : !smt.int
      %15 = smt.not %14
      %16 = smt.and %15, %13
      %17 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %18 = smt.apply_func %F_C(%11) : !smt.func<(!smt.int) !smt.bool>
      %19 = smt.and %17, %16
      %20 = smt.implies %19, %18
      smt.yield %20 : !smt.bool
    }
    smt.assert %10
  }
}

