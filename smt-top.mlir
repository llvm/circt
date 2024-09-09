module {
  smt.solver() : () -> () {
    %at_0 = smt.declare_fun "at_0" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %notAt_0 = smt.declare_fun "notAt_0" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %17 = smt.exists {
      ^bb0(%arg1: !smt.int):
        %c0 = smt.int.constant 0
        %18 = smt.int.cmp gt %arg1, %c0
        %c13 = smt.int.constant 13
        %19 = smt.int.cmp lt %arg1, %c13
        %20 = smt.apply_func %notAt_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
        %21 = smt.not %20
        %22 = smt.and %18, %19, %21
        smt.yield %22 : !smt.bool
      }
      smt.yield %17 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %17 = smt.apply_func %at_0(%arg2, %arg0) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %18 = smt.not %17
      %19 = smt.apply_func %at_0(%arg1, %arg0) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %20 = smt.distinct %arg1, %arg2 : !smt.int
      %21 = smt.and %19, %20
      %22 = smt.implies %21, %18
      smt.yield %22 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %notAt_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %18 = smt.apply_func %at_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %19 = smt.not %18
      %20 = smt.implies %17, %19
      smt.yield %20 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %notAt_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %18 = smt.not %17
      %19 = smt.apply_func %at_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %20 = smt.implies %18, %19
      smt.yield %20 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %at_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %18 = smt.apply_func %notAt_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %19 = smt.not %18
      %20 = smt.implies %17, %19
      smt.yield %20 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %at_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %18 = smt.not %17
      %19 = smt.apply_func %notAt_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %20 = smt.implies %18, %19
      smt.yield %20 : !smt.bool
    }
    smt.assert %5
    %6 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c13 = smt.int.constant 13
      %17 = smt.int.cmp gt %arg1, %c13
      %18 = smt.apply_func %notAt_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %19 = smt.implies %17, %18
      smt.yield %19 : !smt.bool
    }
    smt.assert %6
    %7 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c0 = smt.int.constant 0
      %17 = smt.int.cmp lt %arg1, %c0
      %18 = smt.apply_func %notAt_0(%arg0, %arg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %19 = smt.implies %17, %18
      smt.yield %19 : !smt.bool
    }
    smt.assert %6
    %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.int) !smt.bool>
    %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.int) !smt.bool>
    %F_C = smt.declare_fun "F_C" : !smt.func<(!smt.int) !smt.bool>
    %8 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %c0 = smt.int.constant 0
      %17 = smt.eq %arg1, %c0 : !smt.int
      %18 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %19 = smt.implies %17, %18
      smt.yield %19 : !smt.bool
    }
    smt.assert %8
    %9 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %F_A(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %18 = smt.int.add %arg1, %c1
      %19 = smt.apply_func %F_B(%18) : !smt.func<(!smt.int) !smt.bool>
      %true = smt.constant true
      %20 = smt.and %17, %true
      %21 = smt.implies %20, %19
      smt.yield %21 : !smt.bool
    }
    smt.assert %9
    %10 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %18 = smt.int.add %arg1, %c1
      %19 = smt.apply_func %F_C(%18) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %20 = smt.eq %arg0, %c1_0 : !smt.int
      %21 = smt.and %17, %20
      %22 = smt.implies %21, %19
      smt.yield %22 : !smt.bool
    }
    smt.assert %10
    %11 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %F_B(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %18 = smt.int.add %arg1, %c1
      %19 = smt.apply_func %F_B(%18) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %20 = smt.distinct %arg0, %c1_0 : !smt.int
      %21 = smt.and %17, %20
      %22 = smt.implies %21, %19
      smt.yield %22 : !smt.bool
    }
    smt.assert %11
    %12 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %18 = smt.int.add %arg1, %c1
      %19 = smt.apply_func %F_A(%18) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %20 = smt.eq %arg0, %c1_0 : !smt.int
      %21 = smt.and %17, %20
      %22 = smt.implies %21, %19
      smt.yield %22 : !smt.bool
    }
    smt.assert %12
    %13 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int):
      %17 = smt.apply_func %F_C(%arg1) : !smt.func<(!smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %18 = smt.int.add %arg1, %c1
      %19 = smt.apply_func %F_B(%18) : !smt.func<(!smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %20 = smt.distinct %arg0, %c1_0 : !smt.int
      %21 = smt.and %17, %20
      %22 = smt.implies %21, %19
      smt.yield %22 : !smt.bool
    }
    smt.assert %13
    %14 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %true = smt.constant true
      %17 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %18 = smt.not %17
      %19 = smt.and %true, %18
      %20 = smt.apply_func %F_C(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %21 = smt.not %20
      %22 = smt.and %19, %21
      %23 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %24 = smt.implies %23, %22
      smt.yield %24 : !smt.bool
    }
    smt.assert %14
    %15 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %true = smt.constant true
      %17 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %18 = smt.not %17
      %19 = smt.and %true, %18
      %20 = smt.apply_func %F_C(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %21 = smt.not %20
      %22 = smt.and %19, %21
      %23 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %24 = smt.implies %23, %22
      smt.yield %24 : !smt.bool
    }
    smt.assert %15
    %16 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %true = smt.constant true
      %17 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %18 = smt.not %17
      %19 = smt.and %true, %18
      %20 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %21 = smt.not %20
      %22 = smt.and %19, %21
      %23 = smt.apply_func %F_C(%arg0) : !smt.func<(!smt.int) !smt.bool>
      %24 = smt.implies %23, %22
      smt.yield %24 : !smt.bool
    }
    smt.assert %16
  }
}

