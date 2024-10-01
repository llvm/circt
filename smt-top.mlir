module {
  smt.solver() : () -> () {
    %t01 = smt.declare_fun "t01" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t12 = smt.declare_fun "t12" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t23 = smt.declare_fun "t23" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t22 = smt.declare_fun "t22" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t31 = smt.declare_fun "t31" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %t32 = smt.declare_fun "t32" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %c-1 = smt.int.constant -1
      %40 = smt.eq %arg2, %c-1 : !smt.int
      %41 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.implies %40, %41
      smt.yield %42 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t12(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %43 = smt.and %40, %true
      %true_0 = smt.constant true
      %44 = smt.and %43, %true_0
      %45 = smt.implies %44, %42
      smt.yield %45 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t23(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %43 = smt.and %40, %true
      %c1_0 = smt.int.constant 1
      %44 = smt.eq %arg1, %c1_0 : !smt.int
      %45 = smt.and %43, %44
      %46 = smt.implies %45, %42
      smt.yield %46 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t22(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %43 = smt.and %40, %true
      %c1_0 = smt.int.constant 1
      %44 = smt.distinct %arg1, %c1_0 : !smt.int
      %45 = smt.and %43, %44
      %46 = smt.implies %45, %42
      smt.yield %46 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t31(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %43 = smt.eq %arg0, %c1_0 : !smt.int
      %44 = smt.and %40, %43
      %c1_1 = smt.int.constant 1
      %45 = smt.eq %arg1, %c1_1 : !smt.int
      %46 = smt.and %44, %45
      %47 = smt.implies %46, %42
      smt.yield %47 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t32(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %43 = smt.eq %arg0, %c1_0 : !smt.int
      %44 = smt.and %40, %43
      %c1_1 = smt.int.constant 1
      %45 = smt.distinct %arg1, %c1_1 : !smt.int
      %46 = smt.and %44, %45
      %47 = smt.implies %46, %42
      smt.yield %47 : !smt.bool
    }
    smt.assert %5
    %6 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t23(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %43 = smt.distinct %arg0, %c1_0 : !smt.int
      %44 = smt.and %40, %43
      %c1_1 = smt.int.constant 1
      %45 = smt.eq %arg1, %c1_1 : !smt.int
      %46 = smt.and %44, %45
      %47 = smt.implies %46, %42
      smt.yield %47 : !smt.bool
    }
    smt.assert %6
    %7 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t12(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %43 = smt.eq %arg0, %c1_0 : !smt.int
      %44 = smt.and %40, %43
      %true = smt.constant true
      %45 = smt.and %44, %true
      %46 = smt.implies %45, %42
      smt.yield %46 : !smt.bool
    }
    smt.assert %7
    %8 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t23(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %43 = smt.distinct %arg0, %c1_0 : !smt.int
      %44 = smt.and %40, %43
      %c1_1 = smt.int.constant 1
      %45 = smt.eq %arg1, %c1_1 : !smt.int
      %46 = smt.and %44, %45
      %47 = smt.implies %46, %42
      smt.yield %47 : !smt.bool
    }
    smt.assert %8
    %9 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %41 = smt.int.add %arg2, %c1
      %42 = smt.apply_func %t22(%arg1, %41) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %c1_0 = smt.int.constant 1
      %43 = smt.distinct %arg0, %c1_0 : !smt.int
      %44 = smt.and %40, %43
      %c1_1 = smt.int.constant 1
      %45 = smt.distinct %arg1, %c1_1 : !smt.int
      %46 = smt.and %44, %45
      %47 = smt.implies %46, %42
      smt.yield %47 : !smt.bool
    }
    smt.assert %9
    %10 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %10
    %11 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %11
    %12 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %12
    %13 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %13
    %14 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %14
    %15 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %15
    %16 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %16
    %17 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %17
    %18 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %18
    %19 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %19
    %20 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %20
    %21 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %21
    %22 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %22
    %23 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %23
    %24 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %24
    %25 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %25
    %26 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %26
    %27 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %27
    %28 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %28
    %29 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %29
    %30 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %30
    %31 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %31
    %32 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %32
    %33 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %33
    %34 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %34
    %35 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t01(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %35
    %36 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t12(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %36
    %37 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t23(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %37
    %38 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t22(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %38
    %39 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
      %40 = smt.apply_func %t32(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %t31(%arg0, %arg2) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.implies %40, %42
      smt.yield %43 : !smt.bool
    }
    smt.assert %39
  }
}