module {
  smt.solver() : () -> () {
    %false = smt.constant false
    %false_0 = smt.constant false
    %false_1 = smt.constant false
    %F_CTR_IDLE = smt.declare_fun "F_CTR_IDLE" : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_CTR_INCR = smt.declare_fun "F_CTR_INCR" : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_CTR_ERROR = smt.declare_fun "F_CTR_ERROR" : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %false_2 = smt.constant false
      %false_3 = smt.constant false
      %false_4 = smt.constant false
      %c0 = smt.int.constant 0
      %c0_5 = smt.int.constant 0
      %c0_6 = smt.int.constant 0
      %16 = smt.eq %arg9, %c0_6 : !smt.int
      %17 = smt.apply_func %F_CTR_IDLE(%false_2, %false_3, %false_4, %c0, %c0_5, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %18 = smt.implies %16, %17
      smt.yield %18 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %16 = smt.apply_func %F_CTR_IDLE(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %17 = smt.int.add %arg9, %c1
      %18 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %17) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %19 = smt.and %16, %arg0
      %20 = smt.implies %19, %18
      smt.yield %20 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %16 = smt.apply_func %F_CTR_IDLE(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %17 = smt.int.add %arg9, %c1
      %18 = smt.apply_func %F_CTR_ERROR(%arg4, %arg5, %arg6, %arg7, %arg8, %17) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %19 = smt.or %arg1, %arg2
      %20 = smt.and %16, %19
      %21 = smt.implies %20, %18
      smt.yield %21 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %16 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %17 = smt.int.add %arg9, %c1
      %18 = smt.apply_func %F_CTR_IDLE(%arg4, %arg5, %arg6, %arg7, %arg8, %17) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c-1 = smt.int.constant -1
      %19 = smt.eq %c-1, %arg7 : !smt.int
      %20 = smt.and %16, %19
      %21 = smt.implies %20, %18
      smt.yield %21 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %16 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %17 = smt.int.add %arg9, %c1
      %18 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %17) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c-1 = smt.int.constant -1
      %19 = smt.distinct %c-1, %arg7 : !smt.int
      %20 = smt.or %arg1, %arg2
      %true = smt.constant true
      %21 = smt.xor %20, %true
      %22 = smt.and %19, %21
      %23 = smt.and %16, %22
      %24 = smt.implies %23, %18
      smt.yield %24 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %16 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %17 = smt.int.add %arg9, %c1
      %18 = smt.apply_func %F_CTR_ERROR(%arg4, %arg5, %arg6, %arg7, %arg8, %17) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %19 = smt.or %arg1, %arg2
      %20 = smt.and %16, %19
      %21 = smt.implies %20, %18
      smt.yield %21 : !smt.bool
    }
    smt.assert %5
    %6 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %17 = smt.apply_func %F_CTR_IDLE(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %18 = smt.apply_func %F_CTR_IDLE(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %19 = smt.not %arg0
      %20 = smt.and %17, %19
      %21 = smt.implies %20, %18
      smt.yield %21 : !smt.bool
    }
    smt.assert %6
    %7 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %17 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %18 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c-1 = smt.int.constant -1
      %19 = smt.eq %c-1, %arg7 : !smt.int
      %20 = smt.not %19
      %21 = smt.and %17, %20
      %22 = smt.implies %21, %18
      smt.yield %22 : !smt.bool
    }
    smt.assert %7
    %8 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %c-1 = smt.int.constant -1
      %17 = smt.distinct %c-1, %arg7 : !smt.int
      %18 = smt.or %arg1, %arg2
      %true = smt.constant true
      %19 = smt.xor %18, %true
      %20 = smt.and %17, %19
      %21 = smt.not %20
      %c-1_2 = smt.int.constant -1
      %22 = smt.distinct %c-1_2, %arg7 : !smt.int
      %23 = smt.or %arg1, %arg2
      %true_3 = smt.constant true
      %24 = smt.xor %23, %true_3
      %25 = smt.and %22, %24
      %26 = smt.not %25
      %27 = smt.and %26, %21
      %28 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %29 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %16) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %30 = smt.and %28, %27
      %31 = smt.implies %30, %29
      smt.yield %31 : !smt.bool
    }
    smt.assert %8
    %9 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %17 = smt.or %arg1, %arg2
      %18 = smt.not %17
      %19 = smt.or %arg1, %arg2
      %20 = smt.not %19
      %21 = smt.and %20, %18
      %22 = smt.or %arg1, %arg2
      %23 = smt.not %22
      %24 = smt.and %21, %23
      %25 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %26 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %16) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %27 = smt.and %25, %24
      %28 = smt.implies %27, %26
      smt.yield %28 : !smt.bool
    }
    smt.assert %9
    %10 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %17 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %18 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c-1 = smt.int.constant -1
      %19 = smt.distinct %c-1, %arg7 : !smt.int
      %20 = smt.or %arg1, %arg2
      %true = smt.constant true
      %21 = smt.xor %20, %true
      %22 = smt.and %19, %21
      %23 = smt.not %22
      %24 = smt.and %17, %23
      %25 = smt.implies %24, %18
      smt.yield %25 : !smt.bool
    }
    smt.assert %10
    %11 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %c-1 = smt.int.constant -1
      %17 = smt.eq %c-1, %arg7 : !smt.int
      %18 = smt.not %17
      %c-1_2 = smt.int.constant -1
      %19 = smt.eq %c-1_2, %arg7 : !smt.int
      %20 = smt.not %19
      %21 = smt.and %20, %18
      %22 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %23 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %16) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %24 = smt.and %22, %21
      %25 = smt.implies %24, %23
      smt.yield %25 : !smt.bool
    }
    smt.assert %11
    %12 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %17 = smt.or %arg1, %arg2
      %18 = smt.not %17
      %19 = smt.or %arg1, %arg2
      %20 = smt.not %19
      %21 = smt.and %20, %18
      %22 = smt.or %arg1, %arg2
      %23 = smt.not %22
      %24 = smt.and %21, %23
      %25 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %26 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %16) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %27 = smt.and %25, %24
      %28 = smt.implies %27, %26
      smt.yield %28 : !smt.bool
    }
    smt.assert %12
    %13 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %17 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %18 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %19 = smt.or %arg1, %arg2
      %20 = smt.not %19
      %21 = smt.and %17, %20
      %22 = smt.implies %21, %18
      smt.yield %22 : !smt.bool
    }
    smt.assert %13
    %14 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %c-1 = smt.int.constant -1
      %17 = smt.eq %c-1, %arg7 : !smt.int
      %18 = smt.not %17
      %c-1_2 = smt.int.constant -1
      %19 = smt.eq %c-1_2, %arg7 : !smt.int
      %20 = smt.not %19
      %21 = smt.and %20, %18
      %22 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %23 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %16) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %24 = smt.and %22, %21
      %25 = smt.implies %24, %23
      smt.yield %25 : !smt.bool
    }
    smt.assert %14
    %15 = smt.forall {
    ^bb0(%arg0: !smt.bool, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.int, %arg4: !smt.bool, %arg5: !smt.bool, %arg6: !smt.bool, %arg7: !smt.int, %arg8: !smt.int, %arg9: !smt.int):
      %c1 = smt.int.constant 1
      %16 = smt.int.add %arg9, %c1
      %c-1 = smt.int.constant -1
      %17 = smt.distinct %c-1, %arg7 : !smt.int
      %18 = smt.or %arg1, %arg2
      %true = smt.constant true
      %19 = smt.xor %18, %true
      %20 = smt.and %17, %19
      %21 = smt.not %20
      %c-1_2 = smt.int.constant -1
      %22 = smt.distinct %c-1_2, %arg7 : !smt.int
      %23 = smt.or %arg1, %arg2
      %true_3 = smt.constant true
      %24 = smt.xor %23, %true_3
      %25 = smt.and %22, %24
      %26 = smt.not %25
      %27 = smt.and %26, %21
      %c-1_4 = smt.int.constant -1
      %28 = smt.distinct %c-1_4, %arg7 : !smt.int
      %29 = smt.or %arg1, %arg2
      %true_5 = smt.constant true
      %30 = smt.xor %29, %true_5
      %31 = smt.and %28, %30
      %32 = smt.not %31
      %33 = smt.and %27, %32
      %34 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %35 = smt.apply_func %F_CTR_INCR(%arg4, %arg5, %arg6, %arg7, %arg8, %16) : !smt.func<(!smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %36 = smt.and %34, %33
      %37 = smt.implies %36, %35
      smt.yield %37 : !smt.bool
    }
    smt.assert %15
  }
}

