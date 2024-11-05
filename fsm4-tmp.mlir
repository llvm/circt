module {
  smt.solver() : () -> () {
    %F_SpecSCC_NUM_fsm_Init0_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Init0_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_Proceed_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Proceed_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Stall0_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Stall0_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Rollback_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Rollback_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill0_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill0_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill1_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill1_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill2_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill2_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init0_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Init0_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init1_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Init1_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init2_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Init2_st" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_Proceed_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Proceed_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Stall0_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Stall0_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Rollback_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Rollback_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill0_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill0_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill1_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill1_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill2_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill2_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init1_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Init1_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init2_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Init2_ar" : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
    %In_0 = smt.declare_fun "In_0" : !smt.func<(!smt.int) !smt.bool>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %c0 = smt.int.constant 0
      %false = smt.constant false
      %false_0 = smt.constant false
      %false_1 = smt.constant false
      %false_2 = smt.constant false
      %c0_3 = smt.int.constant 0
      %c0_4 = smt.int.constant 0
      %c0_5 = smt.int.constant 0
      %false_6 = smt.constant false
      %false_7 = smt.constant false
      %c0_8 = smt.int.constant 0
      %c0_9 = smt.int.constant 0
      %false_10 = smt.constant false
      %c0_11 = smt.int.constant 0
      %c0_12 = smt.int.constant 0
      %c0_13 = smt.int.constant 0
      %38 = smt.eq %arg15, %c0_13 : !smt.int
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%c0, %false, %false_0, %false_1, %false_2, %c0_3, %c0_4, %c0_5, %false_6, %false_7, %c0_8, %c0_9, %false_10, %c0_11, %c0_12, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.implies %38, %39
      smt.yield %40 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %arg9, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_0 = smt.constant true
      %41 = smt.and %38, %true_0
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %false = smt.constant false
      %false_0 = smt.constant false
      %true = smt.constant true
      %c5 = smt.int.constant 5
      %c1 = smt.int.constant 1
      %c1_1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1_1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %false, %false_0, %arg10, %arg11, %true, %c5, %c1, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %In_0(%arg15) : !smt.func<(!smt.int) !smt.bool>
      %true_2 = smt.constant true
      %42 = smt.eq %41, %true_2 : !smt.bool
      %43 = smt.and %38, %42
      %44 = smt.implies %43, %40
      smt.yield %44 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c5 = smt.int.constant 5
      %c4 = smt.int.constant 4
      %true = smt.constant true
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %c5, %c4, %arg13, %true, %arg9, %arg14, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_0 = smt.constant true
      %41 = smt.and %38, %true_0
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %5
    %6 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %6
    %7 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %7
    %8 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %8
    %9 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %9
    %10 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %10
    %11 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %true_0 = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %true_0, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_1 = smt.constant true
      %41 = smt.and %38, %true_1
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %11
    %12 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %41 = smt.and %38, %true
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %12
    %13 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %arg9, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_0 = smt.constant true
      %41 = smt.and %38, %true_0
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %13
    %14 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %arg9, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_0 = smt.constant true
      %41 = smt.and %38, %true_0
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %14
    %15 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %arg9, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_0 = smt.constant true
      %41 = smt.and %38, %true_0
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %15
    %16 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %true_0 = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %true_0, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_1 = smt.constant true
      %41 = smt.and %38, %true_1
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %16
    %17 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %arg9, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_0 = smt.constant true
      %41 = smt.and %38, %true_0
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %17
    %18 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true = smt.constant true
      %c0 = smt.int.constant 0
      %c1 = smt.int.constant 1
      %39 = smt.int.add %arg15, %c1
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %true, %arg9, %c0, %arg11, %arg12, %arg13, %arg14, %39) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %true_0 = smt.constant true
      %41 = smt.and %38, %true_0
      %42 = smt.implies %41, %40
      smt.yield %42 : !smt.bool
    }
    smt.assert %18
    %19 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %c1 = smt.int.constant 1
      %38 = smt.int.add %arg15, %c1
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %41 = smt.apply_func %In_0(%arg15) : !smt.func<(!smt.int) !smt.bool>
      %true = smt.constant true
      %42 = smt.eq %41, %true : !smt.bool
      %43 = smt.not %42
      %44 = smt.and %39, %43
      %45 = smt.implies %44, %40
      smt.yield %45 : !smt.bool
    }
    smt.assert %19
    %20 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %20
    %21 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %21
    %22 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %22
    %23 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %23
    %24 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %24
    %25 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %25
    %26 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %26
    %27 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %27
    %28 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %28
    %29 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %29
    %30 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %30
    %31 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %31
    %32 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %32
    %33 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %33
    %34 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %34
    %35 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %35
    %36 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %36
    %37 = smt.forall {
    ^bb0(%arg0: !smt.int, %arg1: !smt.bool, %arg2: !smt.bool, %arg3: !smt.bool, %arg4: !smt.bool, %arg5: !smt.int, %arg6: !smt.int, %arg7: !smt.int, %arg8: !smt.bool, %arg9: !smt.bool, %arg10: !smt.int, %arg11: !smt.int, %arg12: !smt.bool, %arg13: !smt.int, %arg14: !smt.int, %arg15: !smt.int):
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %40 = smt.not %39
      %41 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %42 = smt.not %41
      %43 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %44 = smt.not %43
      %45 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %46 = smt.not %45
      %47 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %48 = smt.not %47
      %49 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %50 = smt.not %49
      %51 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %52 = smt.not %51
      %53 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %54 = smt.not %53
      %55 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %56 = smt.not %55
      %57 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %58 = smt.not %57
      %59 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %60 = smt.not %59
      %61 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %62 = smt.not %61
      %63 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %64 = smt.not %63
      %65 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %66 = smt.not %65
      %67 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %68 = smt.not %67
      %69 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %70 = smt.not %69
      %71 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.int, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.int, !smt.bool, !smt.bool, !smt.int, !smt.int, !smt.bool, !smt.int, !smt.int, !smt.int) !smt.bool>
      %72 = smt.not %71
      %73 = smt.and %40, %42
      %74 = smt.and %73, %44
      %75 = smt.and %74, %46
      %76 = smt.and %75, %48
      %77 = smt.and %76, %50
      %78 = smt.and %77, %52
      %79 = smt.and %78, %54
      %80 = smt.and %79, %56
      %81 = smt.and %80, %58
      %82 = smt.and %81, %60
      %83 = smt.and %82, %62
      %84 = smt.and %83, %64
      %85 = smt.and %84, %66
      %86 = smt.and %85, %68
      %87 = smt.and %86, %70
      %88 = smt.and %87, %72
      %89 = smt.implies %38, %88
      smt.yield %89 : !smt.bool
    }
    smt.assert %37
  }
}

