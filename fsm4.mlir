module {
  smt.solver() : () -> () {
    %F_SpecSCC_NUM_fsm_Init0_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Init0_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_Proceed_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Proceed_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Stall0_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Stall0_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Rollback_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Rollback_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill0_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill0_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill1_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill1_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill2_st = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill2_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init0_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Init0_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init1_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Init1_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init2_st = smt.declare_fun "F_SpecSCC_NUM_fsm_Init2_st" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_Proceed_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Proceed_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Stall0_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Stall0_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Rollback_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Rollback_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill0_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill0_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill1_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill1_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_x1__Fill2_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_x1__Fill2_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init1_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Init1_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %F_SpecSCC_NUM_fsm_Init2_ar = smt.declare_fun "F_SpecSCC_NUM_fsm_Init2_ar" : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
    %In_0 = smt.declare_fun "In_0" : !smt.func<(!smt.bv<32>) !smt.bv<1>>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %c0_bv3 = smt.bv.constant #smt.bv<0> : !smt.bv<3>
      %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv1_0 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv1_2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv32_3 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv32_4 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv1_5 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv1_6 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv32_7 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv32_8 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv1_9 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv32_10 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv32_11 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c0_bv32_12 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %37 = smt.eq %arg15, %c0_bv32_12 : !smt.bv<32>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%c0_bv3, %c0_bv1, %c0_bv1_0, %c0_bv1_1, %c0_bv1_2, %c0_bv32, %c0_bv32_3, %c0_bv32_4, %c0_bv1_5, %c0_bv1_6, %c0_bv32_7, %c0_bv32_8, %c0_bv1_9, %c0_bv32_10, %c0_bv32_11, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.implies %37, %38
      smt.yield %39 : !smt.bool
    }
    smt.assert %0
    %1 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %arg9, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %1
    %2 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c0_bv1_0 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c5_bv32 = smt.bv.constant #smt.bv<5> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %c1_bv32_1 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32_1 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c0_bv1, %c0_bv1_0, %arg10, %arg11, %c-1_bv1, %c5_bv32, %c1_bv32, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %40 = smt.apply_func %In_0(%arg15) : !smt.func<(!smt.bv<32>) !smt.bv<1>>
      %c-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %41 = smt.eq %40, %c-1_bv1_2 : !smt.bv<1>
      %42 = smt.and %37, %41
      %43 = smt.implies %42, %39
      smt.yield %43 : !smt.bool
    }
    smt.assert %2
    %3 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c5_bv32 = smt.bv.constant #smt.bv<5> : !smt.bv<32>
      %c4_bv32 = smt.bv.constant #smt.bv<4> : !smt.bv<32>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %c5_bv32, %c4_bv32, %arg13, %c-1_bv1, %arg9, %arg14, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %3
    %4 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %4
    %5 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %5
    %6 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %6
    %7 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %7
    %8 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %8
    %9 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %9
    %10 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %10
    %11 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %c-1_bv1_0, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %11
    %12 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %12
    %13 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %arg9, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %13
    %14 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %arg9, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %14
    %15 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %arg9, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %15
    %16 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %c-1_bv1_0, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %16
    %17 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %arg9, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %17
    %18 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %38 = smt.bv.add %arg15, %c1_bv32 : !smt.bv<32>
      %39 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %c-1_bv1, %arg9, %c0_bv32, %arg11, %arg12, %arg13, %arg14, %38) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %true = smt.constant true
      %40 = smt.and %37, %true
      %41 = smt.implies %40, %39
      smt.yield %41 : !smt.bool
    }
    smt.assert %18
    %19 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %19
    %20 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %20
    %21 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %21
    %22 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %22
    %23 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %23
    %24 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %24
    %25 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %25
    %26 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %26
    %27 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %27
    %28 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %28
    %29 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %29
    %30 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %30
    %31 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %31
    %32 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %32
    %33 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %33
    %34 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %34
    %35 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %35
    %36 = smt.forall {
    ^bb0(%arg0: !smt.bv<3>, %arg1: !smt.bv<1>, %arg2: !smt.bv<1>, %arg3: !smt.bv<1>, %arg4: !smt.bv<1>, %arg5: !smt.bv<32>, %arg6: !smt.bv<32>, %arg7: !smt.bv<32>, %arg8: !smt.bv<1>, %arg9: !smt.bv<1>, %arg10: !smt.bv<32>, %arg11: !smt.bv<32>, %arg12: !smt.bv<1>, %arg13: !smt.bv<32>, %arg14: !smt.bv<32>, %arg15: !smt.bv<32>):
      %37 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %38 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %39 = smt.not %38
      %40 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %41 = smt.not %40
      %42 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %43 = smt.not %42
      %44 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %45 = smt.not %44
      %46 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %47 = smt.not %46
      %48 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %49 = smt.not %48
      %50 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %51 = smt.not %50
      %52 = smt.apply_func %F_SpecSCC_NUM_fsm_Init0_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %53 = smt.not %52
      %54 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %55 = smt.not %54
      %56 = smt.apply_func %F_SpecSCC_NUM_fsm_Init2_st(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %57 = smt.not %56
      %58 = smt.apply_func %F_SpecSCC_NUM_fsm_Proceed_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %59 = smt.not %58
      %60 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Stall0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %61 = smt.not %60
      %62 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Rollback_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %63 = smt.not %62
      %64 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill0_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %65 = smt.not %64
      %66 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %67 = smt.not %66
      %68 = smt.apply_func %F_SpecSCC_NUM_fsm_x1__Fill2_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %69 = smt.not %68
      %70 = smt.apply_func %F_SpecSCC_NUM_fsm_Init1_ar(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !smt.func<(!smt.bv<3>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<1>, !smt.bv<32>, !smt.bv<32>, !smt.bv<32>) !smt.bool>
      %71 = smt.not %70
      %72 = smt.and %39, %41
      %73 = smt.and %72, %43
      %74 = smt.and %73, %45
      %75 = smt.and %74, %47
      %76 = smt.and %75, %49
      %77 = smt.and %76, %51
      %78 = smt.and %77, %53
      %79 = smt.and %78, %55
      %80 = smt.and %79, %57
      %81 = smt.and %80, %59
      %82 = smt.and %81, %61
      %83 = smt.and %82, %63
      %84 = smt.and %83, %65
      %85 = smt.and %84, %67
      %86 = smt.and %85, %69
      %87 = smt.and %86, %71
      %88 = smt.implies %37, %87
      smt.yield %88 : !smt.bool
    }
    smt.assert %36
  }
}

