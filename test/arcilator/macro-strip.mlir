// RUN: arcilator %s

// CHECK: %[[VAL:.+]] = hw.constant 1 : i1
// CHECK: comb.and %[[VAL]]

module {
  sv.macro.decl @STOP_COND_
  hw.module @SimTop(in %cond : i1, out out : i1) {
    %STOP_COND_ = sv.macro.ref.expr @STOP_COND_() : () -> i1
    %0 = comb.and %STOP_COND_, %cond : i1
    hw.output %0 : i1
  }
}