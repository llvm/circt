// RUN: circt-opt -pass-pipeline='builtin.module(calyx.component(materialize-calyx-to-fsm))' -split-input-file -verify-diagnostics %s
calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
  calyx.wires {
    calyx.assign %r.clk = %clk : i1
  }
// expected-error @+1 {{'calyx.control' op expected an 'fsm.machine' operation as the top-level operation within the control region of this component.}}
  calyx.control {
    calyx.seq {}
  }
}

// -----
calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
  calyx.wires {
  }
  calyx.control {
// expected-error @+1 {{'fsm.machine' op Expected an 'fsm_entry' and 'fsm_exit' state to be present in the FSM.}}
    fsm.machine @control() attributes {initialState = "IDLE"} {
      fsm.state @IDLE
    }
  }
}
