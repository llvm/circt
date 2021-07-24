// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-remove-groups))' %s | FileCheck %s

calyx.program {
  calyx.component @Z(%go : i1, %reset : i1, %clk : i1) -> (%flag: i1, %out :i2, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @main(%go : i1, %reset : i1, %clk : i1) -> (%done: i1) {
    %z.go, %z.reset, %z.clk, %z.flag, %z.out, %z.done = calyx.cell "z" @Z : i1, i1, i1, i1, i2, i1
    %fsm.in, %fsm.write_en, %fsm.clk, %fsm.reset, %fsm.out, %fsm.done = calyx.register "fsm" : i2

    // CHECK-LABEL: calyx.wires
    calyx.wires {
      %undef = calyx.undef : i1

      %signal_on = hw.constant true
      %group_A_fsm_begin = hw.constant 0 : i2
      %fsm_is_group_A_begin_state = comb.icmp eq %fsm.out, %group_A_fsm_begin : i2
      %group_A_not_done = comb.xor %z.done, %signal_on : i1
      %group_A_go_guard = comb.and %fsm_is_group_A_begin_state, %group_A_not_done : i1

      calyx.group @A {
        %A.go = calyx.group_go %signal_on, %group_A_go_guard ? : i1
        calyx.assign %z.go = %z.flag : i1
        calyx.group_done %z.done : i1
      }

      %group_B_fsm_begin = hw.constant 1 : i2
      %group_B_done = comb.and %z.flag, %z.done : i1
      %fsm_is_group_B_begin_state = comb.icmp eq %fsm.out, %group_B_fsm_begin : i2
      %group_B_not_done = comb.xor %group_B_done, %signal_on : i1
      %group_B_go_guard = comb.and %fsm_is_group_B_begin_state, %group_B_not_done : i1
      calyx.group @B {
        %B.go = calyx.group_go %signal_on, %group_B_go_guard ? : i1
        calyx.group_done %z.done, %z.flag ? : i1
      }

      %group_A_assign_guard = comb.and %fsm_is_group_A_begin_state, %z.done : i1
      %fsm_step_1 = hw.constant 1 : i2
      %group_B_assign_guard = comb.and %fsm_is_group_B_begin_state, %group_B_done : i1
      %fsm_step_2 = hw.constant -2 : i2
      %seq_group_done_guard = comb.icmp eq %fsm.out, %fsm_step_2 : i2
      calyx.group @seq {
        calyx.assign %fsm.in = %fsm_step_1, %group_A_assign_guard ? : i2
        calyx.assign %fsm.write_en = %signal_on, %group_A_assign_guard ? : i1
        calyx.assign %fsm.in = %fsm_step_2, %group_B_assign_guard ? : i2
        calyx.assign %fsm.write_en = %signal_on, %group_B_assign_guard ? : i1
        calyx.group_done %signal_on, %seq_group_done_guard ? : i1
      }
      %fsm_step_0 = hw.constant 0 : i2
      calyx.assign %fsm.in = %fsm_step_0, %seq_group_done_guard ? : i2
      calyx.assign %fsm.write_en = %signal_on, %seq_group_done_guard ? : i1
    }

    // CHECK-LABEL: calyx.control
    calyx.control {
      calyx.enable @seq {compiledGroups = [@A, @B]}
    }
  }
}
