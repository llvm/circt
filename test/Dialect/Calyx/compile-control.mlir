// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(calyx-compile-control))' %s | FileCheck %s

calyx.program "main" {
  calyx.component @Z(%go : i1 {go}, %reset : i1 {reset}, %clk : i1 {clk}) -> (%flag :i1, %done : i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  calyx.component @main(%go : i1 {go}, %reset : i1 {reset}, %clk : i1 {clk}) -> (%done : i1 {done}) {
    // CHECK:  %fsm.in, %fsm.write_en, %fsm.clk, %fsm.reset, %fsm.out, %fsm.done = calyx.register @fsm : i2
    %z.go, %z.reset, %z.clk, %z.flag, %z.done = calyx.instance @z @Z : i1, i1, i1, i1, i1
    calyx.wires {
      %undef = calyx.undef : i1
      // CHECK: %[[SIGNAL_ON:.+]] = hw.constant true
      // CHECK: %[[GROUP_A_FSM_BEGIN:.+]] = hw.constant 0 : i2
      // CHECK: %[[FSM_IS_GROUP_A_BEGIN_STATE:.+]] = comb.icmp eq %fsm.out, %[[GROUP_A_FSM_BEGIN]] : i2
      // CHECK: %[[GROUP_A_NOT_DONE:.+]] = comb.xor %z.done, {{.+}} : i1
      // CHECK: %[[GROUP_A_GO_GUARD:.+]] = comb.and %[[FSM_IS_GROUP_A_BEGIN_STATE]], %[[GROUP_A_NOT_DONE]] : i1
      calyx.group @A {
        // CHECK:  %A.go = calyx.group_go %[[GROUP_A_GO_GUARD]] ? %[[SIGNAL_ON]] : i1
        // CHECK:  calyx.assign %z.go = %A.go ? %go : i1
        %A.go = calyx.group_go %undef : i1
        calyx.assign %z.go = %A.go ? %go : i1
        calyx.group_done %z.done : i1
      }

      // CHECK: %[[GROUP_B_FSM_BEGIN:.+]] = hw.constant 1 : i2
      // CHECK: %[[GROUP_B_DONE:.+]] = comb.and %z.flag, %z.done : i1
      // CHECK: %[[FSM_IS_GROUP_B_BEGIN_STATE:.+]] = comb.icmp eq %fsm.out, %[[GROUP_B_FSM_BEGIN]] : i2
      // CHECK: %[[GROUP_B_NOT_DONE:.+]] = comb.xor %[[GROUP_B_DONE]], {{.+}} : i1
      // CHECK: %[[GROUP_B_GO_GUARD:.+]] = comb.and %[[FSM_IS_GROUP_B_BEGIN_STATE]], %[[GROUP_B_NOT_DONE]] : i1
      calyx.group @B {
        // CHECK: %B.go = calyx.group_go %[[GROUP_B_GO_GUARD]] ? %[[SIGNAL_ON]] : i1
        %B.go = calyx.group_go %undef : i1
        calyx.group_done %z.flag ? %z.done : i1
      }

      // CHECK: %[[GROUP_A_ASSIGN_GUARD:.+]] = comb.and %[[FSM_IS_GROUP_A_BEGIN_STATE]], %z.done : i1
      // CHECK: %[[FSM_STEP_1:.+]] = hw.constant 1 : i2
      // CHECK: %[[GROUP_B_ASSIGN_GUARD:.+]] = comb.and %[[FSM_IS_GROUP_B_BEGIN_STATE]], %[[GROUP_B_DONE]] : i1
      // CHECK: %[[FSM_STEP_2:.+]] = hw.constant -2 : i2
      // CHECK: %[[SEQ_GROUP_DONE_GUARD:.+]] = comb.icmp eq %fsm.out, %[[FSM_STEP_2]] : i2

      // CHECK-LABEL: calyx.group @seq {
      // CHECK-NEXT:    calyx.assign %fsm.in = %[[GROUP_A_ASSIGN_GUARD]] ? %[[FSM_STEP_1]] : i2
      // CHECK-NEXT:    calyx.assign %fsm.write_en = %[[GROUP_A_ASSIGN_GUARD]] ? %[[SIGNAL_ON]] : i1
      // CHECK-NEXT:    calyx.assign %fsm.in = %[[GROUP_B_ASSIGN_GUARD]] ? %[[FSM_STEP_2]] : i2
      // CHECK-NEXT:    calyx.assign %fsm.write_en = %[[GROUP_B_ASSIGN_GUARD]] ? %[[SIGNAL_ON]] : i1
      // CHECK-NEXT:    calyx.group_done %[[SEQ_GROUP_DONE_GUARD]] ? %[[SIGNAL_ON]] : i1

      // CHECK: %[[FSM_RESET:.+]] = hw.constant 0 : i2
      // CHECK: calyx.assign %fsm.in = %[[SEQ_GROUP_DONE_GUARD]] ? %[[FSM_RESET]] : i2
      // CHECK: calyx.assign %fsm.write_en =  %[[SEQ_GROUP_DONE_GUARD]] ? %[[SIGNAL_ON]] : i1
    }

    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.enable @seq {compiledGroups = [@A, @B]}
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @A
        calyx.enable @B
      }
    }
  }
}
