// RUN: circt-opt -pass-pipeline='calyx.program(calyx.component(lower-calyx-to-fsm,calyx-materialize-fsm))' %s | FileCheck %s

// todo: non-deterministically fails due to state reordering.

// CHECK: fsm.machine @control(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1, i1) attributes {compiledGroups = [@true, @false, @cond], initialState = "fsm_entry", stateType = i1} {
// CHECK-NEXT:   %true_0 = hw.constant true
// CHECK-NEXT:   %false_1 = hw.constant false
// CHECK-NEXT:   fsm.state "fsm_entry" output {
// CHECK-NEXT:     fsm.output %false_1, %false_1, %false_1 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_0_cond guard {
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state "fsm_exit" output {
// CHECK-NEXT:     fsm.output %false_1, %false_1, %false_1 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state "seq_1_if" output {
// CHECK-NEXT:     fsm.output %false_1, %false_1, %false_1 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_if_then guard {
// CHECK-NEXT:       %0 = comb.and %lt_reg.out : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:     fsm.transition @seq_1_if_else guard {
// CHECK-NEXT:       %true_2 = hw.constant true
// CHECK-NEXT:       %0 = comb.xor %lt_reg.out, %true_2 : i1
// CHECK-NEXT:       %1 = comb.and %0 : i1
// CHECK-NEXT:       fsm.return %1
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state "seq_1_if_then" output {
// CHECK-NEXT:     fsm.output %false_1, %false_1, %false_1 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_if_then_seq_0_true guard {
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state "seq_1_if_then_seq_0_true" output {
// CHECK-NEXT:     fsm.output %false_1, %true_0, %false_1 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @fsm_exit guard {
// CHECK-NEXT:       %0 = comb.and %arg1 : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state "seq_1_if_else" output {
// CHECK-NEXT:     fsm.output %false_1, %false_1, %false_1 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_if_else_seq_0_false guard {
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state "seq_1_if_else_seq_0_false" output {
// CHECK-NEXT:     fsm.output %true_0, %false_1, %false_1 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @fsm_exit guard {
// CHECK-NEXT:       %0 = comb.and %arg0 : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state "seq_0_cond" output {
// CHECK-NEXT:     fsm.output %false_1, %false_1, %true_0 : i1, i1, i1
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_if guard {
// CHECK-NEXT:       %0 = comb.and %arg2 : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     } action {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
    %false = hw.constant false
    %true = hw.constant true
    %lt_reg.in, %lt_reg.write_en, %lt_reg.clk, %lt_reg.reset, %lt_reg.out, %lt_reg.done = calyx.register @lt_reg : i1, i1, i1, i1, i1, i1
    %t.in, %t.write_en, %t.clk, %t.reset, %t.out, %t.done = calyx.register @t : i1, i1, i1, i1, i1, i1
    %f.in, %f.write_en, %f.clk, %f.reset, %f.out, %f.done = calyx.register @f : i1, i1, i1, i1, i1, i1
    %lt.left, %lt.right, %lt.out = calyx.std_lt @lt : i1, i1, i1
    calyx.wires {
      %0 = calyx.undef : i1
      calyx.group @true {
        %true.go = calyx.group_go %0 : i1
        calyx.assign %t.in = %true.go ? %true : i1
        calyx.assign %t.write_en = %true.go ? %true : i1
        calyx.group_done %t.done : i1
      }
      calyx.group @false {
        %false.go = calyx.group_go %0 : i1
        calyx.assign %f.in = %false.go ? %true : i1
        calyx.assign %f.write_en = %false.go ? %true : i1
        calyx.group_done %f.done : i1
      }
      calyx.group @cond {
        %cond.go = calyx.group_go %0 : i1
        calyx.assign %lt_reg.in = %cond.go ? %lt.out : i1
        calyx.assign %lt_reg.write_en = %cond.go ? %true : i1
        calyx.assign %lt.left = %cond.go ? %true : i1
        calyx.assign %lt.right = %cond.go ? %false : i1
        calyx.group_done %lt_reg.done ? %true : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.enable @cond
        calyx.if %lt_reg.out {
          calyx.seq {
            calyx.enable @true
          }
        } else {
          calyx.seq {
            calyx.enable @false
          }
        }
      }
    }
  }
}
