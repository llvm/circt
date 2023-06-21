// RUN: circt-opt --split-input-file -pass-pipeline='builtin.module(calyx.component(lower-calyx-to-fsm))' %s | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   calyx.component @identity(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
// CHECK-NEXT:    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    %true_0 = hw.constant true
// CHECK-NEXT:    %true_1 = hw.constant true
// CHECK-NEXT:    calyx.wires {
// CHECK-NEXT:      calyx.assign %out = %r.out : i32
// CHECK-NEXT:      calyx.group @invoke_0 {
// CHECK-NEXT:        calyx.assign %r.write_en = %true_1 : i1
// CHECK-NEXT:        calyx.assign %r.in = %in : i32
// CHECK-NEXT:        calyx.group_done %r.done : i1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    calyx.control {
// CHECK-NEXT:      fsm.machine @control_identity() attributes {compiledGroups = [@invoke_0], initialState = "fsm_entry"} {
// CHECK-NEXT:        fsm.state @fsm_entry output {
// CHECK-NEXT:          fsm.output
// CHECK-NEXT:        } transitions {
// CHECK-NEXT:          fsm.transition @seq_0_invoke_0
// CHECK-NEXT:        }
// CHECK-NEXT:        fsm.state @fsm_exit output {
// CHECK-NEXT:          fsm.output
// CHECK-NEXT:        } transitions {
// CHECK-NEXT:        }
// CHECK-NEXT:        fsm.state @seq_0_invoke_0 output {
// CHECK-NEXT:          calyx.enable @invoke_0
// CHECK-NEXT:          fsm.output
// CHECK-NEXT:        } transitions {
// CHECK-NEXT:          fsm.transition @fsm_exit
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:}

calyx.component @identity(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1 
  %c1_1 = hw.constant 1 : i1
  %true = hw.constant true
  calyx.wires {
   calyx.assign %out = %r.out :i32
  }
  calyx.control {
    calyx.seq {
      calyx.invoke @r(%r.in = %in) -> (i32) 
    }
 }
}
