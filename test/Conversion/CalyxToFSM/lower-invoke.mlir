// RUN: circt-opt --split-input-file -pass-pipeline='builtin.module(calyx.component(lower-calyx-to-fsm))' %s | FileCheck %s

// CHECK:      fsm.machine @control_identity() attributes {compiledGroups = [@invoke_r_0], initialState = "fsm_entry"} {
// CHECK-NEXT:   fsm.state @fsm_entry output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_0_invoke_r_0
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @fsm_exit output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_0_invoke_r_0 output {
// CHECK-NEXT:     calyx.enable @invoke_r_0
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @fsm_exit
// CHECK-NEXT:   }
// CHECK-NEXT: }
  
// CHECK:      fsm.machine @control_main() attributes {compiledGroups = [@incr, @invoke_r_1, @invoke_id_0, @init], initialState = "fsm_entry"} {
// CHECK-NEXT:   fsm.state @fsm_entry output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_0_init
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @fsm_exit output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_header output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_entry guard {
// CHECK-NEXT:       fsm.return %lt.out
// CHECK-NEXT:     }
// CHECK-NEXT:     fsm.transition @fsm_exit guard {
// CHECK-NEXT:       %true_2 = hw.constant true
// CHECK-NEXT:       %0 = comb.xor %lt.out, %true_2 : i1
// CHECK-NEXT:       fsm.return %0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_entry output {
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_seq_0_invoke_id_0
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_seq_2_incr output {
// CHECK-NEXT:     calyx.enable @incr
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_header
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_seq_1_invoke_r_1 output {
// CHECK-NEXT:     calyx.enable @invoke_r_1
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_seq_2_incr
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_1_while_seq_0_invoke_id_0 output {
// CHECK-NEXT:     calyx.enable @invoke_id_0
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_seq_1_invoke_r_1
// CHECK-NEXT:   }
// CHECK-NEXT:   fsm.state @seq_0_init output {
// CHECK-NEXT:     calyx.enable @init
// CHECK-NEXT:     fsm.output
// CHECK-NEXT:   } transitions {
// CHECK-NEXT:     fsm.transition @seq_1_while_header
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {calyx.entrypoint = "main"} { 
calyx.component @identity(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
  calyx.wires {
    calyx.assign %out = %r.out :i32
  }
  calyx.control {
    calyx.seq {
      calyx.invoke @r(%r.in = %in) -> (i32)
    }
  }
}

calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}, %out : i32) {
  %id.in, %id.go, %id.clk, %id.reset, %id.out, %id.done = calyx.instance @id of @identity : i32, i1, i1, i1, i32, i1
  %counter.in, %counter.write_en, %counter.clk, %counter.reset, %counter.out, %counter.done = calyx.register @counter : i32, i1, i1, i1, i32, i1
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32
  %lt.left, %lt.right, %lt.out = calyx.std_lt @lt : i32, i32, i1
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1 
  %c0 = hw.constant 0 : i32
  %c1 = hw.constant 1 : i32
  %c8 = hw.constant 8 : i32
  %c10 = hw.constant 10 : i32
  %true = hw.constant 1 : i1
  calyx.wires {
    calyx.group @init {
       calyx.assign %counter.in = %c0 : i32
       calyx.assign %counter.write_en = %true : i1
       calyx.group_done %counter.done : i1
      }
      calyx.group @incr {
        calyx.assign %add.left = %counter.out : i32
        calyx.assign %add.right = %c1 : i32
        calyx.assign %counter.in = %add.out : i32
        calyx.assign %counter.write_en = %true : i1
        calyx.group_done %counter.done : i1
      }
      calyx.assign %lt.left = %counter.out : i32
      calyx.assign %lt.right = %c8 : i32
    }
    calyx.control {
      calyx.seq {
        calyx.enable @init
        calyx.while %lt.out {
          calyx.seq {
            calyx.invoke @id(%id.in = %c10) -> (i32)
            calyx.invoke @r(%r.in = %id.out) -> (i32) 
            calyx.enable @incr
          }
        }
      }
    }
  }
}
