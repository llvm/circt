// RUN: circt-opt --split-input-file -pass-pipeline='builtin.module(calyx.component(lower-calyx-to-fsm))' %s | FileCheck %s

// CHECK:      module attributes {calyx.entrypoint = "main"} {
// CHECK-NEXT:   calyx.component @identity(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
// CHECK-NEXT:     %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     calyx.wires {
// CHECK-NEXT:       calyx.group @save {
// CHECK-NEXT:         calyx.assign %r.in = %in : i32
// CHECK-NEXT:         calyx.assign %r.write_en = %true : i1
// CHECK-NEXT:         calyx.group_done %r.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.assign %out = %r.out : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     calyx.control {
// CHECK-NEXT:       fsm.machine @control_identity() attributes {compiledGroups = [@save], initialState = "fsm_entry"} {
// CHECK-NEXT:         fsm.state @fsm_entry output {
// CHECK-NEXT:           fsm.output
// CHECK-NEXT:         } transitions {
// CHECK-NEXT:           fsm.transition @seq_0_save
// CHECK-NEXT:         }
// CHECK-NEXT:         fsm.state @fsm_exit output {
// CHECK-NEXT:           fsm.output
// CHECK-NEXT:         } transitions {
// CHECK-NEXT:         }
// CHECK-NEXT:         fsm.state @seq_0_save output {
// CHECK-NEXT:           calyx.enable @save
// CHECK-NEXT:           fsm.output
// CHECK-NEXT:         } transitions {
// CHECK-NEXT:           fsm.transition @fsm_exit
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}, %out: i32) {
// CHECK-NEXT:     %id.in, %id.go, %id.clk, %id.reset, %id.out, %id.done = calyx.instance @id of @identity : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:     %c10_i32 = hw.constant 10 : i32
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     calyx.wires {
// CHECK-NEXT:       calyx.group @invoke_0 {
// CHECK-NEXT:         calyx.assign %id.go = %true : i1
// CHECK-NEXT:         calyx.assign %id.in = %c10_i32 : i32
// CHECK-NEXT:         calyx.group_done %id.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.group @invoke_1 {
// CHECK-NEXT:         calyx.assign %r.write_en = %true : i1
// CHECK-NEXT:         calyx.assign %r.in = %id.out : i32
// CHECK-NEXT:         calyx.group_done %r.done : i1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:    calyx.control {
// CHECK-NEXT:       fsm.machine @control_main() attributes {compiledGroups = [@invoke_1, @invoke_0], initialState = "fsm_entry"} {
// CHECK-NEXT:         fsm.state @fsm_entry output {
// CHECK-NEXT:           fsm.output
// CHECK-NEXT:        } transitions {
// CHECK-NEXT:          fsm.transition @seq_0_invoke_0
// CHECK-NEXT:        }
// CHECK-NEXT:       fsm.state @fsm_exit output {
// CHECK-NEXT:         fsm.output
// CHECK-NEXT:       } transitions {
// CHECK-NEXT:        }
// CHECK-NEXT:        fsm.state @seq_1_invoke_1 output {
// CHECK-NEXT:          calyx.enable @invoke_1
// CHECK-NEXT:          fsm.output
// CHECK-NEXT:        } transitions {
// CHECK-NEXT:          fsm.transition @fsm_exit
// CHECK-NEXT:        }
// CHECK-NEXT:        fsm.state @seq_0_invoke_0 output {
// CHECK-NEXT:          calyx.enable @invoke_0
// CHECK-NEXT:          fsm.output
// CHECK-NEXT:        } transitions {
// CHECK-NEXT:          fsm.transition @seq_1_invoke_1
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT: }

module attributes {calyx.entrypoint = "main"} { 
calyx.component @identity(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
  %c1_1 = hw.constant 1 : i1 
  calyx.wires {
    calyx.group @save {
      calyx.assign %r.in = %in : i32 
      calyx.assign %r.write_en = %c1_1 : i1 
      calyx.group_done %r.done : i1
   } 
    calyx.assign %out = %r.out :i32
  }

  calyx.control {
    calyx.seq {
      calyx.enable @save
    }
  }
}

calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}, %out : i32) {
  %id.in, %id.go, %id.clk, %id.reset, %id.out, %id.done = calyx.instance @id of @identity : i32, i1, i1, i1, i32, i1
  %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1 
  %c1_10 = hw.constant 10 : i32

    calyx.control {
      calyx.seq {
        calyx.invoke @id(%id.in = %c1_10) -> (i32)
        calyx.invoke @r(%r.in = %id.out) -> (i32) 
      }
    }
  }
}
