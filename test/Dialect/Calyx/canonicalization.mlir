// RUN: circt-opt %s -canonicalize | FileCheck %s

// Nested SeqOps are collapsed.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.seq {
          calyx.enable @A
        }
      }
    }
  }
}

// Nested ParOps are collapsed.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.par {
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.par {
        calyx.par {
          calyx.enable @A
        }
      }
    }
  }
}

// IfOp removes common tails from within SeqOps.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq "eq" : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @B {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.group @C {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.if %eq.out with @Cond {
    // CHECK-NEXT:        calyx.seq {
    // CHECK-NEXT:          calyx.enable @B
    // CHECK-NEXT:        }
    // CHECK-NEXT:      else {
    // CHECK-NEXT:        calyx.seq {
    // CHECK-NEXT:          calyx.enable @C
    // CHECK-NEXT:        }
    // CHECK-NEXT:     }
    // CHECK-NEXT:     calyx.enable @A
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    calyx.control {
      calyx.seq {
        calyx.if %eq.out with @Cond {
          calyx.seq {
            calyx.enable @B
            calyx.enable @A
          }
        } else {
          calyx.seq {
            calyx.enable @C
            calyx.enable @A
          }
        }
      }
    }
  }
}
