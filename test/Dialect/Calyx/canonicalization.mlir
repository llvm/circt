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
