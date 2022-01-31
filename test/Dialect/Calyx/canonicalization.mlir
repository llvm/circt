// RUN: circt-opt %s -canonicalize -split-input-file | FileCheck %s

// Nested SeqOps are collapsed.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
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

// -----

// Nested ParOps are collapsed.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
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

// -----

// IfOp nested in SeqOp removes common tail from within SeqOps.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
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
    // CHECK-NEXT:      } else {
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

// -----

// IfOp nested in ParOp removes common tails from within ParOps.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
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
      calyx.group @D {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.par {
    // CHECK-NEXT:      calyx.if %eq.out with @Cond {
    // CHECK-NEXT:        calyx.par {
    // CHECK-NEXT:          calyx.enable @A
    // CHECK-NEXT:        }
    // CHECK-NEXT:      } else {
    // CHECK-NEXT:        calyx.par {
    // CHECK-NEXT:          calyx.enable @B
    // CHECK-NEXT:        }
    // CHECK-NEXT:     }
    // CHECK-NEXT:     calyx.enable @C
    // CHECK-NEXT:     calyx.enable @D
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    calyx.control {
      calyx.par {
        calyx.if %eq.out with @Cond {
          calyx.par {
            calyx.enable @A
            calyx.enable @C
            calyx.enable @D
          }
        } else {
          calyx.par {
            calyx.enable @B
            calyx.enable @C
            calyx.enable @D
          }
        }
      }
    }
  }
}

// -----

// IfOp nested in ParOp removes common tail from within SeqOps. The important check
// here is ensuring the removed EnableOps are still computed sequentially.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
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
    // CHECK-NEXT:    calyx.par {
    // CHECK-NEXT:      calyx.seq {
    // CHECK-NEXT:        calyx.if %eq.out with @Cond {
    // CHECK-NEXT:          calyx.seq {
    // CHECK-NEXT:            calyx.enable @B
    // CHECK-NEXT:          }
    // CHECK-NEXT:        } else {
    // CHECK-NEXT:          calyx.seq {
    // CHECK-NEXT:            calyx.enable @C
    // CHECK-NEXT:          }
    // CHECK-NEXT:        }
    // CHECK-NEXT:        calyx.enable @A
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.par {
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

// -----

// IfOp nested in SeqOp removes common tail from within ParOps. The important check
// here is ensuring the removed EnableOps are still computed in parallel.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
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
      calyx.group @D {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.par {
    // CHECK-NEXT:        calyx.if %eq.out with @Cond {
    // CHECK-NEXT:          calyx.par {
    // CHECK-NEXT:            calyx.enable @A
    // CHECK-NEXT:          }
    // CHECK-NEXT:        } else {
    // CHECK-NEXT:          calyx.par {
    // CHECK-NEXT:            calyx.enable @B
    // CHECK-NEXT:          }
    // CHECK-NEXT:        }
    // CHECK-NEXT:        calyx.enable @C
    // CHECK-NEXT:        calyx.enable @D
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.if %eq.out with @Cond {
          calyx.par {
            calyx.enable @A
            calyx.enable @C
            calyx.enable @D
          }
        } else {
          calyx.par {
            calyx.enable @B
            calyx.enable @C
            calyx.enable @D
          }
        }
      }
    }
  }
}

// -----

// Empty Then and Else regions lead to the removal of the IfOp (as well as unused cells and groups).
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    // CHECK-NOT: %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // CHECK-NOT: calyx.comp_group @Cond
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
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
        calyx.if %eq.out with @Cond {
          calyx.seq {
            calyx.enable @A
          }
        } else {
          calyx.seq {
            calyx.enable @A
          }
        }
      }
    }
  }
}

// -----

// Empty Then region and no Else region leads to removal of IfOp (as well as unused cells).
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    // CHECK-NOT: %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
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
        calyx.enable @A
        calyx.if %eq.out {}
      }
    }
  }
}

// -----

// Empty body leads to removal of WhileOp (as well as unused cells and groups).
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    // CHECK: %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // CHECK-NOT: calyx.comp_group @Cond
      calyx.comb_group @Cond {
        calyx.assign %eq.left =  %c1_1 : i1
        calyx.assign %eq.right = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        // Use the `std_eq` here to verify it is not removed.
        calyx.assign %r.write_en = %eq.out : i1
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
        calyx.enable @A
        calyx.while %eq.out with @Cond {}
      }
    }
  }
}

// -----

// Empty ParOp and SeqOp are removed.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
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
        calyx.enable @A
        calyx.seq { calyx.seq {} }
        calyx.par { calyx.seq {} }
      }
    }
  }
}

// -----

// Unary control operations are collapsed.
calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
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
    }
    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:    calyx.seq {
    // CHECK-NEXT:      calyx.enable @B
    // CHECK-NEXT:      calyx.enable @A
    // CHECK-NEXT:      calyx.enable @B
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.enable @B
        calyx.par {
          calyx.seq {
            calyx.enable @A
          }
        }
        calyx.enable @B
      }
    }
  }
}
