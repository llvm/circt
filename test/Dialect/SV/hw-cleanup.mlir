// RUN: circt-opt -hw-cleanup='convert-if-to-case=true' %s | FileCheck %s

//CHECK-LABEL: hw.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite [[FD]], "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:     sv.fwrite [[FD]], "A1"
//CHECK-NEXT:     sv.fwrite [[FD]], "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg1)  {
//CHECK-NEXT:     sv.fwrite [[FD]], "B1"
//CHECK-NEXT:     sv.fwrite [[FD]], "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }

hw.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff(posedge %arg0) {
    sv.fwrite %fd, "A1"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite %fd, "B1"
  }
  sv.initial {
    sv.fwrite %fd, "Middle\n"
  }
  sv.alwaysff(posedge %arg0) {
    sv.fwrite %fd, "A2"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite %fd, "B2"
  }
  hw.output
}

// CHECK-LABEL: hw.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "A1"
// CHECK-NEXT:     sv.fwrite [[FD]], "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "B1"
// CHECK-NEXT:     sv.fwrite [[FD]], "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output
// CHECK-NEXT: }

hw.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B2"
  }
  hw.output
}


// CHECK-LABEL: hw.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "A1"
// CHECK-NEXT:     sv.fwrite [[FD]], "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "B1"
// CHECK-NEXT:     sv.fwrite [[FD]], "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "C1"
// CHECK-NEXT:     sv.fwrite [[FD]], "C2"
// CHECK-NEXT:   }(asyncreset : posedge %arg1)  {
// CHECK-NEXT:     sv.fwrite [[FD]], "D1"
// CHECK-NEXT:     sv.fwrite [[FD]], "D2"
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output
// CHECK-NEXT: }

hw.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "C1"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite %fd, "D1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "B2"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "C2"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite %fd, "D2"
  }
  hw.output
}

//CHECK-LABEL: hw.module @alwaysff_ifdef(%arg0: i1) {
//CHECK-NEXT:  [[FD:%.*]] = hw.constant -2147483646 : i32
//CHECK-NEXT:  sv.ifdef "FOO" {
//CHECK-NEXT:     sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:       sv.fwrite [[FD]], "A1"
//CHECK-NEXT:       sv.fwrite [[FD]], "B1"
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }

hw.module @alwaysff_ifdef(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "A1"
    }
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @ifdef_merge(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.ifdef "FOO"  {
// CHECK-NEXT:      sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:        sv.fwrite [[FD]], "A1"
// CHECK-NEXT:        sv.fwrite [[FD]], "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @ifdef_merge(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "A1"
    }
  }
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite %fd, "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @ifdef_proc_merge(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      [[XOR:%.*]] = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.ifdef.procedural "FOO"  {
// CHECK-NEXT:        sv.fwrite [[FD]], "A1"
// CHECK-NEXT:        sv.fwrite [[FD]], "%x"([[XOR]]) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.ifdef.procedural "BAR"  {
// CHECK-NEXT:        sv.fwrite [[FD]], "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @ifdef_proc_merge(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff(posedge %arg0) {
    sv.ifdef.procedural "FOO" {
      sv.fwrite %fd, "A1"
    }
    %true = hw.constant true
    %0 = comb.xor %arg0, %true : i1
    sv.ifdef.procedural "FOO" {
       sv.fwrite %fd, "%x"(%0) : i1
    }
     sv.ifdef.procedural "BAR" {
       sv.fwrite %fd, "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @if_merge(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      [[XOR:%.*]] = comb.xor %arg1, %true : i1
// CHECK-NEXT:      sv.if %arg1  {
// CHECK-NEXT:        sv.fwrite [[FD]], "A1"
// CHECK-NEXT:        sv.fwrite [[FD]], "%x"([[XOR:%.*]]) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.if [[XOR]] {
// CHECK-NEXT:        sv.fwrite [[FD]], "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @if_merge(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.alwaysff(posedge %arg0) {
    sv.if %arg1 {
      sv.fwrite %fd, "A1"
    }
    %true = hw.constant true
    %0 = comb.xor %arg1, %true : i1
    sv.if %arg1 {
      sv.fwrite %fd, "%x"(%0) : i1
    }
    sv.if %0 {
      sv.fwrite %fd, "B1"
    }
  }
  hw.output
}


// CHECK-LABEL: hw.module @initial_merge(%arg0: i1) {
// CHECK-NEXT:    [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.initial {
// CHECK-NEXT:      sv.fwrite [[FD]], "A1"
// CHECK-NEXT:      sv.fwrite [[FD]], "B1"
// CHECK-NEXT:    }
hw.module @initial_merge(%arg0: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.initial {
    sv.fwrite %fd, "A1"
  }
  sv.initial {
    sv.fwrite %fd, "B1"
  }
  hw.output
}

//CHECK-LABEL: hw.module @always_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   [[FD:%.*]] = hw.constant -2147483646 : i32
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite [[FD]], "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg0   {
//CHECK-NEXT:     sv.fwrite [[FD]], "A1"
//CHECK-NEXT:     sv.fwrite [[FD]], "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg1   {
//CHECK-NEXT:     sv.fwrite [[FD]], "B1"
//CHECK-NEXT:     sv.fwrite [[FD]], "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }
hw.module @always_basic(%arg0: i1, %arg1: i1) {
  %fd = hw.constant 0x80000002 : i32

  sv.always posedge %arg0 {
    sv.fwrite %fd, "A1"
  }
  sv.always posedge %arg1 {
    sv.fwrite %fd, "B1"
  }
  sv.initial {
    sv.fwrite %fd, "Middle\n"
  }
  sv.always posedge %arg0 {
    sv.fwrite %fd, "A2"
  }
  sv.always posedge %arg1 {
    sv.fwrite %fd, "B2"
  }
  hw.output
}


// CHECK-LABEL: hw.module @alwayscomb_basic(
hw.module @alwayscomb_basic(%a: i1, %b: i1) -> (x: i1, y: i1) {
  %w1 = sv.reg : !hw.inout<i1>
  %w2 = sv.reg : !hw.inout<i1>
  // CHECK: sv.alwayscomb {
  sv.alwayscomb {
    // CHECK-NEXT: sv.bpassign %w1, %a : i1
    sv.bpassign %w1, %a : i1
  }

  %out1 = sv.read_inout %w1 : !hw.inout<i1>

  sv.alwayscomb {
    // CHECK-NEXT: sv.bpassign %w2, %b : i1
    sv.bpassign %w2, %b : i1
  } // CHECK-NEXT: }

  %out2 = sv.read_inout %w1 : !hw.inout<i1>

  hw.output %out1, %out2 : i1, i1
}

// CHECK-LABEL: hw.module @nested_regions(
// CHECK-NEXT:  [[FD:%.*]] = hw.constant -2147483646 : i32
// CHECK-NEXT:  sv.initial  {
// CHECK-NEXT:    sv.ifdef.procedural "L1"  {
// CHECK-NEXT:      sv.ifdef.procedural "L2"  {
// CHECK-NEXT:        sv.ifdef.procedural "L3"  {
// CHECK-NEXT:          sv.fwrite [[FD]], "A"
// CHECK-NEXT:          sv.fwrite [[FD]], "B"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
hw.module @nested_regions() {
  %fd = hw.constant 0x80000002 : i32

  sv.initial {
    sv.ifdef.procedural "L1" {
      sv.ifdef.procedural "L2" {
        sv.ifdef.procedural "L3" {
          sv.fwrite %fd, "A"
        }
      }
    }
  }
  sv.initial {
    sv.ifdef.procedural "L1" {
      sv.ifdef.procedural "L2" {
        sv.ifdef.procedural "L3" {
          sv.fwrite %fd, "B"
        }
      }
    }
  }
}

// CHECK-LABEL: hw.module @if_condition_hosting(
hw.module @if_condition_hosting(%a: i1, %b: i1, %c:i1, %d:i1) {
  %0 = comb.and %a, %b, %c, %d: i1
  %1 = comb.and %a, %b, %c: i1
  %2 = comb.and %a, %c, %d: i1
  %3 = comb.and %a, %d, %d: i1
  %fd = hw.constant 0 : i32
  // CHECK:      sv.initial  {
  // CHECK-NEXT:   sv.if %a  {
  // CHECK-NEXT:     sv.if %c  {
  // CHECK-NEXT:       sv.if %b  {
  // CHECK-NEXT:         sv.fwrite [[FD:%.*]], "A_B_C"
  // CHECK-NEXT:         sv.if %d  {
  // CHECK-NEXT:           sv.fwrite [[FD]], "A_B_C_D"
  // CHECK-NEXT:         }
  // CHECK-NEXT:       }
  // CHECK-NEXT:       sv.if %d  {
  // CHECK-NEXT:         sv.fwrite [[FD]], "A_C_D"
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.if %d  {
  // CHECK-NEXT:       sv.fwrite [[FD]], "A_D"
  // CHECK-NEXT:       %0 = comb.and %b, %c : i1
  // CHECK-NEXT:       sv.if %0  {
  // CHECK-NEXT:         sv.fwrite [[FD]], "A_B_C_D again"
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sv.initial {
    sv.if %0 {
      sv.fwrite %fd, "A_B_C_D"
    }
    sv.if %1 {
      sv.fwrite %fd, "A_B_C"
    }
    sv.if %2 {
      sv.fwrite %fd, "A_C_D"
    }
    sv.if %3 {
      sv.fwrite %fd, "A_D"
    }
    sv.if %0 {
      sv.fwrite %fd, "A_B_C_D again"
    }
  }

}

hw.module @case(%sel1: i2, %sel2: i2) {
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c3_i2 = hw.constant 3 : i2
  %0 = comb.icmp eq %sel1, %c0_i2 : i2
  %1 = comb.icmp eq %sel1, %c1_i2 : i2
  %2 = comb.icmp eq %sel1, %c3_i2 : i2
  %3 = comb.icmp eq %sel2, %c3_i2 : i2
  %fd = hw.constant 0 : i32

  // CHECK: sv.initial
  sv.initial {
    // CHECK-NEXT:   sv.case %sel1 : i2
    // CHECK-NEXT:   case b00: {
    // CHECK-NEXT:     sv.fwrite [[FD:%.*]], "sel1 = 0"
    // CHECK-NEXT:     sv.fwrite [[FD]], "sel1 = 0 again"
    // CHECK-NEXT:   }
    // CHECK-NEXT:   case b01: {
    // CHECK-NEXT:     sv.fwrite [[FD]], "sel1 = 1"
    // CHECK-NEXT:   }
    // CHECK-NEXT:   case b11: {
    // CHECK-NEXT:     sv.fwrite [[FD]], "sel1 = 3"
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %0  {
    // CHECK-NEXT:     sv.fwrite [[FD]], "it should not be merged"
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.case %sel1 : i2
    // CHECK-NEXT:   case b00: {
    // CHECK-NEXT:     sv.fwrite [[FD]], "sel1 = 0"
    // CHECK-NEXT:   }
    // CHECK-NEXT:   case b01: {
    // CHECK-NEXT:     sv.fwrite [[FD]], "sel1 = 1"
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    sv.if %0 {
      sv.fwrite %fd, "sel1 = 0"
    }
    sv.if %1 {
      sv.fwrite %fd, "sel1 = 1"
    }
    sv.if %2 {
      sv.fwrite %fd, "sel1 = 3"
    }
    sv.if %0 {
      sv.fwrite %fd, "sel1 = 0 again"
    }
    sv.if %3 {
      sv.fwrite %fd, "it should not be merged"
    }
    sv.if %0 {
      sv.fwrite %fd, "sel1 = 0"
    }
    sv.if %1 {
      sv.fwrite %fd, "sel1 = 1"
    }
  }
}
