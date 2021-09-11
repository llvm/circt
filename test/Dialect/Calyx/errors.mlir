// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{'calyx.program' op must contain one component named "main" as the entry point.}}
calyx.program {}

// -----

calyx.program {
  // expected-error @+1 {{'calyx.component' op requires exactly one of each: 'calyx.wires', 'calyx.control'.}}
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.instance' op is referencing component: A, which does not exist.}}
    calyx.instance "a0" @A
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.instance' op has a wrong number of results; expected: 5 but got 0}}
    calyx.instance "a0" @A
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @B(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.instance' op result type for "in" must be 'i16', but got 'i1'}}
    %b0.in, %b0.go, %b0.clk, %b0.reset, %b0.done = calyx.instance "b0" @B : i1, i1, i1, i1, i1
    calyx.wires { calyx.assign %done = %b0.done : i1 }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i16, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @B(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %a.go, %a.clk, %a.reset, %a.out, %a.done = calyx.instance "a" @A : i1, i1, i1, i16, i1
    %b.in, %b.go, %b.clk, %b.reset, %b.done = calyx.instance "b" @B : i16, i1, i1, i1, i1
    // expected-error @+1 {{'calyx.assign' op expects parent op to be one of 'calyx.group, calyx.comb_group, calyx.wires'}}
    calyx.assign %b.in = %a.out : i16

    calyx.wires { calyx.assign %b.in = %a.out : i16 }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    calyx.wires {}
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.enable' op with group 'WrongName', which does not exist.}}
        calyx.enable @WrongName
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A { calyx.group_done %c1_1 : i1 }
    }
    // expected-error @+1 {{'calyx.control' op EnableOp is not a composition operator. It should be nested in a control flow operation, such as "calyx.seq"}}
    calyx.control {
      calyx.enable @A
      calyx.enable @A
    }
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with name: "Group1" is unused in the control execution schedule}}
      calyx.group @Group1 {
        calyx.group_done %c1_1 : i1
      }
      calyx.assign %done = %c1_1 : i1
    }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.comb_group @Group1 { calyx.assign %c0.go = %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{empty 'then' region.}}
        calyx.if %c0.out with @Group1 {}
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    calyx.wires {
      calyx.assign %done = %r.done : i1
    }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Group1 { calyx.assign %c0.go = %c1_1 : i1 }
      calyx.group @Group2 { calyx.group_done %c1_1 : i1 } 
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{empty 'else' region.}}
        calyx.if %c0.out with @Group1 {
          calyx.enable @Group2
        } else {}
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    calyx.wires { calyx.group @Group2 { calyx.group_done %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.if' op with group 'Group1', which does not exist.}}
        calyx.if %c0.out with @Group1 { calyx.enable @Group2}
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%in: i1, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Group1 {}
      calyx.group @Group2 { calyx.group_done %c1_1 : i1 } 
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{conditional op: '%c0.out' expected to be driven from group: 'Group1' but no driver was found.}}
        calyx.if %c0.out with @Group1 {
          calyx.enable @Group2
        }
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.comb_group @Group1 { calyx.assign %c0.go = %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{empty body region.}}
        calyx.while %c0.out with @Group1 {}
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    calyx.wires { calyx.comb_group @Group2 { calyx.assign %c0.go = %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.while' op with group 'Group1', which does not exist.}}
        calyx.while %c0.out with @Group1 {
          calyx.enable @Group2
        }
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%in: i1, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Group1 { }
      calyx.group @Group2 { calyx.group_done %c1_1 : i1 } 
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{conditional op: '%c0.out' expected to be driven from group: 'Group1' but no driver was found.}}
        calyx.while %c0.out with @Group1 {
          calyx.enable @Group2
        }
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.memory' op mismatched number of dimensions (1) and address sizes (2)}}
    %m.addr0, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory "m"<[64] x 8> [6, 6] : i6, i8, i1, i1, i8, i1
    calyx.wires { calyx.assign %done = %m.done : i1 }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.memory' op incorrect number of address ports, expected 2}}
    %m.addr0, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory "m"<[64, 64] x 8> [6, 6] : i6, i8, i1, i1, i8, i1
    calyx.wires { calyx.assign %done = %m.done : i1}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.memory' op address size (5) for dimension 0 can't address the entire range (64)}}
    %m.addr0, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory "m"<[64] x 8> [5] : i5, i8, i1, i1, i5, i1
    calyx.wires { calyx.assign %done = %m.done : i1 }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has an invalid destination port. It must be drive-able.}}
      calyx.assign %c1_1 = %go : i1
    }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a component port as the destination with the incorrect direction.}}
      calyx.assign %go = %c1_1 : i1
    }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a component port as the source with the incorrect direction.}}
      calyx.assign %r.write_en = %done : i1
    }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a cell port as the source with the incorrect direction.}}
      calyx.assign %done = %r.write_en : i1
    }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a cell port as the destination with the incorrect direction.}}
      calyx.assign %r.done = %go : i1
    }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A { calyx.group_done %c1_1 : i1 }
    }
    // expected-error @+1 {{'calyx.control' op has an invalid control sequence. Multiple control flow operations must all be nested in a single calyx.seq or calyx.par}}
    calyx.control {
      calyx.seq { calyx.enable @A }
      calyx.seq { calyx.enable @A }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.group @Group1 { calyx.group_done %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.if' op with group 'Group1', which is not a combinational group.}}
        calyx.if %c0.out with @Group1 { calyx.enable @Group1}
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.group @Group1 { calyx.group_done %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.while' op with group 'Group1', which is not a combinational group.}}
        calyx.while %c0.out with @Group1 { calyx.enable @Group1}
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.comb_group @Group1 { calyx.assign %c0.go = %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        calyx.if %c0.out {
          // expected-error @+1 {{'calyx.enable' op with group 'Group1', which is a combinational group.}}
          calyx.enable @Group1
        }
      }
    }
  }
}

// -----

calyx.program {
  // expected-error @+1 {{'calyx.component' op is missing the following required port attribute identifiers: done, go}}
  calyx.component @main(%go: i1, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with cell: calyx.register "r" is performing a write and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory "m"<[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %c1_i8 = hw.constant 1 : i8
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with cell: calyx.memory "m" is performing a write and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %m.write_data = %c1_i8 : i8
        calyx.group_done %m.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %gt.left, %gt.right, %gt.out = calyx.std_gt "gt" : i8, i8, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i1, i1, i1, i1, i1, i1
    %c1_i8 = hw.constant 1 : i8
    %c1_i1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{calyx.group' op with cell: calyx.std_gt "gt" is performing a write and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %r.in = %gt.out : i1
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.assign %gt.left = %c1_i8 : i8
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory "m"<[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    %c1_i1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with cell: calyx.memory "m" is having a read performed upon it, and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %r.in = %m.read_data : i8
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.comb_group' op with cell: calyx.register "r", which is not combinational.}}
      calyx.comb_group @IncorrectCombGroup {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.if %r.out with @IncorrectCombGroup {
          calyx.enable @A
        }
      }
    }
  }
}
