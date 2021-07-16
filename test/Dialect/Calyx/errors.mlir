// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{'calyx.program' op must contain one component named "main" as the entry point.}}
calyx.program {}

// -----

calyx.program {
  // expected-error @+1 {{'calyx.component' op requires exactly one of each: 'calyx.wires', 'calyx.control'.}}
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    // expected-error @+1 {{'calyx.cell' op is referencing component: A, which does not exist.}}
    calyx.cell "a0" @A

    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A(%in: i16, %go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    // expected-error @+1 {{'calyx.cell' op has a wrong number of results; expected: 5 but got 0}}
    calyx.cell "a0" @A
    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @B(%in: i16, %go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    calyx.wires {}
    calyx.control {}
  }
<<<<<<< HEAD
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    // expected-error @+1 {{'calyx.cell' op result type for "in" must be 'i16', but got 'i1'}}
    %b0.in, %b0.go, %b0.clk, %b0.reset, %b0.done = calyx.cell "b0" @B : i1, i1, i1, i1, i1
=======
  calyx.component @main() -> () {
    // expected-error @+1 {{'calyx.cell' op result type for "in" must be 'i16', but got 'i1'}}
    %0 = calyx.cell "b0" @B : i1
>>>>>>> origin/main

    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A(%go: i1, %clk: i1, %reset: i1) -> (%out: i16, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @B(%in: i16, %go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    %a.go, %a.clk, %a.reset, %a.out, %a.done = calyx.cell "a" @A : i1, i1, i1, i16, i1
    %b.in, %b.go, %b.clk, %b.reset, %b.done = calyx.cell "b" @B : i16, i1, i1, i1, i1
    // expected-error @+1 {{'calyx.assign' op expects parent op to be one of 'calyx.group, calyx.wires'}}
    calyx.assign %b.in = %a.out : i16

    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    calyx.wires {}
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.enable' op with group: WrongName, which does not exist.}}
        calyx.enable @WrongName
      }
    }
  }
}

// -----

calyx.program {
  calyx.component @B(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    %c1_1 = constant 1 : i1
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
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    %c1_1 = constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with name: Group1 is unused in the control execution schedule}}
      calyx.group @Group1 {
        calyx.group_done %c1_1 : i1
      }
    }
    calyx.control {}
  }
}
