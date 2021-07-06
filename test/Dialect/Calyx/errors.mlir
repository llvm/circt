// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{'calyx.program' op must contain one component named "main" as the entry point.}}
calyx.program {}

// -----

calyx.program {
  // expected-error @+1 {{'calyx.component' op requires exactly one of each: 'calyx.wires', 'calyx.control'.}}
  calyx.component @main() -> () {
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main() -> () {
    // expected-error @+1 {{'calyx.cell' op is referencing component: A, which does not exist.}}
    calyx.cell "a0" @A

    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A(%in: i16) -> () {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    // expected-error @+1 {{'calyx.cell' op has a wrong number of results; expected: 1 but got 0}}
    calyx.cell "a0" @A
    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @B(%in: i16) -> () {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    // expected-error @+1 {{'calyx.cell' op result type for "%in" must be 'i16', but got 'i1'}}
    %0 = calyx.cell "b0" @B : i1

    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A() -> (%out: i16) {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @B(%in: i16) -> () {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    %0 = calyx.cell "a0" @A : i16
    %1 = calyx.cell "b0" @B : i16
    // expected-error @+1 {{'calyx.assign' op should only be contained in 'calyx.wires' or 'calyx.group'}}
    calyx.assign %1 = %0 : i16

    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main() -> () {
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
  calyx.component @B() -> () {
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    calyx.wires {}
    // expected-error @+1 {{'calyx.control' op has operation: calyx.cell, which is not allowed in this control-like operation}}
    calyx.control {
      calyx.cell "b0" @B
    }
  }
}

// -----

calyx.program {
  calyx.component @main() -> () {
    %c1_1 = constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with name: Group1 is unused in the control execution schedule}}
      calyx.group @Group1 {
        calyx.done %c1_1 : i1
      }
    }
    calyx.control {}
  }
}
