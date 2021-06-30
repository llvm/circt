// RUN: circt-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{'calyx.program' op must contain one component named "main" as the entry point.}}
calyx.program {}

// -----

calyx.program {
  // expected-error @+1 {{'calyx.component' op requires exactly one of each: 'calyx.cells', 'calyx.wires', 'calyx.control'.}}
  calyx.component @main() -> () {
    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @main() -> () {
    calyx.cells {
      // expected-error @+1 {{'calyx.cell' op is referencing component: A, which does not exist.}}
      calyx.cell "a0" @A
    }
    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @A(%in: i16) -> () {
    calyx.cells {}
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    calyx.cells {
      // expected-error @+1 {{'calyx.cell' op has a wrong number of results; expected: 1 but got 0}}
      calyx.cell "a0" @A
    }
    calyx.wires {}
    calyx.control {}
  }
}

// -----

calyx.program {
  calyx.component @B(%in: i16) -> () {
    calyx.cells {}
    calyx.wires {}
    calyx.control {}
  }
  calyx.component @main() -> () {
    calyx.cells {
      // expected-error @+1 {{'calyx.cell' op result type for "%in" must be 'i16', but got 'i1'}}
      %0 = calyx.cell "b0" @B : i1
    }
    calyx.wires {}
    calyx.control {}
  }
}
