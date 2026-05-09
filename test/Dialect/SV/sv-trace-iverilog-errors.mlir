// RUN: circt-opt --sv-trace-iverilog %s -verify-diagnostics 

// Issue 9375
// expected-error@+1 {{Expected exactly one top level node}}
builtin.module {
  func.func @top () {
      func.return
  }
}
