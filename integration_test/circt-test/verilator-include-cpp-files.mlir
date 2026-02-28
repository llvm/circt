// RUN: circt-test %s -d %t -r \verilator 2>&1 | FileCheck %s
// REQUIRES: verilator

// CHECK: 1 tests passed

sim.func.dpi @someFunction()

verif.simulation @Foo {} {
^bb0(%clock: !seq.clock, %init: i1):
  sim.func.dpi.call @someFunction() clock %clock : () -> ()
  %true = hw.constant true
  verif.yield %true, %true : i1, i1
}

// This abuses verbatim SV quite a bit, but I've seen this in the wild when
// people try to bundle C/C++/Rust code up with some DPI-based testbench.
sv.verbatim "#include <iostream>\nextern \"C\" void someFunction() { std::cout << \"Hello from C++\\n\"; }" {output_file = #hw.output_file<"someFunction.cpp">}
