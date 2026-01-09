// RUN: circt-test %s -d %t -r \sby 2>&1 | FileCheck %s
// REQUIRES: sby

// CHECK: 1 tests passed

verif.formal @Foo {} {
  %a = verif.symbolic_value : i42
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42
  %1 = comb.add %a, %0 : i42
  %2 = comb.mul %a, %c9_i42 : i42
  %3 = comb.icmp eq %1, %2 : i42
  // assert((a<<3)+a == a*9)
  verif.assert %3 : i1
}

// This abuses verbatim SV quite a bit, but I've seen this in the wild when
// people try to bundle C/C++/Rust code up with some DPI-based testbench.
sv.verbatim "void someDummyFunction() {}" {output_file = #hw.output_file<"ignored.cpp">}
