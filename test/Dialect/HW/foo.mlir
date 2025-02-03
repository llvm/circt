// RUN: circt-opt --hw-foo-wires %s | FileCheck %s

hw.module @foo(in %a: i32, in %b: i32, out out: i32) {
  // CHECK:   %c1_i32 = hw.constant 1 : i32
  // CHECK:   %foo_0 = hw.wire %c1_i32  : i32
  // CHECK:   %foo_1 = hw.wire %a  : i32
  // CHECK:   %foo_2 = hw.wire %b  : i32
  // CHECK:   %0 = comb.add %foo_1, %foo_0 : i32
  // CHECK:   %foo_3 = hw.wire %0  : i32
  // CHECK:   %1 = comb.add %foo_3, %foo_2 : i32
  // CHECK:   %foo_4 = hw.wire %1  : i32
  // CHECK:   hw.output %foo_4 : i32
  %c1 = hw.constant 1 : i32
  %wire_1 = hw.wire %c1 name "wire_1" : i32
  %wire_a = hw.wire %a name "wire_a" : i32
  %wire_b = hw.wire %b name "wire_b" : i32
  %ap1 = comb.add %wire_a, %wire_1 : i32
  %wire_ap1 = hw.wire %ap1 name "wire_ap1" : i32
  %ap1pb = comb.add %wire_ap1, %wire_b : i32
  %wire_ap1pb = hw.wire %ap1pb : i32
  hw.output %wire_ap1pb : i32
}
