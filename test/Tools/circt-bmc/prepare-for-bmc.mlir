// RUN: circt-opt %s --prepare-for-bmc="top-module=Top" | FileCheck %s

// CHECK-LABEL: hw.module @Top(
// CHECK-SAME:    in [[CLK:%[^:]+]] : !seq.clock
// CHECK-SAME:    in [[DATA:%[^:]+]] : i1
// CHECK:         [[RAW_CLK:%.+]] = seq.from_clock [[CLK]]
// CHECK-NOT:     seq.to_clock
// CHECK:         seq.firreg [[DATA]] clock [[CLK]]
// CHECK:         verif.assert [[DATA]] if [[RAW_CLK]] : i1
hw.module @Top(in %clk: i1, in %data: i1) {
  %clock = seq.to_clock %clk
  %state = seq.firreg %data clock %clock : i1
  verif.assert %data if %clk : i1
}
