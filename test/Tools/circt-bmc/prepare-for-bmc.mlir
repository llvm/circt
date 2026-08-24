// RUN: circt-opt %s --prepare-for-bmc="top-module=Top" | FileCheck %s

// CHECK-LABEL: hw.module @Top(
// CHECK-SAME:    in [[CLK:%[^:]+]] : !seq.clock
// CHECK-SAME:    in [[DATA:%[^:]+]] : i1
// CHECK-SAME:    in [[ENABLE:%[^:]+]] : i1
// CHECK:         [[RAW_CLK:%.+]] = seq.from_clock [[CLK]]
// CHECK-NOT:     seq.to_clock
// CHECK:         seq.firreg [[DATA]] clock [[CLK]]
// CHECK:         [[INITIAL:%.+]] = seq.initial() {
// CHECK:           [[FALSE:%.+]] = hw.constant false
// CHECK:           seq.yield [[FALSE]] : i1
// CHECK:         }
// CHECK:         [[PREVIOUS_CLK:%.+]] = seq.compreg [[RAW_CLK]], [[CLK]] initial [[INITIAL]] : i1
// CHECK:         [[TRUE:%.+]] = hw.constant true
// CHECK:         [[NOT_PREVIOUS:%.+]] = comb.xor [[PREVIOUS_CLK]], [[TRUE]] : i1
// CHECK:         [[POSEDGE:%.+]] = comb.and [[RAW_CLK]], [[NOT_PREVIOUS]] : i1
// CHECK:         [[ACTIVE:%.+]] = comb.and [[POSEDGE]], [[ENABLE]] : i1
// CHECK:         [[INACTIVE:%.+]] = comb.xor [[ACTIVE]], [[TRUE]] : i1
// CHECK:         [[ASSERT_PROPERTY:%.+]] = comb.or [[INACTIVE]], [[DATA]] : i1
// CHECK:         verif.assert [[ASSERT_PROPERTY]] : i1
// CHECK:         [[NOT_CURRENT:%.+]] = comb.xor [[RAW_CLK]], {{%.+}} : i1
// CHECK:         [[NEGEDGE:%.+]] = comb.and [[NOT_CURRENT]], [[PREVIOUS_CLK]] : i1
// CHECK:         [[INACTIVE_NEGEDGE:%.+]] = comb.xor [[NEGEDGE]], {{%.+}} : i1
// CHECK:         [[ASSUME_PROPERTY:%.+]] = comb.or [[INACTIVE_NEGEDGE]], [[DATA]] : i1
// CHECK:         verif.assume [[ASSUME_PROPERTY]] : i1
// CHECK:         [[BOTH_EDGES:%.+]] = comb.xor [[RAW_CLK]], [[PREVIOUS_CLK]] : i1
// CHECK:         [[INACTIVE_BOTH:%.+]] = comb.xor [[BOTH_EDGES]], {{%.+}} : i1
// CHECK:         [[BOTH_PROPERTY:%.+]] = comb.or [[INACTIVE_BOTH]], [[DATA]] : i1
// CHECK:         verif.assert [[BOTH_PROPERTY]] : i1
// CHECK-NOT:     verif.clocked_
hw.module @Top(in %clk: i1, in %data: i1, in %enable: i1) {
  %clock = seq.to_clock %clk
  %state = seq.firreg %data clock %clock : i1
  verif.clocked_assert %data if %enable, posedge %clk : i1
  verif.clocked_assume %data, negedge %clk : i1
  verif.clocked_assert %data, edge %clk : i1
}
