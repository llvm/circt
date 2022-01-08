// RUN: circt-opt %s --msft-partition -verify-diagnostics -split-input-file | FileCheck %s
// RUN: circt-opt %s --msft-partition --msft-wire-cleanup -verify-diagnostics -split-input-file | FileCheck --check-prefix=CLEANUP %s

msft.module @top {} (%clk : i1) -> (out1: i2, out2: i2) {
  msft.partition @part1, "dp"

  %res1 = msft.instance @b @B(%clk) : (i1) -> (i2)

  %c0 = hw.constant 0 : i2
  %res2 = msft.instance @unit1 @Extern(%c0) { targetDesignPartition = @top::@part1 }: (i2) -> (i2)

  msft.output %res1, %res2 : i2, i2
}

msft.module.extern @Extern (%foo_a: i2) -> (foo_x: i2)

msft.module @B {} (%clk : i1) -> (x: i2)  {
  %c1 = hw.constant 1 : i2
  %0 = msft.instance @unit1 @Extern(%c1) { targetDesignPartition = @top::@part1 }: (i2) -> (i2)
  %1 = seq.compreg %0, %clk { targetDesignPartition = @top::@part1 } : i2

  %2 = msft.instance @unit2 @Extern(%1) { targetDesignPartition = @top::@part1 }: (i2) -> (i2)

  msft.output %2: i2
}

// CHECK-LABEL: msft.module @top {} (%clk: i1) -> (out1: i2, out2: i2) {
// CHECK:    %part1.b.unit1.foo_x, %part1.b.seq.compreg.b.seq.compreg, %part1.b.unit2.foo_x, %part1.unit1.foo_x = msft.instance @part1 @dp(%b.unit1.foo_a, %b.seq.compreg.in0, %b.seq.compreg.in1, %b.unit2.foo_a, %c0_i2)  : (i2, i2, i1, i2, i2) -> (i2, i2, i2, i2)
// CHECK:    %b.x, %b.unit1.foo_a, %b.seq.compreg.in0, %b.seq.compreg.in1, %b.unit2.foo_a = msft.instance @b @B(%clk, %part1.b.unit1.foo_x, %part1.b.seq.compreg.b.seq.compreg, %part1.b.unit2.foo_x)  : (i1, i2, i2, i2) -> (i2, i2, i2, i1, i2)
// CHECK:    %c0_i2 = hw.constant 0 : i2
// CHECK:    msft.output %b.x, %part1.unit1.foo_x : i2, i2
// CHECK-LABEL: msft.module.extern @Extern(%foo_a: i2) -> (foo_x: i2)
// CHECK-LABEL: msft.module @B {} (%clk: i1, %unit1.foo_x: i2, %seq.compreg: i2, %unit2.foo_x: i2) -> (x: i2, unit1.foo_a: i2, seq.compreg.in0: i2, seq.compreg.in1: i1, unit2.foo_a: i2) {
// CHECK:    %c1_i2 = hw.constant 1 : i2
// CHECK:    msft.output %unit2.foo_x, %c1_i2, %unit1.foo_x, %clk, %seq.compreg : i2, i2, i2, i1, i2
// CHECK-LABEL: msft.module @dp {} (%b.unit1.foo_a: i2, %b.seq.compreg.in0: i2, %b.seq.compreg.in1: i1, %b.unit2.foo_a: i2, %unit1.foo_a: i2) -> (b.unit1.foo_x: i2, b.seq.compreg.b.seq.compreg: i2, b.unit2.foo_x: i2, unit1.foo_x: i2) {
// CHECK:    %b.unit1.foo_x = msft.instance @b.unit1 @Extern(%b.unit1.foo_a)  : (i2) -> i2
// CHECK:    %b.seq.compreg = seq.compreg %b.seq.compreg.in0, %b.seq.compreg.in1 : i2
// CHECK:    %b.unit2.foo_x = msft.instance @b.unit2 @Extern(%b.unit2.foo_a)  : (i2) -> i2
// CHECK:    %unit1.foo_x = msft.instance @unit1 @Extern(%unit1.foo_a)  : (i2) -> i2
// CHECK:    msft.output %b.unit1.foo_x, %b.seq.compreg, %b.unit2.foo_x, %unit1.foo_x : i2, i2, i2, i2

// CLEANUP-LABEL: msft.module @top {} (%clk: i1) -> (out1: i2, out2: i2) {
// CLEANUP:    %part1.b.unit2.foo_x, %part1.unit1.foo_x = msft.instance @part1 @dp(%b.unit1.foo_a, %clk, %c0_i2)  : (i2, i1, i2) -> (i2, i2)
// CLEANUP:    %b.unit1.foo_a = msft.instance @b @B()  : () -> i2
// CLEANUP:    %c0_i2 = hw.constant 0 : i2
// CLEANUP:    msft.output %part1.b.unit2.foo_x, %part1.unit1.foo_x : i2, i2
// CLEANUP-LABEL: msft.module.extern @Extern(%foo_a: i2) -> (foo_x: i2)
// CLEANUP-LABEL: msft.module @B {} () -> (unit1.foo_a: i2) {
// CLEANUP:    %c1_i2 = hw.constant 1 : i2
// CLEANUP:    msft.output %c1_i2 : i2
// CLEANUP-LABEL: msft.module @dp {} (%b.unit1.foo_a: i2, %b.seq.compreg.in1: i1, %unit1.foo_a: i2) -> (b.unit2.foo_x: i2, unit1.foo_x: i2)
// CLEANUP:    %b.unit1.foo_x = msft.instance @b.unit1 @Extern(%b.unit1.foo_a)  : (i2) -> i2
// CLEANUP:    %b.seq.compreg = seq.compreg %b.unit1.foo_x, %b.seq.compreg.in1 : i2
// CLEANUP:    %b.unit2.foo_x = msft.instance @b.unit2 @Extern(%b.seq.compreg)  : (i2) -> i2
// CLEANUP:    %unit1.foo_x = msft.instance @unit1 @Extern(%unit1.foo_a)  : (i2) -> i2
// CLEANUP:    msft.output %b.unit2.foo_x, %unit1.foo_x : i2, i2
