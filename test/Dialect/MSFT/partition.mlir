// RUN: circt-opt %s --msft-partition -verify-diagnostics -split-input-file | FileCheck %s

msft.module @top {} (%clk : i1) -> () {
  msft.partition @part1, "dp"

  msft.instance @b @B(%clk) : (i1) -> (i32)

  %c0 = hw.constant 0 : i1
  msft.instance @unit1 @Extern(%c0) { targetDesignPartition = @top::@part1 }: (i1) -> (i1)

  msft.output
}

msft.module.extern @Extern (%foo_a: i1) -> (foo_x: i1)

msft.module @B {} (%clk : i1) -> (x: i1)  {
  %c1 = hw.constant 1 : i1
  %0 = msft.instance @unit1 @Extern(%c1) { targetDesignPartition = @top::@part1 }: (i1) -> (i1)
  %1 = seq.compreg %0, %clk { targetDesignPartition = @top::@part1 } : i1

  %2 = msft.instance @unit2 @Extern(%1) { targetDesignPartition = @top::@part1 }: (i1) -> (i1)

  msft.output %2: i1
}

// CHECK-LABEL: msft.module @top {} (%clk: i1) {
// CHECK:         %part1.b.unit1.foo_x, %part1.b.seq.compreg.b.seq.compreg, %part1.b.unit2.foo_x, %part1.unit1.foo_x = msft.instance @part1 @dp(%b.unit1.foo_a, %b.seq.compreg.in0, %b.seq.compreg.in1, %b.unit2.foo_a, %false) : (i1, i1, i1, i1, i1) -> (i1, i1, i1, i1)
// CHECK:         %b.x, %b.unit1.foo_a, %b.seq.compreg.in0, %b.seq.compreg.in1, %b.unit2.foo_a = msft.instance @b @B(%clk, %part1.b.unit1.foo_x, %part1.b.seq.compreg.b.seq.compreg, %part1.b.unit2.foo_x)  : (i1, i1, i1, i1) -> (i1, i1, i1, i1, i1)
// CHECK:         %false = hw.constant false
// CHECK:         msft.output
// CHECK-LABEL: msft.module @B {} (%clk: i1, %unit1.foo_x: i1, %seq.compreg.out0: i1, %unit2.foo_x: i1) -> (x: i1, unit1.foo_a: i1, seq.compreg.in0: i1, seq.compreg.in1: i1, unit2.foo_a: i1) {
// CHECK:         %true = hw.constant true
// CHECK:         msft.output %unit2.foo_x, %true, %unit1.foo_x, %clk, %seq.compreg.out0 : i1, i1, i1, i1, i1
// CHECK-LABEL: msft.module @dp {} (%b.unit1.foo_a: i1, %b.seq.compreg.in0: i1, %b.seq.compreg.in1: i1, %b.unit2.foo_a: i1, %unit1.foo_a: i1) -> (b.unit1.foo_x: i1, b.seq.compreg.b.seq.compreg: i1, b.unit2.foo_x: i1, unit1.foo_x: i1) {
// CHECK:         %b.unit1.foo_x = msft.instance @b.unit1 @Extern(%b.unit1.foo_a)  : (i1) -> i1
// CHECK:         %b.seq.compreg = seq.compreg %b.seq.compreg.in0, %b.seq.compreg.in1 : i1
// CHECK:         %b.unit2.foo_x = msft.instance @b.unit2 @Extern(%b.unit2.foo_a)  : (i1) -> i1
// CHECK:         %unit1.foo_x = msft.instance @unit1 @Extern(%unit1.foo_a)  : (i1) -> i1
// CHECK:         msft.output %b.unit1.foo_x, %b.seq.compreg, %b.unit2.foo_x, %unit1.foo_x : i1, i1, i1, i1
