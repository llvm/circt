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

// CHECK-LABEL:  hw.module @dp
// CHECK:          msft.instance @b.unit1 @Extern
// CHECK:          msft.instance @b.unit2 @Extern
// CHECK:          msft.instance @unit1 @Extern
