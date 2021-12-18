// RUN: circt-opt %s --msft-partition -verify-diagnostics -split-input-file

msft.module @top {} () -> () {
  msft.partition @part1, "dp"

  msft.instance @b @B() : () -> (i32)

  %c0 = hw.constant 0 : i1
  msft.instance @unit1 @Extern(%c0) { targetDesignPartition = @top::@part1 }: (i1) -> (i1)

  msft.output
}

msft.module.extern @Extern (%in: i1) -> (out: i1)

msft.module @B {} () -> (x: i1)  {
  %c1 = hw.constant 1 : i1
  %0 = msft.instance @unit1 @Extern(%c1) { targetDesignPartition = @top::@part1 }: (i1) -> (i1)

  msft.output %0: i1
}

// CHECK-LABEL:  hw.module @dp(%unit1_in: i1) -> (unit1_out: i1) {
// CHECK:    %unit1.out = hw.instance "unit1" sym @unit1 @Extern(in: %unit1_in: i1) -> (out: i1)
// CHECK:    hw.output %unit1.out : i1
