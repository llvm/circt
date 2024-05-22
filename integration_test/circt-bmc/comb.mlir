// These tests will be only enabled if circt-mc is built.


//  RUN: circt-bmc %s -b 10 --module OrCommutes --shared-libs=%libz3 | FileCheck %s --check-prefix=ORCOMMUTES
//  ORCOMMUTES: Success!

hw.module @OrCommutes(in %i0: i1, in %i1: i1) {
  %or0 = comb.or bin %i0, %i1 : i1
  %or1 = comb.or bin %i1, %i0 : i1
  // Condition
  %cond = comb.icmp bin eq %or0, %or1 : i1
  verif.assert %cond : i1
}

//  RUN: circt-bmc %s -b 10 --module demorgan --shared-libs=%libz3 | FileCheck %s --check-prefix=DEMORGAN
//  DEMORGAN: Success!

hw.module @demorgan(in %i0: i1, in %i1: i1) {
  %c1 = hw.constant 1 : i1
  %ni0 = comb.xor bin %i0, %c1 : i1
  %ni1 = comb.xor bin %i1, %c1 : i1
  %or = comb.or bin %ni0, %ni1 : i1
  // Condition
  %and = comb.and bin %i0, %i1 : i1
  %nand = comb.xor bin %and, %c1 : i1
  %cond = comb.icmp bin eq %or, %nand : i1
  verif.assert %cond : i1
}
