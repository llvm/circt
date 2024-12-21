// RUN: circt-opt %s --egg-synthesis='lib=%S/egg-synthesis-lib.json metric=Area' | FileCheck %s

module {
  hw.module @and(in %a : i2, in %b : i2, in %c : i2, out and : i2) {
    %0 = comb.extract %b from 1 : (i2) -> i1
    %1 = comb.extract %c from 1 : (i2) -> i1
    %2 = aig.and_inv %0, %1 : i1
    %3 = comb.extract %b from 0 : (i2) -> i1
    %4 = comb.extract %c from 0 : (i2) -> i1
    %5 = aig.and_inv %3, not %4 : i1
    %6 = comb.extract %a from 1 : (i2) -> i1
    %7 = aig.and_inv %6, %2 : i1
    %8 = comb.extract %a from 0 : (i2) -> i1
    %9 = aig.and_inv not %8, %5 : i1
    %10 = comb.concat %7, %9 : i1, i1
    hw.output %10 : i2
  }
}
