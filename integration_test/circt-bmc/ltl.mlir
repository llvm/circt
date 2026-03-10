// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

//  RUN: circt-bmc %s -b 10 --module ImplicationSanity --shared-libs=%libz3 | FileCheck %s --check-prefix=IMPLICATIONSANITY
//  IMPLICATIONSANITY: Bound reached with no violations!

hw.module @ImplicationSanity(in %i0: i1) {
  %impl = ltl.implication %i0, %i0 : i1, i1
  verif.assert %impl : !ltl.property
}