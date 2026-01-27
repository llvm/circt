// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-synth %s -o %t.before.mlir -convert-to-comb --output-longest-path=- | FileCheck %s --check-prefix=LONGEST_PATH_BEFORE
// RUN: circt-synth %s -enable-sop-balancing -o %t.after.mlir -convert-to-comb --output-longest-path=- | FileCheck %s --check-prefix=LONGEST_PATH_AFTER
// RUN: circt-lec %t.before.mlir %t.after.mlir -c1=add16 -c2=add16 --shared-libs=%libz3 | FileCheck %s --check-prefix=AND_INVERTER_LEC
// AND_INVERTER_LEC: c1 == c2
// LONGEST_PATH_BEFORE: delay: 12
// LONGEST_PATH_AFTER:  delay: 10
hw.module @add16(in %arg0: i16, in %arg1: i16, out add: i16) {
  %0 = comb.add %arg0, %arg1 : i16
  hw.output %0 : i16
}
