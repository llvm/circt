// REQUIRES: z3

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-synth -o %t.mlir
// RUN: circt-lec.sh %t.mlir %s -c1=add16 -c2=add16
hw.module @add16(in %arg0: i16, in %arg1: i16, in %arg2: i16,  out add: i16) {
  %0 = comb.add %arg0, %arg1, %arg2 : i16
  hw.output %0 : i16
}

// RUN: circt-lec.sh %t.mlir %s -c1=add33 -c2=add33
hw.module @add33(in %arg0: i33, in %arg1: i33, in %arg2: i33,  out add: i33) {
  %0 = comb.add %arg0, %arg1, %arg2 : i33
  hw.output %0 : i33
}
