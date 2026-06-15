// REQUIRES: z3

// RUN: circt-opt %s --convert-comb-to-synth -o %t.mlir
// RUN: circt-lec.sh %t.mlir %s -c1=add -c2=add
hw.module @add(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 : i4
  hw.output %0 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=add_ripple_carry -c2=add_ripple_carry
hw.module @add_ripple_carry(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.test.arch = "RIPPLE-CARRY"} : i4
  hw.output %0 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=add_sklanskey -c2=add_sklanskey
hw.module @add_sklanskey(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.test.arch = "SKLANSKEY"} : i4
  hw.output %0 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=add_kogge_stone -c2=add_kogge_stone
hw.module @add_kogge_stone(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.test.arch = "KOGGE-STONE"} : i4
  hw.output %0 : i4
}

// RUN: circt-lec.sh %t.mlir %s -c1=add_brent_kung -c2=add_brent_kung
hw.module @add_brent_kung(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.test.arch = "BRENT-KUNG"} : i4
  hw.output %0 : i4
}

