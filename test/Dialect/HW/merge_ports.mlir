// RUN: circt-opt --hw-merge-identical-ports %s | FileCheck %s

// Base case - everything is driven by the same thing so this should heavily
// optimize the instantiated modules.

hw.module @DualAnd(%in0 : i1, %in1 : i1) -> (out : i1) {
  %out = comb.and %in0, %in1 : i1
  hw.output %out : i1
}

hw.module @IdenticalOutput(%in0 : i1) -> (out0 : i1, out1 : i1) {
  hw.output %in0, %in0 : i1, i1
}


hw.module @top(%in0 : i1) -> (out0 : i1, out1 : i1, out2 : i1) {
  %out0, %out1 = hw.instance "a" @IdenticalOutput(in0 : %in0 : i1) -> (out0: i1, out1: i1)
  %out2 = hw.instance "b" @DualAnd(in0 : %in0 : i1, in1 : %in0 : i1) -> (out: i1)
  hw.output %out0, %out1, %out2 : i1, i1, i1
}

// -----

// Errant case - base case + another top module which is less optimization
// friendly.

hw.module @DualAnd(%in0 : i1, %in1 : i1) -> (out : i1) {
  %out = comb.and %in0, %in1 : i1
  hw.output %out : i1
}

hw.module @top1(%in0 : i1) -> (out0 : i1, out1 : i1, out2 : i1) {
  %out = hw.instance "b" @DualAnd(in0 : %in0 : i1, in1 : %in0 : i1) -> (out: i1)
  hw.output %out0, %out1, %out2 : i1, i1, i1
}

hw.module @top2(%in0 : i1, %in1 : i1) -> (out0 : i1, out1 : i1) {
  %out = hw.instance "b" @DualAnd(in0 : %in0 : i1, in1 : %i1 : i1) -> (out: i1)
  hw.output %out, %out : i1
}

// -----

// Check that the merge_identical_ports.ignore attribute is respected.

hw.module @DualAnd(%in0 : i1, %in1 : i1) -> (out : i1) {"hw.merge_identical_ports.ignore"} {
  %out = comb.and %in0, %in1 : i1
  hw.output %out : i1
}

hw.module @top(%in0 : i1) -> (out0 : i1, out1 : i1, out2 : i1) {
  %out2 = hw.instance "b" @DualAnd(in0 : %in0 : i1, in1 : %in0 : i1) -> (out: i1)
  hw.output %out0, %out1, %out2 : i1, i1, i1
}

// -----

// Check that things work when there is multiple disjoint set of in and outputs that are identical.
