// RUN: circt-opt %s --convert-comb-to-smt

// Regression test for: https://github.com/llvm/circt/issues/7096
hw.module @test(in %arg0 : i1, in %arg1 : i1, out out : i1) {
  %true = hw.constant true
  %false = hw.constant false
  %0 = comb.mux %arg1, %false, %true : i1
  %1 = comb.mux %arg1, %false, %true : i1
  %2 = comb.mux %arg0, %0, %1 : i1
  hw.output %2 : i1
}
