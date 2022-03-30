// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module @SAExample1 (%clk : i1) -> (out: !hw.array<2 x array<3 x i8>>) {
  %c0_8 = hw.constant 0 : i8
  %c0_12 = hw.constant 0 : i8
  %rowInputs = hw.array_create %c0_8, %c0_8 : i8
  %colInputs = hw.array_create %c0_12, %c0_12, %c0_12 : i8

  %peOuts = msft.systolic.array [2 x %rowInputs : i8]
                                [3 x %colInputs : i8]
                                -> (i8)
    pe (%row, %col) {
      %sum = comb.add %row, %col : i8
      %sumDelay1 = seq.compreg %sum, %clk : i8
      msft.pe.output %sumDelay1 : i8
    }
  hw.output %peOuts : !hw.array<2 x array<3 x i8>>
}
