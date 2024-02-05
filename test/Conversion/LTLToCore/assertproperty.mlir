// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  //CHECK:  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %0 "" : i1)
  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %8 : i1) {
    //CHECK:  %1 = seq.from_clock %clock
    %0 = seq.from_clock %clock 
    //CHECK:  %true = hw.constant true
    %true = hw.constant true

    //CHECK:  %true_0 = hw.constant true
    //CHECK:  %2 = comb.mux %reset, %true_0, %hbr : i1
    //CHECK:  %hbr = seq.compreg sym @hbr  %2, %clock : i1
    %9 = verif.has_been_reset %0, sync %reset

    //CHECK:  %3 = comb.xor bin %2, %true : i1
    %10 = comb.xor bin %9, %true : i1

    //CHECK:  %4 = hw.wire %0 : i1
    %11 = hw.wire %8 : i1

    //CHECK:  %5 = hw.wire %3 : i1
    %12 = hw.wire %10 : i1

    //CHECK:  %6 = comb.or %5, %4 : i1
    %13 = ltl.disable %11 if %12 : i1

    //CHECK:  sv.always posedge %1 {
    //CHECK:    sv.assert %6, immediate
    //CHECK:  }
    %14 = ltl.clock %13, posedge %0 : !ltl.property
    verif.assert %14 : !ltl.property

    //CHECK:  hw.output
    hw.output 
  }
}