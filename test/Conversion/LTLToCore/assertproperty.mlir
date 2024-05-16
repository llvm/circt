// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s
// Tests that an Assert Property high level statement can be converted correctly

module {
  //CHECK:  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %0 "" : i1)
  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %8 : i1) {
    //CHECK:  %1 = seq.from_clock %clock
    %0 = seq.from_clock %clock 
    //CHECK:  %true = hw.constant true
    %true = hw.constant true

    //CHECK:  %false = hw.constant false
    //CHECK:  %true_0 = hw.constant true
    //CHECK:  %2 = comb.or %reset, %hbr : i1
    //CHECK:  %hbr = seq.compreg %2, %clock powerOn %false : i1  
    %9 = verif.has_been_reset %0, sync %reset

    //CHECK:  %3 = comb.xor %reset, %true_0 : i1
    //CHECK:  %4 = comb.and %hbr, %3 : i1
    //CHECK:  %5 = comb.xor bin %4, %true : i1
    %10 = comb.xor bin %9, %true : i1

    //CHECK:  %6 = hw.wire %5 : i1
    %12 = hw.wire %10 : i1

    //CHECK:  %7 = comb.or %6, %0 : i1
    %13 = ltl.disable %8 if %12 : !ltl.disabled_property

    //CHECK:  sv.always posedge %1 {
    //CHECK:    sv.assert %7, immediate
    //CHECK:  }
    %14 = ltl.clock %13, posedge %0 : !ltl.clocked_disabled_property
    verif.assert %14 : !ltl.clocked_disabled_property

    //CHECK:  hw.output
    hw.output 
  }
}

