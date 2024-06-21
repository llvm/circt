// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s
// Tests that an Assert Property high level statement can be converted correctly

module {
  //CHECK:  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %a : i1)
  hw.module @test(in %clock : !seq.clock, in %reset : i1, in %a : i1) {
    //CHECK:  [[CLK:%.+]] = seq.from_clock %clock
    %0 = seq.from_clock %clock 

    //CHECK:  %false = hw.constant false
    //CHECK:  %true = hw.constant true
    //CHECK:  [[TMP:%.+]] = comb.or %reset, %hbr : i1
    //CHECK:  %hbr = seq.compreg [[TMP]], %clock powerOn %false : i1  
    %1 = verif.has_been_reset %0, sync %reset

    //CHECK:  [[TMP1:%.+]] = comb.xor %reset, %true : i1
    //CHECK:  [[TMP2:%.+]] = comb.and %hbr, [[TMP1]] : i1
    //CHECK:  verif.clocked_assert %a if [[TMP2]], posedge [[CLK]] : i1
    verif.clocked_assert %a if %1, posedge %0 : i1

    //CHECK:  hw.output
    hw.output 
  }
}

