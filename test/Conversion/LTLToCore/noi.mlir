// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s
// Tests that Non-Overlapping Implication is being lowered as expected
module {
  //CHECK:  hw.module @NOI(in %clock : !seq.clock, in %reset : i1, in %a : i1, in %b : i1)
  hw.module @NOI(in %clock : !seq.clock, in %reset : i1, in %a : i1, in %b : i1) {
    //CHECK:  %true = hw.constant true
    %true = hw.constant true
    //CHECK:  %0 = seq.from_clock %clock
    %0 = seq.from_clock %clock

    // Has been reset gets converted before all ltl operations
    //CHECK:  %false = hw.constant false
    //CHECK:  %true_0 = hw.constant true
    //CHECK:  %1 = comb.or %reset, %hbr : i1
    //CHECK:  %hbr = seq.compreg sym @hbr  %1, %clock powerOn %false : i1  
    //CHECK:  %2 = comb.xor %reset, %true_0 : i1
    //CHECK:  %3 = comb.and %hbr, %2 : i1
    //CHECK:  %4 = comb.xor bin %3, %true {sv.namehint = "disable"} : i1

    // NOI becomes this big block
    //CHECK:  %false_1 = hw.constant false
    //CHECK:  %true_2 = hw.constant true
    //CHECK:  %5 = comb.add %delay_, %true_2 : i1
    //CHECK:  %true_3 = hw.constant true
    //CHECK:  %6 = comb.icmp bin eq %delay_, %true_3 : i1
    //CHECK:  %7 = comb.mux %6, %true_3, %5 : i1
    //CHECK:  %delay_ = seq.compreg sym @delay_  %7, %clock reset %reset, %false_1 powerOn %false_1 : i1  
    //CHECK:  %false_4 = hw.constant false
    //CHECK:  %_0 = seq.compreg sym @_0  %true, %clock reset %reset, %false_4 powerOn %false_4 : i1  
    //CHECK:  %8 = comb.icmp bin ult %delay_, %true_3 : i1
    //CHECK:  %true_5 = hw.constant true
    //CHECK:  %9 = comb.xor %_0, %true_5 : i1
    //CHECK:  %10 = comb.or %9, %b : i1
    %1 = ltl.delay %true, 1, 0 : i1
    %2 = ltl.concat %a, %1 : i1, !ltl.sequence
    %3 = ltl.implication %2, %b : !ltl.sequence, i1

    // Already converted above
    %4 = verif.has_been_reset %0, sync %reset
    %5 = comb.xor bin %4, %true {sv.namehint = "disable"} : i1

    //CHECK:  %11 = comb.or %8, %10 : i1
    //CHECK:  %12 = comb.or %11, %reset : i1
    //CHECK:  %13 = comb.or %4, %12 : i1
    %6 = ltl.disable %3 if %5 : !ltl.property

    //CHECK:  sv.always posedge %0 {
    //CHECK:    sv.assert %13, immediate
    //CHECK:  }
    %7 = ltl.clock %6, posedge %0 : !ltl.property
    verif.assert %7 : !ltl.property

    //CHECK:  hw.output
    hw.output
  }
}
