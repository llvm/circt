// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s
// Tests that Non-Overlapping Implication is being lowered as expected
module {
  //CHECK:  hw.module @OI(in %clock : !seq.clock, in %reset : i1, in %a : i1, in %b : i1)
  hw.module @OI(in %clock : !seq.clock, in %reset : i1, in %a : i1, in %b : i1) {
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

    // The implication simply becomes a logical a -> b, i.e. !a || b
    //CHECK:  %true_1 = hw.constant true
    //CHECK:  %5 = comb.xor %a, %true_1 : i1
    //CHECK:  %6 = comb.or %5, %b : i1
    %1 = ltl.implication %a, %b : i1, i1

    %2 = verif.has_been_reset %0, sync %reset
    %3 = comb.xor bin %2, %true {sv.namehint = "disable"} : i1

    //CHECK:  %7 = comb.or %4, %6 : i1
    %4 = ltl.disable %1 if %3 : !ltl.property

    //CHECK:  sv.always posedge %0 {
    //CHECK:    sv.assert %7, immediate
    //CHECK:  }
    %5 = ltl.clock %4, posedge %0 : !ltl.property
    verif.assert %5 : !ltl.property

    //CHECK:  hw.output
    hw.output
  }
}
