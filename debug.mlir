// CHECK-LABEL: @ChasePastValuesThroughControlFlow(
hw.module @ChasePastValuesThroughControlFlow(in %clock: i1, in %d: i42) {
  %c0_i42 = hw.constant 0 : i42
  %0 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NOT: llhd.process
  // CHECK: seq.compreg
  %1, %2 = llhd.process -> i42, i1 {
    %true = hw.constant true
    %false = hw.constant false
    cf.br ^bb1(%c0_i42, %false, %clock : i42, i1, i1)
  ^bb1(%3: i42, %4: i1, %5: i1):
    llhd.wait yield (%3, %4 : i42, i1), (%clock : i1), ^bb2(%5 : i1)
  ^bb2(%6: i1):
    %7 = comb.xor bin %6, %true : i1
    %8 = comb.and bin %7, %clock : i1  // posedge clock
    cf.cond_br %8, ^bb3, ^bb1(%c0_i42, %false, %clock : i42, i1, i1)
  ^bb3:
    cf.cond_br %clock, ^bb1(%d, %true, %true : i42, i1, i1), ^bb1(%d, %true, %false : i42, i1, i1)
  }
  %3 = llhd.sig %c0_i42 : i42
  llhd.drv %3, %1 after %0 if %2 : !hw.inout<i42>
}
