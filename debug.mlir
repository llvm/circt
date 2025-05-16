hw.module private @ResetValueFromStaticSignal(in %clk : i1, in %reset : i1, in %d : i42) {
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i42 = hw.constant 0 : i42
  %c9001_i42 = hw.constant 9001 : i42
  %false = hw.constant false
  %true = hw.constant true
  %init = llhd.sig %c0_i42 : i42
  llhd.drv %init, %c9001_i42 after %1 : !hw.inout<i42>
  %q = llhd.sig %c0_i42 : i42
  %2 = llhd.prb %init : !hw.inout<i42>
  %3:2 = llhd.process -> i42, i1 {
    cf.br ^bb1(%c0_i42, %false : i42, i1)
  ^bb1(%4: i42, %5: i1):
    llhd.wait yield (%4, %5 : i42, i1), (%clk, %reset : i1, i1), ^bb2(%clk, %reset : i1, i1)
  ^bb2(%6: i1, %7: i1):
    %8 = comb.xor bin %6, %true : i1
    %9 = comb.and bin %8, %clk : i1
    %10 = comb.xor bin %7, %true : i1
    %11 = comb.and bin %10, %reset : i1
    %12 = comb.or bin %9, %11 : i1
    cf.cond_br %12, ^bb3, ^bb1(%c0_i42, %false : i42, i1)
  ^bb3:
    cf.cond_br %reset, ^bb1(%2, %true : i42, i1), ^bb1(%d, %true : i42, i1)
  }
  llhd.drv %q, %3#0 after %0 if %3#1 : !hw.inout<i42>
}
