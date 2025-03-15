module {
  hw.module @AsyncReset(in %clock : i1, in %reset : i1, in %d : i32, out q : i32) {
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i32 = hw.constant 0 : i32
    %c42_i32 = hw.constant 42 : i32
    %false = hw.constant false
    %clock_0 = llhd.sig name "clock" %false : i1
    %2 = llhd.prb %clock_0 : !hw.inout<i1>
    %reset_1 = llhd.sig name "reset" %false : i1
    %3 = llhd.prb %reset_1 : !hw.inout<i1>
    %d_2 = llhd.sig name "d" %c0_i32 : i32
    %q = llhd.sig %c0_i32 : i32
    %4 = llhd.prb %d_2 : !hw.inout<i32>
    %5:2 = llhd.process -> i32, i1 {
      cf.br ^bb1(%c0_i32, %false : i32, i1)
    ^bb1(%7: i32, %8: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb3
      llhd.wait yield (%7, %8 : i32, i1), (%2, %3 : i1, i1), ^bb2(%2, %3 : i1, i1)
    ^bb2(%9: i1, %10: i1):  // pred: ^bb1
      %11 = comb.xor bin %9, %true : i1
      %12 = comb.and bin %11, %2 : i1
      %13 = comb.xor bin %10, %true : i1
      %14 = comb.and bin %13, %3 : i1
      %15 = comb.or bin %12, %14 : i1
      cf.cond_br %15, ^bb3, ^bb1(%c0_i32, %false : i32, i1)
    ^bb3:  // pred: ^bb2
      cf.cond_br %3, ^bb1(%c42_i32, %true : i32, i1), ^bb1(%4, %true : i32, i1)
    }
    llhd.drv %q, %5#0 after %0 if %5#1 : !hw.inout<i32>
    llhd.drv %clock_0, %clock after %1 : !hw.inout<i1>
    llhd.drv %reset_1, %reset after %1 : !hw.inout<i1>
    llhd.drv %d_2, %d after %1 : !hw.inout<i32>
    %6 = llhd.prb %q : !hw.inout<i32>
    hw.output %6 : i32
  }
}

