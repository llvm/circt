module {
  hw.module @AsyncReset(in %clock : i1, in %reset : i1, in %enable : i1, in %d : i32, out q1 : i32, out q2 : i32, out q3 : i32) {
    %false = arith.constant false
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %true = hw.constant true
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i32 = hw.constant 0 : i32
    %c42_i32 = hw.constant 42 : i32
    %false_0 = hw.constant false
    %clock_1 = llhd.sig name "clock" %false_0 : i1
    %2 = llhd.prb %clock_1 : !hw.inout<i1>
    %reset_2 = llhd.sig name "reset" %false_0 : i1
    %3 = llhd.prb %reset_2 : !hw.inout<i1>
    %enable_3 = llhd.sig name "enable" %false_0 : i1
    %d_4 = llhd.sig name "d" %c0_i32 : i32
    %q1 = llhd.sig %c0_i32 : i32
    %q2 = llhd.sig %c0_i32 : i32
    %q3 = llhd.sig %c0_i32 : i32
    %4 = llhd.prb %enable_3 : !hw.inout<i1>
    %5 = llhd.prb %d_4 : !hw.inout<i32>
    %6:2 = llhd.process -> i32, i1 {
      cf.br ^bb1(%3, %c0_i32, %false_0 : i1, i32, i1)
    ^bb1(%18: i1, %19: i32, %20: i1):  // 5 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb4
      llhd.wait yield (%19, %20 : i32, i1), (%2, %3 : i1, i1), ^bb2(%2, %18 : i1, i1)
    ^bb2(%21: i1, %22: i1):  // pred: ^bb1
      %23 = comb.xor bin %21, %true : i1
      %24 = comb.and bin %23, %2 : i1
      %25 = comb.xor bin %22, %true : i1
      %26 = comb.and bin %25, %3 : i1
      %27 = comb.or bin %24, %26 : i1
      cf.cond_br %27, ^bb3, ^bb1(%3, %c0_i32, %false_0 : i1, i32, i1)
    ^bb3:  // pred: ^bb2
      cf.cond_br %3, ^bb1(%3, %c42_i32, %true : i1, i32, i1), ^bb4
    ^bb4:  // pred: ^bb3
      cf.cond_br %4, ^bb1(%false, %5, %true : i1, i32, i1), ^bb1(%false, %c0_i32, %false_0 : i1, i32, i1)
    }
    llhd.drv %q1, %6#0 after %0 if %6#1 : !hw.inout<i32>
    %7 = llhd.prb %clock_1 : !hw.inout<i1>
    %8 = llhd.prb %reset_2 : !hw.inout<i1>
    %9 = llhd.prb %d_4 : !hw.inout<i32>
    %10:2 = llhd.process -> i32, i1 {
      cf.br ^bb1(%c0_i32, %false_0 : i32, i1)
    ^bb1(%18: i32, %19: i1):  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb3
      llhd.wait yield (%18, %19 : i32, i1), (%2, %3 : i1, i1), ^bb2(%2, %3 : i1, i1)
    ^bb2(%20: i1, %21: i1):  // pred: ^bb1
      %22 = comb.xor bin %20, %true : i1
      %23 = comb.and bin %22, %2 : i1
      %24 = comb.xor bin %21, %true : i1
      %25 = comb.and bin %24, %3 : i1
      %26 = comb.or bin %23, %25 : i1
      cf.cond_br %26, ^bb3, ^bb1(%c0_i32, %false_0 : i32, i1)
    ^bb3:  // pred: ^bb2
      cf.cond_br %3, ^bb1(%c42_i32, %true : i32, i1), ^bb1(%9, %true : i32, i1)
    }
    llhd.drv %q2, %10#0 after %0 if %10#1 : !hw.inout<i32>
    %11 = llhd.prb %clock_1 : !hw.inout<i1>
    %12 = llhd.prb %reset_2 : !hw.inout<i1>
    %13 = llhd.prb %d_4 : !hw.inout<i32>
    %14:2 = llhd.process -> i32, i1 {
      cf.br ^bb1(%c0_i32, %false_0 : i32, i1)
    ^bb1(%18: i32, %19: i1):  // 3 preds: ^bb0, ^bb2, ^bb2
      llhd.wait yield (%18, %19 : i32, i1), (%2, %3 : i1, i1), ^bb2(%2, %3 : i1, i1)
    ^bb2(%20: i1, %21: i1):  // pred: ^bb1
      %22 = comb.xor bin %20, %true : i1
      %23 = comb.and bin %22, %2 : i1
      %24 = comb.xor bin %21, %true : i1
      %25 = comb.and bin %24, %3 : i1
      %26 = comb.or bin %23, %25 : i1
      cf.cond_br %26, ^bb1(%13, %true : i32, i1), ^bb1(%c0_i32, %false_0 : i32, i1)
    }
    llhd.drv %q3, %14#0 after %0 if %14#1 : !hw.inout<i32>
    llhd.drv %clock_1, %clock after %1 : !hw.inout<i1>
    llhd.drv %reset_2, %reset after %1 : !hw.inout<i1>
    llhd.drv %enable_3, %enable after %1 : !hw.inout<i1>
    llhd.drv %d_4, %d after %1 : !hw.inout<i32>
    %15 = llhd.prb %q1 : !hw.inout<i32>
    %16 = llhd.prb %q2 : !hw.inout<i32>
    %17 = llhd.prb %q3 : !hw.inout<i32>
    hw.output %15, %16, %17 : i32, i32, i32
  }
}

