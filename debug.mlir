module {
  hw.module @AsyncReset(in %clock : i1, in %reset : i1, in %enable : i1, in %d : i32, out q1 : i32, out q2 : i32, out q3 : i32, out q4 : i32) {
    %true = hw.constant true
    %true_0 = hw.constant true
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %true_1 = hw.constant true
    %c0_i32 = hw.constant 0 : i32
    %c42_i32 = hw.constant 42 : i32
    %false = hw.constant false
    %clock_2 = llhd.sig name "clock" %false : i1
    %1 = llhd.prb %clock_2 : !hw.inout<i1>
    %reset_3 = llhd.sig name "reset" %false : i1
    %2 = llhd.prb %reset_3 : !hw.inout<i1>
    %enable_4 = llhd.sig name "enable" %false : i1
    %3 = llhd.prb %enable_4 : !hw.inout<i1>
    %d_5 = llhd.sig name "d" %c0_i32 : i32
    %4 = llhd.prb %d_5 : !hw.inout<i32>
    %q1 = llhd.sig %c0_i32 : i32
    %q2 = llhd.sig %c0_i32 : i32
    %q3 = llhd.sig %c0_i32 : i32
    %q4 = llhd.sig %c0_i32 : i32
    %5 = llhd.constant_time <0ns, 1d, 0e>
    %6:2 = llhd.process -> i32, i1 {
      %c0_i32_6 = hw.constant 0 : i32
      %false_7 = hw.constant false
      cf.br ^bb1(%1, %2, %c0_i32_6, %false_7 : i1, i1, i32, i1)
    ^bb1(%17: i1, %18: i1, %19: i32, %20: i1):  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      llhd.wait yield (%19, %20 : i32, i1), (%1, %2 : i1, i1), ^bb2(%17, %18 : i1, i1)
    ^bb2(%21: i1, %22: i1):  // pred: ^bb1
      %23 = comb.xor bin %21, %true_1 : i1
      %24 = comb.and bin %23, %1 : i1
      %25 = comb.xor bin %22, %true_1 : i1
      %26 = comb.and bin %25, %2 : i1
      %27 = comb.or bin %24, %26 : i1
      %c0_i32_8 = hw.constant 0 : i32
      %false_9 = hw.constant false
      cf.cond_br %27, ^bb3, ^bb1(%1, %2, %c0_i32_8, %false_9 : i1, i1, i32, i1)
    ^bb3:  // pred: ^bb2
      cf.cond_br %2, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      cf.br ^bb1(%1, %2, %c42_i32, %true_0 : i1, i1, i32, i1)
    ^bb5:  // pred: ^bb3
      %c0_i32_10 = hw.constant 0 : i32
      %false_11 = hw.constant false
      cf.cond_br %3, ^bb6, ^bb1(%1, %2, %c0_i32_10, %false_11 : i1, i1, i32, i1)
    ^bb6:  // pred: ^bb5
      %true_12 = hw.constant true
      cf.br ^bb1(%1, %2, %4, %true_12 : i1, i1, i32, i1)
    }
    llhd.drv %q1, %6#0 after %5 if %6#1 : !hw.inout<i32>
    %7 = llhd.constant_time <0ns, 1d, 0e>
    %8:2 = llhd.process -> i32, i1 {
      %c0_i32_6 = hw.constant 0 : i32
      %false_7 = hw.constant false
      cf.br ^bb1(%1, %2, %c0_i32_6, %false_7 : i1, i1, i32, i1)
    ^bb1(%17: i1, %18: i1, %19: i32, %20: i1):  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      llhd.wait yield (%19, %20 : i32, i1), (%1, %2 : i1, i1), ^bb2(%17, %18 : i1, i1)
    ^bb2(%21: i1, %22: i1):  // pred: ^bb1
      %23 = comb.xor bin %21, %true_1 : i1
      %24 = comb.and bin %23, %1 : i1
      %25 = comb.xor bin %22, %true_1 : i1
      %26 = comb.and bin %25, %2 : i1
      %27 = comb.or bin %24, %26 : i1
      %c0_i32_8 = hw.constant 0 : i32
      %false_9 = hw.constant false
      cf.cond_br %27, ^bb3, ^bb1(%1, %2, %c0_i32_8, %false_9 : i1, i1, i32, i1)
    ^bb3:  // pred: ^bb2
      cf.cond_br %2, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      cf.br ^bb1(%1, %2, %c42_i32, %true : i1, i1, i32, i1)
    ^bb5:  // pred: ^bb3
      %true_10 = hw.constant true
      cf.br ^bb1(%1, %2, %4, %true_10 : i1, i1, i32, i1)
    }
    llhd.drv %q2, %8#0 after %7 if %8#1 : !hw.inout<i32>
    %9 = llhd.constant_time <0ns, 1d, 0e>
    %10:2 = llhd.process -> i32, i1 {
      %c0_i32_6 = hw.constant 0 : i32
      %false_7 = hw.constant false
      cf.br ^bb1(%1, %2, %c0_i32_6, %false_7 : i1, i1, i32, i1)
    ^bb1(%17: i1, %18: i1, %19: i32, %20: i1):  // 3 preds: ^bb0, ^bb2, ^bb3
      llhd.wait yield (%19, %20 : i32, i1), (%1, %2 : i1, i1), ^bb2(%17, %18 : i1, i1)
    ^bb2(%21: i1, %22: i1):  // pred: ^bb1
      %23 = comb.xor bin %21, %true_1 : i1
      %24 = comb.and bin %23, %1 : i1
      %25 = comb.xor bin %22, %true_1 : i1
      %26 = comb.and bin %25, %2 : i1
      %27 = comb.or bin %24, %26 : i1
      %c0_i32_8 = hw.constant 0 : i32
      %false_9 = hw.constant false
      cf.cond_br %27, ^bb3, ^bb1(%1, %2, %c0_i32_8, %false_9 : i1, i1, i32, i1)
    ^bb3:  // pred: ^bb2
      %true_10 = hw.constant true
      cf.br ^bb1(%1, %2, %4, %true_10 : i1, i1, i32, i1)
    }
    llhd.drv %q3, %10#0 after %9 if %10#1 : !hw.inout<i32>
    %11 = llhd.constant_time <0ns, 1d, 0e>
    %12:2 = llhd.process -> i32, i1 {
      %c0_i32_6 = hw.constant 0 : i32
      %false_7 = hw.constant false
      cf.br ^bb1(%1, %2, %c0_i32_6, %false_7 : i1, i1, i32, i1)
    ^bb1(%17: i1, %18: i1, %19: i32, %20: i1):  // 3 preds: ^bb0, ^bb2, ^bb3
      llhd.wait yield (%19, %20 : i32, i1), (%1, %2 : i1, i1), ^bb2(%17, %18 : i1, i1)
    ^bb2(%21: i1, %22: i1):  // pred: ^bb1
      %23 = comb.xor bin %21, %true_1 : i1
      %24 = comb.and bin %23, %1 : i1
      %25 = comb.xor bin %22, %true_1 : i1
      %26 = comb.and bin %25, %2 : i1
      %27 = comb.or bin %24, %26 : i1
      %c0_i32_8 = hw.constant 0 : i32
      %false_9 = hw.constant false
      cf.cond_br %27, ^bb3, ^bb1(%1, %2, %c0_i32_8, %false_9 : i1, i1, i32, i1)
    ^bb3:  // pred: ^bb2
      %true_10 = hw.constant true
      %28 = comb.mux %2, %c42_i32, %4 : i32
      cf.br ^bb1(%1, %2, %28, %true_10 : i1, i1, i32, i1)
    }
    llhd.drv %q4, %12#0 after %11 if %12#1 : !hw.inout<i32>
    llhd.drv %clock_2, %clock after %0 : !hw.inout<i1>
    llhd.drv %reset_3, %reset after %0 : !hw.inout<i1>
    llhd.drv %enable_4, %enable after %0 : !hw.inout<i1>
    llhd.drv %d_5, %d after %0 : !hw.inout<i32>
    %13 = llhd.prb %q1 : !hw.inout<i32>
    %14 = llhd.prb %q2 : !hw.inout<i32>
    %15 = llhd.prb %q3 : !hw.inout<i32>
    %16 = llhd.prb %q4 : !hw.inout<i32>
    hw.output %13, %14, %15, %16 : i32, i32, i32, i32
  }
}

