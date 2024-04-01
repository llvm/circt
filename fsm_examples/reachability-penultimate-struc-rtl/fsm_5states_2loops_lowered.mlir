module {
  hw.module @fsm5(in %clk : !seq.clock, in %rst : i1) {
    // Cover statement
    %is_second_state = comb.icmp bin eq %c3_i16, %state_reg : i16
    verif.cover %is_second_state : i1
    %c0_i1 = hw.constant 0 : i1
    %no_rst = comb.icmp bin eq %c0_i1, %rst : i1
    verif.assume %no_rst : i1
    %c0_i16 = hw.constant 0 : i16
    %c1_i16 = hw.constant 1 : i16
    %c2_i16 = hw.constant 2 : i16
    %c3_i16 = hw.constant 3 : i16
    %c4_i16 = hw.constant 4 : i16
    %c5_i16 = hw.constant 5 : i16
    %state_reg = seq.compreg sym @state_reg  %45, %clk reset %rst, %c0_i16 : i16  
    %c0_i16_0 = hw.constant 0 : i16
    %x0 = seq.compreg sym @x0  %42, %clk reset %rst, %c0_i16_0 : i16  
    %c3_i16_1 = hw.constant 3 : i16
    %c6_i16 = hw.constant 6 : i16
    %c1_i16_2 = hw.constant 1 : i16
    %0 = comb.icmp bin eq %state_reg, %c0_i16 : i16
    %1 = comb.add bin %x0, %c1_i16_2 : i16
    %2 = comb.mux bin %0, %1, %42 : i16
    %3 = comb.mux bin %0, %c1_i16, %state_reg : i16
    %4 = comb.icmp bin eq %state_reg, %c1_i16 : i16
    %5 = comb.add bin %x0, %c1_i16_2 : i16
    %6 = comb.mux bin %4, %5, %2 : i16
    %7 = comb.mux bin %4, %c2_i16, %3 : i16
    %8 = comb.icmp bin eq %state_reg, %c2_i16 : i16
    %9 = comb.add bin %x0, %c1_i16_2 : i16
    %10 = comb.icmp bin uge %x0, %c3_i16_1 : i16
    %11 = comb.icmp bin eq %state_reg, %c2_i16 : i16
    %12 = comb.add bin %x0, %c1_i16_2 : i16
    %13 = comb.icmp bin ult %x0, %c3_i16_1 : i16
    %14 = comb.icmp bin eq %state_reg, %c2_i16 : i16
    %15 = comb.mux bin %14, %c2_i16, %7 : i16
    %16 = comb.mux bin %13, %c0_i16, %c2_i16 : i16
    %17 = comb.and bin %13, %11 : i1
    %18 = comb.mux bin %17, %12, %6 : i16
    %19 = comb.mux bin %11, %16, %15 : i16
    %20 = comb.mux bin %10, %c3_i16, %16 : i16
    %21 = comb.and bin %10, %8 : i1
    %22 = comb.mux bin %21, %9, %18 : i16
    %23 = comb.mux bin %8, %20, %19 : i16
    %24 = comb.icmp bin eq %state_reg, %c3_i16 : i16
    %25 = comb.add bin %x0, %c1_i16_2 : i16
    %26 = comb.mux bin %24, %25, %22 : i16
    %27 = comb.mux bin %24, %c4_i16, %23 : i16
    %28 = comb.icmp bin eq %state_reg, %c4_i16 : i16
    %29 = comb.add bin %x0, %c1_i16_2 : i16
    %30 = comb.icmp bin uge %x0, %c6_i16 : i16
    %31 = comb.icmp bin eq %state_reg, %c4_i16 : i16
    %32 = comb.add bin %x0, %c1_i16_2 : i16
    %33 = comb.icmp bin ult %x0, %c6_i16 : i16
    %34 = comb.icmp bin eq %state_reg, %c4_i16 : i16
    %35 = comb.mux bin %34, %c4_i16, %27 : i16
    %36 = comb.mux bin %33, %c0_i16, %c4_i16 : i16
    %37 = comb.and bin %33, %31 : i1
    %38 = comb.mux bin %37, %32, %26 : i16
    %39 = comb.mux bin %31, %36, %35 : i16
    %40 = comb.mux bin %30, %c5_i16, %36 : i16
    %41 = comb.and bin %30, %28 : i1
    %42 = comb.mux bin %41, %29, %38 : i16
    %43 = comb.mux bin %28, %40, %39 : i16
    %44 = comb.icmp bin eq %state_reg, %c5_i16 : i16
    %45 = comb.mux bin %44, %c5_i16, %43 : i16
    hw.output
  }
}

