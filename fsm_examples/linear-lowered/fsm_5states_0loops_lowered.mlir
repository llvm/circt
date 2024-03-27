module {
  hw.module @fsm5(in %clk : !seq.clock, in %rst : i1) {
    %c0_i3 = hw.constant 0 : i3
    %c1_i3 = hw.constant 1 : i3
    %c2_i3 = hw.constant 2 : i3
    %c3_i3 = hw.constant 3 : i3
    %c-4_i3 = hw.constant -4 : i3
    %c-3_i3 = hw.constant -3 : i3
    %state_reg = seq.compreg sym @state_reg  %21, %clk reset %rst, %c0_i3 powerOnValue %pon, %c0_i3: i3  
    %c0_i16 = hw.constant 0 : i16
    %x0 = seq.compreg sym @x0  %18, %clk reset %rst, %c0_i16 powerOnValue %pon, %c0_i16: i16  
    %c1_i16 = hw.constant 1 : i16
    %0 = comb.icmp bin eq %state_reg, %c0_i3 : i3
    %1 = comb.add bin %x0, %c1_i16 : i16
    %2 = comb.mux bin %0, %1, %18 : i16
    %3 = comb.mux bin %0, %c1_i3, %state_reg : i3
    %4 = comb.icmp bin eq %state_reg, %c1_i3 : i3
    %5 = comb.add bin %x0, %c1_i16 : i16
    %6 = comb.mux bin %4, %5, %2 : i16
    %7 = comb.mux bin %4, %c2_i3, %3 : i3
    %8 = comb.icmp bin eq %state_reg, %c2_i3 : i3
    %9 = comb.add bin %x0, %c1_i16 : i16
    %10 = comb.mux bin %8, %9, %6 : i16
    %11 = comb.mux bin %8, %c3_i3, %7 : i3
    %12 = comb.icmp bin eq %state_reg, %c3_i3 : i3
    %13 = comb.add bin %x0, %c1_i16 : i16
    %14 = comb.mux bin %12, %13, %10 : i16
    %15 = comb.mux bin %12, %c-4_i3, %11 : i3
    %16 = comb.icmp bin eq %state_reg, %c-4_i3 : i3
    %17 = comb.add bin %x0, %c1_i16 : i16
    %18 = comb.mux bin %16, %17, %14 : i16
    %19 = comb.mux bin %16, %c-3_i3, %15 : i3
    %20 = comb.icmp bin eq %state_reg, %c-3_i3 : i3
    %21 = comb.mux bin %20, %c-3_i3, %19 : i3
    // Assertion:
    %c-1_i1 = hw.constant -1 : i1
    %is_init_state = comb.icmp bin eq %state_reg, %c0_i3 : i3
    %cnt_zero = comb.icmp bin eq %x0, %c0_i16 : i16
    %not_init_state = comb.xor bin %is_init_state, %c-1_i1 : i1
    %prop = comb.or bin %not_init_state, %cnt_zero : i1
    sv.assert %prop : i1
    hw.output
  }
}

