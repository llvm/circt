// RUN: circt-opt --lower-seq-fifo --cse %s | FileCheck %s --implicit-check-not=seq.fifo

// Every status output is a register, stepped from compares of registers against
// constants, and the enables only select between values computed from
// registers. The count register lags a cycle behind, stepped by registered
// copies of the enables, so the enables do not reach it. Reading while empty and
// writing while full are protocol violations: they are not guarded against, but
// asserted on outside of reset.

// A depth which is neither a power of two nor one short of one: the pointers
// wrap on a compare of the current pointer.
// CHECK-LABEL: hw.module @fifo1(
// CHECK:         %fifo_count_lag = seq.compreg sym @fifo_count_lag [[COUNT:%.+]], %clk reset %rst, %c0_i3 : i3
// The memory is not reset; the FIFO leaves reset empty, so nothing written
// during reset is read.
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    %fifo_mem = seq.hlmem @fifo_mem %clk, %false : <6xi32>
// CHECK-NEXT:    %fifo_rd_addr = seq.compreg sym @fifo_rd_addr [[RD_NEXT:%.+]], %clk reset %rst, %c0_i3 : i3
// CHECK-NEXT:    %fifo_wr_addr = seq.compreg sym @fifo_wr_addr [[WR_NEXT:%.+]], %clk reset %rst, %c0_i3 : i3
// CHECK-NEXT:    %fifo_mem_rdata = seq.read %fifo_mem[%fifo_rd_addr] rden %rdEn {latency = 0 : i64} : !seq.hlmem<6xi32>
// CHECK-NEXT:    seq.write %fifo_mem[%fifo_wr_addr] %in wren %wrEn {latency = 1 : i64} : !seq.hlmem<6xi32>
// CHECK:         [[NOT_RD:%.+]] = comb.xor %rdEn, %true
// CHECK-NEXT:    [[UP:%.+]] = comb.and %wrEn, [[NOT_RD]]
// CHECK-NEXT:    [[NOT_WR:%.+]] = comb.xor %wrEn, %true
// CHECK-NEXT:    [[DOWN:%.+]] = comb.and %rdEn, [[NOT_WR]]
// The count register is stepped only by registers.
// CHECK:         %fifo_count_up = seq.compreg sym @fifo_count_up [[UP]], %clk reset %rst, %false : i1
// CHECK-NEXT:    %fifo_count_down = seq.compreg sym @fifo_count_down [[DOWN]], %clk reset %rst, %false : i1
// CHECK:         [[INC:%.+]] = comb.add %fifo_count_lag, %c1_i3
// CHECK-NEXT:    [[DEC:%.+]] = comb.sub %fifo_count_lag, %c1_i3
// CHECK-NEXT:    [[COUNT_DOWN:%.+]] = comb.mux %fifo_count_down, [[DEC]], %fifo_count_lag
// CHECK-NEXT:    [[COUNT]] = comb.mux %fifo_count_up, [[INC]], [[COUNT_DOWN]] {sv.namehint = "fifo_count"}
// Full is set going up from a count of 5: the register is compared against 4, 6
// or 5, depending on the registered step.
// CHECK-NEXT:    %fifo_full = seq.compreg sym @fifo_full [[FULL_NEXT:%.+]], %clk reset %rst, %false : i1
// CHECK-NEXT:    %c-4_i3 = hw.constant -4 : i3
// CHECK-NEXT:    [[LAG_4:%.+]] = comb.icmp eq %fifo_count_lag, %c-4_i3
// CHECK-NEXT:    %c-2_i3 = hw.constant -2 : i3
// CHECK-NEXT:    [[LAG_6:%.+]] = comb.icmp eq %fifo_count_lag, %c-2_i3
// CHECK-NEXT:    %c-3_i3 = hw.constant -3 : i3
// CHECK-NEXT:    [[LAG_5:%.+]] = comb.icmp eq %fifo_count_lag, %c-3_i3
// CHECK-NEXT:    [[AT_5_DOWN:%.+]] = comb.mux %fifo_count_down, [[LAG_6]], [[LAG_5]]
// CHECK-NEXT:    [[AT_5:%.+]] = comb.mux %fifo_count_up, [[LAG_4]], [[AT_5_DOWN]]
// CHECK-NEXT:    [[FULL_DOWN:%.+]] = comb.mux [[DOWN]], %false, %fifo_full
// CHECK-NEXT:    [[FULL_NEXT]] = comb.mux [[UP]], [[AT_5]], [[FULL_DOWN]]
// Empty is set going down from a count of 1.
// CHECK-NEXT:    %fifo_empty = seq.compreg sym @fifo_empty [[EMPTY_NEXT:%.+]], %clk reset %rst, %true : i1
// CHECK-NEXT:    [[LAG_0:%.+]] = comb.icmp eq %fifo_count_lag, %c0_i3
// CHECK-NEXT:    %c2_i3 = hw.constant 2 : i3
// CHECK-NEXT:    [[LAG_2:%.+]] = comb.icmp eq %fifo_count_lag, %c2_i3
// CHECK-NEXT:    [[LAG_1:%.+]] = comb.icmp eq %fifo_count_lag, %c1_i3
// CHECK-NEXT:    [[AT_1_DOWN:%.+]] = comb.mux %fifo_count_down, [[LAG_2]], [[LAG_1]]
// CHECK-NEXT:    [[AT_1:%.+]] = comb.mux %fifo_count_up, [[LAG_0]], [[AT_1_DOWN]]
// CHECK-NEXT:    [[EMPTY_DOWN:%.+]] = comb.mux [[DOWN]], [[AT_1]], %fifo_empty
// CHECK-NEXT:    [[EMPTY_NEXT]] = comb.mux [[UP]], %false, [[EMPTY_DOWN]]
// CHECK-NEXT:    [[WR_INC:%.+]] = comb.add %fifo_wr_addr, %c1_i3
// CHECK-NEXT:    [[WR_WRAP:%.+]] = comb.icmp eq %fifo_wr_addr, %c-3_i3
// CHECK-NEXT:    [[WR_STEP:%.+]] = comb.mux [[WR_WRAP]], %c0_i3, [[WR_INC]]
// CHECK-NEXT:    [[WR_NEXT]] = comb.mux %wrEn, [[WR_STEP]], %fifo_wr_addr {sv.namehint = "fifo_wr_addr_next"}
// CHECK-NEXT:    [[RD_INC:%.+]] = comb.add %fifo_rd_addr, %c1_i3
// CHECK-NEXT:    [[RD_WRAP:%.+]] = comb.icmp eq %fifo_rd_addr, %c-3_i3
// CHECK-NEXT:    [[RD_STEP:%.+]] = comb.mux [[RD_WRAP]], %c0_i3, [[RD_INC]]
// CHECK-NEXT:    [[RD_NEXT]] = comb.mux %rdEn, [[RD_STEP]], %fifo_rd_addr {sv.namehint = "fifo_rd_addr_next"}
// CHECK-NEXT:    [[CLK:%.+]] = seq.from_clock %clk
// CHECK-NEXT:    [[NOT_RST:%.+]] = comb.xor %rst, %true
// CHECK-NEXT:    [[RD_EMPTY:%.+]] = comb.and %rdEn, %fifo_empty
// CHECK-NEXT:    [[RD_OK:%.+]] = comb.xor [[RD_EMPTY]], %true
// CHECK-NEXT:    verif.clocked_assert [[RD_OK]] if [[NOT_RST]], posedge [[CLK]] label "FIFO empty when read enabled" : i1
// CHECK-NEXT:    [[WR_FULL:%.+]] = comb.and %wrEn, %fifo_full
// CHECK-NEXT:    [[WR_OK:%.+]] = comb.xor [[WR_FULL]], %true
// CHECK-NEXT:    verif.clocked_assert [[WR_OK]] if [[NOT_RST]], posedge [[CLK]] label "FIFO full when write enabled" : i1
// CHECK-NEXT:    hw.output %fifo_mem_rdata : i32
hw.module @fifo1(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
  %out, %full, %empty = seq.fifo depth 6 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out : i32
}

// A power-of-two depth: the pointer increment wraps on its own. Almost full
// and almost empty are registers too.
// CHECK-LABEL: hw.module @fifo2(
// CHECK:         %fifo_mem_rdata = seq.read %fifo_mem[%fifo_rd_addr] rden %rdEn {latency = 1 : i64} : !seq.hlmem<4x!hw.array<2xi32>>
// CHECK:         [[UP:%.+]] = comb.and %wrEn,
// CHECK:         [[DOWN:%.+]] = comb.and %rdEn,
// CHECK:         %fifo_full = seq.compreg sym @fifo_full {{%.+}}, %clk reset %rst, %false : i1
// CHECK-NEXT:    %c2_i3 = hw.constant 2 : i3
// CHECK-NEXT:    [[LAG_2:%.+]] = comb.icmp eq %fifo_count_lag, %c2_i3
// CHECK-NEXT:    %c-4_i3 = hw.constant -4 : i3
// CHECK-NEXT:    {{%.+}} = comb.icmp eq %fifo_count_lag, %c-4_i3
// CHECK-NEXT:    %c3_i3 = hw.constant 3 : i3
// CHECK-NEXT:    [[LAG_3:%.+]] = comb.icmp eq %fifo_count_lag, %c3_i3
// CHECK:         %fifo_empty = seq.compreg sym @fifo_empty {{%.+}}, %clk reset %rst, %true : i1
// CHECK-NEXT:    [[LAG_0:%.+]] = comb.icmp eq %fifo_count_lag, %c0_i3
// CHECK-NEXT:    [[LAG_1:%.+]] = comb.icmp eq %fifo_count_lag, %c1_i3
// CHECK-NEXT:    [[AT_1_DOWN:%.+]] = comb.mux %fifo_count_down, [[LAG_2]], [[LAG_1]]
// CHECK-NEXT:    [[AT_1:%.+]] = comb.mux %fifo_count_up, [[LAG_0]], [[AT_1_DOWN]]
// CHECK:         [[WR_INC:%.+]] = comb.add %fifo_wr_addr, %c1_i2
// CHECK-NEXT:    {{%.+}} = comb.mux %wrEn, [[WR_INC]], %fifo_wr_addr {sv.namehint = "fifo_wr_addr_next"}
// CHECK-NEXT:    [[RD_INC:%.+]] = comb.add %fifo_rd_addr, %c1_i2
// CHECK-NEXT:    {{%.+}} = comb.mux %rdEn, [[RD_INC]], %fifo_rd_addr {sv.namehint = "fifo_rd_addr_next"}
// Almost full (count >= 2): set going up from 1, cleared going down from 2.
// CHECK-NEXT:    %fifo_almost_full = seq.compreg sym @fifo_almost_full [[AF_NEXT:%.+]], %clk reset %rst, %false : i1
// CHECK-NEXT:    [[AT_2_DOWN:%.+]] = comb.mux %fifo_count_down, [[LAG_3]], [[LAG_2]]
// CHECK-NEXT:    [[AT_2:%.+]] = comb.mux %fifo_count_up, [[LAG_1]], [[AT_2_DOWN]]
// CHECK-NEXT:    [[NOT_AT_2:%.+]] = comb.xor [[AT_2]], %true
// CHECK-NEXT:    [[AF_UP:%.+]] = comb.or %fifo_almost_full, [[AT_1]]
// CHECK-NEXT:    [[AF_DOWN:%.+]] = comb.and %fifo_almost_full, [[NOT_AT_2]]
// CHECK-NEXT:    [[AF_MUX:%.+]] = comb.mux [[DOWN]], [[AF_DOWN]], %fifo_almost_full
// CHECK-NEXT:    [[AF_NEXT]] = comb.mux [[UP]], [[AF_UP]], [[AF_MUX]]
// Almost empty (count <= 1): cleared going up from 1, set going down from 2.
// CHECK-NEXT:    %fifo_almost_empty = seq.compreg sym @fifo_almost_empty [[AE_NEXT:%.+]], %clk reset %rst, %true : i1
// CHECK-NEXT:    [[NOT_AT_1:%.+]] = comb.xor [[AT_1]], %true
// CHECK-NEXT:    [[AE_UP:%.+]] = comb.and %fifo_almost_empty, [[NOT_AT_1]]
// CHECK-NEXT:    [[AE_DOWN:%.+]] = comb.or %fifo_almost_empty, [[AT_2]]
// CHECK-NEXT:    [[AE_MUX:%.+]] = comb.mux [[DOWN]], [[AE_DOWN]], %fifo_almost_empty
// CHECK-NEXT:    [[AE_NEXT]] = comb.mux [[UP]], [[AE_UP]], [[AE_MUX]]
// CHECK:         verif.clocked_assert
// CHECK:         verif.clocked_assert
// CHECK-NEXT:    hw.output %fifo_mem_rdata, %fifo_empty, %fifo_full, %fifo_almost_empty, %fifo_almost_full
!testType = !hw.array<2xi32>
hw.module @fifo2(in %clk : !seq.clock, in %rst : i1, in %in : !testType, in %rdEn : i1, in %wrEn : i1, out out: !testType, out empty: i1, out full: i1, out almost_empty : i1, out almost_full : i1) {
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 4 rd_latency 1 almost_full 2 almost_empty 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : !testType
  hw.output %out, %empty, %full, %almostEmpty, %almostFull : !testType, i1, i1, i1, i1
}

// Thresholds of `depth` and 0 make almost full and almost empty the full and
// empty flags.
// CHECK-LABEL: hw.module @aliasedThresholds(
// CHECK-NOT:     fifo_almost
// CHECK:         hw.output %fifo_full, %fifo_empty : i1, i1
hw.module @aliasedThresholds(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %rdEn : i1, in %wrEn : i1, out af : i1, out ae : i1) {
  %out, %full, %empty, %af, %ae = seq.fifo depth 8 almost_full 8 almost_empty 0 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i8
  hw.output %af, %ae : i1, i1
}

// Thresholds of 0 and `depth` make almost full and almost empty constant.
// CHECK-LABEL: hw.module @constantThresholds(
// CHECK-NOT:     fifo_almost
// CHECK:         hw.output %true, %true : i1, i1
hw.module @constantThresholds(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %rdEn : i1, in %wrEn : i1, out af : i1, out ae : i1) {
  %out, %full, %empty, %af, %ae = seq.fifo depth 8 almost_full 0 almost_empty 8 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i8
  hw.output %af, %ae : i1, i1
}

// Unused almost full and almost empty outputs get no registers.
// CHECK-LABEL: hw.module @unusedThresholds(
// CHECK-NOT:     fifo_almost
// CHECK:         hw.output %fifo_full : i1
hw.module @unusedThresholds(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %rdEn : i1, in %wrEn : i1, out full : i1) {
  %out, %full, %empty, %af, %ae = seq.fifo depth 8 almost_full 5 almost_empty 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i8
  hw.output %full : i1
}

// A single entry: the pointers are zero-width and never move.
// CHECK-LABEL: hw.module @depth1(
// CHECK:         %fifo_rd_addr = seq.compreg sym @fifo_rd_addr %fifo_rd_addr, %clk reset %rst, %c0_i0 : i0
// CHECK-NEXT:    %fifo_wr_addr = seq.compreg sym @fifo_wr_addr %fifo_wr_addr, %clk reset %rst, %c0_i0 : i0
// CHECK-NOT:     comb.add %fifo_rd_addr
// CHECK-NOT:     comb.add %fifo_wr_addr
// CHECK:         hw.output %fifo_mem_rdata : i8
hw.module @depth1(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %rdEn : i1, in %wrEn : i1, out out : i8) {
  %out, %full, %empty = seq.fifo depth 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i8
  hw.output %out : i8
}

// With rd_latency > 1, the memory is still read in the cycle the entry is
// popped, and the rest of the latency is registers on the read data. No read is
// in flight when a popped slot is reused, so the flags count only the entries
// not yet popped.
// CHECK-LABEL: hw.module @fifo3(
// CHECK:         %fifo_mem_rdata = seq.read %fifo_mem[%fifo_rd_addr] rden %rdEn {latency = 1 : i64}
// CHECK-NEXT:    %fifo_mem_rd_dly1 = seq.compreg sym @fifo_mem_rd_dly1 %fifo_mem_rdata, %clk : i32
// CHECK-NEXT:    %fifo_mem_rd_dly2 = seq.compreg sym @fifo_mem_rd_dly2 %fifo_mem_rd_dly1, %clk : i32
// CHECK:         %fifo_full = seq.compreg sym @fifo_full
// CHECK:         %fifo_almost_full = seq.compreg sym @fifo_almost_full
// CHECK:         hw.output %fifo_mem_rd_dly2, %fifo_full, %fifo_almost_full : i32, i1, i1
// CHECK-NOT:     fifo_used
hw.module @fifo3(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32, out full : i1, out almost_full : i1) {
  %out, %full, %empty, %af = seq.fifo depth 4 rd_latency 3 almost_full 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out, %full, %af : i32, i1, i1
}

// A depth one short of a power of two: the memory has a spare slot, the one at
// the write pointer, so it is written every cycle without a write enable, and
// the pointers wrap on their own. The count is three bits.
// CHECK-LABEL: hw.module @spareSlot(
// CHECK:         %fifo_count_lag = seq.compreg sym @fifo_count_lag {{%.+}}, %clk reset %rst, %c0_i3 : i3
// CHECK:         %fifo_mem = seq.hlmem @fifo_mem %clk, %false : <8xi8>
// CHECK:         %fifo_rd_addr = seq.compreg sym @fifo_rd_addr {{%.+}}, %clk reset %rst, %c0_i3 : i3
// CHECK:         seq.write %fifo_mem[%fifo_wr_addr] %in wren %true {latency = 1 : i64} : !seq.hlmem<8xi8>
// CHECK-NOT:     comb.icmp eq %fifo_wr_addr
// CHECK-NOT:     comb.icmp eq %fifo_rd_addr
// CHECK:         [[WR_INC:%.+]] = comb.add %fifo_wr_addr, %c1_i3
// CHECK-NEXT:    {{%.+}} = comb.mux %wrEn, [[WR_INC]], %fifo_wr_addr {sv.namehint = "fifo_wr_addr_next"}
// CHECK-NEXT:    [[RD_INC:%.+]] = comb.add %fifo_rd_addr, %c1_i3
// CHECK-NEXT:    {{%.+}} = comb.mux %rdEn, [[RD_INC]], %fifo_rd_addr {sv.namehint = "fifo_rd_addr_next"}
hw.module @spareSlot(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %rdEn : i1, in %wrEn : i1, out out : i8) {
  %out, %full, %empty = seq.fifo depth 7 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i8
  hw.output %out : i8
}
