// This is such a large lowering that it doesn't really make that much sense to
// inspect the test output. So this is mostly here for detecting regressions.
// Canonicalize used to remove some of the constants introduced by the lowering.
// RUN: circt-opt --lower-seq-fifo --canonicalize %s | FileCheck %s --implicit-check-not=seq.fifo


// CHECK:  hw.module @fifo1(in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_1:.*]] : i1, in %[[VAL_2:.*]] : i32, in %[[VAL_3:.*]] : i1, in %[[VAL_4:.*]] : i1, out out : i32) {
// CHECK:    %fifo_count = seq.compreg sym @fifo_count %{{.+}}, %clk reset %rst, %c0_i2 : i2  
// CHECK:    %fifo_mem = seq.hlmem @fifo_mem %clk, %rst : <3xi32>
// CHECK:    %fifo_rd_addr = seq.compreg sym @fifo_rd_addr %{{.+}}, %clk reset %rst, %c0_i2 : i2  
// CHECK:    %fifo_wr_addr = seq.compreg sym @fifo_wr_addr %{{.+}}, %clk reset %rst, %c0_i2 : i2  
// CHECK:    %fifo_mem_rdata = seq.read %fifo_mem[%fifo_rd_addr] rden %rdEn {latency = 0 : i64} : !seq.hlmem<3xi32>
// CHECK:    seq.write %fifo_mem[%fifo_wr_addr] %in wren %wrEn {latency = 1 : i64} : !seq.hlmem<3xi32>
//           All of the comb logic which would be here is pretty much impossible
//           to verify visually. As a rule, if it's difficult to verify
//           visually, that's a candidate for a new op: #8002. Since I haven't
//           verified it visually, I'm not including it here. There are
//           integration tests for it.
// CHECK:    verif.clocked_assert %{{.+}}, posedge %{{.+}} label "FIFO empty when read enabled" : i1
// CHECK:    verif.clocked_assert %{{.+}}, posedge %{{.+}} label "FIFO full when write enabled" : i1
// CHECK:    hw.output %fifo_mem_rdata : i32
hw.module @fifo1(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
  %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out : i32
}


// CHECK:   hw.module @fifo2(in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_1:.*]] : i1, in %[[VAL_2:.*]] : [[TY:.+]], in %[[VAL_3:.*]] : i1, in %[[VAL_4:.*]] : i1, out out : [[TY]], out empty : i1, out full : i1, out almost_empty : i1, out almost_full : i1) {
// CHECK:     %fifo_count = seq.compreg sym @fifo_count %16, %clk reset %rst, %c0_i3 : i3  
// CHECK:     %fifo_mem = seq.hlmem @fifo_mem %clk, %rst : <4x!hw.array<2xi32>>
// CHECK:     %fifo_rd_addr = seq.compreg sym @fifo_rd_addr %28, %clk reset %rst, %c0_i2 : i2  
// CHECK:     %fifo_wr_addr = seq.compreg sym @fifo_wr_addr %22, %clk reset %rst, %c0_i2 : i2  
// CHECK:     %fifo_mem_rdata = seq.read %fifo_mem[%fifo_rd_addr] rden %rdEn {latency = 1 : i64} : !seq.hlmem<4x!hw.array<2xi32>>
// CHECK:     seq.write %fifo_mem[%fifo_wr_addr] %in wren %wrEn {latency = 1 : i64} : !seq.hlmem<4x!hw.array<2xi32>>
//            See comment above.
// CHECK:     verif.clocked_assert %{{.+}}, posedge %{{.+}} label "FIFO empty when read enabled" : i1
// CHECK:     verif.clocked_assert %{{.+}}, posedge %{{.+}} label "FIFO full when write enabled" : i1
!testType = !hw.array<2xi32>
hw.module @fifo2(in %clk : !seq.clock, in %rst : i1, in %in : !testType, in %rdEn : i1, in %wrEn : i1, out out: !testType, out empty: i1, out full: i1, out almost_empty : i1, out almost_full : i1) {
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 4 rd_latency 1 almost_full 2 almost_empty 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : !testType
  hw.output %out, %empty, %full, %almostEmpty, %almostFull : !testType, i1, i1, i1, i1
}
