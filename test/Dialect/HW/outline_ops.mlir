// RUN: circt-opt %s --hw-outline-ops=op-names=seq.fifo | FileCheck %s

// CHECK-LABEL:   hw.module @fifo3a(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
// CHECK-NEXT:     %outlined_seq.fifo.out, %outlined_seq.fifo.full, %outlined_seq.fifo.empty = hw.instance "outlined_seq.fifo" @outlined_seq.fifo_opers_i32_i1_i1_seq.clock_i1_res_i32_i1_i1_attrs_depth_3("": %in: i32, "": %rdEn: i1, "": %wrEn: i1, "": %clk: !seq.clock, "": %rst: i1) -> (out: i32, full: i1, empty: i1)
// CHECK-NEXT:     hw.output %outlined_seq.fifo.out : i32
hw.module @fifo3a(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
  %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out : i32
}

// CHECK-LABEL:   hw.module @fifo3b(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
// CHECK-NEXT:     %outlined_seq.fifo.out, %outlined_seq.fifo.full, %outlined_seq.fifo.empty = hw.instance "outlined_seq.fifo" @outlined_seq.fifo_opers_i32_i1_i1_seq.clock_i1_res_i32_i1_i1_attrs_depth_3("": %in: i32, "": %rdEn: i1, "": %wrEn: i1, "": %clk: !seq.clock, "": %rst: i1) -> (out: i32, full: i1, empty: i1)
// CHECK-NEXT:     hw.output %outlined_seq.fifo.out : i32
hw.module @fifo3b(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
  %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out : i32
}

// CHECK-LABEL:   hw.module @fifo7a(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
// CHECK-NEXT:     %outlined_seq.fifo.out, %outlined_seq.fifo.full, %outlined_seq.fifo.empty = hw.instance "outlined_seq.fifo" @outlined_seq.fifo_opers_i32_i1_i1_seq.clock_i1_res_i32_i1_i1_attrs_depth_7("": %in: i32, "": %rdEn: i1, "": %wrEn: i1, "": %clk: !seq.clock, "": %rst: i1) -> (out: i32, full: i1, empty: i1)
// CHECK-NEXT:     hw.output %outlined_seq.fifo.out : i32
hw.module @fifo7a(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1, out out : i32) {
  %out, %full, %empty = seq.fifo depth 7 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
  hw.output %out : i32
}

// CHECK-LABEL:   hw.module @fifo3a_i8(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %rdEn : i1, in %wrEn : i1, out out : i8) {
// CHECK-NEXT:     %outlined_seq.fifo.out, %outlined_seq.fifo.full, %outlined_seq.fifo.empty = hw.instance "outlined_seq.fifo" @outlined_seq.fifo_opers_i8_i1_i1_seq.clock_i1_res_i8_i1_i1_attrs_depth_3("": %in: i8, "": %rdEn: i1, "": %wrEn: i1, "": %clk: !seq.clock, "": %rst: i1) -> (out: i8, full: i1, empty: i1)
// CHECK-NEXT:     hw.output %outlined_seq.fifo.out : i8
hw.module @fifo3a_i8(in %clk : !seq.clock, in %rst : i1, in %in : i8, in %rdEn : i1, in %wrEn : i1, out out : i8) {
  %out, %full, %empty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i8
  hw.output %out : i8
}

// CHECK:        hw.module @outlined_seq.fifo_opers_i32_i1_i1_seq.clock_i1_res_i32_i1_i1_attrs_depth_3(in [[R0:%.]] "" : i32, in [[R1:%.]] "" : i1, in [[R2:%.]] "" : i1, in [[R3:%.]] "" : !seq.clock, in [[R4:%.]] "" : i1, out out : i32, out full : i1, out empty : i1) {
// CHECK-NEXT:     %out, %full, %empty = seq.fifo depth 3   in [[R0]] rdEn [[R1]] wrEn [[R2]] clk [[R3]] rst [[R4]] : i32
// CHECK-NEXT:     hw.output %out, %full, %empty : i32, i1, i1
// CHECK-NEXT:   }
// CHECK:        hw.module @outlined_seq.fifo_opers_i32_i1_i1_seq.clock_i1_res_i32_i1_i1_attrs_depth_7(in [[R0:%.]] "" : i32, in [[R1:%.]] "" : i1, in [[R2:%.]] "" : i1, in [[R3:%.]] "" : !seq.clock, in [[R4:%.]] "" : i1, out out : i32, out full : i1, out empty : i1) {
// CHECK-NEXT:     %out, %full, %empty = seq.fifo depth 7   in [[R0]] rdEn [[R1]] wrEn [[R2]] clk [[R3]] rst [[R4]] : i32
// CHECK-NEXT:     hw.output %out, %full, %empty : i32, i1, i1
// CHECK-NEXT:   }
// CHECK:        hw.module @outlined_seq.fifo_opers_i8_i1_i1_seq.clock_i1_res_i8_i1_i1_attrs_depth_3(in [[R0:%.]] "" : i8, in [[R1:%.]] "" : i1, in [[R2:%.]] "" : i1, in [[R3:%.]] "" : !seq.clock, in [[R4:%.]] "" : i1, out out : i8, out full : i1, out empty : i1) {
// CHECK-NEXT:     %out, %full, %empty = seq.fifo depth 3   in [[R0]] rdEn [[R1]] wrEn [[R2]] clk [[R3]] rst [[R4]] : i8
// CHECK-NEXT:     hw.output %out, %full, %empty : i8, i1, i1
// CHECK-NEXT:   }
