// RUN: circt-opt --lower-seq-hlmem %s | FileCheck %s

hw.module @d1(%clk : i1, %rst : i1) -> () {
  // CHECK:           %[[VAL_2:.*]] = sv.reg  : !hw.inout<uarray<4xi32>>
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>
  
  // CHECK:           sv.alwaysff(posedge %clk) {
  // CHECK:             sv.if %[[VAL_3:.*]] {
  // CHECK:               %[[VAL_4:.*]] = sv.array_index_inout %myMemory[%c0_i2] : !hw.inout<uarray<4xi32>>, i2
  // CHECK:               sv.passign %[[VAL_4]], %c42_i32 : i32
  // CHECK:             }
  // CHECK:           }(syncreset : posedge %rst) {
  // CHECK:           }
  seq.write %myMemory[%c0_i2] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  // CHECK:           %[[VAL_10:.*]] = sv.array_index_inout %myMemory[%c0_i2] : !hw.inout<uarray<4xi32>>, i2
  // CHECK:           %[[VAL_11:.*]] = sv.read_inout %[[VAL_10]] : !hw.inout<i32>
  %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 0} : !seq.hlmem<4xi32>

  // CHECK:           %[[VAL_12:.*]] = sv.array_index_inout %myMemory[%c0_i2] : !hw.inout<uarray<4xi32>>, i2
  // CHECK:           %[[VAL_13:.*]] = sv.read_inout %[[VAL_12]] : !hw.inout<i32>
  // CHECK:           %[[VAL_14:.*]] = seq.compreg sym @myMemory_rd0_dly0 %[[VAL_13]], %clk : i32
  // CHECK:           %[[VAL_15:.*]] = seq.compreg sym @myMemory_rd0_dly1 %[[VAL_14]], %clk : i32
  %myMemory_rdata2 = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 2} : !seq.hlmem<4xi32>
  hw.output
}
