// RUN: circt-opt --seq-unroll-mem-latency %s | FileCheck %s

// CHECK-LABEL: hw.module @hlmem
// CHECK: [[MEM:%.+]] = seq.hlmem @mem %clk, %rst : <4xi32>
// CHECK: [[C0:%.+]] = hw.constant 0 : i2
// CHECK: [[C1:%.+]] = hw.constant 1 : i1
// CHECK: [[C42:%.+]] = hw.constant 42 : i32

// CHECK: [[WR_ADDR:%.+]] = seq.shiftreg [68] [[C0]], %clk, %rst : i2
// CHECK: [[DATA:%.+]] = seq.shiftreg [68] [[C42]], %clk, %rst : i32
// CHECK: seq.write [[MEM]][[[WE_ADDR]]] [[DATA]] wren [[C1]] { latency = 1 } : !seq.hlmem<4xi32>

// CHECK: [[RD_ADDR:%.+]] = seq.shiftreg [42] [[C0]], %clk, %rst : i2
// CHECK: [[READ:%.+]] = seq.read [[MEM]][[[RD_ADDR]]] rden [[C1]] { latency = 0 } : !seq.hlmem<4xi32>

// CHECK: hw.output [[READ]] : i32
hw.module @hlmem(in %clk : !seq.clock, in %rst : i1, out "rd": i32) {
  %mem = seq.hlmem @mem %clk, %rst : <4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32
  
  seq.write %mem[%c0_i2] %c42_i32 wren %c1_i1 { latency = 69 } : !seq.hlmem<4xi32>
  %rddata = seq.read %mem[%c0_i2] rden %c1_i1 { latency = 42 } : !seq.hlmem<4xi32>
  hw.output %rddata : i32
}

// CHECK-LABEL: hw.module @firmem

// CHECK: [[MEM:%.+]] = seq.firmem 0, 1, undefined, port_order : <12 x 42>

// CHECK: [[RD_ADDR:%.+]] = seq.shiftreg [42] %addr, %clk, %rst : i4
// CHECK: [[RD_DATA:%.+]] = seq.firmem.read_port [[MEM][[[RD_DATA]]], clock %clk enable %en : <12 x 42>

// CHECK: [[WR_ADDR:%.+]] = seq.shiftreg [68] %addr, %clk, %rst : i4
// CHECK: [[WR_DATA:%.+]] = seq.shiftreg [68] %wdata, %clk, %rst : i42
// CHECK: seq.firmem.write_port [[MEM]][[[WR_ADDR]]] = [[WR_DATA]], clock %clk enable %en : <12 x 42>

// CHECK: [[RWRD_ADDR:%.+]] = seq.shiftreg [42] %addr, %clk, %rst : i4
// CHECK: [[RWWR_ADDR:%.+]] = seq.shiftreg [68] %addr, %clk, %rst : i4
// CHECK: [[MUX:%.+]] = comb.mux %wmode, [[RWWR_ADDR]], [[RWRD_ADDR]] : i4
// CHECK: [[RWWR_DATA:%.+]] = seq.shiftreg [68] %wdata, %clk, %rst : i42
// CHECK: [[RWRD_DATA]] = seq.firmem.read_write_port [[MEM]][[[MUX]]] = [[RWRW_DATA]] if %wmode, clock %clk enable %en : <12 x 42>
hw.module @firmem(in %clk: !seq.clock, in %rst: i1, in %en: i1, in %addr: i4, in %wdata: i42, in %wmode: i1) {
  %mem = seq.firmem 42, 69, undefined, port_order : <12 x 42>
  %0 = seq.firmem.read_port %mem[%addr], clock %clk enable %en : <12 x 42>
  seq.firmem.write_port %mem[%addr] = %wdata, clock %clk enable %en : <12 x 42>
  %2 = seq.firmem.read_write_port %mem[%addr] = %wdata if %wmode, clock %clk enable %en : <12 x 42>
}
