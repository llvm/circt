// RUN: circt-opt --seq-unroll-mem-latency %s | FileCheck %s

// CHECK-LABEL: hw.module @hlmem

// CHECK: %mem = seq.hlmem @mem %clk, %rst : <4xi32>
// CHECK: [[C0:%.+]] = hw.constant 0 : i2
// CHECK: [[T:%.+]] = hw.constant true
// CHECK: [[C42:%.+]] = hw.constant 42 : i32
// CHECK: [[T0:%.+]] = hw.constant true
// CHECK: [[S0:%.+]] = seq.shiftreg[68] [[C0]], %clk, [[T0]] : i2  
// CHECK: [[S1:%.+]] = seq.shiftreg[68] [[C42]], %clk, [[T0]] : i32  
// CHECK: [[SEN:%.+]] = seq.shiftreg[68] [[T]], %clk, [[T0]] : i1  
// CHECK: seq.write %mem[[[S0]]] [[S1]] wren [[SEN]] {latency = 1 : i64} : !seq.hlmem<4xi32>
// CHECK: [[T1:%.+]] = hw.constant true
// CHECK: [[S2:%.+]] = seq.shiftreg[42] [[C0]], %clk, [[T1]] : i2  
// CHECK: [[SEN1:%.+]] = seq.shiftreg[42] [[T]], %clk, [[T1]] : i1  
// CHECK: [[RD_DATA:%.+]] = seq.read %mem[[[S2]]] rden [[SEN1]] {latency = 0 : i64} : !seq.hlmem<4xi32>
// CHECK: hw.output [[RD_DATA]] : i32
hw.module @hlmem(in %clk : !seq.clock, in %rst : i1, out "rd": i32) {
  %mem = seq.hlmem @mem %clk, %rst : <4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32
  
  seq.write %mem[%c0_i2] %c42_i32 wren %c1_i1 { latency = 69 } : !seq.hlmem<4xi32>
  %rddata = seq.read %mem[%c0_i2] rden %c1_i1 { latency = 42 } : !seq.hlmem<4xi32>
  hw.output %rddata : i32
}

// CHECK-LABEL: hw.module @hlmemManyAddr

// CHECK: %mem = seq.hlmem @mem %clk, %rst : <4x4x4xi32>
// CHECK: [[T:%.+]] = hw.constant true
// CHECK: [[C42:%.+]] = hw.constant 42 : i32
// CHECK: [[T0:%.+]] = hw.constant true
// CHECK: [[SADDR0:%.+]] = seq.shiftreg[68] %addr0, %clk, [[T0]] : i2  
// CHECK: [[SADDR1:%.+]] = seq.shiftreg[68] %addr1, %clk, [[T0]] : i2  
// CHECK: [[SADDR2:%.+]] = seq.shiftreg[68] %addr2, %clk, [[T0]] : i2  
// CHECK: [[S1:%.+]] = seq.shiftreg[68] [[C42]], %clk, [[T0]] : i32  
// CHECK: [[SEN:%.+]] = seq.shiftreg[68] [[T]], %clk, [[T0]] : i1  
// CHECK: seq.write %mem[[[SADDR0]], [[SADDR1]], [[SADDR2]]] [[S1]] wren [[SEN]] {latency = 1 : i64} : !seq.hlmem<4x4x4xi32>
// CHECK: [[T1:%.+]] = hw.constant true
// CHECK: [[SADDR3:%.+]] = seq.shiftreg[42] %addr0, %clk, [[T1]] : i2  
// CHECK: [[SADDR4:%.+]] = seq.shiftreg[42] %addr1, %clk, [[T1]] : i2  
// CHECK: [[SADDR5:%.+]] = seq.shiftreg[42] %addr2, %clk, [[T1]] : i2   
// CHECK: [[SEN1:%.+]] = seq.shiftreg[42] [[T]], %clk, [[T1]] : i1  
// CHECK: [[RD_DATA:%.+]] = seq.read %mem[[[SADDR3]], [[SADDR4]], [[SADDR5]]] rden [[SEN1]] {latency = 0 : i64} : !seq.hlmem<4x4x4xi32>
// CHECK: hw.output [[RD_DATA]] : i32
hw.module @hlmemManyAddr(in %clk : !seq.clock, in %rst : i1, in %addr0 : i2, in %addr1 : i2, in %addr2 : i2, out "rd": i32) {
  %mem = seq.hlmem @mem %clk, %rst : !seq.hlmem<4x4x4xi32>

  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32
  
  seq.write %mem[%addr0, %addr1, %addr2] %c42_i32 wren %c1_i1 { latency = 69 } : !seq.hlmem<4x4x4xi32>
  %rddata = seq.read %mem[%addr0, %addr1, %addr2] rden %c1_i1 { latency = 42 } : !seq.hlmem<4x4x4xi32>
  hw.output %rddata : i32
}

// CHECK-LABEL: hw.module @firmem

// CHECK: [[T:%.+]] = hw.constant true
// CHECK: [[S3:%.+]] = seq.shiftreg[68] %wdata, %clk, [[T]] : i42  
// CHECK: [[S4:%.+]] = seq.shiftreg[68] %addr, %clk, [[T]] : i4  
// CHECK: [[SEN:%.+]] = seq.shiftreg[68] %en, %clk, [[T]] : i1  
// CHECK: seq.firmem.write_port %mem[[[S4]]] = [[S3]], clock %clk enable [[SEN]] : <12 x 42>
// CHECK: [[S5:%.+]] = seq.shiftreg[42] %addr, %clk, [[T]] : i4  
// CHECK: [[SEN1:%.+]] = seq.shiftreg[42] %en, %clk, [[T]] : i1  
// CHECK: [[RD:%.+]] = seq.firmem.read_port %mem[[[S5]]], clock %clk enable [[SEN1]] : <12 x 42>
// CHECK: %mem = seq.firmem 0, 1, undefined, port_order : <12 x 42>
hw.module @firmem(in %clk: !seq.clock, in %rst: i1, in %en: i1, in %addr: i4, in %wdata: i42, in %wmode: i1) {
  %mem = seq.firmem 42, 69, undefined, port_order : <12 x 42>
  %0 = seq.firmem.read_port %mem[%addr], clock %clk enable %en : <12 x 42>
  seq.firmem.write_port %mem[%addr] = %wdata, clock %clk enable %en : <12 x 42>
}

// CHECK-LABEL: hw.module @firmemRdWr

// CHECK: [[T:%.+]] = hw.constant true
// CHECK: [[S0:%.+]] = seq.shiftreg[42] %addr, %clk, [[T]] : i4  
// CHECK: [[S1:%.+]] = seq.shiftreg[68] %addr, %clk, [[T]] : i4  
// CHECK: [[S2:%.+]] = seq.shiftreg[68] %wdata, %clk, [[T]] : i42  
// CHECK: [[SMW0:%.+]] = seq.shiftreg[68] %wmode, %clk, [[T]] : i1  
// CHECK: [[SMR0:%.+]] = seq.shiftreg[42] %wmode, %clk, [[T]] : i1  
// CHECK: [[SM0:%.+]] = comb.mux [[SMR0]], [[SMW0]], [[SMR0]] : i1
// CHECK: [[SS0:%.+]] = comb.mux [[SM0]], [[S1]], [[S0]] : i4
// CHECK: [[SEN0:%.+]] = seq.shiftreg[68] %en, %clk, [[T]] : i1  
// CHECK: [[RDWR0:%.+]] = seq.firmem.read_write_port %mem0[[[SS0]]] = [[S2]] if [[SM0]], clock %clk enable [[SEN0]] : <12 x 42>
// CHECK: %mem0 = seq.firmem 0, 1, undefined, port_order : <12 x 42>

// CHECK: [[T0:%.+]] = hw.constant true
// CHECK: [[S3:%.+]] = seq.shiftreg[69] %addr, %clk, [[T0]] : i4  
// CHECK: [[S4:%.+]] = seq.shiftreg[41] %addr, %clk, [[T0]] : i4  
// CHECK: [[S5:%.+]] = seq.shiftreg[41] %wdata, %clk, [[T0]] : i42  
// CHECK: [[SMW1:%.+]] = seq.shiftreg[41] %wmode, %clk, [[T0]] : i1  
// CHECK: [[SMR1:%.+]] = seq.shiftreg[69] %wmode, %clk, [[T0]] : i1  
// CHECK: [[SM1:%.+]] = comb.mux [[SMW1]], [[SMW1]], [[SMR1]] : i1
// CHECK: [[SS1:%.+]] = comb.mux [[SM1]], [[S4]], [[S3]] : i4
// CHECK: [[SEN1:%.+]] = seq.shiftreg[41] %en, %clk, [[T0]] : i1  
// CHECK: [[RDWR1:%.+]] = seq.firmem.read_write_port %mem1[[[SS1]]] = [[S5]] if [[SM1]], clock %clk enable [[SEN1]] : <12 x 42>
// CHECK: %mem1 = seq.firmem 0, 1, undefined, port_order : <12 x 42>
hw.module @firmemRdWr(in %clk: !seq.clock, in %rst: i1, in %en: i1, in %addr: i4, in %wdata: i42, in %wmode: i1) {
  %mem0 = seq.firmem 42, 69, undefined, port_order : <12 x 42>
  %2 = seq.firmem.read_write_port %mem0[%addr] = %wdata if %wmode, clock %clk enable %en : <12 x 42>

  %mem1 = seq.firmem 69, 42, undefined, port_order : <12 x 42>
  %3 = seq.firmem.read_write_port %mem1[%addr] = %wdata if %wmode, clock %clk enable %en : <12 x 42>
}
