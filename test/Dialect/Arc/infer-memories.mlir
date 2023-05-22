// RUN: circt-opt %s --arc-infer-memories | FileCheck %s

hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]


// CHECK-LABEL: hw.module @TestWOMemory(
hw.module @TestWOMemory(%clock: i1, %addr: i10, %enable: i1, %data: i8) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: [[FOO:%.+]] = arc.memory <1024 x i8, i10> {name = "foo"}
  // CHECK-NEXT: arc.memory_write_port [[FOO]], @mem_write{{.*}}(%addr, %data, %enable) clock %clock enable lat 1 : <1024 x i8, i10>, i10, i8, i1
  // CHECK-NEXT: hw.output
  hw.instance "foo" @WOMemory(W0_addr: %addr: i10, W0_en: %enable: i1, W0_clk: %clock: i1, W0_data: %data: i8) -> ()
}
// CHECK-NEXT: }
// CHECK-NOT: hw.module.generated @WOMemory, @FIRRTLMem
hw.module.generated @WOMemory, @FIRRTLMem(%W0_addr: i10, %W0_en: i1, %W0_clk: i1, %W0_data: i8) attributes {depth = 1024 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}


// CHECK-LABEL: hw.module @TestWOMemoryWithMask(
hw.module @TestWOMemoryWithMask(%clock: i1, %addr: i10, %enable: i1, %data: i16, %mask: i2) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: [[FOO:%.+]] = arc.memory <1024 x i16, i10> {name = "foo"}
  // CHECK-NEXT: [[MASK_BIT0:%.+]] = comb.extract %mask from 0 : (i2) -> i1
  // CHECK-NEXT: [[MASK_BYTE0:%.+]] = comb.replicate [[MASK_BIT0]] : (i1) -> i8
  // CHECK-NEXT: [[MASK_BIT1:%.+]] = comb.extract %mask from 1 : (i2) -> i1
  // CHECK-NEXT: [[MASK_BYTE1:%.+]] = comb.replicate [[MASK_BIT1]] : (i1) -> i8
  // CHECK-NEXT: [[MASK:%.+]] = comb.concat [[MASK_BYTE0]], [[MASK_BYTE1]]
  // CHECK-NEXT: arc.memory_write_port [[FOO]], @mem_write{{.*}}(%addr, %data, %enable, [[MASK]]) clock %clock enable mask lat 1 : <1024 x i16, i10>, i10, i16, i1, i16
  // CHECK-NEXT: hw.output
  hw.instance "foo" @WOMemoryWithMask(W0_addr: %addr: i10, W0_en: %enable: i1, W0_clk: %clock: i1, W0_data: %data: i16, W0_mask: %mask: i2) -> ()
}
// CHECK-NEXT: }
// CHECK-NOT: hw.module.generated @WOMemoryWithMask, @FIRRTLMem
hw.module.generated @WOMemoryWithMask, @FIRRTLMem(%W0_addr: i10, %W0_en: i1, %W0_clk: i1, %W0_data: i16, %W0_mask: i2) attributes {depth = 1024 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 16 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}


// CHECK-LABEL: hw.module @TestROMemory(
hw.module @TestROMemory(%clock: i1, %addr: i10, %enable: i1) -> (data: i8) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: [[FOO:%.+]] = arc.memory <1024 x i8, i10> {name = "foo"}
  // CHECK-NEXT: [[RDATA:%.+]] = arc.memory_read_port [[FOO]][%addr] : <1024 x i8, i10>
  // CHECK-NEXT: hw.output [[RDATA]]
  %0 = hw.instance "foo" @ROMemory(R0_addr: %addr: i10, R0_en: %enable: i1, R0_clk: %clock: i1) -> (R0_data: i8)
  hw.output %0 : i8
}
// CHECK-NEXT: }
// CHECK-NOT: hw.module.generated @ROMemory, @FIRRTLMem
hw.module.generated @ROMemory, @FIRRTLMem(%R0_addr: i10, %R0_en: i1, %R0_clk: i1) -> (R0_data: i8) attributes {depth = 1024 : i64, maskGran = 8 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}


// CHECK-LABEL: hw.module @TestROMemoryWithLatency(
hw.module @TestROMemoryWithLatency(%clock: i1, %addr: i10, %enable: i1) -> (data: i8) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: [[FOO:%.+]] = arc.memory <1024 x i8, i10> {name = "foo"}
  // CHECK-NEXT: [[D0:%.+]] = arc.memory_read_port [[FOO]][%addr] : <1024 x i8, i10>
  // CHECK-NEXT: [[D1:%.+]] = seq.compreg {{.+}} [[D0]], %clock
  // CHECK-NEXT: [[D2:%.+]] = seq.compreg {{.+}} [[D1]], %clock
  // CHECK-NEXT: [[D3:%.+]] = seq.compreg {{.+}} [[D2]], %clock
  // CHECK-NEXT: hw.output [[D3]]
  %0 = hw.instance "foo" @ROMemoryWithLatency(R0_addr: %addr: i10, R0_en: %enable: i1, R0_clk: %clock: i1) -> (R0_data: i8)
  hw.output %0 : i8
}
// CHECK-NEXT: }
// CHECK-NOT: hw.module.generated @ROMemoryWithLatency, @FIRRTLMem
hw.module.generated @ROMemoryWithLatency, @FIRRTLMem(%R0_addr: i10, %R0_en: i1, %R0_clk: i1) -> (R0_data: i8) attributes {depth = 1024 : i64, maskGran = 8 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 3 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}


// CHECK-LABEL: hw.module @TestRWMemory(
hw.module @TestRWMemory(%clock: i1, %addr: i10, %enable: i1, %wmode: i1, %wdata: i8) -> (rdata: i8) {
  // CHECK-NOT: hw.instance
  // CHECK-NEXT: [[FOO:%.+]] = arc.memory <1024 x i8, i10> {name = "foo"}
  // CHECK-NEXT: [[RDATA:%.+]] = arc.memory_read_port [[FOO]][%addr] : <1024 x i8, i10>
  // CHECK-NEXT: [[WENABLE:%.+]] = comb.and %enable, %wmode
  // CHECK-NEXT: arc.memory_write_port [[FOO]], @mem_write{{.*}}(%addr, %wdata, [[WENABLE]]) clock %clock enable lat 1 : <1024 x i8, i10>, i10, i8, i1
  // CHECK-NEXT: hw.output [[RDATA]]
  %0 = hw.instance "foo" @RWMemory(RW0_addr: %addr: i10, RW0_en: %enable: i1, RW0_clk: %clock: i1, RW0_wmode: %wmode: i1, RW0_wdata: %wdata: i8) -> (RW0_rdata: i8)
  hw.output %0 : i8
}
// CHECK-NEXT: }
// CHECK-NOT: hw.module.generated @RWMemory, @FIRRTLMem
hw.module.generated @RWMemory, @FIRRTLMem(%RW0_addr: i10, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i8) -> (RW0_rdata: i8) attributes {depth = 1024 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
