// RUN: circt-opt %s --arc-infer-memories="tap-memories=0" | FileCheck %s --check-prefixes=CHECK,CHECK-TAP-OFF
// RUN: circt-opt %s --arc-infer-memories="tap-memories=1" | FileCheck %s --check-prefixes=CHECK,CHECK-TAP-ON

// CHECK-LABEL: hw.module @TestMemory(
hw.module @TestMemory(in %clock: !seq.clock, in %addr: i10, in %enable: i1, in %data: i8) {
  // CHECK: arc.memory <1024 x i8, i10>
  // CHECK-TAP-OFF-NOT: name = "foo"
  // CHECK-TAP-ON: name = "foo"
  hw.instance "foo" @Memory(W0_addr: %addr: i10, W0_en: %enable: i1, W0_clk: %clock: !seq.clock, W0_data: %data: i8) -> ()
}

hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]
hw.module.generated @Memory, @FIRRTLMem(in %W0_addr: i10, in %W0_en: i1, in %W0_clk: !seq.clock, in %W0_data: i8) attributes {depth = 1024 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : ui32, width = 8 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

