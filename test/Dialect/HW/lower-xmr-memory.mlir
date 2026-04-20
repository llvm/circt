// RUN: circt-opt %s --hw-lower-xmr -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Test memory debug port via hw.probe.send with type inference
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@MemoryDebugTest::@mem_storage]
// CHECK-LABEL: hw.module @MemoryDebugTest
hw.module @MemoryDebugTest(in %clk: !seq.clock, in %addr: i2, out data: i32, out mem_data: !hw.array<4xi32>) {
  // Create a memory with symbol
  // CHECK: %mem = seq.firmem sym @mem_storage
  %mem = seq.firmem sym @mem_storage 0, 0, undefined, undefined : <4 x 32>

  // Send a probe to the memory (type automatically inferred to !hw.probe<!hw.array<4xi32>>)
  // CHECK-NOT: hw.probe.send
  %mem_probe = hw.probe.send %mem : !seq.firmem<4 x 32>

  // Resolve the probe to get memory contents
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory" : !hw.inout<array<4xi32>>
  // CHECK: %[[MEM_DATA:.+]] = sv.read_inout %[[XMR]]
  %mem_contents = hw.probe.resolve %mem_probe : !hw.probe<!hw.array<4xi32>>

  // Read from memory
  // CHECK: %[[DATA:.+]] = seq.firmem.read_port
  %data_val = seq.firmem.read_port %mem[%addr], clock %clk : <4 x 32>

  // CHECK: hw.output %[[DATA]], %[[MEM_DATA]]
  hw.output %data_val, %mem_contents : i32, !hw.array<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Test memory probe through hierarchy (unused, so optimized away)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @MemLeaf
hw.module @MemLeaf(in %clk: !seq.clock) {
  // CHECK: %mem = seq.firmem sym @leaf_mem
  %mem = seq.firmem sym @leaf_mem 0, 0, undefined, undefined : <16 x 8>
  %mem_probe = hw.probe.send %mem : !seq.firmem<16 x 8>
  // CHECK: hw.output
  hw.output
}

// CHECK-LABEL: hw.module @MemMiddle
hw.module @MemMiddle(in %clk: !seq.clock) {
  // CHECK: hw.instance "leaf"
  hw.instance "leaf" sym @leaf @MemLeaf(clk: %clk: !seq.clock) -> ()
  hw.output
}

// CHECK-LABEL: hw.module @MemTop
hw.module @MemTop(in %clk: !seq.clock) {
  // CHECK: hw.instance "mid"
  hw.instance "mid" sym @mid @MemMiddle(clk: %clk: !seq.clock) -> ()
  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test regular wire with forceable (not memory)
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@ForceableWireTest::@xmr_sym]
// CHECK-LABEL: hw.module @ForceableWireTest
hw.module @ForceableWireTest(in %clk: i1, in %enable: i1, in %value: i32) {
  // CHECK: %c0_i32 = hw.constant
  %c0_i32 = hw.constant 0 : i32
  // CHECK: %wire = hw.wire %c0_i32 sym @xmr_sym
  %wire = hw.wire %c0_i32 : i32

  // Forceable probe on regular wire
  // CHECK-NOT: hw.probe.send
  %rw_probe = hw.probe.send forceable %wire : i32

  // Force the wire
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: sv.always posedge %clk
  // CHECK: sv.force %[[XMR]], %value
  hw.probe.force %clk, %enable, %rw_probe, %value : i1, i1, !hw.rwprobe<i32>, i32

  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test mixed probe types (probe and rwprobe)
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@MixedProbeTypes::@mixed_mem]
// CHECK-LABEL: hw.module @MixedProbeTypes
hw.module @MixedProbeTypes(in %clk: !seq.clock) {
  // CHECK: %mem = seq.firmem sym @mixed_mem
  %mem = seq.firmem sym @mixed_mem 0, 0, undefined, undefined : <4 x 32>

  // Read-only probe
  // CHECK-NOT: hw.probe.send
  %ro_probe = hw.probe.send %mem : !seq.firmem<4 x 32>

  // Read-write probe
  // CHECK-NOT: hw.probe.send
  %rw_probe = hw.probe.send forceable %mem : !seq.firmem<4 x 32>

  // Resolve ro_probe
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory"
  // CHECK: %{{.+}} = sv.read_inout %[[XMR]]
  %mem_contents = hw.probe.resolve %ro_probe : !hw.probe<!hw.array<4xi32>>

  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test memory probe with sub operations
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@MemProbeWithSub::@sub_mem]
// CHECK-LABEL: hw.module @MemProbeWithSub
hw.module @MemProbeWithSub(in %clk: !seq.clock, out elem: i8) {
  // CHECK: %mem = seq.firmem sym @sub_mem
  %mem = seq.firmem sym @sub_mem 0, 0, undefined, undefined : <8 x 8>

  // Send probe to memory
  // CHECK-NOT: hw.probe.send
  %mem_probe = hw.probe.send %mem : !seq.firmem<8 x 8>

  // Access specific element via sub
  // CHECK-NOT: hw.probe.sub
  %elem_probe = hw.probe.sub %mem_probe[3] : !hw.probe<!hw.array<8xi8>>

  // Resolve the element (suffix combines Memory + [3])
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory[3]"
  // CHECK: %[[ELEM:.+]] = sv.read_inout %[[XMR]]
  %elem_val = hw.probe.resolve %elem_probe : !hw.probe<i8>

  // CHECK: hw.output %[[ELEM]]
  hw.output %elem_val : i8
}

// -----

//===----------------------------------------------------------------------===//
// Test zero-width memory - should be optimized away
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @ZeroWidthMem
hw.module @ZeroWidthMem(in %clk: !seq.clock) {
  // CHECK: %mem = seq.firmem sym @xmr_sym
  %mem = seq.firmem sym @xmr_sym 0, 0, undefined, undefined : <4 x 0>

  // Probe to zero-width memory should be optimized away
  // CHECK-NOT: hw.probe.send
  // CHECK-NOT: sv.xmr.ref
  %mem_probe = hw.probe.send %mem : !seq.firmem<4 x 0>

  // CHECK: hw.output
  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test zero-width wire probe - should be optimized away
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @ZeroWidthWireProbe
hw.module @ZeroWidthWireProbe(out result: i0) {
  %c0_i0 = hw.constant 0 : i0
  %wire = hw.wire %c0_i0 : i0

  // Probe to zero-width wire should be optimized away
  // CHECK-NOT: hw.probe.send
  %probe = hw.probe.send %wire : i0

  // Resolve zero-width probe
  // CHECK-NOT: sv.xmr.ref
  %val = hw.probe.resolve %probe : !hw.probe<i0>

  // CHECK: hw.output %c0_i0
  hw.output %val : i0
}

// -----

//===----------------------------------------------------------------------===//
// Test forceable memory with force operation
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@ForceMemTest::@force_mem]
// CHECK-LABEL: hw.module @ForceMemTest
hw.module @ForceMemTest(in %clk: i1, in %enable: i1) {
  // CHECK: %mem = seq.firmem sym @force_mem
  %mem = seq.firmem sym @force_mem 0, 0, undefined, undefined : <2 x 8>

  // Create forceable probe
  // CHECK-NOT: hw.probe.send
  %rw_probe = hw.probe.send forceable %mem : !seq.firmem<2 x 8>

  // Force the memory
  %force_data = hw.aggregate_constant [42 : i8, 43 : i8] : !hw.array<2xi8>
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory"
  // CHECK: sv.always posedge %clk
  // CHECK: sv.force %[[XMR]]
  hw.probe.force %clk, %enable, %rw_probe, %force_data : i1, i1, !hw.rwprobe<!hw.array<2xi8>>, !hw.array<2xi8>

  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test release operations on memory
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@ReleaseMemTest::@release_mem]
// CHECK-LABEL: hw.module @ReleaseMemTest
hw.module @ReleaseMemTest(in %clk: i1, in %enable: i1) {
  // CHECK: %mem = seq.firmem sym @release_mem
  %mem = seq.firmem sym @release_mem 0, 0, undefined, undefined : <4 x 16>

  // Create forceable probe
  // CHECK-NOT: hw.probe.send
  %rw_probe = hw.probe.send forceable %mem : !seq.firmem<4 x 16>

  // Release the memory
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory"
  // CHECK: sv.always posedge %clk
  // CHECK: sv.release %[[XMR]]
  hw.probe.release %clk, %enable, %rw_probe : i1, i1, !hw.rwprobe<!hw.array<4xi16>>

  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test initial force on memory
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@ForceInitialMemTest::@init_mem]
// CHECK-LABEL: hw.module @ForceInitialMemTest
hw.module @ForceInitialMemTest(in %enable: i1) {
  // CHECK: %mem = seq.firmem sym @init_mem
  %mem = seq.firmem sym @init_mem 0, 0, undefined, undefined : <2 x 32>

  // Create forceable probe
  // CHECK-NOT: hw.probe.send
  %rw_probe = hw.probe.send forceable %mem : !seq.firmem<2 x 32>

  // Force initial
  %force_data = hw.aggregate_constant [100 : i32, 200 : i32] : !hw.array<2xi32>
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory"
  // CHECK: sv.initial
  // CHECK: sv.force %[[XMR]]
  hw.probe.force_initial %enable, %rw_probe, %force_data : i1, !hw.rwprobe<!hw.array<2xi32>>, !hw.array<2xi32>

  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test release initial on memory
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath private @[[PATH:.+]] [@ReleaseInitialMemTest::@rel_init_mem]
// CHECK-LABEL: hw.module @ReleaseInitialMemTest
hw.module @ReleaseInitialMemTest(in %enable: i1) {
  // CHECK: %mem = seq.firmem sym @rel_init_mem
  %mem = seq.firmem sym @rel_init_mem 0, 0, undefined, undefined : <8 x 8>

  // Create forceable probe
  // CHECK-NOT: hw.probe.send
  %rw_probe = hw.probe.send forceable %mem : !seq.firmem<8 x 8>

  // Release initial
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory"
  // CHECK: sv.initial
  // CHECK: sv.release %[[XMR]]
  hw.probe.release_initial %enable, %rw_probe : i1, !hw.rwprobe<!hw.array<8xi8>>

  hw.output
}
