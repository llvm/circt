// RUN: circt-opt %s | circt-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// Test probe.send with type inference
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @BasicProbe
hw.module @BasicProbe(in %value: i32) {
  %wire = hw.wire %value : i32
  
  // Basic probe.send - infers !hw.probe<i32>
  // CHECK: %{{.+}} = hw.probe.send %wire : i32
  %probe = hw.probe.send %wire : i32
  
  hw.output
}

// CHECK-LABEL: hw.module @ForceableProbe
hw.module @ForceableProbe(in %value: i16) {
  %wire = hw.wire %value : i16
  
  // Forceable probe.send - infers !hw.rwprobe<i16>
  // CHECK: %{{.+}} = hw.probe.send forceable %wire : i16
  %rwprobe = hw.probe.send forceable %wire : i16
  
  hw.output
}

// CHECK-LABEL: hw.module @MemoryProbe
hw.module @MemoryProbe() {
  // Memory - probe.send infers !hw.probe<!hw.array<4xi32>>
  %mem = seq.firmem 0, 0, undefined, undefined : <4 x 32>
  
  // CHECK: %{{.+}} = hw.probe.send %mem : !seq.firmem<4 x 32>
  %mem_probe = hw.probe.send %mem : !seq.firmem<4 x 32>
  
  hw.output
}

// CHECK-LABEL: hw.module @ForceableMemoryProbe
hw.module @ForceableMemoryProbe() {
  // Memory with forceable - infers !hw.rwprobe<!hw.array<8xi16>>
  %mem = seq.firmem 0, 0, undefined, undefined : <8 x 16>
  
  // CHECK: %{{.+}} = hw.probe.send forceable %mem : !seq.firmem<8 x 16>
  %rw_mem_probe = hw.probe.send forceable %mem : !seq.firmem<8 x 16>
  
  hw.output
}

// CHECK-LABEL: hw.module @StructProbe
hw.module @StructProbe() {
  %c0_struct = hw.aggregate_constant [10 : i8, 20 : i16] : !hw.struct<a: i8, b: i16>
  %struct_wire = hw.wire %c0_struct : !hw.struct<a: i8, b: i16>
  
  // Struct probe
  // CHECK: %{{.+}} = hw.probe.send %struct_wire : !hw.struct<a: i8, b: i16>
  %struct_probe = hw.probe.send %struct_wire : !hw.struct<a: i8, b: i16>
  
  hw.output
}

// CHECK-LABEL: hw.module @ArrayProbe
hw.module @ArrayProbe() {
  %c0_array = hw.aggregate_constant [1 : i4, 2 : i4, 3 : i4, 4 : i4] : !hw.array<4xi4>
  %array_wire = hw.wire %c0_array : !hw.array<4xi4>
  
  // Array probe
  // CHECK: %{{.+}} = hw.probe.send %array_wire : !hw.array<4xi4>
  %array_probe = hw.probe.send %array_wire : !hw.array<4xi4>
  
  hw.output
}

// CHECK-LABEL: hw.module @ProbeWithResolve
hw.module @ProbeWithResolve(out result: i64) {
  %c42_i64 = hw.constant 42 : i64
  %wire = hw.wire %c42_i64 : i64
  
  // Send and resolve
  // CHECK: %[[PROBE:.+]] = hw.probe.send %wire : i64
  %probe = hw.probe.send %wire : i64
  
  // CHECK: %{{.+}} = hw.probe.resolve %[[PROBE]] : !hw.probe<i64>
  %value = hw.probe.resolve %probe : !hw.probe<i64>
  
  hw.output %value : i64
}

// CHECK-LABEL: hw.module @ForceableWithForce
hw.module @ForceableWithForce(in %clk: i1, in %enable: i1, in %force_val: i32) {
  %c0_i32 = hw.constant 0 : i32
  %wire = hw.wire %c0_i32 : i32
  
  // Create forceable probe
  // CHECK: %[[RWPROBE:.+]] = hw.probe.send forceable %wire : i32
  %rwprobe = hw.probe.send forceable %wire : i32
  
  // Use it with force
  // CHECK: hw.probe.force %clk, %enable, %[[RWPROBE]], %force_val
  hw.probe.force %clk, %enable, %rwprobe, %force_val : i1, i1, !hw.rwprobe<i32>, i32
  
  hw.output
}

// CHECK-LABEL: hw.module @ProbeInModulePort
hw.module @ProbeInModulePort(in %value: i8, out probe_out: !hw.probe<i8>) {
  %wire = hw.wire %value : i8
  
  // CHECK: %{{.+}} = hw.probe.send %wire : i8
  %probe = hw.probe.send %wire : i8
  
  hw.output %probe : !hw.probe<i8>
}

// CHECK-LABEL: hw.module @RWProbeInModulePort
hw.module @RWProbeInModulePort(in %value: i16, out probe_out: !hw.rwprobe<i16>) {
  %wire = hw.wire %value : i16
  
  // CHECK: %{{.+}} = hw.probe.send forceable %wire : i16
  %rwprobe = hw.probe.send forceable %wire : i16
  
  hw.output %rwprobe : !hw.rwprobe<i16>
}

// CHECK-LABEL: hw.module @MemoryProbeInPort
hw.module @MemoryProbeInPort(out mem_probe: !hw.probe<!hw.array<16xi8>>) {
  %mem = seq.firmem 0, 0, undefined, undefined : <16 x 8>
  
  // CHECK: %{{.+}} = hw.probe.send %mem : !seq.firmem<16 x 8>
  %probe = hw.probe.send %mem : !seq.firmem<16 x 8>
  
  hw.output %probe : !hw.probe<!hw.array<16xi8>>
}

// CHECK-LABEL: hw.module @ForceableMemoryProbeInPort
hw.module @ForceableMemoryProbeInPort(out mem_probe: !hw.rwprobe<!hw.array<32xi32>>) {
  %mem = seq.firmem 0, 0, undefined, undefined : <32 x 32>
  
  // CHECK: %{{.+}} = hw.probe.send forceable %mem : !seq.firmem<32 x 32>
  %rwprobe = hw.probe.send forceable %mem : !seq.firmem<32 x 32>
  
  hw.output %rwprobe : !hw.rwprobe<!hw.array<32xi32>>
}
