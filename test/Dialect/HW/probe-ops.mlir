// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @ProbeTypes
hw.module @ProbeTypes() {
  // CHECK: %wire = hw.wire %c42_i32 : i32
  %c42_i32 = hw.constant 42 : i32
  %wire = hw.wire %c42_i32 : i32
  
  // CHECK: %{{.+}} = hw.probe.send %wire : i32
  %probe_ref = hw.probe.send %wire : i32
  
  // CHECK: %{{.+}} = hw.probe.resolve %{{.+}} : !hw.probe<i32>
  %value = hw.probe.resolve %probe_ref : !hw.probe<i32>
}

// CHECK-LABEL: hw.module @ProbeSend
hw.module @ProbeSend(in %input: i8, out result: !hw.probe<i8>) {
  // CHECK: %[[WIRE:.+]] = hw.wire %input : i8
  %wire = hw.wire %input : i8
  
  // CHECK: %[[REF:.+]] = hw.probe.send %[[WIRE]] : i8
  %ref = hw.probe.send %wire : i8
  
  // CHECK: hw.output %[[REF]] : !hw.probe<i8>
  hw.output %ref : !hw.probe<i8>
}

// CHECK-LABEL: hw.module @ProbeResolve
hw.module @ProbeResolve(in %probe_in: !hw.probe<i16>, out value: i16) {
  // CHECK: %[[VAL:.+]] = hw.probe.resolve %probe_in : !hw.probe<i16>
  %value_out = hw.probe.resolve %probe_in : !hw.probe<i16>
  
  // CHECK: hw.output %[[VAL]] : i16
  hw.output %value_out : i16
}

// CHECK-LABEL: hw.module @ProbeRWProbe
hw.module @ProbeRWProbe() {
  %c0_i32 = hw.constant 0 : i32
  %reg = hw.wire %c0_i32 : i32

  // Using probe.send forceable to create rwprobe
  // CHECK: %{{.+}} = hw.probe.send forceable %{{.+}} : i32
  %rwref = hw.probe.send forceable %reg : i32
}

// CHECK-LABEL: hw.module @ProbeRWProbeViaSymbol
hw.module @ProbeRWProbeViaSymbol() {
  %c0_i32 = hw.constant 0 : i32
  %reg = hw.wire %c0_i32 sym @myReg : i32

  // Alternative: using symbol reference for rwprobe
  // CHECK: %{{.+}} = hw.probe.rwprobe @ProbeRWProbeViaSymbol::@myReg : !hw.rwprobe<i32>
  %rwref = hw.probe.rwprobe @ProbeRWProbeViaSymbol::@myReg : !hw.rwprobe<i32>
}

// CHECK-LABEL: hw.module @ProbeDefine
hw.module @ProbeDefine(in %src_probe: !hw.rwprobe<i32>) {
  %c0_i32 = hw.constant 0 : i32
  %dest_wire = hw.wire %c0_i32 sym @dest : i32
  %dest_probe = hw.probe.rwprobe @ProbeDefine::@dest : !hw.rwprobe<i32>

  // CHECK: hw.probe.define %{{.+}}, %src_probe : !hw.rwprobe<i32>
  hw.probe.define %dest_probe, %src_probe : !hw.rwprobe<i32>

  hw.output
}



// CHECK-LABEL: hw.module @ProbeArrays
hw.module @ProbeArrays() {
  %c0_i32 = hw.constant 0 : i32
  %arr = hw.array_create %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32
  
  // CHECK: %[[ARR_REF:.+]] = hw.probe.send %{{.+}} : !hw.array<4xi32>
  %arr_ref = hw.probe.send %arr : !hw.array<4xi32>
  
  // CHECK: %[[ELEM_REF:.+]] = hw.probe.sub %[[ARR_REF]][2 : i32] : !hw.probe<!hw.array<4xi32>>
  %elem_ref = hw.probe.sub %arr_ref[2 : i32] : !hw.probe<!hw.array<4xi32>>
  
  // CHECK: %{{.+}} = hw.probe.resolve %[[ELEM_REF]] : !hw.probe<i32>
  %elem_val = hw.probe.resolve %elem_ref : !hw.probe<i32>
}

// CHECK-LABEL: hw.module @ProbeStructs
hw.module @ProbeStructs() {
  %c1_i8 = hw.constant 1 : i8
  %c2_i16 = hw.constant 2 : i16
  %struct = hw.struct_create (%c1_i8, %c2_i16) : !hw.struct<a: i8, b: i16>
  
  // CHECK: %[[STRUCT_REF:.+]] = hw.probe.send %{{.+}} : !hw.struct<a: i8, b: i16>
  %struct_ref = hw.probe.send %struct : !hw.struct<a: i8, b: i16>
  
  // CHECK: %[[FIELD_REF:.+]] = hw.probe.sub %[[STRUCT_REF]][1 : i32] : !hw.probe<!hw.struct<a: i8, b: i16>>
  %field_ref = hw.probe.sub %struct_ref[1 : i32] : !hw.probe<!hw.struct<a: i8, b: i16>>
  
  // CHECK: %{{.+}} = hw.probe.resolve %[[FIELD_REF]] : !hw.probe<i16>
  %field_val = hw.probe.resolve %field_ref : !hw.probe<i16>
}

// CHECK-LABEL: hw.module @ProbeCast
hw.module @ProbeCast() {
  %c0_i32 = hw.constant 0 : i32
  %reg = hw.wire %c0_i32 sym @reg : i32

  // Create RWProbe
  // CHECK: %[[RWREF:.+]] = hw.probe.rwprobe @ProbeCast::@reg : !hw.rwprobe<i32>
  %rwref = hw.probe.rwprobe @ProbeCast::@reg : !hw.rwprobe<i32>

  // Cast RWProbe to Probe (demotion)
  // CHECK: %[[PROBE:.+]] = hw.probe.cast %[[RWREF]] : !hw.rwprobe<i32> -> !hw.probe<i32>
  %probe = hw.probe.cast %rwref : !hw.rwprobe<i32> -> !hw.probe<i32>

  // CHECK: %{{.+}} = hw.probe.resolve %[[PROBE]] : !hw.probe<i32>
  %value = hw.probe.resolve %probe : !hw.probe<i32>
}

// CHECK-LABEL: hw.module @ProbeForce
hw.module @ProbeForce(in %clock: i1, in %enable: i1) {
  %c0_i32 = hw.constant 0 : i32
  %reg = hw.wire %c0_i32 : i32

  // Using probe.send forceable to create rwprobe
  // CHECK: %[[RWREF:.+]] = hw.probe.send forceable %{{.+}} : i32
  %rwref = hw.probe.send forceable %reg : i32

  %c123_i32 = hw.constant 123 : i32

  // CHECK: hw.probe.force %clock, %enable, %[[RWREF]], %{{.+}} : i1, i1, !hw.rwprobe<i32>, i32
  hw.probe.force %clock, %enable, %rwref, %c123_i32 : i1, i1, !hw.rwprobe<i32>, i32
}

// CHECK-LABEL: hw.module @ProbeForceInitial
hw.module @ProbeForceInitial(in %enable: i1) {
  %c0_i16 = hw.constant 0 : i16
  %reg = hw.wire %c0_i16 : i16

  // Using probe.send forceable to create rwprobe
  // CHECK: %[[RWREF:.+]] = hw.probe.send forceable %{{.+}} : i16
  %rwref = hw.probe.send forceable %reg : i16

  %c999_i16 = hw.constant 999 : i16

  // CHECK: hw.probe.force_initial %enable, %[[RWREF]], %{{.+}} : i1, !hw.rwprobe<i16>, i16
  hw.probe.force_initial %enable, %rwref, %c999_i16 : i1, !hw.rwprobe<i16>, i16
}

// CHECK-LABEL: hw.module @ProbeRelease
hw.module @ProbeRelease(in %clock: i1, in %enable: i1) {
  %c0_i32 = hw.constant 0 : i32
  %reg = hw.wire %c0_i32 : i32

  // Using probe.send forceable to create rwprobe
  // CHECK: %[[RWREF:.+]] = hw.probe.send forceable %{{.+}} : i32
  %rwref = hw.probe.send forceable %reg : i32

  // CHECK: hw.probe.release %clock, %enable, %[[RWREF]] : i1, i1, !hw.rwprobe<i32>
  hw.probe.release %clock, %enable, %rwref : i1, i1, !hw.rwprobe<i32>
}

// CHECK-LABEL: hw.module @ProbeReleaseInitial
hw.module @ProbeReleaseInitial(in %enable: i1) {
  %c0_i16 = hw.constant 0 : i16
  %reg = hw.wire %c0_i16 : i16

  // Using probe.send forceable to create rwprobe
  // CHECK: %[[RWREF:.+]] = hw.probe.send forceable %{{.+}} : i16
  %rwref = hw.probe.send forceable %reg : i16

  // CHECK: hw.probe.release_initial %enable, %[[RWREF]] : i1, !hw.rwprobe<i16>
  hw.probe.release_initial %enable, %rwref : i1, !hw.rwprobe<i16>
}

// CHECK-LABEL: hw.module @ProbeHierarchy
hw.module @ProbeHierarchy(out probe_out: !hw.probe<i64>) {
  %c42_i64 = hw.constant 42 : i64
  %wire = hw.wire %c42_i64 : i64

  // CHECK: %[[REF:.+]] = hw.probe.send %{{.+}} : i64
  %ref = hw.probe.send %wire : i64

  // CHECK: hw.output %[[REF]] : !hw.probe<i64>
  hw.output %ref : !hw.probe<i64>
}

// CHECK-LABEL: hw.module @ProbeParent
hw.module @ProbeParent() {
  // CHECK: %[[CHILD_REF:.+]] = hw.instance "child" @ProbeHierarchy() -> (probe_out: !hw.probe<i64>)
  %child_ref = hw.instance "child" @ProbeHierarchy() -> (probe_out: !hw.probe<i64>)

  // CHECK: %{{.+}} = hw.probe.resolve %[[CHILD_REF]] : !hw.probe<i64>
  %value = hw.probe.resolve %child_ref : !hw.probe<i64>
}

// CHECK-LABEL: hw.module @NestedProbes
hw.module @NestedProbes() {
  // Nested array of structs
  %c0_i8 = hw.constant 0 : i8
  %c1_i16 = hw.constant 1 : i16
  %struct = hw.struct_create (%c0_i8, %c1_i16) : !hw.struct<x: i8, y: i16>
  %arr = hw.array_create %struct, %struct : !hw.struct<x: i8, y: i16>

  // CHECK: %[[ARR_REF:.+]] = hw.probe.send %{{.+}} : !hw.array<2xstruct<x: i8, y: i16>>
  %arr_ref = hw.probe.send %arr : !hw.array<2xstruct<x: i8, y: i16>>

  // Index into array
  // CHECK: %[[ELEM_REF:.+]] = hw.probe.sub %[[ARR_REF]][0 : i32] : !hw.probe<!hw.array<2xstruct<x: i8, y: i16>>>
  %elem_ref = hw.probe.sub %arr_ref[0 : i32] : !hw.probe<!hw.array<2xstruct<x: i8, y: i16>>>

  // Index into struct field
  // CHECK: %[[FIELD_REF:.+]] = hw.probe.sub %[[ELEM_REF]][1 : i32] : !hw.probe<!hw.struct<x: i8, y: i16>>
  %field_ref = hw.probe.sub %elem_ref[1 : i32] : !hw.probe<!hw.struct<x: i8, y: i16>>

  // Resolve final value
  // CHECK: %{{.+}} = hw.probe.resolve %[[FIELD_REF]] : !hw.probe<i16>
  %value = hw.probe.resolve %field_ref : !hw.probe<i16>
}

// CHECK-LABEL: hw.module @MultipleProbesOnSameValue
hw.module @MultipleProbesOnSameValue() {
  %c100_i32 = hw.constant 100 : i32
  %wire = hw.wire %c100_i32 : i32

  // Multiple probes can reference the same value
  // CHECK: %[[REF1:.+]] = hw.probe.send %{{.+}} : i32
  %ref1 = hw.probe.send %wire : i32

  // CHECK: %[[REF2:.+]] = hw.probe.send %{{.+}} : i32
  %ref2 = hw.probe.send %wire : i32

  // Both can be resolved independently
  // CHECK: %{{.+}} = hw.probe.resolve %[[REF1]] : !hw.probe<i32>
  %val1 = hw.probe.resolve %ref1 : !hw.probe<i32>

  // CHECK: %{{.+}} = hw.probe.resolve %[[REF2]] : !hw.probe<i32>
  %val2 = hw.probe.resolve %ref2 : !hw.probe<i32>
}

// CHECK-LABEL: hw.module @ProbeWithDifferentWidths
hw.module @ProbeWithDifferentWidths() {
  // Test probes with various bit widths
  %c1_i1 = hw.constant 1 : i1
  %c255_i8 = hw.constant 255 : i8
  %c65535_i16 = hw.constant 65535 : i16
  %c4294967295_i32 = hw.constant 4294967295 : i32

  // CHECK: %{{.+}} = hw.probe.send %{{.+}} : i1
  %ref1 = hw.probe.send %c1_i1 : i1

  // CHECK: %{{.+}} = hw.probe.send %{{.+}} : i8
  %ref8 = hw.probe.send %c255_i8 : i8

  // CHECK: %{{.+}} = hw.probe.send %{{.+}} : i16
  %ref16 = hw.probe.send %c65535_i16 : i16

  // CHECK: %{{.+}} = hw.probe.send %{{.+}} : i32
  %ref32 = hw.probe.send %c4294967295_i32 : i32
}

// Test that probe types print and parse correctly
// CHECK-LABEL: hw.module @ProbeTypePrinting(in %p1 : !hw.probe<i32>, in %p2 : !hw.rwprobe<i64>, in %p3 : !hw.probe<!hw.array<4xi8>>, in %p4 : !hw.rwprobe<!hw.struct<a: i8, b: i16>>)
hw.module @ProbeTypePrinting(
  in %p1: !hw.probe<i32>,
  in %p2: !hw.rwprobe<i64>,
  in %p3: !hw.probe<!hw.array<4xi8>>,
  in %p4: !hw.rwprobe<!hw.struct<a: i8, b: i16>>
) {
  hw.output
}
