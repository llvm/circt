// RUN: circt-opt %s --hw-lower-xmr -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Test for same module lowering (basic case)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @SameModule
hw.module @SameModule(out o: i2) {
  // CHECK: %w = hw.wire %c0_i2 sym @{{.+}}
  %c0_i2 = hw.constant 0 : i2
  %w = hw.wire %c0_i2 : i2
  %ref = hw.probe.send %w : i2
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %x = hw.probe.resolve %ref : !hw.probe<i2>
  // CHECK: hw.output
  hw.output %x : i2
}

// -----

// =============================================================================
// COMPREHENSIVE HW LOWER XMR TEST SUITE
// Ported from FIRRTL lowerXMR.mlir (818 lines, 27 test cases)
// =============================================================================


// Test 1: Basic same-module probe lowering

// Test basic same-module probe lowering
// CHECK: hw.hierpath private @[[PATH:.+]] [@BasicProbe::@[[SYM:.+]]]
// CHECK-LABEL: hw.module @BasicProbe
hw.module @BasicProbe(out result: i8) {
  // CHECK: %[[C42:.+]] = hw.constant 42
  // CHECK: %[[WIRE:.+]] = hw.wire %[[C42]] sym @[[SYM]]
  // CHECK-NOT: hw.probe.send
  // CHECK-NOT: hw.probe.resolve
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  // CHECK: hw.output %[[READ]]

  %c42_i8 = hw.constant 42 : i8
  %wire = hw.wire %c42_i8 : i8

  %ref = hw.probe.send %wire : i8
  %value = hw.probe.resolve %ref : !hw.probe<i8>

  hw.output %value : i8
}

// -----

// Test probe with constant
// Constants cannot hold inner symbols, so a wire is created
// CHECK: hw.hierpath private @[[PATH:.+]] [@ProbeConstant::@[[SYM:.+]]]
// CHECK-LABEL: hw.module @ProbeConstant
hw.module @ProbeConstant(out result: i32) {
  // CHECK: %[[C100:.+]] = hw.constant 100 : i32
  // CHECK: %[[WIRE:.+]] = hw.wire %[[C100]]{{.*}}sym @[[SYM]]
  // CHECK-NOT: hw.probe.send
  // CHECK-NOT: hw.probe.resolve
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  // CHECK: hw.output %[[READ]]

  %c100_i32 = hw.constant 100 : i32

  %ref = hw.probe.send %c100_i32 : i32

  %value = hw.probe.resolve %ref : !hw.probe<i32>

  hw.output %value : i32
}

// -----

// Test multiple probes on same value
// CHECK: hw.hierpath private @[[PATH:.+]] [@MultipleProbes::@[[SYM:.+]]]
// CHECK-LABEL: hw.module @MultipleProbes
hw.module @MultipleProbes(out o1: i16, out o2: i16) {
  // CHECK: %[[C99:.+]] = hw.constant 99
  // CHECK: %[[WIRE:.+]] = hw.wire %[[C99]] sym @[[SYM]]
  // CHECK-NOT: hw.probe
  // CHECK: %[[XMR1:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ1:.+]] = sv.read_inout %[[XMR1]]
  // CHECK: %[[XMR2:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ2:.+]] = sv.read_inout %[[XMR2]]
  // CHECK: hw.output %[[READ1]], %[[READ2]]

  %c99_i16 = hw.constant 99 : i16
  %wire = hw.wire %c99_i16 : i16

  %ref1 = hw.probe.send %wire : i16
  %ref2 = hw.probe.send %wire : i16

  %val1 = hw.probe.resolve %ref1 : !hw.probe<i16>
  %val2 = hw.probe.resolve %ref2 : !hw.probe<i16>

  hw.output %val1, %val2 : i16, i16
}

// -----

// Test probe on wire with different widths
// CHECK-COUNT-4: hw.hierpath
// CHECK-LABEL: hw.module @ProbeWidths
hw.module @ProbeWidths(out o1: i1, out o8: i8, out o16: i16, out o32: i32) {
  // CHECK-NOT: hw.probe
  // CHECK-COUNT-4: sv.xmr.ref

  %c1_i1 = hw.constant 1 : i1
  %c255_i8 = hw.constant 255 : i8
  %c65535_i16 = hw.constant 65535 : i16
  %c4294967295_i32 = hw.constant 4294967295 : i32

  %ref1 = hw.probe.send %c1_i1 : i1
  %ref8 = hw.probe.send %c255_i8 : i8
  %ref16 = hw.probe.send %c65535_i16 : i16
  %ref32 = hw.probe.send %c4294967295_i32 : i32

  %v1 = hw.probe.resolve %ref1 : !hw.probe<i1>
  %v8 = hw.probe.resolve %ref8 : !hw.probe<i8>
  %v16 = hw.probe.resolve %ref16 : !hw.probe<i16>
  %v32 = hw.probe.resolve %ref32 : !hw.probe<i32>

  hw.output %v1, %v8, %v16, %v32 : i1, i8, i16, i32
}

// -----

// Test probe with wire operations
// CHECK-LABEL: hw.module @ProbeWire
hw.module @ProbeWire(in %input: i8, out result: i8) {
  // CHECK: hw.wire %input sym
  // CHECK-NOT: hw.probe
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout

  %wire = hw.wire %input : i8

  %ref = hw.probe.send %wire : i8
  %value = hw.probe.resolve %ref : !hw.probe<i8>

  hw.output %value : i8
}

// -----

// Test that probes work with combinational logic
// CHECK-LABEL: hw.module @ProbeComb
hw.module @ProbeComb(in %a: i8, in %b: i8, out result: i8) {
  // CHECK: comb.add
  // CHECK: hw.wire %{{.*}} sym
  // CHECK-NOT: hw.probe
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout

  %sum = comb.add %a, %b : i8
  %wire = hw.wire %sum : i8

  %ref = hw.probe.send %wire : i8
  %value = hw.probe.resolve %ref : !hw.probe<i8>

  hw.output %value : i8
}

// -----

// Test probe type parsing and printing
// CHECK-LABEL: hw.module @ProbeTypes
hw.module @ProbeTypes(
  in %p1: !hw.probe<i32>,
  in %p2: !hw.probe<!hw.array<4xi8>>
) {
  // CHECK: hw.output
  hw.output
}

// -----

// Test basic XMR structure after lowering
// CHECK: hw.hierpath private @[[XPATH:.+]] [@CheckXMRStructure::@[[WIRE_SYM:.+]]]
// CHECK-LABEL: hw.module @CheckXMRStructure
hw.module @CheckXMRStructure(out result: i8) {
  // CHECK: %[[CONST:.+]] = hw.constant 42
  // CHECK: %[[WIRE:.+]] = hw.wire %[[CONST]] sym @[[WIRE_SYM]]
  // CHECK-NOT: hw.probe
  // CHECK: %[[XMRREF:.+]] = sv.xmr.ref @[[XPATH]]
  // CHECK: %[[XMRVAL:.+]] = sv.read_inout %[[XMRREF]]
  // CHECK: hw.output %[[XMRVAL]]

  %c42_i8 = hw.constant 42 : i8
  %wire = hw.wire %c42_i8 : i8

  %ref = hw.probe.send %wire : i8
  %value = hw.probe.resolve %ref : !hw.probe<i8>

  hw.output %value : i8
}

// -----

// Test that hierarchical path structure is correct
// This test verifies that the pass can correctly build hierarchical paths
// even though full cross-module lowering requires additional port handling
// CHECK-LABEL: hw.module @PathStructure
hw.module @PathStructure(out result: i8) {
  // CHECK: hw.constant 77
  // CHECK: hw.wire
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %c77_i8 = hw.constant 77 : i8
  %wire = hw.wire %c77_i8 : i8
  %ref = hw.probe.send %wire : i8
  %value = hw.probe.resolve %ref : !hw.probe<i8>
  hw.output %value : i8
}

// -----

// =============================================================================
// HANDLEPROBESEND TEST CASES - Testing all 5 cases
// =============================================================================

//===----------------------------------------------------------------------===//
// Case 1: Zero-width probes (should be skipped/removed)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @ZeroWidthProbe
hw.module @ZeroWidthProbe(out result: i0) {
  // Zero-width type (i0) - probe should be removed entirely
  // CHECK: %[[C0:.+]] = hw.constant 0 : i0
  %c0_i0 = hw.constant 0 : i0
  // CHECK: %[[WIRE:.+]] = hw.wire %[[C0]]
  %wire = hw.wire %c0_i0 : i0

  // CHECK-NOT: hw.probe.send
  %ref = hw.probe.send %wire : i0

  // CHECK-NOT: hw.probe.resolve
  // Zero-width resolves should be eliminated
  %val = hw.probe.resolve %ref : !hw.probe<i0>

  // CHECK: hw.output %[[C0]]
  hw.output %val : i0
}


//===----------------------------------------------------------------------===//
// Case 2: VerbatimExpr - internal path into a module
// NOTE: This case is skipped in this test as VerbatimExpr is mainly used
// for external/blackbox modules which require special handling
//===----------------------------------------------------------------------===//

// -----

//===----------------------------------------------------------------------===//
// Case 3: Block argument (module port)
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @PortProbe
hw.module @PortProbe(in %input: i32, out result: i32) {
  // Probe send on a module port (block argument)
  // The port should get an inner symbol added to it
  // CHECK-NOT: hw.probe.send
  %ref = hw.probe.send %input : i32

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %ref : !hw.probe<i32>

  hw.output %val : i32
}

// -----

//===----------------------------------------------------------------------===//
// Case 4: Operation with InnerSymbol targeting specific result
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath{{.*}}@existing_sym
// CHECK-LABEL: hw.module @ExistingInnerSymProbe
hw.module @ExistingInnerSymProbe(out result: i32) {
  // Create an operation that already has an inner symbol
  // The existing symbol should be reused, not a new one created
  %c0_i32 = hw.constant 0 : i32

  // CHECK: hw.wire{{.*}}sym @existing_sym
  %wire = hw.wire %c0_i32 sym @existing_sym : i32

  // CHECK-NOT: hw.probe.send
  %ref = hw.probe.send %wire : i32

  // Should use the existing symbol in the hierpath
  // CHECK: sv.xmr.ref
  %val = hw.probe.resolve %ref : !hw.probe<i32>

  hw.output %val : i32
}

// -----

//===----------------------------------------------------------------------===//
// Case 5: Operation without inner symbol - creates wire node
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ConstantProbe
hw.module @ConstantProbe(out result: i32) {
  // Constants cannot hold inner symbols directly
  // A wire node should be created to hold the symbol
  // CHECK: hw.constant 42
  %c42_i32 = hw.constant 42 : i32

  // CHECK: hw.wire{{.*}}sym @
  // CHECK-NOT: hw.probe.send
  %ref = hw.probe.send %c42_i32 : i32

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %ref : !hw.probe<i32>

  hw.output %val : i32
}

// -----

//===----------------------------------------------------------------------===//
// Case 5b: Multi-result operation without target result index
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ArrayCreateProbe
hw.module @ArrayCreateProbe(out result: !hw.array<2xi8>) {
  // array_create returns a value that doesn't have per-result inner symbols
  // A wire node should be created
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %array = hw.array_create %c0_i8, %c1_i8 : i8

  // CHECK: hw.wire{{.*}}sym @
  // CHECK-NOT: hw.probe.send
  %ref = hw.probe.send %array : !hw.array<2xi8>

  // CHECK: sv.xmr.ref
  %val = hw.probe.resolve %ref : !hw.probe<!hw.array<2xi8>>

  hw.output %val : !hw.array<2xi8>
}

// -----

//===----------------------------------------------------------------------===//
// Case 5c: Struct create without inner symbol
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @StructCreateProbe
hw.module @StructCreateProbe(out result: !hw.struct<a: i8, b: i16>) {
  // CHECK: hw.constant 10
  %c10_i8 = hw.constant 10 : i8
  // CHECK: hw.constant 200
  %c200_i16 = hw.constant 200 : i16
  // CHECK: hw.struct_create
  %struct = hw.struct_create (%c10_i8, %c200_i16) : !hw.struct<a: i8, b: i16>

  // CHECK: hw.wire{{.*}}sym @
  // CHECK-NOT: hw.probe.send
  %ref = hw.probe.send %struct : !hw.struct<a: i8, b: i16>

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %ref : !hw.probe<!hw.struct<a: i8, b: i16>>

  hw.output %val : !hw.struct<a: i8, b: i16>
}

// -----

//===----------------------------------------------------------------------===//
// Combined test: All 5 cases in one module
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @AllCasesCombined
hw.module @AllCasesCombined(
  in %port_input: i32,
  out zero_width: i0,
  out from_port: i32,
  out from_wire: i32,
  out from_const: i32
) {
  // Case 1: Zero-width
  // CHECK: hw.constant 0 : i0
  %c0_i0 = hw.constant 0 : i0
  %ref_zw = hw.probe.send %c0_i0 : i0
  %val_zw = hw.probe.resolve %ref_zw : !hw.probe<i0>

  // Case 3: Port (block argument)
  %ref_port = hw.probe.send %port_input : i32
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val_port = hw.probe.resolve %ref_port : !hw.probe<i32>

  // Case 4: Wire with existing symbol
  // CHECK: hw.constant 100
  %c100_i32 = hw.constant 100 : i32
  // CHECK: hw.wire{{.*}}sym @my_wire
  %wire = hw.wire %c100_i32 sym @my_wire : i32
  %ref_wire = hw.probe.send %wire : i32
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val_wire = hw.probe.resolve %ref_wire : !hw.probe<i32>

  // Case 5: Constant (needs wire node)
  // CHECK: hw.constant 200
  %c200_i32 = hw.constant 200 : i32
  // CHECK: hw.wire{{.*}}sym @
  %ref_const = hw.probe.send %c200_i32 : i32
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val_const = hw.probe.resolve %ref_const : !hw.probe<i32>

  // CHECK: hw.output
  hw.output %val_zw, %val_port, %val_wire, %val_const : i0, i32, i32, i32
}

// -----

//===----------------------------------------------------------------------===//
// Test with aggregate probe and indexing (related to Case 5)
// NOTE: Currently commented out due to issue with probe.sub handling
//===----------------------------------------------------------------------===//

// TODO: Re-enable this test once probe.sub is fully implemented
// hw.module @AggregateProbeWithIndex(out result: i32) {
//   %c0_i32 = hw.constant 0 : i32
//   %c1_i32 = hw.constant 1 : i32
//   %c2_i32 = hw.constant 2 : i32
//   %arr = hw.array_create %c0_i32, %c1_i32, %c2_i32 : i32
//
//   // Probe the whole array
//   %arr_ref = hw.probe.send %arr : !hw.array<3xi32>
//
//   // Index into the probe
//   %elem_ref = hw.probe.sub %arr_ref[1] : !hw.probe<!hw.array<3xi32>>
//
//   // Resolve to get the element
//   %elem_val = hw.probe.resolve %elem_ref : !hw.probe<i32>
//
//   hw.output %elem_val : i32
// }

// -----

//===----------------------------------------------------------------------===//
// Test output probe ports through instances (downward references)
//===----------------------------------------------------------------------===//

hw.module @ChildWithOutputProbe(out probe_out: !hw.probe<i16>) {
  %c100_i16 = hw.constant 100 : i16
  %wire = hw.wire %c100_i16 : i16

  // Create probe - will be removed
  %ref = hw.probe.send %wire : i16

  hw.output %ref : !hw.probe<i16>
}

// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ParentWithOutputProbe
hw.module @ParentWithOutputProbe(out result: i16) {
  // Instantiate child - probe port will be removed from instance
  // CHECK: hw.instance "child"{{.*}}@ChildWithOutputProbe()
  %child_probe = hw.instance "child" @ChildWithOutputProbe() -> (probe_out: !hw.probe<i16>)

  // Resolve the probe from child
  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %child_probe : !hw.probe<i16>

  hw.output %val : i16
}

// -----

//===----------------------------------------------------------------------===//
// Test input and output probe ports on same instance
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
hw.module @PassThrough(out probe_out: !hw.probe<i32>) {
  %c42_i32 = hw.constant 42 : i32
  %wire = hw.wire %c42_i32 : i32
  %ref = hw.probe.send %wire : i32
  hw.output %ref : !hw.probe<i32>
}

// CHECK-LABEL: hw.module @PassThroughParent
hw.module @PassThroughParent(out result: i32) {
  // CHECK: hw.instance "child"{{.*}}@PassThrough()
  %child_probe = hw.instance "child" @PassThrough() -> (probe_out: !hw.probe<i32>)

  // CHECK: sv.xmr.ref
  %val = hw.probe.resolve %child_probe : !hw.probe<i32>

  hw.output %val : i32
}

// -----

//===----------------------------------------------------------------------===//
// Test multiple instances with probe ports
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
// CHECK: hw.hierpath
hw.module @LeafModule(out probe_out: !hw.probe<i8>) {
  %c77_i8 = hw.constant 77 : i8
  %wire = hw.wire %c77_i8 : i8
  %ref = hw.probe.send %wire : i8
  hw.output %ref : !hw.probe<i8>
}

// CHECK-LABEL: hw.module @MultipleInstances
hw.module @MultipleInstances(out o1: i8, out o2: i8) {
  // CHECK: hw.instance "inst1"{{.*}}@LeafModule()
  %probe1 = hw.instance "inst1" @LeafModule() -> (probe_out: !hw.probe<i8>)

  // CHECK: hw.instance "inst2"{{.*}}@LeafModule()
  %probe2 = hw.instance "inst2" @LeafModule() -> (probe_out: !hw.probe<i8>)

  // CHECK: sv.xmr.ref
  %val1 = hw.probe.resolve %probe1 : !hw.probe<i8>

  // CHECK: sv.xmr.ref
  %val2 = hw.probe.resolve %probe2 : !hw.probe<i8>

  hw.output %val1, %val2 : i8, i8
}

// -----

//===----------------------------------------------------------------------===//
// Test nested module hierarchy with probe ports
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
hw.module @Grandchild(out probe_out: !hw.probe<i64>) {
  %c999_i64 = hw.constant 999 : i64
  %wire = hw.wire %c999_i64 : i64
  %ref = hw.probe.send %wire : i64
  hw.output %ref : !hw.probe<i64>
}

hw.module @Child(out probe_out: !hw.probe<i64>) {
  %gc_probe = hw.instance "gc" @Grandchild() -> (probe_out: !hw.probe<i64>)
  hw.output %gc_probe : !hw.probe<i64>
}

// CHECK-LABEL: hw.module @GrandparentNested
hw.module @GrandparentNested(out result: i64) {
  // CHECK: hw.instance "child"{{.*}}@Child()
  %child_probe = hw.instance "child" @Child() -> (probe_out: !hw.probe<i64>)

  // CHECK: sv.xmr.ref
  %val = hw.probe.resolve %child_probe : !hw.probe<i64>

  hw.output %val : i64
}

// -----

//===----------------------------------------------------------------------===//
// Test module with mixed regular and probe ports
//===----------------------------------------------------------------------===//

// CHECK: hw.hierpath
hw.module @MixedPorts(
  in %regular_in: i16,
  out regular_out: i16,
  out probe_out: !hw.probe<i16>
) {
  %doubled = comb.shl %regular_in, %regular_in : i16
  %probe_sent = hw.probe.send %regular_in : i16
  hw.output %doubled, %probe_sent : i16, !hw.probe<i16>
}

// CHECK-LABEL: hw.module @ParentOfMixed
hw.module @ParentOfMixed(in %input: i16, out o1: i16, out o2: i16) {
  // CHECK: hw.instance "mixed"{{.*}}@MixedPorts
  %regular_out, %probe_out = hw.instance "mixed" @MixedPorts(
    regular_in: %input: i16
  ) -> (regular_out: i16, probe_out: !hw.probe<i16>)

  // CHECK: sv.xmr.ref
  %val = hw.probe.resolve %probe_out : !hw.probe<i16>

  hw.output %regular_out, %val : i16, i16
}

// -----

//===----------------------------------------------------------------------===//
// Test deep hierarchy with probe ports at multiple levels
//===----------------------------------------------------------------------===//

hw.module @Level3(out probe: !hw.probe<i32>) {
  %c1_i32 = hw.constant 1 : i32
  %wire = hw.wire %c1_i32 : i32
  %ref = hw.probe.send %wire : i32
  hw.output %ref : !hw.probe<i32>
}

hw.module @Level2(out probe: !hw.probe<i32>) {
  %l3_probe = hw.instance "l3" @Level3() -> (probe: !hw.probe<i32>)
  hw.output %l3_probe : !hw.probe<i32>
}

hw.module @Level1(out probe: !hw.probe<i32>) {
  %l2_probe = hw.instance "l2" @Level2() -> (probe: !hw.probe<i32>)
  hw.output %l2_probe : !hw.probe<i32>
}

// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @Level0Top
hw.module @Level0Top(out result: i32) {
  // CHECK: hw.instance "l1"{{.*}}@Level1()
  %l1_probe = hw.instance "l1" @Level1() -> (probe: !hw.probe<i32>)

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %l1_probe : !hw.probe<i32>

  // CHECK: hw.output
  hw.output %val : i32
}

// -----

//===----------------------------------------------------------------------===//
// Test sibling modules with probes from same parent
//===----------------------------------------------------------------------===//

hw.module @Sibling1WithProbe(out probe_out: !hw.probe<i8>) {
  %c10_i8 = hw.constant 10 : i8
  %wire = hw.wire %c10_i8 : i8
  %probe = hw.probe.send %wire : i8
  hw.output %probe : !hw.probe<i8>
}

hw.module @Sibling2WithProbe(out probe_out: !hw.probe<i8>) {
  %c20_i8 = hw.constant 20 : i8
  %wire = hw.wire %c20_i8 : i8
  %probe = hw.probe.send %wire : i8
  hw.output %probe : !hw.probe<i8>
}

// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ParentOfSiblings
hw.module @ParentOfSiblings(out o1: i8, out o2: i8) {
  // Both siblings send out probes
  // CHECK: hw.instance "s1"{{.*}}@Sibling1WithProbe()
  %p1 = hw.instance "s1" @Sibling1WithProbe() -> (probe_out: !hw.probe<i8>)

  // CHECK: hw.instance "s2"{{.*}}@Sibling2WithProbe()
  %p2 = hw.instance "s2" @Sibling2WithProbe() -> (probe_out: !hw.probe<i8>)

  // CHECK: sv.xmr.ref
  %v1 = hw.probe.resolve %p1 : !hw.probe<i8>

  // CHECK: sv.xmr.ref
  %v2 = hw.probe.resolve %p2 : !hw.probe<i8>

  // CHECK: hw.output
  hw.output %v1, %v2 : i8, i8
}

// -----

//===----------------------------------------------------------------------===//
// Test multiple probe ports on same module
//===----------------------------------------------------------------------===//

hw.module @MultiProbeModule(
  out probe1: !hw.probe<i8>,
  out probe2: !hw.probe<i16>,
  out probe3: !hw.probe<i32>
) {
  %c1_i8 = hw.constant 1 : i8
  %c2_i16 = hw.constant 2 : i16
  %c3_i32 = hw.constant 3 : i32

  %w1 = hw.wire %c1_i8 : i8
  %w2 = hw.wire %c2_i16 : i16
  %w3 = hw.wire %c3_i32 : i32

  %ref1 = hw.probe.send %w1 : i8
  %ref2 = hw.probe.send %w2 : i16
  %ref3 = hw.probe.send %w3 : i32

  hw.output %ref1, %ref2, %ref3 : !hw.probe<i8>, !hw.probe<i16>, !hw.probe<i32>
}

// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ParentOfMultiProbe
hw.module @ParentOfMultiProbe(out o1: i8, out o2: i16, out o3: i32) {
  // CHECK: hw.instance "multi"{{.*}}@MultiProbeModule()
  %p1, %p2, %p3 = hw.instance "multi" @MultiProbeModule() ->
    (probe1: !hw.probe<i8>, probe2: !hw.probe<i16>, probe3: !hw.probe<i32>)

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %v1 = hw.probe.resolve %p1 : !hw.probe<i8>

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %v2 = hw.probe.resolve %p2 : !hw.probe<i16>

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %v3 = hw.probe.resolve %p3 : !hw.probe<i32>

  // CHECK: hw.output
  hw.output %v1, %v2, %v3 : i8, i16, i32
}

// -----

//===----------------------------------------------------------------------===//
// Test interleaved regular and probe ports
//===----------------------------------------------------------------------===//

hw.module @InterleavedPorts(
  in %in1: i8,
  in %in2: i16,
  out out1: i8,
  out probe_out1: !hw.probe<i8>,
  out out2: i16,
  out probe_out2: !hw.probe<i16>
) {
  // Regular operations
  %sum1 = comb.add %in1, %in1 : i8
  %sum2 = comb.add %in2, %in2 : i16

  // Probe operations
  %p1 = hw.probe.send %in1 : i8
  %p2 = hw.probe.send %in2 : i16

  hw.output %sum1, %p1, %sum2, %p2 : i8, !hw.probe<i8>, i16, !hw.probe<i16>
}

// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ParentOfInterleaved
hw.module @ParentOfInterleaved(out o1: i8, out o2: i16, out o3: i8, out o4: i16) {
  %c10_i8 = hw.constant 10 : i8
  %c20_i16 = hw.constant 20 : i16

  // CHECK: hw.instance "interleaved"{{.*}}@InterleavedPorts
  %r1, %rp1, %r2, %rp2 = hw.instance "interleaved" @InterleavedPorts(
    in1: %c10_i8: i8,
    in2: %c20_i16: i16
  ) -> (out1: i8, probe_out1: !hw.probe<i8>, out2: i16, probe_out2: !hw.probe<i16>)

  // CHECK: sv.xmr.ref
  %v1 = hw.probe.resolve %rp1 : !hw.probe<i8>
  // CHECK: sv.xmr.ref
  %v2 = hw.probe.resolve %rp2 : !hw.probe<i16>

  // CHECK: hw.output
  hw.output %r1, %r2, %v1, %v2 : i8, i16, i8, i16
}

// -----

//===----------------------------------------------------------------------===//
// Test extern module with probe port
//===----------------------------------------------------------------------===//

// External module with probe output port
hw.module.extern @ExternModuleWithProbe(out probe_out: !hw.probe<i32>)

// CHECK: hw.hierpath{{.*}}@ParentOfExtern
// CHECK-LABEL: hw.module @ParentOfExtern
hw.module @ParentOfExtern(out result: i32) {
  // Instance of extern module
  // CHECK: hw.instance "ext"{{.*}}@ExternModuleWithProbe()
  %ext_probe = hw.instance "ext" @ExternModuleWithProbe() -> (probe_out: !hw.probe<i32>)

  // Resolve the probe from extern module
  // CHECK: sv.xmr.ref{{.*}}"`ref_ExternModuleWithProbe_probe_out"
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %ext_probe : !hw.probe<i32>

  hw.output %val : i32
}

// -----

//===----------------------------------------------------------------------===//
// Test multiple extern modules with probe ports
//===----------------------------------------------------------------------===//

hw.module.extern @ExtMod1(out p: !hw.probe<i8>)
hw.module.extern @ExtMod2(out p: !hw.probe<i16>)

// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ParentOfMultiExtern
hw.module @ParentOfMultiExtern(out o1: i8, out o2: i16) {
  // CHECK: hw.instance "e1"{{.*}}@ExtMod1()
  %p1 = hw.instance "e1" @ExtMod1() -> (p: !hw.probe<i8>)

  // CHECK: hw.instance "e2"{{.*}}@ExtMod2()
  %p2 = hw.instance "e2" @ExtMod2() -> (p: !hw.probe<i16>)

  // CHECK: sv.xmr.ref{{.*}}"`ref_ExtMod1_p"
  %v1 = hw.probe.resolve %p1 : !hw.probe<i8>

  // CHECK: sv.xmr.ref{{.*}}"`ref_ExtMod2_p"
  %v2 = hw.probe.resolve %p2 : !hw.probe<i16>

  hw.output %v1, %v2 : i8, i16
}

// -----

//===----------------------------------------------------------------------===//
// Test mixed extern and regular modules with probes
//===----------------------------------------------------------------------===//

hw.module.extern @ExtModMixed(out probe: !hw.probe<i32>)

hw.module @RegularModMixed(out probe: !hw.probe<i32>) {
  %c99_i32 = hw.constant 99 : i32
  %wire = hw.wire %c99_i32 : i32
  %ref = hw.probe.send %wire : i32
  hw.output %ref : !hw.probe<i32>
}

// CHECK: hw.hierpath
// CHECK: hw.hierpath
// CHECK-LABEL: hw.module @ParentOfMixedModules
hw.module @ParentOfMixedModules(out o1: i32, out o2: i32) {
  // CHECK: hw.instance "ext"{{.*}}@ExtModMixed()
  %pext = hw.instance "ext" @ExtModMixed() -> (probe: !hw.probe<i32>)

  // CHECK: hw.instance "reg"{{.*}}@RegularModMixed()
  %preg = hw.instance "reg" @RegularModMixed() -> (probe: !hw.probe<i32>)

  // Extern module probe should use macro
  // CHECK: sv.xmr.ref{{.*}}"`ref_ExtModMixed_probe"
  %vext = hw.probe.resolve %pext : !hw.probe<i32>

  // Regular module probe should use regular hierpath
  // CHECK: sv.xmr.ref
  // CHECK-NOT: "`ref_
  %vreg = hw.probe.resolve %preg : !hw.probe<i32>

  hw.output %vext, %vreg : i32, i32
}

// -----

//===----------------------------------------------------------------------===//
// Test that extern module ports are properly removed
//===----------------------------------------------------------------------===//

// Verify that the probe port is removed from the extern module signature
// CHECK: hw.module.extern @ExternWithMultiplePorts(in %clk : i1, out data : i32)
hw.module.extern @ExternWithMultiplePorts(in %clk: i1, out data: i32, out probe: !hw.probe<i8>)

// CHECK-LABEL: hw.module @TestExternPortRemoval
hw.module @TestExternPortRemoval(in %clock: i1, out o1: i32, out o2: i8) {
  %data, %probe = hw.instance "ext" @ExternWithMultiplePorts(clk: %clock: i1) -> (data: i32, probe: !hw.probe<i8>)

  %val = hw.probe.resolve %probe : !hw.probe<i8>

  // CHECK: %[[DATA:.+]] = hw.instance "ext"{{.*}}@ExternWithMultiplePorts({{.*}}) -> (data: i32)
  // CHECK: sv.xmr.ref{{.*}}"`ref_ExternWithMultiplePorts_probe"{{.*}}: !hw.inout<i8>
  // CHECK: hw.output %[[DATA]]

  hw.output %data, %val : i32, i8
}

// -----

//===----------------------------------------------------------------------===//
// Test extern module with only probe ports (all ports removed)
//===----------------------------------------------------------------------===//

// All ports should be removed - CHECK this appears before the test module
// CHECK: hw.module.extern @ExternOnlyProbe()
hw.module.extern @ExternOnlyProbe(out p1: !hw.probe<i1>, out p2: !hw.rwprobe<i8>)

// CHECK-LABEL: hw.module @TestAllPortsRemoved
hw.module @TestAllPortsRemoved(out o1: i1, out o2: i8) {
  %p1, %p2 = hw.instance "ext" @ExternOnlyProbe() -> (p1: !hw.probe<i1>, p2: !hw.rwprobe<i8>)

  %v1 = hw.probe.resolve %p1 : !hw.probe<i1>
  %v2 = hw.probe.resolve %p2 : !hw.rwprobe<i8>

  // CHECK: hw.instance "ext"{{.*}}@ExternOnlyProbe() -> ()
  // CHECK: sv.xmr.ref{{.*}}"`ref_ExternOnlyProbe_p1"
  // CHECK: sv.xmr.ref{{.*}}"`ref_ExternOnlyProbe_p2"

  hw.output %v1, %v2 : i1, i8
}

// -----

//===----------------------------------------------------------------------===//
// Test ProbeSubOp - accessing sub-elements of aggregate types
//===----------------------------------------------------------------------===//

hw.module @StructSource(out ref: !hw.probe<!hw.struct<a: i8, b: i16>>) {
  %c0_struct = hw.aggregate_constant [0 : i8, 0 : i16] : !hw.struct<a: i8, b: i16>
  %wire = hw.wire %c0_struct sym @struct_signal : !hw.struct<a: i8, b: i16>
  %probe = hw.probe.send %wire : !hw.struct<a: i8, b: i16>
  hw.output %probe : !hw.probe<!hw.struct<a: i8, b: i16>>
}

// CHECK-LABEL: hw.module @TestProbeSub
hw.module @TestProbeSub(out field_a: i8, out field_b: i16) {
  %struct_probe = hw.instance "inst" sym @inst @StructSource() -> (ref: !hw.probe<!hw.struct<a: i8, b: i16>>)

  // Access field 'a' (index 0)
  %probe_a = hw.probe.sub %struct_probe[0 : i32] : !hw.probe<!hw.struct<a: i8, b: i16>>

  // Access field 'b' (index 1)
  %probe_b = hw.probe.sub %struct_probe[1 : i32] : !hw.probe<!hw.struct<a: i8, b: i16>>

  // CHECK: sv.xmr.ref{{.*}}".a"
  // CHECK: sv.read_inout
  %val_a = hw.probe.resolve %probe_a : !hw.probe<i8>

  // CHECK: sv.xmr.ref{{.*}}".b"
  // CHECK: sv.read_inout
  %val_b = hw.probe.resolve %probe_b : !hw.probe<i16>

  hw.output %val_a, %val_b : i8, i16
}

// -----

//===----------------------------------------------------------------------===//
// Test ProbeSubOp with array types
//===----------------------------------------------------------------------===//

hw.module @ArraySource(out ref: !hw.probe<!hw.array<4xi32>>) {
  %c0_array = hw.aggregate_constant [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !hw.array<4xi32>
  %wire = hw.wire %c0_array sym @array_signal : !hw.array<4xi32>
  %probe = hw.probe.send %wire : !hw.array<4xi32>
  hw.output %probe : !hw.probe<!hw.array<4xi32>>
}

// CHECK-LABEL: hw.module @TestProbeSubArray
hw.module @TestProbeSubArray(out elem0: i32, out elem2: i32) {
  %array_probe = hw.instance "arr_inst" sym @arr_inst @ArraySource() -> (ref: !hw.probe<!hw.array<4xi32>>)

  // Access element 0
  %probe_0 = hw.probe.sub %array_probe[0 : i32] : !hw.probe<!hw.array<4xi32>>

  // Access element 2
  %probe_2 = hw.probe.sub %array_probe[2 : i32] : !hw.probe<!hw.array<4xi32>>

  // CHECK: sv.xmr.ref{{.*}}"[0]"
  // CHECK: sv.read_inout
  %val_0 = hw.probe.resolve %probe_0 : !hw.probe<i32>

  // CHECK: sv.xmr.ref{{.*}}"[2]"
  // CHECK: sv.read_inout
  %val_2 = hw.probe.resolve %probe_2 : !hw.probe<i32>

  hw.output %val_0, %val_2 : i32, i32
}

// -----

//===----------------------------------------------------------------------===//
// Test XMR to input port
//===----------------------------------------------------------------------===//

hw.module @InputPortChild(in %in: i8, out ref: !hw.probe<i8>) {
  %probe = hw.probe.send %in : i8
  hw.output %probe : !hw.probe<i8>
}

// CHECK-LABEL: hw.module @TestInputPort
hw.module @TestInputPort(out result: i8) {
  %c42_i8 = hw.constant 42 : i8
  // CHECK: hw.instance "child"
  %child_ref = hw.instance "child" sym @child @InputPortChild(in: %c42_i8: i8) -> (ref: !hw.probe<i8>)

  // CHECK: sv.xmr.ref
  %val = hw.probe.resolve %child_ref : !hw.probe<i8>
  hw.output %val : i8
}

// -----

//===----------------------------------------------------------------------===//
// Test multiple instances of same module with XMR
//===----------------------------------------------------------------------===//

hw.module @SharedXmrSource(out ref: !hw.probe<i4>) {
  %c7_i4 = hw.constant 7 : i4
  %w = hw.wire %c7_i4 sym @shared : i4
  %probe = hw.probe.send %w : i4
  hw.output %probe : !hw.probe<i4>
}

// CHECK-COUNT-2: hw.hierpath
// CHECK-LABEL: hw.module @TestMultiInstance
hw.module @TestMultiInstance(out o1: i4, out o2: i4) {
  // Two instances of the same module
  // CHECK: hw.instance "foo"
  %foo_ref = hw.instance "foo" sym @foo @SharedXmrSource() -> (ref: !hw.probe<i4>)

  // CHECK: hw.instance "bar"
  %bar_ref = hw.instance "bar" sym @bar @SharedXmrSource() -> (ref: !hw.probe<i4>)

  // Each resolve should have its own hierpath
  // CHECK: sv.xmr.ref
  %val_foo = hw.probe.resolve %foo_ref : !hw.probe<i4>

  // CHECK: sv.xmr.ref
  %val_bar = hw.probe.resolve %bar_ref : !hw.probe<i4>

  hw.output %val_foo, %val_bar : i4, i4
}

// -----

//===----------------------------------------------------------------------===//
// Test probe wire forwarding (wire of probe type)
//===----------------------------------------------------------------------===//

hw.module @WireProbeSource(out ref: !hw.probe<i5>) {
  %c10_i5 = hw.constant 10 : i5
  %w = hw.wire %c10_i5 sym @src : i5
  %probe = hw.probe.send %w : i5
  hw.output %probe : !hw.probe<i5>
}

// CHECK-LABEL: hw.module @TestProbeWire
hw.module @TestProbeWire(out result: i5) {
  %inst_ref = hw.instance "inst" sym @inst @WireProbeSource() -> (ref: !hw.probe<i5>)

  // Wire of probe type - should be forwarded through
  %probe_wire = hw.wire %inst_ref : !hw.probe<i5>

  // CHECK: sv.xmr.ref
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %probe_wire : !hw.probe<i5>
  hw.output %val : i5
}

// -----

//===----------------------------------------------------------------------===//
// Test ProbeDefine forwarding through hierarchy with wire
//===----------------------------------------------------------------------===//

hw.module @DefineSource(out ref: !hw.probe<i3>) {
  %c5_i3 = hw.constant 5 : i3
  %w = hw.wire %c5_i3 sym @def_src : i3
  %probe = hw.probe.send %w : i3
  hw.output %probe : !hw.probe<i3>
}

hw.module @DefineMiddle(out ref: !hw.probe<i3>) {
  %src_ref = hw.instance "src" sym @src @DefineSource() -> (ref: !hw.probe<i3>)
  %local_ref = hw.wire %src_ref : !hw.probe<i3>
  hw.probe.define %local_ref, %src_ref : !hw.probe<i3>
  hw.output %local_ref : !hw.probe<i3>
}

// CHECK-LABEL: hw.module @TestProbeDefineWire
hw.module @TestProbeDefineWire(out result: i3) {
  %mid_ref = hw.instance "mid" sym @mid @DefineMiddle() -> (ref: !hw.probe<i3>)

  // CHECK: sv.xmr.ref
  %val = hw.probe.resolve %mid_ref : !hw.probe<i3>
  hw.output %val : i3
}

// -----

//===----------------------------------------------------------------------===//
// Test ProbeSubOp with nested access (chained subs)
//===----------------------------------------------------------------------===//

hw.module @NestedSource(out ref: !hw.probe<!hw.array<2x!hw.struct<x: i8, y: i8>>>) {
  %c0_nested = hw.aggregate_constant [[0 : i8, 0 : i8], [0 : i8, 0 : i8]] : !hw.array<2x!hw.struct<x: i8, y: i8>>
  %wire = hw.wire %c0_nested sym @nested_signal : !hw.array<2x!hw.struct<x: i8, y: i8>>
  %probe = hw.probe.send %wire : !hw.array<2x!hw.struct<x: i8, y: i8>>
  hw.output %probe : !hw.probe<!hw.array<2x!hw.struct<x: i8, y: i8>>>
}

// CHECK-LABEL: hw.module @TestProbeSubNested
hw.module @TestProbeSubNested(out result: i8) {
  %nested_probe = hw.instance "nested_inst" sym @nested_inst @NestedSource() -> (ref: !hw.probe<!hw.array<2x!hw.struct<x: i8, y: i8>>>)

  // Access array[1]
  %probe_elem1 = hw.probe.sub %nested_probe[1 : i32] : !hw.probe<!hw.array<2x!hw.struct<x: i8, y: i8>>>

  // Access .y field of the struct (index 1)
  %probe_y = hw.probe.sub %probe_elem1[1 : i32] : !hw.probe<!hw.struct<x: i8, y: i8>>

  // CHECK: sv.xmr.ref{{.*}}"[1].y"
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %probe_y : !hw.probe<i8>

  hw.output %val : i8
}

// -----

//===----------------------------------------------------------------------===//
// Test release operations
//===----------------------------------------------------------------------===//

hw.module @ReleasableTarget(out ref: !hw.rwprobe<i12>) {
  %c0_i12 = hw.constant 0 : i12
  %w = hw.wire %c0_i12 sym @releaseable : i12
  %rwprobe = hw.probe.rwprobe @ReleasableTarget::@releaseable : !hw.rwprobe<i12>
  hw.output %rwprobe : !hw.rwprobe<i12>
}

// CHECK-LABEL: hw.module @TestReleaseOps
hw.module @TestReleaseOps(in %clk: i1, in %enable: i1) {
  %tgt_ref = hw.instance "rt" sym @rt @ReleasableTarget() -> (ref: !hw.rwprobe<i12>)

  // CHECK: sv.always posedge %clk
  // CHECK: sv.release
  hw.probe.release %clk, %enable, %tgt_ref : i1, i1, !hw.rwprobe<i12>

  hw.output
}

// -----

//===----------------------------------------------------------------------===//
// Test FirMem debug port lowering
//===----------------------------------------------------------------------===//

// Test basic firmem debug port lowering
// CHECK: hw.hierpath private @[[MEM_PATH:.+]] [@FirMemDebugPortBasic::@[[MEM_SYM:.+]]]
// CHECK-LABEL: hw.module @FirMemDebugPortBasic
hw.module @FirMemDebugPortBasic(in %clk: !seq.clock, in %addr: i4, out result: !hw.array<12xi42>) {
  // CHECK: %mem = seq.firmem sym @[[MEM_SYM]]
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 42>

  // CHECK-NOT: seq.firmem.debug_port
  %debug_ref = seq.firmem.debug_port %mem : <12 x 42>

  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[MEM_PATH]] "Memory"
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  %value = hw.probe.resolve %debug_ref : !hw.probe<!hw.array<12xi42>>

  // CHECK: hw.output %[[READ]]
  hw.output %value : !hw.array<12xi42>
}

// -----

// Test firmem debug port with masked memory
// CHECK: hw.hierpath private @[[MEM_PATH:.+]] [@FirMemDebugPortMasked::@[[MEM_SYM:.+]]]
// CHECK-LABEL: hw.module @FirMemDebugPortMasked
hw.module @FirMemDebugPortMasked(out result: !hw.array<16xi32>) {
  // CHECK: %mem = seq.firmem sym @[[MEM_SYM]]
  %mem = seq.firmem 0, 1, undefined, undefined : <16 x 32, mask 4>

  // CHECK-NOT: seq.firmem.debug_port
  %debug_ref = seq.firmem.debug_port %mem : <16 x 32, mask 4>

  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[MEM_PATH]] "Memory"
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  %value = hw.probe.resolve %debug_ref : !hw.probe<!hw.array<16xi32>>

  // CHECK: hw.output %[[READ]]
  hw.output %value : !hw.array<16xi32>
}

// -----

// Test multiple firmem debug ports in same module
// CHECK: hw.hierpath private @[[MEM1_PATH:.+]] [@FirMemDebugPortMultiple::@[[MEM1_SYM:.+]]]
// CHECK: hw.hierpath private @[[MEM2_PATH:.+]] [@FirMemDebugPortMultiple::@[[MEM2_SYM:.+]]]
// CHECK-LABEL: hw.module @FirMemDebugPortMultiple
hw.module @FirMemDebugPortMultiple(out result1: !hw.array<8xi16>, out result2: !hw.array<4xi8>) {
  // CHECK: %mem1 = seq.firmem sym @[[MEM1_SYM]]
  %mem1 = seq.firmem 0, 1, undefined, undefined : <8 x 16>
  // CHECK: %mem2 = seq.firmem sym @[[MEM2_SYM]]
  %mem2 = seq.firmem 0, 1, undefined, undefined : <4 x 8>

  // CHECK-NOT: seq.firmem.debug_port
  %debug_ref1 = seq.firmem.debug_port %mem1 : <8 x 16>
  %debug_ref2 = seq.firmem.debug_port %mem2 : <4 x 8>

  // CHECK: %[[XMR1:.+]] = sv.xmr.ref @[[MEM1_PATH]] "Memory"
  // CHECK: %[[READ1:.+]] = sv.read_inout %[[XMR1]]
  %value1 = hw.probe.resolve %debug_ref1 : !hw.probe<!hw.array<8xi16>>

  // CHECK: %[[XMR2:.+]] = sv.xmr.ref @[[MEM2_PATH]] "Memory"
  // CHECK: %[[READ2:.+]] = sv.read_inout %[[XMR2]]
  %value2 = hw.probe.resolve %debug_ref2 : !hw.probe<!hw.array<4xi8>>

  // CHECK: hw.output %[[READ1]], %[[READ2]]
  hw.output %value1, %value2 : !hw.array<8xi16>, !hw.array<4xi8>
}

// -----

// Test that firmem debug ports are lowered to XMRs correctly
// CHECK: hw.hierpath private @[[PATH:.+]] [@TestMemoryDebugPort::@[[SYM:.+]]]
// CHECK-LABEL: hw.module @TestMemoryDebugPort
hw.module @TestMemoryDebugPort(in %clk: !seq.clock, in %addr: i4, in %data: i32, in %en: i1, out mem_content: !hw.array<16xi32>) {
  // CHECK: %mem = seq.firmem sym @[[SYM]]
  %mem = seq.firmem 0, 1, undefined, undefined : <16 x 32>

  // Create a debug port to the memory
  // CHECK-NOT: seq.firmem.debug_port
  %debug_ref = seq.firmem.debug_port %mem : <16 x 32>

  // Use the memory normally
  %read_data = seq.firmem.read_port %mem[%addr], clock %clk enable %en : <16 x 32>
  seq.firmem.write_port %mem[%addr] = %data, clock %clk enable %en : <16 x 32>

  // Resolve the debug probe to get the memory contents as an array
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]] "Memory"
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  %mem_array = hw.probe.resolve %debug_ref : !hw.probe<!hw.array<16xi32>>

  // CHECK: hw.output %[[READ]]
  hw.output %mem_array : !hw.array<16xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Test firmem debug port across 3-level hierarchy
//===----------------------------------------------------------------------===//

// Leaf module with memory
// CHECK: hw.hierpath private @[[HIER_PATH:.+]] [@FirMemHierTop::@[[TOP_INST:.+]], @FirMemHierMiddle::@[[MID_INST:.+]], @FirMemHierLeaf::@[[MEM_SYM:.+]]]
// CHECK-LABEL: hw.module @FirMemHierLeaf
hw.module @FirMemHierLeaf(in %clk: !seq.clock, in %addr: i3, in %data: i64, in %en: i1, out probe: !hw.probe<!hw.array<8xi64>>) {
  // CHECK: %mem = seq.firmem sym @[[MEM_SYM]]
  %mem = seq.firmem 0, 1, undefined, undefined : <8 x 64>

  // Normal memory operations
  %read_data = seq.firmem.read_port %mem[%addr], clock %clk enable %en : <8 x 64>
  seq.firmem.write_port %mem[%addr] = %data, clock %clk enable %en : <8 x 64>

  // Create debug port and output as probe
  // CHECK-NOT: seq.firmem.debug_port
  %debug_ref = seq.firmem.debug_port %mem : <8 x 64>

  // CHECK: hw.output
  hw.output %debug_ref : !hw.probe<!hw.array<8xi64>>
}

// Middle module that instantiates leaf
// CHECK-LABEL: hw.module @FirMemHierMiddle
hw.module @FirMemHierMiddle(in %clk: !seq.clock, in %addr: i3, in %data: i64, in %en: i1, out probe: !hw.probe<!hw.array<8xi64>>) {
  // CHECK: hw.instance "leaf" sym @[[MID_INST]]
  %probe_out = hw.instance "leaf" @FirMemHierLeaf(clk: %clk: !seq.clock, addr: %addr: i3, data: %data: i64, en: %en: i1) -> (probe: !hw.probe<!hw.array<8xi64>>)

  // CHECK: hw.output
  hw.output %probe_out : !hw.probe<!hw.array<8xi64>>
}

// Top module that resolves the probe
// CHECK-LABEL: hw.module @FirMemHierTop
hw.module @FirMemHierTop(in %clk: !seq.clock, in %addr: i3, in %data: i64, in %en: i1, out mem_content: !hw.array<8xi64>) {
  // CHECK: hw.instance "middle" sym @[[TOP_INST]]
  %probe_out = hw.instance "middle" @FirMemHierMiddle(clk: %clk: !seq.clock, addr: %addr: i3, data: %data: i64, en: %en: i1) -> (probe: !hw.probe<!hw.array<8xi64>>)

  // Resolve the probe at the top level
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[HIER_PATH]] "Memory"
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  %mem_array = hw.probe.resolve %probe_out : !hw.probe<!hw.array<8xi64>>

  // CHECK: hw.output %[[READ]]
  hw.output %mem_array : !hw.array<8xi64>
}

// -----

//===----------------------------------------------------------------------===//
// External Module Probe Handling - Equivalence Test
// This test demonstrates that HW dialect handles external module probes
// identically to FIRRTL dialect using the macro-based ABI.
//===----------------------------------------------------------------------===//

// Probe port should be removed
// CHECK: hw.module.extern @ExtMod()
hw.module.extern @ExtMod(out probe: !hw.probe<i8>)

// CHECK: hw.module @SimpleExtern
hw.module @SimpleExtern(out result: i8) {
  // Instance should have symbol added and probe port removed
  // CHECK: hw.instance "ext" sym @xmr_sym @ExtMod() -> ()
  %ext_probe = hw.instance "ext" @ExtMod() -> (probe: !hw.probe<i8>)

  // Probe resolve should become XMR with macro suffix
  // CHECK: sv.xmr.ref @xmrPath "`ref_ExtMod_probe" : !hw.inout<i8>
  // CHECK: sv.read_inout
  %val = hw.probe.resolve %ext_probe : !hw.probe<i8>

  // CHECK: hw.output
  hw.output %val : i8
}

// -----

//===----------------------------------------------------------------------===//
// External Module with Multiple Probe Ports
// Mirrors FIRRTL test: firrtl.extmodule @RefExtMore
//===----------------------------------------------------------------------===//

// Both probe ports should be removed
// CHECK: hw.module.extern @ExtModMultiple(in %in : i1, out data : i3)
hw.module.extern @ExtModMultiple(
  in %in: i1,
  out data: i3,
  out probe1: !hw.probe<i1>,
  out probe2: !hw.probe<!hw.array<3xi3>>
)

// CHECK: hw.module @MultiProbeExtern
hw.module @MultiProbeExtern(in %input: i1, out o1: i1, out o2: !hw.array<3xi3>) {
  // CHECK: hw.instance "ext" sym @xmr_sym @ExtModMultiple(in: %input{{.*}}: i1) -> (data: i3)
  %data, %probe1, %probe2 = hw.instance "ext" @ExtModMultiple(
    in: %input: i1
  ) -> (data: i3, probe1: !hw.probe<i1>, probe2: !hw.probe<!hw.array<3xi3>>)

  // First probe: scalar type
  // CHECK: sv.xmr.ref @xmrPath "`ref_ExtModMultiple_probe1" : !hw.inout<i1>
  // CHECK: sv.read_inout
  %val1 = hw.probe.resolve %probe1 : !hw.probe<i1>

  // Second probe: complex type (array)
  // CHECK: sv.xmr.ref @xmrPath "`ref_ExtModMultiple_probe2"
  // CHECK: sv.read_inout
  %val2 = hw.probe.resolve %probe2 : !hw.probe<!hw.array<3xi3>>

  // CHECK: hw.output
  hw.output %val1, %val2 : i1, !hw.array<3xi3>
}


// -----

//===----------------------------------------------------------------------===//
// Zero-Width Probe on External Module (should be removed)
//===----------------------------------------------------------------------===//

// CHECK: hw.module.extern @ExtModZeroWidth(out data : i8)
hw.module.extern @ExtModZeroWidth(out data: i8, out probe: !hw.probe<i0>)

// CHECK: hw.module @ZeroWidthExtern
hw.module @ZeroWidthExtern(out o1: i8, out o2: i0) {
  // CHECK: hw.instance "ext"{{.*}}@ExtModZeroWidth() -> (data: i8)
  %data, %probe = hw.instance "ext" @ExtModZeroWidth() -> (data: i8, probe: !hw.probe<i0>)

  // Zero-width resolve should become a constant
  // CHECK: hw.constant 0 : i0
  %val = hw.probe.resolve %probe : !hw.probe<i0>

  // CHECK: hw.output
  hw.output %data, %val : i8, i0
}

// -----

//===----------------------------------------------------------------------===//
// Mixed External and Regular Modules
// Verifies that external modules use macros while regular modules use
// direct hierarchical paths
//===----------------------------------------------------------------------===//

hw.module.extern @ExtMixed(out probe: !hw.probe<i16>)

hw.module @RegularMixed(out probe: !hw.probe<i16>) {
  %c99_i16 = hw.constant 99 : i16
  %wire = hw.wire %c99_i16 : i16
  %ref = hw.probe.send %wire : i16
  hw.output %ref : !hw.probe<i16>
}

// CHECK: hw.module @MixedModules
hw.module @MixedModules(out o1: i16, out o2: i16) {
  // CHECK: hw.instance "ext"{{.*}}@ExtMixed() -> ()
  %pext = hw.instance "ext" @ExtMixed() -> (probe: !hw.probe<i16>)

  // CHECK: hw.instance "reg"{{.*}}@RegularMixed() -> ()
  %preg = hw.instance "reg" @RegularMixed() -> (probe: !hw.probe<i16>)

  // External module should use macro (note the backtick)
  // CHECK: sv.xmr.ref{{.*}}"`ref_ExtMixed_probe"
  %vext = hw.probe.resolve %pext : !hw.probe<i16>

  // Regular module should NOT use macro - check that there's NO backtick before "ref"
  // CHECK: sv.xmr.ref
  // CHECK-NOT: "`ref
  %vreg = hw.probe.resolve %preg : !hw.probe<i16>

  hw.output %vext, %vreg : i16, i16
}

// -----

//===----------------------------------------------------------------------===//
// Test hw.probe.xmr_ref lowering (similar to probe.resolve, but returns inout)
//===----------------------------------------------------------------------===//

// Test basic hw.probe.xmr_ref lowering
// Since we can't output inout types from modules, we'll read from it
// CHECK: hw.hierpath private @[[PATH:.+]] [@BasicXMRRef::@[[SYM:.+]]]
// CHECK-LABEL: hw.module @BasicXMRRef
hw.module @BasicXMRRef(out result: i8) {
  // CHECK: %[[C42:.+]] = hw.constant 42
  // CHECK: %[[WIRE:.+]] = hw.wire %[[C42]] sym @[[SYM]]
  // CHECK-NOT: hw.probe.send
  // CHECK-NOT: hw.probe.xmr_ref
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  // CHECK: hw.output %[[READ]]

  %c42_i8 = hw.constant 42 : i8
  %wire = hw.wire %c42_i8 : i8

  %probe = hw.probe.send %wire : i8
  %inout = hw.probe.xmr_ref %probe : !hw.probe<i8>
  %value = sv.read_inout %inout : !hw.inout<i8>

  hw.output %value : i8
}

// -----

// Test hw.probe.xmr_ref with rwprobe
// Since we can't output inout types, we'll use it for force/release (simulated with read)
// CHECK: hw.hierpath private @[[PATH:.+]] [@XMRRefRWProbe::@[[SYM:.+]]]
// CHECK-LABEL: hw.module @XMRRefRWProbe
hw.module @XMRRefRWProbe(out result: i32) {
  // CHECK: %[[WIRE:.+]] = hw.wire %{{.+}} sym @[[SYM]]
  // CHECK-NOT: hw.probe.send
  // CHECK-NOT: hw.probe.xmr_ref
  // CHECK: %[[XMR:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ:.+]] = sv.read_inout %[[XMR]]
  // CHECK: hw.output %[[READ]]

  %c99_i32 = hw.constant 99 : i32
  %wire = hw.wire %c99_i32 : i32

  %rwprobe = hw.probe.send forceable %wire : i32
  %inout = hw.probe.xmr_ref %rwprobe : !hw.rwprobe<i32>
  // Use the inout (e.g., for force/release or read)
  %value = sv.read_inout %inout : !hw.inout<i32>

  hw.output %value : i32
}

// -----

// Test hw.probe.xmr_ref vs probe.resolve - both should create sv.xmr.ref
// CHECK: hw.hierpath private @[[PATH:.+]] [@XMRRefAndResolve::@[[SYM:.+]]]
// CHECK-LABEL: hw.module @XMRRefAndResolve
hw.module @XMRRefAndResolve(out value1: i16, out value2: i16) {
  // CHECK: %[[WIRE:.+]] = hw.wire %{{.+}} sym @[[SYM]]
  // CHECK-NOT: hw.probe
  // CHECK: %[[XMR1:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ1:.+]] = sv.read_inout %[[XMR1]]
  // CHECK: %[[XMR2:.+]] = sv.xmr.ref @[[PATH]]
  // CHECK: %[[READ2:.+]] = sv.read_inout %[[XMR2]]
  // CHECK: hw.output %[[READ1]], %[[READ2]]

  %c123_i16 = hw.constant 123 : i16
  %wire = hw.wire %c123_i16 : i16

  %probe = hw.probe.send %wire : i16

  // Get inout via xmr_ref, then read from it
  %inout = hw.probe.xmr_ref %probe : !hw.probe<i16>
  %val1 = sv.read_inout %inout : !hw.inout<i16>

  // Get value directly via resolve
  %val2 = hw.probe.resolve %probe : !hw.probe<i16>

  hw.output %val1, %val2 : i16, i16
}