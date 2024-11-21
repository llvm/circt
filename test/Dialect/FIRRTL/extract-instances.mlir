// RUN: circt-opt --firrtl-extract-instances %s | FileCheck %s

// Tests extracted from:
// - test/scala/firrtl/ExtractBlackBoxes.scala
// - test/scala/firrtl/ExtractClockGates.scala
// - test/scala/firrtl/ExtractSeqMems.scala

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes Simple
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractBlackBoxesSimple"
firrtl.circuit "ExtractBlackBoxesSimple" attributes {annotations = [{class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "BlackBoxes"}]} {
  // CHECK-LABEL: firrtl.extmodule private @MyBlackBox
  firrtl.extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "bb"}], defname = "MyBlackBox"}
  // CHECK-LABEL: firrtl.module private @BBWrapper
  // CHECK-SAME: out %bb_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %bb_0_out: !firrtl.uint<8>
  firrtl.module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK-NOT: firrtl.instance bb @MyBlackBox
    %bb_in, %bb_out = firrtl.instance bb @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    firrtl.matchingconnect %bb_in, %invalid_ui8 : !firrtl.uint<8>
    // CHECK: firrtl.connect %out, %bb_0_out
    // CHECK: firrtl.connect %bb_0_in, %in
    firrtl.connect %out, %bb_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %bb_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module private @DUTModule
  // CHECK-SAME: out %bb_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %bb_0_out: !firrtl.uint<8>
  firrtl.module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NOT: firrtl.instance bb @MyBlackBox
    // CHECK: %mod_in, %mod_out, %mod_bb_0_in, %mod_bb_0_out = firrtl.instance mod
    // CHECK-SAME: sym [[WRAPPER_SYM:@.+]] {annotations =
    // CHECK-SAME: circt.nonlocal
    // CHECK-SAME: id = distinct[0]<>
    // CHECK-SAME: @BBWrapper
    // CHECK-NEXT: firrtl.matchingconnect %bb_0_in, %mod_bb_0_in
    // CHECK-NEXT: firrtl.matchingconnect %mod_bb_0_out, %bb_0_out
    %mod_in, %mod_out = firrtl.instance mod @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ExtractBlackBoxesSimple
  firrtl.module @ExtractBlackBoxesSimple(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, out %metadataObj: !firrtl.anyref) {
    // CHECK: %dut_in, %dut_out, %dut_bb_0_in, %dut_bb_0_out = firrtl.instance dut sym {{@.+}} @DUTModule
    // CHECK-NEXT: %bb_in, %bb_out = firrtl.instance bb @MyBlackBox
    // CHECK-NEXT: firrtl.matchingconnect %bb_in, %dut_bb_0_in
    // CHECK-NEXT: firrtl.matchingconnect %dut_bb_0_out, %bb_out
    %dut_in, %dut_out = firrtl.instance dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
    %sifive_metadata = firrtl.object @SiFive_Metadata()
    // CHECK:  firrtl.object @SiFive_Metadata(out [[extractedInstances_field_0:.+]]: !firrtl.class<@ExtractInstancesMetadata
    %0 = firrtl.object.anyref_cast %sifive_metadata : !firrtl.class<@SiFive_Metadata()>
    firrtl.propassign %metadataObj, %0 : !firrtl.anyref
  }
  firrtl.class @SiFive_Metadata() {}
  // CHECK:  firrtl.class @SiFive_Metadata(
  // CHECK-SAME: out %[[extractedInstances_field_0]]: !firrtl.class<@ExtractInstancesMetadata
  // CHECK-SAME: {
  // CHECK:    %extract_instances_metadata = firrtl.object @ExtractInstancesMetadata(out [[bb_0_field:.+]]: !firrtl.class<@ExtractInstancesSchema
  // CHECK:    firrtl.propassign %[[extractedInstances_field_0]], %extract_instances_metadata : !firrtl.class<@ExtractInstancesMetadata
  // CHECK:  }

  // CHECK:  firrtl.class @ExtractInstancesSchema(in %name_in: !firrtl.string, out %name: !firrtl.string, in %path_in: !firrtl.path, out %path: !firrtl.path, in %filename_in: !firrtl.string, out %filename: !firrtl.string, in %inst_name_in: !firrtl.string, out %inst_name: !firrtl.string) {
  // CHECK:    firrtl.propassign %name, %name_in : !firrtl.string
  // CHECK:    firrtl.propassign %path, %path_in : !firrtl.path
  // CHECK:    firrtl.propassign %filename, %filename_in : !firrtl.string
  // CHECK:    firrtl.propassign %inst_name, %inst_name_in : !firrtl.string
  // CHECK:  }

  // CHECK:  firrtl.class @ExtractInstancesMetadata(out %[[bb_0_field]]: !firrtl.class<@ExtractInstancesSchema
  // CHECK-SAME: {
  // CHECK:    %[[V0:.+]] = firrtl.string "bb_0"
  // CHECK:    %[[bb_0:.+]] = firrtl.object @ExtractInstancesSchema
  // CHECK:    %[[V1:.+]] = firrtl.object.subfield %[[bb_0]][name_in]
  // CHECK:    firrtl.propassign %[[V1]], %[[V0]] : !firrtl.string
  // CHECK:    %[[V2:.+]] = firrtl.path instance distinct[0]<>
  // CHECK:    %[[V3:.+]] = firrtl.object.subfield %[[bb_0]][path_in]
  // CHECK:    firrtl.propassign %[[V3]], %[[V2]] : !firrtl.path
  // CHECK:    %[[V4:.+]] = firrtl.object.subfield %[[bb_0]][filename_in]
  // CHECK:    %[[V5:.+]] = firrtl.string "BlackBoxes.txt"
  // CHECK:    firrtl.propassign %[[V4]], %[[V5]] : !firrtl.string
  // CHECK:    %[[V6:.+]] = firrtl.object.subfield %bb_0[inst_name_in]
  // CHECK:    %[[V7:.+]] = firrtl.string "bb"
  // CHECK;    firrtl.propassign %[[V6]], %[[V7]] : !firrtl.string
  // CHECK:    firrtl.propassign %[[bb_0_field]], %[[bb_0]]
  // CHECK:  }

  // CHECK:               emit.file "BlackBoxes.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     bb_0 -> {{0}}.{{1}}.bb\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:                #hw.innerNameRef<@DUTModule::[[WRAPPER_SYM]]>
  // CHECK-SAME:              ]
}

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes Simple (modified)
// ExtractBlackBoxes RenameTargets (modified)
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractBlackBoxesSimple2"
firrtl.circuit "ExtractBlackBoxesSimple2" attributes {annotations = [{class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "BlackBoxes"}]} {
  // Old style NLAs
  hw.hierpath private @nla_old1 [@DUTModule::@mod, @BBWrapper::@bb]
  hw.hierpath private @nla_old2 [@ExtractBlackBoxesSimple2::@dut, @DUTModule::@mod, @BBWrapper::@bb]
  // New style NLAs on extracted instance
  hw.hierpath private @nla_on1 [@DUTModule::@mod, @BBWrapper]
  hw.hierpath private @nla_on2 [@ExtractBlackBoxesSimple2::@dut, @DUTModule::@mod, @BBWrapper]
  // New style NLAs through extracted instance
  hw.hierpath private @nla_thru1 [@BBWrapper::@bb, @MyBlackBox]
  hw.hierpath private @nla_thru2 [@DUTModule::@mod, @BBWrapper::@bb, @MyBlackBox]
  hw.hierpath private @nla_thru3 [@ExtractBlackBoxesSimple2::@dut, @DUTModule::@mod, @BBWrapper::@bb, @MyBlackBox]
  // CHECK: hw.hierpath private [[THRU1:@nla_thru1]] [@ExtractBlackBoxesSimple2::@bb, @MyBlackBox]
  // CHECK: hw.hierpath private [[THRU2:@nla_thru2]] [@ExtractBlackBoxesSimple2::@bb, @MyBlackBox]
  // CHECK: hw.hierpath private [[THRU3:@nla_thru3]] [@ExtractBlackBoxesSimple2::@bb, @MyBlackBox]

  // Annotation on the extmodule itself
  // CHECK-LABEL: firrtl.extmodule private @MyBlackBox
  firrtl.extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [
      {class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "prefix"},
      {circt.nonlocal = @nla_thru1, class = "Thru1"},
      {circt.nonlocal = @nla_thru2, class = "Thru2"},
      {circt.nonlocal = @nla_thru3, class = "Thru3"}
    ], defname = "MyBlackBox"}
  // Annotation will be on the instance
  // CHECK-LABEL: firrtl.extmodule private @MyBlackBox2
  firrtl.extmodule private @MyBlackBox2(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {defname = "MyBlackBox"}

  // CHECK-LABEL: firrtl.module private @BBWrapper
  // CHECK-SAME: out %prefix_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_0_out: !firrtl.uint<8>
  // CHECK-SAME: out %prefix_1_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_1_out: !firrtl.uint<8>
  firrtl.module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK-NOT: firrtl.instance bb @MyBlackBox
    // CHECK-NOT: firrtl.instance bb2 @MyBlackBox2
    %bb_in, %bb_out = firrtl.instance bb sym @bb {annotations = [
        {circt.nonlocal = @nla_old1, class = "Old1"},
        {circt.nonlocal = @nla_old2, class = "Old2"},
        {circt.nonlocal = @nla_on1, class = "On1"},
        {circt.nonlocal = @nla_on2, class = "On2"}
      ]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %bb2_in, %bb2_out = firrtl.instance bb2 {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "prefix"}]} @MyBlackBox2(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    // CHECK: firrtl.connect %out, %prefix_0_out
    // CHECK: firrtl.connect %prefix_0_in, %prefix_1_out
    // CHECK: firrtl.connect %prefix_1_in, %in
    firrtl.connect %out, %bb2_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %bb2_in, %bb_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %bb_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module private @DUTModule
  // CHECK-SAME: out %prefix_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_0_out: !firrtl.uint<8>
  // CHECK-SAME: out %prefix_1_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_1_out: !firrtl.uint<8>
  firrtl.module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NOT: firrtl.instance bb @MyBlackBox
    // CHECK-NOT: firrtl.instance bb2 @MyBlackBox2
    // CHECK: %mod_in, %mod_out, %mod_prefix_0_in, %mod_prefix_0_out, %mod_prefix_1_in, %mod_prefix_1_out = firrtl.instance mod
    // CHECK-SAME: sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NOT: annotations =
    // CHECK-NEXT: firrtl.matchingconnect %prefix_1_in, %mod_prefix_1_in
    // CHECK-NEXT: firrtl.matchingconnect %mod_prefix_1_out, %prefix_1_out
    // CHECK-NEXT: firrtl.matchingconnect %prefix_0_in, %mod_prefix_0_in
    // CHECK-NEXT: firrtl.matchingconnect %mod_prefix_0_out, %prefix_0_out
    %mod_in, %mod_out = firrtl.instance mod sym @mod @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ExtractBlackBoxesSimple2
  firrtl.module @ExtractBlackBoxesSimple2(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK: %dut_in, %dut_out, %dut_prefix_0_in, %dut_prefix_0_out, %dut_prefix_1_in, %dut_prefix_1_out = firrtl.instance dut
    // CHECK-NOT: annotations =
    // CHECK-SAME: sym {{@.+}} @DUTModule
    // CHECK-NEXT: %bb_in, %bb_out = firrtl.instance bb sym [[BB_SYM:@.+]] {annotations = [{class = "Old1"}, {class = "On1"}, {class = "Old2"}, {class = "On2"}]} @MyBlackBox
    // CHECK-NEXT: firrtl.matchingconnect %bb_in, %dut_prefix_1_in
    // CHECK-NEXT: firrtl.matchingconnect %dut_prefix_1_out, %bb_out
    // CHECK-NEXT: %bb2_in, %bb2_out = firrtl.instance bb2 sym @sym @MyBlackBox2
    // CHECK-NEXT: firrtl.matchingconnect %bb2_in, %dut_prefix_0_in
    // CHECK-NEXT: firrtl.matchingconnect %dut_prefix_0_out, %bb2_out
    %dut_in, %dut_out = firrtl.instance dut sym @dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK:               emit.file "BlackBoxes.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     prefix_0 -> {{0}}.{{1}}.bb2\0A
  // CHECK-SAME{LITERAL}:     prefix_1 -> {{0}}.{{1}}.bb\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:                @DUTModule::[[WRAPPER_SYM]]
  // CHECK-SAME:              ]
}

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes IntoDUTSubmodule
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractBlackBoxesIntoDUTSubmodule"
firrtl.circuit "ExtractBlackBoxesIntoDUTSubmodule"  {
  // CHECK-LABEL: hw.hierpath private @nla_new_0 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes
  // CHECK-SAME:  ]
  // CHECK-LABEL: hw.hierpath private @nla_new_1 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes
  // CHECK-SAME:  ]
  hw.hierpath private @nla_new [
    @ExtractBlackBoxesIntoDUTSubmodule::@tb,
    @TestHarness::@dut,
    @DUTModule::@mod,
    @BBWrapper
  ]
  // CHECK-LABEL: hw.hierpath private @nla_old1 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes::@bb1
  // CHECK-SAME:  ]
  hw.hierpath private @nla_old1 [
    @ExtractBlackBoxesIntoDUTSubmodule::@tb,
    @TestHarness::@dut,
    @DUTModule::@mod,
    @BBWrapper::@bb1
  ]
  // CHECK-LABEL: hw.hierpath private @nla_old2 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes::@bb2
  // CHECK-SAME:  ]
  hw.hierpath private @nla_old2 [
    @ExtractBlackBoxesIntoDUTSubmodule::@tb,
    @TestHarness::@dut,
    @DUTModule::@mod,
    @BBWrapper::@bb2
  ]
  firrtl.extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", dest = "BlackBoxes", filename = "BlackBoxes.txt", prefix = "bb"}], defname = "MyBlackBox"}
  firrtl.module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %bb1_in, %bb1_out = firrtl.instance bb1 sym @bb1 {annotations = [{circt.nonlocal = @nla_old1, class = "Dummy1"}, {circt.nonlocal = @nla_new, class = "Dummy3"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %bb2_in, %bb2_out = firrtl.instance bb2 sym @bb2 {annotations = [{circt.nonlocal = @nla_old2, class = "Dummy2"}, {circt.nonlocal = @nla_new, class = "Dummy4"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %bb2_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %bb2_in, %bb1_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %bb1_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module private @BlackBoxes(
  // CHECK-SAME:    in %bb_0_in: !firrtl.uint<8>
  // CHECK-SAME:    out %bb_0_out: !firrtl.uint<8>
  // CHECK-SAME:    in %bb_1_in: !firrtl.uint<8>
  // CHECK-SAME:    out %bb_1_out: !firrtl.uint<8>
  // CHECK-SAME:  ) {
  // CHECK-NEXT:    %bb2_in, %bb2_out = firrtl.instance bb2 sym [[BB2_SYM:@.+]] {annotations = [{circt.nonlocal = @nla_new_0, class = "Dummy4"}, {circt.nonlocal = @nla_old2, class = "Dummy2"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
  // CHECK-NEXT:    firrtl.matchingconnect %bb2_in, %bb_0_in : !firrtl.uint<8>
  // CHECK-NEXT:    firrtl.matchingconnect %bb_0_out, %bb2_out : !firrtl.uint<8>
  // CHECK-NEXT:    %bb1_in, %bb1_out = firrtl.instance bb1 sym [[BB1_SYM:@.+]] {annotations = [{circt.nonlocal = @nla_new_1, class = "Dummy3"}, {circt.nonlocal = @nla_old1, class = "Dummy1"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
  // CHECK-NEXT:    firrtl.matchingconnect %bb1_in, %bb_1_in : !firrtl.uint<8>
  // CHECK-NEXT:    firrtl.matchingconnect %bb_1_out, %bb1_out : !firrtl.uint<8>
  // CHECK-NEXT:  }
  // CHECK-LABEL: firrtl.module private @DUTModule
  firrtl.module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: %BlackBoxes_bb_0_in, %BlackBoxes_bb_0_out, %BlackBoxes_bb_1_in, %BlackBoxes_bb_1_out = firrtl.instance BlackBoxes sym @BlackBoxes
    // CHECK-SAME: @BlackBoxes
    // CHECK-NEXT: %mod_in, %mod_out, %mod_bb_0_in, %mod_bb_0_out, %mod_bb_1_in, %mod_bb_1_out = firrtl.instance mod
    // CHECK-NOT: annotations =
    // CHECK-SAME: sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NEXT: firrtl.matchingconnect %BlackBoxes_bb_1_in, %mod_bb_1_in
    // CHECK-NEXT: firrtl.matchingconnect %mod_bb_1_out, %BlackBoxes_bb_1_out
    // CHECK-NEXT: firrtl.matchingconnect %BlackBoxes_bb_0_in, %mod_bb_0_in
    // CHECK-NEXT: firrtl.matchingconnect %mod_bb_0_out, %BlackBoxes_bb_0_out
    %mod_in, %mod_out = firrtl.instance mod sym @mod @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  firrtl.module @TestHarness(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %dut_in, %dut_out = firrtl.instance dut sym @dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  firrtl.module @ExtractBlackBoxesIntoDUTSubmodule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %tb_in, %tb_out = firrtl.instance tb sym @tb @TestHarness(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %tb_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %tb_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK:               emit.file "BlackBoxes.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     bb_0 -> {{0}}.{{1}}.bb2\0A
  // CHECK-SAME{LITERAL}:     bb_1 -> {{0}}.{{1}}.bb1\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:                @DUTModule::[[WRAPPER_SYM]]
  // CHECK-SAME:              ]
}

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes Custom Tests
//
// These are tests that were not derived from legacy custom SFC tests.
//===----------------------------------------------------------------------===//

// Test all possible combinations of extraction for modules that are: (1)
// exclusively under the design, (2) exclusively not under the design, and (3)
// both under and not under the design.  Modules can be not under the design if
// they are outside the design or if they are under a layer.
//
// Note: this pass does not do any deduplication, so a module that is of type
// (3) needs to cause extraction outside the design up to the point that it is
// no longer necessary to do more extraction.
firrtl.circuit "CombinationsTest" {
  firrtl.layer @A bind {}
  firrtl.extmodule @bbox_Foo() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
        filename = "BlackBoxes.txt",
        prefix = ""
      }
    ],
    defname = "bbox_Foo"
  }
  firrtl.module @Foo() {
    firrtl.instance bbox_Foo @bbox_Foo()
  }
  firrtl.extmodule @bbox_Bar() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
        filename = "BlackBoxes.txt",
        prefix = ""
      }
    ],
    defname = "bbox_Bar"
  }
  firrtl.module @Bar() {
    firrtl.instance bbox_Bar @bbox_Bar()
  }
  firrtl.extmodule @bbox_Baz() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
        filename = "BlackBoxes.txt",
        prefix = ""
      }
    ],
    defname = "bbox_Baz"
  }
  firrtl.module @Baz() {
    firrtl.instance bbox_Baz @bbox_Baz()
  }
  firrtl.extmodule @bbox_Qux() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
        filename = "BlackBoxes.txt",
        prefix = ""
      }
    ],
    defname = "bbox_Qux"
  }
  firrtl.module @Qux() {
    firrtl.instance bbox_Qux @bbox_Qux()
  }
  firrtl.extmodule @bbox_Quz() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
        filename = "BlackBoxes.txt",
        prefix = ""
      }
    ],
    defname = "bbox_Quz"
  }
  firrtl.module @Quz() {
    firrtl.instance bbox_Quz @bbox_Quz()
  }
  firrtl.module @DUT() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance foo @Foo()
    firrtl.instance baz @Baz()
    firrtl.instance quz1 @Quz()
    firrtl.layerblock @A {
      firrtl.instance qux @Qux()
      firrtl.instance quz2 @Quz()
    }
  }
  firrtl.module @Wrapper() {
    firrtl.instance dut @DUT()
    firrtl.instance bar @Bar()
    firrtl.instance baz @Baz()
  }
  firrtl.module @CombinationsTest() {
    firrtl.instance wrapper @Wrapper()
    // %sifive_metadata = firrtl.object @SiFive_Metadata()
  }
  // firrtl.class @SiFive_Metadata() {}
}

// CHECK-LABEL: firrtl.circuit "CombinationsTest"

// CHECK-LABEL: firrtl.module @Foo()
// CHECK-NOT:     firrtl.instance bbox_Foo @bbox_Foo

// CHECK-LABEL: firrtl.module @Bar()
// CHECK-NEXT:    firrtl.instance bbox_Bar @bbox_Bar

// CHECK-LABEL: firrtl.module @Baz()
// CHECK-NOT:     firrtl.instance bbox_Baz @bbox_Baz

// CHECK-LABEL: firrtl.module @Qux()
// CHECK:         firrtl.instance bbox_Qux @bbox_Qux

// CHECK-LABEL: firrtl.module @Quz()
// CHECK-NOT:     firrtl.instance bbox_Quz @bbox_Quz

// CHECK-LABEL: firrtl.module @DUT()
//
// CHECK:         firrtl.layerblock @A {
// CHECK-NEXT:      firrtl.instance qux @Qux()
// CHECK-NEXT:      firrtl.instance quz2 {{.*}}@Quz()
// CHECK-NEXT:      firrtl.instance bbox_Quz {{.*}}@bbox_Quz()

// CHECK-LABEL: firrtl.module @Wrapper()
// CHECK-NEXT:    firrtl.instance dut {{.*}}@DUT()
// CHECK-NEXT:    firrtl.instance bbox_Foo {{.*}}@bbox_Foo()
// CHECK-NEXT:    firrtl.instance bbox_Baz {{.*}}@bbox_Baz()
// CHECK-NEXT:    firrtl.instance bbox_Quz {{.*}}@bbox_Quz()
//
// CHECK-NEXT:    firrtl.instance bar @Bar()
//
// CHECK-NEXT:    firrtl.instance baz {{.*}}@Baz()
// CHECK-NEXT:    firrtl.instance bbox_Baz {{.*}}@bbox_Baz()

// Test that a circuit without a design-under-test has no extraction.
firrtl.circuit "NoDutBehavior" {

  firrtl.extmodule @bbox_Foo() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
        filename = "BlackBoxes.txt",
        prefix = ""
      }
    ],
    defname = "bbox_Foo"
  }
  firrtl.module @Foo() {
    firrtl.instance bbox_Foo @bbox_Foo()
  }

  firrtl.module @NoDutBehavior() {
    firrtl.instance foo @Foo()
  }
}

// CHECK-LABEL: firrtl.module @Foo
// CHECK-NEXT:    firrtl.instance bbox_Foo @bbox_Foo()

//===----------------------------------------------------------------------===//
// ExtractClockGates Simple
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractClockGatesSimple"
firrtl.circuit "ExtractClockGatesSimple" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %gate_in, %gate_en, %gate_out = firrtl.instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @ExtractClockGatesSimple
  firrtl.module @ExtractClockGatesSimple(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK: firrtl.instance gate @EICG_wrapper
    %dut_clock, %dut_en = firrtl.instance dut @DUTModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    firrtl.connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK:               emit.file "ClockGates.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     clock_gate_0 -> {{0}}.gate\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:              ]
}

//===----------------------------------------------------------------------===//
// ExtractClockGates TestHarnessOnly
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractClockGatesTestHarnessOnly"
firrtl.circuit "ExtractClockGatesTestHarnessOnly" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %0 = firrtl.add %in, %en : (!firrtl.uint<8>, !firrtl.uint<1>) -> !firrtl.uint<9>
    %_io_out_T = firrtl.node %0 : !firrtl.uint<9>
    %1 = firrtl.tail %_io_out_T, 1 : (!firrtl.uint<9>) -> !firrtl.uint<8>
    %_io_out_T_1 = firrtl.node %1 : !firrtl.uint<8>
    firrtl.connect %out, %_io_out_T_1 : !firrtl.uint<8>, !firrtl.uint<8>
  }
  firrtl.module @ExtractClockGatesTestHarnessOnly(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) {
    %gate_in, %gate_en, %gate_out = firrtl.instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %dut_clock, %dut_in, %dut_out, %dut_en = firrtl.instance dut @DUTModule(in clock: !firrtl.clock, in in: !firrtl.uint<8>, out out: !firrtl.uint<8>, in en: !firrtl.uint<1>)
    firrtl.connect %dut_clock, %gate_out : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-NOT: sv.verbatim "clock_gate
}

//===----------------------------------------------------------------------===//
// ExtractClockGates Mixed
//===----------------------------------------------------------------------===//

// Mixed ClockGate extraction should only extract clock gates in the DUT
// CHECK: firrtl.circuit "ExtractClockGatesMixed"
firrtl.circuit "ExtractClockGatesMixed" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  // CHECK-LABEL: firrtl.module private @Child
  firrtl.module private @Child(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: firrtl.instance gate sym @ckg1 @EICG_wrapper
    %gate_in, %gate_en, %gate_out = firrtl.instance gate sym @ckg1 @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module private @DUTModule
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %inst_clock, %inst_in, %inst_out, %inst_en = firrtl.instance inst sym @inst @Child(in clock: !firrtl.clock, in in: !firrtl.uint<8>, out out: !firrtl.uint<8>, in en: !firrtl.uint<1>)
    firrtl.connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %inst_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %inst_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: firrtl.instance gate sym @ckg2 @EICG_wrapper
    %gate_in, %gate_en, %gate_out = firrtl.instance gate sym @ckg2 @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ExtractClockGatesMixed
  firrtl.module @ExtractClockGatesMixed(in %clock: !firrtl.clock, in %intf_in: !firrtl.uint<8>, out %intf_out: !firrtl.uint<8>, in %intf_en: !firrtl.uint<1>, in %en: !firrtl.uint<1>) {
    // CHECK: firrtl.instance gate sym @ckg3 @EICG_wrapper
    %gate_in, %gate_en, %gate_out = firrtl.instance gate sym @ckg3 @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %dut_clock, %dut_in, %dut_out, %dut_en = firrtl.instance dut sym @dut @DUTModule(in clock: !firrtl.clock, in in: !firrtl.uint<8>, out out: !firrtl.uint<8>, in en: !firrtl.uint<1>)
    // CHECK: firrtl.instance gate sym @ckg2 @EICG_wrapper
    // CHECK: firrtl.instance gate sym @ckg1 @EICG_wrapper
    firrtl.connect %dut_clock, %gate_out : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_en, %intf_en : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %intf_out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %intf_in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK:               emit.file "ClockGates.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     clock_gate_0 -> {{0}}.{{1}}.gate\0A
  // CHECK-SAME{LITERAL}:     clock_gate_1 -> {{0}}.gate\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:                @DUTModule::@inst
  // CHECK-SAME:              ]
}

//===----------------------------------------------------------------------===//
// ExtractClockGates Composed
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractClockGatesComposed"
firrtl.circuit "ExtractClockGatesComposed" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"},
  {class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}
]} {
  // CHECK: hw.hierpath private @nla0 [@ExtractClockGatesComposed::[[SYM0:.+]], @EICG_wrapper]
  hw.hierpath private @nla0 [@DUTModule::@sym, @EICG_wrapper]
  // CHECK: hw.hierpath private @nla1 [@ExtractClockGatesComposed::[[SYM1:.+]], @EICG_wrapper]
  hw.hierpath private @nla1 [@Child::@sym, @EICG_wrapper]
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.memmodule @mem_ext() attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  firrtl.module private @Child(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    %gate_in, %gate_en, %gate_out = firrtl.instance gate sym @sym @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    firrtl.instance mem_ext @mem_ext()
    %gate_in, %gate_en, %gate_out = firrtl.instance gate sym @sym @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %child_clock, %child_en = firrtl.instance child @Child(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    firrtl.connect %child_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %child_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @ExtractClockGatesComposed
  firrtl.module @ExtractClockGatesComposed(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, out %metadataObj: !firrtl.anyref) {
    // CHECK: firrtl.instance gate sym [[SYM0]] @EICG_wrapper
    // CHECK: firrtl.instance gate sym [[SYM1]] @EICG_wrapper
    // CHECK: firrtl.instance mem_ext @mem_ext
    %dut_clock, %dut_en = firrtl.instance dut @DUTModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    firrtl.connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %sifive_metadata = firrtl.object @SiFive_Metadata()
    // CHECK:  firrtl.object @SiFive_Metadata(
    // CHECK-SAME: out extractedInstances_field0: !firrtl.class<@ExtractInstancesMetadata
    // CHECK-SAME: (out mem_wiring_0_field0: !firrtl.class<@ExtractInstancesSchema(in name_in: !firrtl.string, out name: !firrtl.string, in path_in: !firrtl.path, out path: !firrtl.path, in filename_in: !firrtl.string, out filename: !firrtl.string, in inst_name_in: !firrtl.string, out inst_name: !firrtl.string)>
    // CHECK-SAME: out clock_gate_0_field1: !firrtl.class<@ExtractInstancesSchema(in name_in: !firrtl.string, out name: !firrtl.string, in path_in: !firrtl.path, out path: !firrtl.path, in filename_in: !firrtl.string, out filename: !firrtl.string, in inst_name_in: !firrtl.string, out inst_name: !firrtl.string)>
    // CHECK-SAME: out clock_gate_1_field3: !firrtl.class<@ExtractInstancesSchema(in name_in: !firrtl.string, out name: !firrtl.string, in path_in: !firrtl.path, out path: !firrtl.path, in filename_in: !firrtl.string, out filename: !firrtl.string, in inst_name_in: !firrtl.string, out inst_name: !firrtl.string)>)>)
    %0 = firrtl.object.anyref_cast %sifive_metadata : !firrtl.class<@SiFive_Metadata()>
    firrtl.propassign %metadataObj, %0 : !firrtl.anyref
  }
  firrtl.class @SiFive_Metadata() {}

  // CHECK:               emit.file "SeqMems.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     mem_wiring_0 -> {{0}}.mem_ext\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:              ]

  // CHECK:               emit.file "ClockGates.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     clock_gate_0 -> {{0}}.{{1}}.gate\0A
  // CHECK-SAME{LITERAL}:     clock_gate_1 -> {{0}}.gate\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:                #hw.innerNameRef<@DUTModule::[[SYM0]]>
  // CHECK-SAME:              ]
}

//===----------------------------------------------------------------------===//
// ExtractSeqMems Simple2
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractSeqMemsSimple2"
firrtl.circuit "ExtractSeqMemsSimple2" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}]} {
  firrtl.memmodule @mem_ext() attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK-LABEL: firrtl.module @mem
  firrtl.module @mem() {
    // CHECK-NOT: firrtl.instance mem_ext @mem_ext
    firrtl.instance mem_ext @mem_ext()
  }
  // CHECK-LABEL: firrtl.module private @DUTModule
  firrtl.module private @DUTModule() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NEXT: firrtl.instance mem sym [[MEM_SYM:@.+]] @mem
    firrtl.instance mem @mem()
  }
  // CHECK-LABEL: firrtl.module @ExtractSeqMemsSimple2
  firrtl.module @ExtractSeqMemsSimple2() {
    firrtl.instance dut @DUTModule()
    // CHECK-NEXT: firrtl.instance dut sym [[DUT_SYM:@.+]] @DUTModule
    // CHECK-NEXT: firrtl.instance mem_ext @mem_ext
  }
  // CHECK:               emit.file "SeqMems.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     mem_wiring_0 -> {{0}}.{{1}}.mem_ext\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:                @DUTModule::[[MEM_SYM]]
  // CHECK-SAME:              ]
}

//===----------------------------------------------------------------------===//
// ExtractSeqMems NoExtraction
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractSeqMemsNoExtraction"
firrtl.circuit "ExtractSeqMemsNoExtraction"  attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}]} {
  firrtl.module @ExtractSeqMemsNoExtraction() {}
  // CHECK: emit.file "SeqMems.txt"
}

//===----------------------------------------------------------------------===//
// Conflicting Instance Symbols
// https://github.com/llvm/circt/issues/3089
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "InstSymConflict"
firrtl.circuit "InstSymConflict" {
  // CHECK-NOT: hw.hierpath private @nla_1
  // CHECK-NOT: hw.hierpath private @nla_2
  hw.hierpath private @nla_1 [
    @InstSymConflict::@dut,
    @DUTModule::@mod1,
    @BBWrapper::@bb
  ]
  hw.hierpath private @nla_2 [
    @InstSymConflict::@dut,
    @DUTModule::@mod2,
    @BBWrapper::@bb
  ]
  firrtl.extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {defname = "MyBlackBox"}
  firrtl.module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %bb_in, %bb_out = firrtl.instance bb sym @bb {annotations = [
        {class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "bb"},
        {circt.nonlocal = @nla_1, class = "DummyA"},
        {circt.nonlocal = @nla_2, class = "DummyB"}
      ]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %bb_in, %in : !firrtl.uint<8>
    firrtl.matchingconnect %out, %bb_out : !firrtl.uint<8>
  }
  firrtl.module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %mod1_in, %mod1_out = firrtl.instance mod1 sym @mod1 @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %mod2_in, %mod2_out = firrtl.instance mod2 sym @mod2 @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %mod1_in, %in : !firrtl.uint<8>
    firrtl.matchingconnect %mod2_in, %mod1_out : !firrtl.uint<8>
    firrtl.matchingconnect %out, %mod2_out : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @InstSymConflict
  firrtl.module @InstSymConflict(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK-NEXT: firrtl.instance dut sym @dut @DUTModule
    // CHECK: firrtl.instance bb sym @bb {annotations = [{class = "DummyB"}]} @MyBlackBox
    // CHECK: firrtl.instance bb sym @bb_0 {annotations = [{class = "DummyA"}]} @MyBlackBox
    %dut_in, %dut_out = firrtl.instance dut sym @dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %dut_in, %in : !firrtl.uint<8>
    firrtl.matchingconnect %out, %dut_out : !firrtl.uint<8>
  }
  // CHECK:               emit.file "BlackBoxes.txt" {
  // CHECK-NEXT:            sv.verbatim "
  // CHECK-SAME{LITERAL}:     bb_1 -> {{0}}.{{1}}.bb\0A
  // CHECK-SAME{LITERAL}:     bb_0 -> {{0}}.{{2}}.bb\0A
  // CHECK-SAME:              symbols = [
  // CHECK-SAME:                @DUTModule
  // CHECK-SAME:                #hw.innerNameRef<@DUTModule::@mod1>
  // CHECK-SAME:                #hw.innerNameRef<@DUTModule::@mod2>
  // CHECK-SAME:              ]
}

// Test that clock gate extraction composes with Chisel-time module prefixing.
// Chisel may add any number of prefixes to the clock gates.  Ensure that this
// will not block extraction.  Additionally, test that suffixes will block
// extraction.
//
// CHECK-LABEL: firrtl.circuit "PrefixedClockGate"
firrtl.circuit "PrefixedClockGate" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation",
      filename = "ckgates.txt",
      group = "ClockGates"
    }
  ]
} {
  firrtl.extmodule private @Prefix_EICG_wrapper() attributes {
    defname = "Prefix_EICG_wrapper"
  }
  firrtl.extmodule private @Prefix_EICG_wrapper_Suffix() attributes {
    defname = "Prefix_EICG_wrapper_Suffix"
  }
  // CHECK:      firrtl.module private @ClockGates() {
  // CHECK-NEXT:   firrtl.instance clockGate_0 @Prefix_EICG_wrapper()
  // CHECK-NOT:    firrtl.instance clockGate_1 @Prefix_EICG_wrapper_Suffix()
  //
  // CHECK:      firrtl.module @Foo
  // CHECK-NEXT:   firrtl.instance ClockGates {{.*}} @ClockGates()
  // CHECK-NEXT:   firrtl.instance clockGate_1 @Prefix_EICG_wrapper_Suffix()
  firrtl.module @Foo() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      }
    ]
  } {
    firrtl.instance clockGate_0 @Prefix_EICG_wrapper()
    firrtl.instance clockGate_1 @Prefix_EICG_wrapper_Suffix()
  }
  firrtl.module @PrefixedClockGate() {
    firrtl.instance foo @Foo()
  }
}
