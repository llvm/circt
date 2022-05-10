// RUN: circt-opt --firrtl-extract-instances --verify-diagnostics %s | FileCheck %s

// Tests extracted from:
// - test/scala/firrtl/ExtractBlackBoxes.scala
// - test/scala/firrtl/ExtractClockGates.scala

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
    firrtl.strictconnect %bb_in, %invalid_ui8 : !firrtl.uint<8>
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
    // CHECK: %mod_in, %mod_out, %mod_bb_0_in, %mod_bb_0_out = firrtl.instance mod sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NEXT: firrtl.strictconnect %bb_0_in, %mod_bb_0_in
    // CHECK-NEXT: firrtl.strictconnect %mod_bb_0_out, %bb_0_out
    %mod_in, %mod_out = firrtl.instance mod @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ExtractBlackBoxesSimple
  firrtl.module @ExtractBlackBoxesSimple(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK: %dut_in, %dut_out, %dut_bb_0_in, %dut_bb_0_out = firrtl.instance dut sym {{@.+}} @DUTModule
    // CHECK-NEXT: %bb_in, %bb_out = firrtl.instance bb sym [[BB_SYM:@.+]] @MyBlackBox
    // CHECK-NEXT: firrtl.strictconnect %bb_in, %dut_bb_0_in
    // CHECK-NEXT: firrtl.strictconnect %dut_bb_0_out, %bb_out
    %dut_in, %dut_out = firrtl.instance dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: bb_0 -> {{0}}.{{1}}.{{2}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"BlackBoxes.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::[[WRAPPER_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@ExtractBlackBoxesSimple::[[BB_SYM]]>
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes Simple (modified)
// ExtractBlackBoxes RenameTargets (modified)
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractBlackBoxesSimple2"
firrtl.circuit "ExtractBlackBoxesSimple2" attributes {annotations = [{class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "BlackBoxes"}]} {
  // CHECK: firrtl.nla @nla_1 [#hw.innerNameRef<@ExtractBlackBoxesSimple2::@bb>, @MyBlackBox]
  // CHECK-NOT: firrtl.nla @nla_2
  // CHECK-NOT: firrtl.nla @nla_3
  firrtl.nla @nla_1 [
    #hw.innerNameRef<@BBWrapper::@bb>,
    @MyBlackBox
  ]
  firrtl.nla @nla_2 [
    #hw.innerNameRef<@DUTModule::@mod>,
    #hw.innerNameRef<@BBWrapper::@bb>
  ]
  firrtl.nla @nla_3 [
    #hw.innerNameRef<@ExtractBlackBoxesSimple2::@dut>,
    #hw.innerNameRef<@DUTModule::@mod>,
    #hw.innerNameRef<@BBWrapper::@bb>
  ]
  // Annotation on the extmodule itself
  // CHECK-LABEL: firrtl.extmodule private @MyBlackBox
  firrtl.extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [
      {class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "prefix"},
      {circt.nonlocal = @nla_1, class = "DummyA"}
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
        {circt.nonlocal = @nla_1, class = "circt.nonlocal"},
        {circt.nonlocal = @nla_2, class = "DummyB"},
        {circt.nonlocal = @nla_3, class = "DummyC"}
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
    // CHECK-NOT: annotations =
    // CHECK-SAME: sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NEXT: firrtl.strictconnect %prefix_1_in, %mod_prefix_1_in
    // CHECK-NEXT: firrtl.strictconnect %mod_prefix_1_out, %prefix_1_out
    // CHECK-NEXT: firrtl.strictconnect %prefix_0_in, %mod_prefix_0_in
    // CHECK-NEXT: firrtl.strictconnect %mod_prefix_0_out, %prefix_0_out
    %mod_in, %mod_out = firrtl.instance mod sym @mod {annotations = [
        {circt.nonlocal = @nla_2, class = "circt.nonlocal"},
        {circt.nonlocal = @nla_3, class = "circt.nonlocal"}
      ]} @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ExtractBlackBoxesSimple2
  firrtl.module @ExtractBlackBoxesSimple2(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK: %dut_in, %dut_out, %dut_prefix_0_in, %dut_prefix_0_out, %dut_prefix_1_in, %dut_prefix_1_out = firrtl.instance dut
    // CHECK-NOT: annotations =
    // CHECK-SAME: sym {{@.+}} @DUTModule
    // CHECK-NEXT: %bb_in, %bb_out = firrtl.instance bb sym [[BB_SYM:@.+]] {annotations = [{class = "DummyB"}, {class = "DummyC"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @MyBlackBox
    // CHECK-NEXT: firrtl.strictconnect %bb_in, %dut_prefix_1_in
    // CHECK-NEXT: firrtl.strictconnect %dut_prefix_1_out, %bb_out
    // CHECK-NEXT: %bb2_in, %bb2_out = firrtl.instance bb2 sym [[BB2_SYM:@.+]] @MyBlackBox2
    // CHECK-NEXT: firrtl.strictconnect %bb2_in, %dut_prefix_0_in
    // CHECK-NEXT: firrtl.strictconnect %dut_prefix_0_out, %bb2_out
    %dut_in, %dut_out = firrtl.instance dut sym @dut {annotations = [
        {circt.nonlocal = @nla_3, class = "circt.nonlocal"}
      ]} @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: prefix_0 -> {{0}}.{{1}}.{{2}}\0A
  // CHECK-SAME{LITERAL}: prefix_1 -> {{0}}.{{1}}.{{3}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"BlackBoxes.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::[[WRAPPER_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@ExtractBlackBoxesSimple2::[[BB2_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@ExtractBlackBoxesSimple2::[[BB_SYM]]>
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes IntoDUTSubmodule
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractBlackBoxesIntoDUTSubmodule"
firrtl.circuit "ExtractBlackBoxesIntoDUTSubmodule"  {
  // CHECK-LABEL: firrtl.nla @nla_1 [
  // CHECK-SAME:    #hw.innerNameRef<@ExtractBlackBoxesIntoDUTSubmodule::@tb>
  // CHECK-SAME:    #hw.innerNameRef<@TestHarness::@dut>
  // CHECK-SAME:    #hw.innerNameRef<@DUTModule::@BlackBoxes>
  // CHECK-SAME:    #hw.innerNameRef<@BlackBoxes::@bb1>
  // CHECK-SAME:  ]
  firrtl.nla @nla_1 [
    #hw.innerNameRef<@ExtractBlackBoxesIntoDUTSubmodule::@tb>,
    #hw.innerNameRef<@TestHarness::@dut>,
    #hw.innerNameRef<@DUTModule::@mod>,
    #hw.innerNameRef<@BBWrapper::@bb1>
  ]
  // CHECK-LABEL: firrtl.nla @nla_2 [
  // CHECK-SAME:    #hw.innerNameRef<@ExtractBlackBoxesIntoDUTSubmodule::@tb>
  // CHECK-SAME:    #hw.innerNameRef<@TestHarness::@dut>
  // CHECK-SAME:    #hw.innerNameRef<@DUTModule::@BlackBoxes>
  // CHECK-SAME:    #hw.innerNameRef<@BlackBoxes::@bb2>
  // CHECK-SAME:  ]
  firrtl.nla @nla_2 [
    #hw.innerNameRef<@ExtractBlackBoxesIntoDUTSubmodule::@tb>,
    #hw.innerNameRef<@TestHarness::@dut>,
    #hw.innerNameRef<@DUTModule::@mod>,
    #hw.innerNameRef<@BBWrapper::@bb2>
  ]
  firrtl.extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", dest = "BlackBoxes", filename = "BlackBoxes.txt", prefix = "bb"}], defname = "MyBlackBox"}
  firrtl.module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %bb1_in, %bb1_out = firrtl.instance bb1 sym @bb1 {annotations = [{circt.nonlocal = @nla_1, class = "Dummy1"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %bb2_in, %bb2_out = firrtl.instance bb2 sym @bb2 {annotations = [{circt.nonlocal = @nla_2, class = "Dummy2"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %bb2_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %bb2_in, %bb1_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %bb1_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @BlackBoxes(
  // CHECK-SAME:    in %bb_0_in: !firrtl.uint<8>
  // CHECK-SAME:    out %bb_0_out: !firrtl.uint<8>
  // CHECK-SAME:    in %bb_1_in: !firrtl.uint<8>
  // CHECK-SAME:    out %bb_1_out: !firrtl.uint<8>
  // CHECK-SAME:  ) {
  // CHECK-NEXT:    %bb2_in, %bb2_out = firrtl.instance bb2 sym [[BB2_SYM:@.+]] {annotations = [{circt.nonlocal = @nla_2, class = "Dummy2"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
  // CHECK-NEXT:    firrtl.strictconnect %bb2_in, %bb_0_in : !firrtl.uint<8>
  // CHECK-NEXT:    firrtl.strictconnect %bb_0_out, %bb2_out : !firrtl.uint<8>
  // CHECK-NEXT:    %bb1_in, %bb1_out = firrtl.instance bb1 sym [[BB1_SYM:@.+]] {annotations = [{circt.nonlocal = @nla_1, class = "Dummy1"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
  // CHECK-NEXT:    firrtl.strictconnect %bb1_in, %bb_1_in : !firrtl.uint<8>
  // CHECK-NEXT:    firrtl.strictconnect %bb_1_out, %bb1_out : !firrtl.uint<8>
  // CHECK-NEXT:  }
  // CHECK-LABEL: firrtl.module private @DUTModule
  firrtl.module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: %BlackBoxes_bb_0_in, %BlackBoxes_bb_0_out, %BlackBoxes_bb_1_in, %BlackBoxes_bb_1_out = firrtl.instance BlackBoxes sym @BlackBoxes
    // CHECK-SAME: {circt.nonlocal = @nla_2, class = "circt.nonlocal"}
    // CHECK-SAME: {circt.nonlocal = @nla_1, class = "circt.nonlocal"}
    // CHECK-SAME: @BlackBoxes
    // CHECK-NEXT: %mod_in, %mod_out, %mod_bb_0_in, %mod_bb_0_out, %mod_bb_1_in, %mod_bb_1_out = firrtl.instance mod
    // CHECK-NOT: annotations =
    // CHECK-SAME: sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NEXT: firrtl.strictconnect %BlackBoxes_bb_1_in, %mod_bb_1_in
    // CHECK-NEXT: firrtl.strictconnect %mod_bb_1_out, %BlackBoxes_bb_1_out
    // CHECK-NEXT: firrtl.strictconnect %BlackBoxes_bb_0_in, %mod_bb_0_in
    // CHECK-NEXT: firrtl.strictconnect %mod_bb_0_out, %BlackBoxes_bb_0_out
    %mod_in, %mod_out = firrtl.instance mod sym @mod {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}]} @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  firrtl.module @TestHarness(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %dut_in, %dut_out = firrtl.instance dut sym @dut {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}]} @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  firrtl.module @ExtractBlackBoxesIntoDUTSubmodule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %tb_in, %tb_out = firrtl.instance tb sym @tb {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}]} @TestHarness(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %out, %tb_out : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %tb_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: bb_0 -> {{0}}.{{1}}.{{2}}\0A
  // CHECK-SAME{LITERAL}: bb_1 -> {{0}}.{{1}}.{{3}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"BlackBoxes.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::[[WRAPPER_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@BlackBoxes::[[BB2_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@BlackBoxes::[[BB1_SYM]]>
  // CHECK-SAME: ]
}

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
    // CHECK: firrtl.instance gate sym [[CKG_SYM:@.+]] @EICG_wrapper
    %dut_clock, %dut_en = firrtl.instance dut @DUTModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    firrtl.connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@ExtractClockGatesSimple::[[CKG_SYM]]>
  // CHECK-SAME: ]
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
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}.{{1}}.{{2}}\0A
  // CHECK-SAME{LITERAL}: clock_gate_1 -> {{0}}.{{3}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::@inst>
  // CHECK-SAME: #hw.innerNameRef<@ExtractClockGatesMixed::@ckg1>
  // CHECK-SAME: #hw.innerNameRef<@ExtractClockGatesMixed::@ckg2>
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractClockGates Composed
//===----------------------------------------------------------------------===//

// CHECK: firrtl.circuit "ExtractClockGatesComposed"
firrtl.circuit "ExtractClockGatesComposed" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"},
  {class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}
]} {
  firrtl.extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  firrtl.memmodule @mem_ext() attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    firrtl.instance mem_ext @mem_ext()
    %gate_in, %gate_en, %gate_out = firrtl.instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @ExtractClockGatesComposed
  firrtl.module @ExtractClockGatesComposed(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK: firrtl.instance gate sym [[CKG_SYM:@.+]] @EICG_wrapper
    // CHECK: firrtl.instance mem_ext sym [[MEM_SYM:@.+]] @mem_ext
    %dut_clock, %dut_en = firrtl.instance dut @DUTModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    firrtl.connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@ExtractClockGatesComposed::[[CKG_SYM]]>
  // CHECK-SAME: ]
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: mem_wiring_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"SeqMems.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@ExtractClockGatesComposed::[[MEM_SYM]]>
  // CHECK-SAME: ]
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
    // CHECK-NEXT: firrtl.instance mem_ext sym [[MEM_EXT_SYM:@.+]] @mem_ext
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: mem_wiring_0 -> {{0}}.{{1}}.{{2}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"SeqMems.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::[[MEM_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@ExtractSeqMemsSimple2::[[MEM_EXT_SYM]]>
  // CHECK-SAME: ]
}
