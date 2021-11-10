// RUN: circt-opt %s --firrtl-grand-central-taps | FileCheck %s

firrtl.circuit "TestHarness" attributes {
  annotations = [{
    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    directory = "outputDirectory",
    filename = "outputDirectory/bindings.sv"
  }]
} {
  // CHECK-LABEL: firrtl.module @Bar
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
  firrtl.module @Bar(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.reset,
    in %in: !firrtl.uint<1>,
    out %out: !firrtl.uint<1>
  ) attributes
   {portAnnotations = [ [ {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 2 : i64,
      type = "source"
    }], [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 3 : i64,
      type = "source"
    } ],[],[] ] }
  {
    // CHECK-LABEL: %wire = firrtl.wire
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %wire = firrtl.wire {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 1 : i64,
      type = "source"
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<1>

    // CHECK-LABEL: %node = firrtl.node
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %node = firrtl.node %in {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 5 : i64,
      type = "source"
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<1>

    // CHECK-LABEL: %reg = firrtl.reg
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %reg = firrtl.reg %clock {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 6 : i64,
      type = "source"
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<1>

    // CHECK-LABEL: %regreset = firrtl.regreset
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %regreset = firrtl.regreset %clock, %reset, %in {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 7 : i64,
      type = "source"
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK-LABEL: firrtl.mem Undefined
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.MemTapAnnotation"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %mem = firrtl.mem Undefined {
      annotations = [{
        class = "sifive.enterprise.grandcentral.MemTapAnnotation",
        id = 4 : i64
      }, {
        class = "firrtl.transforms.DontTouchAnnotation"
      }],
      name = "mem",
      depth = 2 : i64,
      portNames = ["MPORT"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %mem_addr = firrtl.subfield %mem(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
    %mem_en = firrtl.subfield %mem(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
    %mem_clk = firrtl.subfield %mem(2) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.clock
    firrtl.connect %mem_addr, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %mem_en, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %mem_clk, %clock : !firrtl.clock, !firrtl.clock

    %42 = firrtl.not %in : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %wire, %42  : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %wire : !firrtl.uint<1>, !firrtl.uint<1>
  }

  firrtl.module @Foo(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.reset,
    in %in: !firrtl.uint<1>,
    out %out: !firrtl.uint<1>
  ) {
    %bar_clock, %bar_reset, %bar_in, %bar_out = firrtl.instance "bar" @Bar(in clock: !firrtl.clock, in reset: !firrtl.reset, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %bar_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %bar_reset, %reset : !firrtl.reset, !firrtl.reset
    firrtl.connect %bar_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %bar_out : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module @[[DT:DataTap.*]](
  // CHECK-SAME: out %_10: !firrtl.uint<4>
  // CHECK-SAME: out %_9: !firrtl.uint<1>
  // CHECK-SAME: out %_8: !firrtl.sint<8>
  // CHECK-SAME: out %_7: !firrtl.uint<1>
  // CHECK-SAME: out %_6: !firrtl.uint<1>
  // CHECK-SAME: out %_5: !firrtl.uint<1>
  // CHECK-SAME: out %_4: !firrtl.uint<1>
  // CHECK-SAME: out %_3: !firrtl.uint<1>
  // CHECK-SAME: out %_2: !firrtl.uint<1>
  // CHECK-SAME: out %_1: !firrtl.clock
  // CHECK-SAME: out %_0: !firrtl.uint<1>
  // CHECK-SAME: #hw.output_file<"outputDirectory/[[DT]].sv">
  // CHECK-NEXT: [[V10:%.+]] = firrtl.verbatim.expr "TestHarness.harnessWire"
  // CHECK-NEXT: firrtl.connect %_10, [[V10]]
  // CHECK-NEXT: [[V9:%.+]] = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %_9, [[V9]]
  // CHECK-NEXT: [[V8:%.+]] = firrtl.constant -42 : !firrtl.sint<8>
  // CHECK-NEXT: firrtl.connect %_8, [[V8]]
  // CHECK-NEXT: [[V7:%.+]] = firrtl.verbatim.expr "TestHarness.extmoduleWithTappedPort.out"
  // CHECK-NEXT: firrtl.connect %_7, [[V7]]
  // CHECK-NEXT: [[V6:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.regreset"
  // CHECK-NEXT: firrtl.connect %_6, [[V6]]
  // CHECK-NEXT: [[V5:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.reg"
  // CHECK-NEXT: firrtl.connect %_5, [[V5]]
  // CHECK-NEXT: [[V4:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.node"
  // CHECK-NEXT: firrtl.connect %_4, [[V4]]
  // CHECK-NEXT: [[V3:%.+]] = firrtl.verbatim.expr "TestHarness.bigScary.schwarzschild.no.more"
  // CHECK-NEXT: firrtl.connect %_3, [[V3]]
  // CHECK-NEXT: [[V2:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.reset"
  // CHECK-NEXT: firrtl.connect %_2, [[V2]]
  // CHECK-NEXT: [[V1:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.clock"
  // CHECK-NEXT: firrtl.connect %_1, [[V1]]
  // CHECK-NEXT: [[V0:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.wire"
  // CHECK-NEXT: firrtl.connect %_0, [[V0]]
  firrtl.extmodule @DataTap(
    out _10: !firrtl.uint<4>,
    out _9: !firrtl.uint<1>,
    out _8: !firrtl.sint<8>,
    out _7: !firrtl.uint<1>,
    out _6: !firrtl.uint<1>,
    out _5: !firrtl.uint<1>,
    out _4: !firrtl.uint<1>,
    out _3: !firrtl.uint<1>,
    out _2: !firrtl.uint<1>,
    out _1: !firrtl.clock,
    out _0: !firrtl.uint<1>
  ) attributes {
    annotations = [
      { class = "sifive.enterprise.grandcentral.DataTapsAnnotation" },
      { class = "firrtl.transforms.NoDedupAnnotation" }
    ],
    portAnnotations = [
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 11 : i64, type = "portName"}],
      [{class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "UInt<1>(\"h0\")", id = 0 : i64, portID = 10 : i64 }],
      [{class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "SInt<8>(\"h-2A\")", id = 0 : i64, portID = 9 : i64 }],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 8 : i64, type = "portName"}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 7 : i64, type = "portName"}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 6 : i64, type = "portName"}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 5 : i64, type = "portName"}],
      [{class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey", id = 0 : i64, portID = 4 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 3 : i64, type = "portName"}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 2 : i64, type = "portName"}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 1 : i64, type = "portName"}]
    ],
    defname = "DataTap"
  }

  // CHECK: firrtl.module @[[MT:MemTap.*]](
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.MemTapAnnotation"
  // CHECK-SAME: out %mem_0: !firrtl.uint<1>
  // CHECK-SAME: out %mem_1: !firrtl.uint<1>
  // CHECK-SAME: class = "firrtl.transforms.NoDedupAnnotation"
  // CHECK-SAME: #hw.output_file<"outputDirectory/[[MT]].sv">
  // CHECK-NEXT: [[V0:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.mem.Memory[0]"
  // CHECK-NEXT: firrtl.connect %mem_0, [[V0:%.+]]
  // CHECK-NEXT: [[V1:%.+]] = firrtl.verbatim.expr "TestHarness.foo.bar.mem.Memory[1]"
  // CHECK-NEXT: firrtl.connect %mem_1, [[V1:%.+]]
  firrtl.extmodule @MemTap(
    out mem_0: !firrtl.uint<1>,
    out mem_1: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {class = "firrtl.transforms.NoDedupAnnotation"}
    ],
    portAnnotations = [
      [{class = "sifive.enterprise.grandcentral.MemTapAnnotation", id = 4 : i64, word = 0 : i64}],
      [{class = "sifive.enterprise.grandcentral.MemTapAnnotation", id = 4 : i64, word = 1 : i64}]
    ],
    defname = "MemTap"
  }

  // CHECK-LABEL: firrtl.extmodule @BlackHole()
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey"
  firrtl.extmodule @BlackHole() attributes {
    annotations = [{
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "schwarzschild.no.more",
      id = 0 : i64,
      portID = 4 : i64 }]
  }

  firrtl.extmodule @ExtmoduleWithTappedPort(
    out out: !firrtl.uint<1>) attributes {portAnnotations = [[{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 8 : i64,
      type = "source" }]]}

  // CHECK: firrtl.module @TestHarness
  firrtl.module @TestHarness(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK-LABEL: %harnessWire = firrtl.wire
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %harnessWire = firrtl.wire {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 11 : i64,
      type = "source"
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<4>

    %foo_clock, %foo_reset, %foo_in, %foo_out = firrtl.instance foo @Foo(in clock: !firrtl.clock, in reset: !firrtl.reset, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %foo_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %foo_reset, %reset : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %foo_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %foo_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.instance bigScary @BlackHole()
    %0 = firrtl.instance extmoduleWithTappedPort @ExtmoduleWithTappedPort(out out: !firrtl.uint<1>)
    // CHECK: firrtl.instance dataTap @[[DT]]
    %DataTap_10, %DataTap_9, %DataTap_8, %DataTap_7, %DataTap_6, %DataTap_5, %DataTap_4, %DataTap_3, %DataTap_2, %DataTap_1, %DataTap_0 = firrtl.instance dataTap @DataTap(out _10: !firrtl.uint<4>, out _9: !firrtl.uint<1>, out _8: !firrtl.sint<8>, out _7: !firrtl.uint<1>, out _6: !firrtl.uint<1>, out _5: !firrtl.uint<1>, out _4: !firrtl.uint<1>, out _3: !firrtl.uint<1>, out _2: !firrtl.uint<1>, out _1: !firrtl.clock, out _0: !firrtl.uint<1>)
    // CHECK: firrtl.instance memTap @[[MT]]
    %MemTap_mem_0, %MemTap_mem_1 = firrtl.instance memTap @MemTap(out mem_0: !firrtl.uint<1>, out mem_1: !firrtl.uint<1>)
  }
}
