// RUN: circt-opt %s --firrtl-grand-central-taps --symbol-dce --split-input-file | FileCheck %s

firrtl.circuit "TestHarness" attributes {
  annotations = [{
    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    directory = "outputDirectory",
    filename = "bindings.sv"
  }]
} {
  // CHECK: firrtl.module private @Bar
  // CHECK-SAME: in %clock: !firrtl.clock sym [[BAR_CLOCK:@[0-9a-zA-Z_]+]]
  // CHECK-SAME: in %reset: !firrtl.reset sym [[BAR_RESET:@[0-9a-zA-Z_]+]]
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
  firrtl.module private @Bar(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.reset,
    in %in: !firrtl.uint<1>,
    out %out: !firrtl.uint<1>
  ) attributes
   {portAnnotations = [ [ {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 2 : i64
    }], [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 3 : i64
    } ],[],[] ] }
  {
    // CHECK: %wire = firrtl.wire sym [[WIRE:@[0-9a-zA-Z_]+]]
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %wire = firrtl.wire {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 1 : i64
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<1>

    // CHECK: %node = firrtl.node sym [[NODE:@[0-9a-zA-Z_]+]]
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %node = firrtl.node %in {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 5 : i64
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<1>

    // CHECK: %reg = firrtl.reg sym [[REG:@[0-9a-zA-Z_]+]]
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %reg = firrtl.reg %clock {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 6 : i64
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<1>

    // CHECK: %regreset = firrtl.regreset sym [[REGRESET:@[0-9a-zA-Z_]+]]
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %regreset = firrtl.regreset %clock, %reset, %in {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 7 : i64
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>

    %array = firrtl.wire {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 12 : i64,
      circt.fieldID = 2
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.vector<uint<1>, 2>

    %mem_0 = firrtl.reg %clock  {annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 4 : i64, portID = 0 : i64}, {class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    %mem_1 = firrtl.reg %clock  {annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 4 : i64, portID = 1 : i64}, {class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK:   %mem_0 = firrtl.reg sym @[[gct_sym_5:.+]] %clock  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK:   %mem_1 = firrtl.reg sym @[[gct_sym_6:.+]] %clock  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    %mem = firrtl.mem Undefined {
      annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 42 : i64}],
      name = "mem",
      depth = 2 : i64,
      portNames = ["MPORT"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    // CHECK: firrtl.mem sym @[[gct_sym_7:.+]] Undefined
    %mem_addr = firrtl.subfield %mem[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %mem_en = firrtl.subfield %mem[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %mem_clk = firrtl.subfield %mem[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.connect %mem_addr, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %mem_en, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %mem_clk, %clock : !firrtl.clock, !firrtl.clock

    %42 = firrtl.not %in : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %wire, %42  : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %wire : !firrtl.uint<1>, !firrtl.uint<1>
  }

  firrtl.module private @Foo(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.reset,
    in %in: !firrtl.uint<1>,
    out %out: !firrtl.uint<1>
  ) {
    // CHECK: firrtl.instance bar sym [[BAR:@[0-9a-zA-Z_]+]]
    %bar_clock, %bar_reset, %bar_in, %bar_out = firrtl.instance bar @Bar(in clock: !firrtl.clock, in reset: !firrtl.reset, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %bar_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %bar_reset, %reset : !firrtl.reset, !firrtl.reset
    firrtl.connect %bar_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %bar_out : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module private @[[DT:DataTap.*]](
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
  // CHECK-SAME: #hw.output_file<"outputDirectory{{/|\\\\}}[[DT]].sv">
  // CHECK-NEXT: [[V11:%.+]] = firrtl.verbatim.expr
  // CHECK-SAME{LITERAL}: "{{0}}.{{1}}.{{2}}.{{3}}[1]"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[HARNESSWIRE:@[0-9a-zA-Z_]+]]>
  // CHECK-NEXT: firrtl.connect %_11, [[V11]]
  // CHECK-NEXT: [[V10:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[HARNESSWIRE:@[0-9a-zA-Z_]+]]>
  // CHECK-NEXT: firrtl.connect %_10, [[V10]]
  // CHECK-NEXT: [[V9:%.+]] = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK-NEXT: firrtl.connect %_9, [[V9]]
  // CHECK-NEXT: [[V8:%.+]] = firrtl.constant -42 : !firrtl.sint<8>
  // CHECK-NEXT: firrtl.connect %_8, [[V8]]
  // CHECK-NEXT: [[V7:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[EXTMODULEWITHTAPPEDPORT:@[0-9a-zA-Z_]+]]>,
  // CHECK-SAME:   #hw.innerNameRef<@ExtmoduleWithTappedPort::[[EXTMODULEWITHTAPPEDPORTOUT:@[0-9a-zA-Z_]+]]>
  // CHECK-NEXT: firrtl.connect %_7, [[V7]]
  // CHECK-NEXT: [[V6:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO:@[0-9a-zA-Z_]+]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::[[REGRESET]]>
  // CHECK-NEXT: firrtl.connect %_6, [[V6]]
  // CHECK-NEXT: [[V5:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::[[REG]]>
  // CHECK-NEXT: firrtl.connect %_5, [[V5]]
  // CHECK-NEXT: [[V4:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::[[NODE]]>
  // CHECK-NEXT: firrtl.connect %_4, [[V4]]
  // CHECK-NEXT: [[V3:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.schwarzschild.no.more"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[BIGSCARY:@[0-9a-zA-Z_]+]]>
  // CHECK-NEXT: firrtl.connect %_3, [[V3]]
  // CHECK-NEXT: [[V2:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::[[BAR_RESET]]>
  // CHECK-NEXT: firrtl.connect %_2, [[V2]]
  // CHECK-NEXT: [[V1:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::[[BAR_CLOCK]]>
  // CHECK-NEXT: firrtl.connect %_1, [[V1]]
  // CHECK-NEXT: [[V0:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::[[WIRE]]>
  // CHECK-NEXT: firrtl.connect %_0, [[V0]]
  firrtl.extmodule private @DataTap(
    out _11: !firrtl.uint<1>,
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
      { class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox" },
      { class = "firrtl.transforms.NoDedupAnnotation" }
    ],
    portAnnotations = [
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 12 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 11 : i64}],
      [{class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "UInt<1>(\"h0\")", id = 0 : i64, portID = 10 : i64 }],
      [{class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "SInt<8>(\"h-2A\")", id = 0 : i64, portID = 9 : i64 }],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 8 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 7 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 6 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 5 : i64}],
      [{class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.port", id = 0 : i64, portID = 4 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 3 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 2 : i64}],
      [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port", id = 0 : i64, portID = 1 : i64}]
    ],
    defname = "DataTap"
  }

  // CHECK: firrtl.module private @[[MT:MemTap.*]](
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.MemTapAnnotation.port"
  // CHECK-SAME: out %mem_0: !firrtl.uint<1>
  // CHECK-SAME: out %mem_1: !firrtl.uint<1>
  // CHECK-SAME: class = "firrtl.transforms.NoDedupAnnotation"
  // CHECK-SAME: #hw.output_file<"outputDirectory{{/|\\\\}}[[MT]].sv">
  // CHECK-NEXT: [[V0:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::@[[gct_sym_5]]>
  // CHECK-NEXT: firrtl.connect %mem_0, [[V0:%.+]]
  // CHECK-NEXT: [[V1:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::@[[gct_sym_6]]>
  // CHECK-NEXT: firrtl.connect %mem_1, [[V1:%.+]]
  firrtl.extmodule private @MemTap(
    out mem_0: !firrtl.uint<1>,
    out mem_1: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {class = "firrtl.transforms.NoDedupAnnotation"}
    ],
    portAnnotations = [
      [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.port", id = 4 : i64, portID = 0 : i64}],
      [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.port", id = 4 : i64, portID = 1 : i64}]
    ],
    defname = "MemTap"
  }

  // CHECK: firrtl.module private @[[MT2:MemTap2.*]](
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.MemTapAnnotation.port"
  // CHECK-SAME: out %mem_0: !firrtl.uint<1>
  // CHECK-SAME: out %mem_1: !firrtl.uint<1>
  // CHECK-SAME: class = "firrtl.transforms.NoDedupAnnotation"
  // CHECK-SAME: #hw.output_file<"outputDirectory{{/|\\\\}}[[MT2]].sv">
  // CHECK-NEXT: [[V0:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}.Memory[0]"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::@[[gct_sym_7]]>
  // CHECK-NEXT: firrtl.connect %mem_0, [[V0:%.+]]
  // CHECK-NEXT: [[V1:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}}.{{[{][{]1[}][}]}}.{{[{][{]2[}][}]}}.{{[{][{]3[}][}]}}.Memory[1]"
  // CHECK-SAME:   @TestHarness,
  // CHECK-SAME:   #hw.innerNameRef<@TestHarness::[[FOO]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Foo::[[BAR]]>,
  // CHECK-SAME:   #hw.innerNameRef<@Bar::@[[gct_sym_7]]>
  // CHECK-NEXT: firrtl.connect %mem_1, [[V1:%.+]]
  firrtl.extmodule private @MemTap2(
    out mem_0: !firrtl.uint<1>,
    out mem_1: !firrtl.uint<1>
  ) attributes {
    annotations = [
      {class = "firrtl.transforms.NoDedupAnnotation"}
    ],
    portAnnotations = [
      [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.port", id = 42 : i64, portID = 0 : i64}],
      [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.port", id = 42 : i64, portID = 1 : i64}]
    ],
    defname = "MemTap2"
  }

  // CHECK-LABEL: firrtl.extmodule private @BlackHole()
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.source"
  firrtl.extmodule private @BlackHole() attributes {
    annotations = [{
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.source",
      internalPath = "schwarzschild.no.more",
      id = 0 : i64,
      portID = 4 : i64 }]
  }

  // CHECK-LABEL: firrtl.extmodule private @ExtmoduleWithTappedPort
  // CHECK-SAME: out out: !firrtl.uint<1> sym [[EXTMODULEWITHTAPPEDPORTOUT]]
  firrtl.extmodule private @ExtmoduleWithTappedPort(
    out out: !firrtl.uint<1>) attributes {portAnnotations = [[{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 8 : i64}]]}

  // CHECK: firrtl.module @TestHarness
  firrtl.module @TestHarness(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK: %harnessWire = firrtl.wire sym [[HARNESSWIRE]]
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %harnessWire = firrtl.wire {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 11 : i64
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<4>

    // CHECK: firrtl.instance foo sym [[FOO]]
    %foo_clock, %foo_reset, %foo_in, %foo_out = firrtl.instance foo @Foo(in clock: !firrtl.clock, in reset: !firrtl.reset, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.connect %foo_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %foo_reset, %reset : !firrtl.reset, !firrtl.uint<1>
    firrtl.connect %foo_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %foo_out : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.instance bigScary sym [[BIGSCARY]]
    firrtl.instance bigScary @BlackHole()
    // CHECK: firrtl.instance extmoduleWithTappedPort sym [[EXTMODULEWITHTAPPEDPORT]]
    %0 = firrtl.instance extmoduleWithTappedPort @ExtmoduleWithTappedPort(out out: !firrtl.uint<1>)
    // CHECK: firrtl.instance dataTap @[[DT]]
    %DataTap_11, %DataTap_10, %DataTap_9, %DataTap_8, %DataTap_7, %DataTap_6, %DataTap_5, %DataTap_4, %DataTap_3, %DataTap_2, %DataTap_1, %DataTap_0 = firrtl.instance dataTap @DataTap(out _11: !firrtl.uint<1>, out _10: !firrtl.uint<4>, out _9: !firrtl.uint<1>, out _8: !firrtl.sint<8>, out _7: !firrtl.uint<1>, out _6: !firrtl.uint<1>, out _5: !firrtl.uint<1>, out _4: !firrtl.uint<1>, out _3: !firrtl.uint<1>, out _2: !firrtl.uint<1>, out _1: !firrtl.clock, out _0: !firrtl.uint<1>)
    // CHECK: firrtl.instance memTap @[[MT]]
    // CHECK: firrtl.instance memTap2 @[[MT2]]
    %MemTap_mem_0, %MemTap_mem_1 = firrtl.instance memTap @MemTap(out mem_0: !firrtl.uint<1>, out mem_1: !firrtl.uint<1>)
    %MemTap2_mem_0, %MemTap2_mem_1 = firrtl.instance memTap2 @MemTap2(out mem_0: !firrtl.uint<1>, out mem_1: !firrtl.uint<1>)
  }
}

// -----

// Test that NLAs are properly garbage collected.  After this runs, no NLAs
// should exist in the circuit.
//
// CHECK-LABEL: firrtl.circuit "NLAGarbageCollection"
firrtl.circuit "NLAGarbageCollection" {
  // CHECK-NOT: @nla_1
  // CHECK-NOT: @nla_2
  // CHECK-NOT: @nla_3
  hw.hierpath private @nla_1 [@NLAGarbageCollection::@dut, @DUT::@submodule, @Submodule::@foo]
  hw.hierpath private @nla_2 [@NLAGarbageCollection::@dut, @DUT::@submodule, @Submodule::@port]
  hw.hierpath private @nla_3 [@NLAGarbageCollection::@dut, @DUT::@submodule, @Submodule::@bar_0]
  firrtl.module private @Submodule(
    in %port: !firrtl.uint<1> sym @port [
      {circt.nonlocal = @nla_2,
       class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
       id = 1 : i64,
       portID = 3 : i64}]
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %foo = firrtl.wire sym @foo {
      annotations = [
        {circt.nonlocal = @nla_1,
         class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
         id = 1 : i64,
         portID = 2 : i64}]} : !firrtl.uint<1>
    %bar_out_MPORT_clk = firrtl.wire  : !firrtl.clock
    %bar_0 = firrtl.reg sym @bar_0 %bar_out_MPORT_clk  {annotations = [{circt.nonlocal = @nla_3, class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 0 : i64, portID = 0 : i64}]} : !firrtl.uint<1>
    // CHECK:  %bar_0 = firrtl.reg sym @[[bar_0:.+]] %bar_out_MPORT_clk  : !firrtl.uint<1>
    %bar_out_MPORT = firrtl.mem Undefined {
      depth = 1 : i64,
      name = "bar",
      portNames = ["out_MPORT"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  }
  firrtl.extmodule private @MemTap(
    out mem_0: !firrtl.uint<1> [
      {class = "sifive.enterprise.grandcentral.MemTapAnnotation.port",
       id = 0 : i64,
       portID = 0 : i64}]
  ) attributes {
    annotations = [
      {class = "firrtl.transforms.NoDedupAnnotation"}],
    defname = "MemTap"
  }
  firrtl.extmodule private @DataTap_1(
    out _1: !firrtl.uint<1> sym @_1 [
      {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
       id = 1 : i64,
       portID = 3 : i64}],
    out _0: !firrtl.uint<1> sym @_0 [
      {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
       id = 1 : i64,
       portID = 2 : i64}]
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"},
      {class = "firrtl.transforms.DontTouchAnnotation"},
      {class = "firrtl.transforms.NoDedupAnnotation"}],
    defname = "DataTap_1"}
  firrtl.module private @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {
    %submodule_port = firrtl.instance submodule sym @submodule @Submodule(in port : !firrtl.uint<1>)
    %MemTap_0 = firrtl.instance mem_tap_MemTap  @MemTap(out mem_0: !firrtl.uint<1>)
    %DataTap_0, %DataTap_1 = firrtl.instance DataTap_1  @DataTap_1(out _1: !firrtl.uint<1>, out _0: !firrtl.uint<1>)
  }
  firrtl.module @NLAGarbageCollection() {
    firrtl.instance dut sym @dut @DUT()
  }
}

// -----

// Check that NLAs are used to wire up the data tap port connections properly.
// See https://github.com/llvm/circt/issues/2691.

// CHECK-LABEL: firrtl.circuit "NLAUsedInWiring"
firrtl.circuit "NLAUsedInWiring"  {
  // CHECK-NOT: @nla_1
  // CHECK-NOT: @nla_2
  hw.hierpath private @nla_1 [@NLAUsedInWiring::@foo, @Foo::@f]
  hw.hierpath private @nla_2 [@NLAUsedInWiring::@foo, @Foo::@g]

  // CHECK-LABEL: firrtl.module private @DataTap
  // CHECK-NEXT: [[TMP:%.+]] = firrtl.verbatim.expr
  // CHECK-SAME:   symbols = [@NLAUsedInWiring, #hw.innerNameRef<@NLAUsedInWiring::@foo>, #hw.innerNameRef<@Foo::@f>]
  // CHECK-NEXT: firrtl.connect %b, [[TMP]] : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: [[TMP:%.+]] = firrtl.verbatim.expr
  // CHECK-SAME:   symbols = [@NLAUsedInWiring, #hw.innerNameRef<@NLAUsedInWiring::@foo>, #hw.innerNameRef<@Foo::@g>]
  // CHECK-NEXT: firrtl.connect %c, [[TMP]] : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: [[TMP:%.+]] = firrtl.verbatim.expr
  // CHECK-SAME:   symbols = [@NLAUsedInWiring, #hw.innerNameRef<@NLAUsedInWiring::@k>]
  // CHECK-NEXT: firrtl.connect %d, [[TMP]] : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.extmodule private @DataTap(
    out b: !firrtl.uint<1> sym @b [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
      id = 0 : i64,
      portID = 1 : i64
    }],
    out c: !firrtl.uint<1> sym @c [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
      id = 0 : i64,
      portID = 2 : i64
    }],
    out d: !firrtl.uint<1> sym @d [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
      id = 0 : i64,
      portID = 3 : i64
    }]
  ) attributes {
    annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}],
    defname = "DataTap"
  }

  firrtl.module private @Foo(
    out %g: !firrtl.uint<1> sym @g [{
      circt.nonlocal = @nla_2,
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 2 : i64
    }]
  ) {
    %f = firrtl.wire sym @f {annotations = [{
      circt.nonlocal = @nla_1,
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 1 : i64
    }]} : !firrtl.uint<1>
  }

  firrtl.module @NLAUsedInWiring() {
    %foo_g = firrtl.instance foo sym @foo @Foo(out g: !firrtl.uint<1>)
    %bar_g = firrtl.instance bar @Foo(out g: !firrtl.uint<1>)
    %k = firrtl.wire sym @k {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
      id = 0 : i64,
      portID = 3 : i64
    }]} : !firrtl.uint<1>
    %dataTap_b, %dataTap_c, %dataTap_d = firrtl.instance dataTap @DataTap(out b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d: !firrtl.uint<1>)
  }
}

// -----

// Check that NLAs not rooted at the top module still produce correct XMRs.
// See https://github.com/llvm/circt/issues/2767.

firrtl.circuit "Top" {
  hw.hierpath private @nla_0 [@DUT::@submodule_1, @Submodule::@bar_0]
  hw.hierpath private @nla [@DUT::@submodule_2, @Submodule::@bar_0]
  firrtl.module private @Submodule(in %clock: !firrtl.clock, out %out: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %bar_0 = firrtl.reg sym @bar_0 %clock  {annotations = [{circt.nonlocal = @nla, class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 1 : i64, portID = 0 : i64}, {circt.nonlocal = @nla_0, class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 0 : i64, portID = 0 : i64}]} : !firrtl.uint<1>
    // CHECK:  %bar_0 = firrtl.reg sym @[[bar_0:.+]] %clock  : !firrtl.uint<1>
    %bar_out_MPORT = firrtl.mem sym @bar Undefined  {depth = 1 : i64, modName = "bar_ext", name = "bar", portNames = ["out_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %0 = firrtl.subfield %bar_out_MPORT[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %1 = firrtl.subfield %bar_out_MPORT[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %2 = firrtl.subfield %bar_out_MPORT[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %3 = firrtl.subfield %bar_out_MPORT[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    firrtl.strictconnect %0, %c0_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %1, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %2, %clock : !firrtl.clock
    firrtl.strictconnect %out, %3 : !firrtl.uint<1>
  }
  firrtl.module private @DUT(in %clock: !firrtl.clock, out %out: !firrtl.uint<1>) {
    %submodule_1_clock, %submodule_1_out = firrtl.instance submodule_1 sym @submodule_1 @Submodule(in clock: !firrtl.clock, out out: !firrtl.uint<1>)
    firrtl.strictconnect %submodule_1_clock, %clock : !firrtl.clock
    %submodule_2_clock, %submodule_2_out = firrtl.instance submodule_2 sym @submodule_2 @Submodule(in clock: !firrtl.clock, out out: !firrtl.uint<1>)
    firrtl.strictconnect %submodule_2_clock, %clock : !firrtl.clock
    %0 = firrtl.or %submodule_1_out, %submodule_2_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %out, %0 : !firrtl.uint<1>
    %mem_tap_MemTap_1_mem_0 = firrtl.instance mem_tap_MemTap_1  @MemTap_1(out mem_0: !firrtl.uint<1>)
    %mem_tap_MemTap_2_mem_0 = firrtl.instance mem_tap_MemTap_2  @MemTap_2(out mem_0: !firrtl.uint<1>)
  }
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %dut_clock, %dut_out = firrtl.instance dut sym @dut @DUT(in clock: !firrtl.clock, out out: !firrtl.uint<1>)
    firrtl.strictconnect %dut_clock, %clock : !firrtl.clock
    firrtl.strictconnect %out, %dut_out : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module private @MemTap_1_impl_0
  // CHECK-NEXT{LITERAL}: firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}"
  // CHECK-SAME: symbols = [@DUT, #hw.innerNameRef<@DUT::@submodule_1>, #hw.innerNameRef<@Submodule::@[[bar_0]]>]
  firrtl.extmodule private @MemTap_1(out mem_0: !firrtl.uint<1> [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.port", id = 0 : i64, portID = 0 : i64}]) attributes {defname = "MemTap"}
  // CHECK-LABEL: firrtl.module private @MemTap_2_impl_0
  // CHECK-NEXT{LITERAL}: firrtl.verbatim.expr "{{0}}.{{1}}.{{2}}"
  // CHECK-SAME: symbols = [@DUT, #hw.innerNameRef<@DUT::@submodule_2>, #hw.innerNameRef<@Submodule::@[[bar_0]]>]
  firrtl.extmodule private @MemTap_2(out mem_0: !firrtl.uint<1> [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.port", id = 1 : i64, portID = 0 : i64}]) attributes {defname = "MemTap"}
}

// -----

// Check that an empty data tap module and its instantiations is deleted.

// CHECK-LABEL: "Top"
// CHECK-NOT: firrtl.extmodule {{.+}}@DataTap
// CHECK-NOT: firrtl.instance DataTap @DataTap()
firrtl.circuit "Top" {
  firrtl.extmodule @DataTap() attributes {annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}]}
  firrtl.module @DUT() {
    firrtl.instance DataTap @DataTap()
  }
  firrtl.module @Top() {
    firrtl.instance dut @DUT()
  }
}

// -----

// Check that zero-width data taps are no-ops.  These should generate no XMRs.
//
firrtl.circuit "Top"  {
  firrtl.extmodule private @DataTap(
    out _0: !firrtl.uint<0> [
      {
        class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
        id = 0 : i64,
        portID = 1 : i64
      }
    ],
    out _1: !firrtl.uint<0> [
      {
        class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
        id = 0 : i64,
        portID = 2 : i64
      }
    ]) attributes {annotations = [
      {class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}
    ]}
  firrtl.module @Top(
    out %p: !firrtl.uint<0> [
      {
        class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
        id = 0 : i64,
        portID = 2 : i64
      }
    ]) {
    %w = firrtl.wire {annotations = [
      {
        class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
        id = 0 : i64,
        portID = 1 : i64
      }
    ]} : !firrtl.uint<0>
    %tap_0, %tap_1 = firrtl.instance tap @DataTap(
      out _0: !firrtl.uint<0>,
      out _1: !firrtl.uint<0>
    )
  }
}

// CHECK-NOT: firrtl.verbatim.expr

// -----

// Check that constants are sunk into XMRs and symbols are not unnecessarily
// created.
//
// CHECK-LABEL: "ConstantSinking"
firrtl.circuit "ConstantSinking"  {
  // CHECK:      firrtl.module private @DataTap
  // CHECK-NEXT:   %[[one:.+]] = firrtl.constant 1
  // CHECK-NEXT:   firrtl.connect %a, %[[one]]
  firrtl.extmodule private @DataTap(
    out a: !firrtl.uint<1> [
      {
        class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
        id = 0 : i64,
        portID = 1 : i64
      }
    ]) attributes {annotations = [
      {
        class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"
      }
    ]}
  // CHECK: firrtl.module @ConstantSinking
  firrtl.module @ConstantSinking() {
    %dataTap_a = firrtl.instance dataTap interesting_name  @DataTap(out a: !firrtl.uint<1>)
    // CHECK:    %w = firrtl.wire
    // CHECK-NOT   sym
    %w = firrtl.wire {annotations = [
      {
        class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
        id = 0 : i64,
        portID = 1 : i64
      }
    ]} : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %w, %c1_ui1 : !firrtl.uint<1>
  }
}
