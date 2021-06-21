// RUN: circt-opt %s --firrtl-grand-central-taps | FileCheck %s

firrtl.circuit "TestHarness" attributes {
  annotations = [{
    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    directory = "builds/sandbox/dataTaps/firrtl",
    filename = "builds/sandbox/dataTaps/firrtl/bindings.sv"
  }]
} {
  // CHECK-LABEL: firrtl.module @Bar
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
  firrtl.module @Bar(
    in %clock: !firrtl.clock {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 2 : i64
    }]},
    in %reset: !firrtl.reset {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 3 : i64
    }]},
    in %in: !firrtl.uint<1>,
    out %out: !firrtl.uint<1>
  ) {
    // CHECK-LABEL: %wire = firrtl.wire
    // CHECK-NOT: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: class = "firrtl.transforms.DontTouchAnnotation"
    %wire = firrtl.wire {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 1 : i64
    }, {
      class = "firrtl.transforms.DontTouchAnnotation"
    }]} : !firrtl.uint<1>

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
    } : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>>
    %mem_addr = firrtl.subfield %mem("addr") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>>) -> !firrtl.uint<1>
    %mem_en = firrtl.subfield %mem("en") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>>) -> !firrtl.uint<1>
    %mem_clk = firrtl.subfield %mem("clk") : (!firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>>) -> !firrtl.clock
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
    %bar_clock, %bar_reset, %bar_in, %bar_out = firrtl.instance @Bar  {name = "bar"} : !firrtl.flip<clock>, !firrtl.flip<reset>, !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %bar_clock, %clock : !firrtl.flip<clock>, !firrtl.clock
    firrtl.connect %bar_reset, %reset : !firrtl.flip<reset>, !firrtl.reset
    firrtl.connect %bar_in, %in : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %out, %bar_out : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module [[DT:@DataTap.*]](
  // CHECK-SAME: out %_3: !firrtl.uint<1>
  // CHECK-SAME: out %_2: !firrtl.uint<1>
  // CHECK-SAME: out %_1: !firrtl.clock
  // CHECK-SAME: out %_0: !firrtl.uint<1>
  // CHECK-NEXT: [[V3:%.+]] = firrtl.verbatim.expr "bigScary.schwarzschild.no.more"
  // CHECK-NEXT: firrtl.connect %_3, [[V3]]
  // CHECK-NEXT: [[V2:%.+]] = firrtl.verbatim.expr "foo.bar.reset"
  // CHECK-NEXT: firrtl.connect %_2, [[V2]]
  // CHECK-NEXT: [[V1:%.+]] = firrtl.verbatim.expr "foo.bar.clock"
  // CHECK-NEXT: firrtl.connect %_1, [[V1]]
  // CHECK-NEXT: [[V0:%.+]] = firrtl.verbatim.expr "foo.bar.wire"
  // CHECK-NEXT: firrtl.connect %_0, [[V0]]
  firrtl.extmodule @DataTap(
    out %_3: !firrtl.uint<1> {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "schwarzschild.no.more",
      id = 0 : i64,
      portID = 4 : i64 }]},
    out %_2: !firrtl.uint<1> {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 3 : i64 }]},
    out %_1: !firrtl.clock {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 2 : i64 }]},
    out %_0: !firrtl.uint<1> {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 1 : i64 }]}
  ) attributes {
    annotations = [
      { class = "sifive.enterprise.grandcentral.DataTapsAnnotation" },
      { class = "firrtl.transforms.NoDedupAnnotation" }
    ],
    defname = "DataTap"
  }

  // CHECK: firrtl.module [[MT:@MemTap.*]](
  // CHECK-NOT: class = "sifive.enterprise.grandcentral.MemTapAnnotation"
  // CHECK-SAME: out %mem_0: !firrtl.uint<1>
  // CHECK-SAME: out %mem_1: !firrtl.uint<1>
  // CHECK-SAME: class = "firrtl.transforms.NoDedupAnnotation"
  // CHECK-NEXT: [[V0:%.+]] = firrtl.verbatim.expr "foo.bar.mem[0]"
  // CHECK-NEXT: firrtl.connect %mem_0, [[V0:%.+]]
  // CHECK-NEXT: [[V1:%.+]] = firrtl.verbatim.expr "foo.bar.mem[1]"
  // CHECK-NEXT: firrtl.connect %mem_1, [[V1:%.+]]
  firrtl.extmodule @MemTap(
    out %mem_0: !firrtl.uint<1> {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.MemTapAnnotation",
      id = 4 : i64 }]},
    out %mem_1: !firrtl.uint<1> {firrtl.annotations = [{
      class = "sifive.enterprise.grandcentral.MemTapAnnotation",
      id = 4 : i64 }]}
  ) attributes {
    annotations = [
      {class = "firrtl.transforms.NoDedupAnnotation"}
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

  // CHECK: firrtl.module @TestHarness
  firrtl.module @TestHarness(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %foo_clock, %foo_reset, %foo_in, %foo_out = firrtl.instance @Foo {name = "foo"} : !firrtl.flip<clock>, !firrtl.flip<reset>, !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %foo_clock, %clock : !firrtl.flip<clock>, !firrtl.clock
    firrtl.connect %foo_reset, %reset : !firrtl.flip<reset>, !firrtl.uint<1>
    firrtl.connect %foo_in, %in : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %out, %foo_out : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.instance @BlackHole {name = "bigScary"}
    // CHECK: firrtl.instance [[DT]] {name = "dataTap"}
    %DataTap_3, %DataTap_2, %DataTap_1, %DataTap_0 = firrtl.instance @DataTap {name = "dataTap"} : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.instance [[MT]] {name = "memTap"}
    %MemTap_mem_0, %MemTap_mem_1 = firrtl.instance @MemTap {name = "memTap"} : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
