// RUN: circt-opt --sifive-gct-taps --lower-firrtl-to-hw %s | FileCheck %s

firrtl.circuit "TestHarness" attributes {
  annotations = [{
    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
    directory = "builds/sandbox/dataTaps/firrtl",
    filename = "builds/sandbox/dataTaps/firrtl/bindings.sv"
  }]
} {
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
    %wire = firrtl.wire {annotations = [{
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      id = 0 : i64,
      portID = 1 : i64
    }]} : !firrtl.uint<1>

    %mem = firrtl.cmem {
      annotations = [{
        class = "sifive.enterprise.grandcentral.MemTapAnnotation",
        id = 4 : i64
      }, {
        class = "firrtl.transforms.DontTouchAnnotation"
      }],
      name = "mem"
    } : !firrtl.vector<uint<1>, 2>

    %MPORT = firrtl.memoryport Read %mem, %in, %clock {annotations = [], name = "MPORT"} : (!firrtl.vector<uint<1>, 2>, !firrtl.uint<1>, !firrtl.clock) -> !firrtl.uint<1>
    firrtl.connect %wire, %MPORT : !firrtl.uint<1>, !firrtl.uint<1>
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

  firrtl.extmodule @DataTap_1(
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
    defname = "DataTap_1"
  }

  firrtl.extmodule @MemTap_1(
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
    defname = "MemTap_1"
  }

  firrtl.extmodule @BlackHole() attributes {
    annotations = [{
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "schwarzschild.no.more",
      id = 0 : i64,
      portID = 4 : i64 }]
  }

  firrtl.module @TestHarness(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %foo_clock, %foo_reset, %foo_in, %foo_out = firrtl.instance @Foo  {name = "foo"} : !firrtl.flip<clock>, !firrtl.flip<reset>, !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %foo_clock, %clock : !firrtl.flip<clock>, !firrtl.clock
    firrtl.connect %foo_reset, %reset : !firrtl.flip<reset>, !firrtl.uint<1>
    firrtl.connect %foo_in, %in : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    firrtl.connect %out, %foo_out : !firrtl.uint<1>, !firrtl.uint<1>

    %dutDataTap_0 = firrtl.wire  : !firrtl.uint<1>
    %dutDataTap_1 = firrtl.wire  : !firrtl.clock
    %dutDataTap_2 = firrtl.wire  : !firrtl.uint<1>
    %dutDataTap_3 = firrtl.wire  : !firrtl.uint<1>
    %DataTap_1__3, %DataTap_1__2, %DataTap_1__1, %DataTap_1__0 = firrtl.instance @DataTap_1  {name = "DataTap_1"} : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.clock, !firrtl.uint<1>
    firrtl.connect %dutDataTap_0, %DataTap_1__0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %dutDataTap_1, %DataTap_1__1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %dutDataTap_2, %DataTap_1__2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %dutDataTap_3, %DataTap_1__3 : !firrtl.uint<1>, !firrtl.uint<1>

    %MemTap_1_mem_0, %MemTap_1_mem_1 = firrtl.instance @MemTap_1  {name = "MemTap_1"} : !firrtl.uint<1>, !firrtl.uint<1>
    %dutMemTap_0 = firrtl.wire  : !firrtl.uint<1>
    %dutMemTap_1 = firrtl.wire  : !firrtl.uint<1>
    firrtl.connect %dutMemTap_0, %MemTap_1_mem_0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %dutMemTap_1, %MemTap_1_mem_1 : !firrtl.uint<1>, !firrtl.uint<1>

    firrtl.instance @BlackHole {name = "bigScary0"}
  }
}
