// RUN: circt-opt --firrtl-grand-central-signal-mappings %s | FileCheck %s

firrtl.circuit "SubCircuit" {
  firrtl.extmodule @FooExtern(in %clockIn: !firrtl.clock, out %clockOut: !firrtl.clock)
  firrtl.extmodule @BarExtern(in %someInput: !firrtl.uint<42>, out %someOutput: !firrtl.uint<42>)

  // CHECK-LABEL: firrtl.module @Foo_signal_mappings
  // CHECK-SAME:    in %clock_sink: !firrtl.clock
  // CHECK-SAME:    out %clock_source: !firrtl.clock
  // CHECK-SAME:  ) {
  // CHECK:         [[T1:%.+]] = firrtl.verbatim.wire "MainA.clock"
  // CHECK:         firrtl.force [[T1]], %clock_sink
  // CHECK:         [[T2:%.+]] = firrtl.verbatim.wire "MainB.clock"
  // CHECK:         firrtl.connect %clock_source, [[T2]]
  // CHECK:       }
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo() attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
      sinkTargets = [{_1 = "~Main|MainA>clock", _2 = "clock_sink"}],
      sourceTargets = [{_1 = "~Main|MainB>clock", _2 = "clock_source"}]
  }]} {
    %clock_source = firrtl.wire  : !firrtl.clock
    %clock_sink = firrtl.wire  : !firrtl.clock
    %ext_clockIn, %ext_clockOut = firrtl.instance @FooExtern {name = "ext"} : !firrtl.clock, !firrtl.clock
    firrtl.connect %ext_clockIn, %clock_source : !firrtl.clock, !firrtl.clock
    firrtl.connect %clock_sink, %ext_clockOut : !firrtl.clock, !firrtl.clock
  }

  // CHECK-LABEL: firrtl.module @Bar_signal_mappings
  // CHECK-SAME:    in %data_sink: !firrtl.uint<42>
  // CHECK-SAME:    out %data_source: !firrtl.uint<42>
  // CHECK-SAME:  ) {
  // CHECK:         [[T1:%.+]] = firrtl.verbatim.wire "MainA.dataIn_a_b_c"
  // CHECK:         firrtl.force [[T1]], %data_sink
  // CHECK:         [[T2:%.+]] = firrtl.verbatim.wire "MainA.dataOut_x_y_z"
  // CHECK:         firrtl.connect %data_source, [[T2]]
  // CHECK:       }
  firrtl.module @Bar() attributes {annotations = [{
    class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
    sinkTargets = [{_1 = "~Main|MainA>dataIn.a.b.c", _2 = "data_sink"}],
    sourceTargets = [{_1 = "~Main|MainA>dataOut.x.y.z", _2 = "data_source"}]
  }]} {
    %data_source = firrtl.wire  : !firrtl.uint<42>
    %data_sink = firrtl.wire  : !firrtl.uint<42>
    %ext_someInput, %ext_someOutput = firrtl.instance @BarExtern {name = "ext"} : !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %ext_someInput, %data_source : !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %data_sink, %ext_someOutput : !firrtl.uint<42>, !firrtl.uint<42>
  }

  firrtl.module @SubCircuit() {
    firrtl.instance @Foo {name = "foo"}
    firrtl.instance @Bar {name = "bar"}
  }
}
