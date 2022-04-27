// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-grand-central-signal-mappings)' --split-input-file %s | FileCheck %s

firrtl.circuit "SubCircuit" {
  firrtl.extmodule @FooExtern(in clockIn: !firrtl.clock, out clockOut: !firrtl.clock)
  firrtl.extmodule @BarExtern(in someInput: !firrtl.uint<42>, out someOutput: !firrtl.uint<42>)

  // Create a name collision for the signal mappings file to ensure it's
  // properly handled.
  firrtl.extmodule @Bar_signal_mappings()

  // CHECK-LABEL: firrtl.module @Foo_signal_mappings
  // CHECK-SAME:    out %clock_source: !firrtl.clock
  // CHECK-SAME:    in %clock_sink: !firrtl.clock
  // CHECK-SAME:  ) {
  // CHECK:         [[T1:%.+]] = firrtl.verbatim.wire "Main.clock"
  // CHECK:         firrtl.connect %clock_source, [[T1]]
  // CHECK:         [[T2:%.+]] = firrtl.verbatim.wire "Main.clock"
  // CHECK:         firrtl.force [[T2]], %clock_sink
  // CHECK:       }
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = 0 : i64}]} {
    %clock_source = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "source", id = 0 : i64, peer = "~Main|Main>clock", side = "local", targetId = 0 : i64}]} : !firrtl.clock
    %clock_sink = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "sink", id = 0 : i64, peer = "~Main|Main>clock", side = "local", targetId = 1 : i64}]} : !firrtl.clock
    %ext_clockIn, %ext_clockOut = firrtl.instance ext @FooExtern(in clockIn: !firrtl.clock, out clockOut: !firrtl.clock)
    firrtl.connect %ext_clockIn, %clock_source : !firrtl.clock, !firrtl.clock
    firrtl.connect %clock_sink, %ext_clockOut : !firrtl.clock, !firrtl.clock
    // CHECK: [[T1:%.+]], [[T2:%.+]] = firrtl.instance signal_mappings @Foo_signal_mappings
    // CHECK: firrtl.connect %clock_source, [[T1]] :
    // CHECK: firrtl.connect [[T2]], %clock_sink :
  }

  // CHECK-LABEL: firrtl.module @Bar_signal_mappings_0
  // CHECK-SAME:    out %data_source: !firrtl.uint<42>
  // CHECK-SAME:    in %data_sink: !firrtl.uint<42>
  // CHECK-SAME:  ) {
  // CHECK:         [[T1:%.+]] = firrtl.verbatim.wire "Main.dataOut_x_y_z"
  // CHECK:         firrtl.connect %data_source, [[T1]]
  // CHECK:         [[T2:%.+]] = firrtl.verbatim.wire "Main.dataIn_a_b_c"
  // CHECK:         firrtl.force [[T2]], %data_sink
  // CHECK:       }
  // CHECK-LABEL: firrtl.module @Bar
  firrtl.module @Bar() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = 1 : i64}]} {
    %data_source = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "source", id = 1 : i64, peer = "~Main|Main>dataOut.x.y.z", side = "local", targetId = 0 : i64}]} : !firrtl.uint<42>
    %data_sink = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "sink", id = 1 : i64, peer = "~Main|Main>dataIn.a.b.c", side = "local", targetId = 1 : i64}]} : !firrtl.uint<42>
    %ext_someInput, %ext_someOutput = firrtl.instance ext @BarExtern(in someInput: !firrtl.uint<42>, out someOutput: !firrtl.uint<42>)
    firrtl.connect %ext_someInput, %data_source : !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %data_sink, %ext_someOutput : !firrtl.uint<42>, !firrtl.uint<42>
    // CHECK: [[T1:%.+]], [[T2:%.+]] = firrtl.instance signal_mappings @Bar_signal_mappings_0
    // CHECK: firrtl.connect %data_source, [[T1]] :
    // CHECK: firrtl.connect [[T2]], %data_sink :
  }

  // CHECK-LABEL: firrtl.module @Baz_signal_mappings
  // CHECK-SAME:    out %data_source: !firrtl.uint<42>
  // CHECK-SAME:    in %data_sink: !firrtl.uint<42>
  // CHECK-SAME:  ) {
  // CHECK:         [[T1:%.+]] = firrtl.verbatim.wire "Main.foo.dataOut"
  // CHECK:         firrtl.connect %data_source, [[T1]]
  // CHECK:         [[T2:%.+]] = firrtl.verbatim.wire "Main.foo.dataIn"
  // CHECK:         firrtl.force [[T2]], %data_sink
  // CHECK:       }
  // CHECK-LABEL: firrtl.module @Baz
  firrtl.module @Baz(
    out %data_source: !firrtl.uint<42> [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "source", id = 2 : i64, peer = "~Main|Main/foo:MainA>dataOut", side = "local", targetId = 0 : i64}],
    in %data_sink: !firrtl.uint<42> [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "sink", id = 2 : i64, peer = "~Main|Main/foo:MainA>dataIn", side = "local", targetId = 1 : i64}]
  ) attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = 2 : i64}]} {
    // CHECK: [[T1:%.+]], [[T2:%.+]] = firrtl.instance signal_mappings @Baz_signal_mappings
    // CHECK: firrtl.connect %data_source, [[T1]] :
    // CHECK: firrtl.connect [[T2]], %data_sink :
  }

  firrtl.module @SubCircuit() {
    firrtl.instance foo @Foo()
    firrtl.instance bar @Bar()
    %baz_data_source, %baz_data_sink = firrtl.instance baz @Baz(out data_source: !firrtl.uint<42>, in data_sink: !firrtl.uint<42>)
  }
}

// -----

// Check that no work is done for a GCT signal driver that only drives
// zero-width ports.
//
// CHECK-LABEL: "signal_driver"
// CHECK-COUNT-1: firrtl.module
// CHECK-NOT:     firrtl.module
firrtl.circuit "signal_driver" {
  firrtl.module @signal_driver() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = 0 : i64}]} {
    %_w_sink = firrtl.wire sym @w_sink  {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "sink", id = 0 : i64, peer = "~Foo|Bar>w", side = "local", targetId = 1 : i64}]} : !firrtl.uint<0>
    %w_sink = firrtl.node sym @w_sink_0 %_w_sink  : !firrtl.uint<0>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.tail %c1_ui1, 1 : (!firrtl.uint<1>) -> !firrtl.uint<0>
    firrtl.connect %_w_sink, %0 : !firrtl.uint<0>, !firrtl.uint<0>
  }
}

// -----

// Check that GCT-SM generates a SiFive-specific JSON configuration file.  The
// following things are specifically checked:
//   1. "vendor.verilator.vsrcs" contains the subcircuit and mappings
//   2. "vendor.verilator.vsrcs" also contains subcircuit inline blackboxes
//   3. "load_jsons" contains undefined external modules with a .json suffix
//
// CHECK-LABEL: "GenerateJSON"
firrtl.circuit "GenerateJSON"  attributes {
  annotations = [
    {annotations = [],
    circuit = "",
    circuitPackage = "driving",
    class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
    emitJSON,
    id = 0 : i64}]} {
  firrtl.extmodule private @ExternalModule(out out: !firrtl.uint<1>) attributes {defname = "ExternalModule"}
  firrtl.extmodule private @InlineExternalModule(out out: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "firrtl.transforms.BlackBoxInlineAnno",
       name = "InlineExternalModule",
       text = "// hello"}],
       defname = "InlineExternalModule"}
  firrtl.module @GenerateJSON() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
       id = 0 : i64}]} {
    %source = firrtl.wire sym @source  {
      annotations = [
        {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
         dir = "source",
         id = 0 : i64,
         peer = "~SignalDrivingTop|SignalDrivingTop>_y",
         side = "local",
         targetId = 1 : i64}]} : !firrtl.uint<1>
    %sub_out = firrtl.instance sub  @ExternalModule(out out: !firrtl.uint<1>)
    %sub2_out = firrtl.instance sub2  @InlineExternalModule(out out: !firrtl.uint<1>)
  }
  // CHECK:      sv.verbatim "{
  // CHECK-SAME:   \22vendor\22: {
  // CHECK-SAME:     \22vcs\22: {
  // CHECK-SAME:       \22vsrcs\22: [
  // CHECK-SAME:         \22{{.+}}/GenerateJSON_signal_mappings.sv\22,
  // CHECK-SAME:         \22{{.+}}/GenerateJSON.sv\22,
  // CHECK-SAME:         \22{{.+}}/InlineExternalModule.sv\22
  // CHECK-SAME:       ]
  // CHECK-SAME:     },
  // CHECK-SAME:     \22verilator\22: {
  // CHECK-SAME:       \22error\22: [
  // CHECK-SAME:         \22force statement is not supported in verilator\22
  // CHECK-SAME:       ]
  // CHECK-SAME:     }
  // CHECK-SAME:   },
  // CHECK-SAME:   \22remove_vsrcs\22: [],
  // CHECK-SAME:   \22vsrcs\22: [],
  // CHECK-SAME:   \22load_jsons\22: [
  // CHECK-SAME:     \22ExternalModule.json\22
  // CHECK-SAME:   ]
  // CHECK-SAME: }"
  // CHECK-SAME: #hw.output_file<"driving.subcircuit.json", excludeFromFileList>
}

// -----

// Check that GCT-SM removes drivers of sources in the signal mapping module.
// This is needed because the Chisel-level API for describing GCT-SM sources
// will invalidate the sources.  MFC needs to clean these up to avoid doubly
// driven wires in the output.
//
// CHECK-LABEL: "RemoveDrivers"
firrtl.circuit "RemoveDrivers" {
  // CHECK: firrtl.module @RemoveDrivers
  firrtl.module @RemoveDrivers() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = 0 : i64}]} {
    %source = firrtl.wire sym @source  {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "source", id = 0 : i64, peer = "~Foo|Bar>w", side = "local", targetId = 1 : i64}]} : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    // CHECK-NOT: firrtl.strictconnect %source, %invalid_ui1
    firrtl.strictconnect %source, %invalid_ui1 : !firrtl.uint<1>
  }
}

// -----

// Check that GCT-SM generates a dummy wire for any input ports which are
// forced.  This is done to work around the way that SystemVerilog force
// statements work.  If an input port is forced, this will force that entire
// net.  If there is NOT a wire used at the connection of the input port, the
// effect of the force can be extremely far-reaching.  The Scala-based FIRRTL
// Compiler (SFC) _always_ emits a wire and never saw this problem.
// Specifically, check the following things:
//   1. An input port that is forced gets a dummy wire.
//   2. An output port that is forced does NOT get a dummy wire.
//
// CHECK-LABEL: firrtl.circuit "AddWireToForcedInputs"
firrtl.circuit "AddWireToForcedInputs"  attributes {
  annotations = [
    {annotations = [],
     circuit = "circuit empty :\0A  module empty :\0A\0A    skip\0A",
     circuitPackage = "driving",
     class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
     id = 0 : i64}]} {
  firrtl.module private @ForcedPort(
    in %in: !firrtl.uint<1> sym @in [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
       dir = "sink",
       id = 0 : i64,
       peer = "~signal_driver|signal_driver>in_sink",
       side = "remote",
       targetId = 3 : i64}],
    out %out: !firrtl.uint<1> sym @out [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
       dir = "sink",
       id = 0 : i64,
       peer = "~signal_driver|signal_driver>out_sink",
       side = "remote",
       targetId = 4 : i64}]) attributes {
    annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = 0 : i64}]} {}
  // CHECK: firrtl.module @AddWireToForcedInputs
  firrtl.module @AddWireToForcedInputs(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = 0 : i64}]} {
    // CHECK-NEXT: firrtl.instance sub
    %sub_in, %sub_out = firrtl.instance sub  @ForcedPort(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    // CHECK-NEXT: %[[sub_in_wire:.+]] = firrtl.wire sym @[[sub_in_wire_sym:.+]] : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %sub_in, %[[sub_in_wire]]
    // CHECK-NEXT: firrtl.strictconnect %[[sub_in_wire]], %in
    firrtl.strictconnect %sub_in, %in : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %out, %sub_out
    firrtl.strictconnect %out, %sub_out : !firrtl.uint<1>
    // CHECK-NEXT: }
  }
}
