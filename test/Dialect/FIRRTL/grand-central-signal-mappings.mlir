// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-grand-central-signal-mappings))' --split-input-file %s | FileCheck %s

firrtl.circuit "SubCircuit" attributes {
  annotations = [
    {annotations = [],
    circuit = "",
    circuitPackage = "driving",
    class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
    isSubCircuit = true,
    id = 0 : i64}]} {
  firrtl.extmodule @FooExtern(in clockIn: !firrtl.clock, out clockOut: !firrtl.clock)
  firrtl.extmodule @BarExtern(in someInput: !firrtl.uint<42>, out someOutput: !firrtl.uint<42>)

  // Create a name collision for the signal mappings file to ensure it's
  // properly handled.
  // CHECK-LABEL: firrtl.extmodule @Bar_signal_mappings(
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
  firrtl.module @Foo() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 0 : i64}]} {
    %clock_source = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = 0 : i64, peer = "~Main|Main>clock", side = "local", targetId = 0 : i64}]} : !firrtl.clock
    %clock_sink = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = 0 : i64, peer = "~Main|Main>clock", side = "local", targetId = 1 : i64}]} : !firrtl.clock
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
  firrtl.module @Bar() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 1 : i64}]} {
    %data_source = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = 1 : i64, peer = "~Main|Main>dataOut.x.y.z", side = "local", targetId = 0 : i64}]} : !firrtl.uint<42>
    %data_sink = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = 1 : i64, peer = "~Main|Main>dataIn.a.b.c", side = "local", targetId = 1 : i64}]} : !firrtl.uint<42>
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
    out %data_source: !firrtl.uint<42> [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = 2 : i64, peer = "~Main|Main/foo:MainA>dataOut", side = "local", targetId = 0 : i64}],
    in %data_sink: !firrtl.uint<42> [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = 2 : i64, peer = "~Main|Main/foo:MainA>dataIn", side = "local", targetId = 1 : i64}]
  ) attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 2 : i64}]} {
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
firrtl.circuit "signal_driver" attributes {
  annotations = [
    {annotations = [],
    circuit = "",
    circuitPackage = "driving",
    class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
    isSubCircuit = true,
    id = 0 : i64}]} {
  firrtl.module @signal_driver() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 0 : i64}]} {
    %_w_sink = firrtl.wire sym @w_sink  {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "sink", id = 0 : i64, peer = "~Foo|Bar>w", side = "local", targetId = 1 : i64}]} : !firrtl.uint<0>
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
    isSubCircuit = true,
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
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module",
       id = 0 : i64}]} {
    %source = firrtl.wire sym @source  {
      annotations = [
        {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
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
  // CHECK-SAME:         \22{{.+}}GenerateJSON_signal_mappings.sv\22,
  // CHECK-SAME:         \22{{.+}}GenerateJSON.sv\22,
  // CHECK-SAME:         \22{{.+}}InlineExternalModule.sv\22
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
firrtl.circuit "RemoveDrivers" attributes {
  annotations = [
    {annotations = [],
    circuit = "",
    circuitPackage = "driving",
    class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
    isSubCircuit = true,
    id = 0 : i64}]} {
  // CHECK: firrtl.module @RemoveDrivers
  firrtl.module @RemoveDrivers() attributes {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 0 : i64}]} {
    %source = firrtl.wire sym @source  {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target", dir = "source", id = 0 : i64, peer = "~Foo|Bar>w", side = "local", targetId = 1 : i64}]} : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    // CHECK-NOT: firrtl.strictconnect %source, %invalid_ui1
    firrtl.strictconnect %source, %invalid_ui1 : !firrtl.uint<1>
  }
}

// -----

// Check that GCT-SM generates 2 dummy wires for any ports which are forced.
// This is done to work around the way that SystemVerilog force statements
// work.  If a port is forced, this will force that entire net.  If there is
// NOT a wire used at the connection of the port, the effect of the force can
// be extremely far-reaching.  The Scala-based FIRRTL Compiler (SFC) _always_
// emits a wire and never saw this problem. Two wires are used so an 'assign'
// is created, which is what breaks the net.
//
// For forced input ports, the buffer wires are created at the instantiation
// site.  For forced output ports, the buffer wires are created inside the
// module.
// CHECK-LABEL: firrtl.circuit "AddWireToForcedPorts"
firrtl.circuit "AddWireToForcedPorts"  attributes {
  annotations = [
    {annotations = [],
     circuit = "circuit empty :\0A  module empty :\0A\0A    skip\0A",
     circuitPackage = "driving",
     class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
     isSubCircuit = false,
     id = 0 : i64}]} {
  // CHECK: firrtl.module private @ForcedPort
  firrtl.module private @ForcedPort(
    in %in: !firrtl.uint<1> sym @in [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
       dir = "sink",
       id = 0 : i64,
       peer = "~signal_driver|signal_driver>in_sink",
       side = "remote",
       targetId = 3 : i64}],
    out %out: !firrtl.uint<1> sym @out [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
       dir = "sink",
       id = 0 : i64,
       peer = "~signal_driver|signal_driver>out_sink",
       side = "remote",
       targetId = 4 : i64}]) attributes {
    annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 0 : i64}]} {
      // CHECK-NEXT: %[[buffer_wire:.+]] = firrtl.wire sym @{{.+}}
      // CHECK-NEXT: %[[port_wire:.+]] = firrtl.wire sym @{{.+}}
      // CHECK-NEXT: firrtl.strictconnect %out, %[[buffer_wire]]
      // CHECK-NEXT: firrtl.strictconnect %[[buffer_wire]], %[[port_wire]]
      // CHECK-NEXT: firrtl.strictconnect %[[port_wire]], %in
      firrtl.strictconnect %out, %in : !firrtl.uint<1>
      // CHECK-NEXT: }
    }
  // CHECK: firrtl.module @AddWireToForcedPorts
  firrtl.module @AddWireToForcedPorts(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 0 : i64}]} {
    // CHECK-NEXT: firrtl.instance sub
    %sub_in, %sub_out = firrtl.instance sub  @ForcedPort(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    // CHECK-NEXT: %[[buffer_wire:.+]] = firrtl.wire sym @{{.+}} : !firrtl.uint<1>
    // CHECK-NEXT: %[[sub_in_wire:.+]] = firrtl.wire sym @{{.+}} : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %sub_in, %[[buffer_wire]]
    // CHECK-NEXT: firrtl.strictconnect %[[buffer_wire]], %[[sub_in_wire]]
    // CHECK-NEXT: firrtl.strictconnect %[[sub_in_wire]], %in
    firrtl.strictconnect %sub_in, %in : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %out, %sub_out
    firrtl.strictconnect %out, %sub_out : !firrtl.uint<1>
    // CHECK-NEXT: }
  }
}

// -----

// Check remote-side handles targets needing hierpath's,
// with and without path going through the DUT.

// CHECK-LABEL: firrtl.circuit "MainWithNLA"
firrtl.circuit "MainWithNLA" attributes {
  annotations = [
    {annotations = [],
     circuit = "circuit empty :\0A  module empty :\0A\0A    skip\0A",
     circuitPackage = "driving",
     class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
     id = 0 : i64,
     isSubCircuit = false
    }
  ]} {
  // Starting from DUT
  firrtl.hierpath private @nla_dut_rel [@DUT::@l, @Leaf::@w]
  // Not through DUT, describes multiple paths
  firrtl.hierpath private @nla_segment [@Mid::@l, @Leaf::@in]
  // Top to leaf, through the DUT
  firrtl.hierpath private @nla_top_thru_dut_to_w [@MainWithNLA::@dut, @DUT::@m, @Mid::@l, @Leaf::@w]
  firrtl.module private @Leaf(
    in %in: !firrtl.uint<1> sym @in [{
      circt.nonlocal = @nla_segment,
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "source",
      id = 0 : i64,
      peer = "~Sub|Sub>in_source",
      side = "remote",
      targetId = 2 : i64
    }, {
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "sink",
      id = 0 : i64,
      peer = "~Sub|Sub>in_sink",
      side = "remote",
      targetId = 3 : i64
    }],
    out %out: !firrtl.uint<1>) attributes {
      annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 0 : i64}]
    } {
    %w = firrtl.wire sym @w  {
      annotations = [{
        circt.nonlocal = @nla_top_thru_dut_to_w,
        class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
        dir = "source",
        id = 0 : i64,
        peer = "~Sub|Sub>w_source",
        side = "remote",
        targetId = 1 : i64
      }, {
        circt.nonlocal = @nla_dut_rel,
        class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
        dir = "sink",
        id = 0 : i64,
        peer = "~Sub|Sub>w_sink",
        side = "remote",
        targetId = 4 : i64}
       ]} : !firrtl.uint<1>
    firrtl.strictconnect %w, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %w : !firrtl.uint<1>
  }
  firrtl.module private @Mid(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %l_in, %l_out = firrtl.instance l sym @l  @Leaf(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %l_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %l_out : !firrtl.uint<1>
  }
  firrtl.module @MainWithNLA(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %dut_in, %dut_out = firrtl.instance dut sym @dut  @DUT(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %dut_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %dut_out : !firrtl.uint<1>
    %m_in, %m_out = firrtl.instance m  @Mid(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %m_in, %in : !firrtl.uint<1>
  }
  firrtl.module private @DUT(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %l_in, %l_out = firrtl.instance l sym @l  @Leaf(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %m_in, %m_out = firrtl.instance m sym @m  @Mid(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %l_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %m_in, %in : !firrtl.uint<1>
    %0 = firrtl.or %l_out, %m_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %out, %0 : !firrtl.uint<1>
  }
  // CHECK:      sv.verbatim "[
  // CHECK-SAME:   {
  // CHECK-SAME:     \22sinkTargets\22: [
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{1}}>{{2}}\22
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>in_sink\22
  // CHECK-SAME:       },
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{0}}/{{4}}:{{1}}>{{3}}\22,
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>w_sink\22
  // CHECK-SAME:       }
  // CHECK-SAME:     ],
  // CHECK-SAME:     \22sourceTargets\22: [
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{5}}/{{6}}:{{1}}>{{2}}\22,
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>in_source\22
  // CHECK-SAME:       },
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{0}}/{{7}}:{{5}}/{{6}}:{{1}}>{{3}}\22,
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>w_source\22
  // CHECK-SAME:       }
  // CHECK-SAME:     ],
  // CHECK-SAME:   }
  // CHECK-SAME: ]"
  // CHECK-SAME: {
  // CHECK-SAME:   output_file =
  // CHECK-SAME:     #hw.output_file<"sigdrive.json", excludeFromFileList>
  // CHECK-SAME:   symbols = [
  // CHECK-SAME:     @DUT,
  // CHECK-SAME:     @Leaf,
  // CHECK-SAME:     #hw.innerNameRef<@Leaf::@in>,
  // CHECK-SAME:     #hw.innerNameRef<@Leaf::@w>,
  // CHECK-SAME:     #hw.innerNameRef<@DUT::@l>,
  // CHECK-SAME:     @Mid,
  // CHECK-SAME:     #hw.innerNameRef<@Mid::@l>,
  // CHECK-SAME:     #hw.innerNameRef<@DUT::@m>
  // CHECK-SAME:    ]
  // CHECK-SAME: }
}

// -----

// Check local-side emits XMR's walking paths

// CHECK-LABEL: firrtl.circuit "Sub"
firrtl.circuit "Sub" attributes {
  annotations = [
    {annotations = [],
    circuit = "",
    circuitPackage = "driving",
    class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
    isSubCircuit = true,
    id = 0 : i64}]} {
  firrtl.extmodule private @SubExtern(
    in a: !firrtl.uint<1>, in b: !firrtl.uint<1>,
    out c: !firrtl.uint<1>, out d: !firrtl.uint<1>)

  firrtl.module @Sub() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module",
      id = 0 : i64}]} {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %w_source = firrtl.wire sym @w_source  {annotations = [{
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "source",
      id = 0 : i64,
      peer = "~DUT|DUT/m:Mid/l:Leaf>w",
      side = "local",
      targetId = 1 : i64}]} : !firrtl.uint<1>
    firrtl.strictconnect %w_source, %c0_ui1 : !firrtl.uint<1>
    %w_sink = firrtl.wire sym @w_sink  {annotations = [{
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "sink",
      id = 0 : i64,
      peer = "~DUT|DUT/l:Leaf>w",
      side = "local",
      targetId = 2 : i64}]} : !firrtl.uint<1>
    %in_source = firrtl.wire sym @in_source  {annotations = [{
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "source",
      id = 0 : i64,
      peer = "~DUT|Mid/l:Leaf>in",
      side = "local",
      targetId = 3 : i64}]} : !firrtl.uint<1>
    firrtl.strictconnect %in_source, %c0_ui1 : !firrtl.uint<1>
    %in_sink = firrtl.wire sym @in_sink  {annotations = [{
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "sink",
      id = 0 : i64,
      peer = "~DUT|Leaf>in",
      side = "local",
      targetId = 4 : i64}]} : !firrtl.uint<1>
    %ext_a, %ext_b, %ext_c, %ext_d = firrtl.instance ext  @SubExtern(
      in a: !firrtl.uint<1>, in b: !firrtl.uint<1>,
      out c: !firrtl.uint<1>, out d: !firrtl.uint<1>)
    firrtl.strictconnect %ext_a, %w_source : !firrtl.uint<1>
    firrtl.strictconnect %ext_b, %in_source : !firrtl.uint<1>
    firrtl.strictconnect %w_sink, %ext_c : !firrtl.uint<1>
    firrtl.strictconnect %in_sink, %ext_d : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @Sub_signal_mappings
  // CHECK:         %[[T1:.+]] = firrtl.verbatim.wire "DUT.m.l.w"
  // CHECK-NEXT:    firrtl.connect %w_source, %[[T1]]
  // CHECK-NEXT:    %[[T2:.+]] = firrtl.verbatim.wire "DUT.l.w"
  // CHECK-NEXT:    firrtl.force %[[T2]], %w_sink
  // CHECK-NEXT:    %[[T3:.+]] = firrtl.verbatim.wire "Mid.l.in"
  // CHECK-NEXT:    firrtl.connect %in_source, %[[T3]]
  // CHECK-NEXT:    %[[T4:.+]] = firrtl.verbatim.wire "Leaf.in"
  // CHECK-NEXT:    firrtl.force %[[T4]], %in_sink
}

// -----

// Check remote-side handles targets needing new format hierpath's,
// with and without path going through the DUT.

// CHECK-LABEL: firrtl.circuit "MainWithnewNLA"
firrtl.circuit "MainWithnewNLA" attributes {
  annotations = [
    {annotations = [],
     circuit = "circuit empty :\0A  module empty :\0A\0A    skip\0A",
     circuitPackage = "driving",
     class = "sifive.enterprise.grandcentral.SignalDriverAnnotation",
     id = 0 : i64,
     isSubCircuit = false
    }
  ]} {
  // Starting from DUT
  firrtl.hierpath private @nla_dut_rel [@DUT::@l, @Leaf]
  // Not through DUT, describes multiple paths
  firrtl.hierpath private @nla_segment [@Mid::@l, @Leaf]
  // Top to leaf, through the DUT
  firrtl.hierpath private @nla_top_thru_dut_to_w [@MainWithnewNLA::@dut, @DUT::@m, @Mid::@l, @Leaf]
  firrtl.module private @Leaf(
    in %in: !firrtl.uint<1> [{
      circt.nonlocal = @nla_segment,
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "source",
      id = 0 : i64,
      peer = "~Sub|Sub>in_source",
      side = "remote",
      targetId = 2 : i64
    }, {
      class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
      dir = "sink",
      id = 0 : i64,
      peer = "~Sub|Sub>in_sink",
      side = "remote",
      targetId = 3 : i64
    }],
    out %out: !firrtl.uint<1>) attributes {
      annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.module", id = 0 : i64}]
    } {
    %w = firrtl.wire {
      annotations = [{
        circt.nonlocal = @nla_top_thru_dut_to_w,
        class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
        dir = "source",
        id = 0 : i64,
        peer = "~Sub|Sub>w_source",
        side = "remote",
        targetId = 1 : i64
      }, {
        circt.nonlocal = @nla_dut_rel,
        class = "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
        dir = "sink",
        id = 0 : i64,
        peer = "~Sub|Sub>w_sink",
        side = "remote",
        targetId = 4 : i64}
       ]} : !firrtl.uint<1>
    // CHECK:  %w = firrtl.wire sym @w   : !firrtl.uint<1>
    firrtl.strictconnect %w, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %w : !firrtl.uint<1>
  }
  firrtl.module private @Mid(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %l_in, %l_out = firrtl.instance l sym @l  @Leaf(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %l_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %l_out : !firrtl.uint<1>
  }
  firrtl.module @MainWithnewNLA(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %dut_in, %dut_out = firrtl.instance dut sym @dut  @DUT(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %dut_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %dut_out : !firrtl.uint<1>
    %m_in, %m_out = firrtl.instance m  @Mid(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %m_in, %in : !firrtl.uint<1>
  }
  firrtl.module private @DUT(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %l_in, %l_out = firrtl.instance l sym @l  @Leaf(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %m_in, %m_out = firrtl.instance m sym @m  @Mid(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    firrtl.strictconnect %l_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %m_in, %in : !firrtl.uint<1>
    %0 = firrtl.or %l_out, %m_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %out, %0 : !firrtl.uint<1>
  }
  // CHECK:      sv.verbatim "[
  // CHECK-SAME:   {
  // CHECK-SAME:     \22sinkTargets\22: [
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{1}}>{{2}}\22
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>in_sink\22
  // CHECK-SAME:       },
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{0}}/{{4}}:{{1}}>{{3}}\22,
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>w_sink\22
  // CHECK-SAME:       }
  // CHECK-SAME:     ],
  // CHECK-SAME:     \22sourceTargets\22: [
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{5}}/{{6}}:{{1}}>{{2}}\22,
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>in_source\22
  // CHECK-SAME:       },
  // CHECK-SAME:       {
  // CHECK-SAME{LITERAL}: \22_1\22: \22~{{0}}|{{0}}/{{7}}:{{5}}/{{6}}:{{1}}>{{3}}\22,
  // CHECK-SAME{LITERAL}: \22_2\22: \22~Sub|Sub>w_source\22
  // CHECK-SAME:       }
  // CHECK-SAME:     ],
  // CHECK-SAME:   }
  // CHECK-SAME: ]"
  // CHECK-SAME: {
  // CHECK-SAME:   output_file =
  // CHECK-SAME:     #hw.output_file<"sigdrive.json", excludeFromFileList>
  // CHECK-SAME:   symbols = [
  // CHECK-SAME:     @DUT,
  // CHECK-SAME:     @Leaf,
  // CHECK-SAME:     #hw.innerNameRef<@Leaf::@in>,
  // CHECK-SAME:     #hw.innerNameRef<@Leaf::@w>,
  // CHECK-SAME:     #hw.innerNameRef<@DUT::@l>,
  // CHECK-SAME:     @Mid,
  // CHECK-SAME:     #hw.innerNameRef<@Mid::@l>,
  // CHECK-SAME:     #hw.innerNameRef<@DUT::@m>
  // CHECK-SAME:    ]
  // CHECK-SAME: }
}
