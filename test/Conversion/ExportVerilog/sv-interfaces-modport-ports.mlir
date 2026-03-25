// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

module {
  sv.interface @MyBundle {
    sv.interface.signal @data : i32
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.modport @sink (input @data, input @valid, output @ready)
    sv.interface.modport @source (output @data, output @valid, input @ready)
  }

  // CHECK-LABEL: interface MyBundle;
  // CHECK:         logic [31:0] data;
  // CHECK:         logic valid;
  // CHECK:         logic ready;
  // CHECK:         modport sink(
  // CHECK:         modport source(
  // CHECK:       endinterface
  // CHECK-EMPTY:
  // CHECK-LABEL: module ModportPorts(
  // CHECK-NEXT:  MyBundle.sink       p
  // CHECK-NEXT:       );
  hw.module @ModportPorts(in %p : !sv.modport<@MyBundle::@sink>) {}

  // CHECK-LABEL: module MultipleModportPorts(
  // CHECK:  MyBundle.sink         sink_port,
  // CHECK:  MyBundle.source       source_port
  // CHECK:       );
  hw.module @MultipleModportPorts(
    in %sink_port : !sv.modport<@MyBundle::@sink>,
    in %source_port : !sv.modport<@MyBundle::@source>
  ) {}

  // CHECK-LABEL: module MixedPorts(
  // CHECK:  input  [31:0]        data_in,
  // CHECK:  MyBundle.sink        intf_port,
  // CHECK:  output [31:0]        data_out
  // CHECK:       );
  hw.module @MixedPorts(
    in %data_in : i32,
    in %intf_port : !sv.modport<@MyBundle::@sink>,
    out data_out : i32
  ) {
    hw.output %data_in : i32
  }
}
