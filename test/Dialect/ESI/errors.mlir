// RUN: circt-opt %s -split-input-file -verify-diagnostics

sv.interface @IData {
  sv.interface.signal @data : i32
  sv.interface.signal @valid : i1
  sv.interface.signal @stall : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test() {
  %ifaceOut = sv.interface.instance : !sv.interface<@IData>
  %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // expected-error @+1 {{Interface is not a valid ESI interface.}}
  %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>
}

// -----

sv.interface @IData {
  sv.interface.signal @data : i2
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test() {
  %ifaceOut = sv.interface.instance : !sv.interface<@IData>
  %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // expected-error @+1 {{Operation specifies '!esi.channel<i32>' but type inside doesn't match interface data type 'i2'.}}
  %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>
}

// -----

sv.interface @IData {
  sv.interface.signal @data : i2
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test(%m : !sv.modport<@IData::@Noexist>) {
  // expected-error @+1 {{Could not find modport @IData::@Noexist in symbol table.}}
  %idataChanOut = esi.wrap.iface %m: !sv.modport<@IData::@Noexist> -> !esi.channel<i32>
}

// -----

esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<i16>
  esi.service.to_client @Recv : !esi.channel<i32>
}

hw.module @Loopback (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
  // expected-error @+1 {{'esi.service.req.to_server' op Request type does not match port type '!esi.channel<i16>'}}
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i32>
}

// -----

esi.service.decl @HostComms {
}

hw.module @Loopback (%clk: i1) -> () {
  // expected-error @+1 {{'esi.service.req.to_client' op Cannot find port named "Recv"}}
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
}
// -----

esi.service.decl @HostComms {
  esi.service.to_client @Recv : !esi.channel<i8>
}

hw.module @Loopback (%clk: i1) -> () {
  // expected-error @+1 {{'esi.service.req.to_client' op Request type does not match port type '!esi.channel<i8>'}}
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
}

// -----

hw.module @Loopback (%clk: i1) -> () {
  // expected-error @+1 {{Cannot find module "HostComms"}}
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
}
