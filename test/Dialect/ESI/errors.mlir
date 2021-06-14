// RUN: circt-opt %s -split-input-file -verify-diagnostics

sv.interface @IData {
  sv.interface.signal @data : i32
  sv.interface.signal @valid : i1
  sv.interface.signal @stall : i1
  sv.interface.modport @Sink ("output" @data, "output" @valid, "input" @ready)
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
  sv.interface.modport @Sink ("output" @data, "output" @valid, "input" @ready)
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
  sv.interface.modport @Sink ("output" @data, "output" @valid, "input" @ready)
}

hw.module @test(%m : !sv.modport<@IData::@Noexist>) {
  // expected-error @+1 {{Could not find modport @IData::@Noexist in symbol table.}}
  %idataChanOut = esi.wrap.iface %m: !sv.modport<@IData::@Noexist> -> !esi.channel<i32>
}
