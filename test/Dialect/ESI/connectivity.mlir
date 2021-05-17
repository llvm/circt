// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module @Sender() -> (%x: !esi.channel<i1>) {
  %0 = constant 0 : i1
  // Don't transmit any data.
  %ch, %rcvrRdy = esi.wrap.vr %0, %0 : i1
  hw.output %ch : !esi.channel<i1>
}
hw.module @Reciever(%a: !esi.channel<i1>) {
  %rdy = constant 1 : i1
  // Recieve bits.
  %data, %valid = esi.unwrap.vr %a, %rdy : i1
}
!FooStruct = type !esi.struct<FooStruct, a: si4, b: !hw.array<3 x ui4>>
hw.module @StructRcvr(%a: !esi.channel<!FooStruct>) {
  %rdy = constant 1 : i1
  // Recieve bits.
  %data, %valid = esi.unwrap.vr %a, %rdy : !FooStruct
}

// CHECK-LABEL: hw.module @Sender() -> (%x: !esi.channel<i1>) {
// CHECK:        %chanOutput, %ready = esi.wrap.vr %false, %false : i1
// CHECK-LABEL: hw.module @Reciever(%a: !esi.channel<i1>) {
// CHECK:        %rawOutput, %valid = esi.unwrap.vr %a, %true : i1
// CHECK-LABEL: hw.module @StructRcvr(%a: !esi.channel<!esi.struct<FooStruct, a: si4, b: !hw.array<3xui4>>>)

hw.module @test(%clk: i1, %rstn: i1) {
  %esiChan = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
  %bufferedChan = esi.buffer %clk, %rstn, %esiChan { } : i1
  hw.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

  // CHECK:  %sender.x = hw.instance "sender" @Sender()  : () -> !esi.channel<i1>
  // CHECK-NEXT:  %0 = esi.buffer %clk, %rstn, %sender.x {} : i1
  // CHECK-NEXT:  hw.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

  %esiChan2 = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
  %bufferedChan2 = esi.buffer %clk, %rstn, %esiChan2 { stages = 4 } : i1
  hw.instance "recv" @Reciever (%bufferedChan2) : (!esi.channel<i1>) -> ()

  // CHECK-NEXT:  %sender.x_0 = hw.instance "sender" @Sender()  : () -> !esi.channel<i1>
  // CHECK-NEXT:  %1 = esi.buffer %clk, %rstn, %sender.x_0 {stages = 4 : i64} : i1
  // CHECK-NEXT:  hw.instance "recv" @Reciever(%1)  : (!esi.channel<i1>) -> ()

  %nullBit = esi.null : !esi.channel<i1>
  hw.instance "nullRcvr" @Reciever(%nullBit) : (!esi.channel<i1>) -> ()
  // CHECK-NEXT:  [[NULLI1:%.+]] = esi.null : !esi.channel<i1>
  // CHECK-NEXT:  hw.instance "nullRcvr" @Reciever([[NULLI1]]) : (!esi.channel<i1>) -> ()
}

hw.module.extern @IFaceSender(!sv.modport<@IData::@Source>) -> ()
hw.module.extern @IFaceRcvr(!sv.modport<@IData::@Sink>) -> ()
sv.interface @IData {
  sv.interface.signal @data : i32
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @Source ("input" @data, "input" @valid, "output" @ready)
  sv.interface.modport @Sink ("output" @data, "output" @valid, "input" @ready)
}
hw.module @testIfaceWrap() {
  %ifaceOut = sv.interface.instance : !sv.interface<@IData>
  %ifaceOutSource = sv.modport.get %ifaceOut @Source : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
  hw.instance "ifaceSender" @IFaceSender (%ifaceOutSource) : (!sv.modport<@IData::@Source>) -> ()
  %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>

  // CHECK-LABEL:  hw.module @testIfaceWrap() {
  // CHECK-NEXT:     %0 = sv.interface.instance : !sv.interface<@IData>
  // CHECK-NEXT:     %1 = sv.modport.get %0 @Source : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
  // CHECK-NEXT:     hw.instance "ifaceSender" @IFaceSender(%1) : (!sv.modport<@IData::@Source>) -> ()
  // CHECK-NEXT:     %2 = sv.modport.get %0 @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // CHECK-NEXT:     %3 = esi.wrap.iface %2 : !sv.modport<@IData::@Sink> -> !esi.channel<i32>

  %ifaceIn = sv.interface.instance : !sv.interface<@IData>
  %ifaceInSink = sv.modport.get %ifaceIn @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  hw.instance "ifaceRcvr" @IFaceRcvr (%ifaceInSink) : (!sv.modport<@IData::@Sink>) -> ()
  %ifaceInSource = sv.modport.get %ifaceIn @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
  esi.unwrap.iface %idataChanOut into %ifaceInSource : (!esi.channel<i32>, !sv.modport<@IData::@Source>)

  // CHECK-NEXT:     %4 = sv.interface.instance : !sv.interface<@IData>
  // CHECK-NEXT:     %5 = sv.modport.get %4 @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // CHECK-NEXT:     hw.instance "ifaceRcvr" @IFaceRcvr(%5) : (!sv.modport<@IData::@Sink>) -> ()
  // CHECK-NEXT:     %6 = sv.modport.get %4 @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
  // CHECK-NEXT:     esi.unwrap.iface %3 into %6 : (!esi.channel<i32>, !sv.modport<@IData::@Source>)
}
