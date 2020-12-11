// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @Sender() -> (%x: !esi.channel<i1>) {
    %0 = constant 0 : i1
    // Don't transmit any data.
    %ch, %rcvrRdy = esi.wrap.vr %0, %0 : i1
    rtl.output %ch : !esi.channel<i1>
  }
  rtl.module @Reciever(%a: !esi.channel<i1>) {
    %rdy = constant 1 : i1
    // Recieve bits.
    %data, %valid = esi.unwrap.vr %a, %rdy : i1
  }

  // CHECK-LABEL: rtl.module @Sender() -> (%x: !esi.channel<i1>) {
  // CHECK:        %output, %ready = esi.wrap.vr %false, %false : i1
  // CHECK-LABEL: rtl.module @Reciever(%a: !esi.channel<i1>) {
  // CHECK:        %output, %valid = esi.unwrap.vr %a, %true : i1

  rtl.module @test(%clk: i1, %rstn: i1) {
    %esiChan = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan = esi.buffer %clk, %rstn, %esiChan { } : i1
    rtl.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // CHECK:  %sender.x = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %0 = esi.buffer %clk, %rstn, %sender.x {} : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%0)  : (!esi.channel<i1>) -> ()

    %esiChan2 = rtl.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    %bufferedChan2 = esi.buffer %clk, %rstn, %esiChan2 { stages = 4 } : i1
    rtl.instance "recv" @Reciever (%bufferedChan2) : (!esi.channel<i1>) -> ()

    // CHECK-NEXT:  %sender.x_0 = rtl.instance "sender" @Sender()  : () -> !esi.channel<i1>
    // CHECK-NEXT:  %1 = esi.buffer %clk, %rstn, %sender.x_0 {stages = 4 : i64} : i1
    // CHECK-NEXT:  rtl.instance "recv" @Reciever(%1)  : (!esi.channel<i1>) -> ()
  }

  rtl.externmodule @IFaceSender(!sv.modport<@IData::@Source>) -> ()
  rtl.externmodule @IFaceRcvr(!sv.modport<@IData::@Sink>) -> ()
  sv.interface @IData {
    sv.interface.signal @data : i32
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.modport @Source ("input" @data, "input" @valid, "output" @ready)
    sv.interface.modport @Sink ("output" @data, "output" @valid, "input" @ready)
  }
  rtl.module @testIfaceWrap() {
    %ifaceOut = sv.interface.instance : !sv.interface<@IData>
    %ifaceOutSource = sv.modport.get %ifaceOut @Source : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
    rtl.instance "ifaceSender" @IFaceSender (%ifaceOutSource) : (!sv.modport<@IData::@Source>) -> ()
    %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
    %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>

    // CHECK-LABEL:  rtl.module @testIfaceWrap() {
    // CHECK-NEXT:     %0 = sv.interface.instance : !sv.interface<@IData>
    // CHECK-NEXT:     %1 = sv.modport.get %0 @Source : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
    // CHECK-NEXT:     rtl.instance "ifaceSender" @IFaceSender(%1) : (!sv.modport<@IData::@Source>) -> ()
    // CHECK-NEXT:     %2 = sv.modport.get %0 @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
    // CHECK-NEXT:     %3 = esi.wrap.iface %2 : !sv.modport<@IData::@Sink> -> !esi.channel<i32>

    %ifaceIn = sv.interface.instance : !sv.interface<@IData>
    %ifaceInSink = sv.modport.get %ifaceIn @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
    rtl.instance "ifaceRcvr" @IFaceRcvr (%ifaceInSink) : (!sv.modport<@IData::@Sink>) -> ()
    %ifaceInSource = sv.modport.get %ifaceIn @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
    esi.unwrap.iface %idataChanOut into %ifaceInSource : (!esi.channel<i32>, !sv.modport<@IData::@Source>)

    // CHECK-NEXT:     %4 = sv.interface.instance : !sv.interface<@IData>
    // CHECK-NEXT:     %5 = sv.modport.get %4 @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
    // CHECK-NEXT:     rtl.instance "ifaceRcvr" @IFaceRcvr(%5) : (!sv.modport<@IData::@Sink>) -> ()
    // CHECK-NEXT:     %6 = sv.modport.get %4 @Sink : !sv.interface<@IData> -> !sv.modport<@IData::@Source>
    // CHECK-NEXT:     esi.unwrap.iface %3 into %6 : (!esi.channel<i32>, !sv.modport<@IData::@Source>)
  }
}
