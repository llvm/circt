// RUN: circt-opt %s --lower-esi-to-physical -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-ports -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=IFACE %s
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-rtl -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=RTL %s

rtl.externmodule @Sender(%clk: i1) -> ( %x: !esi.channel<i4>, %y: i8 )
rtl.externmodule @ArrSender() -> (%x: !esi.channel<!rtl.array<4xi64>>)
rtl.externmodule @Reciever(%a: !esi.channel<i4>, %clk: i1)

// CHECK-LABEL: rtl.externmodule @Sender(i1 {rtl.name = "clk"}) -> (%x: !esi.channel<i4>, %y: i8)
// CHECK-LABEL: rtl.externmodule @Reciever(!esi.channel<i4> {rtl.name = "a"}, i1 {rtl.name = "clk"})

// IFACE-LABEL: sv.interface @IValidReady_i4 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : i4
// IFACE-NEXT:    sv.interface.modport @sink ("input" @ready, "output" @valid, "output" @data)
// IFACE-NEXT:    sv.interface.modport @source ("input" @valid, "input" @data, "output" @ready)
// IFACE-LABEL: sv.interface @IValidReady_ArrayOf4xi64 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : !rtl.array<4xi64>
// IFACE-NEXT:    sv.interface.modport @sink  ("input" @ready, "output" @valid, "output" @data)
// IFACE-NEXT:    sv.interface.modport @source  ("input" @valid, "input" @data, "output" @ready)
// IFACE-LABEL: rtl.externmodule @Sender(i1 {rtl.name = "clk"}, !sv.modport<@sink> {rtl.name = "x"}) -> (%y: i8)
// IFACE-LABEL: rtl.externmodule @Reciever(!sv.modport<@source> {rtl.name = "a"}, i1 {rtl.name = "clk"})


rtl.module @test(%clk:i1, %rstn:i1) {

  %esiChan2, %0 = rtl.instance "sender2" @Sender (%clk) : (i1) -> (!esi.channel<i4>, i8)
  %bufferedChan2 = esi.buffer %clk, %rstn, %esiChan2 { stages = 4 } : i4
  rtl.instance "recv2" @Reciever (%bufferedChan2, %clk) : (!esi.channel<i4>, i1) -> ()

  // CHECK:      %sender2.x, %sender2.y = rtl.instance "sender2" @Sender(%clk) : (i1) -> (!esi.channel<i4>, i8)
  // CHECK-NEXT:  %0 = esi.stage %clk, %rstn, %sender2.x : i4
  // CHECK-NEXT:  %1 = esi.stage %clk, %rstn, %0 : i4
  // CHECK-NEXT:  %2 = esi.stage %clk, %rstn, %1 : i4
  // CHECK-NEXT:  %3 = esi.stage %clk, %rstn, %2 : i4
  // CHECK-NEXT:  rtl.instance "recv2" @Reciever(%3, %clk)  : (!esi.channel<i4>, i1) -> ()

  // IFACE-LABEL: rtl.module @test(%clk: i1, %rstn: i1) {
  // IFACE-NEXT:    %0 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %1 = sv.modport.get %0 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    %2 = esi.wrap.iface %1 : !sv.modport<@IValidReady_i4::@source> -> !esi.channel<i4>
  // IFACE-NEXT:    %3 = sv.modport.get %0 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    %sender2.y = rtl.instance "sender2" @Sender(%clk, %3) : (i1, !sv.modport<@IValidReady_i4::@sink>) -> i8
  // IFACE-NEXT:    %4 = esi.buffer %clk, %rstn, %2 {stages = 4 : i64} : i4
  // IFACE-NEXT:    %5 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %6 = sv.modport.get %5 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    esi.unwrap.iface %4 into %6 : (!esi.channel<i4>, !sv.modport<@IValidReady_i4::@sink>)
  // IFACE-NEXT:    %7 = sv.modport.get %5 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    rtl.instance "recv2" @Reciever(%7, %clk) : (!sv.modport<@IValidReady_i4::@source>, i1) -> ()

  // After all 3 ESI lowering passes, there shouldn't be any ESI constructs!
  // RTL-NOT: esi
}
