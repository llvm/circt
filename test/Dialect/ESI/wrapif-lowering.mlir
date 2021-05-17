// RUN: circt-opt %s --lower-esi-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

sv.interface @IValidReady_i4 {
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.signal @data : i4
  sv.interface.modport @source  ("input" @ready, "output" @valid, "output" @data)
  sv.interface.modport @sink  ("input" @valid, "input" @data, "output" @ready)
}

hw.module @test(%clk:i1, %rstn:i1) {

  %0 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  %1 = sv.modport.get %0 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  %2 = esi.wrap.iface %1 : !sv.modport<@IValidReady_i4::@source> -> !esi.channel<i4>

  %5 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  %6 = sv.modport.get %5 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  esi.unwrap.iface %2 into %6 : (!esi.channel<i4>, !sv.modport<@IValidReady_i4::@sink>)

  // CHECK:         %0 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // CHECK-NEXT:    %1 = sv.modport.get %0 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // CHECK-NEXT:    %2 = sv.interface.signal.read %0(@IValidReady_i4::@valid) : i1
  // CHECK-NEXT:    %3 = sv.interface.signal.read %0(@IValidReady_i4::@data) : i4
  // CHECK-NEXT:    sv.interface.signal.assign %0(@IValidReady_i4::@ready) = %6 : i1
  // CHECK-NEXT:    %4 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // CHECK-NEXT:    %5 = sv.modport.get %4 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // CHECK-NEXT:    %6 = sv.interface.signal.read %4(@IValidReady_i4::@ready) : i1
  // CHECK-NEXT:    sv.interface.signal.assign %4(@IValidReady_i4::@valid) = %2 : i1
  // CHECK-NEXT:    sv.interface.signal.assign %4(@IValidReady_i4::@data) = %3 : i4
}
