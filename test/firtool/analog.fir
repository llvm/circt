; RUN: firtool %s --format=fir -verilog | FileCheck %s

FIRRTL version 7.0.0

circuit AnalogPortCase :
  extmodule AnalogDevice:
    input a : Analog<8>
    output b : Analog<4>

  public module AnalogPortCase :
    input a : Analog<8>
    output b : Analog<4>

    inst device of AnalogDevice
    attach (a, device.a)
    attach (b, device.b)

; CHECK-LABEL: module AnalogPortCase(
; CHECK:         inout [7:0] a,
; CHECK:         inout [3:0] b
; CHECK:       );
; CHECK: AnalogDevice device (
; CHECK:   .a (a),
; CHECK:   .b (b)
