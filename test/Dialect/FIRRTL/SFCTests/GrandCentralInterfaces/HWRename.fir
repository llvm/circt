; RUN: firtool --firrtl-grand-central --verilog --annotation-file %S/HWRename.anno.json %s | FileCheck %s --check-prefixes CHECK
; XFAIL: true

circuit Top:
  module Companion :
    output io : { }

    wire _WIRE : UInt<1>
    _WIRE <= UInt<1>("h0")

  module DUT:
    input a: UInt<1>
    output b: UInt<1>

    wire signed: UInt<1>

    signed <= a
    b <= signed

    inst companion of Companion

  module Top:
    input a: UInt<1>
    output b: UInt<1>

    inst signed of DUT
    signed.a <= a
    b <= signed.b

    ; CHECK:      module DUT
    ; CHECK:        wire [[wireName:.+]];
    ; CHECK:      endmodule

    ; CHECK:      module Top
    ; CHECK:        DUT [[dutName:.+]] (
    ; CHECK:      endmodule

    ; CHECK:      module MyView_mapping();
    ; CHECK-NEXT:   assign MyView.[[elementName:.+]] = [[dutName]].[[wireName]]
    ; CHECK-NEXT: endmodule

    ; CHECK: interface MyInterface
    ; CHECK-NEXT:   logic [[elementName]]
    ; CHECK-NEXT: endinterface
