; RUN: firtool --verilog %s | FileCheck %s

circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo/b:Bar>inv",
        "wireName": "~Top|Top>tap"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Bar>inv"
  }
]]
  module Bar :
    input in : UInt<1>
    output out : UInt<1>
    wire inv : UInt<1>
    inv <= not(in)
    out <= inv

  module Foo :
    input in : UInt<1>
    output out : UInt<1>

    inst b of Bar
    b.in <= in
    out <= b.out

  module Top:
    input in : UInt<1>
    output out : UInt<1>

    inst foo of Foo
    foo.in <= in
    out <= foo.out

    wire tap : UInt<1>
    tap is invalid

; CHECK: module Top(
; CHECK-NOT: module
; CHECK:   assign tap = Top.foo.b.inv;
; CHECK: endmodule

