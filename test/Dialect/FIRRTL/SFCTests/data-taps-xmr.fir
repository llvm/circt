; RUN: firtool --verilog %s | FileCheck %s

circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/foo:Foo/b:Bar>inv",
        "sink": "~Top|Top>tap"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|BlackBox",
        "internalPath":"random.something",
        "sink": "~Top|Top>tap2"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~Top|InternalPathChild",
        "internalPath":"io_out",
        "sink": "~Top|Top/unsigned:ChildWrapper/signed:Child>tap"
      }
    ]
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Top>tap2"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Child>tap"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Top|Bar>inv"
  }
]]
  module InternalPathChild :
    output io : { flip in : UInt<8>, out : UInt<8>}

    node sum = tail(add(io.in, UInt<1>(1)), 1)

    io.out <= sum

  extmodule BlackBox :
    input in : UInt<1>
    output out : UInt<1>
    defname = BlackBox

  module Child :
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst localparam of BlackBox
    localparam.out is invalid
    localparam.in is invalid
    localparam.in <= io.in
    io.out <= localparam.out
    wire tap : UInt<1>
    tap is invalid

  module ChildWrapper :
    output io : { flip in : UInt<1>, out : UInt<1>}

    inst signed of Child
    signed.io.in <= io.in
    io.out <= signed.io.out

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
    output io : { flip in : UInt<1>, out : UInt<1>}
    output io2 : { flip in : UInt<8>, out : UInt<8>}

    inst foo of Foo
    foo.in <= in
    out <= foo.out

    wire tap : UInt<1>
    tap is invalid

    wire tap2 : UInt<1>
    tap2 is invalid

    inst unsigned of ChildWrapper
    node _child_io_in_T = and(io.in, in)
    unsigned.io.in <= _child_io_in_T
    node _io_out_T = and(unsigned.io.out, out)
    io.out <= _io_out_T

    inst int of InternalPathChild
    io2 <= int.io

; CHECK-LABEL: module Child(
; CHECK:       wire tap
; CHECK-NEXT:  assign tap = Top.int_0.io_out;
; CHECK:       BlackBox localparam_0 (

; CHECK: module Top(
; CHECK-NOT: module
; CHECK:   tap = Top.foo.b.inv;
; CHECK:   assign tap2 = Top.unsigned_0.signed_0.localparam_0.random.something;
; CHECK:   InternalPathChild int_0 (
; CHECK: endmodule
