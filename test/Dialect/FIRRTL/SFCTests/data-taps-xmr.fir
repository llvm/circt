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
        "module":"~Top|BlackBox",
        "internalPath":"random.something_else",
        "sink": "~Top|Top>tap3"
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
    "target": "~Top|Top>tap3"
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

    skip

  extmodule BlackBox :

    defname = BlackBox

  module Child :

    inst localparam of BlackBox
    localparam is invalid

    wire tap : UInt<1>
    tap is invalid

  module ChildWrapper :

    inst signed of Child

  module Bar :

    wire inv : UInt<1>
    inv is invalid

  module Foo :

    inst b of Bar

  module Top:

    inst foo of Foo

    wire tap : UInt<1>
    tap is invalid

    wire tap2 : UInt<1>
    tap2 is invalid

    wire tap3 : UInt<1>
    tap3 is invalid

    inst unsigned of ChildWrapper

    inst int of InternalPathChild

; CHECK-LABEL: module Child(
; CHECK-NOT:   endmodule
; CHECK:         input [[Child_boredPort:[a-zA-Z0-9_]+]]
; CHECK:         wire tap = [[Child_boredPort]];
; CHECK;       endmodule

; CHECK-LABEL: module ChildWrapper(
; CHECK-NOT:   endmodule
; CHECK:         input [[ChildWrapper_boredPort:[a-zA-Z0-9_]+]]
; CHECK:         Child signed_0 (
; CHECK-NEXT:      .[[Child_boredPort]] ([[ChildWrapper_boredPort]])
; CHECK:       endmodule

; CHECK-LABEL: module Top(
; CHECK-NOT:   endmodule
; CHECK:         assign tap = Top.foo.b.inv;
; CHECK-NEXT:    assign tap2 = Top.unsigned_0.signed_0.localparam_0.random.something;
; CHECK-NEXT:    assign tap3 = Top.unsigned_0.signed_0.localparam_0.random.something_else;
; CHECK:         ChildWrapper unsigned_0 (
; CHECK-NEXT:      .[[ChildWrapper_boredPort]] (Top.int_0.io_out)
; CHECK:       endmodule
