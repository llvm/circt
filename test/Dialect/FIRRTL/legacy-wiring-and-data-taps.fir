; RUN: firtool --verilog %s | FileCheck %s
; Regression test for Legacy Wiring + Data Taps, see:
; https://github.com/llvm/circt/pull/4496#discussion_r1065935380
; Ensure Legacy Wiring doesn't trip over the Data Taps issue:
; https://github.com/llvm/circt/pull/4390

; Check doesn't crash, primarily.
; While testing, check the constant reaches the target port.

; CHECK:      BlackBox localparam{{[^ ]*}} (
; CHECK-NEXT:   .sink (1'h1)
; CHECK-NEXT: )

circuit Top : %[[
  {
    "class": "firrtl.passes.wiring.SinkAnnotation",
    "target": "Top.BlackBox.sink",
    "pin": "test_pin"
  },
  {
    "class": "firrtl.passes.wiring.SourceAnnotation",
    "target": "Top.Top.legacy_tap",
    "pin": "test_pin"
  },
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
    input sink : UInt<1>

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

    wire legacy_tap : UInt<1>
    legacy_tap <= UInt<1>(1)

    inst unsigned of ChildWrapper

    inst int of InternalPathChild

