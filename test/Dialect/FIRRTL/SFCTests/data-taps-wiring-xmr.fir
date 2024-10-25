; RUN: firtool %s --split-input-file | FileCheck %s

FIRRTL version 4.0.0
circuit SinkThroughWhens: %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~SinkThroughWhens|SinkThroughWhens>val",
        "sink":"~SinkThroughWhens|SinkThroughWhens/mid:Middle/leaf:Leaf>tap"
      }
    ]
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
    "target":"~SinkThroughWhens|Leaf>tap"
  }
]]

; CHECK-LABEL: module Leaf
  module Leaf :
    input c: UInt<1>

    when c:
      wire tap : UInt<4>
      invalidate tap
      ; CHECK-NEXT: tap = 4'h3;
      ; CHECK-NEXT: endmodule

  module Middle :
    input c: UInt<1>

    when c:
      inst leaf of Leaf
      connect leaf.c, c

  public module SinkThroughWhens :
    input c: UInt<1>

    wire val : UInt<3>
    connect val, UInt(3)

    inst mid of Middle
    connect mid.c, c

; // -----

FIRRTL version 4.0.0
circuit TapWhenVal : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TapWhenVal|TapWhenVal>val",
        "sink":"~TapWhenVal|TapWhenVal>tap"
      }
    ]
  }
]]
; CHECK-LABEL: module TapWhenVal
  public module TapWhenVal :
    input c: UInt<1>
    output o: UInt<1>

    wire tap: UInt<3>
    invalidate tap

    when c:
      wire val : UInt<3>
      connect val, UInt(1)

    connect o, tap
; CHECK: assign o = 1'h1;

; // -----

FIRRTL version 4.0.0
circuit TapIntoWhen : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TapIntoWhen|TapIntoWhen>val",
        "sink":"~TapIntoWhen|TapIntoWhen>tap"
      }
    ]
  }
]]
; CHECK-LABEL: module TapIntoWhen
  public module TapIntoWhen :
    input c: UInt<1>
    output o: UInt<1>

    wire val : UInt<3>
    connect val, UInt(3)

    invalidate o

    when c:
      wire tap: UInt
      invalidate tap
      connect o, tap
; CHECK: assign  o = 1'h1;
