; RUN: firtool %s --split-input-file -verify-diagnostics

circuit TapBetweenWhens : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~TapBetweenWhens|TapBetweenWhens>val",
        "sink":"~TapBetweenWhens|TapBetweenWhens>tap"
      }
    ]
  }
]]
  module TapBetweenWhens :
    input c1: UInt<1>
    input c2: UInt<1>
    output o: UInt

    when c1:
      ; expected-error @below {{This value is involved with a Wiring Problem where the destination}}
      wire val : UInt<3>
      val <= UInt(1)

    o is invalid

    when c2:
      ; expected-note @below {{The destination is here.}}
      wire tap: UInt<3>
      tap is invalid
      o <= tap

; // -----

circuit NoSafeInsertionPoint : %[[
  {
    "class":"sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys":[
      {
        "class":"sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source":"~NoSafeInsertionPoint|NoSafeInsertionPoint>val",
        "sink":"~NoSafeInsertionPoint|NoSafeInsertionPoint>tap"
      }
    ]
  }
]]
  module NoSafeInsertionPoint :
    input c: UInt<1>
    output o: UInt<1>

    when c:
      ; expected-error @below {{This value is involved with a Wiring Problem where the destination}}
      wire val : UInt<3>
      val <= UInt(1)

    ; expected-note @below {{The destination is here.}}
    wire tap: UInt<3>
    tap is invalid
    o <= tap
