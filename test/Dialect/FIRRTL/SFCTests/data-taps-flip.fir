; RUN: firtool --parse-only --split-input-file %s | FileCheck %s
; RUN: firtool --verilog --split-input-file %s
; RUN: firtool --parse-only --lower-annotations-no-ref-type-ports --split-input-file %s | FileCheck %s --check-prefixes NOREF --implicit-check-not firrtl.ref
; RUN: firtool --verilog --lower-annotations-no-ref-type-ports --split-input-file %s

; Test tapping a bundle with flips, tap sink is passive.

; CHECK-LABEL: module private @Child(
; CHECK: firrtl.ref.send
; CHECK-SAME: !firrtl.bundle<a: uint<2>, b flip: uint<2>>
; CHECK-LABEL: module @Top(
; CHECK: firrtl.ref.resolve
; CHECK-SAME: !firrtl.probe<bundle<a: uint<2>, b: uint<2>>>

; NOREF-LABEL: module private @Child(
; NOREF-SAME: out %[[OUT_PORT:.+]]: !firrtl.bundle<a: uint<2>, b: uint<2>>
; NOREF-DAG: %[[A:.+]] = firrtl.subfield %[[OUT_PORT]][a]
; NOREF-DAG: firrtl.matchingconnect %[[A]],
; NOREF-DAG: %[[B:.+]] = firrtl.subfield %[[OUT_PORT]][b]
; NOREF-DAG: firrtl.matchingconnect %[[B]],
FIRRTL version 4.0.0
circuit Top : %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Top|Top/c:Child>x",
        "sink": "~Top|Top>sink"
      }
    ]
  }
]]
  module Child :
    input x: {a : UInt<2>, flip b: UInt<2>}
    x.b <= UInt<2>(2)

  public module Top :
    input x: {a : UInt<2>, flip b: UInt<2>}
    output tap : {a : UInt<2>, b: UInt<2>}

    inst c of Child
    c.x <= x

    wire sink : {a : UInt<2>, b: UInt<2>}
    tap <= sink

; // -----

; Check same but where no ports are inserted.

; CHECK-LABEL: circuit "Local"
; NOREF-LABEL: circuit "Local"
FIRRTL version 4.0.0
circuit Local: %[[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Local|Local>x",
        "sink": "~Local|Local>sink"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~Local|Local>x",
        "sink": "~Local|Local>unused_sink"
      }
    ]
  }
]]
  public module Local:
    input x: {a : UInt<2>, flip b: UInt<2>}
    output tap : {a : UInt<2>, b: UInt<2>}

    x.b <= x.a

    wire sink : {a : UInt<2>, b: UInt<2>}
    tap <= sink

    wire unused_sink : {a : UInt<2>, b: UInt<2>}
