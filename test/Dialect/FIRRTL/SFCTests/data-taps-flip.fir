; RUN: firtool --parse-only %s -verify-each=true | FileCheck %s
; RUN: firtool --verilog %s
; RUN: firtool --parse-only %s --lower-annotations-no-ref-type-ports | FileCheck %s --check-prefixes NOREF --implicit-check-not firrtl.ref
; RUN: firtool --verilog %s --lower-annotations-no-ref-type-ports

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
; NOREF-DAG: firrtl.strictconnect %[[A]],
; NOREF-DAG: %[[B:.+]] = firrtl.subfield %[[OUT_PORT]][b]
; NOREF-DAG: firrtl.strictconnect %[[B]],
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

  module Top :
    input x: {a : UInt<2>, flip b: UInt<2>}
    output tap : {a : UInt<2>, b: UInt<2>}

    inst c of Child
    c.x <= x

    wire sink : {a : UInt<2>, b: UInt<2>}
    tap <= sink


