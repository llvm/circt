; RUN: firtool --parse-only %s -verify-each=true | FileCheck %s
; RUN: firtool --verilog %s

; Test tapping a bundle with flips, tap sink is passive.

; CHECK-DAG: firrtl.ref.send
; CHECK-DAG-SAME: !firrtl.ref<bundle<a: uint<2>, b flip: uint<2>>>
; CHECK-DAG: firrtl.ref.resolve
; CHECK-DAG-SAME: !firrtl.ref<bundle<a: uint<2>, b: uint<2>>>
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


