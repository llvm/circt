; RUN: firtool %s --emit-metadata --verilog | FileCheck %s

circuit Top : %[[{
  "class": "sifive.enterprise.grandcentral.ModuleReplacementAnnotation",
  "targets": [
    "~Top|Child",
    "~Top|Top/child:Child",
    "~Top|Top/childWrapper:ChildWrapper/child:Child"
  ],
  "circuit": "",
  "annotations": ["foo", "bar"],
  "circuitPackage": "subcircuit",
  "dontTouches":[
    "~Top|Child>in",
    "~Top|Child>out"
  ]
}]]
  module ChildWrapper :
    input in : UInt<123>
    output out : UInt<456>

    inst child of Child
    child.in <= in
    out <= child.out

  extmodule Child :
    input in : UInt<123>
    output out : UInt<456>

  module Top :
    input in : UInt<123>
    output out : UInt<456>
    inst child of Child
    inst childWrapper of ChildWrapper
    child.in <= in
    childWrapper.in <= child.out
    out <= childWrapper.out

; CHECK-LABEL: FILE "./subcircuit/original_modules.F"
; CHECK: Child

; CHECK-LABEL: FILE "./subcircuit/runtime_config.txt"
; CHECK: Child
; CHECK-NEXT: Top.child
; CHECK-NEXT: Top.childWrapper.child

; CHECK-LABEL: FILE "./subcircuit/compile_config.txt"
; CHECK: replace module {Child} with module {subcircuit}
