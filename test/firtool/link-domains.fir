; RUN: split-file %s %t
; RUN: firtool %t/parent.fir -output-final-mlir=%t/parent.mlir -domain-mode=infer -strip-domain=false -o %t/parent.sv
; RUN: firtool %t/child.fir -output-final-mlir=%t/child.mlir -domain-mode=infer -strip-domain=false -o %t/child.sv
; RUN: not om-linker %t/parent.mlir %t/child.mlir 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not om-linker %t/child.mlir %t/parent.mlir 2>&1 | FileCheck %s --check-prefix=ERROR

; ERROR: OM property assertion failed: flag must be set

;--- child.fir
FIRRTL version 7.0.0
circuit Child:
  domain D:
    flag: Bool
  public module Child:
    input d: Domain of D
    input rst: AsyncReset domains [d]
    propassert d.flag, String("flag must be set")

;--- parent.fir
FIRRTL version 7.0.0
circuit Parent:
  domain D:
    flag: Bool
  extmodule Child:
    input d: Domain of D
    input rst: AsyncReset domains [d]
  public module Parent:
    output d: Domain of D
    input rst: AsyncReset domains [d]
    domain dDomain of D(Bool(false))
    domain_define d = dDomain
    inst child of Child
    connect child.rst, rst
