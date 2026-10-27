; RUN: split-file %s %t
; RUN: firtool %t/parent.fir -output-final-mlir=%t/parent.mlir -domain-mode=infer -strip-domain=false -o %t/parent.sv
; RUN: firtool %t/child.fir -output-final-mlir=%t/child.mlir -domain-mode=infer -strip-domain=false -o %t/child.sv
; RUN: not om-linker %t/parent.mlir %t/child.mlir 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not om-linker %t/child.mlir %t/parent.mlir 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: om-linker %t/parent.mlir %t/child.mlir --disable-elaboration | FileCheck %s

; ERROR: OM property assertion failed: reset must be cold

; CHECK-DAG: om.class @Parent_Class(%{{.*}}: !om.frozenbasepath) -> (d_out: !om.class.type<@ResetDomain_out>)
; CHECK-DAG: om.class @Child_Class(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.class.type<@ResetDomain>) -> (d_out: !om.class.type<@ResetDomain_out>)
; CHECK-DAG: om.class @ResetDomain(%{{.*}}: !om.frozenbasepath, %{{.*}}: i1) -> (isColdReset_out: i1)
; CHECK-DAG: om.class @ResetDomain_out(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.class.type<@ResetDomain>, %{{.*}}: !om.list<!om.frozenpath>) -> (domainInfo_out: !om.class.type<@ResetDomain>, associations_out: !om.list<!om.frozenpath>)
; CHECK-DAG: %[[COLD:.+]] = om.object.field %{{.+}}["isColdReset_out"] : (!om.class.type<@ResetDomain>) -> i1
; CHECK-DAG: om.property_assert %[[COLD]], %{{.+}} : i1

;--- child.fir
FIRRTL version 7.0.0
circuit Child:
  domain ResetDomain:
    isColdReset: Bool
  public module Child:
    input d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    output out: UInt<1> domains [d]
    propassert d.isColdReset, String("reset must be cold")
    connect out, UInt<1>(0)

;--- parent.fir
FIRRTL version 7.0.0
circuit Parent:
  domain ResetDomain:
    isColdReset: Bool
  extmodule Child:
    input d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    output out: UInt<1> domains [d]
  public module Parent:
    output d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    domain resetDomain of ResetDomain(Bool(false))
    domain_define d = resetDomain
    inst child of Child
    connect child.rst, rst
