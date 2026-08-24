; RUN: firtool %s -output-final-mlir - -o /dev/null                    | FileCheck %s -implicit-check-not om.frozenpath_create -implicit-check-not Domain
; RUN: firtool %s -output-final-mlir - -domain-mode=check -o /dev/null | FileCheck %s -check-prefixes=CHECK,DOMAIN
; RUN: firtool %s -output-final-mlir - -domain-mode=strip -o /dev/null | FileCheck %s -implicit-check-not om.frozenpath_create -implicit-check-not Domain

FIRRTL version 7.0.0
circuit Foo :
  domain ClockDomain :
    name : String
    frequency : Integer
    clockGates : Registry<Path>

  domain PowerDomain :
    name : String
    voltage : Integer
    always_on : Bool

  extmodule ClockGate:

  public module Foo :
    input A : Domain of ClockDomain
    input B : Domain of PowerDomain
    input a : UInt<1> domains [A, B]
    input b : UInt<1> domains [A, B]

    inst clockGate of ClockGate
    register A[clockGates], ref_instance(clockGate)

; CHECK-LABEL: om.class @Foo_Class
; DOMAIN:        om.elaborated_object @ClockDomain_out(%A,
; DOMAIN:        om.elaborated_object @PowerDomain_out(%B,
; DOMAIN-DAG:    om.frozenpath_create reference %basepath "Foo>a"
; DOMAIN-DAG:    om.frozenpath_create reference %basepath "Foo>b"
; DOMAIN-DAG:    om.frozenpath_create instance %basepath "Foo/clockGate:ClockGate"
;
; DOMAIN-LABEL: om.class @ClockDomain
; DOMAIN-LABEL: om.class @ClockDomain_out
; DOMAIN:       %clockGates_registry_in: !om.list<!om.frozenpath>
; DOMAIN-LABEL: om.class @PowerDomain
; DOMAIN-LABEL: om.class @PowerDomain_out
