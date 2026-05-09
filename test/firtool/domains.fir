; RUN: firtool %s -output-final-mlir - -o /dev/null                    | FileCheck %s -implicit-check-not om.frozenpath_create -implicit-check-not Domain
; RUN: firtool %s -output-final-mlir - -domain-mode=check -o /dev/null | FileCheck %s -check-prefixes=CHECK,DOMAIN
; RUN: firtool %s -output-final-mlir - -domain-mode=strip -o /dev/null | FileCheck %s -implicit-check-not om.frozenpath_create -implicit-check-not Domain

FIRRTL version 6.0.0
circuit Foo :
  domain ClockDomain :
    name : String
    frequency : Integer

  domain PowerDomain :
    name : String
    voltage : Integer
    always_on : Bool

  public module Foo :
    input A : Domain of ClockDomain
    input B : Domain of PowerDomain
    input a : UInt<1> domains [A, B]
    input b : UInt<1> domains [A, B]

; CHECK-LABEL: om.class @Foo_Class
; DOMAIN-NEXT:   om.object @ClockDomain_out(%basepath, %A, %[[#association:]])
; DOMAIN-NEXT:   %[[#a:]] = om.frozenpath_create reference %basepath "Foo>a"
; DOMAIN-NEXT:   %[[#b:]] = om.frozenpath_create reference %basepath "Foo>b"
; DOMAIN-NEXT:   %[[#association]] = om.list_create %[[#a]], %[[#b]]
;
; DOMAIN:        om.object @PowerDomain_out(%basepath, %B, %[[#association:]])
; DOMAIN-NEXT:   %[[#a:]] = om.frozenpath_create reference %basepath "Foo>a"
; DOMAIN-NEXT:   %[[#b:]] = om.frozenpath_create reference %basepath "Foo>b"
; DOMAIN-NEXT:   %[[#association]] = om.list_create %[[#a]], %[[#b]]
;
; DOMAIN-LABEL: om.class @ClockDomain
; DOMAIN-LABEL: om.class @ClockDomain_out
; DOMAIN-LABEL: om.class @PowerDomain
; DOMAIN-LABEL: om.class @PowerDomain_out
