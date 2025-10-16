; RUN: firtool %s -output-final-mlir %t.mlir && FileCheck %s < %t.mlir

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
; CHECK-NEXT:    om.object @ClockDomain_out(%basepath, %A, %[[#association:]])
; CHECK-NEXT:    %[[#a:]] = om.frozenpath_create reference %basepath "Foo>a"
; CHECK-NEXT:    %[[#b:]] = om.frozenpath_create reference %basepath "Foo>b"
; CHECK-NEXT:    %[[#association]] = om.list_create %[[#a]], %[[#b]]
;
; CHECK:         om.object @PowerDomain_out(%basepath, %B, %[[#association:]])
; CHECK-NEXT:    %[[#a:]] = om.frozenpath_create reference %basepath "Foo>a"
; CHECK-NEXT:    %[[#b:]] = om.frozenpath_create reference %basepath "Foo>b"
; CHECK-NEXT:    %[[#association]] = om.list_create %[[#a]], %[[#b]]
;
; CHECK-LABEL: om.class @ClockDomain
; CHECK-LABEL: om.class @ClockDomain_out
; CHECK-LABEL: om.class @PowerDomain
; CHECK-LABEL: om.class @PowerDomain_out
