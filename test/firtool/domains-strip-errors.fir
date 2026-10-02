; RUN: firtool %s -domain-mode=check -strip-domain=true -o /dev/null --verify-diagnostics
; RUN: firtool %s -domain-mode=infer -strip-domain=true -o /dev/null --verify-diagnostics
; RUN: firtool %s -domain-mode=infer-all -strip-domain=true -o /dev/null --verify-diagnostics
; RUN: firtool %s -domain-mode=disable -strip-domain=true -o - | FileCheck %s

; Stripping domains must happen after checking, so an illegal crossing is still
; diagnosed. Disabling checking allows the same circuit to compile.

FIRRTL version 7.0.0
circuit Crossing:
  domain ClockDomain:

  public module Crossing:
    ; expected-note @below {{input module port A declared here}}
    input A: Domain of ClockDomain
    ; expected-note @below {{input module port B declared here}}
    input B: Domain of ClockDomain
    ; expected-note @below {{a has domains [A : ClockDomain]}}
    input a: UInt<1> domains [A]
    ; expected-note @below {{b has domains [B : ClockDomain]}}
    output b: UInt<1> domains [B]

    ; expected-error @below {{illegal domain crossing}}
    connect b, a

; CHECK-LABEL: module Crossing(
; CHECK: assign b = a;
