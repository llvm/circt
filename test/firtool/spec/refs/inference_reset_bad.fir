; RUN: firtool %s -verify-diagnostics
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false -verify-diagnostics
; XFAIL: *
; https://github.com/llvm/circt/issues/4813
FIRRTL version 3.0.0
circuit ResetInferBad :
  ; expected-error @+3 {{reset inference failed}}
  ; SPEC EXAMPLE BEGIN
  module ResetInferBad :
    input in : Reset
    output out : AsyncReset
    connect out, read(probe(in))
  ; SPEC EXAMPLE END
