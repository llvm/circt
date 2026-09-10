; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0
circuit ResetInferGood :
  ; SPEC EXAMPLE BEGIN
  module ResetInferGood :
    input in : Reset
    output out : Reset
    output out2 : AsyncReset
    connect out, read(probe(in))
    connect out2, in
  ; SPEC EXAMPLE END
