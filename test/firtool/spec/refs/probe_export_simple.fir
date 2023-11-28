; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
circuit MyModule :
  ; SPEC EXAMPLE BEGIN
  module MyModule :
    input in: UInt<5> ; XXX: Added width.
    output r : Probe<UInt>

    define r = probe(in)
    ; SPEC EXAMPLE END
