; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 4.0.0
circuit MyModule :
  ; SPEC EXAMPLE BEGIN
  public module MyModule :
    input in: UInt<5> ; XXX: Added width.
    output r : Probe<UInt<5>>

    define r = probe(in)
    ; SPEC EXAMPLE END
