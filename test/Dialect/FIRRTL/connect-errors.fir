; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

FIRRTL version 4.0.0
circuit Foo:
  public module Foo:
    output a: { flip a: UInt<1> }
    output b: { a: UInt<1> }

; expected-error @+1 {{cannot connect non-equivalent}}
    b <= a

;// -----

FIRRTL version 4.0.0
circuit Bar:
  public module Bar:
; expected-note @+1 {{destination}}
    output a: { flip a: UInt<1> }
    input b: { flip a: UInt<1> }

; expected-error @+1 {{invalid flow}}
    b <= a
