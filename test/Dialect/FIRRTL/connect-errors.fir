; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

circuit Foo:
  module Foo:
    output a: { flip a: UInt<1> }
    output b: { a: UInt<1> }

; expected-error @+1 {{cannot connect non-equivalent}}
    b <= a

;// -----

circuit PartialFoo:
  module PartialFoo:
    output a: { flip a: UInt<1> }
    output b: { a: UInt<1> }

; expected-error @+1 {{cannot partially connect non-weakly-equivalent}}
    b <- a

;// -----

circuit PartialConnectMismatch:
  module PartialConnectMismatch:
    output a: UInt
    output b: SInt

; expected-error @+1 {{cannot partially connect non-weakly-equivalent}}
    b <- a

;// -----

circuit Bar:
  module Bar:
    output a: { flip a: UInt<1> }
; expected-note @+1 {{destination}}
    input b: { flip a: UInt<1> }

; expected-error @+1 {{invalid flow}}
    b <= a
