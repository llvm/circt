; RUN: firtool %s --verify-diagnostics --split-input-file
; The Chisel's Definition API generates modules that are not instantiated,
; whose reset cannot be inferred properly. These modules should be removed
; before checking abstract type resets.

FIRRTL version 3.3.0
circuit Foo:
  module Adder :
    input clock : Clock
    input reset : Reset
    input in : UInt<10>
    output out : UInt<10>

    node _out_T = add(in, UInt<1>(0h1))
    node _out_T_1 = tail(_out_T, 1)
    connect out, _out_T_1

  ; expected-warning @below {{module `Foo` is empty}}
  module Foo :
    input clock : Clock
    input reset : UInt<1>

;// -----

FIRRTL version 3.3.0
circuit Bar:
    ; expected-note @below {{the module with this uninferred reset port was defined here}}
  module Bar :
    input clock : Clock
    ; expected-error @below {{a port "reset" with abstract reset type was unable to be inferred by InferResets}}
    input reset : Reset
