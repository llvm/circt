; RUN: firtool %s -parse-only | circt-opt -firrtl-infer-resets | FileCheck %s --check-prefix POST-INFER-RESETS
; RUN: firtool %s -parse-only | circt-opt -firrtl-infer-resets -firrtl-sfc-compat | FileCheck %s --implicit-check-not POST-SFC-COMPAT-NOT

; Check that FullAsyncResetAnnotation exists after infer-resets pass
; but is deleted after sfc-compat

FIRRTL version 3.3.0
circuit test :%[[{
  "class":"sifive.enterprise.firrtl.FullAsyncResetAnnotation",
    "target":"~test|test>reset"
    }]]
  module test :
    input clock : Clock
    input reset : AsyncReset
    ; POST-INFER-RESETS: [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]
    ; POST-SFC-COMPAT-NOT: [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}
    connect out, in
