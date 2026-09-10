; RUN: firtool %s -parse-only | circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets))'  | FileCheck %s --check-prefixes COMMON,POST-INFER-RESETS
; RUN: firtool %s -parse-only | circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets,firrtl.module(firrtl-sfc-compat)))' | FileCheck %s --check-prefixes COMMON,POST-SFC-COMPAT

; Check that FullResetAnnotation exists after infer-resets pass
; but is deleted after sfc-compat

FIRRTL version 3.3.0
circuit test :%[[
{ "class":"circt.FullResetAnnotation",
  "target":"~test|test>reset",
  "resetType":"async" },
{ "class":"circt.FullResetAnnotation",
  "target":"~test|foo>r",
  "resetType":"async" }
]]
  ; COMMON-LABEL: module @test
  module test :
    input clock : Clock
    input reset : AsyncReset
    ; POST-INFER-RESETS: [{class = "circt.FullResetAnnotation", resetType = "async"}]
    ; POST-SFC-COMPAT-NOT: [{class = "circt.FullResetAnnotation", resetType = "async"}]
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}
    connect out, in

  ; COMMON-LABEL: module private @foo
  module foo :
    input clock : Clock
    input reset : AsyncReset
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}

    ; POST-INFER-RESETS: [{class = "circt.FullResetAnnotation", resetType = "async"}]
    ; POST-SFC-COMPAT-NOT: [{class = "circt.FullResetAnnotation", resetType = "async"}]
     wire r : AsyncReset
    connect out, in
