; Test that the --verify option works with all modes
; This test contains invalid FIRRTL (layerblock connecting to outside destination)
;
; With verify=all or verify=default, the verifier should catch the layerblock error
; RUN: not firtool %s --parse-only --verify=all 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not firtool %s --parse-only --verify=default 2>&1 | FileCheck %s --check-prefix=ERROR
;
; With verify=none, the verifier error is not caught.
; RUN: firtool %s --parse-only --verify=none | FileCheck %s --check-prefix=SUCCESS

FIRRTL version 4.0.0
; SUCCESS: "VerifyTest"
circuit VerifyTest :
  layer A, bind:
  public module VerifyTest :
    input a : UInt<8>
    output b : UInt<8>
    layerblock A:
      ; ERROR: op connects to a destination which is defined outside its enclosing layer block
      connect b, a
