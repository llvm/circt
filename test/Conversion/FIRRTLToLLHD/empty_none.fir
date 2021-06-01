; RUN: circt-translate --import-firrtl %s | circt-opt --convert-firrtl-to-llhd | FileCheck %s

circuit test_mod :
  module test_mod :

; CHECK-LABEL: llhd.entity @test_mod () -> () {
; CHECK-NEXT: }
