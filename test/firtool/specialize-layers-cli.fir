; RUN: firtool -parse-only %s | FileCheck %s --check-prefixes=NONE
; RUN: firtool -parse-only -disable-layers=A %s | FileCheck %s --check-prefixes=DISABLEA
; RUN: firtool -parse-only -enable-layers=A %s | FileCheck %s --check-prefixes=ENABLEA
; RUN: firtool -parse-only -enable-layers=A.B %s | FileCheck %s --check-prefixes=ENABLEB
; RUN: firtool -parse-only -disable-layers=A,A.B %s | FileCheck %s --check-prefixes=DISABLEBOTH
; RUN: firtool -parse-only -disable-layers=A -enable-layers=A.B %s | FileCheck %s --check-prefixes=BOTH

; Check that the command line options are working correctly.
; https://github.com/llvm/circt/issues/7345

FIRRTL version 4.0.0
; NONE:             firrtl.circuit "Foo"
; DISABLEA:         firrtl.circuit "Foo"
; DISABLEA-SAME:      disable_layers = [@A]
; ENABLEA:          firrtl.circuit "Foo"
; ENABLEA-SAME:       enable_layers = [@A]
; ENABLEB:          firrtl.circuit "Foo"
; ENABLEB-SAME:       enable_layers = [@A::@B]
; DISABLEBOTH:      firrtl.circuit "Foo"
; DISABLEBOTH-SAME:   disable_layers = [@A, @A::@B]
; BOTH:             firrtl.circuit "Foo"
; BOTH-SAME:          disable_layers = [@A], enable_layers = [@A::@B]
circuit Foo:
  layer A, bind:
    layer B, bind:
  public module Foo:
