; RUN: firtool %s --format=fir --parse-only --split-input-file | FileCheck %s

; CHECK-LABEL: @regularConnect
; CHECK: %0 = firrtl.zext %b : (!firrtl.uint<3>) -> !firrtl.uint<5>
; CHECK: firrtl.matchingconnect %a, %0 : !firrtl.uint<5>
 FIRRTL version 4.0.0
circuit regularConnect :
 public module regularConnect :
   output a: UInt<5>
   input b: UInt<3>
   connect a, b

; // -----

; CHECK-LABEL: @regularBundleConnect
; CHECK: %0 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<5>, b: uint<3>>
; CHECK: %1 = firrtl.subfield %b[a] : !firrtl.bundle<a: uint<3>, b: uint<2>>
; CHECK: %2 = firrtl.zext %1 : (!firrtl.uint<3>) -> !firrtl.uint<5>
; CHECK: firrtl.matchingconnect %0, %2 : !firrtl.uint<5>
; CHECK: %3 = firrtl.subfield %a[b] : !firrtl.bundle<a: uint<5>, b: uint<3>>
; CHECK: %4 = firrtl.subfield %b[b] : !firrtl.bundle<a: uint<3>, b: uint<2>>
; CHECK: %5 = firrtl.zext %4 : (!firrtl.uint<2>) -> !firrtl.uint<3>
; CHECK: firrtl.matchingconnect %3, %5 : !firrtl.uint<3>
FIRRTL version 4.0.0
circuit regularBundleConnect :
 public module regularBundleConnect :
   output a: { a: UInt<5>, b: UInt<3> }
   input b: { a: UInt<3>, b: UInt<2> }
   connect a, b
