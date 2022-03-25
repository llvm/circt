; RUN: firtool %s --format=fir --parse-only | FileCheck %s

; Ensure connect is demoted to partial connect in the presence of implicit truncation.

; CHECK-LABEL: @regularConnect
; CHECK: firrtl.connect %a, %b : !firrtl.uint<5>, !firrtl.uint<3>
circuit regularConnect :
 module regularConnect :
   output a: UInt<5>
   input b: UInt<3>
   a <= b

; CHECK-LABEL: @truncatingIntegerConnect
; CHECK: firrtl.partialconnect %a, %b : !firrtl.uint<3>, !firrtl.uint<5>
circuit truncatingIntegerConnect :
 module truncatingIntegerConnect :
   output a: UInt<3>
   input b: UInt<5>
   a <= b

; CHECK-LABEL: @regularBundleConnect
; CHECK: firrtl.connect %a, %b : !firrtl.bundle<a: uint<5>, b: uint<3>>, !firrtl.bundle<a: uint<3>, b: uint<2>>
circuit regularBundleConnect :
 module regularBundleConnect :
   output a: { a: UInt<5>, b: UInt<3> }
   input b: { a: UInt<3>, b: UInt<2> }
   a <= b

; CHECK-LABEL: @truncatingBundleConnect
; CHECK: firrtl.partialconnect %a, %b : !firrtl.bundle<a: uint<5>, b: uint<3>>, !firrtl.bundle<a: uint<6>, b: uint<1>>
circuit truncatingBundleConnect :
 module truncatingBundleConnect :
   output a: { a: UInt<5>, b: UInt<3> }
   input b: { a: UInt<6>, b: UInt<1> }
   a <= b
