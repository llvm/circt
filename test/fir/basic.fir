; expected-error {{firrtl parsing not implemented}}

; RUN: spt-translate -import-firrtl -verify-diagnostics %s | FileCheck %s

module MyModule :
  input in: UInt<8>
  output out: UInt<8>
  out <= in


; CHECK: TOKEN: 'module'
; CHECK-NEXT: TOKEN: 'MyModule'
; CHECK-NEXT: TOKEN: ':'
; CHECK-NEXT: TOKEN: 'input'
; CHECK-NEXT: TOKEN: 'in'
; CHECK-NEXT: TOKEN: ':'
; CHECK-NEXT: TOKEN: 'UInt'
; CHECK-NEXT: TOKEN: '<'
; CHECK-NEXT: TOKEN: '8'
; CHECK-NEXT: TOKEN: '>'
; CHECK-NEXT: TOKEN: 'output'
; CHECK-NEXT: TOKEN: 'out'
; CHECK-NEXT: TOKEN: ':'
; CHECK-NEXT: TOKEN: 'UInt'
; CHECK-NEXT: TOKEN: '<'
; CHECK-NEXT: TOKEN: '8'
; CHECK-NEXT: TOKEN: '>'
; CHECK-NEXT: TOKEN: 'out'
; CHECK-NEXT: TOKEN: '<'
; CHECK-NEXT: TOKEN: '='
; CHECK-NEXT: TOKEN: 'in'
