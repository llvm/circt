; RUN: spt-translate -import-firrtl -verify-diagnostics %s | FileCheck %s

circuit basic :     ; CHECK: firrtl.circuit "basic" {

  ; CHECK-LABEL: firrtl.module @MyModule(%in: !firrtl.uint, %out: ui8) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>
    out <= in

  ; CHECK: }


  ; CHECK-LABEL: firrtl.module @circuit(%in: ui8) {
  module circuit :    ; Module with a keyword id.
    input in: UInt<8>

  ; CHECK: }


  ; Module to test type parsing.

  ; CHECK-LABEL: firrtl.module @types(
  module types :
    input c: Clock     ; CHECK: %c: !firrtl.clock,
    input r: Reset     ; CHECK: %r: !firrtl.reset,
    input a: Analog    ; CHECK: %a: !firrtl.analog,
    input s: SInt      ; CHECK: %s: !firrtl.sint,
    input s8: SInt<8>  ; CHECK: %s8: si8,
    input u: UInt      ; CHECK: %u: !firrtl.uint) {

