; RUN: spt-translate -import-firrtl -verify-diagnostics %s | FileCheck %s

circuit basic :     ; CHECK: firrtl.circuit "basic" {

  ; CHECK: firrtl.module @MyModule(%in: ui8, %out: ui8) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt<8>
    output out: UInt<8>
    out <= in

  ; CHECK: }


  ; CHECK: firrtl.module @circuit(%in: ui8) {
  module circuit :    ; Module with a keyword id.
    input in: UInt<8>


  ; CHECK: }
