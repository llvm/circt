; RUN: spt-translate -import-firrtl -verify-diagnostics %s | spt-opt | FileCheck %s

circuit basic :     ; CHECK: firrtl.circuit "basic" {

  ; CHECK-LABEL: firrtl.module @MyModule(%in: !firrtl.uint, %out: !firrtl.flip<uint<8>>) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>

    ; CHECK: firrtl.connect %out, %in : !firrtl.flip<uint<8>>, !firrtl.uint
    out <= in

  ; CHECK: }


  ; CHECK-LABEL: firrtl.module @circuit(%in: !firrtl.uint<80>) {
  module circuit :    ; Module with a keyword id.
    input in: UInt<80>

  ; CHECK: }


  ; Module to test type parsing.

  ; CHECK-LABEL: firrtl.module @types(
  module types :
    input c: Clock         ; CHECK: %c: !firrtl.clock,
    input r: Reset         ; CHECK: %r: !firrtl.reset,
    input a: Analog        ; CHECK: %a: !firrtl.analog,
    input a8: Analog<8>    ; CHECK: %a8: !firrtl.analog<8>,
    input s: SInt          ; CHECK: %s: !firrtl.sint,
    input s4: SInt<4>      ; CHECK: %s4: !firrtl.sint<4>,
    input u: UInt          ; CHECK: %u: !firrtl.uint,
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %bf: !firrtl.bundle<int_1: flip<uint<1>>, int_out: uint<2>>
    
    input vec: UInt<1>[4] ; CHECK: %vec: !firrtl.vector<uint<1>, 4>) {
    

  ; CHECK-LABEL: firrtl.module @stmts(
  module stmts :
    output auto : UInt<1>    ; CHECK: %auto: !firrtl.flip<uint<1>>)

    ; CHECK: %0 = firrtl.wire {name = "_T"} : !firrtl.vector<uint<1>, 12>
    wire _T : UInt<1>[12] @[Nodes.scala 370:76]

    ; CHECK: %1 = firrtl.wire {name = "_T_2"} : !firrtl.vector<uint<1>, 13>
    wire _T_2 : UInt<1>[13]

    ; CHECK: %2 = firrtl.wire {name = "out_0"} : !firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}

    ; CHECK: firrtl.connect %0, %1 : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 13>
    _T <= _T_2

    ; CHECK: firrtl.partialconnect %0, %1
    _T <- _T_2

    ; CHECK: firrtl.invalid %auto : !firrtl.flip<uint<1>>
    auto is invalid

    ;_T_2[0] <= _T[0] @[Xbar.scala 21:44]
    ;auto.out_0 <- out_0.member.0.reset @[LazyModule.scala 173:49]
    ;node _T_23 = or(_T_2[0], _T_2[1]) @[Parameters.scala 337:46]

