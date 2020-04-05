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
    input reset : UInt<1>    ; CHECK: %reset: !firrtl.uint<1>,
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

    ; CHECK: firrtl.skip
    skip  @[SKipLoc.scala 42:24]

    ; CHECK: firrtl.invalid %reset : !firrtl.uint<1>
    reset is invalid

    ; CHECK: %3 = firrtl.subfield %2("member") : (!firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>) -> !firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>
    ; CHECK: %4 = firrtl.subfield %3("0") : (!firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>) -> !firrtl.bundle<clock: clock, reset: uint<1>>
    ; CHECK: %5 = firrtl.subfield %4("reset") : (!firrtl.bundle<clock: clock, reset: uint<1>>) -> !firrtl.uint<1>
    ; CHECK: firrtl.partialconnect %auto, %5 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    auto <- out_0.member.0.reset @[Field 173:49]

    ; CHECK: %6 = firrtl.subindex %1[0] : (!firrtl.vector<uint<1>, 13>) -> !firrtl.uint<1>
    ; CHECK: %7 = firrtl.subindex %0[0] : (!firrtl.vector<uint<1>, 12>) -> !firrtl.uint<1>
    ; CHECK: firrtl.connect %6, %7
    _T_2[0] <= _T[0] @[Xbar.scala 21:44]

    ; CHECK: %8 = firrtl.node %auto {name = "n1"} : !firrtl.flip<uint<1>>
    node n1 = auto

    ; CHECK: firrtl.add %reset, %reset : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    node n2 = add(reset, reset)

    ; CHECK: firrtl.asClock %reset : !firrtl.uint<1>
    node n3 = asClock(reset)

    ; CHECK: firrtl.constant 42 : !firrtl.uint<10>
    ; CHECK: firrtl.constant 171 : !firrtl.uint<8>
    ; CHECK: firrtl.add
    ; CHECK: firrtl.connect %auto
    auto <= add(UInt<10>(42), UInt<8>("hAB"))

    ; CHECK: firrtl.connect %0, %1
    ; CHECK: firrtl.partialconnect %0
    when reset : _T <= _T_2 else : _T <- _T_2

    ; CHECK: [[N4A:%.+]] = firrtl.node %1
    ; CHECK: firrtl.connect %0, [[N4A]]
    ; CHECK: [[N4B:%.+]] = firrtl.node %1
    ; CHECK: firrtl.partialconnect %0, [[N4B]]
    when reset :
      node n4 = _T_2
      _T <= n4
    else :
      node n4 = _T_2   ; 'n4' name is in unique scopes.
      _T <- n4
    
