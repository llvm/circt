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

  ; CHECK-LABEL: firrtl.extmodule @MyExtModule(!firrtl.uint {firrtl.name = "in"}, !firrtl.flip<uint<8>> {firrtl.name = "out"})
  ; CHECK-NOT: {
  extmodule MyExtModule : 
    input in: UInt
    output ,,, out ,,: ,, UInt,<,8,>  ; Commas are whitespace

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
    input clock : Clock      ; CHECK: %clock: !firrtl.clock,
    output auto : UInt<1>    ; CHECK: %auto: !firrtl.flip<uint<1>>,
    input i8 : UInt<8>       ; CHECK: %i8: !firrtl.uint<8>)

    ; CHECK: %_T = firrtl.wire {name = "_T"} : !firrtl.vector<uint<1>, 12>
    wire _T : UInt<1>[12] @[Nodes.scala 370:76]

    ; CHECK: %_T_2 = firrtl.wire {name = "_T_2"} : !firrtl.vector<uint<1>, 13>
    wire _T_2 : UInt<1>[13]

    ; CHECK: %out_0 = firrtl.wire {name = "out_0"} : !firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}

    ; CHECK: firrtl.connect %_T, %_T_2 : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 13>
    _T <= _T_2

    ; CHECK: firrtl.partialconnect %_T, %_T_2
    _T <- _T_2

    ; CHECK: firrtl.invalid %auto : !firrtl.flip<uint<1>>
    auto is invalid

    ; CHECK: firrtl.skip
    skip  @[SKipLoc.scala 42:24]

    ; CHECK: firrtl.invalid %reset : !firrtl.uint<1>
    reset is invalid

    ; CHECK: %0 = firrtl.subfield %out_0("member") : (!firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>) -> !firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>
    ; CHECK: %1 = firrtl.subfield %0("0") : (!firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>) -> !firrtl.bundle<clock: clock, reset: uint<1>>
    ; CHECK: %2 = firrtl.subfield %1("reset") : (!firrtl.bundle<clock: clock, reset: uint<1>>) -> !firrtl.uint<1>
    ; CHECK: firrtl.partialconnect %auto, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    auto <- out_0.member.0.reset @[Field 173:49]

    ; CHECK: %3 = firrtl.subindex %_T_2[0] : (!firrtl.vector<uint<1>, 13>) -> !firrtl.uint<1>
    ; CHECK: %4 = firrtl.subindex %_T[0] : (!firrtl.vector<uint<1>, 12>) -> !firrtl.uint<1>
    ; CHECK: firrtl.connect %3, %4
    _T_2[0] <= _T[0] @[Xbar.scala 21:44]

    ; CHECK: %n1 = firrtl.node %auto {name = "n1"} : !firrtl.flip<uint<1>>
    node n1 = auto

    ; CHECK: firrtl.add %reset, %reset : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    node n2 = add(reset, reset)

    ; CHECK: firrtl.asClock %reset : (!firrtl.uint<1>) -> !firrtl.clock
    node n3 = asClock(reset)

    ; CHECK: firrtl.asAsyncReset %reset : (!firrtl.uint<1>) -> !firrtl.asyncreset
    node ar = asAsyncReset(reset)

    ; CHECK: firrtl.constant(42 : ui10) : !firrtl.uint<10>
    ; CHECK: firrtl.constant(171 : ui8) : !firrtl.uint<8>
    ; CHECK: firrtl.add
    ; CHECK: firrtl.connect %auto
    auto <= add(UInt<10>(42), UInt<8>("hAB"))

    ; CHECK: firrtl.when %reset {
    ; CHECK:   firrtl.connect %_T, %_T_2
    ; CHECK: } else {
    ; CHECK:   firrtl.partialconnect %_T, %_T_2
    ; CHECK: }
    when reset : _T <= _T_2 else : _T <- _T_2

    ; CHECK: firrtl.when %reset {
    ; CHECK:   [[N4A:%.+]] = firrtl.node %_T_2
    ; CHECK:   firrtl.connect %_T, [[N4A]]
    ; CHECK: } else {
    ; CHECK:   [[N4B:%.+]] = firrtl.node %_T_2
    ; CHECK:   firrtl.partialconnect %_T, [[N4B]]
    ; CHECK: }
    when reset :
      node n4 = _T_2
      _T <= n4
    else :
      node n4 = _T_2   ; 'n4' name is in unique scopes.
      _T <- n4

    ; CHECK: [[TMP:%.+]] = firrtl.constant(4 : ui1024)
    ; CHECK: [[COND:%.+]] = firrtl.lt %reset, [[TMP]]
    ; CHECK: firrtl.when [[COND]] {
    ; CHECK:   firrtl.connect %_T, %_T_2
    ; CHECK: }
    ; CHECK-NOT: else
    when lt(reset, UInt(4)) :   ;; When with no else.
      _T <= _T_2

    ; CHECK: firrtl.printf %clock, %reset, "\22Something interesting! %x %x\22" %_T, %_T_2 : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 13>
    printf(clock, reset, "Something interesting! %x %x", _T, _T_2)

    ; CHECK: firrtl.stop %clock, %reset, 42
    stop(clock, reset, 42)

    ; CHECK: firrtl.bits %i8 4 to 2 : (!firrtl.uint<8>) -> !firrtl.uint<3>
    node n4 = bits(i8, 4, 2)

    ; CHECK: firrtl.shl %i8, 4 : (!firrtl.uint<8>) -> !firrtl.uint<12>
    ; CHECK: firrtl.shr %i8, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    node n5 = or(shl(i8, 4), shr(i8, 8))
  
    ; CHECK: firrtl.dshl %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<23>
    node n6 = dshl(i8, UInt<4>(7))

    ; CHECK: firrtl.cat %{{.*}}, %{{.*}} : (!firrtl.uint<12>, !firrtl.uint<23>) -> !firrtl.uint<35>
    node n7 = cat(n5, n6)

    ; CHECK: firrtl.mux(%reset, %i8, %{{.*}}) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint) -> !firrtl.uint
    node n8 = mux(reset, i8, UInt(4))

    ; CHECK: firrtl.reginit %clock, %reset, %{{.*}} {name = "_T_2621"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>) -> !firrtl.uint<4>
    reg _T_2621 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0")) @[Edges.scala 230:27]

    ; CHECK: firrtl.div %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<8>
    node n9 = div(i8, UInt<4>(4))

    ; CHECK: firrtl.tail %i8 7 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    ; CHECK: firrtl.tail %i8 0 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    node n10 = add(tail(i8, 7), tail(i8, 0))

    ; The Scala implementation of FIRRTL prints registers without a reset value
    ; using the register name as the reset.  Make sure we handle this for
    ; compatibility.
    ; CHECK: firrtl.reg %clock {name = "_T_2622"} : (!firrtl.clock) -> !firrtl.uint<4>
    reg _T_2622 : UInt<4>, clock with :
      reset => (UInt<1>("h0"), _T_2622)

    ; CHECK: [[xyz:%.+]] = firrtl.instance @circuit {name = "xyz"} : !firrtl.bundle<in: flip<uint<80>>>
    inst xyz of circuit
    ; CHECK: [[xyz_in:%.+]] = firrtl.subfield [[xyz]]("in") : (!firrtl.bundle<in: flip<uint<80>>>) -> !firrtl.flip<uint<80>>
    ; CHECK: firrtl.connect [[xyz_in]], %i8 : !firrtl.flip<uint<80>>, !firrtl.uint<8>
    xyz.in <= i8

    ; CHECK: [[myext:%.+]] = firrtl.instance @MyExtModule {name = "myext"} : !firrtl.bundle<in: flip<uint>, out: uint<8>>
    inst myext of MyExtModule
    myext.in <= i8
    printf(clock, reset, "Something interesting! %x", myext.out)

    ; CHECK: firrtl.subaccess %_T[%i8] : (!firrtl.vector<uint<1>, 12>, !firrtl.uint<8>) -> !firrtl.uint<1>
    auto <= _T[i8]

    ; CHECK: firrtl.subaccess %_T[%auto] : (!firrtl.vector<uint<1>, 12>, !firrtl.flip<uint<1>>) -> !firrtl.uint<1>
    auto <= _T[auto]

    ; CHECK: %myMem = firrtl.cmem {name = "myMem"} : !firrtl.vector<bundle<id: uint<4>, resp: uint<2>>, 8>
    cmem myMem : { id : UInt<4>, resp : UInt<2>} [8] @[Decoupled.scala 209:24]

    ; CHECK: %memValue = firrtl.memoryport infer %myMem, %i8, %clock {name = "memValue"} : (!firrtl.vector<bundle<id: uint<4>, resp: uint<2>>, 8>, !firrtl.uint<8>, !firrtl.clock) -> !firrtl.bundle<id: uint<4>, resp: uint<2>>
    infer mport memValue = myMem[i8], clock
    auto <= memValue.id

  ; CHECK-LABEL: firrtl.module @type_handling(
  module type_handling :
    wire _T_6 : { flip b : { bits : { source : UInt<7> } } }
    node _T_8 = bits(_T_6.b.bits.source, 5, 0)

  ; CHECK-LABEL: firrtl.module @expr_stmt_ambiguity(
  module expr_stmt_ambiguity :
    ; CHECK: %reg = firrtl.wire {name = "reg"} : !firrtl.uint
    wire reg : UInt
    ; CHECK: firrtl.connect %reg,
    reg <= UInt(42)

 ; CHECK-LABEL: firrtl.module @oversize_shift(
  module oversize_shift :
    wire value : UInt<2>
    ; CHECK: firrtl.shr %value, 5 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    node n = shr(value, 5)

