; RUN: circt-translate -import-firrtl -verify-diagnostics %s | circt-opt | FileCheck %s

circuit MyModule :     ; CHECK: firrtl.circuit "MyModule" {

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

  ; CHECK-LABEL: firrtl.extmodule @MyExtModule(!firrtl.uint {firrtl.name = "in"},
  ; CHECK: !firrtl.flip<uint<8>> {firrtl.name = "out"})
  ; CHECK: attributes {defname = "myextmodule"}
  ; CHECK-NOT: {
  extmodule MyExtModule :
    input in: UInt
    output ,,, out ,,: ,, UInt,<,8,>  ; Commas are whitespace
    defname = myextmodule

  ; CHECK-LABEL: firrtl.extmodule @MyParameterizedExtModule(!firrtl.uint {firrtl.name = "in"}, !firrtl.flip<uint<8>> {firrtl.name = "out"})
  ; CHECK: attributes {defname = "name_thing",
  ; CHECK: parameters = {DEFAULT = 0 : i64,
  ; CHECK:               DEPTH = 3.242000e+01 : f64,
  ; CHECK:               FORMAT = "xyz_timeout=%d\0A",
  ; CHECK:               WIDTH = 32 : i8}}
  ; CHECK-NOT: {
  extmodule MyParameterizedExtModule :
    input in: UInt
    output out: UInt<8>
    defname = name_thing
    parameter FORMAT = "xyz_timeout=%d\n"
    parameter DEFAULT = 0
    parameter WIDTH = 32
    parameter DEPTH = 32.42

  ; Module to test type parsing.

  ; CHECK-LABEL: firrtl.module @types(
  module types :
    input c: Clock         ; CHECK: %c: !firrtl.clock,
    input r: Reset         ; CHECK: %r: !firrtl.reset,
    input ar: AsyncReset   ; CHECK: %ar: !firrtl.asyncreset,
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
    input reset : UInt<1>         ; CHECK: %reset: !firrtl.uint<1>,
    input reset_async: AsyncReset ; CHECK: %reset_async: !firrtl.asyncreset,
    input reset_abstract: Reset   ; CHECK: %reset_abstract: !firrtl.reset,
    input clock : Clock           ; CHECK: %clock: !firrtl.clock,
    output auto : UInt<1>         ; CHECK: %auto: !firrtl.flip<uint<1>>,
    output auto11 : UInt<11>      ; CHECK: %auto11: !firrtl.flip<uint<11>>,
    output sauto : SInt<9>        ; CHECK: %sauto: !firrtl.flip<sint<9>>,
    input i8 : UInt<8>            ; CHECK: %i8: !firrtl.uint<8>,
    input s1 : SInt<1>            ; CHECK: %s1: !firrtl.sint<1>,
    input s8 : SInt<8>            ; CHECK: %s8: !firrtl.sint<8>,
    input a1 : Analog<1>          ; CHECK: %a1: !firrtl.analog<1>,
    input a8 : Analog<8>          ; CHECK: %a8: !firrtl.analog<8>)

    ; CHECK: %_t = firrtl.wire : !firrtl.vector<uint<1>, 12>
    wire _t : UInt<1>[12] @[Nodes.scala 370:76]

    ; CHECK: %_t_2 = firrtl.wire : !firrtl.vector<uint<1>, 12>
    wire _t_2 : UInt<1>[12]

    ; CHECK: %out_0 = firrtl.wire : !firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}

    ; CHECK: firrtl.connect %_t, %_t_2 : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    _t <= _t_2

    ; CHECK: firrtl.partialconnect %_t, %_t_2
    _t <- _t_2

    ; CHECK: [[INV:%.+]]  = firrtl.invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: firrtl.connect %auto, [[INV]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    auto is invalid

    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue : !firrtl.analog<1>
    ; CHECK-NEXT: firrtl.attach %a1, [[INV]] : !firrtl.analog<1>, !firrtl.analog<1>
    a1 is invalid

    ; CHECK: firrtl.skip
    skip  @[SKipLoc.scala 42:24]

    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: firrtl.connect %reset, [[INV]] : !firrtl.uint<1>, !firrtl.uint<1>
    reset is invalid

    ; CHECK: [[A:%.+]] = firrtl.subfield %out_0("member") : (!firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>) -> !firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>
    ; CHECK: [[B:%.+]] = firrtl.subfield [[A]]("0") : (!firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>) -> !firrtl.bundle<clock: clock, reset: uint<1>>
    ; CHECK: [[C:%.+]] = firrtl.subfield [[B]]("reset") : (!firrtl.bundle<clock: clock, reset: uint<1>>) -> !firrtl.uint<1>
    ; CHECK: firrtl.partialconnect %auto, [[C]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    auto <- out_0.member.0.reset @[Field 173:49]

    ; CHECK: [[A:%.+]] = firrtl.subindex %_t_2[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: [[B:%.+]] = firrtl.subindex %_t[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: firrtl.connect [[A]], [[B]]
    _t_2[0] <= _t[0] @[Xbar.scala 21:44]

    ; CHECK: %n1 = firrtl.node %i8 : !firrtl.uint<8>
    node n1 = i8

    ; CHECK: firrtl.add %reset, %reset : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    node n2 = add(reset, reset)

    ; CHECK: firrtl.asClock %reset : (!firrtl.uint<1>) -> !firrtl.clock
    node n3 = asClock(reset)

    ; CHECK: firrtl.asUInt %clock : (!firrtl.clock) -> !firrtl.uint<1>
    node check_u0 = asUInt(clock)
    ; CHECK: firrtl.asUInt %i8 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    node check_u1 = asUInt(i8)
    ; CHECK: firrtl.asUInt %s8 : (!firrtl.sint<8>) -> !firrtl.uint<8>
    node check_u2 = asUInt(s8)
    ; CHECK: firrtl.asUInt %a8 : (!firrtl.analog<8>) -> !firrtl.uint<8>
    node check_u3 = asUInt(a8)
    ; CHECK: firrtl.asUInt %reset_abstract : (!firrtl.reset) -> !firrtl.uint<1>
    node check_u5 = asUInt(reset_abstract)
    ; CHECK: firrtl.asUInt %reset_async : (!firrtl.asyncreset) -> !firrtl.uint<1>
    node check_u6 = asUInt(reset_async)

    ; CHECK: firrtl.asSInt %clock : (!firrtl.clock) -> !firrtl.sint<1>
    node check_s0 = asSInt(clock)
    ; CHECK: firrtl.asSInt %i8 : (!firrtl.uint<8>) -> !firrtl.sint<8>
    node check_s1 = asSInt(i8)
    ; CHECK: firrtl.asSInt %s8 : (!firrtl.sint<8>) -> !firrtl.sint<8>
    node check_s2 = asSInt(s8)
    ; CHECK: firrtl.asSInt %a8 : (!firrtl.analog<8>) -> !firrtl.sint<8>
    node check_s3 = asSInt(a8)
    ; CHECK: firrtl.asSInt %reset_abstract : (!firrtl.reset) -> !firrtl.sint<1>
    node check_s5 = asSInt(reset_abstract)
    ; CHECK: firrtl.asSInt %reset_async : (!firrtl.asyncreset) -> !firrtl.sint<1>
    node check_s6 = asSInt(reset_async)

    ; CHECK: firrtl.asAsyncReset %clock : (!firrtl.clock) -> !firrtl.asyncreset
    node check_ar0 = asAsyncReset(clock)
    ; CHECK: firrtl.asAsyncReset %reset : (!firrtl.uint<1>) -> !firrtl.asyncreset
    node check_ar1 = asAsyncReset(reset)
    ; CHECK: firrtl.asAsyncReset %s1 : (!firrtl.sint<1>) -> !firrtl.asyncreset
    node check_ar2 = asAsyncReset(s1)
    ; CHECK: firrtl.asAsyncReset %a1 : (!firrtl.analog<1>) -> !firrtl.asyncreset
    node check_ar3 = asAsyncReset(a1)
    ; CHECK: firrtl.asAsyncReset %reset_abstract : (!firrtl.reset) -> !firrtl.asyncreset
    node check_ar4 = asAsyncReset(reset_abstract)
    ; CHECK: firrtl.asAsyncReset %reset_async : (!firrtl.asyncreset) -> !firrtl.asyncreset
    node check_ar5 = asAsyncReset(reset_async)

    ; CHECK: firrtl.asClock %clock : (!firrtl.clock) -> !firrtl.clock
    node check_c0 = asClock(clock)
    ; CHECK: firrtl.asClock %reset : (!firrtl.uint<1>) -> !firrtl.clock
    node check_c1 = asClock(reset)
    ; CHECK: firrtl.asClock %s1 : (!firrtl.sint<1>) -> !firrtl.clock
    node check_c2 = asClock(s1)
    ; CHECK: firrtl.asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    node check_c3 = asClock(a1)
    ; CHECK: firrtl.asClock %reset_abstract : (!firrtl.reset) -> !firrtl.clock
    node check_c4 = asClock(reset_abstract)
    ; CHECK: firrtl.asClock %reset_async : (!firrtl.asyncreset) -> !firrtl.clock
    node check_c5 = asClock(reset_async)

    ; CHECK: %c42_ui10 = firrtl.constant(42 : ui10) : !firrtl.uint<10>
    ; CHECK: %c171_ui8 = firrtl.constant(171 : ui8) : !firrtl.uint<8>
    ; CHECK: firrtl.add %c42_ui10, %c171_ui8
    ; CHECK: firrtl.connect %auto
    auto11 <= add(UInt<10>(42), UInt<8>("hAB"))

    ; CHECK: %c-85_si8 = firrtl.constant(-85 : si8) : !firrtl.sint<8>
    sauto <= add(s8, SInt<8>(-85))

    ; CHECK: firrtl.when %reset {
    ; CHECK:   firrtl.connect %_t, %_t_2
    ; CHECK: } else {
    ; CHECK:   firrtl.partialconnect %_t, %_t_2
    ; CHECK: }
    when reset : _t <= _t_2 else : _t <- _t_2

    ; CHECK: firrtl.when %reset {
    ; CHECK:   [[N4A:%.+]] = firrtl.node %_t_2
    ; CHECK:   firrtl.connect %_t, [[N4A]]
    ; CHECK: } else {
    ; CHECK:   [[N4B:%.+]] = firrtl.node %_t_2
    ; CHECK:   firrtl.partialconnect %_t, [[N4B]]
    ; CHECK: }
    when reset :
      node n4 = _t_2
      _t <= n4
    else :
      node n4 = _t_2   ; 'n4' name is in unique scopes.
      _t <- n4

    ; CHECK: [[TMP:%.+]] = firrtl.constant(4 : ui4)
    ; CHECK: [[COND:%.+]] = firrtl.lt %reset, [[TMP]]
    ; CHECK: firrtl.when [[COND]] {
    ; CHECK:   firrtl.connect %_t, %_t_2
    ; CHECK: }
    ; CHECK-NOT: else
    when lt(reset, UInt(4)) :   ;; When with no else.
      _t <= _t_2

    ; CHECK: firrtl.printf %clock, %reset, "Something interesting!\0A %x %x"(%_t, %_t_2) : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    printf(clock, reset, "Something interesting!\n %x %x", _t, _t_2)

    ; CHECK: firrtl.stop %clock, %reset, 42
    stop(clock, reset, 42)

    ; CHECK: firrtl.bits %i8 4 to 2 : (!firrtl.uint<8>) -> !firrtl.uint<3>
    node n4 = bits(i8, 4, 2)

    ; CHECK: firrtl.shl %i8, 4 : (!firrtl.uint<8>) -> !firrtl.uint<12>
    ; CHECK: firrtl.shr %i8, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    node n5 = or(shl(i8, 4), shr(i8, 8))

    ; CHECK: firrtl.dshl %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<23>
    node n6 = dshl(i8, UInt<4>(7))
    ; CHECK: firrtl.dshlw %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<8>
    node n6s = dshlw(i8, UInt<4>(7))

    ; CHECK: firrtl.cat %{{.*}}, %{{.*}} : (!firrtl.uint<12>, !firrtl.uint<23>) -> !firrtl.uint<35>
    node n7 = cat(n5, n6)

    ; CHECK: firrtl.mux(%reset, %i8, %{{.*}}) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint) -> !firrtl.uint
    node n8 = mux(reset, i8, UInt(4))

    ; CHECK: firrtl.regreset %clock, %reset, %{{.*}} {name = "_t_2621"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>) -> !firrtl.uint<4>
    reg _t_2621 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0")) @[Edges.scala 230:27]

    ; CHECK: firrtl.regreset %clock, %reset, %{{.*}} {name = "_t_1601"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<2>
    reg _t_1601 : UInt<2>, clock with :
      (reset => (reset, UInt<2>("h00"))) @[Edges.scala 230:27]

    ; CHECK: firrtl.div %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<8>
    node n9 = div(i8, UInt<4>(4))

    ; CHECK: firrtl.tail %i8, 7 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    ; CHECK: firrtl.tail %i8, 0 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    ; CHECK: firrtl.head %i8, 4 : (!firrtl.uint<8>) -> !firrtl.uint<4>
    node n10 = add(add(tail(i8, 7), tail(i8, 0)), head(i8, 4))

    ; CHECK: firrtl.tail %{{.*}}, 3 : (!firrtl.sint<8>) -> !firrtl.uint<5>
    node n10s = tail(asSInt(i8), 3)

    ; The Scala implementation of FIRRTL prints registers without a reset value
    ; using the register name as the reset.  Make sure we handle this for
    ; compatibility.
    ; CHECK: firrtl.reg %clock {name = "_t_2622"} : (!firrtl.clock) -> !firrtl.uint<4>
    reg _t_2622 : UInt<4>, clock with :
      reset => (UInt<1>("h0"), _t_2622)

    ; CHECK: %xyz_in = firrtl.instance @circuit {name = "xyz", portNames = ["in"]} : !firrtl.flip<uint<80>>
    inst xyz of circuit
    ; CHECK: firrtl.connect %xyz_in, %i8 : !firrtl.flip<uint<80>>, !firrtl.uint<8>
    xyz.in <= i8

    ; CHECK: %myext_in, %myext_out = firrtl.instance @MyExtModule
    ; CHECK: {name = "myext", portNames = ["in", "out"]} :
    ; CHECK: !firrtl.flip<uint>, !firrtl.uint<8>
    inst myext of MyExtModule
    myext.in <= i8
    printf(clock, reset, "Something interesting! %x", myext.out)

    ; CHECK: firrtl.when %reset  {
    when reset :
      ; CHECK: %reset_myext_in, %reset_myext_out = firrtl.instance @MyExtModule
      ; CHECK: {name = "reset_myext", portNames = ["in", "out"]} :
      ; CHECK: !firrtl.flip<uint>, !firrtl.uint<8>
      inst reset_myext of MyExtModule
      reset_myext.in <= i8
    ; CHECK: }

    ; CHECK: firrtl.subaccess %_t[%i8] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<8>
    auto <= _t[i8]

    ; CHECK: [[autoP:%.+]] = firrtl.asPassive %auto : !firrtl.flip<uint<1>>
    ; CHECK: firrtl.subaccess %_t{{\[}}[[autoP]]] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<1>
    auto <= _t[auto]

    ; CHECK: %myMem = firrtl.cmem {name = "myMem"} : !firrtl.vector<bundle<id: uint<4>, resp: uint<2>>, 8>
    cmem myMem : { id : UInt<4>, resp : UInt<2>} [8] @[Decoupled.scala 209:24]

    ; CHECK: %memValue = firrtl.memoryport Infer %myMem, %i8, %clock {name = "memValue"} : (!firrtl.vector<bundle<id: uint<4>, resp: uint<2>>, 8>, !firrtl.uint<8>, !firrtl.clock) -> !firrtl.bundle<id: uint<4>, resp: uint<2>>
    infer mport memValue = myMem[i8], clock
    auto11 <= memValue.id

    ; CHECK: %base_table_0 = firrtl.smem Undefined {name = "base_table_0"} : !firrtl.vector<vector<uint<1>, 9>, 256>
    smem base_table_0 : UInt<1>[9] [256]
    ; CHECK: %base_table_1 = firrtl.smem Old {name = "base_table_1"} : !firrtl.vector<vector<uint<1>, 9>, 256>
    smem base_table_1 : UInt<1>[9] [256] old

    ; CHECK: %tableValue = firrtl.memoryport Read %base_table_0, %i8, %clock {name = "tableValue"} : (!firrtl.vector<vector<uint<1>, 9>, 256>, !firrtl.uint<8>, !firrtl.clock) -> !firrtl.vector<uint<1>, 9>
    read mport tableValue = base_table_0[i8], clock

    ; CHECK: firrtl.pad %i8, 10 : (!firrtl.uint<8>) -> !firrtl.uint<10>
    node n11 = pad(i8, 10)

    ; CHECK: firrtl.andr %n11 : (!firrtl.uint<10>) -> !firrtl.uint<1>
    node n12 = andr(n11)

    ;  Test that _T and _GEN names are ignored by the parser.
    ; CHECK: %{{[0-9]+}} = firrtl.wire
    wire _T_42 : UInt<1>

    ; Test that node decls with ignored names are not generated at all.
    ; CHECK-NOT: firrtl.node
    node _GEN_43 = n12

    ; CHECK: firrtl.andr %n12
    node value = andr(_GEN_43)    ;; Uses n12 directly.

    ; CHECK: [[autoP:%.+]] = firrtl.asPassive %auto : !firrtl.flip<uint<1>>
    ; CHECK-NEXT: = firrtl.not [[autoP]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
    node n13 = not(auto)


    ; CHECK: %_M__T_10, %_M__T_11, %_M__T_18 = firrtl.mem Undefined {depth = 8 : i64, name = "_M", portNames = ["_T_10", "_T_11", "_T_18"]
    ; CHECK:   readLatency = 0 : i32, writeLatency = 1 : i32} :
    ; CHECK:       !firrtl.bundle<addr: flip<uint<3>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<id: analog<4>>, mask: flip<bundle<id: uint<1>>>>,
    ; CHECK:       !firrtl.bundle<addr: flip<uint<3>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<id: analog<4>>, mask: flip<bundle<id: uint<1>>>>,
    ; CHECK:       !firrtl.bundle<addr: flip<uint<3>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<id: analog<4>>
    mem _M : @[Decoupled.scala 209:24]
        data-type => { id : Analog<4> }
        depth => 8
        read-latency => 0
        write-latency => 1
        reader => _T_18
        writer => _T_10 _T_11
        read-under-write => undefined
    _M._T_18.addr is invalid @[Decoupled.scala 209:24]
    _M._T_18.clk is invalid @[Decoupled.scala 209:24]
    _M._T_18.en <= UInt<1>("h0") @[Decoupled.scala 209:24]
    _M._T_10.addr is invalid @[Decoupled.scala 209:24]
    _M._T_10.clk is invalid @[Decoupled.scala 209:24]
    _M._T_10.en <= UInt<1>("h0") @[Decoupled.scala 209:24]
    _M._T_10.data is invalid @[Decoupled.scala 209:24]
    _M._T_10.mask is invalid @[Decoupled.scala 209:24]

    ; CHECK: firrtl.attach %a8, %a8, %a8 :
    attach (a8, a8, a8)

    wire pred: UInt <1>
    wire en: UInt <1>
    pred <= eq(i8, i8)
    en <= not(reset)
    ; CHECK: firrtl.assert %clock, %pred, %en, "X equals Y when Z is valid"
    assert(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.assume %clock, %pred, %en, "X equals Y when Z is valid"
    assume(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.cover %clock, %pred, %en, "X equals Y when Z is valid"
    cover(clock, pred, en, "X equals Y when Z is valid")

  ; CHECK-LABEL: firrtl.module @type_handling(
  module type_handling :
    wire _t_6 : { flip b : { bits : { source : UInt<7> } } }
    node _t_8 = bits(_t_6.b.bits.source, 5, 0)

    ; CHECK: %flip1 = firrtl.wire : !firrtl.bundle<x: bundle<a: uint>>
    wire flip1 : { flip x : { flip a : UInt } }
    ; CHECK: %flip2 = firrtl.wire : !firrtl.bundle<x: bundle<a: uint, b: analog>>
    wire flip2 : { flip x : { flip a : UInt, b: Analog } }
    ; CHECK: %flip3 = firrtl.wire : !firrtl.bundle<x: bundle<a: uint, b: analog>>
    wire flip3 : { flip x : { flip a : UInt, flip b: Analog } }
    ; CHECK: %flip4 = firrtl.wire : !firrtl.bundle<x: vector<bundle<a: uint>, 4>>
    wire flip4 : { flip x : { flip a : UInt }[4] }


  ; CHECK-LABEL: firrtl.module @expr_stmt_ambiguity(
  module expr_stmt_ambiguity :
    ; CHECK: %reg = firrtl.wire : !firrtl.uint
    wire reg : UInt
    ; CHECK: firrtl.connect %reg,
    reg <= UInt(42)

    ; CHECK: %write = firrtl.wire
    wire write : { id : UInt<4>, resp : UInt<2>}

    ; CHECK: firrtl.subfield %write("id")
    write.id <= UInt(1)

  ; CHECK-LABEL: firrtl.module @oversize_shift(
  module oversize_shift :
    wire value : UInt<2>
    ; CHECK: firrtl.shr %value, 5 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    node n = shr(value, 5)

  ; CHECK-LABEL: firrtl.module @when_else_ambiguity(
  module when_else_ambiguity :
    output out : UInt
    input in : UInt
    wire reset : UInt<1>

  ; CHECK: firrtl.when {{.*}} {
    when reset : @[Debug.scala 1176:37]
    ; CHECK: firrtl.when {{.*}} {
      when reset :
        out <= in
    ; CHECK: }
    ; CHECK: } else {
    else :
        ; CHECK: firrtl.when {{.*}} {
      when reset : @[Debug.scala 1180:39]
        out <= in
    ; CHECK: }
    ; CHECK: }


  ; CHECK-LABEL: firrtl.module @chisel_when_mport_bug(
  module chisel_when_mport_bug :
    input cond : UInt<1>
    input foo : UInt
    input clock : Clock

    ; CHECK: [[HACKWIRE:%.+]] = firrtl.wire : !firrtl.vector<uint<1>, 9>
    ; CHECK: [[HACKWIRE2:%.+]] = firrtl.wire : !firrtl.vector<uint<1>, 9>
    ; CHECK: %memory = firrtl.smem Undefined {name = "memory"} : !firrtl.vector<vector<uint<1>, 9>, 256>
    smem memory : UInt<1>[9] [256]

    ; CHECK: firrtl.when %cond {
    when cond :
      node idx = foo
      ; CHECK: %xyz = firrtl.memoryport Read %memory, %idx
      ; CHECK: firrtl.connect [[HACKWIRE]], %xyz
      read mport xyz = memory[idx], clock
    ; CHECK: }
    node sometest = eq(foo, UInt(42))

    ; CHECK: firrtl.when %cond {
    when cond :
      node idx = foo
      ; CHECK: %xyz2 = firrtl.memoryport Read %memory, %idx
      ; CHECK: firrtl.skip
      ; CHECK: firrtl.connect [[HACKWIRE2]], %xyz2
      read mport xyz2 = memory[idx], clock
      skip @[Debug.scala 1176:37]
    ; CHECK: }
    node sometest2 = eq(foo, UInt(42))

    ; CHECK: firrtl.when %sometest {
    when sometest :
      ; CHECK: %awesome = firrtl.node [[HACKWIRE]]
      ; CHECK: %awesome2 = firrtl.node [[HACKWIRE2]]
      node awesome = xyz
      node awesome2 = xyz2


   ; CHECK-LABEL: firrtl.module @constant_implicit_cse(
  module constant_implicit_cse :
    input cond : UInt<1>

    ; CHECK: [[CST15:%.+]] = firrtl.constant(15 : ui4) : !firrtl.uint<4>
    ; CHECK: %a = firrtl.node [[CST15]]
    node a = UInt<4>(15)
    ; CHECK: %b = firrtl.node [[CST15]]
    node b = UInt<4>(15)

    ;; Constants always get emitted to the top level.
    ; CHECK: [[CST7:%.+]] = firrtl.constant(7 : ui4) : !firrtl.uint<4>
    ; CHECK: firrtl.when %cond {
    when cond :
      ; CHECK: %c = firrtl.node [[CST15]]
      node c = UInt<4>(15)
      ; CHECK: %d = firrtl.node [[CST7]]
      node d = UInt<4>(7)
      ; CHECK: firrtl.when %cond {
      when cond :
        ; CHECK:  %e = firrtl.node [[CST7]]
        node e = UInt<4>(7)
    ; CHECK: }
    ; CHECK: }

    ; CHECK:  %f = firrtl.node [[CST15]]
    node f = UInt<4>(15)
    node g = UInt<4>(7)

  ; CHECK-LABEL: firrtl.module @flip_one
  module flip_one :
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %0 = firrtl.subfield %bf("int_1")
    node _T = bf.int_1
    ; CHECK: %1 = firrtl.asPassive %0 : !firrtl.flip<uint<1>>
    when _T :  ; CHECK: firrtl.when %1 {
      skip

  ; CHECK-LABEL: firrtl.module @mem_depth_1
  module mem_depth_1 :
    input clock : Clock
    input reset : UInt<1>

    mem bar : @[Decoupled.scala 218:16]
      data-type => UInt<3>
      depth => 1
      read-latency => 0
      write-latency => 1
      reader => io_deq_bits_MPORT
      writer => MPORT
      read-under-write => undefined
      ; CHECK: %bar_MPORT, %bar_io_deq_bits_MPORT = firrtl.mem Undefined {depth = 1 : i64, name = "bar", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} :
      ; CHECK: !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<3>, mask: uint<1>>>,
      ; CHECK: !firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<3>>

  ; CHECK-LABEL: firrtl.module @issue354(%tmp5: !firrtl.flip<sint<19>>) {
  module issue354 :
    output tmp5: SInt<19>
    tmp5 <= SInt<19>(8)
     ; CHECK: %c8_si19 = firrtl.constant(8 : si19) : !firrtl.sint<19>
     ; CHECK: firrtl.connect %tmp5, %c8_si19 : !firrtl.flip<sint<19>>, !firrtl.sint<19>

   ; CHECK-LABEL: firrtl.module @issue347
  module issue347 :
    output tmp12: SInt<4>
    tmp12 <= SInt<4>(-4)
    ; CHECK: %c-4_si4 = firrtl.constant(-4 : si4) : !firrtl.sint<4>

  ; CHECK-LABEL: firrtl.extmodule @issue183()
  ; CHECK: attributes {parameters = {A = -1 : i4}}
  extmodule issue183:
     parameter A = -1

  ; COM: The Scala FIRRTL Compiler allows this for an aggregate node with an internal analog.
  ; CHECK-LABEL: firrtl.module @analog_in_aggregate_node
  module analog_in_aggregate_node:
    input a: { a: UInt<1>, b: Analog<1>}
    ; CHECK: %b = firrtl.node %a : !firrtl.bundle<a: uint<1>, b: analog<1>>
    node b = a

  ; COM: Check that a register clock sink is converted to passive
  ; CHECK-LABEL: firrtl.module @register_clock_passive
  module register_clock_passive:
    input clkIn: Clock
    output clkOut: Clock
    clkOut <= clkIn
    ; CHECK: [[clkOut_passive:%.+]] = firrtl.asPassive %clkOut
    ; CHECK: firrtl.reg [[clkOut_passive]]
    reg r: UInt<1>, clkOut

  ; COM: Check that a register reset sink is converted to passive
  ; CHECK-LABEL: firrtl.module @register_reset_passive
  module register_reset_passive:
    input clk: Clock
    output rst: UInt<1>
    rst is invalid
    ; CHECK: [[rst_passive:%.+]] = firrtl.asPassive %rst
    ; CHECK: firrtl.regreset %clk, [[rst_passive]]
    reg r: UInt<1>, clk with : (reset => (rst, UInt<1>(0)))

  ; COM: Check that a register init sink is converted to passive
  ; CHECK-LABEL: firrtl.module @register_init_passive
  module register_init_passive:
    input clk: Clock
    input rst: UInt<1>
    output init: UInt<1>
    init is invalid
    ; CHECK: [[init_passive:%.+]] = firrtl.asPassive %init
    ; CHECK: firrtl.regreset %clk, %rst, [[init_passive]]
    reg r: UInt<1>, clk with : (reset => (rst, init))
