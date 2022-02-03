; RUN: circt-translate -import-firrtl -verify-diagnostics %s | circt-opt | FileCheck %s

circuit MyModule :     ; CHECK: firrtl.circuit "MyModule" {

  ; CHECK-LABEL: firrtl.module @MyModule(in %in: !firrtl.uint, out %out: !firrtl.uint<8>) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>

    ; CHECK: firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint
    out <= in

  ; CHECK: }


  ; CHECK-LABEL: firrtl.module @circuit(in %in: !firrtl.uint<80>) {
  module circuit :    ; Module with a keyword id.
    input in: UInt<80>

  ; CHECK: }

  ; CHECK-LABEL: firrtl.extmodule @MyExtModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
  ; CHECK: attributes {defname = "myextmodule"}
  ; CHECK-NOT: {
  extmodule MyExtModule :
    input in: UInt
    output ,,, out ,,: ,, UInt,<,8,>  ; Commas are whitespace
    defname = myextmodule

  ; CHECK-LABEL: firrtl.extmodule @MyParameterizedExtModule
  ; CHECK-SAME:    <FORMAT: none = "xyz_timeout=%d\0A",
  ; CHECK-SAME:     DEFAULT: ui32 = 0,
  ; CHECK-SAME:     WIDTH: ui32 = 32,
  ; CHECK-SAME:     DEPTH: f64 = 3.242000e+01>
  ; CHECK-SAME:    (in in: !firrtl.uint,
  ; CHECK-SAME:     out out: !firrtl.uint<8>)
  ; CHECK-SAME:    attributes {defname = "name_thing"}
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
    ; CHECK: %bf: !firrtl.bundle<int_1 flip: uint<1>, int_out: uint<2>>

    input vec: UInt<1>[4] ; CHECK: %vec: !firrtl.vector<uint<1>, 4>) {


  ; CHECK-LABEL: firrtl.module @stmts(
  module stmts :
    input reset : UInt<1>         ; CHECK: in %reset: !firrtl.uint<1>,
    input reset_async: AsyncReset ; CHECK: in %reset_async: !firrtl.asyncreset,
    input reset_abstract: Reset   ; CHECK: in %reset_abstract: !firrtl.reset,
    input clock : Clock           ; CHECK: in %clock: !firrtl.clock,
    output auto : UInt<1>         ; CHECK: out %auto: !firrtl.uint<1>,
    output auto11 : UInt<11>      ; CHECK: out %auto11: !firrtl.uint<11>,
    output sauto : SInt<9>        ; CHECK: out %sauto: !firrtl.sint<9>,
    input i8 : UInt<8>            ; CHECK: in %i8: !firrtl.uint<8>,
    input s1 : SInt<1>            ; CHECK: in %s1: !firrtl.sint<1>,
    input s8 : SInt<8>            ; CHECK: in %s8: !firrtl.sint<8>,
    input a1 : Analog<1>          ; CHECK: in %a1: !firrtl.analog<1>,
    input a8 : Analog<8>          ; CHECK: in %a8: !firrtl.analog<8>)

    ; CHECK: %_t = firrtl.wire : !firrtl.vector<uint<1>, 12>
    wire _t : UInt<1>[12] @[Nodes.scala 370:76]

    ; CHECK: %_t_2 = firrtl.wire : !firrtl.vector<uint<1>, 12>
    wire _t_2 : UInt<1>[12]

    ; CHECK: firrtl.connect %_t, %_t_2 : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    _t <= _t_2

    ; CHECK: firrtl.partialconnect %_t, %_t_2
    _t <- _t_2

    ; CHECK: [[INV:%.+]]  = firrtl.invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: firrtl.connect %auto, [[INV]] : !firrtl.uint<1>, !firrtl.uint<1>
    auto is invalid

    ; CHECK-NOT: firrtl.attach %a1
    a1 is invalid

    ; CHECK: firrtl.skip
    skip  @[SKipLoc.scala 42:24]

    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: firrtl.connect %auto, [[INV]] : !firrtl.uint<1>, !firrtl.uint<1>
    auto is invalid

    ; CHECK-NOT: firrtl.connect %reset
    reset is invalid

    ; CHECK: %out_0 = firrtl.wire : !firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}

    ; CHECK: [[A:%.+]] = firrtl.subfield %out_0(0) : (!firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>) -> !firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>
    ; CHECK: [[B:%.+]] = firrtl.subfield [[A]](0) : (!firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>) -> !firrtl.bundle<clock: clock, reset: uint<1>>
    ; CHECK: [[C:%.+]] = firrtl.subfield [[B]](1) : (!firrtl.bundle<clock: clock, reset: uint<1>>) -> !firrtl.uint<1>
    ; CHECK: firrtl.partialconnect %auto, [[C]] : !firrtl.uint<1>, !firrtl.uint<1>
    auto <- out_0.member.0.reset @[Field 173:49]

    ; CHECK: %_t_3 = firrtl.wire : !firrtl.vector<uint<1>, 12>
    ; CHECK: [[A:%.+]] = firrtl.subindex %_t_3[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: %_t_4 = firrtl.wire : !firrtl.vector<uint<1>, 12>
    ; CHECK: [[B:%.+]] = firrtl.subindex %_t_4[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: firrtl.connect [[A]], [[B]]
    wire _t_3 : UInt<1>[12] @[Nodes.scala 370:76]
    wire _t_4 : UInt<1>[12]
    _t_3[0] <= _t_4[0] @[Xbar.scala 21:44]

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

    ; CHECK: firrtl.node %auto : !firrtl.uint<1>
    node check_output = auto

    ; CHECK: %c42_ui10 = firrtl.constant 42 : !firrtl.uint<10>
    ; CHECK: %c171_ui8 = firrtl.constant 171 : !firrtl.uint<8>
    ; CHECK: firrtl.add %c42_ui10, %c171_ui8
    ; CHECK: firrtl.connect %auto
    auto11 <= add(UInt<10>(42), UInt<8>("hAB"))

    ; CHECK: %c-85_si8 = firrtl.constant -85 : !firrtl.sint<8>
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

    ; CHECK: [[TMP:%.+]] = firrtl.constant 4
    ; CHECK: [[COND:%.+]] = firrtl.lt %reset, [[TMP]]
    ; CHECK: firrtl.when [[COND]] {
    ; CHECK:   firrtl.connect %_t, %_t_2
    ; CHECK: }
    ; CHECK-NOT: else
    when lt(reset, UInt(4)) :   ;; When with no else.
      _t <= _t_2

    ; CHECK: firrtl.when %reset  {
    ; CHECK:   firrtl.connect %_t, %_t_2
    ; CHECK: } else  {
    ; CHECK:   [[COND:%.+]] = firrtl.not %reset
    ; CHECK:   firrtl.when [[COND]]  {
    ; CHECK:     firrtl.connect %_t, %_t_2
    ; CHECK:   }
    ; CHECK: }
    when reset :
      _t <= _t_2
    else when not(reset) :
      _t <= _t_2

    ; CHECK: firrtl.when %reset  {
    ; CHECK:   firrtl.connect %_t, %_t
    ; CHECK: } else  {
    ; CHECK:   [[COND:%.+]] = firrtl.not %reset
    ; CHECK:   firrtl.when [[COND]]  {
    ; CHECK:     firrtl.connect %_t, %_t_2
    ; CHECK:   } else  {
    ; CHECK:     firrtl.connect %_t, %_t_2
    ; CHECK:   }
    ; CHECK: }
    when reset:
      _t <= _t_2
    else when not(reset) :
      _t <= _t_2
    else :
      _t <= _t_2

    ; CHECK: firrtl.printf %clock, %reset, "Something interesting!\0A %x %x" (%_t, %_t_2) : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    printf(clock, reset, "Something interesting!\n %x %x", _t, _t_2)

    ; CHECK: firrtl.printf %clock, %reset, "Something interesting!\0A %x %x" {name = "printf_0"}(%_t, %_t_2) : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    printf(clock, reset, "Something interesting!\n %x %x", _t, _t_2) : printf_0

    ; CHECK: firrtl.stop %clock, %reset, 42
    stop(clock, reset, 42)

    ; CHECK: firrtl.stop %clock, %reset, 42 {name = "stop_0"}
    stop(clock, reset, 42) : stop_0

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

    ; CHECK: %_t_2621 = firrtl.regreset %clock, %reset, %{{.*}} : !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
    reg _t_2621 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0")) @[Edges.scala 230:27]

    ; CHECK: %_t_1601 = firrtl.regreset %clock, %reset, %{{.*}} : !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>
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
    ; CHECK: %_t_2622 = firrtl.reg %clock : !firrtl.uint<4>
    reg _t_2622 : UInt<4>, clock with :
      reset => (UInt<1>("h0"), _t_2622)

    ; CHECK: %xyz_in = firrtl.instance xyz @circuit(in in: !firrtl.uint<80>)
    inst xyz of circuit
    ; CHECK: firrtl.connect %xyz_in, %i8 : !firrtl.uint<80>, !firrtl.uint<8>
    xyz.in <= i8

    ; CHECK: %myext_in, %myext_out = firrtl.instance myext @MyExtModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
    inst myext of MyExtModule
    myext.in <= i8
    printf(clock, reset, "Something interesting! %x", myext.out)

    ; CHECK: firrtl.when %reset  {
    when reset :
      ; CHECK: %reset_myext_in, %reset_myext_out = firrtl.instance reset_myext @MyExtModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
      inst reset_myext of MyExtModule
      reset_myext.in <= i8
    ; CHECK: }

    ; CHECK: firrtl.subaccess %_t[%i8] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<8>
    auto <= _t[i8]

    ; CHECK: firrtl.subaccess %_t[%auto] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<1>
    auto <= _t[auto]

    ; CHECK: %myMem = chirrtl.combmem : !chirrtl.cmemory<bundle<id: uint<4>, resp: uint<2>>, 8>
    cmem myMem : { id : UInt<4>, resp : UInt<2>} [8] @[Decoupled.scala 209:24]

    ; CHECK: %memValue_data, %memValue_port = chirrtl.memoryport Infer %myMem {name = "memValue"} : (!chirrtl.cmemory<bundle<id: uint<4>, resp: uint<2>>, 8>) -> (!firrtl.bundle<id: uint<4>, resp: uint<2>>, !chirrtl.cmemoryport)
    ; CHECK: chirrtl.memoryport.access %memValue_port[%i8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    infer mport memValue = myMem[i8], clock
    auto11 <= memValue.id

    ; CHECK: %base_table_0 = chirrtl.seqmem Undefined : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem base_table_0 : UInt<1>[9] [256]
    ; CHECK: %base_table_1 = chirrtl.seqmem Old : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem base_table_1 : UInt<1>[9] [256] old

    ; CHECK: %tableValue_data, %tableValue_port = chirrtl.memoryport Read %base_table_1 {name = "tableValue"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    ; CHECK: chirrtl.memoryport.access %tableValue_port[%i8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    read mport tableValue = base_table_1[i8], clock

    ; Check that we can handle large memory sizes.
    ; CHECK: %testharness = chirrtl.seqmem Undefined : !chirrtl.cmemory<vector<uint<8>, 16>, 2147483648>
    smem testharness : UInt<8>[16][2147483648], undefined

    ; CHECK: firrtl.pad %i8, 10 : (!firrtl.uint<8>) -> !firrtl.uint<10>
    node n11 = pad(i8, 10)

    ; CHECK: firrtl.andr %n11 : (!firrtl.uint<10>) -> !firrtl.uint<1>
    node n12 = andr(n11)

    ; CHECK: = firrtl.not %auto : (!firrtl.uint<1>) -> !firrtl.uint<1>
    node n13 = not(auto)


    ; CHECK: %_M__T_10, %_M__T_11, %_M__T_18 = firrtl.mem Undefined {depth = 8 : i64, name = "_M", portNames = ["_T_10", "_T_11", "_T_18"]
    ; CHECK-SAME: readLatency = 0 : i32, writeLatency = 1 : i32} :
    ; CHECK-SAME: !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>,
    ; CHECK-SAME: !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>,
    ; CHECK-SAME: !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<id: uint<4>>
    mem _M : @[Decoupled.scala 209:24]
        data-type => { id : UInt<4> }
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
    ; CHECK: firrtl.assert %clock, %pred, %en, "X equals Y when Z is valid"  {eventControl = 0 : i32, isConcurrent = false}
    assert(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.assert %clock, %pred, %en, "X equals Y when Z is valid"  {eventControl = 0 : i32, isConcurrent = false, name = "assert_0"}
    assert(clock, pred, en, "X equals Y when Z is valid") : assert_0
    ; CHECK: firrtl.assume %clock, %pred, %en, "X equals Y when Z is valid"  {eventControl = 0 : i32, isConcurrent = false}
    assume(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.assume %clock, %pred, %en, "X equals Y when Z is valid" {eventControl = 0 : i32, isConcurrent = false, name = "assume_0"}
    assume(clock, pred, en, "X equals Y when Z is valid") : assume_0
    ; CHECK: firrtl.cover %clock, %pred, %en, "X equals Y when Z is valid"  {eventControl = 0 : i32, isConcurrent = false}
    cover(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.cover %clock, %pred, %en, "X equals Y when Z is valid" {eventControl = 0 : i32, isConcurrent = false, name = "cover_0"}
    cover(clock, pred, en, "X equals Y when Z is valid") : cover_0

  ; CHECK-LABEL: firrtl.module @type_handling(
  module type_handling :
    wire _t_6 : { flip b : { bits : { source : UInt<7> } } }
    node _t_8 = bits(_t_6.b.bits.source, 5, 0)

    ; CHECK: %flip1 = firrtl.wire : !firrtl.bundle<x flip: bundle<a flip: uint>>
    wire flip1 : { flip x : { flip a : UInt } }
    ; CHECK: %flip2 = firrtl.wire : !firrtl.bundle<x flip: bundle<a flip: uint, b: analog>>
    wire flip2 : { flip x : { flip a : UInt, b: Analog } }
    ; CHECK: %flip3 = firrtl.wire : !firrtl.bundle<x flip: bundle<a flip: uint, b flip: analog>>
    wire flip3 : { flip x : { flip a : UInt, flip b: Analog } }
    ; CHECK: %flip4 = firrtl.wire : !firrtl.bundle<x flip: vector<bundle<a flip: uint>, 4>>
    wire flip4 : { flip x : { flip a : UInt }[4] }


  ; CHECK-LABEL: firrtl.module @expr_stmt_ambiguity(
  module expr_stmt_ambiguity :
    ; CHECK: %reg = firrtl.wire : !firrtl.uint
    wire reg : UInt
    ; CHECK: firrtl.connect %reg,
    reg <= UInt(42)

    ; CHECK: %write = firrtl.wire
    wire write : { id : UInt<4>, resp : UInt<2>}

    ; CHECK: firrtl.subfield %write(0)
    write.id <= UInt(1)

  ; CHECK-LABEL: firrtl.module @expr_stmt_ambiguity2(
  module expr_stmt_ambiguity2 :
    ; CHECK: firrtl.instance write @circuit
    inst write of circuit
    ; CHECK: firrtl.connect %write_in
    write.in <= UInt(1)

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
    input addr : UInt
    input clock : Clock

    ; Memory ports should be declared in the scope of the cmemory, but should
    ; be enabled at the location of the mport.

    ; CHECK: %memory = chirrtl.seqmem Undefined  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem memory : UInt<1>[9] [256]

    ; CHECK: %xyz0_data, %xyz0_port = chirrtl.memoryport Read %memory {name = "xyz0"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    ; CHECK: firrtl.when %cond  {
    ; CHECK:    chirrtl.memoryport.access %xyz0_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint, !firrtl.clock
    ; CHECK: }
    when cond :
      read mport xyz0 = memory[addr], clock

    ; CHECK: firrtl.when %cond  {
    ; CHECK:    %n0 = firrtl.node %xyz0_data  : !firrtl.vector<uint<1>, 9>
    ; CHECK: }
    when cond :
      node n0 = xyz0

    ; CHECK: %n1 = firrtl.node %xyz0_data  : !firrtl.vector<uint<1>, 9>
    node n1 = xyz0


  ; CHECK-LABEL: firrtl.module @constant_implicit_cse(
  module constant_implicit_cse :
    input cond : UInt<1>

    ; CHECK: [[CST15:%.+]] = firrtl.constant 15 : !firrtl.uint<4>
    ; CHECK: %a = firrtl.node [[CST15]]
    node a = UInt<4>(15)
    ; CHECK: %b = firrtl.node [[CST15]]
    node b = UInt<4>(15)

    ;; Constants always get emitted to the top level.
    ; CHECK: [[CST7:%.+]] = firrtl.constant 7 : !firrtl.uint<4>
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

  ; CHECK-LABEL: firrtl.module @subfield_implicit_cse
  module subfield_implicit_cse :
    input i: {x: UInt<1>}
    input cond: UInt<1>
    output o: UInt<1>

    ; Subfields always get emitted by their declarations.
    ; CHECK: [[SUB:%.+]] = firrtl.subfield %i(0)

    ; CHECK: %n3 = firrtl.node [[SUB]]
    node n3 = i.x

    ; CHECK: firrtl.when %cond {
    when cond:
      ; CHECK: %n4 = firrtl.node [[SUB]]
      node n4 = i.x
    ; CHECK: }

    ; Check that invalidation reuses subfields
    wire w: {a: UInt<1>}[1]
    ; CHECK: %1 = firrtl.subindex %w[0]
    ; CHECK: %2 = firrtl.subfield %1(0)
    ; CHECK: %invalid_ui1 = firrtl.invalidvalue
    ; CHECK: firrtl.connect %2, %invalid_ui1
    w is invalid
    ; CHECK: %invalid_ui1_0 = firrtl.invalidvalue
    ; CHECK: firrtl.connect %2, %invalid_ui1_0
    w is invalid

  ; CHECK-LABEL: firrtl.module @flip_one
  module flip_one :
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %0 = firrtl.subfield %bf(0)
    ; CHECK: %_T = firrtl.node %0
    node _T = bf.int_1
    ; CHECK: firrtl.when %_T {
    when _T :
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
      ; CHECK: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<3>, mask: uint<1>>,
      ; CHECK: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<3>>

  ; CHECK-LABEL: firrtl.module @mem_no_ports() {
  ; CHECK-NEXT: }
  ; https://github.com/llvm/circt/issues/531
  module mem_no_ports :
    mem bar : @[Decoupled.scala 218:16]
      data-type => UInt<3>
      depth => 1
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

  ; CHECK-LABEL: firrtl.module @issue354(out %tmp5: !firrtl.sint<19>) {
  module issue354 :
    output tmp5: SInt<19>
    tmp5 <= SInt<19>(8)
     ; CHECK: %c8_si19 = firrtl.constant 8 : !firrtl.sint<19>
     ; CHECK: firrtl.connect %tmp5, %c8_si19 : !firrtl.sint<19>, !firrtl.sint<19>

   ; CHECK-LABEL: firrtl.module @issue347
  module issue347 :
    output tmp12: SInt<4>
    tmp12 <= SInt<4>(-4)
    ; CHECK: %c-4_si4 = firrtl.constant -4 : !firrtl.sint<4>

  ; CHECK-LABEL: firrtl.extmodule @issue183<A: si32 = -1>()
  extmodule issue183:
     parameter A = -1

  ; The Scala FIRRTL Compiler allows this for an aggregate node with an internal
  ; analog.
  ; CHECK-LABEL: firrtl.module @analog_in_aggregate_node
  module analog_in_aggregate_node:
    input a: { a: UInt<1>, b: Analog<1>}
    ; CHECK: %b = firrtl.node %a : !firrtl.bundle<a: uint<1>, b: analog<1>>
    node b = a

  ; Check that a register clock sink is converted to passive
  ; CHECK-LABEL: firrtl.module @register_clock_passive
  module register_clock_passive:
    input clkIn: Clock
    output clkOut: Clock
    clkOut <= clkIn
    ; CHECK: firrtl.reg %clkOut
    reg r: UInt<1>, clkOut

  ; Check that a register reset sink is converted to passive
  ; CHECK-LABEL: firrtl.module @register_reset_passive
  module register_reset_passive:
    input clk: Clock
    output rst: UInt<1>
    rst is invalid
    ; CHECK: firrtl.regreset %clk, %rst
    reg r: UInt<1>, clk with : (reset => (rst, UInt<1>(0)))

  ; Check that a register init sink is converted to passive
  ; CHECK-LABEL: firrtl.module @register_init_passive
  module register_init_passive:
    input clk: Clock
    input rst: UInt<1>
    output init: UInt<1>
    init is invalid
    ; CHECK: firrtl.regreset %clk, %rst, %init
    reg r: UInt<1>, clk with : (reset => (rst, init))

  ; https://github.com/llvm/circt/issues/492
  ; CHECK-LABEL: firrtl.module @WriteOnlyMemIssue492
  module WriteOnlyMemIssue492 :
    input clock: Clock
    input wAddr: UInt<4>
    input wEn: UInt<1>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 16
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    memory.w.clk <= clock
    memory.w.en <= wEn
    memory.w.addr <= wAddr
    memory.w.mask <= wMask
    memory.w.data <= wData

  ; https://github.com/llvm/circt/issues/559
  ; CHECK-LABEL: firrtl.module @TrickyIssue559
  module TrickyIssue559:
    input input: UInt<1>
    output output: UInt<1>
    ; CHECK: firrtl.connect %output, %input
    output <= input

  ; CHECK-LABEL: firrtl.module @CheckInvalids
  module CheckInvalids_in0 :
    input in0 : UInt<1>
    ; CHECK-NOT: firrtl.connect %in0
    in0 is invalid

  module CheckInvalids_in1 :
    input in1 : { a : UInt<1>, b : UInt<1> }
    ; CHECK-NOT: firrtl.connect %in1
    in1 is invalid

  module CheckInvalids_in2 :
    input in2 : { a : UInt<1>, flip b : UInt<1>}
    ; CHECK: [[IN2_B:%.+]] = firrtl.subfield %in2(1)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[IN2_B]], [[INV]]
    in2 is invalid

  module CheckInvalids_in3 :
    input in3 : {a : { b : UInt<1>, flip c : UInt<1>}}
    ; CHECK: [[IN3_A:%.+]] = firrtl.subfield %in3(0)
    ; CHECK: [[IN3_A_C:%.+]] = firrtl.subfield [[IN3_A]](1)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[IN3_A_C]], [[INV]]
    in3 is invalid

  module CheckInvalids_out0 :
    output out0 : UInt<1>
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect %out0, [[INV]]
    out0 is invalid

  module CheckInvalids_out1 :
    output out1 : { a : UInt<1>, b : UInt<1> }
    ; CHECK: [[OUT1_B:%.+]] = firrtl.subfield %out1(1)
    ; CHECK: [[OUT1_A:%.+]] = firrtl.subfield %out1(0)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[OUT1_A]], [[INV]]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[OUT1_B]], [[INV]]
    out1 is invalid

  module CheckInvalids_out2 :
    output out2 : { a : UInt<1>, flip b : UInt<1>}
    ; CHECK: [[OUT2_A:%.+]] = firrtl.subfield %out2(0)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[OUT2_A]], [[INV]]
    out2 is invalid

  module CheckInvalids_out3 :
    output out3 : {a : { b : UInt<1>, flip c : UInt<1>}}
    ; CHECK: [[OUT3_A:%.+]] = firrtl.subfield %out3(0)
    ; CHECK: [[OUT3_A_B:%.+]] = firrtl.subfield [[OUT3_A]](0)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[OUT3_A_B]], [[INV]]
    out3 is invalid

  module CheckInvalids_wires :
    ; CHECK: %wire0 = firrtl.wire
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect %wire0, [[INV]]
    wire wire0 : UInt<1>
    wire0 is invalid

    ; CHECK: %wire1 = firrtl.wire
    ; CHECK: [[WIRE1_B:%.+]] = firrtl.subfield %wire1(1)
    ; CHECK: [[WIRE1_A:%.+]] = firrtl.subfield %wire1(0)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE1_A]], [[INV]]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE1_B]], [[INV]]
    wire wire1 : {a : UInt<1>, flip b : UInt<1> }
    wire1 is invalid

    ; An analog in the leaf of a wire should be attached not connected.
    ; CHECK: %wire2 = firrtl.wire
    ; CHECK: [[WIRE2_X:%.+]] = firrtl.subfield %wire2(0)
    ; CHECK: [[WIRE2_X_B:%.+]] = firrtl.subfield [[WIRE2_X]](1)
    ; CHECK: [[WIRE2_X_A:%.+]] = firrtl.subfield [[WIRE2_X]](0)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[WIRE2_X_A]], [[INV]]
    ; CHECK-NOT: firrtl.attach [[WIRE2_X_B]], [[INV]]
    wire wire2 : {x : {flip a : UInt<1>, flip b: Analog<1> } }
    wire2 is invalid

    ; https://github.com/llvm/circt/issues/563
    ; CHECK: %U0_in0, %U0_in1, %U0_out0, %U0_out1 = firrtl.instance U0 @mod_0_563
    inst U0 of mod_0_563

    ; CHECK: [[U0_IN1_A:%.+]] = firrtl.subfield %U0_in1(0)
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect %U0_in0, [[INV]]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.connect [[U0_IN1_A]], [[INV]]
    U0 is invalid

  ; This reference is declared after its first use.
  ; https://github.com/llvm/circt/issues/163
  module mod_0_563 :
    input in0: UInt<5>
    input in1: { a : UInt<5> }
    output out0: UInt<5>
    output out1: { a : UInt<5> }
    out0 <= in0
    out1 <= in1

  ; https://github.com/llvm/circt/issues/606
  ; CHECK-LABEL: firrtl.module @mutableSubIndex606
  module mutableSubIndex606 :
    output io : UInt<1>[8]
    ; CHECK:  %0 = firrtl.subindex %io[0] : !firrtl.vector<uint<1>, 8>
    ; CHECK: firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    io[0] <= UInt<1>("h00")


  ; https://github.com/llvm/circt/issues/782
  ; CHECK-LABEL: mem_madness782
  module mem_madness782:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>

    ; CHECK: %mem_r = firrtl.mem Undefined {depth = 16 : i64, name = "mem", portNames = ["r"], readLatency = 2 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    mem mem:
      data-type => UInt<8>
      depth => 16
      reader => r
      read-latency => 2
      write-latency => 1
      read-under-write => undefined

    mem.r.clk <= clock
    mem.r.en <= rEn
    mem.r.addr <= rAddr
    rData <= mem.r.data

  ; https://github.com/llvm/circt/issues/886
  ; CHECK-LABEL: ParseValidifAsMux
  module ParseValidifAsMux:
    input in1: Clock
    input in2: UInt<42>
    input en: UInt<1>

    ; CHECK: %out1 = firrtl.node %in1
    node out1 = validif(en, in1)

    ; CHECK: %out2 = firrtl.node %in2
    node out2 = validif(en, in2)

    ; This is required for SFC compatibility.
    ; https://github.com/llvm/circt/issues/1186
    ; CHECK: %out3 = firrtl.node %in1
    node out3 = validif(UInt<1>(0), in1)

  ; Test that behavioral memory reads and writes both work and that flow checks
  ; don't fail here.  (A memory port should have duplex flow.)
  ; See: https://github.com/llvm/circt/issues/1058
  ; CHECK-LABEL: firrtl.module @BehavioralMemory
  module BehavioralMemory:
    input clock: Clock
    input rAddr: UInt<3>
    output rData: UInt<1>
    input wAddr: UInt<3>
    input wData: UInt<1>

    cmem a: UInt<1>[8]

    ; CHECK: firrtl.connect %rData, %r
    infer mport r = a[rAddr], clock
    rData <= r

    ; CHECK: firrtl.connect %w_data, %wData
    infer mport w = a[wAddr], clock
    w <= wData

  ; Test that a mux with an unknown width select line parses.  This is a check
  ; of the predicate enforced on UInt1Type.
  ; CHECK-LABEL: firrtl.module @MuxUnknownWidthSelect_Issue1108
  module MuxUnknownWidthSelect_Issue1108:
    input a: UInt<1>
    input b: UInt<1>
    input sel: UInt
    output c: UInt<8>
    c <= mux(sel, a, b)

  ; CHECK-LABEL: firrtl.extmodule @RawStringParam
  ; CHECK-SAME:    <TYPE: none = "bit",
  ; CHECK-SAME:     FORMAT: none = "xyz_timeout=%d\\n",
  ; CHECK-SAME:     MIXED_QUOTES: none = "\22'\\\22">
  extmodule RawStringParam :
    parameter TYPE = 'bit'
    parameter FORMAT = 'xyz_timeout=%d\n'
    parameter MIXED_QUOTES = '"\'\"'
  ; "

  ; CHECK-LABEL: firrtl.module @issue1303
  module issue1303:
    output out: Reset
    out <= UInt(1)
    ; CHECK: %[[c1:.*]] = firrtl.constant 1 : !firrtl.uint
    ; CHECK-NEXT: firrtl.connect %out, %[[c1]] : !firrtl.reset, !firrtl.uint


  ; CHECK-LABEL: @resetBundle
  module resetBundle:
    input a: {a: UInt<1>, b: AsyncReset}
    output b: {a: Reset, b: Reset}

    b <= a
    ; CHECK: firrtl.connect %b, %a : !firrtl.bundle<a: reset, b: reset>, !firrtl.bundle<a: uint<1>, b: asyncreset>

  ; CHECK-LABEL: @WhenEncodedVerification
  module WhenEncodedVerification:
    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>
    input not_reset: UInt<1>
    input value: UInt<42>

    ; rocket-chip properties
    when cond:
      printf(clock, enable, "assert:foo 0", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "foo 0" {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "assume:foo 1", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "foo 1" {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "cover:foo 2", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.cover %clock, [[TMP]], %enable, "foo 2" {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "assert:custom label 0:foo 3", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "foo 3" {eventControl = 0 : i32, isConcurrent = true, name = "custom label 0"}

    when cond:
      printf(clock, enable, "assume:custom label 1:foo 4", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "foo 4" {eventControl = 0 : i32, isConcurrent = true, name = "custom label 1"}

    when cond:
      printf(clock, enable, "cover:custom label 2:foo 5", value)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.cover %clock, [[TMP]], %enable, "foo 5" {eventControl = 0 : i32, isConcurrent = true, name = "custom label 2"}

    ; Optional `stop` with same clock and condition should be removed.
    when cond:
      printf(clock, enable, "assert:without_stop")
      stop(clock, enable, 1)
    ; CHECK: firrtl.assert %clock, {{%.+}}, %enable, "without_stop"
    ; CHECK-NOT: firrtl.stop

    when cond:
      ; expected-warning @+1 {{printf-encoded assertion has format string arguments which may cause lint warnings}}
      printf(clock, enable, "assert:foo 6, %d", value, value)
    ; CHECK: firrtl.assert {{.+}} "foo 6, %d"(%value)

    ; AssertNotX -- usually `cond` only checks for not-in-reset, and `enable` is
    ; just set to 1; the actual check `^value !== 'x` is implicit.
    when cond:
      printf(clock, enable, "assertNotX:%d:value must not be X!", value)
    ; CHECK: [[TMP1:%.+]] = firrtl.not %cond
    ; CHECK: [[TMP2:%.+]] = firrtl.xorr %value
    ; CHECK: [[TMP3:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}} !== 1'bx"([[TMP2]])
    ; CHECK: [[TMP4:%.+]] = firrtl.or [[TMP1]], [[TMP3]]
    ; CHECK: firrtl.assert %clock, [[TMP4]], %enable, "value must not be X!"
    ; CHECK-SAME: name = "notX"

    ; Chisel built-in assertions
    when cond:
      printf(clock, enable, "Assertion failed with value %d", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed with value %d"(%value) : !firrtl.uint<42>
    ; CHECK-SAME: isConcurrent = false
    ; CHECK-SAME: name = "chisel3_builtin"

    when cond:
      printf(clock, enable, "Assertion failed: some message with value %d", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed: some message with value %d"(%value) : !firrtl.uint<42>
    ; CHECK-SAME: isConcurrent = false
    ; CHECK-SAME: name = "chisel3_builtin"

    ; Verification Library Assertions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Hello Assert"(%value) : !firrtl.uint<42>
    ; CHECK-SAME: format = "sva"
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_magic"

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP1:%.+]] = firrtl.xorr %cond
    ; CHECK-NEXT: [[TMP2:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}} === 1'bx"([[TMP1]])
    ; CHECK-NEXT: [[TMP3:%.+]] = firrtl.or %cond, [[TMP2]]
    ; CHECK-NEXT: [[TMP:%.+]] = firrtl.not [[TMP3]]
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Hello Assert"(%value) : !firrtl.uint<42>
    ; CHECK-SAME: format = "sva"
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_magic"

    ; Verification Library Assumptions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "Hello Assume"(%value) : !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_voodoo"

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP1:%.+]] = firrtl.xorr %cond
    ; CHECK-NEXT: [[TMP2:%.+]] = firrtl.verbatim.expr "{{[{][{]0[}][}]}} === 1'bx"([[TMP1]])
    ; CHECK-NEXT: [[TMP3:%.+]] = firrtl.or %cond, [[TMP2]]
    ; CHECK-NEXT: [[TMP:%.+]] = firrtl.not [[TMP3]]
    ; CHECK-NEXT: firrtl.assume %clock, [[TMP]], %enable, "Hello Assume"(%value) : !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_voodoo"

    ; New flavor of when-encoded verification that also includes an assert
    assert(clock, cond, enable, "hello")
    node not_cond = eq(cond, UInt<1>(0))
    when not_cond:
      printf(clock, enable, "Assertion failed: hello")
    ; CHECK-NOT: firrtl.assert %clock, %cond, %enable, "hello"
    ; CHECK: [[TMP:%.+]] = firrtl.not %not_cond
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed: hello"
    ; CHECK-SAME: name = "chisel3_builtin"

    when not_reset:
      assert(clock, cond, enable, "hello outside reset")
      node not_cond2 = eq(cond, UInt<1>(0))
      when not_cond2:
        printf(clock, enable, "Assertion failed: hello outside reset")
    ; CHECK: firrtl.when %not_reset {
    ; CHECK-NOT: firrtl.assert %clock, %cond, %enable, "hello outside reset"
    ; CHECK: [[TMP:%.+]] = firrtl.not %not_cond2
    ; CHECK-NEXT: firrtl.assert %clock, [[TMP]], %enable, "Assertion failed: hello outside reset"
    ; CHECK-SAME: name = "chisel3_builtin"
    ; CHECK: }

    ; Check that the above doesn't error if the assert is a double user of the
    ; condition.
    when not_reset:
      assert(clock, UInt<1>(1), UInt<1>(1), "double user assert")
      node not_cond3 = eq(UInt<1>(1), UInt<1>(0))
      when not_cond3:
        printf(clock, UInt<1>(1), "Assertion failed: double user assert")
    ; CHECK-NOT: firrtl.assert %clock, {{.+}} "double user assert"

    when cond :
        printf(clock, not(enable), "assert: bar")
    ; CHECK: [[NOT_ENABLE:%.+]] = firrtl.not %enable
    ; CHECK-NEXT: [[NOT_COND:%.+]] = firrtl.not %cond
    ; CHECK-NEXT: firrtl.assert %clock, [[NOT_COND]], [[NOT_ENABLE]]
