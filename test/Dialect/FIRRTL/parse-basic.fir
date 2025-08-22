; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s | circt-opt | FileCheck %s

FIRRTL version 5.1.0
circuit MyModule :     ; CHECK: firrtl.circuit "MyModule"

  ; CHECK-LABEL: firrtl.module @MyModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
  public module MyModule :   @[FooBar.scala 369:27]
    input in: UInt<8>
    output out: UInt<8>

    ; CHECK: firrtl.matchingconnect %out, %in : !firrtl.uint<8>
    connect out, in

  ; CHECK: }


  ; CHECK-LABEL: firrtl.module private @circuit(in %in: !firrtl.uint<80>) {
  module circuit :    ; Module with a keyword id.
    input in: UInt<80>

  ; CHECK: }

  ; CHECK-LABEL: firrtl.extmodule private @MyExtModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
  ; CHECK-SAME: attributes {defname = "myextmodule"}
  ; CHECK-NOT: {
  extmodule MyExtModule :
    input in: UInt<8>
    output out : UInt<8>
    defname = myextmodule

  ; CHECK-LABEL: firrtl.extmodule private @MyParameterizedExtModule
  ; CHECK-SAME:    <FORMAT: none = "xyz_timeout=%d\0A",
  ; CHECK-SAME:     DEFAULT: ui32 = 0,
  ; CHECK-SAME:     WIDTH: ui32 = 32,
  ; CHECK-SAME:     DEPTH: f64 = 3.242000e+01>
  ; CHECK-SAME:    (in in: !firrtl.uint<8>,
  ; CHECK-SAME:     out out: !firrtl.uint<8>)
  ; CHECK-SAME:    attributes {defname = "name_thing"}
  ; CHECK-NOT: {
  extmodule MyParameterizedExtModule :
    input in: UInt<8>
    output out: UInt<8>
    defname = name_thing
    parameter FORMAT = "xyz_timeout=%d\n"
    parameter DEFAULT = 0
    parameter WIDTH = 32
    parameter DEPTH = 32.42

  ; Check that integers are extended to 32 bits if they are smaller.
  ; CHECK-LABEL: firrtl.extmodule private @IntegerParamsModule
  extmodule IntegerParamsModule :
    ; CHECK-SAME: a: ui32 = 1
    parameter a = 1
    ; CHECK-SAME: b: ui40 = 4294967296
    parameter b = 4294967296

  ; Module to test type parsing.

  ; CHECK-LABEL: firrtl.module private @types(
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


  ; CHECK-LABEL: firrtl.module private @stmts(
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
    input a8 : Analog<8>          ; CHECK: in %a8: !firrtl.analog<8>,
    input ab : {x : Analog<1>}    ; CHECK: in %ab: !firrtl.bundle<x: analog<1>>)

    ; CHECK: %_t = firrtl.wire interesting_name : !firrtl.vector<uint<1>, 12>
    wire _t : UInt<1>[12] @[Nodes.scala 370:76]

    ; CHECK: %_t_2 = firrtl.wire interesting_name : !firrtl.vector<uint<1>, 12>
    wire _t_2 : UInt<1>[12]

    ; CHECK: firrtl.matchingconnect %_t, %_t_2 : !firrtl.vector<uint<1>, 12>
    connect _t, _t_2

    ; CHECK: [[INV:%.+]]  = firrtl.invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: firrtl.matchingconnect %auto, [[INV]] : !firrtl.uint<1>
    invalidate auto

    ; CHECK-NOT: firrtl.attach %a1
    invalidate a1

    ; CHECK-NOT: firrtl.attach %ab
    invalidate ab

    ; CHECK: firrtl.skip
    skip  @[SKipLoc.scala 42:24]

    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: firrtl.matchingconnect %auto, [[INV]] : !firrtl.uint<1>
    invalidate auto

    ; CHECK-NOT: firrtl.connect %reset
    ; CHECK-NOT: firrtl.matchingconnect %reset
    invalidate reset

    ; CHECK: %out_0 = firrtl.wire interesting_name : !firrtl.bundle<member: bundle<"0": bundle<clock: clock, reset: uint<1>>>>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}

    ; CHECK: %_t_3 = firrtl.wire interesting_name : !firrtl.vector<uint<1>, 12>
    ; CHECK: [[A:%.+]] = firrtl.subindex %_t_3[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: %_t_4 = firrtl.wire interesting_name : !firrtl.vector<uint<1>, 12>
    ; CHECK: [[B:%.+]] = firrtl.subindex %_t_4[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: firrtl.matchingconnect [[A]], [[B]]
    wire _t_3 : UInt<1>[12] @[Nodes.scala 370:76]
    wire _t_4 : UInt<1>[12]
    connect _t_3[0], _t_4[0] @[Xbar.scala 21:44]

    ; CHECK: %n1 = firrtl.node interesting_name %i8 : !firrtl.uint<8>
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

    ; CHECK: firrtl.node interesting_name %auto : !firrtl.uint<1>
    node check_output = auto

    ; CHECK: %c42_ui10 = firrtl.constant 42 : !firrtl.const.uint<10>
    ; CHECK: %c171_ui8 = firrtl.constant 171 : !firrtl.const.uint<8>
    ; CHECK: firrtl.add %c42_ui10, %c171_ui8
    ; CHECK: firrtl.constCast
    ; CHECK: firrtl.matchingconnect %auto
    connect auto11, add(UInt<10>(42), UInt<8>(0hAB))

    ; CHECK: %c-85_si8 = firrtl.constant -85 : !firrtl.const.sint<8>
    connect sauto, add(s8, SInt<8>(-85))

    ; CHECK: firrtl.when %reset : !firrtl.uint<1> {
    ; CHECK:   firrtl.matchingconnect %_t, %_t_2
    ; CHECK: } else {
    ; CHECK:   firrtl.matchingconnect %_t, %_t_2
    ; CHECK: }
    when reset : connect _t, _t_2 else : connect _t, _t_2

    ; CHECK: [[TMP:%.+]] = firrtl.constant 4
    ; CHECK: [[COND:%.+]] = firrtl.lt %reset, [[TMP]]
    ; CHECK: firrtl.when [[COND]] : !firrtl.uint<1> {
    ; CHECK:   firrtl.matchingconnect %_t, %_t_2
    ; CHECK: }
    ; CHECK-NOT: else
    when lt(reset, UInt(4)) :   ;; When with no else.
      connect _t, _t_2

    ; CHECK: firrtl.when %reset : !firrtl.uint<1> {
    ; CHECK:   firrtl.matchingconnect %_t, %_t_2
    ; CHECK: } else  {
    ; CHECK:   [[COND:%.+]] = firrtl.not %reset
    ; CHECK:   firrtl.when [[COND]] : !firrtl.uint<1> {
    ; CHECK:     firrtl.matchingconnect %_t, %_t_2
    ; CHECK:   }
    ; CHECK: }
    when reset :
      connect _t, _t_2
    else when not(reset) :
      connect _t, _t_2

    ; CHECK: firrtl.when %reset : !firrtl.uint<1> {
    ; CHECK:   firrtl.matchingconnect %_t, %_t
    ; CHECK: } else  {
    ; CHECK:   [[COND:%.+]] = firrtl.not %reset
    ; CHECK:   firrtl.when [[COND]] : !firrtl.uint<1> {
    ; CHECK:     firrtl.matchingconnect %_t, %_t_2
    ; CHECK:   } else  {
    ; CHECK:     firrtl.matchingconnect %_t, %_t_2
    ; CHECK:   }
    ; CHECK: }
    when reset:
      connect _t, _t_2
    else when not(reset) :
      connect _t, _t_2
    else :
      connect _t, _t_2

    ; CHECK: firrtl.printf %clock, %reset, "Something interesting!\0A %x %x" (%_t, %_t_2) : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    printf(clock, reset, "Something interesting!\n %x %x", _t, _t_2)

    ; CHECK: firrtl.printf %clock, %reset, "Something interesting!\0A %x %x" {name = "printf_0"} (%_t, %_t_2) : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    printf(clock, reset, "Something interesting!\n %x %x", _t, _t_2) : printf_0

    ; CHECK: firrtl.stop %clock, %reset, 42 : !firrtl.clock, !firrtl.uint<1>
    stop(clock, reset, 42)

    ; CHECK: firrtl.stop %clock, %reset, 42 {name = "stop_0"} : !firrtl.clock, !firrtl.uint<1>
    stop(clock, reset, 42) : stop_0

    ; CHECK: firrtl.bits %i8 4 to 2 : (!firrtl.uint<8>) -> !firrtl.uint<3>
    node n4 = bits(i8, 4, 2)

    ; CHECK: firrtl.shl %i8, 4 : (!firrtl.uint<8>) -> !firrtl.uint<12>
    ; CHECK: firrtl.shr %i8, 8 : (!firrtl.uint<8>) -> !firrtl.uint<0>
    node n5 = or(shl(i8, 4), shr(i8, 8))

    ; CHECK: firrtl.dshl %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.const.uint<4>) -> !firrtl.uint<23>
    node n6 = dshl(i8, UInt<4>(7))
    ; CHECK: firrtl.dshlw %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.const.uint<4>) -> !firrtl.uint<8>
    node n6s = dshlw(i8, UInt<4>(7))

    ; CHECK: firrtl.cat %{{.*}}, %{{.*}}, %{{.*}} : (!firrtl.uint<12>, !firrtl.uint<23>, !firrtl.uint<23>) -> !firrtl.uint<58>
    node n7 = cat(n5, n6, n6)

    ; CHECK: firrtl.mux(%reset, %i8, %{{.*}}) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.const.uint) -> !firrtl.uint
    node n8 = mux(reset, i8, UInt(4))

    ; CHECK: %_t_2621 = firrtl.regreset interesting_name %clock, %reset, %{{.*}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.const.uint<4>, !firrtl.uint<4>
    regreset _t_2621 : UInt<4>, clock, reset, UInt<4>(0h0) @[Edges.scala 230:27]

    ; CHECK: %_t_1601 = firrtl.regreset interesting_name %clock, %reset, %{{.*}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.const.uint<2>, !firrtl.uint<2>
    regreset _t_1601 : UInt<2>, clock, reset, UInt<2>(0h00) @[Edges.scala 230:27]

    ; CHECK: firrtl.div %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.const.uint<4>) -> !firrtl.uint<8>
    node n9 = div(i8, UInt<4>(4))

    ; CHECK: firrtl.tail %i8, 7 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    ; CHECK: firrtl.tail %i8, 0 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    ; CHECK: firrtl.head %i8, 4 : (!firrtl.uint<8>) -> !firrtl.uint<4>
    node n10 = add(add(tail(i8, 7), tail(i8, 0)), head(i8, 4))

    ; CHECK: firrtl.tail %{{.*}}, 3 : (!firrtl.sint<8>) -> !firrtl.uint<5>
    node n10s = tail(asSInt(i8), 3)

    ; CHECK: %_t_2622 = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<4>
    reg _t_2622 : UInt<4>, clock

    ; CHECK: %xyz_in = firrtl.instance xyz interesting_name @circuit(in in: !firrtl.uint<80>)
    inst xyz of circuit
    ; CHECK: [[PAD:%.*]] = firrtl.pad %i8, 80 : (!firrtl.uint<8>) -> !firrtl.uint<80>
    ; CHECK: firrtl.matchingconnect %xyz_in, [[PAD]] : !firrtl.uint<80>
    connect xyz.in, i8

    ; CHECK: %myext_in, %myext_out = firrtl.instance myext interesting_name @MyExtModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    inst myext of MyExtModule
    connect myext.in, i8
    printf(clock, reset, "Something interesting! %x", myext.out)

    ; CHECK: firrtl.when %reset : !firrtl.uint<1> {
    when reset :
      ; CHECK: %reset_myext_in, %reset_myext_out = firrtl.instance reset_myext interesting_name @MyExtModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
      inst reset_myext of MyExtModule
      connect reset_myext.in, i8
    ; CHECK: }

    ; CHECK: firrtl.subaccess %_t[%i8] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<8>
    connect auto, _t[i8]

    ; CHECK: firrtl.subaccess %_t[%auto] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<1>
    connect auto, _t[auto]

    ; CHECK: %myMem = chirrtl.combmem interesting_name : !chirrtl.cmemory<bundle<id: uint<4>, resp: uint<2>>, 8>
    cmem myMem : { id : UInt<4>, resp : UInt<2>} [8] @[Decoupled.scala 209:24]

    ; CHECK: %memValue_data, %memValue_port = chirrtl.memoryport Infer %myMem {name = "memValue"} : (!chirrtl.cmemory<bundle<id: uint<4>, resp: uint<2>>, 8>) -> (!firrtl.bundle<id: uint<4>, resp: uint<2>>, !chirrtl.cmemoryport)
    ; CHECK: chirrtl.memoryport.access %memValue_port[%i8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    infer mport memValue = myMem[i8], clock
    connect auto11, memValue.id

    ; CHECK: %base_table_0 = chirrtl.seqmem interesting_name Undefined : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem base_table_0 : UInt<1>[9] [256]
    ; CHECK: %base_table_1 = chirrtl.seqmem interesting_name Old : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem base_table_1 : UInt<1>[9] [256], old

    ; CHECK: %tableValue_data, %tableValue_port = chirrtl.memoryport Read %base_table_1 {name = "tableValue"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    ; CHECK: chirrtl.memoryport.access %tableValue_port[%i8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    read mport tableValue = base_table_1[i8], clock

    ; Check that we can handle large memory sizes.
    ; CHECK: %testharness = chirrtl.seqmem interesting_name Undefined : !chirrtl.cmemory<vector<uint<8>, 16>, 2147483648>
    smem testharness : UInt<8>[16][2147483648], undefined

    ; CHECK: firrtl.pad %i8, 10 : (!firrtl.uint<8>) -> !firrtl.uint<10>
    node n11 = pad(i8, 10)

    ; CHECK: firrtl.andr %n11 : (!firrtl.uint<10>) -> !firrtl.uint<1>
    node n12 = andr(n11)

    ; CHECK: = firrtl.not %auto : (!firrtl.uint<1>) -> !firrtl.uint<1>
    node n13 = not(auto)


    ; CHECK: %_M__T_10, %_M__T_11, %_M__T_18 = firrtl.mem interesting_name Undefined {depth = 8 : i64, name = "_M", portNames = ["_T_10", "_T_11", "_T_18"]
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
    invalidate _M._T_18.addr  @[Decoupled.scala 209:24]
    invalidate _M._T_18.clk  @[Decoupled.scala 209:24]
    connect _M._T_18.en, UInt<1>(0h0) @[Decoupled.scala 209:24]
    invalidate _M._T_10.addr  @[Decoupled.scala 209:24]
    invalidate _M._T_10.clk  @[Decoupled.scala 209:24]
    connect _M._T_10.en, UInt<1>(0h0) @[Decoupled.scala 209:24]
    invalidate _M._T_10.data  @[Decoupled.scala 209:24]
    invalidate _M._T_10.mask  @[Decoupled.scala 209:24]

    ; CHECK: firrtl.attach %a8, %a8, %a8 :
    attach (a8, a8, a8)

    wire pred: UInt <1>
    wire en: UInt <1>
    connect pred, eq(i8, i8)
    connect en, not(reset)
    ; CHECK: firrtl.assert %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
    assert(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.assert %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "assert_0"}
    assert(clock, pred, en, "X equals Y when Z is valid") : assert_0
    ; CHECK: firrtl.assert %clock, %pred, %en, "pred=%d, en=%d"(%pred, %en) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "assert_1"}
    assert(clock, pred, en, "pred=%d, en=%d", pred, en) : assert_1
    ; CHECK: firrtl.assume %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
    assume(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.assume %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "assume_0"}
    assume(clock, pred, en, "X equals Y when Z is valid") : assume_0
    ; CHECK: firrtl.assume %clock, %pred, %en, "pred=%d, en=%d"(%pred, %en) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "assume_1"}
    assume(clock, pred, en, "pred=%d, en=%d", pred, en) : assume_1
    ; CHECK: firrtl.cover %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
    cover(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: firrtl.cover %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "cover_0"}
    cover(clock, pred, en, "X equals Y when Z is valid") : cover_0

  ; CHECK-LABEL: firrtl.module private @type_handling(
  module type_handling :
    wire _t_6 : { flip b : { bits : { source : UInt<7> } } }
    node _t_8 = bits(_t_6.b.bits.source, 5, 0)

    ; CHECK: %flip1 = firrtl.wire interesting_name : !firrtl.bundle<x flip: bundle<a flip: uint>>
    wire flip1 : { flip x : { flip a : UInt } }
    ; CHECK: %flip2 = firrtl.wire interesting_name : !firrtl.bundle<x flip: bundle<a flip: uint, b: analog>>
    wire flip2 : { flip x : { flip a : UInt, b: Analog } }
    ; CHECK: %flip3 = firrtl.wire interesting_name : !firrtl.bundle<x flip: bundle<a flip: uint, b flip: analog>>
    wire flip3 : { flip x : { flip a : UInt, flip b: Analog } }
    ; CHECK: %flip4 = firrtl.wire interesting_name : !firrtl.bundle<x flip: vector<bundle<a flip: uint>, 4>>
    wire flip4 : { flip x : { flip a : UInt }[4] }


  ; CHECK-LABEL: firrtl.module private @expr_stmt_ambiguity(
  module expr_stmt_ambiguity :
    ; CHECK: %reg = firrtl.wire interesting_name : !firrtl.uint
    wire reg : UInt
    ; CHECK: firrtl.connect %reg,
    connect reg, UInt(42)

    ; CHECK: %write = firrtl.wire
    wire write : { id : UInt<4>, resp : UInt<2>}

    ; CHECK: firrtl.subfield %write[id]
    connect write.id, UInt(1)

  ; CHECK-LABEL: firrtl.module private @expr_stmt_ambiguity2(
  module expr_stmt_ambiguity2 :
    ; CHECK: firrtl.instance write interesting_name @circuit
    inst write of circuit
    ; CHECK: firrtl.connect %write_in
    connect write.in, UInt(1)

  ; CHECK-LABEL: firrtl.module private @oversize_shift(
  module oversize_shift :
    wire value : UInt<2>
    ; CHECK: firrtl.shr %value, 5 : (!firrtl.uint<2>) -> !firrtl.uint<0>
    node n = shr(value, 5)

  ; CHECK-LABEL: firrtl.module private @when_else_ambiguity(
  module when_else_ambiguity :
    output out : UInt
    input in : UInt
    wire reset : UInt<1>

  ; CHECK: firrtl.when {{.*}} : !firrtl.uint<1> {
    when reset : @[Debug.scala 1176:37]
    ; CHECK: firrtl.when {{.*}} : !firrtl.uint<1> {
      when reset :
        connect out, in
    ; CHECK: }
    ; CHECK: } else {
    else :
        ; CHECK: firrtl.when {{.*}} : !firrtl.uint<1> {
      when reset : @[Debug.scala 1180:39]
        connect out, in
    ; CHECK: }
    ; CHECK: }


  ; CHECK-LABEL: firrtl.module private @chisel_when_mport_bug(
  module chisel_when_mport_bug :
    input cond : UInt<1>
    input addr : UInt
    input clock : Clock

    ; Memory ports should be declared in the scope of the cmemory, but should
    ; be enabled at the location of the mport.

    ; CHECK: %memory = chirrtl.seqmem interesting_name Undefined  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem memory : UInt<1>[9] [256]

    ; CHECK: %xyz0_data, %xyz0_port = chirrtl.memoryport Read %memory {name = "xyz0"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    ; CHECK: firrtl.when %cond : !firrtl.uint<1> {
    ; CHECK:    chirrtl.memoryport.access %xyz0_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint, !firrtl.clock
    ; CHECK: }
    when cond :
      read mport xyz0 = memory[addr], clock

    ; CHECK: firrtl.when %cond : !firrtl.uint<1> {
    ; CHECK:    %n0 = firrtl.node interesting_name %xyz0_data  : !firrtl.vector<uint<1>, 9>
    ; CHECK: }
    when cond :
      node n0 = xyz0

    ; CHECK: %n1 = firrtl.node interesting_name %xyz0_data  : !firrtl.vector<uint<1>, 9>
    node n1 = xyz0


  ; CHECK-LABEL: firrtl.module private @constant_implicit_cse(
  module constant_implicit_cse :
    input cond : UInt<1>
    input enum : {|A|}

    ; CHECK: [[CST15:%.+]] = firrtl.constant 15 : !firrtl.const.uint<4>
    ; CHECK: %a = firrtl.node interesting_name [[CST15]]
    node a = UInt<4>(15)
    ; CHECK: %b = firrtl.node interesting_name [[CST15]]
    node b = UInt<4>(15)

    ;; Constants always get emitted to the top level.
    ; CHECK: [[CST7:%.+]] = firrtl.constant 7 : !firrtl.const.uint<4>
    ; CHECK: firrtl.when %cond : !firrtl.uint<1> {
    when cond :
      ; CHECK: %c = firrtl.node interesting_name [[CST15]]
      node c = UInt<4>(15)
      ; CHECK: %d = firrtl.node interesting_name [[CST7]]
      node d = UInt<4>(7)
      ; CHECK: firrtl.when %cond : !firrtl.uint<1> {
      when cond :
        ; CHECK:  %e = firrtl.node interesting_name [[CST7]]
        node e = UInt<4>(7)
    ; CHECK: }
    ; CHECK: }

    ; CHECK:  %f = firrtl.node interesting_name [[CST15]]
    node f = UInt<4>(15)
    node g = UInt<4>(7)

    ; Should be able to use the constant cache under a match without crashing.
    ; CHECK: [[CST8:%.+]] = firrtl.constant 8 : !firrtl.const.uint<4>
    match enum:
      A:
        ; CHECK:  %h = firrtl.node interesting_name [[CST7]]
        node h = UInt<4>(7)
         ; CHECK: %i = firrtl.node interesting_name [[CST8]]
        node i = UInt<4>(8)

  ; CHECK-LABEL: firrtl.module private @subfield_implicit_cse
  module subfield_implicit_cse :
    input i: {x: UInt<1>}
    input cond: UInt<1>
    output o: UInt<1>

    ; Subfields always get emitted by their declarations.
    ; CHECK: [[SUB:%.+]] = firrtl.subfield %i[x]

    ; CHECK: %n3 = firrtl.node interesting_name [[SUB]]
    node n3 = i.x

    ; CHECK: firrtl.when %cond : !firrtl.uint<1> {
    when cond:
      ; CHECK: %n4 = firrtl.node interesting_name [[SUB]]
      node n4 = i.x
    ; CHECK: }

    ; Check that invalidation reuses subfields
    wire w: {a: UInt<1>}[1]
    ; CHECK: %invalid = firrtl.invalidvalue : !firrtl.vector<bundle<a: uint<1>>, 1>
    ; CHECK: firrtl.matchingconnect %w, %invalid
    invalidate w
    ; CHECK: %invalid_0 = firrtl.invalidvalue : !firrtl.vector<bundle<a: uint<1>>, 1>
    ; CHECK: firrtl.matchingconnect %w, %invalid_0
    invalidate w

  ; CHECK-LABEL: firrtl.module private @flip_one
  module flip_one :
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %0 = firrtl.subfield %bf[int_1]
    ; CHECK: %_T = firrtl.node interesting_name %0
    node _T = bf.int_1
    ; CHECK: firrtl.when %_T : !firrtl.uint<1> {
    when _T :
      skip

  ; CHECK-LABEL: firrtl.module private @mem_depth_1
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
      ; CHECK: %bar_MPORT, %bar_io_deq_bits_MPORT = firrtl.mem interesting_name Undefined {depth = 1 : i64, name = "bar", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} :
      ; CHECK: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<3>, mask: uint<1>>,
      ; CHECK: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<3>>

  ; CHECK-LABEL: firrtl.module private @mem_no_ports() {
  ; CHECK-NEXT: }
  ; https://github.com/llvm/circt/issues/531
  module mem_no_ports :
    mem bar : @[Decoupled.scala 218:16]
      data-type => UInt<3>
      depth => 1
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

  ; CHECK-LABEL: firrtl.module private @issue354(out %tmp5: !firrtl.sint<19>) {
  module issue354 :
    output tmp5: SInt<19>
    connect tmp5, SInt<19>(8)
     ; CHECK: %c8_si19 = firrtl.constant 8 : !firrtl.const.sint<19>
     ; CHECK: [[VAL:%.*]] = firrtl.constCast %c8_si19 : (!firrtl.const.sint<19>) -> !firrtl.sint<19>
     ; CHECK: firrtl.matchingconnect %tmp5, [[VAL]] : !firrtl.sint<19>

   ; CHECK-LABEL: firrtl.module private @issue347
  module issue347 :
    output tmp12: SInt<4>
    connect tmp12, SInt<4>(-4)
    ; CHECK: %c-4_si4 = firrtl.constant -4 : !firrtl.const.sint<4>

  ; CHECK-LABEL: firrtl.extmodule private @issue183<A: si32 = -1>()
  extmodule issue183:
     parameter A = -1

  ; The Scala FIRRTL Compiler allows this for an aggregate node with an internal
  ; analog.
  ; CHECK-LABEL: firrtl.module private @analog_in_aggregate_node
  module analog_in_aggregate_node:
    input a: { a: UInt<1>, b: Analog<1>}
    ; CHECK: %b = firrtl.node interesting_name %a : !firrtl.bundle<a: uint<1>, b: analog<1>>
    node b = a

  ; Check that a register clock sink is converted to passive
  ; CHECK-LABEL: firrtl.module private @register_clock_passive
  module register_clock_passive:
    input clkIn: Clock
    output clkOut: Clock
    connect clkOut, clkIn
    ; CHECK: firrtl.reg interesting_name %clkOut
    reg r: UInt<1>, clkOut

  ; Check that a register reset sink is converted to passive
  ; CHECK-LABEL: firrtl.module private @register_reset_passive
  module register_reset_passive:
    input clk: Clock
    output rst: UInt<1>
    invalidate rst
    ; CHECK: firrtl.regreset interesting_name %clk, %rst
    regreset r: UInt<1>, clk, rst, UInt<1>(0)

  ; Check that a register init sink is converted to passive
  ; CHECK-LABEL: firrtl.module private @register_init_passive
  module register_init_passive:
    input clk: Clock
    input rst: UInt<1>
    output init: UInt<1>
    invalidate init
    ; CHECK: firrtl.regreset interesting_name %clk, %rst, %init
    regreset r: UInt<1>, clk, rst, init

  ; https://github.com/llvm/circt/issues/492
  ; CHECK-LABEL: firrtl.module private @WriteOnlyMemIssue492
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

    connect memory.w.clk, clock
    connect memory.w.en, wEn
    connect memory.w.addr, wAddr
    connect memory.w.mask, wMask
    connect memory.w.data, wData

  ; https://github.com/llvm/circt/issues/559
  ; CHECK-LABEL: firrtl.module private @TrickyIssue559
  module TrickyIssue559:
    input input: UInt<1>
    output output: UInt<1>
    ; CHECK: firrtl.matchingconnect %output, %input
    connect output, input

  ; CHECK-LABEL: firrtl.module private @CheckInvalids
  module CheckInvalids_in0 :
    input in0 : UInt<1>
    ; CHECK-NOT: firrtl.connect %in0
    ; CHECK-NOT: firrtl.matchingconnect %in0
    invalidate in0

  module CheckInvalids_in1 :
    input in1 : { a : UInt<1>, b : UInt<1> }
    ; CHECK-NOT: firrtl.connect %in1
    ; CHECK-NOT: firrtl.matchingconnect %in1
    invalidate in1

  module CheckInvalids_in2 :
    input in2 : { a : UInt<1>, flip b : UInt<1>}
    ; CHECK: [[IN2_B:%.+]] = firrtl.subfield %in2[b]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect [[IN2_B]], [[INV]]
    invalidate in2

  module CheckInvalids_in3 :
    input in3 : {a : { b : UInt<1>, flip c : UInt<1>}}
    ; CHECK: [[IN3_A:%.+]] = firrtl.subfield %in3[a]
    ; CHECK: [[IN3_A_C:%.+]] = firrtl.subfield [[IN3_A]][c]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect [[IN3_A_C]], [[INV]]
    invalidate in3

  module CheckInvalids_out0 :
    output out0 : UInt<1>
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect %out0, [[INV]]
    invalidate out0

  module CheckInvalids_out1 :
    output out1 : { a : UInt<1>, b : UInt<1> }
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue : !firrtl.bundle<a: uint<1>, b: uint<1>>
    ; CHECK: firrtl.matchingconnect %out1, [[INV]]
    invalidate out1

  module CheckInvalids_out2 :
    output out2 : { a : UInt<1>, flip b : UInt<1>}
    ; CHECK: [[OUT2_A:%.+]] = firrtl.subfield %out2[a]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect [[OUT2_A]], [[INV]]
    invalidate out2

  module CheckInvalids_out3 :
    output out3 : {a : { b : UInt<1>, flip c : UInt<1>}}
    ; CHECK: [[OUT3_A:%.+]] = firrtl.subfield %out3[a]
    ; CHECK: [[OUT3_A_B:%.+]] = firrtl.subfield [[OUT3_A]][b]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect [[OUT3_A_B]], [[INV]]
    invalidate out3

  module CheckInvalids_wires :
    ; CHECK: %wire0 = firrtl.wire
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect %wire0, [[INV]]
    wire wire0 : UInt<1>
    invalidate wire0

    ; CHECK: %wire1 = firrtl.wire
    ; CHECK: [[WIRE1_B:%.+]] = firrtl.subfield %wire1[b]
    ; CHECK: [[WIRE1_A:%.+]] = firrtl.subfield %wire1[a]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect [[WIRE1_A]], [[INV]]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect [[WIRE1_B]], [[INV]]
    wire wire1 : {a : UInt<1>, flip b : UInt<1> }
    invalidate wire1

    ; An analog in the leaf of a wire should be attached not connected.
    ; CHECK: %wire2 = firrtl.wire
    ; CHECK: [[WIRE2_X:%.+]] = firrtl.subfield %wire2[x]
    ; CHECK: [[WIRE2_X_B:%.+]] = firrtl.subfield [[WIRE2_X]][b]
    ; CHECK: [[WIRE2_X_A:%.+]] = firrtl.subfield [[WIRE2_X]][a]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect [[WIRE2_X_A]], [[INV]]
    ; CHECK-NOT: firrtl.attach [[WIRE2_X_B]], [[INV]]
    wire wire2 : {x : {flip a : UInt<1>, flip b: Analog<1> } }
    invalidate wire2

    ; https://github.com/llvm/circt/issues/563
    ; CHECK: %U0_in0, %U0_in1, %U0_out0, %U0_out1 = firrtl.instance U0 interesting_name @mod_0_563
    inst U0 of mod_0_563

    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue
    ; CHECK: firrtl.matchingconnect %U0_in0, [[INV]]
    ; CHECK: [[INV:%.+]] = firrtl.invalidvalue : !firrtl.bundle<a: uint<5>>
    ; CHECK: firrtl.matchingconnect %U0_in1, [[INV]]
    invalidate U0

  ; This reference is declared after its first use.
  ; https://github.com/llvm/circt/issues/163
  module mod_0_563 :
    input in0: UInt<5>
    input in1: { a : UInt<5> }
    output out0: UInt<5>
    output out1: { a : UInt<5> }
    connect out0, in0
    connect out1, in1

  ; https://github.com/llvm/circt/issues/606
  ; CHECK-LABEL: firrtl.module private @mutableSubIndex606
  module mutableSubIndex606 :
    output io : UInt<1>[8]
    ; CHECK:  %0 = firrtl.subindex %io[0] : !firrtl.vector<uint<1>, 8>
    ; CHECK: [[VAL:%.*]] = firrtl.constCast %c0_ui1 : (!firrtl.const.uint<1>) -> !firrtl.uint<1>
    ; CHECK: firrtl.matchingconnect %0, [[VAL]] : !firrtl.uint<1>
    connect io[0], UInt<1>(0h00)


  ; https://github.com/llvm/circt/issues/782
  ; CHECK-LABEL: mem_madness782
  module mem_madness782:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>

    ; CHECK: %mem_r = firrtl.mem interesting_name Undefined {depth = 16 : i64, name = "mem", portNames = ["r"], readLatency = 2 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    mem mem:
      data-type => UInt<8>
      depth => 16
      reader => r
      read-latency => 2
      write-latency => 1
      read-under-write => undefined

    connect mem.r.clk, clock
    connect mem.r.en, rEn
    connect mem.r.addr, rAddr
    connect rData, mem.r.data

  ; Test that behavioral memory reads and writes both work and that flow checks
  ; don't fail here.  (A memory port should have duplex flow.)
  ; See: https://github.com/llvm/circt/issues/1058
  ; CHECK-LABEL: firrtl.module private @BehavioralMemory
  module BehavioralMemory:
    input clock: Clock
    input rAddr: UInt<3>
    output rData: UInt<1>
    input wAddr: UInt<3>
    input wData: UInt<1>

    cmem a: UInt<1>[8]

    ; CHECK: firrtl.matchingconnect %rData, %r
    infer mport r = a[rAddr], clock
    connect rData, r

    ; CHECK: firrtl.matchingconnect %w_data, %wData
    infer mport w = a[wAddr], clock
    connect w, wData

  ; Test that a mux with an unknown width select line parses.  This is a check
  ; of the predicate enforced on UInt1Type.
  ; CHECK-LABEL: firrtl.module private @MuxUnknownWidthSelect_Issue1108
  module MuxUnknownWidthSelect_Issue1108:
    input a: UInt<1>
    input b: UInt<1>
    input sel: UInt
    output c: UInt<8>
    connect c, mux(sel, a, b)

  ; Test that a mux with aggregate type is still compatible even if the leaf
  ; types disagree in their width.
  ; CHECK-LABEL: firrtl.module private @MuxAggregateWidthMismatch_Issue2806
  module MuxAggregateWidthMismatch_Issue2806:
    input a: UInt<1>[1]
    input b: UInt<32>[1]
    input x: {u: UInt<1>, v: UInt<2>}
    input y: {u: UInt<32>, v: UInt<2>}
    input sel: UInt<1>
    output c: UInt[1]
    output z: {u: UInt, v: UInt}
    ; CHECK: firrtl.mux(%sel, %a, %b)
    ; CHECK-SAME: -> !firrtl.vector<uint<32>, 1>
    ; CHECK: firrtl.mux(%sel, %x, %y)
    ; CHECK-SAME: -> !firrtl.bundle<u: uint<32>, v: uint<2>>
    connect c, mux(sel, a, b)
    connect z, mux(sel, x, y)

  ; CHECK-LABEL: firrtl.extmodule private @VerbatimStringParam
  ; CHECK-SAME:    <TYPE: none = #hw.param.verbatim<"bit">,
  ; CHECK-SAME:     FORMAT: none = #hw.param.verbatim<"xyz_timeout=%d\\n">,
  ; CHECK-SAME:     MIXED_QUOTES: none = #hw.param.verbatim<"\22'\\\22">>
  extmodule VerbatimStringParam :
    parameter TYPE = 'bit'
    parameter FORMAT = 'xyz_timeout=%d\n'
    parameter MIXED_QUOTES = '"\'\"'
  ; "

  ; CHECK-LABEL: firrtl.module private @issue1303
  module issue1303:
    output out: Reset
    connect out, UInt(1)
    ; CHECK: %[[c1:.*]] = firrtl.constant 1 : !firrtl.const.uint
    ; CHECK-NEXT: %[[c2:.*]] = firrtl.resetCast %[[c1]]
    ; CHECK-NEXT: %[[c3:.*]] = firrtl.constCast %[[c2]]
    ; CHECK-NEXT: firrtl.matchingconnect %out, %[[c3]] : !firrtl.reset


  ; CHECK-LABEL: @resetBundle
  module resetBundle:
    input a: {a: UInt<1>, b: AsyncReset}
    output b: {a: Reset, b: Reset}

    connect b, a
    ; CHECK: %1 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<1>, b: asyncreset>
    ; CHECK: %[[r1:.*]] = firrtl.resetCast %1
    ; CHECK: firrtl.matchingconnect %0, %[[r1]] : !firrtl.reset
    ; CHECK: %3 = firrtl.subfield %b[b] : !firrtl.bundle<a: reset, b: reset>
    ; CHECK: %4 = firrtl.subfield %a[b] : !firrtl.bundle<a: uint<1>, b: asyncreset>
    ; CHECK: %[[r4:.*]] = firrtl.resetCast %4
    ; CHECK: firrtl.matchingconnect %3, %[[r4]] : !firrtl.reset

  module LargeMem :
    input clock : Clock
    input reset : Reset
    ; CHECK: !chirrtl.cmemory<vector<uint<8>, 16>, 34359738368>
    smem testharness : UInt<8>[16] [34359738368]
    node w_addr = UInt<36>(42) @[Cat.scala 31:58]
    write mport MPORT = testharness[w_addr], clock

  ; Module as identifier
  ; Test parsing of "module" as an identifier for instance and module names
  extmodule SomeModule:
    input in: UInt<8>

  module ModuleAsIdentifier:
    inst module of SomeModule
    ; CHECK: firrtl.instance module interesting_name @SomeModule
    connect module.in, UInt(1)

  ; CHECK-LABEL: firrtl.module private @EnumTypes
  module EnumTypes:
    ; CHECK: !firrtl.enum<>
    wire a : {||}
    ; CHECK: !firrtl.enum<a>
    wire b : {|a|}
    ; CHECK: !firrtl.enum<a = 1>
    wire c : {|a = 1|}
    ; CHECK: !firrtl.enum<a: uint<8>>
    wire d : {|a : UInt<8>|}
    ; CHECK: !firrtl.enum<a = 1: uint<8>>
    wire e : {|a = 1 : UInt<8>|}

  ; CHECK-LABEL: firrtl.module private @EnumValue
  module EnumValues:
    ; CHECK-SAME: in %i: !firrtl.enum<Some: uint<8>, None>
    input i : {| Some : UInt<8>, None |}
    output o : UInt<8>

    ; CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.const.uint<8>
    ; CHECK: %0 = firrtl.enumcreate Some(%c0_ui8) : (!firrtl.const.uint<8>) -> !firrtl.enum<Some: uint<8>, None>
    ; CHECK: %some = firrtl.node interesting_name %0 : !firrtl.enum<Some: uint<8>, None>
    node some = {|Some : UInt<8>, None|}(Some, UInt<8>(0))
    ; CHECK: %c0_ui0 = firrtl.constant 0 : !firrtl.const.uint<0>
    ; CHECK: %1 = firrtl.enumcreate None(%c0_ui0) : (!firrtl.const.uint<0>) -> !firrtl.enum<Some: uint<8>, None>
    ; CHECK: %none = firrtl.node {{.*}} %1 : !firrtl.enum<Some: uint<8>, None>
    node none = {|Some : UInt<8>, None|}(None)

    ; CHECK: firrtl.match %i : !firrtl.enum<Some: uint<8>, None> {
    match i:
      ; CHECK: case Some(%arg0) {
      ; CHECK:   firrtl.matchingconnect %o, %arg0 : !firrtl.uint<8>
      ; CHECK: }
      Some(x):
        connect o, x
      ; CHECK: case None(%arg0) {
      ; CHECK:   %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
      ; CHECK:   firrtl.matchingconnect %o, %invalid_ui8 : !firrtl.uint<8>
      ; CHECK: }
      None:
        invalidate o

  ; CHECK-LABEL: firrtl.module private @TagExtract
  module TagExtract:
    input i : {|A, B, C = 8|}
    output o : UInt<4>
    ; CHECK: [[TAGEXTRACT:%[0-9]+]] = firrtl.tagextract %i : !firrtl.enum<A, B, C = 8>
    ; CHECK: firrtl.matchingconnect %o, [[TAGEXTRACT]] : !firrtl.uint<4>
    connect o, tagExtract(i)

  ; CHECK-LABEL: module private @RefsChild(
  ; CHECK-SAME: out %r: !firrtl.probe<uint<1>>
  ; CHECK-SAME: out %rw: !firrtl.rwprobe<uint<1>>
  module RefsChild :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    output rw : RWProbe<UInt<1>>

    ; CHECK-NEXT: %[[NODE:.+]] = firrtl.node sym @[[NODE_RW_SYM:[^ ]+]]
    node n = in
    ; CHECK-NEXT: %[[REF:.+]] = firrtl.ref.send %[[NODE]]
    ; CHECK-NEXT: ref.define %r, %[[REF]]
    define r = probe(n)
    ; CHECK: %[[NODE_RWREF:.+]] = firrtl.ref.rwprobe <@RefsChild::@[[NODE_RW_SYM]]>
    ; CHECK-NEXT: ref.define %rw, %[[NODE_RWREF]]
    define rw = rwprobe(n)

  ; CHECK-LABEL: module private @RefsChildOpenAgg(
  ; CHECK-SAME: in %in: !firrtl.openbundle<a: uint<1>, rw flip: rwprobe<uint<1>>> sym [<@[[SYM:[^,]+]],1,
  module RefsChildOpenAgg :
    input in : { a : UInt<1>, flip rw: RWProbe<UInt<1>> }
    ; CHECK-NEXT: %[[IN_RW:[^ ]+]] = firrtl.opensubfield %in[rw]
    ; CHECK-NEXT: %[[RWPROBE_IN_A:[^ ]+]] = firrtl.ref.rwprobe <@RefsChildOpenAgg::@[[SYM]]>
    ; CHECK-NEXT: firrtl.ref.define %[[IN_RW]], %[[RWPROBE_IN_A]]
    define in.rw = rwprobe(in.a)

  ; CHECK-LABEL: module private @Refs(
  module Refs :
    input in : const UInt<1>
    output r : Probe<const UInt>
    output rw : RWProbe<UInt>
    output notrw : Probe<UInt>
    output out : UInt<1>
    output out2 : UInt<1>
    output outconst : const UInt<1>
    ; CHECK-SAME: out %agg_out: !firrtl.probe<bundle<a: uint<1>, b: const.uint>>
    output agg_out : Probe<{a: UInt<1>, b: const UInt}>
    output field_rw : RWProbe<UInt<1>>

    ; CHECK-NEXT: %[[RC_IN:.+]], %[[RC_R:.+]], %[[RC_RW:.+]] = firrtl.instance rc
    inst rc of RefsChild
    connect rc.in, in
    ; CHECK: %[[OUTREF:.+]] = firrtl.ref.send %outconst
    ; CHECK-NEXT: %[[OUTREF_CAST:.+]] = firrtl.ref.cast %[[OUTREF]] : (!firrtl.probe<const.uint<1>>) -> !firrtl.probe<const.uint>
    ; CHECK-NEXT: ref.define %r, %[[OUTREF_CAST]]
    define r = probe(outconst)
    ; CHECK-NEXT: %[[RC_RW_CAST:.+]] = firrtl.ref.cast %[[RC_RW]] : (!firrtl.rwprobe<uint<1>>) -> !firrtl.rwprobe<uint>
    ; CHECK-NEXT: ref.define %rw, %[[RC_RW_CAST]]
    ; CHECK-SAME: rwprobe<uint>
    define rw = rc.rw

    ; CHECK-NEXT: %[[RC_RW_CAST_PROBE:.+]] = firrtl.ref.cast %[[RC_RW]] : (!firrtl.rwprobe<uint<1>>) -> !firrtl.probe<uint>
    ; CHECK-NEXT: ref.define %notrw, %[[RC_RW_CAST_PROBE]]
    ; CHECK-SAME: firrtl.probe
    define notrw = rc.rw

    ; CHECK-NEXT: %[[READ_RC_R:.+]] = firrtl.ref.resolve %[[RC_R]]
    ; CHECK-NEXT: connect %out, %[[READ_RC_R]]
    connect out, read(rc.r)
    ; CHECK-NEXT: %[[READ_RC_RW:.+]] = firrtl.ref.resolve %[[RC_RW]]
    ; CHECK-NEXT: connect %out, %[[READ_RC_RW]]
    connect out, read(rc.rw)

    ; ref.sub parsing
    ; CHECK-DAG: %[[AGG:.+]] = firrtl.wire interesting_name : !firrtl.bundle<a flip: const.uint<1>, b: uint>
    ; CHECK-DAG: %[[AGG2:.+]] = firrtl.wire interesting_name : !firrtl.bundle<a: uint, b flip: uint<1>>
    wire agg : { flip a : const UInt<1>, b : UInt }
    wire agg2 : { a : UInt, flip b : UInt<1> }
    ; CHECK-DAG: %[[AGG_B:.+]] = firrtl.subfield %[[AGG]][b]
    ; CHECK-DAG: %[[AGG_B_PROBE:.+]] = firrtl.ref.send %[[AGG_B]]
    ; CHECK-DAG: %[[READ_AGG_B_PROBE:.+]] = firrtl.ref.resolve %[[AGG_B_PROBE]]
    ; CHECK-DAG: connect %out2, %[[READ_AGG_B_PROBE]]
    connect out2, read(probe(agg.b))
    ; CHECK-DAG: %[[AGG2_PROBE:.+]] = firrtl.ref.send %[[AGG2]]
    ; CHECK-DAG: %[[READ_AGG2_PROBE:.+]] = firrtl.ref.resolve %[[AGG2_PROBE]]
    ; CHECK-DAG: %[[READ_AGG2_PROBE__B:.+]] = firrtl.subfield %[[READ_AGG2_PROBE]][b]
    ; CHECK-DAG: connect %out2, %[[READ_AGG2_PROBE__B]]
    connect out2, read(probe(agg2)).b

    ; CHECK: %[[AGG3:.+]] = firrtl.wire
    wire agg3 : const { a : UInt<1>, b : UInt }
    ; CHECK-NEXT: %[[AGG3_PROBE:.+]] = firrtl.ref.send %[[AGG3]]
    ; CHECK-NEXT: %[[AGG3_PROBE_CAST:.+]] = firrtl.ref.cast %[[AGG3_PROBE]] : (!firrtl.probe<const.bundle<a: uint<1>, b: uint>>) -> !firrtl.probe<bundle<a: uint<1>, b: const.uint>>
    ; CHECK-NEXT: ref.define %agg_out, %[[AGG3_PROBE_CAST]]
    define agg_out = probe(agg3)

    ; CHECK: %[[PROBE_IN:.+]] = firrtl.ref.send %in
    ; CHECK-DAG: %[[READ_PROBE_IN:.+]] = firrtl.ref.resolve %[[PROBE_IN]]
    ; CHECK-DAG: %[[SUM:.+]] = firrtl.and %[[READ_PROBE_IN]],
    connect outconst, and(read(probe(in)), UInt(1))

    ; CHECK: %[[AGG4:.+]] = firrtl.wire sym [<@[[AGG4_0_b_x_SYM:[^ ]+]],4,public>]
    wire agg4 : { a : UInt, flip b : {x : UInt<1>} }[2]
    ; (static ref expr creates dead subfield accesses, skip)
    ; CHECK: %[[AGG4_RW_0_b_x:.+]] = firrtl.ref.rwprobe <@Refs::@[[AGG4_0_b_x_SYM]]>
    ; CHECK-NEXT: firrtl.ref.define %field_rw, %[[AGG4_RW_0_b_x]]
    define field_rw = rwprobe(agg4[0].b.x)

   ; CHECK: %inst_rw = firrtl.wire : !firrtl.rwprobe<uint<1>>
   ; CHECK-NEXT: %inst_rw2 = firrtl.wire : !firrtl.rwprobe<uint<1>>
   ; CHECK-NEXT: %rc2_in_bounce = firrtl.wire sym @[[RC2_IN_BOUNCE_SYM:[^ ]+]]
   ; CHECK-NEXT: %rc2_in, %rc2_r, %rc2_rw = firrtl.instance rc2
   ; CHECK-NEXT: firrtl.matchingconnect %rc2_in, %rc2_in_bounce
    wire inst_rw : RWProbe<UInt<1>>
    wire inst_rw2 : RWProbe<UInt<1>>
    inst rc2 of RefsChild

    ; CHECK: %[[IN_CAST:[^ ]+]] = firrtl.constCast %in :
    ; CHECK: firrtl.matchingconnect %rc2_in_bounce, %[[IN_CAST]]
    connect rc2.in, in
   ; CHECK-NEXT: firrtl.when %rc2_in_bounce :
   ; CHECK-NEXT:   %[[RWPROBE_RC2_IN_BOUNCE_1:[^ ]+]] = firrtl.ref.rwprobe <@Refs::@[[RC2_IN_BOUNCE_SYM]]>
   ; CHECK-NEXT:   firrtl.ref.define %inst_rw, %[[RWPROBE_RC2_IN_BOUNCE_1]]
   ; CHECK-NEXT: }
    when rc2.in:
      define inst_rw = rwprobe(rc2.in)
   ; CHECK-NEXT: firrtl.matchingconnect %rc2_in_bounce,
    connect rc2.in, rc.in
   ; CHECK-NEXT: %[[RWPROBE_RC2_IN_BOUNCE_2:[^ ]+]] = firrtl.ref.rwprobe <@Refs::@[[RC2_IN_BOUNCE_SYM]]>
   ; CHECK-NEXT: firrtl.ref.define %inst_rw2, %[[RWPROBE_RC2_IN_BOUNCE_2]]
    define inst_rw2 = rwprobe(rc2.in)

  ; CHECK-LABEL: module private @ForceRelease(
  module ForceRelease :
    input in : UInt<1>
    input clock : Clock
    input cond : UInt<1>

    ; CHECK-NEXT: %{{.+}}, %{{.+}}, %[[RC_RW:.+]] = firrtl.instance rc
    inst rc of RefsChild
    connect rc.in, in

    ; Check (const) literal works, even if uninferred width.
    ; Cast reference to more general form as needed.
    ; CHECK: %[[RC_RW_CAST:.+]] = firrtl.ref.cast %[[RC_RW]] : (!firrtl.rwprobe<uint<1>>) -> !firrtl.rwprobe<uint>
    ; CHECK: firrtl.ref.force_initial %[[TRUE:.+]], %[[RC_RW_CAST]], %{{.+}} : !firrtl.uint<1>, !firrtl.rwprobe<uint>, !firrtl.const.uint
    force_initial(rc.rw, UInt(0))

    ; CHECK: firrtl.ref.force %clock, %cond, %[[RC_RW]], %{{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>, !firrtl.const.uint<1>
    force(clock, cond, rc.rw, UInt<1>(1))
    ; CHECK: %[[NOT_COND:.+]] = firrtl.not %cond
    ; CHECK: firrtl.ref.release %clock, %[[NOT_COND]], %[[RC_RW]] : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    release(clock, not(cond), rc.rw)

    ; CHECK-NEXT: %{{.+}}, %{{.+}}, %[[RC2_RW:.+]] = firrtl.instance rc2
    inst rc2 of RefsChild
    connect rc2.in, in
    ; CHECK: firrtl.ref.release_initial %[[TRUE]], %[[RC2_RW]] : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    release_initial(rc2.rw)

  ; CHECK-LABEL: extmodule private @RefExtABI
  ; CHECK-NOT: internalPaths
  extmodule RefExtABI :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    output data : UInt<3>
    output r2 : Probe<{a : UInt<3>}[3]>

  ; CHECK-LABEL: @RWProbePort(
  module RWProbePort:
    ; CHECK: in %in: !firrtl.vector<uint<1>, 2> sym [<@[[IN_SYM:.+]],2,public>],
    input in : UInt<1>[2]
    output p : RWProbe<UInt<1>>
    ; CHECK:  firrtl.ref.rwprobe <@RWProbePort::@[[IN_SYM]]> : !firrtl.rwprobe<uint<1>>
    define p = rwprobe(in[1])

  ; CHECK-LABEL: @RWProbeUninferredPort(
  module RWProbeUninferredPort:
    ; CHECK: in %in: !firrtl.vector<uint, 2> sym [<@[[IN_SYM:.+]],2,public>],
    input in : UInt[2]
    output p : RWProbe<UInt>
    ; CHECK:  firrtl.ref.rwprobe <@RWProbeUninferredPort::@[[IN_SYM]]> : !firrtl.rwprobe<uint>
    define p = rwprobe(in[1])

  ; CHECK-LABEL: @RWProbeUninferredReset(
  module RWProbeUninferredReset:
    ; CHECK: in %in: !firrtl.bundle<a: reset> sym [<@[[IN_SYM:.+]],1,public>],
    input in : {a : Reset}
    output p : RWProbe<Reset>
    ; CHECK:  firrtl.ref.rwprobe <@RWProbeUninferredReset::@[[IN_SYM]]> : !firrtl.rwprobe<reset>
    define p = rwprobe(in.a)

  ; CHECK-LABEL: module private @ProbeInvalidate
  ; CHECK-NEXT: }
  module ProbeInvalidate:
    output p : Probe<UInt<1>>
    invalidate p

  ; CHECK-LABEL: module private @NumericFields
  ; See: https://github.com/llvm/circt/issues/5110
  module NumericFields:
    input a: {0: {0: {bar: UInt<1>}}}
    output b: UInt<1>
    input c: {0: {0: {0: {bar: UInt<1>}}}}
    output d: UInt<1>

    ; CHECK:      %0 = firrtl.subfield %c["0"]
    ; CHECK-NEXT: %1 = firrtl.subfield %0["0"]
    ; CHECK-NEXT: %2 = firrtl.subfield %1["0"]
    ; CHECK-NEXT: %3 = firrtl.subfield %2[bar]
    ; CHECK-NEXT: %4 = firrtl.subfield %a["0"]
    ; CHECK-NEXT: %5 = firrtl.subfield %4["0"]
    ; CHECK-NEXT: %6 = firrtl.subfield %5[bar]
    connect b, a.0.0.bar
    connect d, c.0.0.0.bar
    ; CHECK-NEXT: firrtl.matchingconnect %b, %6
    ; CHECK-NEXT: firrtl.matchingconnect %d, %3

  ; CHECK-LABEL: firrtl.module private @ConstTypes(
  module ConstTypes:
    input c: const Clock         ; CHECK: %c: !firrtl.const.clock,
    input r: const Reset         ; CHECK: %r: !firrtl.const.reset,
    input ar: const AsyncReset   ; CHECK: %ar: !firrtl.const.asyncreset,
    input a: const Analog        ; CHECK: %a: !firrtl.const.analog,
    input a8: const Analog<8>    ; CHECK: %a8: !firrtl.const.analog<8>,
    input s: const SInt          ; CHECK: %s: !firrtl.const.sint,
    input s4: const SInt<4>      ; CHECK: %s4: !firrtl.const.sint<4>,
    input u: const UInt          ; CHECK: %u: !firrtl.const.uint,

    ; CHECK: %b: !firrtl.const.bundle<int_1 flip: uint<1>, int_out: uint<2>>
    input b: const {flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %b_constfields: !firrtl.bundle<int_1 flip: const.uint<1>, int_out: const.uint<2>>
    input b_constfields: {flip int_1 : const UInt<1>, int_out : const UInt<2>}
    ; CHECK: %mixedb: !firrtl.bundle<a: sint<1>, b: const.uint<2>>
    input mixedb: {a: SInt<1>, b: const UInt<2>}
    ; CHECK: %vec: !firrtl.const.vector<uint<1>, 4>) {
    input vec: const UInt<1>[4]

    ; CHECK-NEXT: %w = firrtl.wire interesting_name : !firrtl.const.sint<4>
    wire w: const SInt<4>
    ; CHECK-NEXT: firrtl.matchingconnect %w, %s4 : !firrtl.const.sint<4>
    connect w, s4
    ; CHECK-NEXT: %nonconst_w = firrtl.wire interesting_name : !firrtl.sint<4>
    wire nonconst_w: SInt<4>
    ; CHECK-NEXT: [[CAST:%.+]] = firrtl.constCast %s4 : (!firrtl.const.sint<4>) -> !firrtl.sint<4>
    ; CHECK-NEXT: firrtl.matchingconnect %nonconst_w, [[CAST]] : !firrtl.sint<4>
    connect nonconst_w, s4

  ; CHECK-LABEL: firrtl.module @Printf
  public module Printf:
    input clock: Clock
    input i8: UInt<8>

    ; CHECK:      firrtl.printf %clock, %c1_ui1, "%b, %0b, %8b" (%i8, %i8, %i8) : !firrtl.clock, !firrtl.const.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>, !firrtl.uint<8>
    printf(clock, UInt<1>(1), "%b, %0b, %8b", i8, i8, i8)
    ; CHECK-NEXT: firrtl.printf %clock, %c1_ui1, "%d, %0d, %8d" (%i8, %i8, %i8) : !firrtl.clock, !firrtl.const.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>, !firrtl.uint<8>
    printf(clock, UInt<1>(1), "%d, %0d, %8d", i8, i8, i8)
    ; CHECK-NEXT: firrtl.printf %clock, %c1_ui1, "%x, %0x, %8x" (%i8, %i8, %i8) : !firrtl.clock, !firrtl.const.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>, !firrtl.uint<8>
    printf(clock, UInt<1>(1), "%x, %0x, %8x", i8, i8, i8)

    ; CHECK-NEXT: firrtl.printf %clock, %c1_ui1, "%c" (%i8) : !firrtl.clock, !firrtl.const.uint<1>, !firrtl.uint<8>
    printf(clock, UInt<1>(1), "%c", i8)

    ; CHECK-NEXT: firrtl.printf %clock, %c1_ui1, "%%" : !firrtl.clock, !firrtl.const.uint<1>
    printf(clock, UInt<1>(1), "%%")

    ; CHECK-NEXT: firrtl.fprintf %clock, %c1_ui1, "test.txt", "%%" : !firrtl.clock, !firrtl.const.uint<1>
    fprintf(clock, UInt<1>(1), "test.txt", "%%")

;// -----

; CHECK-LABEL: firrtl.circuit "Foo_v3p0p0"
FIRRTL version 3.0.0
circuit Foo_v3p0p0:
  module Foo_v3p0p0:
    input clock: Clock
    input reset: UInt<1>
    input a: UInt<1>
    output b: UInt<1>

    ; CHECK: firrtl.matchingconnect %b, %a
    connect b, a

    ; CHECK: [[INV:%.+]]  = firrtl.invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: firrtl.matchingconnect %b, [[INV]] : !firrtl.uint<1>
    invalidate b

    ; CHECK:      %[[zero:[0-9A-Za-z_]+]] = firrtl.constant 0
    ; CHECK-NEXT: %r = firrtl.regreset interesting_name %clock, %reset, %[[zero]]
    regreset r: UInt<1>, clock, reset, UInt<1>(0)

  ; CHECK: module private @LiteralIdentifiers
  ; CHECK-SAME: in %_0: !firrtl.bundle<"1": uint<1>>
  ; CHECK-SAME: out %_2: !firrtl.bundle<"3": uint<1>>
  ; CHECK-SAME: portNames = ["0", "2"]
  module LiteralIdentifiers:
    input `0`: {`1`: UInt<1>}
    output `2`: {`3`: UInt<1>}

    ; CHECK-NEXT: %0 = firrtl.subfield %_0["1"] : !firrtl.bundle<"1": uint<1>>
    ; CHECK-NEXT: %1 = firrtl.subfield %_2["3"] : !firrtl.bundle<"3": uint<1>>
    ; CHECK-NEXT: firrtl.matchingconnect %1, %0 : !firrtl.uint<1>
    connect `2`.`3`, `0`.`1`

    ; Ensure that `a` is equivalent to a.
    ; CHECK-NEXT: %a = firrtl.node {{.+}}%_0
    node `a` = `0`
    ; CHECK-NEXT: %b = firrtl.node {{.+}}%a
    node b = a

;// -----
; Check reference expressions using literal identifiers or keywords.

; CHECK-LABEL: firrtl.circuit "Probes_refexprs"
FIRRTL version 3.0.0
circuit Probes_refexprs:
  extmodule refs:
    output `0`: {`1`: Probe<{`2`: UInt<1>}>, rwprobe: RWProbe<UInt<1>>}
    output ref: {module: Probe<{when: UInt<1>}>}

  module Probes_refexprs:
    output `0`: {`1`: Probe<{`2`: UInt<1>}>}
    output out: UInt<1>

    inst `9` of refs
    inst ref of refs

    ; CHECK: firrtl.ref.define
    define `0`.`1` = `9`.`0`.`1`

    ; CHECK-COUNT-4: firrtl.ref.resolve
    node a = read(`9`.`0`.`1`).`2`
    node b = read(`9`.`0`.`1`.`2`)
    node c = read(`9`.ref.module).when
    ; Keyword as leading part of static ref expression:
    node d = read(ref.`0`.`1`.`2`)

    connect out, and(and(a, b), and(c, d))

    ; CHECK: %[[TEST:.+]] = firrtl.wire sym @[[TEST_SYM:[^ ]+]]
    wire `test`: {`0`: UInt<1>, `b`: UInt<1>}
    connect `test`.`0`, a
    connect `test`.`b`, b
    ; CHECK: %[[TEST_REF:.+]] = firrtl.ref.rwprobe <@Probes_refexprs::@[[TEST_SYM]]>
    ; CHECK: force_initial %{{.+}}, %[[TEST_REF]], %[[TEST]]
    force_initial(rwprobe(`test`), `test`)
    ; CHECK: force_initial
    force_initial(`9`.`0`.rwprobe, `test`.`0`)

;// -----

; CHECK-LABEL: firrtl.circuit "RadixEncodedIntegerLiterals"
FIRRTL version 2.4.0
circuit RadixEncodedIntegerLiterals:
  module RadixEncodedIntegerLiterals:
    output bu: UInt<8>
    output ou: UInt<8>
    output du0: UInt<8>
    output du1: UInt<8>
    output hu: UInt<8>
    output bs: SInt<8>
    output os: SInt<8>
    output dus0: SInt<8>
    output dus1: SInt<8>
    output hs: SInt<8>

    ; CHECK:      %[[c42_ui7:[-a-zA-Z_0-9]+]] = firrtl.constant 42
    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[c42_ui7]]
    ; CHECK-NEXT: firrtl.connect %bu, %[[c_constCast]]
    bu <= UInt(0b101010)

    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[c42_ui7]]
    ; CHECK-NEXT: firrtl.connect %ou, %[[c_constCast]]
    ou <= UInt(0o052)

    ; Note: this creates a second constant because the width of a parsed
    ; constant is dependent on the overestimation of LLVM::StringRef.
    ;
    ; CHECK:      %[[c42_ui8:[-a-zA-Z_0-9]+]] = firrtl.constant 42
    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[c42_ui8]]
    ; CHECK-NEXT: firrtl.connect %du0, %[[c_constCast]]
    du0 <= UInt(42)

    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[c42_ui8]]
    ; CHECK-NEXT: firrtl.connect %du1, %[[c_constCast]]
    du1 <= UInt(0d42)

    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[c42_ui7]]
    ; CHECK-NEXT: firrtl.connect %hu, %[[c_constCast]]
    hu <= UInt(0h2a)

    ; CHECK:      %[[cn42_si7:[-a-zA-Z_0-9]+]] = firrtl.constant -42
    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[cn42_si7]]
    ; CHECK-NEXT: firrtl.connect %bs, %[[c_constCast]]
    bs <= SInt(-0b101010)

    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[cn42_si7]]
    ; CHECK-NEXT: firrtl.connect %os, %[[c_constCast]]
    os <= SInt(-0o52)

    ; Note: this creates a second constant because the width of a parsed
    ; constant is dependent on the overestimation of LLVM::StringRef.
    ;
    ; CHECK:      %[[cn42_si8:[-a-zA-Z_0-9]+]] = firrtl.constant -42
    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[cn42_si8]]
    ; CHECK-NEXT: firrtl.connect %dus0, %[[c_constCast]]
    dus0 <= SInt(-42)

    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[cn42_si8]]
    ; CHECK-NEXT: firrtl.connect %dus1, %[[c_constCast]]
    dus1 <= SInt(-0d42)

    ; CHECK-NEXT: %[[c_constCast:[-a-zA-Z_0-9]+]] = firrtl.constCast %[[cn42_si7]]
    ; CHECK-NEXT: firrtl.connect %hs, %[[c_constCast]]
    hs <= SInt(-0h2a)

;// -----
; Tests for type alias
FIRRTL version 4.0.0
circuit Top:
  type WordType = const UInt<32>
  type ValidType = UInt<1>
  type Data = {w: const WordType[2], valid: ValidType, flip ready: UInt<1>}
  type Complex = { real: SInt<10>, imag: SInt<10> }
  type Complex_id = Complex
  ; expected-warning @+1 {{type alias for non-base type '!firrtl.probe<alias<Complex, bundle<real: sint<10>, imag: sint<10>>>>' is currently not supported. Type alias is stripped immediately}}
  type ProbeComplex = Probe<Complex>
  ; CHECK: @Top
  ; CHECK-SAME: in %in_data: !firrtl.alias<Data, bundle<w: const.vector<const.alias<WordType, const.uint<32>>, 2>,
  ; CHECK-SAME:                                         valid: alias<ValidType, uint<1>>, ready flip: uint<1>>>,
  ; CHECK-SAME: in %in_complex: !firrtl.alias<Complex, bundle<real: sint<10>, imag: sint<10>>>
  ; CHECK-SAME: out %out: !firrtl.probe<alias<Complex, bundle<real: sint<10>, imag: sint<10>>>>
  ; CHECK-SAME: out %out_valid: !firrtl.alias<ValidType, uint<1>>

  ; CHECK-NEXT: %[[RESULT:.+]] = firrtl.subfield %in_data[valid] : !firrtl.alias<Data, bundle<w: const.vector<const.alias<WordType, const.uint<32>>, 2>, valid: alias<ValidType, uint<1>>, ready flip: uint<1>>>
  ; CHECK-NEXT: %c = firrtl.wire interesting_name : !firrtl.alias<Complex_id, alias<Complex, bundle<real: sint<10>, imag: sint<10>>>>
  ; CHECK-NEXT: firrtl.matchingconnect %out_valid, %[[RESULT]] : !firrtl.alias<ValidType, uint<1>>
  public module Top:
    input in_data: Data
    input in_complex: Complex
    output out: ProbeComplex
    output out_valid: ValidType
    wire c: Complex_id
    connect out_valid, in_data.valid

  ; CHECK-LABEL: firrtl.module private @Const
  ; CHECK-SAME: (in %a: !firrtl.const.alias<ConstI1, const.uint<1>>)
  type ConstI1 = const UInt<1>
  module Const :
    input a: ConstI1

;// -----

; CHECK-LABEL: firrtl.circuit "Layers"
FIRRTL version 4.1.0
circuit Layers:
  layer A, bind:
    layer B, bind:
      layer C, bind:
      layer D, bind:
        layer E, inline:
    layer F, bind:
    ; CHECK-NEXT: firrtl.layer @A bind {
    ; CHECK-NEXT:   firrtl.layer @B bind {
    ; CHECK-NEXT:     firrtl.layer @C bind {
    ; CHECK-NEXT:     }
    ; CHECK-NEXT:     firrtl.layer @D bind {
    ; CHECK-NEXT:       firrtl.layer @E inline {
    ; CHECK-NEXT:       }
    ; CHECK-NEXT:     }
    ; CHECK-NEXT:   }
    ; CHECK-NEXT:   firrtl.layer @F bind {
    ; CHECK-NEXT:   }
    ; CHECK-NEXT: }

  ; CHECK:      firrtl.module @Layers
  ; CHECK-SAME:   out %b: !firrtl.probe<uint<1>, @A>
  ; CHECK-SAME:   out %c: !firrtl.rwprobe<uint<1>, @A::@B>
  public module Layers:
    input a: UInt<1>
    output b: Probe<UInt<1>, A>
    output c: RWProbe<UInt<1>, A.B>

    layerblock A:
      node A_a = a
      layerblock B:
        node B_a = a
        layerblock C:
          node C_a = a
        layerblock D:
          node D_a = a
          layerblock E:
            node E_a = a
      layerblock F:
        node F_a = a

    ; CHECK-NEXT: firrtl.layerblock @A {
    ; CHECK-NEXT:   %A_a = firrtl.node{{.*}} %a
    ; CHECK-NEXT:   firrtl.layerblock @A::@B {
    ; CHECK-NEXT:     %B_a = firrtl.node{{.*}} %a
    ; CHECK-NEXT:     firrtl.layerblock @A::@B::@C {
    ; CHECK-NEXT:       %C_a = firrtl.node{{.*}} %a
    ; CHECK-NEXT:     }
    ; CHECK-NEXT:     firrtl.layerblock @A::@B::@D {
    ; CHECK-NEXT:       %D_a = firrtl.node{{.*}} %a
    ; CHECK-NEXT:       firrtl.layerblock @A::@B::@D::@E {
    ; CHECK-NEXT:         %E_a = firrtl.node{{.*}} %a
    ; CHECK-NEXT:       }
    ; CHECK-NEXT:     }
    ; CHECK-NEXT:   }
    ; CHECK-NEXT:   firrtl.layerblock @A::@F {
    ; CHECK-NEXT:     %F_a = firrtl.node{{.*}} %a
    ; CHECK-NEXT:   }
    ; CHECK-NEXT: }

  ; CHECK: firrtl.layer @WithDirectory1 bind attributes {output_file = #hw.output_file<"foo{{/|\\\\}}">}
  layer WithDirectory1, bind, "foo":
  ; CHECK: firrtl.layer @WithDirectory2 bind attributes {output_file = #hw.output_file<"foo{{/|\\\\}}">}
  layer WithDirectory2, bind, "foo/":

;// -----
; CHECK-LABEL: firrtl.circuit "BasicProps"
FIRRTL version 5.1.0
circuit BasicProps :
  public module BasicProps :
  ; CHECK-LABEL: module private @Integer
  module Integer :
    ; CHECK-SAME: out %a: !firrtl.integer
    output a : Integer
    ; CHECK: %0 = firrtl.integer -10
    ; CHECK: firrtl.propassign %a, %0 : !firrtl.integer
    propassign a, Integer(-10)

  ; CHECK-LABEL: module private @Bool
  module Bool :
    ; CHECK-SAME: out %true: !firrtl.bool
    ; CHECK-SAME: out %false: !firrtl.bool
    output true : Bool
    output false : Bool
    ; CHECK-NEXT: %0 = firrtl.bool true
    ; CHECK-NEXT: firrtl.propassign %true, %0 : !firrtl.bool
    ; CHECK-NEXT: %1 = firrtl.bool false
    ; CHECK-NEXT: firrtl.propassign %false, %1 : !firrtl.bool
    propassign true, Bool(true)
    propassign false, Bool(false)

  ; CHECK-LABEL: module private @Double
  module Double :
     ; CHECK-SAME: out %x: !firrtl.double
     ; CHECK-SAME: out %y: !firrtl.double
     ; CHECK-SAME: out %negzero: !firrtl.double
     ; CHECK-SAME: out %twoTo64: !firrtl.double
     ; CHECK-SAME: out %exp: !firrtl.double
     output x : Double
     output y : Double
     output negzero : Double
     output twoTo64 : Double
     output exp : Double
     ; Try not to overly rely on double printing/rounding/precision.
     ; CHECK-NEXT: firrtl.double  0.1111111111111111 : f64
     ; CHECK: propassign
     propassign x, Double(0.1111111111111111111111111111111111111111)
     ; CHECK: firrtl.double 1.{{0*[eE]}}+00
     propassign y, Double(1.0)
     ; CHECK: firrtl.double -0.{{0+[eE]}}+00
     propassign negzero, Double(-0.0)
     ; CHECK: firrtl.double 1.844674407370955{{[0-9]*[eE]}}+
     propassign twoTo64, Double(18446744073709551616.0)
     ; CHECK: firrtl.double 1.2{{.+[eE]}}+30
     propassign exp, Double(1.2E+30)

  ; CHECK-LABEL: module private @String
  module String :
    ; CHECK-SAME: out %a: !firrtl.string
    output a : String
    ; CHECK: %0 = firrtl.string "hello"
    ; CHECK: firrtl.propassign %a, %0 : !firrtl.string
    propassign a, String("hello")

  ; CHECK-LABEL: module private @Path
  module Path :
    ; CHECK-SAME: out %path: !firrtl.path
    output path : Path
    ; CHECK: firrtl.unresolved_path "OMDeleted:"
    ; CHECK: firrtl.propassign %path, %0 : !firrtl.path
    propassign path, path("OMDeleted:")

  ; CHECK-LABEL: firrtl.class private @SimpleClass(in %a: !firrtl.string, out %b: !firrtl.string) {
  ; CHECK-NEXT:     firrtl.propassign %b, %a : !firrtl.string
  ; CHECK-NEXT:  }
  class SimpleClass:
    input a: String
    output b: String
    propassign b, a

  ; CHECK-LABEL: firrtl.class private @Client(out %a: !firrtl.class<@SimpleClass(in a: !firrtl.string, out b: !firrtl.string)>)
  class Client:
    output a: Inst<SimpleClass>
    ; CHECK-NEXT:  %b = firrtl.object @SimpleClass(in a: !firrtl.string, out b: !firrtl.string)
    object b of SimpleClass
    ; CHECK-NEXT: firrtl.propassign %a, %b : !firrtl.class<@SimpleClass(in a: !firrtl.string, out b: !firrtl.string)>
    propassign a, b

  ; CHECK-LABEL firrtl.class private @OtherClient(in %a: !firrtl.string, out %b: !firrtl.string)
  class OtherClient:
    input a : String
    output b : String
    object o of SimpleClass
    ; TODO: This test needs to be fixed up once flow-checking for objects is implemented correctly.
    propassign b, o.b

  ; CHECK-LABEL firrtl.extclass private @ExtClass(in in: !firrtl.string, out out: !firrtl.string)
  extclass ExtClass:
    input in : String
    output out : String

  ; CHECK-LABEL: firrtl.class private @UserOfExtClass(in %ext_object: !firrtl.class<@ExtClass(in in: !firrtl.string, out out: !firrtl.string)>, out %out: !firrtl.string)
  class UserOfExtClass:
    input ext_object : Inst<ExtClass>
    output out : String
    ; CHECK-NEXT: %0 = firrtl.object.subfield %ext_object[out] : !firrtl.class<@ExtClass(in in: !firrtl.string, out out: !firrtl.string)
    ; CHECK-NEXT: firrtl.propassign %out, %0 : !firrtl.string
    propassign out, ext_object.out

  ; CHECK-LABEL: module private @List
  module List :
    ; CHECK-SAME: out %strings: !firrtl.list<string>
    ; CHECK-SAME: out %objs: !firrtl.list<class<@SimpleClass({{.*}})>>
    ; CHECK-SAME: out %nested: !firrtl.list<list<string>>
    output strings : List<String>
    output objs : List<Inst<SimpleClass>>
    output nested : List<List<String>>

    ; CHECK-NEXT: %[[HELLO:.+]] = firrtl.string "hello"
    ; CHECK-NEXT: %[[WORLD:.+]] = firrtl.string "world"
    ; CHECK-NEXT: %[[STRINGS:.+]] = firrtl.list.create %[[HELLO]], %[[WORLD]] : !firrtl.list<string>
    ; CHECK-NEXT: firrtl.propassign %strings, %[[STRINGS]] : !firrtl.list<string>
    propassign strings, List<String>(String("hello"), String("world"))

    ; CHECK-NEXT: %[[OBJ:.+]] = firrtl.object
    ; CHECK-NEXT: %[[OBJS:.+]] = firrtl.list.create %[[OBJ]], %[[OBJ]] : !firrtl.list<class<@SimpleClass({{.*}})>>
    ; CHECK-NEXT: firrtl.propassign %objs, %[[OBJS]] : !firrtl.list<class
    object obj of SimpleClass
    propassign objs, List<Inst<SimpleClass>>(obj, obj)

    ; CHECK-NEXT: %[[EMPTY1:.+]] = firrtl.list.create : !firrtl.list<string>
    ; CHECK-NEXT: %[[TEST:.+]] = firrtl.string "test"
    ; CHECK-NEXT: %[[TESTLIST:.+]] = firrtl.list.create %[[TEST]] : !firrtl.list<string>
    ; CHECK-NEXT: %[[EMPTY2:.+]] = firrtl.list.create : !firrtl.list<string>
    ; CHECK-NEXT: %[[NESTED:.+]] = firrtl.list.create %[[EMPTY1]], %[[TESTLIST]], %[[EMPTY2]] : !firrtl.list<list<string>>
    ; CHECK-NEXT: firrtl.propassign %nested, %[[NESTED]]
    propassign nested, List<List<String>>(List<String>(), List<String>(String("test")), List<String>())

;// -----
FIRRTL version 4.0.0

; CHECK-LABEL: firrtl.circuit "IntegerArithmetic"
circuit IntegerArithmetic :
  public module IntegerArithmetic :
    input a : Integer
    input b : Integer
    output c : Integer
    output d : Integer
    output e : Integer
    output f : Integer

    ; CHECK: [[C:%.+]] = firrtl.integer.add %a, %b
    ; CHECK: firrtl.propassign %c, [[C]]
    propassign c, integer_add(a, b)

    ; CHECK: [[D:%.+]] = firrtl.integer.mul %a, %b
    ; CHECK: firrtl.propassign %d, [[D]]
    propassign d, integer_mul(a, b)

    ; CHECK: [[E:%.+]] = firrtl.integer.shr %a, %b
    ; CHECK: firrtl.propassign %e, [[E]]
    propassign e, integer_shr(a, b)

    ; CHECK: [[F:%.+]] = firrtl.integer.shl %a, %b
    ; CHECK: firrtl.propassign %f, [[F]]
    propassign f, integer_shl(a, b)

;// -----
FIRRTL version 4.0.0

; CHECK-LABEL: firrtl.circuit "PropertyListOps"
circuit PropertyListOps :
  public module PropertyListOps :
    input a : List<Integer>
    input b : List<Integer>
    output c : List<Integer>

    ; CHECK: [[C:%.+]] = firrtl.list.concat %a, %b
    ; CHECK: firrtl.propassign %c, [[C]]
    propassign c, list_concat(a, b)

;// -----
FIRRTL version 3.1.0

; CHECK-LABEL: circuit "BundleOfProps"
circuit BundleOfProps:
  module BundleOfProps:
    input x : {a : String}
    wire y : {a: String}
    ; CHECK-COUNT-2: !firrtl.openbundle<a: string>

;// -----
FIRRTL version 3.1.0

; CHECK-LABEL: circuit "VecOfProps"
circuit VecOfProps:
  module VecOfProps:
    input x : String[2]
    wire y : String[2]
    ; CHECK-COUNT-2: openvector<string, 2>

;// -----
; Test parsing of wires of probes, wires of agg of probes, and namekinds.
; CHECK-LABEL: circuit "WireOfProbesAndNames"
FIRRTL version 4.0.0
circuit WireOfProbesAndNames:
  public module WireOfProbesAndNames:
    ; CHECK: %mixed = firrtl.wire interesting_name : !firrtl.openbundle
    wire mixed : { a : UInt<3>, b : Probe<UInt<3>> }
    ; CHECK: %probe = firrtl.wire : !firrtl.probe
    wire probe : Probe<UInt<1>>

;// -----
FIRRTL version 5.1.0

; CHECK-LABEL: circuit "AnyRef"
circuit AnyRef:
  class Foo:
    skip
  public module AnyRef:
    input x : AnyRef
    ; CHECK: !firrtl.anyref
    output y : AnyRef
    ; CHECK: !firrtl.anyref
    output listOfAny : List<AnyRef>
    ; CHECK: !firrtl.list<anyref>

    object foo of Foo
    propassign y, foo
    ; CHECK: %[[OBJ:.+]] = firrtl.object
    ; CHECK: %[[CAST:.+]] = firrtl.object.anyref_cast %[[OBJ]]
    ; CHECK: firrtl.propassign %y, %[[CAST]]

    propassign listOfAny, List<AnyRef>(foo, x)
    ; CHECK-NEXT: %[[CAST:.+]] = firrtl.object.anyref_cast %[[OBJ]]
    ; CHECK-NEXT: %[[LIST:.+]] = firrtl.list.create %[[CAST]], %x
    ; CHECK-NEXT: propassign %listOfAny, %[[LIST]]

;// -----

FIRRTL version 4.0.0
; CHECK-LABEL: circuit "PublicModules"
circuit PublicModules:
  ; CHECK: firrtl.module @Foo
  public module Foo:
  ; CHECK: firrtl.module private @Bar
  module Bar:
  ; CHECK: firrtl.module @PublicModules
  public module PublicModules:

;// -----

FIRRTL version 4.0.0
; CHECK-LABEL: firrtl.circuit "LayerEnabledModule"
circuit LayerEnabledModule:
  layer A, bind:
  layer B, bind:
    layer C, bind:
  ; CHECK: firrtl.module @LayerEnabledModule
  ; CHECK-SAME: layers = [@A, @B::@C]
  public module LayerEnabledModule enablelayer A enablelayer B.C:

  ; CHECK: firrtl.module @LayerEnabledModule2
  ; CHECK-SAME: layers = [@A, @B::@C]
  public module LayerEnabledModule2 enablelayer A, B.C:

  ; CHECK: firrtl.module private @UserOfLayerEnabledModule
  ; CHECK-SAME: layers = [@A, @B::@C]
  module UserOfLayerEnabledModule enablelayer A enablelayer B.C:
    ; CHECK: firrtl.instance i interesting_name {layers = [@A, @B::@C]} @LayerEnabledModule()
    inst i of LayerEnabledModule

;// -----

FIRRTL version 5.2.0

; CHECK-LABEL: firrtl.circuit "KnownLayers"
circuit KnownLayers:
  layer A, bind:
  layer B, bind:
    layer C, bind:

  ; CHECK: firrtl.extmodule private @ExtModuleWithKnownLayers
  ; CHECK-SAME: knownLayers = [@A, @B, @B::@C]}
  extmodule ExtModuleWithKnownLayers knownlayer A knownlayer B knownlayer B.C:

  ; CHECK: firrtl.extmodule private @ExtModuleWithKnownLayers2
  ; CHECK-SAME: knownLayers = [@A, @B, @B::@C]}
  extmodule ExtModuleWithKnownLayers2 knownlayer A, B, B.C:

  ; CHECK: firrtl.extmodule private @ExtModuleWithKnownLayerAndEnabledLayers()
  ; CHECK-SAME: knownLayers = [@A, @B::@C]
  ; CHECK-SAME: layers = [@A, @B]
  extmodule ExtModuleWithKnownLayerAndEnabledLayers enablelayer A knownlayer A enablelayer B knownlayer B.C:

  public module KnownLayers enablelayer A enablelayer B.C:
    ; CHECK: firrtl.instance ext1 interesting_name @ExtModuleWithKnownLayers()
    inst ext1 of ExtModuleWithKnownLayers
    ; CHECK: firrtl.instance ext2 interesting_name {layers = [@A, @B]} @ExtModuleWithKnownLayerAndEnabledLayers()
    inst ext2 of ExtModuleWithKnownLayerAndEnabledLayers

;// -----

FIRRTL version 3.3.0
; CHECK-LABEL: circuit "StaticShiftRight"
circuit StaticShiftRight:
  ; CHECK: firrtl.module @StaticShiftRight
  module StaticShiftRight:
    input a : UInt<8>
    input b : UInt<0>
    input c : SInt<8>
    input d : SInt<0>

    wire w : UInt
    connect w, a

    wire x : SInt
    connect x, c

    ; CHECK: %0 = firrtl.shr %a, 1
    ; CHECK: %1 = firrtl.pad %0, 1
    ; CHECK: %a_1 = firrtl.node {{.*}} %1 : !firrtl.uint<7>
    node a_1 = shr(a, 1)
    ; CHECK: %2 = firrtl.shr %a, 8
    ; CHECK: %3 = firrtl.pad %2, 1
    ; CHECK: %a_2 = firrtl.node {{.*}} %3 : !firrtl.uint<1>
    node a_2 = shr(a, 8)
    ; CHECK: %4 = firrtl.shr %a, 10
    ; CHECK: %5 = firrtl.pad %4, 1
    ; CHECK: %a_3 = firrtl.node {{.*}} %5 : !firrtl.uint<1>
    node a_3 = shr(a, 10)
    ; CHECK: %6 = firrtl.shr %b, 0
    ; CHECK: %7 = firrtl.pad %6, 1
    ; CHECK: %b_1 = firrtl.node {{.*}} %7 : !firrtl.uint<1>
    node b_1 = shr(b, 0)
    ; CHECK: %8 = firrtl.shr %b, 1
    ; CHECK: %9 = firrtl.pad %8, 1
    ; CHECK: %b_2 = firrtl.node {{.*}} %9 : !firrtl.uint<1>
    node b_2 = shr(b, 1)
    ; CHECK: %10 = firrtl.shr %w, 10
    ; CHECK: %11 = firrtl.pad %10, 1
    ; CHECK: %w_1 = firrtl.node {{.*}} %11 : !firrtl.uint
    node w_1 = shr(w, 10)

    ; CHECK: %12 = firrtl.shr %c, 1
    ; CHECK: %c_1 = firrtl.node {{.*}} %12 : !firrtl.sint<7>
    node c_1 = shr(c, 1)
    ; CHECK: %13 = firrtl.shr %c, 8
    ; CHECK: %c_2 = firrtl.node {{.*}} %13 : !firrtl.sint<1>
    node c_2 = shr(c, 8)
    ; CHECK: %14 = firrtl.shr %c, 10
    ; CHECK: %c_3 = firrtl.node {{.*}} %14 : !firrtl.sint<1>
    node c_3 = shr(c, 10)
    ; CHECK: %15 = firrtl.shr %d, 0
    ; CHECK: %d_1 = firrtl.node {{.*}} %15 : !firrtl.sint<1>
    node d_1 = shr(d, 0)
    ; CHECK: %16 = firrtl.shr %d, 1
    ; CHECK: %d_2 = firrtl.node {{.*}} %16 : !firrtl.sint<1>
    node d_2 = shr(d, 1)
    ; CHECK: %17 = firrtl.shr %x, 10
    ; CHECK: %x_1 = firrtl.node {{.*}} %17 : !firrtl.sint
    node x_1 = shr(x, 10)

;// -----

FIRRTL version 4.0.0
; CHECK-LABEL: circuit "StaticShiftRight"
circuit StaticShiftRight:
  ; CHECK: firrtl.module @StaticShiftRight
  public module StaticShiftRight:
    input a : UInt<8>
    input b : UInt<0>
    input c : SInt<8>
    input d : SInt<0>

    wire w : UInt
    connect w, a

    wire x : SInt
    connect x, c

    ; CHECK: %0 = firrtl.shr %a, 1
    ; CHECK: %a_1 = firrtl.node {{.*}} %0 : !firrtl.uint<7>
    node a_1 = shr(a, 1)
    ; CHECK: %1 = firrtl.shr %a, 8
    ; CHECK: %a_2 = firrtl.node {{.*}} %1 : !firrtl.uint<0>
    node a_2 = shr(a, 8)
    ; CHECK: %2 = firrtl.shr %a, 10
    ; CHECK: %a_3 = firrtl.node {{.*}} %2 : !firrtl.uint<0>
    node a_3 = shr(a, 10)
    ; CHECK: %3 = firrtl.shr %b, 0
    ; CHECK: %b_1 = firrtl.node {{.*}} %3 : !firrtl.uint<0>
    node b_1 = shr(b, 0)
    ; CHECK: %4 = firrtl.shr %b, 1
    ; CHECK: %b_2 = firrtl.node {{.*}} %4 : !firrtl.uint<0>
    node b_2 = shr(b, 1)
    ; CHECK: %5 = firrtl.shr %w, 10
    ; CHECK: %w_1 = firrtl.node {{.*}} %5 : !firrtl.uint
    node w_1 = shr(w, 10)

    ; CHECK: %6 = firrtl.shr %c, 1
    ; CHECK: %c_1 = firrtl.node {{.*}} %6 : !firrtl.sint<7>
    node c_1 = shr(c, 1)
    ; CHECK: %7 = firrtl.shr %c, 8
    ; CHECK: %c_2 = firrtl.node {{.*}} %7 : !firrtl.sint<1>
    node c_2 = shr(c, 8)
    ; CHECK: %8 = firrtl.shr %c, 10
    ; CHECK: %c_3 = firrtl.node {{.*}} %8 : !firrtl.sint<1>
    node c_3 = shr(c, 10)
    ; CHECK: %9 = firrtl.shr %d, 0
    ; CHECK: %d_1 = firrtl.node {{.*}} %9 : !firrtl.sint<1>
    node d_1 = shr(d, 0)
    ; CHECK: %10 = firrtl.shr %d, 1
    ; CHECK: %d_2 = firrtl.node {{.*}} %10 : !firrtl.sint<1>
    node d_2 = shr(d, 1)
    ; CHECK: %11 = firrtl.shr %x, 10
    ; CHECK: %x_1 = firrtl.node {{.*}} %11 : !firrtl.sint
    node x_1 = shr(x, 10)

;// -----
FIRRTL version 4.0.0
; CHECK-LABEL: circuit "GenericIntrinsics"
circuit GenericIntrinsics:
  ; CHECK: firrtl.module @GenericIntrinsics
  public module GenericIntrinsics:
    input clock : Clock
    input data : UInt<32>
    input c : UInt<1>

    ; Statements
    ; CHECK-NEXT: firrtl.int.generic "circt_verif_assert" %c : (!firrtl.uint<1>) -> ()
    intrinsic(circt_verif_assert, c)
    ; CHECK-NEXT: firrtl.int.generic "circt_fpga_probe" %data, %clock : (!firrtl.uint<32>, !firrtl.clock) -> ()
    intrinsic(circt_fpga_probe, data, clock)

    ; Expressions
    ; CHECK-NEXT: %[[PAV:.+]] = firrtl.int.generic "circt_plusargs_value" <FORMAT: none = "foo"> : () -> !firrtl.bundle<found: uint<1>, result: uint<5>>
    ; CHECK-NEXT: %n = firrtl.node interesting_name %[[PAV]]
    node n = intrinsic(circt_plusargs_value<FORMAT = "foo"> : { found : UInt<1>, result : UInt<5> })
    ; CHECK-NEXT: %[[PAT:.+]] = firrtl.int.generic "circt_plusargs_test" <FORMAT: none = "bar"> : () -> !firrtl.uint<1>
    ; CHECK-NEXT: %n2 = firrtl.node interesting_name %[[PAT]]
    node n2 = intrinsic(circt_plusargs_test<FORMAT = "bar"> : UInt<1>)


    ; Statement with unused return value.
    ; CHECK-NEXT: firrtl.int.generic "circt_clock_gate" %clock, %c : (!firrtl.clock, !firrtl.uint<1>) -> !firrtl.clock
    intrinsic(circt_clock_gate : Clock, clock, c)

    ; CHECK-NEXT: %[[SZ:.+]] = firrtl.int.generic "circt_isX"
    ; CHECK-NEXT: "circt_verif_assert" %[[SZ]]
    intrinsic(circt_verif_assert, intrinsic(circt_isX: UInt<1>, data))

;// -----
FIRRTL version 4.0.0
; CHECK-LABEL: circuit "Foo"
circuit Foo:

  ; CHECK-LABEL: firrtl.module @Foo()
  public module Foo:
    skip

  ; CHECK-LABEL: firrtl.module @Bar(in %a: !firrtl.uint<42>)
  public module Bar:
    input a: UInt<42>

  ; CHECK: firrtl.formal @someNameA, @Foo {bound = 20 : i32}
  formal someNameA of Foo, bound = 20

  ; CHECK: firrtl.formal @someNameB, @Bar {
  ; CHECK-SAME: a_int = 42
  ; CHECK-SAME: b_string = "hello"
  ; CHECK-SAME: c_array = [42 : ui32, "hello", [9001 : ui32], {foo = 1337 : ui32}]
  ; CHECK-SAME: d_map = {a = 42 : ui32, b = "hello", c = [9001 : ui32], d = {foo = 1337 : ui32}}
  ; CHECK-SAME: }
  formal someNameB of Bar:
    a_int = 42
    b_string = "hello"
    c_array = [42, "hello", [9001], {foo = 1337}]
    d_map = {a = 42, b = "hello", c = [9001], d = {foo = 1337}}

;// -----
FIRRTL version 5.1.0
; CHECK-LABEL: circuit "Foo"
circuit Foo:

  ; CHECK-LABEL: firrtl.extmodule @Foo(
  extmodule Foo:
    input clock: Clock
    input init: UInt<1>
    output done: UInt<1>
    output success: UInt<1>

  ; CHECK: firrtl.simulation @someNameA, @Foo {bound = 20 : i32}
  simulation someNameA of Foo, bound = 20

  ; CHECK: firrtl.simulation @someNameB, @Foo {
  ; CHECK-SAME: a_int = 42
  ; CHECK-SAME: b_string = "hello"
  ; CHECK-SAME: c_array = [42 : ui32, "hello", [9001 : ui32], {foo = 1337 : ui32}]
  ; CHECK-SAME: d_map = {a = 42 : ui32, b = "hello", c = [9001 : ui32], d = {foo = 1337 : ui32}}
  ; CHECK-SAME: }
  simulation someNameB of Foo:
    a_int = 42
    b_string = "hello"
    c_array = [42, "hello", [9001], {foo = 1337}]
    d_map = {a = 42, b = "hello", c = [9001], d = {foo = 1337}}

;// -----
FIRRTL version 3.9.0
circuit Foo:

  ; CHECK-LABEL: firrtl.extmodule private @Bar(in in: !firrtl.uint)
  extmodule Bar:
    input in: UInt

  ; CHECK-LABEL: firrtl.intmodule private @MyIntModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
  ; CHECK: attributes {intrinsic = "testIntrinsic1"}
  ; CHECK-NOT: {
  intmodule MyIntModule :
    input in: UInt
    output out: UInt<8>
    intrinsic = testIntrinsic1

  ; CHECK-LABEL: firrtl.intmodule private @MyParameterizedIntModule
  ; CHECK-SAME:    <FORMAT: none = "xyz_timeout=%d\0A",
  ; CHECK-SAME:     DEFAULT: ui32 = 0,
  ; CHECK-SAME:     WIDTH: ui32 = 32,
  ; CHECK-SAME:     DEPTH: f64 = 3.242000e+01>
  ; CHECK-SAME:    (in in: !firrtl.uint,
  ; CHECK-SAME:     out out: !firrtl.uint<8>)
  ; CHECK-SAME:    attributes {intrinsic = "testIntrinsic2"}
  ; CHECK-NOT: {
  intmodule MyParameterizedIntModule :
    input in: UInt
    output out: UInt<8>
    intrinsic = testIntrinsic2
    parameter FORMAT = "xyz_timeout=%d\n"
    parameter DEFAULT = 0
    parameter WIDTH = 32
    parameter DEPTH = 32.42

  ; CHECK-LABEL: extmodule private @RefExt(
  ; CHECK-SAME:  in in: !firrtl.uint<1>, out r: !firrtl.probe<uint<1>>
  ; CHECK-SAME: internalPaths = [#firrtl.internalpath, #firrtl.internalpath<"in">]
  extmodule RefExt :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    ref r is "in"

  ; CHECK-LABEL: extmodule private @RefExtMore(
  ; CHECK-SAME:  in in: !firrtl.uint<1>
  ; CHECK-SAME: out r: !firrtl.probe<uint<1>>
  ; CHECK-SAME: out data: !firrtl.uint<3>
  ; CHECK-SAME: out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>
  ; CHECK-SAME: internalPaths = [#firrtl.internalpath, #firrtl.internalpath<"path.to.internal.signal">, #firrtl.internalpath, #firrtl.internalpath<"in">]
  extmodule RefExtMore :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    output data : UInt<3>
    output r2 : Probe<{a : UInt<3>}[3]>
    ref r2 is "in"
    ref r is "path.to.internal.signal"

  ; CHECK-LABEL: module @Foo(
  ; CHECK-SAME:    out %out: !firrtl.uint,
  module Foo:
    output out: UInt
    output out3 : UInt<3>

    ; CHECK: %{{.+}}, %[[REM_R:.+]], %{{.+}}, %[[REM_R2:.+]] = firrtl.instance rem
    ; CHECK-NEXT: %[[REM_R2_1:.+]] = firrtl.ref.sub %[[REM_R2]][1]
    ; CHECK-NEXT: %[[REM_R2_1_A:.+]] = firrtl.ref.sub %[[REM_R2_1]][0]
    ; CHECK-NEXT: %[[READ_REM_R2_1_A:.+]] = firrtl.ref.resolve %[[REM_R2_1_A]]
    ; CHECK-NEXT: connect %out3, %[[READ_REM_R2_1_A]]
    inst rem of RefExtMore
    connect out3, read(rem.r2[1].a)

;// -----
FIRRTL version 2.9.0
circuit Foo:
  module Bar:
    input a: UInt<1>

  module Foo:
    input a: UInt<1>
    output b: UInt<1>

    node binary = UInt<4>("b1010")
    node octal = UInt<4>("o12")
    node decimal = UInt<4>(10)
    node hexadecimal = UInt<4>("ha")

    inst bar of Bar
    bar is invalid

    b <= a

;// -----
FIRRTL version 5.1.0
; CHECK-LABEL: firrtl.circuit "Contracts"
circuit Contracts:
  ; CHECK-LABEL: firrtl.module @Contracts(
  ; CHECK-SAME:    in %a: !firrtl.uint<42>
  ; CHECK-SAME:    in %b: !firrtl.uint<1337>
  ; CHECK-SAME:    in %c: !firrtl.bundle<x: uint<4>, y: uint<5>>
  ; CHECK-SAME:  )
  public module Contracts:
    input a : UInt<42>
    input b : UInt<1337>
    input c : {x: UInt<4>, y: UInt<5>}

    ; CHECK:      firrtl.contract {
    ; CHECK-NEXT: }
    contract: @[FooBar.scala 369:27]

    ; CHECK:      [[C1:%.+]] = firrtl.contract %a : !firrtl.uint<42> {
    ; CHECK-NEXT: ^bb0(%arg0: !firrtl.uint<42>):
    ; CHECK-NEXT:   %t2 = firrtl.node {{.+}} %arg0 : !firrtl.uint<42>
    ; CHECK-NEXT: }
    contract t = a:
      node t2 = t

    ; CHECK:      [[C2:%.+]]:3 = firrtl.contract %a, %b, %c : !firrtl.uint<42>, !firrtl.uint<1337>, !firrtl.bundle<x: uint<4>, y: uint<5>> {
    ; CHECK-NEXT: ^bb0(%arg0: !firrtl.uint<42>, %arg1: !firrtl.uint<1337>, %arg2: !firrtl.bundle<x: uint<4>, y: uint<5>>):
    ; CHECK-NEXT:   %u2 = firrtl.node {{.+}} %arg0 : !firrtl.uint<42>
    ; CHECK-NEXT:   %v2 = firrtl.node {{.+}} %arg1 : !firrtl.uint<1337>
    ; CHECK-NEXT:   %w2 = firrtl.node {{.+}} %arg2 : !firrtl.bundle<x: uint<4>, y: uint<5>>
    ; CHECK-NEXT: }
    contract u, v, w = a, b, c:
      node u2 = u
      node v2 = v
      node w2 = w

    ; CHECK: %t3 = firrtl.node {{.+}} [[C1]] : !firrtl.uint<42>
    ; CHECK: %u3 = firrtl.node {{.+}} [[C2]]#0 : !firrtl.uint<42>
    ; CHECK: %v3 = firrtl.node {{.+}} [[C2]]#1 : !firrtl.uint<1337>
    ; CHECK: %w3 = firrtl.node {{.+}} [[C2]]#2 : !firrtl.bundle<x: uint<4>, y: uint<5>>
    node t3 = t
    node u3 = u
    node v3 = v
    node w3 = w

;// -----
FIRRTL version 4.1.0
circuit Foo:
  public module Foo:
    input clock: Clock
    ; In FIRRTL versions < 4.2.0, the following is not a "special substitution".
    ; CHECK{{LITERAL}}: firrtl.printf %clock, %c1_ui1, "[{{SimulationTime}}]: hello from {{HierarchicalModuleName}}"
    printf(clock, UInt<1>(1), "[{{SimulationTime}}]: hello from {{HierarchicalModuleName}}")

;// -----
FIRRTL version 5.0.0
circuit Foo:
  public module Foo:
    input clock: Clock
    ; In FIRRTL versions >= 4.2.0, the following is not a "special substitution".
    ; CHECK:            %time = firrtl.fstring.time : !firrtl.fstring
    ; CHECK:            %hierarchicalmodulename = firrtl.fstring.hierarchicalmodulename : !firrtl.fstring
    ; CHECK{{LITERAL}}: firrtl.printf %clock, %c1_ui1, "[{{}}]: hello world {{}}"(%time, %hierarchicalmodulename)
    printf(clock, UInt<1>(1), "[{{SimulationTime}}]: hello from {{HierarchicalModuleName}}")

;// -----
FIRRTL version 5.1.0
circuit Foo:
  public module Foo:
    input clock: Clock
    input c: UInt<8>
    ; CHECK:            %[[HIERARCHICALMODULENAME_0:.+]] = firrtl.fstring.hierarchicalmodulename : !firrtl.fstring
    ; CHECK-NEXT:       %[[TIME_0:.+]] = firrtl.fstring.time : !firrtl.fstring
    ; CHECK-NEXT:       %[[TIME_1:.+]] = firrtl.fstring.time : !firrtl.fstring
    ; CHECK-NEXT:       %[[HIERARCHICALMODULENAME_1:.+]] = firrtl.fstring.hierarchicalmodulename : !firrtl.fstring
    ; CHECK{{LITERAL}}:      firrtl.fprintf %clock, %c1_ui1, "{{}}{{}}%d.txt" (%[[HIERARCHICALMODULENAME_0]], %[[TIME_0]], %c),
    ; CHECK-SAME{{LITERAL}}:   "[{{}}]: hello world {{}}"(%[[TIME_1]], %[[HIERARCHICALMODULENAME_1]])
    fprintf(clock, UInt<1>(1), "{{HierarchicalModuleName}}{{SimulationTime}}%d.txt", c, "[{{SimulationTime}}]: hello from {{HierarchicalModuleName}}")

;// -----
FIRRTL version 5.0.0
circuit Foo:
  public module Foo:
    input clock: Clock
    ; expected-error @+1 {{fprintf are a FIRRTL 5.1.0+ feature, but the specified FIRRTL version was 5.0.0}}
    fprintf(clock, UInt<1>(1), "test.txt", "[{{SimulationTime}}]: hello from {{HierarchicalModuleName}}")

;// -----
FIRRTL version 5.1.0
circuit Foo:
  public module Foo:
    input clock: Clock
    ; CHECK: firrtl.fflush %clock, %c1_ui1
    fflush(clock, UInt<1>(1))
    ; CHECK: firrtl.fflush %clock, %c1_ui1
    ; CHECK-SAME{{LITERAL}}: "{{}}{{}}%d.txt"(%hierarchicalmodulename, %time, %c), "[{{}}]: hello from {{}}"(%time_0, %hierarchicalmodulename_1) : !firrtl.clock, !firrtl.const.uint<1>, !firrtl.fstring, !firrtl.fstring, !firrtl.uint<8>, !firrtl.fstring, !firrtl.fstring
    fflush(clock, UInt<1>(1), "foo%d{{HierarchicalModuleName}}.txt", UInt<1>(1))

;// -----
FIRRTL version 5.0.0
circuit Foo:
  public module Foo:
    input clock: Clock
    ; expected-error @+1 {{fflush are a FIRRTL 5.1.0+ feature, but the specified FIRRTL version was 5.0.0}}
    fflush(clock, UInt<1>(1))

;// -----
FIRRTL version 5.0.0
circuit Foo:
  public module Foo:
    input a: UInt<1>
    ; expected-error @+1 {{variadic cat are a FIRRTL 5.1.0+ feature, but the specified FIRRTL version was 5.0.0}}
    node b = cat(a, a, a)

;// -----
FIRRTL version 5.1.0
; CHECK-LABEL: firrtl.circuit "NullaryCat"
circuit NullaryCat:
  public module NullaryCat:
    input a: UInt<1>
    ; CHECK: firrtl.cat : () -> !firrtl.uint<0>
    node b = cat()

;// -----
FIRRTL version 5.1.0
circuit Domains:
  ; CHECK-LABEL: firrtl.domain @ClockDomain
  domain ClockDomain:

  ; CHECK-LABEL: firrtl.module @Domains
  ; CHECK-SAME:    in %A: !firrtl.domain
  ; CHECK-SAME:    in %B: !firrtl.domain
  ; CHECK-SAME:    in %a: !firrtl.uint<1> domains [%A]
  ; CHECK-SAME:    in %ab: !firrtl.uint<1> domains [%A, %B]
  public module Domains:
    input A: Domain
    input B: Domain
    input a: UInt<1> domains [A]
    input ab: UInt<1> domains [A, B]
