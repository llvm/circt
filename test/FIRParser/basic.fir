; RUN: circt-translate -parse-fir -verify-diagnostics %s | circt-opt | FileCheck %s

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
    input reset : UInt<1>    ; CHECK: %reset: !firrtl.uint<1>,
    input clock : Clock      ; CHECK: %clock: !firrtl.clock,
    output auto : UInt<1>    ; CHECK: %auto: !firrtl.flip<uint<1>>,
    input i8 : UInt<8>       ; CHECK: %i8: !firrtl.uint<8>,
    input s8 : SInt<8>       ; CHECK: %s8: !firrtl.sint<8>,
    input a8 : Analog<8>     ; CHECK: %a8: !firrtl.analog<8>)

    ; CHECK: %_t = firrtl.wire {name = "_t"} : !firrtl.vector<uint<1>, 12>
    wire _t : UInt<1>[12] @[Nodes.scala 370:76]

    ; CHECK: %_t_2 = firrtl.wire {name = "_t_2"} : !firrtl.vector<uint<1>, 13>
    wire _t_2 : UInt<1>[13]

    ; CHECK: %out_0 = firrtl.wire {name = "out_0"} : !firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}

    ; CHECK: firrtl.connect %_t, %_t_2 : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 13>
    _t <= _t_2

    ; CHECK: firrtl.partialconnect %_t, %_t_2
    _t <- _t_2

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

    ; CHECK: %3 = firrtl.subindex %_t_2[0] : (!firrtl.vector<uint<1>, 13>) -> !firrtl.uint<1>
    ; CHECK: %4 = firrtl.subindex %_t[0] : (!firrtl.vector<uint<1>, 12>) -> !firrtl.uint<1>
    ; CHECK: firrtl.connect %3, %4
    _t_2[0] <= _t[0] @[Xbar.scala 21:44]

    ; CHECK: %n1 = firrtl.node %auto {name = "n1"} : !firrtl.flip<uint<1>>
    node n1 = auto

    ; CHECK: firrtl.add %reset, %reset : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    node n2 = add(reset, reset)

    ; CHECK: firrtl.asClock %reset : (!firrtl.uint<1>) -> !firrtl.clock
    node n3 = asClock(reset)

    ; CHECK: firrtl.asAsyncReset %reset : (!firrtl.uint<1>) -> !firrtl.asyncreset
    node ar = asAsyncReset(reset)

    ; CHECK: %c42_ui10 = firrtl.constant(42 : ui10) : !firrtl.uint<10>
    ; CHECK: %c171_ui8 = firrtl.constant(171 : ui8) : !firrtl.uint<8>
    ; CHECK: firrtl.add %c42_ui10, %c171_ui8
    ; CHECK: firrtl.connect %auto
    auto <= add(UInt<10>(42), UInt<8>("hAB"))

    ; CHECK: %c-85_si8 = firrtl.constant(-85 : si8) : !firrtl.sint<8>
    auto <= add(s8, SInt<8>("hAB"))

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

    ; CHECK: firrtl.printf %clock, %reset, "Something interesting!\0A %x %x"(%_t, %_t_2) : !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 13>
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

    ; CHECK: firrtl.cat %{{.*}}, %{{.*}} : (!firrtl.uint<12>, !firrtl.uint<23>) -> !firrtl.uint<35>
    node n7 = cat(n5, n6)

    ; CHECK: firrtl.mux(%reset, %i8, %{{.*}}) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint) -> !firrtl.uint
    node n8 = mux(reset, i8, UInt(4))

    ; CHECK: firrtl.reginit %clock, %reset, %{{.*}} {name = "_t_2621"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>) -> !firrtl.uint<4>
    reg _t_2621 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0")) @[Edges.scala 230:27]

    ; CHECK: firrtl.reginit %clock, %reset, %{{.*}} {name = "_t_1601"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<2>
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

    ; CHECK: [[xyz:%.+]] = firrtl.instance @circuit {name = "xyz"} : !firrtl.flip<bundle<in: uint<80>>>
    inst xyz of circuit
    ; CHECK: [[xyz_in:%.+]] = firrtl.subfield [[xyz]]("in") : (!firrtl.flip<bundle<in: uint<80>>>) -> !firrtl.flip<uint<80>>
    ; CHECK: firrtl.connect [[xyz_in]], %i8 : !firrtl.flip<uint<80>>, !firrtl.uint<8>
    xyz.in <= i8

    ; CHECK: [[myext:%.+]] = firrtl.instance @MyExtModule {name = "myext"} : !firrtl.bundle<in: flip<uint>, out: uint<8>>
    inst myext of MyExtModule
    myext.in <= i8
    printf(clock, reset, "Something interesting! %x", myext.out)

    ; CHECK: firrtl.subaccess %_t[%i8] : (!firrtl.vector<uint<1>, 12>, !firrtl.uint<8>) -> !firrtl.uint<1>
    auto <= _t[i8]

    ; CHECK: [[autoP:%.+]] = firrtl.asPassive %auto : (!firrtl.flip<uint<1>>) -> !firrtl.uint<1>
    ; CHECK: firrtl.subaccess %_t{{\[}}[[autoP]]] : (!firrtl.vector<uint<1>, 12>, !firrtl.uint<1>) -> !firrtl.uint<1>
    auto <= _t[auto]

    ; CHECK: %myMem = firrtl.cmem {name = "myMem"} : !firrtl.vector<bundle<id: uint<4>, resp: uint<2>>, 8>
    cmem myMem : { id : UInt<4>, resp : UInt<2>} [8] @[Decoupled.scala 209:24]

    ; CHECK: %memValue = firrtl.memoryport "Infer" %myMem, %i8, %clock {name = "memValue"} : (!firrtl.vector<bundle<id: uint<4>, resp: uint<2>>, 8>, !firrtl.uint<8>, !firrtl.clock) -> !firrtl.bundle<id: uint<4>, resp: uint<2>>
    infer mport memValue = myMem[i8], clock
    auto <= memValue.id

    ; CHECK: %base_table_0 = firrtl.smem "Undefined" {name = "base_table_0"} : !firrtl.vector<vector<uint<1>, 9>, 256>
    smem base_table_0 : UInt<1>[9] [256]
    ; CHECK: %base_table_1 = firrtl.smem "Old" {name = "base_table_1"} : !firrtl.vector<vector<uint<1>, 9>, 256>
    smem base_table_1 : UInt<1>[9] [256] old

    ; CHECK: %tableValue = firrtl.memoryport "Read" %base_table_0, %i8, %clock {name = "tableValue"} : (!firrtl.vector<vector<uint<1>, 9>, 256>, !firrtl.uint<8>, !firrtl.clock) -> !firrtl.vector<uint<1>, 9>
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

    ; CHECK: [[autoP:%.+]] = firrtl.asPassive %auto : (!firrtl.flip<uint<1>>) -> !firrtl.uint<1>
    ; CHECK-NEXT: = firrtl.not [[autoP]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
    node n13 = not(auto)


    ; CHECK: %_M = firrtl.mem "Undefined" {depth = 8 : i64, name = "_M",
    ; CHECK:   readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<
    ; CHECK:       _T_10: bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: analog<4>>, mask: bundle<id: uint<1>>>,
    ; CHECK:       _T_11: bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: analog<4>>, mask: bundle<id: uint<1>>>,
    ; CHECK:       _T_18: bundle<addr: uint<3>, en: uint<1>, clk: clock, data: flip<bundle<id: analog<4>>>>>
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

    ; CHECK: firrtl.attach(%a8, %a8, %a8) :
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

    ; CHECK: %flip1 = firrtl.wire {{.*}} !firrtl.bundle<x: bundle<a: uint>>
    wire flip1 : { flip x : { flip a : UInt } }
    ; CHECK: %flip2 = firrtl.wire {{.*}} !firrtl.bundle<x: bundle<a: uint, b: flip<analog>>>
    wire flip2 : { flip x : { flip a : UInt, b: Analog } }
    ; CHECK: %flip3 = firrtl.wire {{.*}} !firrtl.bundle<x: bundle<a: uint, b: analog>>
    wire flip3 : { flip x : { flip a : UInt, flip b: Analog } }
    ; CHECK: %flip4 = firrtl.wire {{.*}} !firrtl.bundle<x: vector<bundle<a: uint>, 4>>
    wire flip4 : { flip x : { flip a : UInt }[4] }


  ; CHECK-LABEL: firrtl.module @expr_stmt_ambiguity(
  module expr_stmt_ambiguity :
    ; CHECK: %reg = firrtl.wire {name = "reg"} : !firrtl.uint
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
    ; CHECK: %memory = firrtl.smem "Undefined" {name = "memory"} : !firrtl.vector<vector<uint<1>, 9>, 256>
    smem memory : UInt<1>[9] [256]

    ; CHECK: firrtl.when %cond {
    when cond :
      node idx = foo
      ; CHECK: %xyz = firrtl.memoryport "Read" %memory, %idx
      ; CHECK: firrtl.connect [[HACKWIRE]], %xyz
      read mport xyz = memory[idx], clock
    ; CHECK: }
    node sometest = eq(foo, UInt(42))

    ; CHECK: firrtl.when %cond {
    when cond :
      node idx = foo
      ; CHECK: %xyz2 = firrtl.memoryport "Read" %memory, %idx
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
    ; CHECK: %1 = firrtl.asPassive %0 : (!firrtl.flip<uint<1>>) -> !firrtl.uint<1>
    when _T :  ; CHECK: firrtl.when %1 {
      skip
  
 