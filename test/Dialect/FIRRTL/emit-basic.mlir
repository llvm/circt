// RUN: circt-translate --export-firrtl --verify-diagnostics %s -o %t
// RUN: cat %t | FileCheck %s --strict-whitespace
// RUN: circt-translate --import-firrtl %t --mlir-print-debuginfo | circt-translate --export-firrtl | diff - %t

// CHECK-LABEL: circuit Foo :
firrtl.circuit "Foo" {
  // CHECK-LABEL: module Foo :
  firrtl.module @Foo() {}

  // CHECK-LABEL: module PortsAndTypes :
  firrtl.module @PortsAndTypes(
    // CHECK-NEXT: input a00 : Clock
    // CHECK-NEXT: input a01 : Reset
    // CHECK-NEXT: input a02 : AsyncReset
    // CHECK-NEXT: input a03 : UInt
    // CHECK-NEXT: input a04 : SInt
    // CHECK-NEXT: input a05 : Analog
    // CHECK-NEXT: input a06 : UInt<42>
    // CHECK-NEXT: input a07 : SInt<42>
    // CHECK-NEXT: input a08 : Analog<42>
    // CHECK-NEXT: input a09 : { a : UInt, flip b : UInt }
    // CHECK-NEXT: input a10 : UInt[42]
    // CHECK-NEXT: output b0 : UInt
    in %a00: !firrtl.clock,
    in %a01: !firrtl.reset,
    in %a02: !firrtl.asyncreset,
    in %a03: !firrtl.uint,
    in %a04: !firrtl.sint,
    in %a05: !firrtl.analog,
    in %a06: !firrtl.uint<42>,
    in %a07: !firrtl.sint<42>,
    in %a08: !firrtl.analog<42>,
    in %a09: !firrtl.bundle<a: uint, b flip: uint>,
    in %a10: !firrtl.vector<uint, 42>,
    out %b0: !firrtl.uint
  ) {}

  // CHECK-LABEL: module Simple :
  // CHECK:         input someIn : UInt<1>
  // CHECK:         output someOut : UInt<1>
  firrtl.module @Simple(in %someIn: !firrtl.uint<1>, out %someOut: !firrtl.uint<1>) {
    firrtl.skip
  }

  // CHECK-LABEL: module Statements :
  firrtl.module @Statements(in %ui1: !firrtl.uint<1>, in %someAddr: !firrtl.uint<8>, in %someClock: !firrtl.clock, in %someReset: !firrtl.reset, out %someOut: !firrtl.uint<1>) {
    // CHECK: when ui1 :
    // CHECK:   skip
    firrtl.when %ui1 {
      firrtl.skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else :
    // CHECK:   skip
    firrtl.when %ui1 {
      firrtl.skip
    } else {
      firrtl.skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else when ui1 :
    // CHECK:   skip
    firrtl.when %ui1 {
      firrtl.skip
    } else {
      firrtl.when %ui1 {
        firrtl.skip
      }
    }
    // CHECK: wire someWire : UInt<1>
    %someWire = firrtl.wire : !firrtl.uint<1>
    // CHECK: reg someReg : UInt<1>, someClock
    %someReg = firrtl.reg %someClock : !firrtl.uint<1>
    // CHECK: reg someReg2 : UInt<1>, someClock with :
    // CHECK:   reset => (someReset, ui1)
    %someReg2 = firrtl.regreset %someClock, %someReset, %ui1 : !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: node someNode = ui1
    %someNode = firrtl.node %ui1 : !firrtl.uint<1>
    // CHECK: stop(someClock, ui1, 42) : foo
    firrtl.stop %someClock, %ui1, 42 {name = "foo"}
    // CHECK: skip
    firrtl.skip
    // CHECK: printf(someClock, ui1, "some\n magic\"stuff\"", ui1, someReset) : foo
    firrtl.printf %someClock, %ui1, "some\n magic\"stuff\"" {name = "foo"} (%ui1, %someReset) : !firrtl.uint<1>, !firrtl.reset
    // CHECK: assert(someClock, ui1, ui1, "msg") : foo
    // CHECK: assume(someClock, ui1, ui1, "msg") : foo
    // CHECK: cover(someClock, ui1, ui1, "msg") : foo
    firrtl.assert %someClock, %ui1, %ui1, "msg" {name = "foo"}
    firrtl.assume %someClock, %ui1, %ui1, "msg" {name = "foo"}
    firrtl.cover %someClock, %ui1, %ui1, "msg" {name = "foo"}
    // CHECK: someOut <= ui1
    firrtl.connect %someOut, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: inst someInst of Simple
    // CHECK: someInst.someIn <= ui1
    // CHECK: someOut <= someInst.someOut
    %someInst_someIn, %someInst_someOut = firrtl.instance someInst @Simple(in someIn: !firrtl.uint<1>, out someOut: !firrtl.uint<1>)
    firrtl.connect %someInst_someIn, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %someOut, %someInst_someOut : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: _invalid
    // CHECK: someOut is invalid
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %someOut, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: _invalid
    // CHECK: someOut is invalid
    %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.strictconnect %someOut, %invalid_ui2 : !firrtl.uint<1>
    // CHECK: attach(an0, an1)
    %an0 = firrtl.wire : !firrtl.analog<1>
    %an1 = firrtl.wire : !firrtl.analog<1>
    firrtl.attach %an0, %an1 : !firrtl.analog<1>, !firrtl.analog<1>

    // CHECK: node k0 = UInt<19>(42)
    // CHECK: node k1 = SInt<19>(42)
    // CHECK: node k2 = UInt(42)
    // CHECK: node k3 = SInt(42)
    %0 = firrtl.constant 42 : !firrtl.uint<19>
    %1 = firrtl.constant 42 : !firrtl.sint<19>
    %2 = firrtl.constant 42 : !firrtl.uint
    %3 = firrtl.constant 42 : !firrtl.sint
    %k0 = firrtl.node %0 : !firrtl.uint<19>
    %k1 = firrtl.node %1 : !firrtl.sint<19>
    %k2 = firrtl.node %2 : !firrtl.uint
    %k3 = firrtl.node %3 : !firrtl.sint

    // CHECK: node k4 = asClock(UInt<1>(0))
    // CHECK: node k5 = asAsyncReset(UInt<1>(0))
    // CHECK: node k6 = UInt<1>(0)
    %4 = firrtl.specialconstant 0 : !firrtl.clock
    %5 = firrtl.specialconstant 0 : !firrtl.asyncreset
    %6 = firrtl.specialconstant 0 : !firrtl.reset
    %k4 = firrtl.node %4 : !firrtl.clock
    %k5 = firrtl.node %5 : !firrtl.asyncreset
    %k6 = firrtl.node %6 : !firrtl.reset

    // CHECK: wire bundle : { a : UInt, flip b : UInt }
    // CHECK: wire vector : UInt[42]
    // CHECK: node subfield = bundle.a
    // CHECK: node subindex = vector[19]
    // CHECK: node subaccess = vector[ui1]
    %bundle = firrtl.wire : !firrtl.bundle<a: uint, b flip: uint>
    %vector = firrtl.wire : !firrtl.vector<uint, 42>
    %subfield_tmp = firrtl.subfield %bundle[a] : !firrtl.bundle<a: uint, b flip: uint>
    %subindex_tmp = firrtl.subindex %vector[19] : !firrtl.vector<uint, 42>
    %subaccess_tmp = firrtl.subaccess %vector[%ui1] : !firrtl.vector<uint, 42>, !firrtl.uint<1>
    %subfield = firrtl.node %subfield_tmp : !firrtl.uint
    %subindex = firrtl.node %subindex_tmp : !firrtl.uint
    %subaccess = firrtl.node %subaccess_tmp : !firrtl.uint

    %x = firrtl.node %2 : !firrtl.uint
    %y = firrtl.node %2 : !firrtl.uint

    // CHECK: node addPrimOp = add(x, y)
    // CHECK: node subPrimOp = sub(x, y)
    // CHECK: node mulPrimOp = mul(x, y)
    // CHECK: node divPrimOp = div(x, y)
    // CHECK: node remPrimOp = rem(x, y)
    // CHECK: node andPrimOp = and(x, y)
    // CHECK: node orPrimOp = or(x, y)
    // CHECK: node xorPrimOp = xor(x, y)
    // CHECK: node leqPrimOp = leq(x, y)
    // CHECK: node ltPrimOp = lt(x, y)
    // CHECK: node geqPrimOp = geq(x, y)
    // CHECK: node gtPrimOp = gt(x, y)
    // CHECK: node eqPrimOp = eq(x, y)
    // CHECK: node neqPrimOp = neq(x, y)
    // CHECK: node catPrimOp = cat(x, y)
    // CHECK: node dShlPrimOp = dshl(x, y)
    // CHECK: node dShlwPrimOp = dshlw(x, y)
    // CHECK: node dShrPrimOp = dshr(x, y)
    %addPrimOp_tmp = firrtl.add %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %subPrimOp_tmp = firrtl.sub %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %mulPrimOp_tmp = firrtl.mul %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %divPrimOp_tmp = firrtl.div %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %remPrimOp_tmp = firrtl.rem %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %andPrimOp_tmp = firrtl.and %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %orPrimOp_tmp = firrtl.or %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %xorPrimOp_tmp = firrtl.xor %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %leqPrimOp_tmp = firrtl.leq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %ltPrimOp_tmp = firrtl.lt %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %geqPrimOp_tmp = firrtl.geq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %gtPrimOp_tmp = firrtl.gt %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %eqPrimOp_tmp = firrtl.eq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %neqPrimOp_tmp = firrtl.neq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %catPrimOp_tmp = firrtl.cat %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShlPrimOp_tmp = firrtl.dshl %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShlwPrimOp_tmp = firrtl.dshlw %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShrPrimOp_tmp = firrtl.dshr %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %addPrimOp = firrtl.node %addPrimOp_tmp : !firrtl.uint
    %subPrimOp = firrtl.node %subPrimOp_tmp : !firrtl.uint
    %mulPrimOp = firrtl.node %mulPrimOp_tmp : !firrtl.uint
    %divPrimOp = firrtl.node %divPrimOp_tmp : !firrtl.uint
    %remPrimOp = firrtl.node %remPrimOp_tmp : !firrtl.uint
    %andPrimOp = firrtl.node %andPrimOp_tmp : !firrtl.uint
    %orPrimOp = firrtl.node %orPrimOp_tmp : !firrtl.uint
    %xorPrimOp = firrtl.node %xorPrimOp_tmp : !firrtl.uint
    %leqPrimOp = firrtl.node %leqPrimOp_tmp : !firrtl.uint<1>
    %ltPrimOp = firrtl.node %ltPrimOp_tmp : !firrtl.uint<1>
    %geqPrimOp = firrtl.node %geqPrimOp_tmp : !firrtl.uint<1>
    %gtPrimOp = firrtl.node %gtPrimOp_tmp : !firrtl.uint<1>
    %eqPrimOp = firrtl.node %eqPrimOp_tmp : !firrtl.uint<1>
    %neqPrimOp = firrtl.node %neqPrimOp_tmp : !firrtl.uint<1>
    %catPrimOp = firrtl.node %catPrimOp_tmp : !firrtl.uint
    %dShlPrimOp = firrtl.node %dShlPrimOp_tmp : !firrtl.uint
    %dShlwPrimOp = firrtl.node %dShlwPrimOp_tmp : !firrtl.uint
    %dShrPrimOp = firrtl.node %dShrPrimOp_tmp : !firrtl.uint

    // CHECK: node asSIntPrimOp = asSInt(x)
    // CHECK: node asUIntPrimOp = asUInt(x)
    // CHECK: node asAsyncResetPrimOp = asAsyncReset(x)
    // CHECK: node asClockPrimOp = asClock(x)
    // CHECK: node cvtPrimOp = cvt(x)
    // CHECK: node negPrimOp = neg(x)
    // CHECK: node notPrimOp = not(x)
    // CHECK: node andRPrimOp = andr(x)
    // CHECK: node orRPrimOp = orr(x)
    // CHECK: node xorRPrimOp = xorr(x)
    %asSIntPrimOp_tmp = firrtl.asSInt %x : (!firrtl.uint) -> !firrtl.sint
    %asUIntPrimOp_tmp = firrtl.asUInt %x : (!firrtl.uint) -> !firrtl.uint
    %asAsyncResetPrimOp_tmp = firrtl.asAsyncReset %x : (!firrtl.uint) -> !firrtl.asyncreset
    %asClockPrimOp_tmp = firrtl.asClock %x : (!firrtl.uint) -> !firrtl.clock
    %cvtPrimOp_tmp = firrtl.cvt %x : (!firrtl.uint) -> !firrtl.sint
    %negPrimOp_tmp = firrtl.neg %x : (!firrtl.uint) -> !firrtl.sint
    %notPrimOp_tmp = firrtl.not %x : (!firrtl.uint) -> !firrtl.uint
    %andRPrimOp_tmp = firrtl.andr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %orRPrimOp_tmp = firrtl.orr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %xorRPrimOp_tmp = firrtl.xorr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %asSIntPrimOp = firrtl.node %asSIntPrimOp_tmp : !firrtl.sint
    %asUIntPrimOp = firrtl.node %asUIntPrimOp_tmp : !firrtl.uint
    %asAsyncResetPrimOp = firrtl.node %asAsyncResetPrimOp_tmp : !firrtl.asyncreset
    %asClockPrimOp = firrtl.node %asClockPrimOp_tmp : !firrtl.clock
    %cvtPrimOp = firrtl.node %cvtPrimOp_tmp : !firrtl.sint
    %negPrimOp = firrtl.node %negPrimOp_tmp : !firrtl.sint
    %notPrimOp = firrtl.node %notPrimOp_tmp : !firrtl.uint
    %andRPrimOp = firrtl.node %andRPrimOp_tmp : !firrtl.uint<1>
    %orRPrimOp = firrtl.node %orRPrimOp_tmp : !firrtl.uint<1>
    %xorRPrimOp = firrtl.node %xorRPrimOp_tmp : !firrtl.uint<1>

    // CHECK: node bitsPrimOp = bits(x, 4, 2)
    // CHECK: node headPrimOp = head(x, 4)
    // CHECK: node tailPrimOp = tail(x, 4)
    // CHECK: node padPrimOp = pad(x, 16)
    // CHECK: node muxPrimOp = mux(ui1, x, y)
    // CHECK: node shlPrimOp = shl(x, 4)
    // CHECK: node shrPrimOp = shr(x, 4)
    %bitsPrimOp_tmp = firrtl.bits %x 4 to 2 : (!firrtl.uint) -> !firrtl.uint<3>
    %headPrimOp_tmp = firrtl.head %x, 4 : (!firrtl.uint) -> !firrtl.uint<4>
    %tailPrimOp_tmp = firrtl.tail %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %padPrimOp_tmp = firrtl.pad %x, 16 : (!firrtl.uint) -> !firrtl.uint
    %muxPrimOp_tmp = firrtl.mux(%ui1, %x, %y) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %shlPrimOp_tmp = firrtl.shl %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %shrPrimOp_tmp = firrtl.shr %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %bitsPrimOp = firrtl.node %bitsPrimOp_tmp : !firrtl.uint<3>
    %headPrimOp = firrtl.node %headPrimOp_tmp : !firrtl.uint<4>
    %tailPrimOp = firrtl.node %tailPrimOp_tmp : !firrtl.uint
    %padPrimOp = firrtl.node %padPrimOp_tmp : !firrtl.uint
    %muxPrimOp = firrtl.node %muxPrimOp_tmp : !firrtl.uint
    %shlPrimOp = firrtl.node %shlPrimOp_tmp : !firrtl.uint
    %shrPrimOp = firrtl.node %shrPrimOp_tmp : !firrtl.uint

    %MyMem_a, %MyMem_b, %MyMem_c = firrtl.mem Undefined {depth = 8, name = "MyMem", portNames = ["a", "b", "c"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<4>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint<4>, wmode: uint<1>, wdata: uint<4>, wmask: uint<1>>
    %MyMem_a_clk = firrtl.subfield %MyMem_a[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<4>>
    %MyMem_b_clk = firrtl.subfield %MyMem_b[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
    %MyMem_c_clk = firrtl.subfield %MyMem_c[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint<4>, wmode: uint<1>, wdata: uint<4>, wmask: uint<1>>
    firrtl.connect %MyMem_a_clk, %someClock : !firrtl.clock, !firrtl.clock
    firrtl.connect %MyMem_b_clk, %someClock : !firrtl.clock, !firrtl.clock
    firrtl.connect %MyMem_c_clk, %someClock : !firrtl.clock, !firrtl.clock
    // CHECK:       mem MyMem :
    // CHECK-NEXT:    data-type => UInt<4>
    // CHECK-NEXT:    depth => 8
    // CHECK-NEXT:    read-latency => 0
    // CHECK-NEXT:    write-latency => 1
    // CHECK-NEXT:    reader => a
    // CHECK-NEXT:    writer => b
    // CHECK-NEXT:    readwriter => c
    // CHECK-NEXT:    read-under-write => undefined
    // CHECK-NEXT:  MyMem.a.clk <= someClock
    // CHECK-NEXT:  MyMem.b.clk <= someClock
    // CHECK-NEXT:  MyMem.c.clk <= someClock

    %combmem = chirrtl.combmem : !chirrtl.cmemory<uint<3>, 256>
    %port0_data, %port0_port = chirrtl.memoryport Infer %combmem {name = "port0"} : (!chirrtl.cmemory<uint<3>, 256>) -> (!firrtl.uint<3>, !chirrtl.cmemoryport)
    firrtl.when %ui1 {
      chirrtl.memoryport.access %port0_port[%someAddr], %someClock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    }
    // CHECK:      cmem combmem : UInt<3>[256]
    // CHECK-NEXT: when ui1 :
    // CHECK-NEXT:   infer mport port0 = combmem[someAddr], someClock

    %seqmem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<3>, 256>
    %port1_data, %port1_port = chirrtl.memoryport Infer %seqmem {name = "port1"} : (!chirrtl.cmemory<uint<3>, 256>) -> (!firrtl.uint<3>, !chirrtl.cmemoryport)
    firrtl.when %ui1 {
      chirrtl.memoryport.access %port1_port[%someAddr], %someClock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    }
    // CHECK:      smem seqmem : UInt<3>[256] undefined
    // CHECK-NEXT: when ui1 :
    // CHECK-NEXT:   infer mport port1 = seqmem[someAddr], someClock

    firrtl.connect %port0_data, %port1_data : !firrtl.uint<3>, !firrtl.uint<3>
    // CHECK: port0 <= port1

    %invalid_clock = firrtl.invalidvalue : !firrtl.clock
    %dummyReg = firrtl.reg %invalid_clock : !firrtl.uint<42>
    // CHECK: wire [[INV:_invalid.*]] : Clock
    // CHECK-NEXT: [[INV]] is invalid
    // CHECK-NEXT: reg dummyReg : UInt<42>, [[INV]]
  }

  firrtl.extmodule @MyParameterizedExtModule<DEFAULT: i64 = 0, DEPTH: f64 = 3.242000e+01, FORMAT: none = "xyz_timeout=%d\0A", WIDTH: i8 = 32>(in in: !firrtl.uint, out out: !firrtl.uint<8>) attributes {defname = "name_thing"}
  // CHECK-LABEL: extmodule MyParameterizedExtModule :
  // CHECK-NEXT:    input in : UInt
  // CHECK-NEXT:    output out : UInt<8>
  // CHECK-NEXT:    defname = name_thing
  // CHECK-NEXT:    parameter DEFAULT = 0
  // CHECK-NEXT:    parameter DEPTH = 32.42
  // CHECK-NEXT:    parameter FORMAT = "xyz_timeout=%d\n"
  // CHECK-NEXT:    parameter WIDTH = 32
}
