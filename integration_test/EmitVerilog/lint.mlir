// REQUIRES: verilator
// RUN: circt-translate %s -export-verilog -verify-diagnostics > %t1.sv
// RUN: verilator --lint-only --top-module A %t1.sv
// RUN: verilator --lint-only --top-module AB %t1.sv
// RUN: verilator --lint-only --top-module shl %t1.sv
// RUN: verilator --lint-only --top-module TESTSIMPLE %t1.sv
// RUN: verilator --lint-only --top-module casts %t1.sv

module {
  rtl.module @B(%a: i1 { rtl.inout }) -> (i1 {rtl.name = "b"}, i1 {rtl.name = "c"}) {
    %0 = rtl.or %a, %a : i1
    %1 = rtl.and %a, %a : i1
    rtl.output %0, %1 : i1, i1
  }

  rtl.module @A(%d: i1, %e: i1) -> (i1 {rtl.name = "f"}) {
    %1 = rtl.mux %d, %d, %e : i1
    rtl.output %1 : i1
  }

  rtl.module @AAA(%d: i1, %e: i1) -> (i1 {rtl.name = "f"}) {
    %z = rtl.constant ( 0 : i1 ) : i1
    rtl.output %z : i1
  }

  rtl.module @AB(%w: i1, %x: i1) -> (i1 {rtl.name = "y"}, i1 {rtl.name = "z"}) {
    %w2 = rtl.instance "a1" @AAA(%w, %w1) : (i1, i1) -> (i1)
    %w1, %y = rtl.instance "b1" @B(%w2) : (i1) -> (i1, i1)
    rtl.output %y, %x : i1, i1
  }

  rtl.module @shl(%a: i1) -> (i1 {rtl.name = "b"}) {
    %0 = rtl.shl %a, %a : i1
    rtl.output %0 : i1
  }

  rtl.module @TESTSIMPLE(%a: i4, %b: i4, %cond: i1, %array: !rtl.array<10xi4>,
                         %uarray: !rtl.uarray<16xi8>) -> (
    %r0: i4, %r2: i4, %r4: i4, %r6: i4,
    %r7: i4, %r8: i4, %r9: i4, %r10: i4,
    %r11: i4, %r12: i4, %r13: i4, %r14: i4,
    %r15: i4, %r16: i1,
    %r17: i1, %r18: i1, %r19: i1, %r20: i1,
    %r21: i1, %r22: i1, %r23: i1, %r24: i1,
    %r25: i1, %r26: i1, %r27: i1, %r28: i1,
    %r29: i12, %r30: i2, %r31: i9, %r33: i4, %r34: i4,
    %r35: !rtl.array<3xi4>
    ) {

    %0 = rtl.add %a, %b : i4
    %2 = rtl.sub %a, %b : i4
    %4 = rtl.mul %a, %b : i4
    %6 = rtl.divu %a, %b : i4
    %7 = rtl.divs %a, %b : i4
    %8 = rtl.modu %a, %b : i4
    %9 = rtl.mods %a, %b : i4
    %10 = rtl.shl %a, %b : i4
    %11 = rtl.shru %a, %b : i4
    %12 = rtl.shrs %a, %b : i4
    %13 = rtl.or %a, %b : i4
    %14 = rtl.and %a, %b : i4
    %15 = rtl.xor %a, %b : i4
    %16 = rtl.icmp eq %a, %b : i4
    %17 = rtl.icmp ne %a, %b : i4
    %18 = rtl.icmp slt %a, %b : i4
    %19 = rtl.icmp sle %a, %b : i4
    %20 = rtl.icmp sgt %a, %b : i4
    %21 = rtl.icmp sge %a, %b : i4
    %22 = rtl.icmp ult %a, %b : i4
    %23 = rtl.icmp ule %a, %b : i4
    %24 = rtl.icmp ugt %a, %b : i4
    %25 = rtl.icmp uge %a, %b : i4
    %26 = rtl.andr %a : i4
    %27 = rtl.orr %a : i4
    %28 = rtl.xorr %a : i4
    %29 = rtl.concat %a, %a, %b : (i4, i4, i4) -> i12
    %30 = rtl.extract %a from 1 : (i4) -> i2
    %31 = rtl.sext %a : (i4) -> i9
    %33 = rtl.mux %cond, %a, %b : i4

    %allone = rtl.constant (15 : i4) : i4
    %34 = rtl.xor %a, %allone : i4

    %one = rtl.constant (1 : i4) : i4
    %aPlusOne = rtl.add %a, %one : i4
    %35 = rtl.array_slice %array at %aPlusOne: (!rtl.array<10xi4>) -> !rtl.array<3xi4>

    rtl.output %0, %2, %4, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %33, %34, %35:
     i4,i4, i4,i4,i4,i4,i4, i4,i4,i4,i4,i4,
     i4,i1,i1,i1,i1, i1,i1,i1,i1,i1, i1,i1,i1,i1,
     i12, i2,i9,i4, i4, !rtl.array<3xi4>
  }

  rtl.module @casts(%in1: i64) -> (%r1: !rtl.array<5xi8>) {
    %bits = rtl.bitcast %in1 : (i64) -> !rtl.array<64xi1>
    %idx = rtl.constant (10 : i6) : i6
    %midBits = rtl.array_slice %bits at %idx : (!rtl.array<64xi1>) -> !rtl.array<40xi1>
    %r1 = rtl.bitcast %midBits : (!rtl.array<40xi1>) -> !rtl.array<5xi8>
    rtl.output %r1 : !rtl.array<5xi8>
  }
}
