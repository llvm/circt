// RUN: circt-opt -lower-firrtl-to-rtl %s | FileCheck %s

module attributes {firrtl.mainModule = "Simple"} {

  // CHECK-LABEL: rtl.module @Simple
  rtl.module @Simple(%in1: i4, %in2: i2, %in3: i8) -> (%out4: i4, %out5: i4) {
    %in1c = firrtl.stdIntCast %in1 : (i4) -> !firrtl.uint<4>
    %in2c = firrtl.stdIntCast %in2 : (i2) -> !firrtl.uint<2>
    %in3c = firrtl.stdIntCast %in3 : (i8) -> !firrtl.sint<8>
    // CHECK-NEXT: [[OUT4:%.+]] = rtl.wire : !rtl.inout<i4>
    %out4 = firrtl.wire : !firrtl.flip<uint<4>>
    // CHECK-NEXT:  [[OUT5:%.+]] = rtl.wire : !rtl.inout<i4>
    %out5 = firrtl.wire : !firrtl.flip<uint<4>>

    // CHECK: [[ZERO4:%.+]] = rtl.constant(0 : i4) : i4
    // CHECK: rtl.connect [[OUT5]], [[ZERO4]] : i4
    %tmp1 = firrtl.invalidvalue : !firrtl.uint<4>
    firrtl.connect %out5, %tmp1 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK: rtl.constant(-4 : i4) : i4
    %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>

    // CHECK: rtl.constant(2 : i3) : i3
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>

    // CHECK: [[ZEXT:%.+]] = rtl.concat %false_0, %in1 : (i1, i4) -> i5
    // CHECK: [[ADD:%.+]] = rtl.add %c12_i5, [[ZEXT]] : i5
    %0 = firrtl.add %c12_ui4, %in1c : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    %1 = firrtl.asUInt %in1c : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[ZEXT1:%.+]] = rtl.concat %false_1, [[ADD]] : (i1, i5) -> i6
    // CHECK: [[ZEXT2:%.+]] = rtl.concat %c0_i2, %in1 : (i2, i4) -> i6
    // CHECK-NEXT: [[SUB:%.+]] = rtl.sub [[ZEXT1]], [[ZEXT2]] : i6
    %2 = firrtl.sub %0, %1 : (!firrtl.uint<5>, !firrtl.uint<4>) -> !firrtl.uint<6>

    %in2s = firrtl.asSInt %in2c : (!firrtl.uint<2>) -> !firrtl.sint<2>

    // CHECK: [[PADRES:%.+]] = rtl.sext %in2 : (i2) -> i3
    %3 = firrtl.pad %in2s, 3 : (!firrtl.sint<2>) -> !firrtl.sint<3>

    // CHECK: [[PADRES2:%.+]] = rtl.concat %c0_i2_2, %in2 : (i2, i2) -> i4
    %4 = firrtl.pad %in2c, 4 : (!firrtl.uint<2>) -> !firrtl.uint<4>

    // CHECK: [[IN2EXT:%.+]] = rtl.concat %c0_i2_3, %in2 : (i2, i2) -> i4
    // CHECK: [[XOR:%.+]] = rtl.xor [[IN2EXT]], [[PADRES2]] : i4
    %5 = firrtl.xor %in2c, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: rtl.and [[XOR]]
    %and = firrtl.and %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: rtl.or [[XOR]]
    %or = firrtl.or %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[CONCAT1:%.+]] = rtl.concat [[PADRES2]], [[XOR]] : (i4, i4) -> i8
    %6 = firrtl.cat %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>

    // CHECK: rtl.concat %in1, %in2
    %7 = firrtl.cat %in1c, %in2c : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>

    // CHECK-NEXT: rtl.connect [[OUT5]], [[PADRES2]] : i4
    firrtl.connect %out5, %4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: rtl.connect [[OUT4]], [[XOR]] : i4
    firrtl.connect %out4, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: %c0_i2_4 = rtl.constant
    // CHECK-NEXT: [[ZEXT:%.+]] = rtl.concat %c0_i2_4, %in2 : (i2, i2) -> i4
    // CHECK-NEXT: rtl.connect [[OUT4]], [[ZEXT]] : i4
    firrtl.connect %out4, %in2c : !firrtl.flip<uint<4>>, !firrtl.uint<2>

    // CHECK-NEXT: %test-name = rtl.wire : !rtl.inout<i4>
    firrtl.wire {name = "test-name"} : !firrtl.uint<4>

    // CHECK-NEXT: = rtl.wire : !rtl.inout<i2>
    firrtl.wire : !firrtl.uint<2>

    // CHECK-NEXT: = firrtl.wire : !firrtl.vector<uint<1>, 13>
    %_t_2 = firrtl.wire : !firrtl.vector<uint<1>, 13>

    // CHECK-NEXT: = firrtl.wire : !firrtl.vector<uint<2>, 13>
    %_t_3 = firrtl.wire : !firrtl.vector<uint<2>, 13>

    // CHECK-NEXT: = rtl.extract [[CONCAT1]] from 3 : (i8) -> i5
    %8 = firrtl.bits %6 7 to 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = rtl.extract [[CONCAT1]] from 5 : (i8) -> i3
    %9 = firrtl.head %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<3>

    // CHECK-NEXT: = rtl.extract [[CONCAT1]] from 0 : (i8) -> i5
    %10 = firrtl.tail %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = rtl.extract [[CONCAT1]] from 3 : (i8) -> i5
    %11 = firrtl.shr %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = rtl.constant(false) : i1
    %12 = firrtl.shr %6, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.extract %in3 from 7 : (i8) -> i1
    %13 = firrtl.shr %in3c, 8 : (!firrtl.sint<8>) -> !firrtl.sint<1>

    // CHECK-NEXT: [[ZERO:%.+]] = rtl.constant(0 : i3) : i3
    // CHECK-NEXT: = rtl.concat [[CONCAT1]], [[ZERO]] : (i8, i3) -> i11
    %14 = firrtl.shl %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<11>

    // CHECK-NEXT: = rtl.xorr [[CONCAT1]] : i8
    %15 = firrtl.xorr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.andr [[CONCAT1]] : i8
    %16 = firrtl.andr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.orr [[CONCAT1]] : i8
    %17 = firrtl.orr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: %c0_i6 = rtl.constant
    // CHECK-NEXT: [[ZEXTC1:%.+]] = rtl.concat %c0_i6, [[CONCAT1]] : (i6, i8) -> i14
    // CHECK-NEXT: %c0_i8 = rtl.constant
    // CHECK-NEXT: [[ZEXT2:%.+]] = rtl.concat %c0_i8, [[SUB]] : (i8, i6) -> i14
    // CHECK-NEXT: [[VAL18:%.+]] = rtl.mul  [[ZEXTC1]], [[ZEXT2]] : i14
    %18 = firrtl.mul %6, %2 : (!firrtl.uint<8>, !firrtl.uint<6>) -> !firrtl.uint<14>

    // CHECK-NEXT: [[IN3SEXT:%.+]] = rtl.sext %in3 : (i8) -> i9
    // CHECK-NEXT: [[PADRESSEXT:%.+]] = rtl.sext [[PADRES]] : (i3) -> i9
    // CHECK-NEXT: = rtl.divs [[IN3SEXT]], [[PADRESSEXT]] : i9
    %19 = firrtl.div %in3c, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<9>

    // CHECK-NEXT: [[IN3EX:%.+]] = rtl.sext [[PADRES]] : (i3) -> i8
    // CHECK-NEXT: [[MOD1:%.+]] = rtl.mods %in3, [[IN3EX]] : i8
    // CHECK-NEXT: = rtl.extract [[MOD1]] from 0 : (i8) -> i3
    %20 = firrtl.rem %in3c, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[IN4EX:%.+]] = rtl.sext [[PADRES]] : (i3) -> i8
    // CHECK-NEXT: [[MOD2:%.+]] = rtl.mods [[IN4EX]], %in3 : i8
    // CHECK-NEXT: = rtl.extract [[MOD2]] from 0 : (i8) -> i3
    %21 = firrtl.rem %3, %in3c : (!firrtl.sint<3>, !firrtl.sint<8>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[WIRE:%n1]] = rtl.wire : !rtl.inout<i2>
    // CHECK-NEXT: rtl.connect [[WIRE]], %in2 : i2
    %n1 = firrtl.node %in2c  {name = "n1"} : !firrtl.uint<2>

    // Nodes with no names are just dropped.
    %22 = firrtl.node %n1 : !firrtl.uint<2>

    // CHECK-NEXT: %false_{{.*}} = rtl.constant(false) : i1
    // CHECK-NEXT: [[CVT:%.+]] = rtl.concat %false_{{.*}}, %in2 : (i1, i2) -> i3
    %23 = firrtl.cvt %22 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // Will be dropped, here because this triggered a crash
    %s23 = firrtl.cvt %in3c : (!firrtl.sint<8>) -> !firrtl.sint<8>

    // CHECK-NEXT: %c-1_i3 = rtl.constant(-1 : i3) : i3
    // CHECK-NEXT: [[XOR:%.+]] = rtl.xor [[CVT]], %c-1_i3 : i3
    %24 = firrtl.not %23 : (!firrtl.sint<3>) -> !firrtl.uint<3>

    %s24 = firrtl.asSInt %24 : (!firrtl.uint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[SEXT:%.+]] = rtl.sext [[XOR]] : (i3) -> i4
    // CHECK-NEXT: [[ZERO4b:%.+]] = rtl.constant(0 : i4) : i4
    // CHECK-NEXT: [[SUB:%.+]] = rtl.sub [[ZERO4b]], [[SEXT]] : i4
    %25 = firrtl.neg %s24 : (!firrtl.sint<3>) -> !firrtl.sint<4>

    // CHECK-NEXT: [[CVT4:%.+]] = rtl.sext [[CVT]] : (i3) -> i4
    // CHECK-NEXT: rtl.mux {{.*}}, [[CVT4]], [[SUB]] : i4
    %26 = firrtl.mux(%17, %23, %25) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.sint<4>

    // Noop
    %27 = firrtl.validif %12, %18 : (!firrtl.uint<1>, !firrtl.uint<14>) -> !firrtl.uint<14>
    // CHECK-NEXT: rtl.andr
    %28 = firrtl.andr %27 : (!firrtl.uint<14>) -> !firrtl.uint<1>

    // CHECK-NEXT: %c0_i11 = rtl.constant(0 : i11)
    // CHECK-NEXT: [[XOREXT:%.+]] = rtl.concat %c0_i11, [[XOR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = rtl.shru [[XOREXT]], [[VAL18]] : i14
    // CHECK-NEXT: [[DSHR:%.+]] = rtl.extract [[SHIFT]] from 0 : (i14) -> i3
    %29 = firrtl.dshr %24, %18 : (!firrtl.uint<3>, !firrtl.uint<14>) -> !firrtl.uint<3>

    // CHECK-NEXT: %c0_i5 = rtl.constant(0 : i5)
    // CHECK-NEXT: = rtl.concat %c0_i5, {{.*}} : (i5, i3) -> i8
    // CHECK-NEXT: [[SHIFT:%.+]] = rtl.shrs %in3, {{.*}} : i8
    %a29 = firrtl.dshr %in3c, %9 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<8>

    // CHECK-NEXT: = rtl.sext %in3 : (i8) -> i15
    // CHECK-NEXT: %c0_i12 = rtl.constant(0 : i12)
    // CHECK-NEXT: = rtl.concat %c0_i12, [[DSHR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = rtl.shl {{.*}}, {{.*}} : i15
    %30 = firrtl.dshl %in3c, %29 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<15>

    // CHECK-NEXT: = rtl.shru [[DSHR]], [[DSHR]] : i3
    %dshlw = firrtl.dshlw %29, %29 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>

    // Issue #367: https://github.com/llvm/circt/issues/367
    // CHECK-NEXT: = rtl.sext {{.*}} : (i4) -> i14
    // CHECK-NEXT: [[SHIFT:%.+]] = rtl.shrs {{.*}}, {{.*}} : i14
    // CHECK-NEXT: = rtl.extract [[SHIFT]] from 0 : (i14) -> i4
    %31 = firrtl.dshr %25, %27 : (!firrtl.sint<4>, !firrtl.uint<14>) -> !firrtl.sint<4>

    // CHECK-NEXT: rtl.icmp ule {{.*}}, {{.*}} : i4
    %41 = firrtl.leq %in1c, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: rtl.icmp ult {{.*}}, {{.*}} : i4
    %42 = firrtl.lt %in1c, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: rtl.icmp uge {{.*}}, {{.*}} : i4
    %43 = firrtl.geq %in1c, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: rtl.icmp ugt {{.*}}, {{.*}} : i4
    %44 = firrtl.gt %in1c, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: rtl.icmp eq {{.*}}, {{.*}} : i4
    %45 = firrtl.eq %in1c, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: rtl.icmp ne {{.*}}, {{.*}} : i4
    %46 = firrtl.neq %in1c, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>

    // Noop
    %47 = firrtl.asClock %44 : (!firrtl.uint<1>) -> !firrtl.clock
    %48 = firrtl.asAsyncReset %44 : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // Issue #353
    // CHECK: [[PADRES_EXT:%.+]] = rtl.sext [[PADRES]] : (i3) -> i8
    // CHECK: = rtl.and %in3, [[PADRES_EXT]] : i8
    %49 = firrtl.and %in3c, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.uint<8>

    // Issue #355: https://github.com/llvm/circt/issues/355
    // CHECK: [[DIV:%.+]] = rtl.divu %c104_i10, %c306_i10 : i10
    // CHECK: = rtl.extract [[DIV]] from 0 : (i10) -> i8
    %c104_ui8 = firrtl.constant(104 : ui8) : !firrtl.uint<8>
    %c306_ui10 = firrtl.constant(306 : ui10) : !firrtl.uint<10>
    %50 = firrtl.div %c104_ui8, %c306_ui10 : (!firrtl.uint<8>, !firrtl.uint<10>) -> !firrtl.uint<8>

    // Issue #364: https://github.com/llvm/circt/issues/364
    // CHECK:      %c-873_i12 = rtl.constant(-873 : i12) : i12
    // CHECK-NEXT: %c0_i12_9 = rtl.constant(0 : i12) : i12
    // CHECK-NEXT: = rtl.sub %c0_i12_9, %c-873_i12 : i12
    %c1175_ui11 = firrtl.constant(1175 : ui11) : !firrtl.uint<11>
    %51 = firrtl.neg %c1175_ui11 : (!firrtl.uint<11>) -> !firrtl.sint<12>

    %out4c = firrtl.asPassive %out4 : !firrtl.flip<uint<4>>
    %out4d = firrtl.stdIntCast %out4c : (!firrtl.uint<4>) -> i4
    %out5c = firrtl.asPassive %out5 : !firrtl.flip<uint<4>>
    %out5d = firrtl.stdIntCast %out5c : (!firrtl.uint<4>) -> i4
    rtl.output %out4d, %out5d : i4, i4
  }

//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: rtl.module @Print
  rtl.module @Print(%clock: i1, %reset: i1, %a: i4, %b: i4) {
    %clock1 = firrtl.stdIntCast %clock : (i1) -> !firrtl.clock
    %reset1 = firrtl.stdIntCast %reset : (i1) -> !firrtl.uint<1>
    %a1 = firrtl.stdIntCast %a : (i4) -> !firrtl.uint<4>
    %b1 = firrtl.stdIntCast %b : (i4) -> !firrtl.uint<4>

    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     [[TV:%.+]] = sv.textual_value "`PRINTF_COND_" : i1
    // CHECK-NEXT:     [[AND:%.+]] = rtl.and [[TV]], %reset
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       sv.fwrite "No operands!\0A"
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
   firrtl.printf %clock1, %reset1, "No operands!\0A"

    // CHECK: [[ADD:%.+]] = rtl.add
    %0 = firrtl.add %a1, %a1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // CHECK: sv.fwrite "Hi %x %x\0A"({{.*}}) : i5, i4
    firrtl.printf %clock1, %reset1, "Hi %x %x\0A"(%0, %b1) : !firrtl.uint<5>, !firrtl.uint<4>

    firrtl.skip

    // CHECK: rtl.output
    rtl.output
   }



// module Stop3 :
//    input clock1: Clock
//    input clock2: Clock
//    input reset: UInt<1>
//    stop(clock1, reset, 42)
//    stop(clock2, reset, 0)

  // CHECK-LABEL: rtl.module @Stop
  rtl.module @Stop(%clock1: i1, %clock2: i1, %reset: i1) {
    %clock1c = firrtl.stdIntCast %clock1 : (i1) -> !firrtl.clock
    %clock2c = firrtl.stdIntCast %clock2 : (i1) -> !firrtl.clock
    %resetc = firrtl.stdIntCast %reset : (i1) -> !firrtl.uint<1>

    // CHECK-NEXT: sv.always posedge %clock1 {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     %0 = sv.textual_value "`STOP_COND_" : i1
    // CHECK-NEXT:     %1 = rtl.and %0, %reset : i1
    // CHECK-NEXT:     sv.if %1 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock1c, %resetc, 42

    // CHECK-NEXT: sv.always posedge %clock2 {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     %0 = sv.textual_value "`STOP_COND_" : i1
    // CHECK-NEXT:     %1 = rtl.and %0, %reset : i1
    // CHECK-NEXT:     sv.if %1 {
    // CHECK-NEXT:       sv.finish
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock2c, %resetc, 0
  }

// circuit Verification:
//   module Verification:
//     input clock: Clock
//     input aCond: UInt<8>
//     input aEn: UInt<8>
//     input bCond: UInt<1>
//     input bEn: UInt<1>
//     input cCond: UInt<1>
//     input cEn: UInt<1>
//     assert(clock, bCond, bEn, "assert0")
//     assume(clock, aCond, aEn, "assume0")
//     cover(clock,  cCond, cEn, "cover0")

  // CHECK-LABEL: rtl.module @Verification
  rtl.module @Verification(%clock: i1, %aCond: i1, %aEn: i1, %bCond: i1, %bEn: i1, %cCond: i1, %cEn: i1) {
    %clockC = firrtl.stdIntCast %clock : (i1) -> !firrtl.clock
    %aCondC = firrtl.stdIntCast %aCond : (i1) -> !firrtl.uint<1>
    %aEnC = firrtl.stdIntCast %aEn : (i1) -> !firrtl.uint<1>
    %bCondC = firrtl.stdIntCast %bCond : (i1) -> !firrtl.uint<1>
    %bEnC = firrtl.stdIntCast %bEn : (i1) -> !firrtl.uint<1>
    %cCondC = firrtl.stdIntCast %cCond : (i1) -> !firrtl.uint<1>
    %cEnC = firrtl.stdIntCast %cEn : (i1) -> !firrtl.uint<1>

    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.if %aEn {
    // CHECK-NEXT:     sv.assert %aCond : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assert %clockC, %aCondC, %aEnC, "assert0"
    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.if %bEn {
    // CHECK-NEXT:     sv.assume %bCond  : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assume %clockC, %bCondC, %bEnC, "assume0"
    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.if %cEn {
    // CHECK-NEXT:     sv.cover %cCond : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.cover %clockC, %cCondC, %cEnC, "cover0"
    // CHECK-NEXT: rtl.output
    rtl.output
  }

  rtl.module @bar(%io_cpu_flush: i1) {
    rtl.output
  }

  // CHECK-LABEL: rtl.module @foo
  rtl.module @foo() {
    // CHECK-NEXT:  %io_cpu_flush.wire = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT:  [[IO:%.+]] = rtl.read_inout %io_cpu_flush.wire
    %io_cpu_flush.wire = rtl.wire : !rtl.inout<i1>
    %io_cpu_flush.wireV = rtl.read_inout %io_cpu_flush.wire : !rtl.inout<i1>
    // CHECK-NEXT: rtl.instance "fetch"
    rtl.instance "fetch" @bar(%io_cpu_flush.wireV)  : (i1) -> ()
    %0 = firrtl.stdIntCast %io_cpu_flush.wireV : (i1) -> !firrtl.uint<1>

    %hits_1_7 = firrtl.node %0 {name = "hits_1_7"} : !firrtl.uint<1>
    // CHECK-NEXT:  %hits_1_7 = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT:  rtl.connect %hits_1_7, [[IO]] : i1
    %1455 = firrtl.asPassive %hits_1_7 : !firrtl.uint<1>
  }

  // https://github.com/llvm/circt/issues/314
  // CHECK-LABEL: rtl.module @issue314
  rtl.module @issue314(%inp2: i27, %inpi: i65) {
    %inp_2 = firrtl.stdIntCast %inp2 : (i27) -> !firrtl.uint<27>
    %inp_i = firrtl.stdIntCast %inpi : (i65) -> !firrtl.uint<65>

    // CHECK-NEXT: %tmp48 = rtl.wire : !rtl.inout<i27>
    %tmp48 = firrtl.wire : !firrtl.uint<27>

    // CHECK-NEXT: %c0_i38 = rtl.constant(0 : i38)
    // CHECK-NEXT: %0 = rtl.concat %c0_i38, %inp2 : (i38, i27) -> i65
    // CHECK-NEXT: %1 = rtl.divu %0, %inpi : i65
    %0 = firrtl.div %inp_2, %inp_i : (!firrtl.uint<27>, !firrtl.uint<65>) -> !firrtl.uint<27>
    // CHECK-NEXT: %2 = rtl.extract %1 from 0 : (i65) -> i27
    // CHECK-NEXT: rtl.connect %tmp48, %2 : i27
    firrtl.connect %tmp48, %0 : !firrtl.uint<27>, !firrtl.uint<27>
  }

  // https://github.com/llvm/circt/issues/318
  // CHECK-LABEL: rtl.module @test_rem
  // CHECK-NEXT:     %0 = rtl.modu
  // CHECK-NEXT:     rtl.output %0
  rtl.module @test_rem(%tmp85: i1, %tmp79: i1) -> (%tmp106: i1) {
    %0 = firrtl.stdIntCast %tmp85 : (i1) -> !firrtl.uint<1>
    %1 = firrtl.stdIntCast %tmp79 : (i1) -> !firrtl.uint<1>
    %2 = firrtl.rem %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %3 = firrtl.stdIntCast %2 : (!firrtl.uint<1>) -> i1
    rtl.output %3 : i1
  }

  // CHECK-LABEL: rtl.module @Analog(%a1: !rtl.inout<i1>, %b1: !rtl.inout<i1>,
  // CHECK:                          %c1: !rtl.inout<i1>) -> (%outClock: i1) {
  // CHECK-NEXT:   sv.ifdef "!SYNTHESIS"  {
  // CHECK-NEXT:     sv.alias %a1, %b1, %c1 : !rtl.inout<i1>
  // CHECK-NEXT:   }
  // CHECK-NEXT:   sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT:     %1 = rtl.read_inout %a1 : !rtl.inout<i1>
  // CHECK-NEXT:     %2 = rtl.read_inout %b1 : !rtl.inout<i1>
  // CHECK-NEXT:     %3 = rtl.read_inout %c1 : !rtl.inout<i1>
  // CHECK-NEXT:     rtl.connect %a1, %2 : i1
  // CHECK-NEXT:     rtl.connect %a1, %3 : i1
  // CHECK-NEXT:     rtl.connect %b1, %1 : i1
  // CHECK-NEXT:     rtl.connect %b1, %3 : i1
  // CHECK-NEXT:     rtl.connect %c1, %1 : i1
  // CHECK-NEXT:     rtl.connect %c1, %2 : i1
  // CHECK-NEXT:   }
  // CHECK-NEXT:    %0 = rtl.read_inout %a1 : !rtl.inout<i1>
  // CHECK-NEXT:    rtl.output %0 : i1
  rtl.module @Analog(%a1: !rtl.inout<i1>, %b1: !rtl.inout<i1>,
                     %c1: !rtl.inout<i1>) -> (%outClock: i1) {
    %a = firrtl.analogInOutCast %a1 : (!rtl.inout<i1>) -> !firrtl.analog<1>
    %b = firrtl.analogInOutCast %b1 : (!rtl.inout<i1>) -> !firrtl.analog<1>
    %c = firrtl.analogInOutCast %c1 : (!rtl.inout<i1>) -> !firrtl.analog<1>

    firrtl.attach %a, %b, %c : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>

    %1 = firrtl.asClock %a : (!firrtl.analog<1>) -> !firrtl.clock
    %2 = firrtl.stdIntCast %1 : (!firrtl.clock) -> i1
    rtl.output %2 : i1
  }


 // module UninitReg1 :
 //   input clock: Clock
 //   input reset : UInt<1>
 //   input cond: UInt<1>
 //   input value: UInt<2>
 //   reg count : UInt<2>, clock with :
 //     reset => (UInt<1>("h0"), count)
 //   node x = count
 //   node _GEN_0 = mux(cond, value, count)
 //   count <= mux(reset, UInt<2>("h0"), _GEN_0)

  // CHECK-LABEL: rtl.module @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
  rtl.module @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {
    // CHECK-NEXT: %c0_i2 = rtl.constant(0 : i2) : i2
    %c0_ui2 = firrtl.constant(0 : ui2) : !firrtl.uint<2>

    %0 = firrtl.stdIntCast %clock : (i1) -> !firrtl.clock
    %1 = firrtl.stdIntCast %reset : (i1) -> !firrtl.uint<1>
    %2 = firrtl.stdIntCast %cond : (i1) -> !firrtl.uint<1>
    %3 = firrtl.stdIntCast %value : (i2) -> !firrtl.uint<2>
    // CHECK-NEXT: %count = sv.reg : !rtl.inout<i2>
    %count = firrtl.reg %0 {name = "count"} : (!firrtl.clock) -> !firrtl.uint<2>

    // CHECK-NEXT: sv.ifdef "!SYNTHESIS"  {
    // CHECK-NEXT:    sv.initial {
    // CHECK-NEXT:    sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:    sv.ifdef "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       %3 = sv.textual_value "`RANDOM" : i2
    // CHECK-NEXT:        sv.bpassign %count, %3 : i2
    // CHECK-NEXT:     }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }

    // CHECK-NEXT: %0 = rtl.read_inout %count : !rtl.inout<i2>
    // CHECK-NEXT: %1 = rtl.mux %cond, %value, %0 : i2
    // CHECK-NEXT: %2 = rtl.mux %reset, %c0_i2, %1 : i2
    %4 = firrtl.mux(%2, %3, %count) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    %5 = firrtl.mux(%1, %c0_ui2, %4) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>

    // CHECK-NEXT: sv.alwaysff posedge, %clock  {
    // CHECK-NEXT:   sv.passign %count, %2 : i2
    // CHECK-NEXT: }
    firrtl.connect %count, %5 : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK-NEXT: rtl.output
    rtl.output
  }

  // module InitReg1 :
  //     input clock : Clock
  //     input reset : UInt<1>
  //     input io_d : UInt<32>
  //     output io_q : UInt<32>
  //     input io_en : UInt<1>
  //
  //     node _T = asAsyncReset(reset)
  //     reg reg : UInt<32>, clock with :
  //       reset => (_T, UInt<32>("h0"))
  //     io_q <= reg
  //     reg <= mux(io_en, io_d, reg)

  // CHECK-LABEL: rtl.module @InitReg1(
  rtl.module @InitReg1(%clock: i1, %reset: i1, %io_d: i32, %io_en: i1) -> (%io_q: i32) {
    // CHECK-NEXT: %c0_i32 = rtl.constant(0 : i32) : i32
    %c0_ui32 = firrtl.constant(0 : ui32) : !firrtl.uint<32>

    %0 = firrtl.stdIntCast %clock : (i1) -> !firrtl.clock
    %1 = firrtl.stdIntCast %reset : (i1) -> !firrtl.uint<1>
    %2 = firrtl.stdIntCast %io_d : (i32) -> !firrtl.uint<32>
    %3 = firrtl.stdIntCast %io_en : (i1) -> !firrtl.uint<1>
    %4 = firrtl.asAsyncReset %1 : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // CHECK-NEXT: %reg = sv.reg : !rtl.inout<i32>
    // CHECK-NEXT: sv.always posedge %clock, posedge %reset  {
    // CHECK-NEXT:   sv.if %reset  {
    // CHECK-NEXT:     sv.passign %reg, %c0_i32 : i32
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: sv.ifdef "!SYNTHESIS"  {
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       %true = rtl.constant(true) : i1
    // CHECK-NEXT:       %8 = rtl.xor %reset, %true : i1
    // CHECK-NEXT:       sv.if %8  {
    // CHECK-NEXT:         %9 = sv.textual_value "`RANDOM" : i32
    // CHECK-NEXT:         sv.bpassign %reg, %9 : i32
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: %reg2 = sv.reg : !rtl.inout<i32>
    // CHECK-NEXT: sv.always posedge %clock  {
    // CHECK-NEXT:   sv.if %reset  {
    // CHECK-NEXT:    sv.passign %reg2, %c0_i32 : i32
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: sv.ifdef "!SYNTHESIS"  {
    // CHECK-NEXT:   sv.initial  {
    // CHECK-NEXT:     sv.ifdef "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       %true = rtl.constant(true) : i1
    // CHECK-NEXT:       %8 = rtl.xor %reset, %true : i1
    // CHECK-NEXT:       sv.if %8  {
    // CHECK-NEXT:         %9 = sv.textual_value "`RANDOM" : i32
    // CHECK-NEXT:         sv.bpassign %reg2, %9 : i32
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    %reg = firrtl.regreset %0, %4, %c0_ui32 {name = "reg"} : (!firrtl.clock, !firrtl.asyncreset, !firrtl.uint<32>) -> !firrtl.uint<32>
    %reg2 = firrtl.regreset %0, %1, %c0_ui32 {name = "reg2"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<32>) -> !firrtl.uint<32>

    // CHECK-NEXT: %0 = rtl.read_inout %reg : !rtl.inout<i32>
    // CHECK-NEXT: %false = rtl.constant(false) : i1
    // CHECK-NEXT: %1 = rtl.concat %false, %0 : (i1, i32) -> i33
    // CHECK-NEXT: %2 = rtl.read_inout %reg2 : !rtl.inout<i32>
    // CHECK-NEXT: %false_0 = rtl.constant(false) : i1
    // CHECK-NEXT: %3 = rtl.concat %false_0, %2 : (i1, i32) -> i33
    // CHECK-NEXT: %4 = rtl.add %1, %3 : i33
    // CHECK-NEXT: %5 = rtl.extract %4 from 1 : (i33) -> i32
    // CHECK-NEXT: %6 = rtl.mux %io_en, %io_d, %5 : i32
    %sum = firrtl.add %reg, %reg2 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
    %shorten = firrtl.head %sum, 32 : (!firrtl.uint<33>) -> !firrtl.uint<32>
    %5 = firrtl.mux(%3, %2, %shorten) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>

    // CHECK-NEXT: sv.alwaysff posedge, %clock, asyncreset, posedge, %reset {
    // CHECK-NEXT: }  {
    // CHECK-NEXT:   sv.passign %reg, %6 : i32
    // CHECK-NEXT: }
    firrtl.connect %reg, %5 : !firrtl.uint<32>, !firrtl.uint<32>
    %6 = firrtl.stdIntCast %reg : (!firrtl.uint<32>) -> i32

    // CHECK-NEXT: %7 = rtl.read_inout %reg : !rtl.inout<i32>
    // CHECK-NEXT: rtl.output %7 : i32
    rtl.output %6 : i32
  }

  //  module MemSimple :
  //     input clock1  : Clock
  //     input clock2  : Clock
  //     input inpred  : UInt<1>
  //     input indata  : SInt<42>
  //     output result : SInt<42>
  //
  //     mem _M : @[Decoupled.scala 209:27]
  //           data-type => SInt<42>
  //           depth => 12
  //           read-latency => 0
  //           write-latency => 1
  //           reader => read
  //           writer => write
  //           read-under-write => undefined
  //
  //     result <= _M.read.data
  //
  //     _M.read.addr <= UInt<1>("h0")
  //     _M.read.en <= UInt<1>("h1")
  //     _M.read.clk <= clock1
  //     _M.write.addr <= validif(inpred, UInt<3>("h0"))
  //     _M.write.en <= mux(inpred, UInt<1>("h1"), UInt<1>("h0"))
  //     _M.write.clk <= validif(inpred, clock2)
  //     _M.write.data <= validif(inpred, indata)
  //     _M.write.mask <= validif(inpred, UInt<1>("h1"))

  // CHECK-LABEL: rtl.module @MemSimple(
  rtl.module @MemSimple(%clock1: i1, %clock2: i1, %inpred: i1, %indata: i42) -> (%result: i42) {
    %0 = firrtl.stdIntCast %clock1 : (i1) -> !firrtl.clock
    %1 = firrtl.stdIntCast %clock2 : (i1) -> !firrtl.clock
    %2 = firrtl.stdIntCast %inpred : (i1) -> !firrtl.uint<1>
    %3 = firrtl.stdIntCast %indata : (i42) -> !firrtl.sint<42>
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
    %c0_ui3 = firrtl.constant(0 : ui3) : !firrtl.uint<3>

    // CHECK:  %_M = sv.reg : !rtl.inout<uarray<12xi42>>
    %_M_read, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>

    // CHECK-NEXT: sv.ifdef "!SYNTHESIS"  {
    // CHECK-NEXT:   sv.initial  {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef "RANDOMIZE_MEM_INIT"  {
    // CHECK-NEXT:       sv.verbatim "integer {{.*}}_initvar < 12{{.*}}`RANDOM;"(%_M) : !rtl.inout<uarray<12xi42>>
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    // Write port.

    // Read port.
    // CHECK-NEXT: %_M_read_addr = rtl.wire : !rtl.inout<i4>
    // CHECK-NEXT: %_M_read_en = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT: %_M_read_clk = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT: %_M_read_data = rtl.wire : !rtl.inout<i42>
    // CHECK-NEXT: sv.ifdef "!RANDOMIZE_GARBAGE_ASSIGN"  {
    // CHECK-NEXT:   %2 = rtl.read_inout %_M_read_addr : !rtl.inout<i4>
    // CHECK-NEXT:   %3 = rtl.arrayindex %_M[%2]
    // CHECK-NEXT:   %4 = rtl.read_inout %3 : !rtl.inout<i42>
    // CHECK-NEXT:   rtl.connect %_M_read_data, %4 : i42
    // CHECK-NEXT: } else  {
    // CHECK-NEXT:   %2 = rtl.read_inout %_M_read_addr : !rtl.inout<i4>
    // CHECK-NEXT:   %3 = rtl.arrayindex %_M[%2]
    // CHECK-NEXT:   %4 = rtl.read_inout %3 : !rtl.inout<i42>
    // CHECK-NEXT:   %c-4_i4 = rtl.constant(-4 : i4) : i4
    // CHECK-NEXT:   %5 = rtl.icmp ult %2, %c-4_i4 : i4
    // CHECK-NEXT:   %6 = sv.textual_value "`RANDOM" : i42
    // CHECK-NEXT:   %7 = rtl.mux %5, %4, %6 : i42
    // CHECK-NEXT:   rtl.connect %_M_read_data, %7 : i42
    // CHECK-NEXT: }

    // CHECK:      %_M_write_addr = rtl.wire : !rtl.inout<i4>
    // CHECK-NEXT: %_M_write_en = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT: %_M_write_clk = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT: %_M_write_data = rtl.wire : !rtl.inout<i42>
    // CHECK-NEXT: %_M_write_mask = rtl.wire : !rtl.inout<i1>

    // CHECK-NEXT: %0 = rtl.read_inout %_M_write_clk
    // CHECK-NEXT: sv.always posedge %0  {
    // CHECK-NEXT:     %2 = rtl.read_inout %_M_write_en : !rtl.inout<i1>
    // CHECK-NEXT:     %3 = rtl.read_inout %_M_write_mask : !rtl.inout<i1>
    // CHECK-NEXT:     %4 = rtl.and %2, %3 : i1
    // CHECK-NEXT:     sv.if %4  {
    // CHECK-NEXT:       %5 = rtl.read_inout %_M_write_data : !rtl.inout<i42>
    // CHECK-NEXT:       %6 = rtl.read_inout %_M_write_addr : !rtl.inout<i4>
    // CHECK-NEXT:       %7 = rtl.arrayindex %_M[%6]
    // CHECK-NEXT:       sv.bpassign %7, %5 : i42
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }

    %5 = firrtl.subfield %_M_read("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.sint<42>
    %6 = firrtl.subfield %_M_read("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %6, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
    %7 = firrtl.subfield %_M_read("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %7, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %8 = firrtl.subfield %_M_read("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
    firrtl.connect %8, %0 : !firrtl.flip<clock>, !firrtl.clock

    %10 = firrtl.subfield %_M_write("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
    %11 = firrtl.validif %2, %c0_ui3 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %10, %11 : !firrtl.flip<uint<4>>, !firrtl.uint<3>
    %12 = firrtl.subfield %_M_write("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %12, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %13 = firrtl.subfield %_M_write("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<clock>
    %14 = firrtl.validif %2, %1 : (!firrtl.uint<1>, !firrtl.clock) -> !firrtl.clock
    firrtl.connect %13, %14 : !firrtl.flip<clock>, !firrtl.clock
    %15 = firrtl.subfield %_M_write("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<sint<42>>
    %16 = firrtl.validif %2, %3 : (!firrtl.uint<1>, !firrtl.sint<42>) -> !firrtl.sint<42>
    firrtl.connect %15, %16 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
    %17 = firrtl.subfield %_M_write("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
    %18 = firrtl.validif %2, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %17, %18 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %19 = firrtl.stdIntCast %5 : (!firrtl.sint<42>) -> i42
    rtl.output %19 : i42
  }

  // module MemAggregate :
  //    input clock1 : Clock
  //    input clock2 : Clock
  //
  //    mem _M : @[Decoupled.scala 209:24]
  //          data-type => { id : Analog<4>, other: SInt<8> }
  //          depth => 20
  //          read-latency => 0
  //          write-latency => 1
  //          reader => read
  //          writer => write
  //          read-under-write => undefined
  //
  // CHECK-LABEL: rtl.module @MemAggregate(%clock1: i1, %clock2: i1) {
  // CHECK-NEXT:  %_M_id = sv.reg : !rtl.inout<uarray<20xi4>>
  // CHECK-NEXT:  %_M_other = sv.reg : !rtl.inout<uarray<20xi8>>
  // CHECK-NEXT:  sv.ifdef "!SYNTHESIS"  {
  // CHECK-NEXT:    sv.initial  {
  // CHECK-NEXT:      sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:      sv.ifdef "RANDOMIZE_MEM_INIT"  {
  // CHECK-NEXT:        sv.verbatim "integer {{.*}}_initvar < 20{{.*}}
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK:  rtl.output
  // CHECK-NEXT:}
  rtl.module @MemAggregate(%clock1: i1, %clock2: i1) {
      %0 = firrtl.stdIntCast %clock1 : (i1) -> !firrtl.clock
      %1 = firrtl.stdIntCast %clock2 : (i1) -> !firrtl.clock
      %_M_read, %_M_write = firrtl.mem Undefined {depth = 20 : i64, name = "_M", portNames = ["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<5>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<id: analog<4>, other: sint<8>>>, !firrtl.flip<bundle<addr: uint<5>, en: uint<1>, clk: clock, data: bundle<id: analog<4>, other: sint<8>>, mask: bundle<id: uint<1>, other: uint<1>>>>
      rtl.output
    }

  // module MemEmpty :
  //    mem Empty :
  //      data-type => UInt<32>
  //      depth => 16
  //      read-latency => 0
  //      write-latency => 1
  //      read-under-write => undefined
  //
  // CHECK-LABEL: rtl.module @MemEmpty() {
  // CHECK-NEXT:   sv.ifdef "!SYNTHESIS"  {
  // CHECK-NEXT:     sv.initial  {
  // CHECK-NEXT:       sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       sv.ifdef "RANDOMIZE_MEM_INIT"  {
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT:   rtl.output
  // CHECK-NEXT: }
  rtl.module @MemEmpty() {
    firrtl.mem Undefined {depth = 16 : i64, name = "Empty", portNames = [], readLatency = 0 : i32, writeLatency = 1 : i32}
    rtl.output
  }

  // module MemOne :
  //   mem _M : @[Decoupled.scala 209:24]
  //         data-type => { id : Analog<4>, other: SInt<8> }
  //         depth => 1
  //         read-latency => 0
  //         write-latency => 1
  //         reader => read
  //         writer => write
  //         read-under-write => undefined
  //
  // CHECK-LABEL: rtl.module @MemOne() {
  // CHECK-NEXT:   %_M_id = sv.reg : !rtl.inout<uarray<1xi4>>
  // CHECK-NEXT:   %_M_other = sv.reg : !rtl.inout<uarray<1xi8>>
  // CHECK-NEXT:   sv.ifdef "!SYNTHESIS"  {
  // CHECK-NEXT:     sv.initial  {
  // CHECK-NEXT:       sv.verbatim "`INIT_RANDOM_PROLOG_"
  // CHECK-NEXT:       sv.ifdef "RANDOMIZE_MEM_INIT"  {
  // CHECK-NEXT:         %0 = sv.textual_value "`RANDOM" : i4
  // CHECK-NEXT:         %false = rtl.constant(false) : i1
  // CHECK-NEXT:         %1 = rtl.arrayindex %_M_id[%false] : !rtl.inout<uarray<1xi4>>, i1
  // CHECK-NEXT:         sv.bpassign %1, %0 : i4
  // CHECK-NEXT:         %2 = sv.textual_value "`RANDOM" : i8
  // CHECK-NEXT:         %false_0 = rtl.constant(false) : i1
  // CHECK-NEXT:         %3 = rtl.arrayindex %_M_other[%false_0] : !rtl.inout<uarray<1xi8>>, i1
  // CHECK-NEXT:         sv.bpassign %3, %2 : i8
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK:   rtl.output
  // CHECK-NEXT: }
  rtl.module @MemOne() {
    %_M_read, %_M_write = firrtl.mem Undefined {depth = 1 : i64, name = "_M", portNames=["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: bundle<id: analog<4>, other: sint<8>>>, !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<id: analog<4>, other: sint<8>>, mask: bundle<id: uint<1>, other: uint<1>>>>
    rtl.output
  }

  // CHECK-LABEL: rtl.module @IncompleteRead(
  // The read port has no use of the data field.
  rtl.module @IncompleteRead(%clock1: i1) {
    %0 = firrtl.stdIntCast %clock1 : (i1) -> !firrtl.clock
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>

    // CHECK:  %_M = sv.reg : !rtl.inout<uarray<12xi42>>
    %_M_read = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>
    // Read port.
    // CHECK: %_M_read_addr = rtl.wire : !rtl.inout<i4>
    // CHECK-NEXT: %_M_read_en = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT: %_M_read_clk = rtl.wire : !rtl.inout<i1>
    // CHECK-NEXT: %_M_read_data = rtl.wire : !rtl.inout<i42>
    // CHECK-NEXT: sv.ifdef "!RANDOMIZE_GARBAGE_ASSIGN"  {
    %6 = firrtl.subfield %_M_read("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %6, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
    %7 = firrtl.subfield %_M_read("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %7, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %8 = firrtl.subfield %_M_read("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
    firrtl.connect %8, %0 : !firrtl.flip<clock>, !firrtl.clock
    rtl.output
  }

  // CHECK-LABEL: rtl.module @top_mod() -> (%tmp27: i23) {
  // CHECK-NEXT:    %c42_i23 = rtl.constant(42 : i23) : i23
  // CHECK-NEXT:    %c0_i23 = rtl.constant(0 : i23) : i23
  // CHECK-NEXT:    rtl.output %c0_i23 : i23
  // CHECK-NEXT:  }
  rtl.module @top_mod() -> (%tmp27: i23) {
    %0 = firrtl.wire : !firrtl.flip<uint<0>>
    %c42_ui23 = firrtl.constant(42 : ui23) : !firrtl.uint<23>
    %1 = firrtl.tail %c42_ui23, 23 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    firrtl.connect %0, %1 : !firrtl.flip<uint<0>>, !firrtl.uint<0>
    %2 = firrtl.head %c42_ui23, 0 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    %3 = firrtl.pad %2, 23 : (!firrtl.uint<0>) -> !firrtl.uint<23>
    %4 = firrtl.stdIntCast %3 : (!firrtl.uint<23>) -> i23
    rtl.output %4 : i23
  }
}
