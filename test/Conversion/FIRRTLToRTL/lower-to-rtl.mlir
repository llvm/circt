// RUN: circt-opt -lower-firrtl-to-rtl %s | FileCheck %s

firrtl.circuit "Simple" {

  // CHECK-LABEL: rtl.module @Simple
  firrtl.module @Simple(%in1: !firrtl.uint<4>,
                        %in2: !firrtl.uint<2>,
                        %in3: !firrtl.sint<8>,
                        %in4: !firrtl.uint<0>, 
                        %in5: !firrtl.sint<0>, 
                        %out1: !firrtl.flip<sint<1>>,
                        %out2: !firrtl.flip<sint<1>>  ) {
    // Issue #364: https://github.com/llvm/circt/issues/364
    // CHECK: = rtl.constant -1175 : i12
    // CHECK-DAG: rtl.constant -4 : i4
    %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>

    // CHECK-DAG: rtl.constant 2 : i3
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>


    // CHECK: %out4 = sv.wire  : !rtl.inout<i4>
    // CHECK: %out5 = sv.wire  : !rtl.inout<i4>
    %out4 = firrtl.wire : !firrtl.uint<4>
    %out5 = firrtl.wire : !firrtl.uint<4>

    // CHECK: sv.connect %out5, %c0_i4 : i4
    %tmp1 = firrtl.invalidvalue : !firrtl.uint<4>
    firrtl.connect %out5, %tmp1 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: [[ZEXT:%.+]] = comb.concat %false, %in1 : (i1, i4) -> i5
    // CHECK: [[ADD:%.+]] = comb.add %c12_i5, [[ZEXT]] : i5
    %0 = firrtl.add %c12_ui4, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[ZEXT1:%.+]] = comb.concat %false, [[ADD]] : (i1, i5) -> i6
    // CHECK: [[ZEXT2:%.+]] = comb.concat %c0_i2, %in1 : (i2, i4) -> i6
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub [[ZEXT1]], [[ZEXT2]] : i6
    %2 = firrtl.sub %0, %1 : (!firrtl.uint<5>, !firrtl.uint<4>) -> !firrtl.uint<6>

    %in2s = firrtl.asSInt %in2 : (!firrtl.uint<2>) -> !firrtl.sint<2>

    // CHECK: [[PADRES:%.+]] = comb.sext %in2 : (i2) -> i3
    %3 = firrtl.pad %in2s, 3 : (!firrtl.sint<2>) -> !firrtl.sint<3>

    // CHECK: [[PADRES2:%.+]] = comb.concat %c0_i2, %in2 : (i2, i2) -> i4
    %4 = firrtl.pad %in2, 4 : (!firrtl.uint<2>) -> !firrtl.uint<4>

    // CHECK: [[IN2EXT:%.+]] = comb.concat %c0_i2, %in2 : (i2, i2) -> i4
    // CHECK: [[XOR:%.+]] = comb.xor [[IN2EXT]], [[PADRES2]] : i4
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.and [[XOR]]
    %and = firrtl.and %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.or [[XOR]]
    %or = firrtl.or %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[CONCAT1:%.+]] = comb.concat [[PADRES2]], [[XOR]] : (i4, i4) -> i8
    %6 = firrtl.cat %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>

    // CHECK: comb.concat %in1, %in2
    %7 = firrtl.cat %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>

    // CHECK-NEXT: sv.connect %out5, [[PADRES2]] : i4
    firrtl.connect %out5, %4 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: sv.connect %out4, [[XOR]] : i4
    firrtl.connect %out4, %5 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: [[ZEXT:%.+]] = comb.concat %c0_i2, %in2 : (i2, i2) -> i4
    // CHECK-NEXT: sv.connect %out4, [[ZEXT]] : i4
    firrtl.connect %out4, %in2 : !firrtl.uint<4>, !firrtl.uint<2>

    // CHECK-NEXT: %test-name = sv.wire : !rtl.inout<i4>
    firrtl.wire {name = "test-name"} : !firrtl.uint<4>

    // CHECK-NEXT: = sv.wire : !rtl.inout<i2>
    %_t_1 = firrtl.wire : !firrtl.uint<2>

    // CHECK-NEXT: = firrtl.wire : !firrtl.vector<uint<1>, 13>
    %_t_2 = firrtl.wire : !firrtl.vector<uint<1>, 13>

    // CHECK-NEXT: = firrtl.wire : !firrtl.vector<uint<2>, 13>
    %_t_3 = firrtl.wire : !firrtl.vector<uint<2>, 13>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %8 = firrtl.bits %6 7 to 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 5 : (i8) -> i3
    %9 = firrtl.head %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 0 : (i8) -> i5
    %10 = firrtl.tail %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %11 = firrtl.shr %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    %12 = firrtl.shr %6, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.extract %in3 from 7 : (i8) -> i1
    %13 = firrtl.shr %in3, 8 : (!firrtl.sint<8>) -> !firrtl.sint<1>

    // CHECK-NEXT: = comb.concat [[CONCAT1]], %c0_i3 : (i8, i3) -> i11
    %14 = firrtl.shl %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<11>

    // CHECK-NEXT: = comb.parity [[CONCAT1]] : i8
    %15 = firrtl.xorr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp eq  {{.*}}, %c-1_i8 : i8
    %16 = firrtl.andr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp ne {{.*}}, %c0_i8 : i8
    %17 = firrtl.orr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[ZEXTC1:%.+]] = comb.concat %c0_i6, [[CONCAT1]] : (i6, i8) -> i14
    // CHECK-NEXT: [[ZEXT2:%.+]] = comb.concat %c0_i8, [[SUB]] : (i8, i6) -> i14
    // CHECK-NEXT: [[VAL18:%.+]] = comb.mul  [[ZEXTC1]], [[ZEXT2]] : i14
    %18 = firrtl.mul %6, %2 : (!firrtl.uint<8>, !firrtl.uint<6>) -> !firrtl.uint<14>

    // CHECK-NEXT: [[IN3SEXT:%.+]] = comb.sext %in3 : (i8) -> i9
    // CHECK-NEXT: [[PADRESSEXT:%.+]] = comb.sext [[PADRES]] : (i3) -> i9
    // CHECK-NEXT: = comb.divs [[IN3SEXT]], [[PADRESSEXT]] : i9
    %19 = firrtl.div %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<9>

    // CHECK-NEXT: [[IN3EX:%.+]] = comb.sext [[PADRES]] : (i3) -> i8
    // CHECK-NEXT: [[MOD1:%.+]] = comb.mods %in3, [[IN3EX]] : i8
    // CHECK-NEXT: = comb.extract [[MOD1]] from 0 : (i8) -> i3
    %20 = firrtl.rem %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[IN4EX:%.+]] = comb.sext [[PADRES]] : (i3) -> i8
    // CHECK-NEXT: [[MOD2:%.+]] = comb.mods [[IN4EX]], %in3 : i8
    // CHECK-NEXT: = comb.extract [[MOD2]] from 0 : (i8) -> i3
    %21 = firrtl.rem %3, %in3 : (!firrtl.sint<3>, !firrtl.sint<8>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[WIRE:%n1]] = sv.wire : !rtl.inout<i2>
    // CHECK-NEXT: sv.connect [[WIRE]], %in2 : i2
    %n1 = firrtl.node %in2  {name = "n1"} : !firrtl.uint<2>

    // Nodes with no names are just dropped.
    %22 = firrtl.node %n1 {name = ""} : !firrtl.uint<2>

    // CHECK-NEXT: [[CVT:%.+]] = comb.concat %false, %in2 : (i1, i2) -> i3
    %23 = firrtl.cvt %22 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // Will be dropped, here because this triggered a crash
    %s23 = firrtl.cvt %in3 : (!firrtl.sint<8>) -> !firrtl.sint<8>

    // CHECK-NEXT: [[XOR:%.+]] = comb.xor [[CVT]], %c-1_i3 : i3
    %24 = firrtl.not %23 : (!firrtl.sint<3>) -> !firrtl.uint<3>

    %s24 = firrtl.asSInt %24 : (!firrtl.uint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[SEXT:%.+]] = comb.sext [[XOR]] : (i3) -> i4
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub %c0_i4, [[SEXT]] : i4
    %25 = firrtl.neg %s24 : (!firrtl.sint<3>) -> !firrtl.sint<4>

    // CHECK-NEXT: [[CVT4:%.+]] = comb.sext [[CVT]] : (i3) -> i4
    // CHECK-NEXT: comb.mux {{.*}}, [[CVT4]], [[SUB]] : i4
    %26 = firrtl.mux(%17, %23, %25) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.sint<4>

    // CHECK-NEXT: = comb.icmp eq  {{.*}}, %c-1_i14 : i14
    %28 = firrtl.andr %18 : (!firrtl.uint<14>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[XOREXT:%.+]] = comb.concat %c0_i11, [[XOR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shru [[XOREXT]], [[VAL18]] : i14
    // CHECK-NEXT: [[DSHR:%.+]] = comb.extract [[SHIFT]] from 0 : (i14) -> i3
    %29 = firrtl.dshr %24, %18 : (!firrtl.uint<3>, !firrtl.uint<14>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.concat %c0_i5, {{.*}} : (i5, i3) -> i8
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs %in3, {{.*}} : i8
    %a29 = firrtl.dshr %in3, %9 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<8>

    // CHECK-NEXT: = comb.sext %in3 : (i8) -> i15
    // CHECK-NEXT: = comb.concat %c0_i12, [[DSHR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shl {{.*}}, {{.*}} : i15
    %30 = firrtl.dshl %in3, %29 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<15>

    // CHECK-NEXT: = comb.shru [[DSHR]], [[DSHR]] : i3
    %dshlw = firrtl.dshlw %29, %29 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>

    // Issue #367: https://github.com/llvm/circt/issues/367
    // CHECK-NEXT: = comb.sext {{.*}} : (i4) -> i14
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs {{.*}}, {{.*}} : i14
    // CHECK-NEXT: = comb.extract [[SHIFT]] from 0 : (i14) -> i4
    %31 = firrtl.dshr %25, %18 : (!firrtl.sint<4>, !firrtl.uint<14>) -> !firrtl.sint<4>

    // CHECK-NEXT: comb.icmp ule {{.*}}, {{.*}} : i4
    %41 = firrtl.leq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp ult {{.*}}, {{.*}} : i4
    %42 = firrtl.lt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp uge {{.*}}, {{.*}} : i4
    %43 = firrtl.geq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp ugt {{.*}}, {{.*}} : i4
    %44 = firrtl.gt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp eq {{.*}}, {{.*}} : i4
    %45 = firrtl.eq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp ne {{.*}}, {{.*}} : i4
    %46 = firrtl.neq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>

    // Noop
    %47 = firrtl.asClock %44 : (!firrtl.uint<1>) -> !firrtl.clock
    %48 = firrtl.asAsyncReset %44 : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // Issue #353
    // CHECK: [[PADRES_EXT:%.+]] = comb.sext [[PADRES]] : (i3) -> i8
    // CHECK: = comb.and %in3, [[PADRES_EXT]] : i8
    %49 = firrtl.and %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.uint<8>

    // Issue #355: https://github.com/llvm/circt/issues/355
    // CHECK: [[DIV:%.+]] = comb.divu %c104_i10, %c306_i10 : i10
    // CHECK: = comb.extract [[DIV]] from 0 : (i10) -> i8
    %c104_ui8 = firrtl.constant(104 : ui8) : !firrtl.uint<8>
    %c306_ui10 = firrtl.constant(306 : ui10) : !firrtl.uint<10>
    %50 = firrtl.div %c104_ui8, %c306_ui10 : (!firrtl.uint<8>, !firrtl.uint<10>) -> !firrtl.uint<8>

    %c1175_ui11 = firrtl.constant(1175 : ui11) : !firrtl.uint<11>
    %51 = firrtl.neg %c1175_ui11 : (!firrtl.uint<11>) -> !firrtl.sint<12>
    // https://github.com/llvm/circt/issues/821
    // CHECK: [[CONCAT:%.+]] = comb.concat %false, %in1 : (i1, i4) -> i5
    // CHECK:  = comb.sub %c0_i5, [[CONCAT]] : i5
    %52 = firrtl.neg %in1 : (!firrtl.uint<4>) -> !firrtl.sint<5>
    %53 = firrtl.neg %in4 : (!firrtl.uint<0>) -> !firrtl.sint<1>
    // CHECK: [[SEXT:%.+]] = comb.sext %in3 : (i8) -> i9
    // CHECK: = comb.sub %c0_i9, [[SEXT]] : i9
    %54 = firrtl.neg %in3 : (!firrtl.sint<8>) -> !firrtl.sint<9>
    // CHECK: rtl.output %false, %false : i1, i1
    firrtl.connect %out1, %53 : !firrtl.flip<sint<1>>, !firrtl.sint<1>
    %55 = firrtl.neg %in5 : (!firrtl.sint<0>) -> !firrtl.sint<1>
    firrtl.connect %out2, %55 : !firrtl.flip<sint<1>>, !firrtl.sint<1>
  }

//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: rtl.module @Print
  firrtl.module @Print(%clock: !firrtl.clock, %reset: !firrtl.uint<1>,
                       %a: !firrtl.uint<4>, %b: !firrtl.uint<4>) {

    // CHECK: sv.always posedge %clock {
    // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
    // CHECK-NEXT:   } else  {
    // CHECK-NEXT:     %PRINTF_COND_ = sv.verbatim.expr "`PRINTF_COND_" : () -> i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and %PRINTF_COND_, %reset
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       sv.fwrite "No operands!\0A"
    // CHECK-NEXT:     }
    // CHECK-NEXT:     %PRINTF_COND__0 = sv.verbatim.expr "`PRINTF_COND_" : () -> i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and %PRINTF_COND__0, %reset : i1
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       sv.fwrite "Hi %x %x\0A"(%2, %b) : i5, i4
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
   firrtl.printf %clock, %reset, "No operands!\0A"

    // CHECK: [[ADD:%.+]] = comb.add
    %0 = firrtl.add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    firrtl.printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.uint<5>, !firrtl.uint<4>

    firrtl.skip

    // CHECK: rtl.output
   }



// module Stop3 :
//    input clock1: Clock
//    input clock2: Clock
//    input reset: UInt<1>
//    stop(clock1, reset, 42)
//    stop(clock2, reset, 0)

  // CHECK-LABEL: rtl.module @Stop
  firrtl.module @Stop(%clock1: !firrtl.clock, %clock2: !firrtl.clock, %reset: !firrtl.uint<1>) {

    // CHECK-NEXT: sv.always posedge %clock1 {
    // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     %STOP_COND_ = sv.verbatim.expr "`STOP_COND_" : () -> i1
    // CHECK-NEXT:     %0 = comb.and %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT: sv.always posedge %clock2 {
    // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     %STOP_COND_ = sv.verbatim.expr "`STOP_COND_" : () -> i1
    // CHECK-NEXT:     %0 = comb.and %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.finish
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock2, %reset, 0
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
  firrtl.module @Verification(%clock: !firrtl.clock, %aCond: !firrtl.uint<1>,
   %aEn: !firrtl.uint<1>, %bCond: !firrtl.uint<1>, %bEn: !firrtl.uint<1>,
    %cCond: !firrtl.uint<1>, %cEn: !firrtl.uint<1>) {

    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.if %aEn {
    // CHECK-NEXT:     sv.assert %aCond : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %bEn {
    // CHECK-NEXT:     sv.assume %bCond  : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %cEn {
    // CHECK-NEXT:     sv.cover %cCond : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assert %clock, %aCond, %aEn, "assert0"
    firrtl.assume %clock, %bCond, %bEn, "assume0"
    firrtl.cover %clock, %cCond, %cEn, "cover0"
    // CHECK-NEXT: rtl.output
  }

  firrtl.module @bar(%io_cpu_flush: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: rtl.module @foo
  firrtl.module @foo() {
    // CHECK-NEXT:  %io_cpu_flush.wire = sv.wire : !rtl.inout<i1>
    %io_cpu_flush.wire = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: rtl.instance "fetch" @bar([[IO:%[0-9]+]])
    %i = firrtl.instance @bar {name = "fetch", portNames=["io_cpu_flush"]} : !firrtl.flip<uint<1>>
    firrtl.connect %i, %io_cpu_flush.wire : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    %hits_1_7 = firrtl.node %io_cpu_flush.wire {name = "hits_1_7"} : !firrtl.uint<1>
    // CHECK-NEXT:  [[IO]] = sv.read_inout %io_cpu_flush.wire
    // CHECK-NEXT:  [[IO:%.+]] = sv.read_inout %io_cpu_flush.wire
    // CHECK-NEXT:  %hits_1_7 = sv.wire : !rtl.inout<i1>
    // CHECK-NEXT:  sv.connect %hits_1_7, [[IO]] : i1
    %1455 = firrtl.asPassive %hits_1_7 : !firrtl.uint<1>
  }

  // https://github.com/llvm/circt/issues/314
  // CHECK-LABEL: rtl.module @issue314
  firrtl.module @issue314(%inp_2: !firrtl.uint<27>, %inpi: !firrtl.uint<65>) {
    // CHECK: %c0_i38 = rtl.constant 0 : i38
    // CHECK: %tmp48 = sv.wire : !rtl.inout<i27>
    %tmp48 = firrtl.wire : !firrtl.uint<27>

    // CHECK-NEXT: %0 = comb.concat %c0_i38, %inp_2 : (i38, i27) -> i65
    // CHECK-NEXT: %1 = comb.divu %0, %inpi : i65
    %0 = firrtl.div %inp_2, %inpi : (!firrtl.uint<27>, !firrtl.uint<65>) -> !firrtl.uint<27>
    // CHECK-NEXT: %2 = comb.extract %1 from 0 : (i65) -> i27
    // CHECK-NEXT: sv.connect %tmp48, %2 : i27
    firrtl.connect %tmp48, %0 : !firrtl.uint<27>, !firrtl.uint<27>
  }

  // https://github.com/llvm/circt/issues/318
  // CHECK-LABEL: rtl.module @test_rem
  // CHECK-NEXT:     %0 = comb.modu
  // CHECK-NEXT:     rtl.output %0
  firrtl.module @test_rem(%tmp85: !firrtl.uint<1>, %tmp79: !firrtl.uint<1>,
       %out: !firrtl.flip<uint<1>>) {
    %2 = firrtl.rem %tmp79, %tmp85 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  }

  // CHECK-LABEL: rtl.module @Analog(%a1: !rtl.inout<i1>, %b1: !rtl.inout<i1>,
  // CHECK:                          %c1: !rtl.inout<i1>) -> (%outClock: i1) {
  // CHECK-NEXT:   sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT:     %1 = sv.read_inout %a1 : !rtl.inout<i1>
  // CHECK-NEXT:     %2 = sv.read_inout %b1 : !rtl.inout<i1>
  // CHECK-NEXT:     %3 = sv.read_inout %c1 : !rtl.inout<i1>
  // CHECK-NEXT:     sv.connect %a1, %2 : i1
  // CHECK-NEXT:     sv.connect %a1, %3 : i1
  // CHECK-NEXT:     sv.connect %b1, %1 : i1
  // CHECK-NEXT:     sv.connect %b1, %3 : i1
  // CHECK-NEXT:     sv.connect %c1, %1 : i1
  // CHECK-NEXT:     sv.connect %c1, %2 : i1
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:     sv.ifdef "verilator" {
  // CHECK-NEXT:       sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.alias %a1, %b1, %c1 : !rtl.inout<i1>
  // CHECK-NEXT:     }
  // CHECK-NEXT:    }
  // CHECK-NEXT:    %0 = sv.read_inout %a1 : !rtl.inout<i1>
  // CHECK-NEXT:    rtl.output %0 : i1
  firrtl.module @Analog(%a1: !firrtl.analog<1>, %b1: !firrtl.analog<1>,
                        %c1: !firrtl.analog<1>, %outClock: !firrtl.flip<clock>) {
    firrtl.attach %a1, %b1, %c1 : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>

    %1 = firrtl.asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    firrtl.connect %outClock, %1 : !firrtl.flip<clock>, !firrtl.clock
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

  firrtl.module @UninitReg1(%clock: !firrtl.clock, %reset: !firrtl.uint<1>,
                            %cond: !firrtl.uint<1>, %value: !firrtl.uint<2>) {
    // CHECK-NEXT: %c0_i2 = rtl.constant 0 : i2
    %c0_ui2 = firrtl.constant(0 : ui2) : !firrtl.uint<2>
    // CHECK-NEXT: %count = sv.reg : !rtl.inout<i2>
    %count = firrtl.reg %clock {name = "count"} : (!firrtl.clock) -> !firrtl.uint<2>

    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:    sv.initial {
    // CHECK-NEXT:    sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:    sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       %RANDOM = sv.verbatim.expr "`RANDOM" : () -> i2
    // CHECK-NEXT:       sv.bpassign %count, %RANDOM : i2
    // CHECK-NEXT:     }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }

    // CHECK-NEXT: %0 = sv.read_inout %count : !rtl.inout<i2>
    // CHECK-NEXT: %1 = comb.mux %cond, %value, %0 : i2
    // CHECK-NEXT: %2 = comb.mux %reset, %c0_i2, %1 : i2
    %4 = firrtl.mux(%cond, %value, %count) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    %5 = firrtl.mux(%reset, %c0_ui2, %4) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>

    // CHECK-NEXT: sv.alwaysff(posedge %clock)  {
    // CHECK-NEXT:   sv.passign %count, %2 : i2
    // CHECK-NEXT: }
    firrtl.connect %count, %5 : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK-NEXT: rtl.output
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
  firrtl.module @InitReg1(%clock: !firrtl.clock, %reset: !firrtl.uint<1>,
                          %io_d: !firrtl.uint<32>, %io_en: !firrtl.uint<1>,
                          %io_q: !firrtl.flip<uint<32>>) {
    // CHECK: %c0_i32 = rtl.constant 0 : i32
    %c0_ui32 = firrtl.constant(0 : ui32) : !firrtl.uint<32>

    %4 = firrtl.asAsyncReset %reset : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // CHECK-NEXT: %reg = sv.reg : !rtl.inout<i32>
    // CHECK-NEXT: sv.alwaysff(posedge %clock) {
    // CHECK-NEXT:   sv.passign %reg, %6 : i32
    // CHECK-NEXT: }(asyncreset : posedge %reset) {
    // CHECK-NEXT:   sv.passign %reg, %c0_i32 : i32
    // CHECK-NEXT: }
    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.if %reset  {
    // CHECK-NEXT:       } else {
    // CHECK-NEXT:         %RANDOM = sv.verbatim.expr "`RANDOM" : () -> i32
    // CHECK-NEXT:         sv.bpassign %reg, %RANDOM : i32
    // CHECK-NEXT:         %RANDOM_0 = sv.verbatim.expr "`RANDOM" : () -> i32
    // CHECK-NEXT:         sv.bpassign %reg2, %RANDOM_0 : i32
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: %reg2 = sv.reg : !rtl.inout<i32>
    // CHECK-NEXT: sv.alwaysff(posedge %clock) {
    // CHECK-NEXT: }(syncreset : posedge %reset) {
    // CHECK-NEXT:    sv.passign %reg2, %c0_i32 : i32
    // CHECK-NEXT: }
    %reg = firrtl.regreset %clock, %4, %c0_ui32 {name = "reg"} : (!firrtl.clock, !firrtl.asyncreset, !firrtl.uint<32>) -> !firrtl.uint<32>
    %reg2 = firrtl.regreset %clock, %reset, %c0_ui32 {name = "reg2"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<32>) -> !firrtl.uint<32>

    // CHECK-NEXT: %0 = sv.read_inout %reg : !rtl.inout<i32>
    // CHECK-NEXT: %1 = comb.concat %false, %0 : (i1, i32) -> i33
    // CHECK-NEXT: %2 = sv.read_inout %reg2 : !rtl.inout<i32>
    // CHECK-NEXT: %3 = comb.concat %false, %2 : (i1, i32) -> i33
    // CHECK-NEXT: %4 = comb.add %1, %3 : i33
    // CHECK-NEXT: %5 = comb.extract %4 from 1 : (i33) -> i32
    // CHECK-NEXT: %6 = comb.mux %io_en, %io_d, %5 : i32
    %sum = firrtl.add %reg, %reg2 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
    %shorten = firrtl.head %sum, 32 : (!firrtl.uint<33>) -> !firrtl.uint<32>
    %5 = firrtl.mux(%io_en, %io_d, %shorten) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>

    firrtl.connect %reg, %5 : !firrtl.uint<32>, !firrtl.uint<32>
    firrtl.connect %io_q, %reg: !firrtl.flip<uint<32>>, !firrtl.uint<32>

    // CHECK-NEXT: %7 = sv.read_inout %reg : !rtl.inout<i32>
    // CHECK-NEXT: rtl.output %7 : i32
  }

  //  module MemSimple :
  //     input clock1  : Clock
  //     input clock2  : Clock
  //     input inpred  : UInt<1>
  //     input indata  : SInt<42>
  //     output result : SInt<42>
  //     output result2 : SInt<42>
  //
  //     mem _M : @[Decoupled.scala 209:27]
  //           data-type => SInt<42>
  //           depth => 12
  //           read-latency => 0
  //           write-latency => 1
  //           reader => read
  //           writer => write
  //           readwriter => rw
  //           read-under-write => undefined
  //
  //     result <= _M.read.data
  //     result2 <= _M.rw.rdata
  //
  //     _M.read.addr <= UInt<1>("h0")
  //     _M.read.en <= UInt<1>("h1")
  //     _M.read.clk <= clock1
  //     _M.rw.addr <= UInt<1>("h0")
  //     _M.rw.en <= UInt<1>("h1")
  //     _M.rw.clk <= clock1
  //     _M.rw.wmask <= UInt<1>("h1")
  //     _M.rw.wmode <= UInt<1>("h1")
  //     _M.write.addr <= validif(inpred, UInt<3>("h0"))
  //     _M.write.en <= mux(inpred, UInt<1>("h1"), UInt<1>("h0"))
  //     _M.write.clk <= clock2
  //     _M.write.data <= validif(inpred, indata)
  //     _M.write.mask <= validif(inpred, UInt<1>("h1"))

  // CHECK-LABEL: rtl.module @MemSimple(
  firrtl.module @MemSimple(%clock1: !firrtl.clock, %clock2: !firrtl.clock,
                           %inpred: !firrtl.uint<1>, %indata: !firrtl.sint<42>,
                           %result: !firrtl.flip<sint<42>>, 
                           %result2: !firrtl.flip<sint<42>>) {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
    %c0_ui3 = firrtl.constant(0 : ui3) : !firrtl.uint<3>
    %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "rw", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>, !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, wmode: flip<uint<1>>, rdata: sint<42>, wdata: flip<sint<42>>, wmask: flip<uint<1>>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>
  // CHECK:      %_M.ro_data_0, %_M.rw_rdata_0 = rtl.instance "_M" @FIRRTLMem_1_1_1_42_12_0_1_0(%clock1, %true, %c0_i4, %clock1, %true, %c0_i4_0, %true, %true, %0, %clock2, %inpred, %c0_i4_1, %[[mask:.+]], %[[data:.+]]) : (i1, i1, i4, i1, i1, i4, i1, i1, i42, i1, i1, i4, i1, i42) -> (i42, i42)
  // CHECK: rtl.output %_M.ro_data_0, %_M.rw_rdata_0 : i42, i42

      %0 = firrtl.subfield %_M_read("data") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.sint<42>
      firrtl.connect %result, %0 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
      %1 = firrtl.subfield %_M_rw("rdata") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, wmode: flip<uint<1>>, rdata: sint<42>, wdata: flip<sint<42>>, wmask: flip<uint<1>>>) -> !firrtl.sint<42>
      firrtl.connect %result2, %1 : !firrtl.flip<sint<42>>, !firrtl.sint<42>
      %2 = firrtl.subfield %_M_read("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
      firrtl.connect %2, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
      %3 = firrtl.subfield %_M_read("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
      firrtl.connect %3, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
      %4 = firrtl.subfield %_M_read("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
      firrtl.connect %4, %clock1 : !firrtl.flip<clock>, !firrtl.clock

      %5 = firrtl.subfield %_M_rw("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, wmode: flip<uint<1>>, rdata: sint<42>, wdata: flip<sint<42>>, wmask: flip<uint<1>>>) -> !firrtl.flip<uint<4>>
      firrtl.connect %5, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
      %6 = firrtl.subfield %_M_rw("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, wmode: flip<uint<1>>, rdata: sint<42>, wdata: flip<sint<42>>, wmask: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
      firrtl.connect %6, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
      %7 = firrtl.subfield %_M_rw("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, wmode: flip<uint<1>>, rdata: sint<42>, wdata: flip<sint<42>>, wmask: flip<uint<1>>>) -> !firrtl.flip<clock>
      firrtl.connect %7, %clock1 : !firrtl.flip<clock>, !firrtl.clock
      %8 = firrtl.subfield %_M_rw("wmask") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, wmode: flip<uint<1>>, rdata: sint<42>, wdata: flip<sint<42>>, wmask: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
      firrtl.connect %8, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
      %9 = firrtl.subfield %_M_rw("wmode") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, wmode: flip<uint<1>>, rdata: sint<42>, wdata: flip<sint<42>>, wmask: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
      firrtl.connect %9, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

      %10 = firrtl.subfield %_M_write("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<4>>
      firrtl.connect %10, %c0_ui3 : !firrtl.flip<uint<4>>, !firrtl.uint<3>
      %11 = firrtl.subfield %_M_write("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
      firrtl.connect %11, %inpred : !firrtl.flip<uint<1>>, !firrtl.uint<1>
      %12 = firrtl.subfield %_M_write("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<clock>
      firrtl.connect %12, %clock2 : !firrtl.flip<clock>, !firrtl.clock
      %13 = firrtl.subfield %_M_write("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<sint<42>>
      firrtl.connect %13, %indata : !firrtl.flip<sint<42>>, !firrtl.sint<42>
      %14 = firrtl.subfield %_M_write("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>>) -> !firrtl.flip<uint<1>>
      firrtl.connect %14, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  }

  // CHECK-LABEL: rtl.module @IncompleteRead(
  // The read port has no use of the data field.
  firrtl.module @IncompleteRead(%clock1: !firrtl.clock) {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>

    // CHECK:  %_M.ro_data_0 = rtl.instance "_M" @FIRRTLMem_1_0_0_42_12_0_1_0(%clock1, %true, %c0_i4) : (i1, i1, i4) -> i42
    %_M_read = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>
    // Read port.
    %6 = firrtl.subfield %_M_read("addr") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %6, %c0_ui1 : !firrtl.flip<uint<4>>, !firrtl.uint<1>
    %7 = firrtl.subfield %_M_read("en") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %7, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %8 = firrtl.subfield %_M_read("clk") : (!firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: sint<42>>) -> !firrtl.flip<clock>
    firrtl.connect %8, %clock1 : !firrtl.flip<clock>, !firrtl.clock
  }

  // CHECK-LABEL: rtl.module @top_mod() -> (%tmp27: i23) {
  // CHECK-NEXT:    %c0_i23 = rtl.constant 0 : i23
  // CHECK-NEXT:    %c42_i23 = rtl.constant 42 : i23
  // CHECK-NEXT:    rtl.output %c0_i23 : i23
  // CHECK-NEXT:  }
  firrtl.module @top_mod(%tmp27: !firrtl.flip<uint<23>>) {
    %0 = firrtl.wire : !firrtl.flip<uint<0>>
    %c42_ui23 = firrtl.constant(42 : ui23) : !firrtl.uint<23>
    %1 = firrtl.tail %c42_ui23, 23 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    firrtl.connect %0, %1 : !firrtl.flip<uint<0>>, !firrtl.uint<0>
    %2 = firrtl.head %c42_ui23, 0 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    %3 = firrtl.pad %2, 23 : (!firrtl.uint<0>) -> !firrtl.uint<23>
    firrtl.connect %tmp27, %3 : !firrtl.flip<uint<23>>, !firrtl.uint<23>
  }

  //CHECK-LABEL: rtl.module @test_partialconnect(%clock: i1) {
  //CHECK: sv.alwaysff(posedge %clock)
  firrtl.module @test_partialconnect(%clock : !firrtl.clock) {
    %b = firrtl.reg %clock {name = "pcon"} : (!firrtl.clock) -> !firrtl.uint<1>
    %a = firrtl.constant(0 : ui2) : !firrtl.uint<2>
    firrtl.partialconnect %b, %a : !firrtl.uint<1>, !firrtl.uint<2>
  }

  // CHECK-LABEL: rtl.module @SimpleStruct(%source: !rtl.struct<valid: i1, ready: i1, data: i64>) -> (%fldout: i64) {
  // CHECK-NEXT:    %0 = rtl.struct_extract %source["data"] : !rtl.struct<valid: i1, ready: i1, data: i64>
  // CHECK-NEXT:    rtl.output %0 : i64
  // CHECK-NEXT:  }
  firrtl.module @SimpleStruct(%source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              %fldout: !firrtl.flip<uint<64>>) {
    %2 = firrtl.subfield %source ("data") : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
    firrtl.connect %fldout, %2 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
  }

  // CHECK-LABEL: IsInvalidIssue572
  // https://github.com/llvm/circt/issues/572
  firrtl.module @IsInvalidIssue572(%a: !firrtl.analog<1>) {
    
    // CHECK-NEXT: %.invalid_analog = sv.wire : !rtl.inout<i1>
    %0 = firrtl.invalidvalue : !firrtl.analog<1>

    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   %0 = sv.read_inout %a : !rtl.inout<i1>
    // CHECK-NEXT:   %1 = sv.read_inout %.invalid_analog : !rtl.inout<i1>
    // CHECK-NEXT:   sv.connect %a, %1 : i1
    // CHECK-NEXT:   sv.connect %.invalid_analog, %0 : i1
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "verilator" {
    // CHECK-NEXT:     sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     sv.alias %a, %.invalid_analog : !rtl.inout<i1>, !rtl.inout<i1>
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.attach %a, %0 : !firrtl.analog<1>, !firrtl.analog<1>
  }

  // CHECK-LABEL: IsInvalidIssue654
  // https://github.com/llvm/circt/issues/654
  firrtl.module @IsInvalidIssue654() {
    %w = firrtl.wire : !firrtl.flip<uint<0>>
    %0 = firrtl.invalidvalue : !firrtl.uint<0>
    firrtl.connect %w, %0 : !firrtl.flip<uint<0>>, !firrtl.uint<0>
  }

  // CHECK-LABEL: ASQ
  // https://github.com/llvm/circt/issues/699
  firrtl.module @ASQ(%clock: !firrtl.clock, %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    %widx_widx_bin = firrtl.regreset %clock, %reset, %c0_ui1 {name = "widx_widx_bin"} : (!firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>) -> !firrtl.uint<4>
  }

  // CHECK-LABEL: rtl.module @Struct0bits(%source: !rtl.struct<valid: i1, ready: i1, data: i0>) {
  // CHECK-NEXT:    rtl.output 
  // CHECK-NEXT:  }
  firrtl.module @Struct0bits(%source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) {
    %2 = firrtl.subfield %source ("data") : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) -> !firrtl.uint<0>
  }

  // CHECK-LABEL: rtl.module @MemDepth1
  firrtl.module @MemDepth1(%clock: !firrtl.clock, %en: !firrtl.uint<1>,
                         %addr: !firrtl.uint<1>, %data: !firrtl.flip<uint<32>>) {
    // CHECK: %mem0.ro_data_0 = rtl.instance "mem0" @FIRRTLMem_1_0_0_32_1_0_1_1(%clock, %en, %addr) : (i1, i1, i1) -> i32
    // CHECK: rtl.output %mem0.ro_data_0 : i32
    %mem0_load0 = firrtl.mem Old {depth = 1 : i64, name = "mem0", portNames = ["load0"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<32>>
    %0 = firrtl.subfield %mem0_load0("clk") : (!firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<32>>) -> !firrtl.flip<clock>
    firrtl.connect %0, %clock : !firrtl.flip<clock>, !firrtl.clock
    %1 = firrtl.subfield %mem0_load0("addr") : (!firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<32>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %1, %addr : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %2 = firrtl.subfield %mem0_load0("data") : (!firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<32>>) -> !firrtl.uint<32>
    firrtl.connect %data, %2 : !firrtl.flip<uint<32>>, !firrtl.uint<32>
    %3 = firrtl.subfield %mem0_load0("en") : (!firrtl.bundle<addr: flip<uint<1>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<32>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %3, %en : !firrtl.flip<uint<1>>, !firrtl.uint<1>
}

}
