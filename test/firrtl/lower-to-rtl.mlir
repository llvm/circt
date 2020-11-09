// RUN: circt-opt -lower-firrtl-to-rtl %s | FileCheck %s

module attributes {firrtl.mainModule = "Simple"} {

  // CHECK-LABEL: rtl.module @Simple
  rtl.module @Simple(%arg0: i4 {rtl.name = "in1"}, %arg1: i2 {rtl.name = "in2"}, %arg2: i8 {rtl.name = "in3"}) -> (i4 {rtl.name = "out4"}) {
    %in1 = firrtl.stdIntCast %arg0 : (i4) -> !firrtl.uint<4>
    %in2 = firrtl.stdIntCast %arg1 : (i2) -> !firrtl.uint<2>
    %in3 = firrtl.stdIntCast %arg2 : (i8) -> !firrtl.sint<8>
    // CHECK-NEXT: %tmp3 = rtl.wire : i4
    %tmp3 = rtl.wire : i4
    %tmp4 = firrtl.stdIntCast %tmp3 : (i4) -> !firrtl.uint<4>
    %out4 = firrtl.asNonPassive %tmp4 : (!firrtl.uint<4>) -> !firrtl.flip<uint<4>>

    // CHECK: rtl.constant(-4 : i4) : i4
    %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>

    // CHECK: rtl.constant(2 : i3) : i3
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>

    // CHECK: %0 = rtl.add %c-4_i4, %c-4_i4 : i4
    %0 = firrtl.add %c12_ui4, %c12_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK-NEXT: [[SUB:%.+]] = rtl.sub %0, %arg0 : i4
    %2 = firrtl.sub %0, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[PADRES:%.+]] = rtl.sext %arg1 : (i2) -> i3
    %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // CHECK: [[PADRES2:%.+]] = rtl.zext [[PADRES]] : (i3) -> i4
    %4 = firrtl.pad %3, 4 : (!firrtl.sint<3>) -> !firrtl.uint<4>

    // CHECK: [[IN2EXT:%.+]] = rtl.zext %arg1 : (i2) -> i4
    // CHECK: [[XOR:%.+]] = rtl.xor [[IN2EXT]], [[PADRES2]] : i4
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: rtl.and [[XOR]]
    %and = firrtl.and %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[CONCAT1:%.+]] = rtl.concat [[PADRES2]], [[XOR]] : (i4, i4) -> i8
    %6 = firrtl.cat %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>

    // CHECK: rtl.concat %arg0, %arg1
    %7 = firrtl.cat %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>

    // CHECK-NEXT: rtl.connect [[SUB]], [[PADRES2]] : i4
    firrtl.connect %2, %4 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: rtl.connect %tmp3, [[XOR]] : i4
    firrtl.connect %out4, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: [[ZEXT:%.+]] = rtl.zext %arg1 : (i2) -> i4
    // CHECK-NEXT: rtl.connect %tmp3, [[ZEXT]] : i4
    firrtl.connect %out4, %in2 : !firrtl.flip<uint<4>>, !firrtl.uint<2>

    // CHECK-NEXT: %test-name = rtl.wire : i4
    firrtl.wire {name = "test-name"} : !firrtl.uint<4>

    // CHECK-NEXT: = rtl.wire : i2
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

    // CHECK-NEXT: = rtl.extract %arg2 from 7 : (i8) -> i1
    %13 = firrtl.shr %in3, 8 : (!firrtl.sint<8>) -> !firrtl.sint<1>

    // CHECK-NEXT: [[ZERO:%.+]] = rtl.constant(0 : i3) : i3
    // CHECK-NEXT: = rtl.concat [[CONCAT1]], [[ZERO]] : (i8, i3) -> i11
    %14 = firrtl.shl %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<11>

    // CHECK-NEXT: = rtl.xorr [[CONCAT1]] : i8
    %15 = firrtl.xorr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.andr [[CONCAT1]] : i8
    %16 = firrtl.andr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = rtl.orr [[CONCAT1]] : i8
    %17 = firrtl.orr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[ZEXTC1:%.+]] = rtl.zext [[CONCAT1]] : (i8) -> i12
    // CHECK-NEXT: [[ZEXT2:%.+]] = rtl.zext [[SUB]] : (i4) -> i12
    // CHECK-NEXT: = rtl.mul [[ZEXTC1]], [[ZEXT2]] : i12
    %18 = firrtl.mul %6, %2 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<12>

    // CHECK-NEXT: [[IN3SEXT:%.+]] = rtl.sext %arg2 : (i8) -> i9
    // CHECK-NEXT: [[PADRESSEXT:%.+]] = rtl.sext [[PADRES]] : (i3) -> i9
    // CHECK-NEXT: = rtl.div [[IN3SEXT]], [[PADRESSEXT]] : i9
    %19 = firrtl.div %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<9>

    // CHECK-NEXT: [[IN3TRUNC:%.+]] = rtl.extract %arg2 from 0 : (i8) -> i3
    // CHECK-NEXT: = rtl.mod [[IN3TRUNC]], [[PADRES]] : i3
    %20 = firrtl.rem %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[IN3TRUNC:%.+]] = rtl.extract %arg2 from 0 : (i8) -> i3
    // CHECK-NEXT: = rtl.mod [[PADRES]], [[IN3TRUNC]] : i3
    %21 = firrtl.rem %3, %in3 : (!firrtl.sint<3>, !firrtl.sint<8>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[WIRE:%n1]] = rtl.wire : i2
    // CHECK-NEXT: rtl.connect [[WIRE]], %arg1 : i2
    %n1 = firrtl.node %in2  {name = "n1"} : !firrtl.uint<2>

    // Nodes with no names are just dropped.
    %22 = firrtl.node %n1 : !firrtl.uint<2>

    // CHECK-NEXT: %false_{{.*}} = rtl.constant(false) : i1
    // CHECK-NEXT: [[CVT:%.+]] = rtl.concat %false_{{.*}}, %arg1 : (i1, i2) -> i3
    %23 = firrtl.cvt %22 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // CHECK-NEXT: %c-1_i3 = rtl.constant(-1 : i3) : i3
    // CHECK-NEXT: [[XOR:%.+]] = rtl.xor [[CVT]], %c-1_i3 : i3
    %24 = firrtl.not %23 : (!firrtl.sint<3>) -> !firrtl.uint<3>

    %s24 = firrtl.asSInt %24 : (!firrtl.uint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[SEXT:%.+]] = rtl.sext [[XOR]] : (i3) -> i4
    // CHECK-NEXT: %c0_i4 = rtl.constant(0 : i4) : i4
    // CHECK-NEXT: [[SUB:%.+]] = rtl.sub %c0_i4, [[SEXT]] : i4
    %25 = firrtl.neg %s24 : (!firrtl.sint<3>) -> !firrtl.sint<4>

    // CHECK-NEXT: [[CVT4:%.+]] = rtl.sext [[CVT]] : (i3) -> i4
    // CHECK-NEXT: rtl.mux %false, [[CVT4]], [[SUB]] : i4
    %26 = firrtl.mux(%12, %23, %25) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.sint<4>
  
    // Noop
    %27 = firrtl.validif %12, %18 : (!firrtl.uint<1>, !firrtl.uint<12>) -> !firrtl.uint<12>
    // CHECK-NEXT: rtl.andr
    %28 = firrtl.andr %27 : (!firrtl.uint<12>) -> !firrtl.uint<1>

    // CHECK-NEXT: rtl.output %tmp3 : i4
    rtl.output %tmp3 : i4
  }

//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: rtl.module @Print
  rtl.module @Print(%arg0: i1 {rtl.name = "clock"}, %arg1: i1 {rtl.name = "reset"}, %arg2: i4 {rtl.name = "a"}, %arg3: i4 {rtl.name = "b"}) {
    %clock = firrtl.stdIntCast %arg0 : (i1) -> !firrtl.clock
    %reset = firrtl.stdIntCast %arg1 : (i1) -> !firrtl.uint<1>
    %a = firrtl.stdIntCast %arg2 : (i4) -> !firrtl.uint<4>
    %b = firrtl.stdIntCast %arg3 : (i4) -> !firrtl.uint<4>
 
    // CHECK-NEXT: sv.alwaysat_posedge %arg0 {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     [[TV:%.+]] = sv.textual_value "`PRINTF_COND_" : i1
    // CHECK-NEXT:     [[AND:%.+]] = rtl.and [[TV]], %arg1
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       sv.fwrite "No operands!\0A"
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
   firrtl.printf %clock, %reset, "No operands!\0A"

    // CHECK: [[ADD:%.+]] = rtl.add
    %0 = firrtl.add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // CHECK: sv.fwrite "Hi %x %x\0A"({{.*}}) : i5, i4
    firrtl.printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.uint<5>, !firrtl.uint<4>

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
  rtl.module @Stop(%arg0: i1 {rtl.name = "clock1"}, %arg1: i1 {rtl.name = "clock2"}, %arg2: i1 {rtl.name = "reset"}) {
    %clock1 = firrtl.stdIntCast %arg0 : (i1) -> !firrtl.clock
    %clock2 = firrtl.stdIntCast %arg1 : (i1) -> !firrtl.clock
    %reset = firrtl.stdIntCast %arg2 : (i1) -> !firrtl.uint<1>

    // CHECK-NEXT: sv.alwaysat_posedge %arg0 {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     %0 = sv.textual_value "`STOP_COND_" : i1
    // CHECK-NEXT:     %1 = rtl.and %0, %arg2 : i1
    // CHECK-NEXT:     sv.if %1 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT: sv.alwaysat_posedge %arg1 {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     %0 = sv.textual_value "`STOP_COND_" : i1
    // CHECK-NEXT:     %1 = rtl.and %0, %arg2 : i1
    // CHECK-NEXT:     sv.if %1 {
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
  rtl.module @Verification(%arg0: i1 {rtl.name = "clock"}, %arg1: i1 {rtl.name = "aCond"}, %arg2: i1 {rtl.name = "aEn"}, %arg3: i1 {rtl.name = "bCond"}, %arg4: i1 {rtl.name = "bEn"}, %arg5: i1 {rtl.name = "cCond"}, %arg6: i1 {rtl.name = "cEn"}) {
    %clock = firrtl.stdIntCast %arg0 : (i1) -> !firrtl.clock
    %aCond = firrtl.stdIntCast %arg1 : (i1) -> !firrtl.uint<1>
    %aEn = firrtl.stdIntCast %arg2 : (i1) -> !firrtl.uint<1>
    %bCond = firrtl.stdIntCast %arg3 : (i1) -> !firrtl.uint<1>
    %bEn = firrtl.stdIntCast %arg4 : (i1) -> !firrtl.uint<1>
    %cCond = firrtl.stdIntCast %arg5 : (i1) -> !firrtl.uint<1>
    %cEn = firrtl.stdIntCast %arg6 : (i1) -> !firrtl.uint<1>

    // CHECK-NEXT: sv.alwaysat_posedge %arg0 {
    // CHECK-NEXT:   sv.if %arg2 {
    // CHECK-NEXT:     sv.assert %arg1 : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assert %clock, %aCond, %aEn, "assert0"
    // CHECK-NEXT: sv.alwaysat_posedge %arg0 {
    // CHECK-NEXT:   sv.if %arg4 {
    // CHECK-NEXT:     sv.assume %arg3  : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assume %clock, %bCond, %bEn, "assume0"
    // CHECK-NEXT: sv.alwaysat_posedge %arg0 {
    // CHECK-NEXT:   sv.if %arg6 {
    // CHECK-NEXT:     sv.cover %arg5 : i1
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.cover %clock, %cCond, %cEn, "cover0"
    // CHECK-NEXT: rtl.output
    rtl.output
  }

  rtl.module @bar(%arg0: i1 {rtl.name = "io_cpu_flush"}) {
    rtl.output
  }

  // CHECK-LABEL: rtl.module @foo
  rtl.module @foo() {
    // CHECK-NEXT:  %io_cpu_flush.wire = rtl.wire : i1
    %io_cpu_flush.wire = rtl.wire : i1
    // CHECK-NEXT: rtl.instance "fetch"
    rtl.instance "fetch" @bar(%io_cpu_flush.wire)  : (i1) -> ()
    %0 = firrtl.stdIntCast %io_cpu_flush.wire : (i1) -> !firrtl.uint<1>
    %1454 = firrtl.asNonPassive %0 : (!firrtl.uint<1>) -> !firrtl.flip<uint<1>>

    %hits_1_7 = firrtl.node %1454 {name = "hits_1_7"} : !firrtl.flip<uint<1>>
    // CHECK-NEXT:  %hits_1_7 = rtl.wire : i1
    // CHECK-NEXT:  rtl.connect %hits_1_7, %io_cpu_flush.wire : i1
    %1455 = firrtl.asPassive %hits_1_7 : (!firrtl.flip<uint<1>>) -> !firrtl.uint<1>
  }
}

