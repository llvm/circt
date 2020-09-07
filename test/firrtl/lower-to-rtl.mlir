// RUN: circt-opt -pass-pipeline='firrtl.circuit(lower-firrtl-to-rtl)' %s | FileCheck %s

 firrtl.circuit "Circuit" {

  // CHECK-LABEL: firrtl.module @Simple
  firrtl.module @Simple(%in1: !firrtl.uint<4>,
                        %in2: !firrtl.uint<2>,
                        %in3: !firrtl.sint<8>,
                        %out1: !firrtl.flip<uint<4>>) {

    // CHECK: rtl.constant(-4 : i4) : i4
    %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>

    // CHECK: rtl.constant(2 : i3) : i3
    %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>

    // CHECK: %0 = rtl.add %c-4_i4, %c-4_i4 : i4
    %0 = firrtl.add %c12_ui4, %c12_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: %1 = firrtl.stdIntCast %in1 : (!firrtl.uint<4>) -> i4
    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: %2 = rtl.sub %0, %1 : i4
    %2 = firrtl.sub %0, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: %3 = firrtl.stdIntCast %in2 : (!firrtl.uint<2>) -> i2
    // CHECK: %4 = rtl.sext %3 : (i2) -> i3
    %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // CHECK: %5 = rtl.zext %4 : (i3) -> i4
    %4 = firrtl.pad %3, 4 : (!firrtl.sint<3>) -> !firrtl.uint<4>

    // CHECK: %6 = firrtl.stdIntCast %in2 : (!firrtl.uint<2>) -> i2
    // CHECK: %7 = rtl.zext %6 : (i2) -> i4
    // CHECK: [[XOR:%.+]] = rtl.xor %7, %5 : i4
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: rtl.and [[XOR]]
    %and = firrtl.and %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[CONCAT1:%.+]] = rtl.concat %5, [[XOR]] : (i4, i4) -> i8
    %6 = firrtl.cat %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>

    // CHECK:  [[CAST1:%.+]] = firrtl.stdIntCast %in1
    // CHECK:  [[CAST2:%.+]] = firrtl.stdIntCast %in2
    // CHECK: rtl.concat [[CAST1]], [[CAST2]]
    %7 = firrtl.cat %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>

    // CHECK-NEXT: rtl.connect %2, %5 : i4
    firrtl.connect %2, %4 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: [[CAST:%.+]] = firrtl.stdIntCast %out1 : (!firrtl.flip<uint<4>>) -> i4
    // CHECK-NEXT: rtl.connect [[CAST]], [[XOR]] : i4
    firrtl.connect %out1, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: [[CAST1:%.+]] = firrtl.stdIntCast %out1 : (!firrtl.flip<uint<4>>) -> i4
    // CHECK-NEXT: [[CAST2:%.+]] = firrtl.stdIntCast %in2 : (!firrtl.uint<2>) -> i2
    // CHECK-NEXT: [[ZEXT:%.+]] = rtl.zext [[CAST2]] : (i2) -> i4
    // CHECK-NEXT: rtl.connect [[CAST1]], [[ZEXT]] : i4
    firrtl.connect %out1, %in2 : !firrtl.flip<uint<4>>, !firrtl.uint<2>

    // CHECK-NEXT: = rtl.wire {name = "test-name"} : i4
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

    // CHECK-NEXT: [[CAST:%.+]] = firrtl.stdIntCast %in3
    // CHECK-NEXT: = rtl.extract [[CAST]] from 7 : (i8) -> i1
    %13 = firrtl.shr %in3, 8 : (!firrtl.sint<8>) -> !firrtl.sint<1>

    // CHECK-NEXT: [[ZERO:%.+]] = rtl.constant(0 : i3) : i3
    // CHECK-NEXT: = rtl.concat [[CONCAT1]], [[ZERO]] : (i8, i3) -> i11
    %14 = firrtl.shl %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<11>
  }


//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: firrtl.module @Print
  firrtl.module @Print(%clock: !firrtl.clock, %reset: !firrtl.uint<1>,
                       %a: !firrtl.uint<4>, %b: !firrtl.uint<4>) {
 
    // CHECK-NEXT: [[CL:%.+]] = firrtl.stdIntCast %clock : (!firrtl.clock) -> i1
    // CHECK-NEXT: [[R:%.+]] = firrtl.stdIntCast %reset : (!firrtl.uint<1>) -> i1
    // CHECK-NEXT: sv.alwaysat_posedge [[CL]] {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     [[TV:%.+]] = sv.textual_value "`PRINTF_COND_" : i1
    // CHECK-NEXT:     [[AND:%.+]] = rtl.and [[TV]], [[R]]
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
  }



// module Stop3 :
//    input clock1: Clock
//    input clock2: Clock
//    input reset: UInt<1>
//    stop(clock1, reset, 42)
//    stop(clock2, reset, 0)

  // CHECK-LABEL: firrtl.module @Stop
  firrtl.module @Stop(%clock1: !firrtl.clock, %clock2: !firrtl.clock, %reset: !firrtl.uint<1>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %clock1 : (!firrtl.clock) -> i1
    // CHECK-NEXT: %1 = firrtl.stdIntCast %reset : (!firrtl.uint<1>) -> i1
    // CHECK-NEXT: sv.alwaysat_posedge %0 {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     %4 = sv.textual_value "`STOP_COND_" : i1
    // CHECK-NEXT:     %5 = rtl.and %4, %1 : i1
    // CHECK-NEXT:     sv.if %5 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT: %2 = firrtl.stdIntCast %clock2 : (!firrtl.clock) -> i1
    // CHECK-NEXT: %3 = firrtl.stdIntCast %reset : (!firrtl.uint<1>) -> i1
    // CHECK-NEXT: sv.alwaysat_posedge %2 {
    // CHECK-NEXT:   sv.ifdef "!SYNTHESIS" {
    // CHECK-NEXT:     %4 = sv.textual_value "`STOP_COND_" : i1
    // CHECK-NEXT:     %5 = rtl.and %4, %3 : i1
    // CHECK-NEXT:     sv.if %5 {
    // CHECK-NEXT:       sv.finish
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock2, %reset, 0
  }
}
