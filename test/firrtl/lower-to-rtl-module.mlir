// RUN: circt-opt -pass-pipeline='lower-firrtl-to-rtl-module' %s -verify-diagnostics  | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Simple"
 firrtl.circuit "Simple" {

   // CHECK-LABEL: rtl.module @Simple(
   // CHECK: %arg0: i4 {rtl.direction = "input", rtl.name = "in1"},
   // CHECK: %arg1: i2 {rtl.direction = "input", rtl.name = "in2"},
   // CHECK: %arg2: i8 {rtl.direction = "input", rtl.name = "in3"},
   // CHECK: %arg3: i4 {rtl.direction = "out", rtl.name = "out4"}) {
   firrtl.module @Simple(%in1: !firrtl.uint<4>,
                        %in2: !firrtl.uint<2>,
                        %in3: !firrtl.sint<8>,
                        %out4: !firrtl.flip<uint<4>>) {

    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    %2 = firrtl.sub %1, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
    %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.sint<3>
    %4 = firrtl.pad %3, 4 : (!firrtl.sint<3>) -> !firrtl.uint<4>
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
    firrtl.connect %out4, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
  }

  // CHECK-LABEL: rtl.module @Print(
  // CHECK: %arg0: i1 {rtl.direction = "input", rtl.name = "clock"},
  // CHECK: %arg1: i1 {rtl.direction = "input", rtl.name = "reset"},
  // CHECK: %arg2: i4 {rtl.direction = "input", rtl.name = "a"},
  // CHECK: %arg3: i4 {rtl.direction = "input", rtl.name = "b"}) {
  firrtl.module @Print(%clock: !firrtl.clock, %reset: !firrtl.uint<1>,
                       %a: !firrtl.uint<4>, %b: !firrtl.uint<4>) {
 
    firrtl.printf %clock, %reset, "No operands!\0A"
    %0 = firrtl.add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
    firrtl.printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.uint<5>, !firrtl.uint<4>
  }

  // CHECK-LABEL: rtl.module @Stop(
  // CHECK: %arg0: i1 {rtl.direction = "input", rtl.name = "clock1"},
  // CHECK: %arg1: i1 {rtl.direction = "input", rtl.name = "clock2"},
  // CHECK: %arg2: i1 {rtl.direction = "input", rtl.name = "reset"}) {
  firrtl.module @Stop(%clock1: !firrtl.clock,
                      %clock2: !firrtl.clock,
                      %reset: !firrtl.uint<1>) {
    firrtl.stop %clock1, %reset, 42
    firrtl.stop %clock2, %reset, 0
  }

  // CHECK-LABEL: firrtl.module @CantLowerArgument(%arg:
  // expected-error @+1 {{cannot lower this port type to RTL}}
  firrtl.module @CantLowerArgument(%arg: !firrtl.bundle<int_1: flip<uint<1>>, int_out: uint<2>>) {
  }
}
