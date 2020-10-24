// RUN: circt-opt -pass-pipeline='lower-firrtl-to-rtl-module' %s -verify-diagnostics  | FileCheck %s

 // The firrtl.circuit should be removed, the main module name moved to an
 // attribute on the module.
 // CHECK-LABEL: {{^}}module attributes {firrtl.mainModule = "Simple"} {
 // CHECK-NOT: firrtl.circuit
 firrtl.circuit "Simple" {

   // CHECK-LABEL: rtl.externmodule @MyParameterizedExtModule(
   // CHECK: i1 {rtl.name = "in"}) -> (i8 {rtl.name = "out"})
   firrtl.extmodule @MyParameterizedExtModule(!firrtl.uint<1> {firrtl.name = "in"}, !firrtl.flip<uint<8>> {firrtl.name = "out"})
      attributes {defname = "name_thing",
                  parameters = {DEFAULT = 0 : i64,
                                DEPTH = 3.242000e+01 : f64,
                                FORMAT = "xyz_timeout=%d\0A",
                                WIDTH = 32 : i8}}

   // CHECK-LABEL: rtl.module @Simple(
   // CHECK: %arg0: i4 {rtl.name = "in1"},
   // CHECK: %arg1: i2 {rtl.name = "in2"},
   // CHECK: %arg2: i8 {rtl.name = "in3"})
   // CHECK: -> (i4 {rtl.name = "out4"}) {
   firrtl.module @Simple(%in1: !firrtl.uint<4>,
                        %in2: !firrtl.uint<2>,
                        %in3: !firrtl.sint<8>,
                        %out4: !firrtl.flip<uint<4>>) {

   // CHECK-NEXT: %0 = firrtl.stdIntCast %arg0 : (i4) -> !firrtl.uint<4>
   // CHECK-NEXT: %1 = firrtl.stdIntCast %arg1 : (i2) -> !firrtl.uint<2>
   // CHECK-NEXT: %2 = firrtl.stdIntCast %arg2 : (i8) -> !firrtl.sint<8>
   // CHECK-NEXT: %3 = rtl.wire : i4
   // CHECK-NEXT: %4 = firrtl.stdIntCast %3 : (i4) -> !firrtl.uint<4>
   // CHECK-NEXT: %5 = firrtl.asNonPassive %4 : (!firrtl.uint<4>) -> !firrtl.flip<uint<4>>

    // CHECK-NEXT: firrtl.asUInt %0
    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK-NEXT: firrtl.sub
    %2 = firrtl.sub %1, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK-NEXT: firrtl.pad %1, 3
    %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.sint<3>
    // CHECK-NEXT: firrtl.pad
    %4 = firrtl.pad %3, 4 : (!firrtl.sint<3>) -> !firrtl.uint<4>
    // CHECK-NEXT: firrtl.xor %1
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
    // CHECK-NEXT: firrtl.connect %5
    firrtl.connect %out4, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    // CHECK-NEXT: rtl.output %3 : i4
  }    // CHECK-NEXT: }

  // CHECK-LABEL: rtl.module @Print(
  // CHECK: %arg0: i1 {rtl.name = "clock"},
  // CHECK: %arg1: i1 {rtl.name = "reset"},
  // CHECK: %arg2: i4 {rtl.name = "a"},
  // CHECK: %arg3: i4 {rtl.name = "b"}) {
  firrtl.module @Print(%clock: !firrtl.clock, %reset: !firrtl.uint<1>,
                       %a: !firrtl.uint<4>, %b: !firrtl.uint<4>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %arg0 : (i1) -> !firrtl.clock
    // CHECK-NEXT: %1 = firrtl.stdIntCast %arg1 : (i1) -> !firrtl.uint<1>
    // CHECK-NEXT: %2 = firrtl.stdIntCast %arg2 : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %3 = firrtl.stdIntCast %arg3 : (i4) -> !firrtl.uint<4>
    
    // CHECK-NEXT: firrtl.printf %0, %1, "No operands!\0A"
    firrtl.printf %clock, %reset, "No operands!\0A"

    // CHECK-NEXT: = firrtl.add %2, %2 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
    %0 = firrtl.add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // CHECK-NEXT: firrtl.printf %0, %1
    firrtl.printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.uint<5>, !firrtl.uint<4>
    // CHECK-NEXT: rtl.output
}  // CHECK-NEXT: }

  // CHECK-LABEL: rtl.module @Stop(
  // CHECK: %arg0: i1 {rtl.name = "clock1"},
  // CHECK: %arg1: i1 {rtl.name = "clock2"},
  // CHECK: %arg2: i1 {rtl.name = "reset"}) {
  firrtl.module @Stop(%clock1: !firrtl.clock,
                      %clock2: !firrtl.clock,
                      %reset: !firrtl.uint<1>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %arg0 : (i1) -> !firrtl.clock
    // CHECK-NEXT: %1 = firrtl.stdIntCast %arg1 : (i1) -> !firrtl.clock
    // CHECK-NEXT: %2 = firrtl.stdIntCast %arg2 : (i1) -> !firrtl.uint<1>

    // CHECK-NEXT: firrtl.stop %0, %2, 42
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT: firrtl.stop %1, %2, 0
    firrtl.stop %clock2, %reset, 0
    // CHECK-NEXT: rtl.output
  }  // CHECK-NEXT: }

  // expected-error @+1 {{cannot lower this port type to RTL}}
  firrtl.module @CantLowerArgument(%arg: !firrtl.bundle<int_1: flip<uint<1>>, int_out: uint<2>>) {
  }   // CHECK-NEXT: }

  // expected-error @+1 {{unexpected operation 'func' in a firrtl.circuit}}
  func @UnknownFunction() {
  }
}
