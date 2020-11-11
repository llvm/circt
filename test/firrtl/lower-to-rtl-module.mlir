// RUN: circt-opt -pass-pipeline='lower-firrtl-to-rtl-module' %s -verify-diagnostics  | FileCheck %s

 // The firrtl.circuit should be removed, the main module name moved to an
 // attribute on the module.
 // CHECK-LABEL: {{^}}module attributes {firrtl.mainModule = "Simple"} {
 // CHECK-NOT: firrtl.circuit
 firrtl.circuit "Simple" {

   // CHECK-LABEL: rtl.externmodule @MyParameterizedExtModule(
   // CHECK: i1 {rtl.name = "in"}) -> (i8 {rtl.name = "out"})
   // CHECK: attributes {verilogName = "name_thing"}
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

    // CHECK-NEXT: firrtl.asUInt %0
    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK-NEXT: firrtl.sub
    %2 = firrtl.sub %1, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK-NEXT: firrtl.pad %1, 3
    %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.sint<3>
    // CHECK-NEXT: firrtl.pad
    %4 = firrtl.pad %3, 4 : (!firrtl.sint<3>) -> !firrtl.uint<4>
    // CHECK-NEXT: [[XOR:%.+]] = firrtl.xor %1
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK-NEXT: [[RESULT:%.+]] = firrtl.stdIntCast [[XOR]]
    firrtl.connect %out4, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    // CHECK-NEXT: rtl.output [[RESULT]] : i4
  }

 // CHECK-LABEL: rtl.module @TestInstance(
  firrtl.module @TestInstance(%u2: !firrtl.uint<2>, %s8: !firrtl.sint<8>,
                              %clock: !firrtl.clock,
                              %reset: !firrtl.uint<1>) {
    // CHECK: [[U2CAST:%.+]] = firrtl.stdIntCast %arg0 : (i2) -> !firrtl.uint<2>

    // CHECK: %in1.wire = rtl.wire : i4
    // CHECK-NEXT: %in2.wire = rtl.wire : i2
    // CHECK-NEXT: %in3.wire = rtl.wire : i8
    // CHECK-NEXT: [[INSTOUT:%.+]] = rtl.instance "xyz" @Simple(%in1.wire, %in2.wire, %in3.wire) : (i4, i2, i8) -> i4
    %xyz = firrtl.instance @Simple {name = "xyz"}
     : !firrtl.bundle<in1: flip<uint<4>>, in2: flip<uint<2>>,
                      in3: flip<sint<8>>, out4: uint<4>>

    // CHECK-NEXT: [[INSTOUTC1:%.+]] = firrtl.stdIntCast [[INSTOUT]] : (i4) -> !firrtl.uint<4>


    // CHECK: [[IN1C:%.+]] = firrtl.stdIntCast %in1.wire : (i4) -> !firrtl.uint<4>
    // CHECK: [[IN1C2:%.+]] = firrtl.asNonPassive [[IN1C]] : (!firrtl.uint<4>) -> !firrtl.flip<uint<4>>
    // CHECK:  firrtl.connect [[IN1C2]], [[U2CAST]]
    %0 = firrtl.subfield %xyz("in1") : (!firrtl.bundle<in1: flip<uint<4>>, in2: flip<uint<2>>, in3: flip<sint<8>>, out4: uint<4>>) -> !firrtl.flip<uint<4>>
    firrtl.connect %0, %u2 : !firrtl.flip<uint<4>>, !firrtl.uint<2>

    %1 = firrtl.subfield %xyz("in2") : (!firrtl.bundle<in1: flip<uint<4>>, in2: flip<uint<2>>, in3: flip<sint<8>>, out4: uint<4>>) -> !firrtl.flip<uint<2>>
    firrtl.connect %1, %u2 : !firrtl.flip<uint<2>>, !firrtl.uint<2>

    %2 = firrtl.subfield %xyz("in3")  : (!firrtl.bundle<in1: flip<uint<4>>, in2: flip<uint<2>>, in3: flip<sint<8>>, out4: uint<4>>) -> !firrtl.flip<sint<8>>
    firrtl.connect %2, %s8 : !firrtl.flip<sint<8>>, !firrtl.sint<8>

    %3 = firrtl.subfield %xyz("out4")  : (!firrtl.bundle<in1: flip<uint<4>>, in2: flip<uint<2>>, in3: flip<sint<8>>, out4: uint<4>>) -> !firrtl.uint<4>

    // CHECK: firrtl.printf {{.*}}"%x"([[INSTOUTC1]])
    firrtl.printf %clock, %reset, "%x"(%3) : !firrtl.uint<4>
 

    // Parameterized module reference.
    // rtl.instance carries the parameters, unlike at the FIRRTL layer.

    // CHECK-NEXT: %in.wire = rtl.wire : i1

    // CHECK-NEXT: [[OUT:%.+]] = rtl.instance "myext" @MyParameterizedExtModule(%in.wire)  {parameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}} : (i1) -> i8
    %myext = firrtl.instance @MyParameterizedExtModule {name = "myext"}
      : !firrtl.bundle<in: flip<uint<1>>, out: uint<8>>

    // CHECK-NEXT: [[OUTC:%.+]] = firrtl.stdIntCast [[OUT]] : (i8) -> !firrtl.uint<8>
    // CHECK-NEXT: [[INC:%.+]]  = firrtl.stdIntCast %in.wire : (i1) -> !firrtl.uint<1>
    // CHECK-NEXT: [[INC2:%.+]] = firrtl.asNonPassive [[INC]] : (!firrtl.uint<1>) -> !firrtl.flip<uint<1>>

    // CHECK-NEXT: firrtl.connect [[INC2]], {{.*}} : !firrtl.flip<uint<1>>, !firrtl.uint<1>
    %9 = firrtl.subfield %myext("in") : (!firrtl.bundle<in: flip<uint<1>>, out: uint<8>>) -> !firrtl.flip<uint<1>>
    firrtl.connect %9, %reset : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // CHECK-NEXT: firrtl.printf {{.*}}, {{.*}}, "Something interesting! %x"([[OUTC]]) : !firrtl.uint<8>
    %10 = firrtl.subfield %myext("out") : (!firrtl.bundle<in: flip<uint<1>>, out: uint<8>>) -> !firrtl.uint<8>
    firrtl.printf %clock, %reset, "Something interesting! %x"(%10) : !firrtl.uint<8>
  }

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
  }

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
  }

  // expected-error @+1 {{cannot lower this port type to RTL}}
  firrtl.module @CantLowerArgument(%arg: !firrtl.bundle<int_1: flip<uint<1>>, int_out: uint<2>>) {
  }   // CHECK-NEXT: }

  // expected-error @+1 {{unexpected operation 'func' in a firrtl.circuit}}
  func @UnknownFunction() {
  }

  // CHECK-LABEL: rtl.module @OutputFirst(%arg0: i1 {rtl.name = "in1"}, %arg1: i4 {rtl.name = "in4"}) -> (i4 {rtl.name = "out4"}) {
  firrtl.module @OutputFirst(%out4: !firrtl.flip<uint<4>>,
                             %in1: !firrtl.uint<1>,
                             %in4: !firrtl.uint<4>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %arg0 : (i1) -> !firrtl.uint<1>
    // CHECK-NEXT: %1 = firrtl.stdIntCast %arg1 : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %2 = firrtl.stdIntCast %1 : (!firrtl.uint<4>) -> i4
    firrtl.connect %out4, %in4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: rtl.output %2 : i4
  }

  // CHECK-LABEL: rtl.module @PortMadness(
  // CHECK: %arg0: i4 {rtl.name = "inA"}, %arg1: i4 {rtl.name = "inB"}, %arg2: i4 {rtl.name = "inC"})
  // CHECK: -> (i4 {rtl.name = "outA"}, i4 {rtl.name = "outB"}, i4 {rtl.name = "outC"}, i4 {rtl.name = "outD"}) {
  firrtl.module @PortMadness(%inA: !firrtl.uint<4>,
                             %inB: !firrtl.uint<4>,
                             %inC: !firrtl.uint<4>,
                             %outA: !firrtl.flip<uint<4>>,
                             %outB: !firrtl.flip<uint<4>>,
                             %outC: !firrtl.flip<uint<4>>,
                             %outD: !firrtl.flip<uint<4>>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %arg0 : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT:%1 = firrtl.stdIntCast %arg1 : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT:%2 = firrtl.stdIntCast %arg2 : (i4) -> !firrtl.uint<4>

    // CHECK: [[OUTB:%.+]] = rtl.wire : i4
    // CHECK: [[OUTC:%.+]] = rtl.wire : i4
    // CHECK: [[OUTD:%.+]] = rtl.wire : i4

    // Normal
    firrtl.connect %outA, %inA : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // Multi connect
    firrtl.connect %outB, %inA : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    firrtl.connect %outB, %inB : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // Use of output as an input.
    %tmp = firrtl.asPassive %outC : (!firrtl.flip<uint<4>>) -> !firrtl.uint<4>
    %0 = firrtl.sub %inA, %tmp : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // Use of an input as an output.
    %tmp2 = firrtl.asNonPassive %inC : (!firrtl.uint<4>) -> !firrtl.flip<uint<4>>
    firrtl.connect %tmp2, %inA : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // No connections to outD.

    // CHECK: [[ACAST:%.+]] = firrtl.stdIntCast %0 : (!firrtl.uint<4>) -> i4
    // CHECK: rtl.output [[ACAST]], %3,
  }
}
