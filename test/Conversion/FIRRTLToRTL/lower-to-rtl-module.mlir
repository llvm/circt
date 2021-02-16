// RUN: circt-opt -lower-firrtl-to-rtl-module %s -verify-diagnostics  | FileCheck %s

// The firrtl.circuit should be removed, the main module name moved to an
// attribute on the module.
// CHECK-LABEL: {{^}}module attributes {firrtl.mainModule = "Simple"} {
// CHECK-NOT: firrtl.circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef "RANDOMIZE_GARBAGE_ASSIGN"  {
// CHECK-NEXT:   sv.verbatim "`define RANDOMIZE"
// CHECK-NEXT:  }
firrtl.circuit "Simple" {

   // CHECK-LABEL: rtl.module.extern @MyParameterizedExtModule(
   // CHECK: i1 {rtl.name = "in"}) -> (%out: i8)
   // CHECK: attributes {verilogName = "name_thing"}
   firrtl.extmodule @MyParameterizedExtModule(!firrtl.uint<1> {firrtl.name = "in"}, !firrtl.flip<uint<8>> {firrtl.name = "out"})
      attributes {defname = "name_thing",
                  parameters = {DEFAULT = 0 : i64,
                                DEPTH = 3.242000e+01 : f64,
                                FORMAT = "xyz_timeout=%d\0A",
                                WIDTH = 32 : i8}}

   // CHECK-LABEL: rtl.module @Simple(%in1: i4, %in2: i2, %in3: i8) -> (%out4: i4) {
   firrtl.module @Simple(%in1: !firrtl.uint<4>,
                        %in2: !firrtl.uint<2>,
                        %in3: !firrtl.sint<8>,
                        %out4: !firrtl.flip<uint<4>>) {

   // CHECK-NEXT: %0 = firrtl.stdIntCast %in1 : (i4) -> !firrtl.uint<4>
   // CHECK-NEXT: %1 = firrtl.stdIntCast %in2 : (i2) -> !firrtl.uint<2>
   // CHECK-NEXT: %2 = firrtl.stdIntCast %in3 : (i8) -> !firrtl.sint<8>

    // CHECK-NEXT: firrtl.asUInt %0
    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK-NEXT: firrtl.sub
    %2 = firrtl.sub %1, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // CHECK-NEXT: firrtl.pad %1, 3
    %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.uint<3>
    // CHECK-NEXT: firrtl.pad
    %4 = firrtl.pad %3, 4 : (!firrtl.uint<3>) -> !firrtl.uint<4>
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
    // CHECK-NEXT: [[U2CAST:%.+]] = firrtl.stdIntCast %u2 : (i2) -> !firrtl.uint<2>
    // CHECK-NEXT: [[S8CAST:%.+]] = firrtl.stdIntCast %s8 : (i8) -> !firrtl.sint<8>
    // CHECK-NEXT: [[CLOCKCAST:%.+]] = firrtl.stdIntCast %clock : (i1) -> !firrtl.clock
    // CHECK-NEXT: [[RESETCAST:%.+]] = firrtl.stdIntCast %reset : (i1) -> !firrtl.uint<1>

    // CHECK: [[ARG1:%.+]] = firrtl.pad [[U2CAST]], 4
    // CHECK-NEXT: [[ARG1CAST:%.+]] = firrtl.stdIntCast [[ARG1]] : (!firrtl.uint<4>) -> i4
    // CHECK-NEXT: %xyz.out4 = rtl.instance "xyz" @Simple([[ARG1CAST]], %u2, %s8) : (i4, i2, i8) -> i4
    %xyz:4 = firrtl.instance @Simple {name = "xyz", portNames=["in1", "in2", "in3", "out4"]}
     : !firrtl.flip<uint<4>>, !firrtl.flip<uint<2>>, !firrtl.flip<sint<8>>, !firrtl.uint<4>

    // CHECK-NEXT: [[INSTOUTC1:%.+]] = firrtl.stdIntCast %xyz.out4 : (i4) -> !firrtl.uint<4>

    firrtl.connect %xyz#0, %u2 : !firrtl.flip<uint<4>>, !firrtl.uint<2>

    // CHECK-NOT: firrtl.connect
    firrtl.connect %xyz#1, %u2 : !firrtl.flip<uint<2>>, !firrtl.uint<2>

    firrtl.connect %xyz#2, %s8 : !firrtl.flip<sint<8>>, !firrtl.sint<8>

    // CHECK: firrtl.printf {{.*}}"%x"([[INSTOUTC1]])
    firrtl.printf %clock, %reset, "%x"(%xyz#3) : !firrtl.uint<4>
 

    // Parameterized module reference.
    // rtl.instance carries the parameters, unlike at the FIRRTL layer.

    // CHECK-NEXT: [[OUT:%.+]] = rtl.instance "myext" @MyParameterizedExtModule(%reset)  {parameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}} : (i1) -> i8
    %myext:2 = firrtl.instance @MyParameterizedExtModule {name = "myext", portNames=["in", "out"]}
      : !firrtl.flip<uint<1>>, !firrtl.uint<8>

    // CHECK-NEXT: [[OUTC:%.+]] = firrtl.stdIntCast [[OUT]] : (i8) -> !firrtl.uint<8>

    firrtl.connect %myext#0, %reset : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // CHECK-NEXT: firrtl.printf {{.*}}, {{.*}}, "Something interesting! %x"([[OUTC]]) : !firrtl.uint<8>
    firrtl.printf %clock, %reset, "Something interesting! %x"(%myext#1) : !firrtl.uint<8>
  }

  // CHECK-LABEL: rtl.module @Print(%clock: i1, %reset: i1, %a: i4, %b: i4) {
  firrtl.module @Print(%clock: !firrtl.clock, %reset: !firrtl.uint<1>,
                       %a: !firrtl.uint<4>, %b: !firrtl.uint<4>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %clock : (i1) -> !firrtl.clock
    // CHECK-NEXT: %1 = firrtl.stdIntCast %reset : (i1) -> !firrtl.uint<1>
    // CHECK-NEXT: %2 = firrtl.stdIntCast %a : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %3 = firrtl.stdIntCast %b : (i4) -> !firrtl.uint<4>
    
    // CHECK-NEXT: firrtl.printf %0, %1, "No operands!\0A"
    firrtl.printf %clock, %reset, "No operands!\0A"

    // CHECK-NEXT: = firrtl.add %2, %2 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
    %0 = firrtl.add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // CHECK-NEXT: firrtl.printf %0, %1
    firrtl.printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.uint<5>, !firrtl.uint<4>
    // CHECK-NEXT: rtl.output
  }

  // CHECK-LABEL: rtl.module @Stop(%clock1: i1, %clock2: i1, %reset: i1) {
  firrtl.module @Stop(%clock1: !firrtl.clock,
                      %clock2: !firrtl.clock,
                      %reset: !firrtl.uint<1>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %clock1 : (i1) -> !firrtl.clock
    // CHECK-NEXT: %1 = firrtl.stdIntCast %clock2 : (i1) -> !firrtl.clock
    // CHECK-NEXT: %2 = firrtl.stdIntCast %reset : (i1) -> !firrtl.uint<1>

    // CHECK-NEXT: firrtl.stop %0, %2, 42
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT: firrtl.stop %1, %2, 0
    firrtl.stop %clock2, %reset, 0
    // CHECK-NEXT: rtl.output
  }

  // CHECK-LABEL: rtl.module @OutputFirst(%in1: i1, %in4: i4) -> (%out4: i4) {
  firrtl.module @OutputFirst(%out4: !firrtl.flip<uint<4>>,
                             %in1: !firrtl.uint<1>,
                             %in4: !firrtl.uint<4>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %in1 : (i1) -> !firrtl.uint<1>
    // CHECK-NEXT: %1 = firrtl.stdIntCast %in4 : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %2 = firrtl.stdIntCast %1 : (!firrtl.uint<4>) -> i4
    firrtl.connect %out4, %in4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: rtl.output %2 : i4
  }

  // CHECK-LABEL: rtl.module @PortMadness(
  // CHECK: %inA: i4, %inB: i4, %inC: i4, %inE: i3)
  // CHECK: -> (%outA: i4, %outB: i4, %outC: i4, %outD: i4, %outE: i4) {
  firrtl.module @PortMadness(%inA: !firrtl.uint<4>,
                             %inB: !firrtl.uint<4>,
                             %inC: !firrtl.uint<4>,
                             %outA: !firrtl.flip<uint<4>>,
                             %outB: !firrtl.flip<uint<4>>,
                             %outC: !firrtl.flip<uint<4>>,
                             %outD: !firrtl.flip<uint<4>>,
                             %inE: !firrtl.uint<3>,
                             %outE: !firrtl.flip<uint<4>>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %inA : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %1 = firrtl.stdIntCast %inB : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %2 = firrtl.stdIntCast %inC : (i4) -> !firrtl.uint<4>

    // CHECK: [[OUTC:%.+]] = firrtl.wire : !firrtl.flip<uint<4>>
    // CHECK: [[OUTD:%.+]] = firrtl.wire : !firrtl.flip<uint<4>>

    // CHECK: [[INE:%.+]] = firrtl.stdIntCast %inE : (i3) -> !firrtl.uint<3>

    // Normal
    firrtl.connect %outA, %inA : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // Multi connect
    firrtl.connect %outB, %inA : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    firrtl.connect %outB, %inB : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // Use of output as an input.
    %tmp = firrtl.asPassive %outC : !firrtl.flip<uint<4>>
    %0 = firrtl.sub %inA, %tmp : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // Use of an input as an output.
    // NOTE: This isn't valid but needs to be accepted until the verifier
    // rejects it.
    %tmp2 = firrtl.asNonPassive %inC : !firrtl.flip<uint<4>>
    firrtl.connect %tmp2, %inA : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // No connections to outD.

    firrtl.connect %outE, %inE : !firrtl.flip<uint<4>>, !firrtl.uint<3>

    // CHECK: [[OUTBY:%.+]] = comb.merge %inB, %inA : i4

    // CHECK: [[OUTCX:%.+]] = firrtl.asPassive [[OUTC]]
    // CHECK: [[OUTCY:%.+]] = firrtl.stdIntCast [[OUTCX]]
    // CHECK: [[OUTDX:%.+]] = firrtl.asPassive [[OUTD]]
    // CHECK: [[OUTDY:%.+]] = firrtl.stdIntCast [[OUTDX]]

    // Extension for outE
    // CHECK: [[OUTE:%.+]] = firrtl.pad [[INE]], 4 : (!firrtl.uint<3>) -> !firrtl.uint<4>
    // CHECK: [[OUTE_CAST:%.+]] = firrtl.stdIntCast [[OUTE]]
    // CHECK: rtl.output %inA, [[OUTBY]], [[OUTCY]], [[OUTDY]], [[OUTE_CAST]]
  }

  // CHECK-LABEL: rtl.module @Analog(%a1: !rtl.inout<i1>) -> (%outClock: i1) {
  // CHECK-NEXT:    %0 = firrtl.analogInOutCast %a1 : (!rtl.inout<i1>) -> !firrtl.analog<1>
  // CHECK-NEXT:    %1 = firrtl.asClock %0 : (!firrtl.analog<1>) -> !firrtl.clock
  // CHECK-NEXT:    %2 = firrtl.stdIntCast %1 : (!firrtl.clock) -> i1
  // CHECK-NEXT:    rtl.output %2 : i1
  firrtl.module @Analog(%a1: !firrtl.analog<1>,
                        %outClock: !firrtl.flip<clock>) {

    %clock = firrtl.asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    firrtl.connect %outClock, %clock : !firrtl.flip<clock>, !firrtl.clock
  }

  // Issue #373: https://github.com/llvm/circt/issues/373
  // CHECK-LABEL: rtl.module @instance_ooo
  firrtl.module @instance_ooo(%arg0: !firrtl.uint<2>, %arg1: !firrtl.uint<2>,
                              %out0: !firrtl.flip<uint<8>>) {
    // The add and eq get hoisted.
    // CHECK: firrtl.add
    // CHECK-NEXT: [[ARG:%.+]] = firrtl.eq
    // CHECK-NEXT: [[ARGC:%.+]] = firrtl.stdIntCast [[ARG]] : (!firrtl.uint<1>) -> i1
    // CHECK-NEXT: rtl.instance "myext" @MyParameterizedExtModule([[ARGC]])
    %myext:2 = firrtl.instance @MyParameterizedExtModule {name = "myext", portNames=["in", "out"]}
      : !firrtl.flip<uint<1>>, !firrtl.uint<8>

    // Calculation of input (the firrtl.add + firrtl.eq) happens after the
    // instance.
    %0 = firrtl.add %arg0, %arg0 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<3>

    // Multiple uses of the add.
    %a = firrtl.eq %0, %0 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<1>
    firrtl.connect %myext#0, %a : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    firrtl.connect %out0, %myext#1 : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    // Casts for the output.
    // CHECK-NEXT: %5 = firrtl.stdIntCast %myext.out : (i8) -> !firrtl.uint<8>
    // CHECK-NEXT: %6 = firrtl.stdIntCast %5 : (!firrtl.uint<8>) -> i8
    // CHECK-NEXT: rtl.output %6
  }

  // CHECK-LABEL: rtl.module @instance_cyclic
  firrtl.module @instance_cyclic(%arg0: !firrtl.uint<2>, %arg1: !firrtl.uint<2>) {
    // This can't be hoisted so we end up with a wire.
    // CHECK: %.in.wire = firrtl.wire : !firrtl.uint<1>
    // CHECK: rtl.instance
    %myext:2 = firrtl.instance @MyParameterizedExtModule {name = "myext", portNames=["in", "out"]}
      : !firrtl.flip<uint<1>>, !firrtl.uint<8>

    // Output of the instance is fed into the input!
    %11 = firrtl.bits %myext#1 2 to 2 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    firrtl.connect %myext#0, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    // CHECK: firrtl.bits
    // CHECK: firrtl.connect
  }

  // CHECK-LABEL: rtl.module @ZeroWidthPorts(
  // CHECK: %inA: i4) -> (%outa: i4) {
  firrtl.module @ZeroWidthPorts(%inA: !firrtl.uint<4>,
                                %inB: !firrtl.uint<0>,
                                %inC: !firrtl.analog<0>,
                                %outa: !firrtl.flip<uint<4>>,
                                %outb: !firrtl.flip<uint<0>>) {
    // CHECK-NEXT: %0 = firrtl.stdIntCast %inA : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %.inB.0width_input = firrtl.wire : !firrtl.flip<uint<0>>
    // CHECK-NEXT: %1 = firrtl.asPassive %.inB.0width_input : !firrtl.flip<uint<0>>
    // CHECK-NEXT: %.inC.output = firrtl.wire : !firrtl.analog<0>
    // CHECK-NEXT: %.outb.output = firrtl.wire : !firrtl.flip<uint<0>>

    // CHECK: [[OUTA:%.+]] = firrtl.mul %0, %1 : (!firrtl.uint<4>, !firrtl.uint<0>) -> !firrtl.uint<4>
    %0 = firrtl.mul %inA, %inB : (!firrtl.uint<4>, !firrtl.uint<0>) -> !firrtl.uint<4>
    firrtl.connect %outa, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    %1 = firrtl.mul %inB, %inB : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
    firrtl.connect %outb, %1 : !firrtl.flip<uint<0>>, !firrtl.uint<0>

    firrtl.attach %inC, %inC : !firrtl.analog<0>, !firrtl.analog<0>

    // CHECK: [[OUTAC:%.+]] = firrtl.stdIntCast [[OUTA]] : (!firrtl.uint<4>) -> i4
    // CHECK-NEXT: rtl.output [[OUTAC]] : i4
  }

  // CHECK-LABEL: rtl.module @ZeroWidthInstance
  firrtl.module @ZeroWidthInstance(%iA: !firrtl.uint<4>,
                                   %iB: !firrtl.uint<0>,
                                   %iC: !firrtl.analog<0>,
                                   %oA: !firrtl.flip<uint<4>>,
                                   %oB: !firrtl.flip<uint<0>>) {

    // CHECK: %myinst.outa = rtl.instance "myinst" @ZeroWidthPorts(%iA) : (i4) -> i4
    // CHECK: [[OUTA:%.+]] = firrtl.stdIntCast %myinst.outa : (i4) -> !firrtl.uint<4>
    %myinst:5 = firrtl.instance @ZeroWidthPorts {name = "myinst", portNames=["inA", "inB", "inC", "outa", "outb"]}
      : !firrtl.flip<uint<4>>, !firrtl.flip<uint<0>>, !firrtl.analog<0>, !firrtl.uint<4>, !firrtl.uint<0>

    // Output of the instance is fed into the input!
    firrtl.connect %myinst#0, %iA : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    firrtl.connect %myinst#1, %iB : !firrtl.flip<uint<0>>, !firrtl.uint<0>
    firrtl.attach %myinst#2, %iC : !firrtl.analog<0>, !firrtl.analog<0>
    firrtl.connect %oA, %myinst#3 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    firrtl.connect %oB, %myinst#4 : !firrtl.flip<uint<0>>, !firrtl.uint<0>

    // CHECK: [[OUTAC:%.+]] = firrtl.stdIntCast [[OUTA]] : (!firrtl.uint<4>) -> i4
    // CHECK: rtl.output [[OUTAC]] : i4
  }

  // CHECK-LABEL: rtl.module @SimpleStruct(%source: !rtl.struct<valid: i1, ready: i1, data: i64>) -> (%sink: !rtl.struct<valid: i1, ready: i1, data: i64>) {
  // CHECK-NEXT:  [[st1:%.+]] = firrtl.rtlStructCast %source : (!rtl.struct<valid: i1, ready: i1, data: i64>) -> !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
  // CHECK-NEXT:  [[st2:%.+]] = firrtl.rtlStructCast [[st1]] : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) -> !rtl.struct<valid: i1, ready: i1, data: i64>
  // CHECK-NEXT:  rtl.output [[st2]] : !rtl.struct<valid: i1, ready: i1, data: i64>
  // CHECK-NEXT: }
  firrtl.module @SimpleStruct(%source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              %sink: !firrtl.flip<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>) {
    firrtl.connect %sink, %source : !firrtl.flip<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
  }

}
