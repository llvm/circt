// RUN: circt-opt -lower-firrtl-to-rtl %s -verify-diagnostics | FileCheck %s

// The firrtl.circuit should be removed, the main module name moved to an
// attribute on the module.
// CHECK-LABEL: {{^}}module attributes {firrtl.mainModule = "Simple"} {
// CHECK-NOT: firrtl.circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef.procedural "PRINTF_COND" {
// CHECK-NEXT:   sv.verbatim "`define PRINTF_COND_ (`PRINTF_COND)"
// CHECK-NEXT:  } else  {
firrtl.circuit "Simple" {

   // CHECK-LABEL: rtl.module.extern @MyParameterizedExtModule(%in: i1) -> (%out: i8)
   // CHECK: attributes {verilogName = "name_thing"}
   firrtl.extmodule @MyParameterizedExtModule(!firrtl.uint<1> , !firrtl.flip<uint<8>> )
      attributes {defname = "name_thing",
                  parameters = {DEFAULT = 0 : i64,
                                DEPTH = 3.242000e+01 : f64,
                                FORMAT = "xyz_timeout=%d\0A",
                                WIDTH = 32 : i8},
                  portNames = ["in", "out"]
                                }

   // CHECK-LABEL: rtl.module @Simple(%in1: i4, %in2: i2, %in3: i8) -> (%out4: i4) {
   firrtl.module @Simple(%in1: !firrtl.uint<4>,
                        %in2: !firrtl.uint<2>,
                        %in3: !firrtl.sint<8>,
                        %out4: !firrtl.flip<uint<4>>) {

    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.concat %false, %in1
    // CHECK: comb.concat %false, %in1 

    // CHECK: comb.sub
    %2 = firrtl.sub %1, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // CHECK: %3 = comb.concat %false, %in2 : (i1, i2) -> i3
    %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.uint<3>
    // CHECK: comb.concat %false, %3 : (i1, i3) -> i4
    %4 = firrtl.pad %3, 4 : (!firrtl.uint<3>) -> !firrtl.uint<4>
    // CHECK: [[RESULT:%.+]] = comb.xor
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    firrtl.connect %out4, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    // CHECK-NEXT: rtl.output [[RESULT]] : i4
  }

  // CHECK-LABEL: rtl.module @TestInstance(
  firrtl.module @TestInstance(%u2: !firrtl.uint<2>, %s8: !firrtl.sint<8>,
                              %clock: !firrtl.clock,
                              %reset: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_i2 = rtl.constant
    // CHECK-NEXT: %xyz.out4 = rtl.instance "xyz" @Simple([[ARG1:%.+]], %u2, %s8) : (i4, i2, i8) -> i4
    %xyz:4 = firrtl.instance @Simple {name = "xyz", portNames=["in1", "in2", "in3", "out4"]}
     : !firrtl.flip<uint<4>>, !firrtl.flip<uint<2>>, !firrtl.flip<sint<8>>, !firrtl.uint<4>

    // CHECK: [[ARG1]] = comb.concat %c0_i2, %u2 : (i2, i2) -> i4
    firrtl.connect %xyz#0, %u2 : !firrtl.flip<uint<4>>, !firrtl.uint<2>

    // CHECK-NOT: rtl.connect
    firrtl.connect %xyz#1, %u2 : !firrtl.flip<uint<2>>, !firrtl.uint<2>

    firrtl.connect %xyz#2, %s8 : !firrtl.flip<sint<8>>, !firrtl.sint<8>

    // CHECK: sv.fwrite "%x"(%xyz.out4) : i4
    firrtl.printf %clock, %reset, "%x"(%xyz#3) : !firrtl.uint<4>
 
    // CHECK: sv.fwrite "Something interesting! %x"(%myext.out) : i8

    // Parameterized module reference.
    // rtl.instance carries the parameters, unlike at the FIRRTL layer.

    // CHECK: %myext.out = rtl.instance "myext" @MyParameterizedExtModule(%reset)  {parameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}} : (i1) -> i8
    %myext:2 = firrtl.instance @MyParameterizedExtModule {name = "myext", portNames=["in", "out"]}
      : !firrtl.flip<uint<1>>, !firrtl.uint<8>

    firrtl.connect %myext#0, %reset : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    firrtl.printf %clock, %reset, "Something interesting! %x"(%myext#1) : !firrtl.uint<8>
  }

  // CHECK-LABEL: rtl.module @OutputFirst(%in1: i1, %in4: i4) -> (%out4: i4) {
  firrtl.module @OutputFirst(%out4: !firrtl.flip<uint<4>>,
                             %in1: !firrtl.uint<1>,
                             %in4: !firrtl.uint<4>) {
    firrtl.connect %out4, %in4 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // CHECK-NEXT: rtl.output %in4 : i4
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
    // CHECK: %0 = firrtl.stdIntCast %inA : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %1 = firrtl.stdIntCast %inB : (i4) -> !firrtl.uint<4>
    // CHECK-NEXT: %2 = firrtl.stdIntCast %inC : (i4) -> !firrtl.uint<4>

    // CHECK: [[OUTC:%.+]] = firrtl.wire {{.*}} : !firrtl.flip<uint<4>>
    // CHECK: [[OUTD:%.+]] = firrtl.wire {{.*}} : !firrtl.flip<uint<4>>

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

    // expected-error @+2 {{'firrtl.connect' op LowerToRTL couldn't handle this operation}}
    // expected-error @+1 {{destination isn't an inout type}}
    firrtl.connect %tmp2, %inA : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    // No connections to outD.

    firrtl.connect %outE, %inE : !firrtl.flip<uint<4>>, !firrtl.uint<3>

    // CHECK: [[OUTBY:%.+]] = comb.merge %inB, %inA : i4
    // CHECK: [[OUTCR:%.+]] = sv.read_inout %.outC.output
    // CHECK: [[OUTDR:%.+]] = sv.read_inout %.outD.output

    // Extension for outE
    // CHECK: [[OUTE:%.+]] = comb.concat %false, %inE : (i1, i3) -> i4
    // CHECK: rtl.output %inA, [[OUTBY]], [[OUTCR]], [[OUTDR]], [[OUTE]]
  }

  // CHECK-LABEL: rtl.module @Analog(%a1: !rtl.inout<i1>) -> (%outClock: i1) {
  // CHECK-NEXT:    %0 = sv.read_inout %a1 : !rtl.inout<i1>
  // CHECK-NEXT:    rtl.output %0 : i1
  firrtl.module @Analog(%a1: !firrtl.analog<1>,
                        %outClock: !firrtl.flip<clock>) {

    %clock = firrtl.asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    firrtl.connect %outClock, %clock : !firrtl.flip<clock>, !firrtl.clock
  }

  // Issue #373: https://github.com/llvm/circt/issues/373
  // CHECK-LABEL: rtl.module @instance_ooo
  firrtl.module @instance_ooo(%arg0: !firrtl.uint<2>, %arg1: !firrtl.uint<2>,
                              %arg2: !firrtl.uint<3>,
                              %out0: !firrtl.flip<uint<8>>) {
    // CHECK: %false = rtl.constant false

    // CHECK-NEXT: rtl.instance "myext" @MyParameterizedExtModule([[ARG:%.+]]) {parameters
    %myext:2 = firrtl.instance @MyParameterizedExtModule {name = "myext", portNames=["in", "out"]}
      : !firrtl.flip<uint<1>>, !firrtl.uint<8>

    // CHECK: [[ADD:%.+]] = comb.add %0, %1

    // Calculation of input (the firrtl.add + firrtl.eq) happens after the
    // instance.
    %0 = firrtl.add %arg0, %arg0 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<3>

    // Multiple uses of the add.
    %a = firrtl.eq %0, %arg2 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<1>
    // CHECK-NEXT: [[ARG]] = comb.icmp eq [[ADD]], %arg2 : i3
    firrtl.connect %myext#0, %a : !firrtl.flip<uint<1>>, !firrtl.uint<1>

    firrtl.connect %out0, %myext#1 : !firrtl.flip<uint<8>>, !firrtl.uint<8>

    // CHECK-NEXT: rtl.output %myext.out
  }

  // CHECK-LABEL: rtl.module @instance_cyclic
  firrtl.module @instance_cyclic(%arg0: !firrtl.uint<2>, %arg1: !firrtl.uint<2>) {
    // CHECK: %myext.out = rtl.instance "myext" @MyParameterizedExtModule(%0)
    %myext:2 = firrtl.instance @MyParameterizedExtModule {name = "myext", portNames=["in", "out"]}
      : !firrtl.flip<uint<1>>, !firrtl.uint<8>

    // Output of the instance is fed into the input!
    %11 = firrtl.bits %myext#1 2 to 2 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    // CHECK: %0 = comb.extract %myext.out from 2 : (i8) -> i1

    firrtl.connect %myext#0, %11 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  }

  // CHECK-LABEL: rtl.module @ZeroWidthPorts(%inA: i4) -> (%outa: i4) {
  firrtl.module @ZeroWidthPorts(%inA: !firrtl.uint<4>,
                                %inB: !firrtl.uint<0>,
                                %inC: !firrtl.analog<0>,
                                %outa: !firrtl.flip<uint<4>>,
                                %outb: !firrtl.flip<uint<0>>) {
     %0 = firrtl.mul %inA, %inB : (!firrtl.uint<4>, !firrtl.uint<0>) -> !firrtl.uint<4>
    firrtl.connect %outa, %0 : !firrtl.flip<uint<4>>, !firrtl.uint<4>

    %1 = firrtl.mul %inB, %inB : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
    firrtl.connect %outb, %1 : !firrtl.flip<uint<0>>, !firrtl.uint<0>

    firrtl.attach %inC, %inC : !firrtl.analog<0>, !firrtl.analog<0>

    // CHECK: [[OUTAC:%.+]] = rtl.constant 0 : i4
    // CHECK-NEXT: rtl.output [[OUTAC]] : i4
  }

  // CHECK-LABEL: rtl.module @ZeroWidthInstance
  firrtl.module @ZeroWidthInstance(%iA: !firrtl.uint<4>,
                                   %iB: !firrtl.uint<0>,
                                   %iC: !firrtl.analog<0>,
                                   %oA: !firrtl.flip<uint<4>>,
                                   %oB: !firrtl.flip<uint<0>>) {

    // CHECK: %myinst.outa = rtl.instance "myinst" @ZeroWidthPorts(%iA) : (i4) -> i4
    %myinst:5 = firrtl.instance @ZeroWidthPorts {name = "myinst", portNames=["inA", "inB", "inC", "outa", "outb"]}
      : !firrtl.flip<uint<4>>, !firrtl.flip<uint<0>>, !firrtl.analog<0>, !firrtl.uint<4>, !firrtl.uint<0>

    // Output of the instance is fed into the input!
    firrtl.connect %myinst#0, %iA : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    firrtl.connect %myinst#1, %iB : !firrtl.flip<uint<0>>, !firrtl.uint<0>
    firrtl.attach %myinst#2, %iC : !firrtl.analog<0>, !firrtl.analog<0>
    firrtl.connect %oA, %myinst#3 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
    firrtl.connect %oB, %myinst#4 : !firrtl.flip<uint<0>>, !firrtl.uint<0>

    // CHECK: rtl.output %myinst.outa
  }

  // CHECK-LABEL: rtl.module @SimpleStruct(%source: !rtl.struct<valid: i1, ready: i1, data: i64>) -> (%sink: !rtl.struct<valid: i1, ready: i1, data: i64>) {
  // CHECK-NEXT:    rtl.output %source : !rtl.struct<valid: i1, ready: i1, data: i64>
  firrtl.module @SimpleStruct(%source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              %sink: !firrtl.flip<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>) {
    firrtl.connect %sink, %source : !firrtl.flip<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
  }

  // https://github.com/llvm/circt/issues/690
  // CHECK-LABEL: rtl.module @bar690(%led_0: !rtl.inout<i1>) {
  firrtl.module @bar690(%led_0: !firrtl.analog<1>) {
  }
  // CHECK-LABEL: rtl.module @foo690()
  firrtl.module @foo690() {
    // CHECK: %.led_0.wire = sv.wire
    // CHECK: rtl.instance "fpga" @bar690(%.led_0.wire) : (!rtl.inout<i1>) -> ()
    %result = firrtl.instance @bar690 {name = "fpga", portNames = ["led_0"]} : !firrtl.analog<1>
  }
  // CHECK-LABEL: rtl.module @foo690a(%a: !rtl.inout<i1>) {
  firrtl.module @foo690a(%a: !firrtl.analog<1>) {
    %result = firrtl.instance @bar690 {name = "fpga", portNames = ["led_0"]} : !firrtl.analog<1>
    firrtl.attach %result, %a: !firrtl.analog<1>, !firrtl.analog<1>
  }

  // https://github.com/llvm/circt/issues/740
  // CHECK-LABEL: rtl.module @foo740(%led_0: !rtl.inout<i1>) {
  // CHECK:  %.led_0.wire = sv.wire
  // CHECK-NEXT:  rtl.instance "fpga" @bar740(%.led_0.wire)
  firrtl.extmodule @bar740(%led_0: !firrtl.analog<1>)
  firrtl.module @foo740(%led_0: !firrtl.analog<1>) {
    %result = firrtl.instance @bar740 {name = "fpga", portNames = ["led_0"]} : !firrtl.analog<1>
    firrtl.attach %result, %led_0 : !firrtl.analog<1>, !firrtl.analog<1>
  }
}
