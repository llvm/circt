// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intmodules))' --split-input-file %s | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: "BasicIntmoduleInstances"
firrtl.circuit "BasicIntmoduleInstances" {
  // CHECK-NOT: NameDoesNotMatter5
  firrtl.intmodule @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {intrinsic = "circt.sizeof"}
  // CHECK-NOT: NameDoesNotMatter6
  firrtl.intmodule @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.isX"}
  // CHECK-NOT: NameDoesNotMatter7
  firrtl.intmodule @NameDoesNotMatter7<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.plusargs.test"}
  // CHECK-NOT: NameDoesNotMatter8
  firrtl.intmodule @NameDoesNotMatter8<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {intrinsic = "circt.plusargs.value"}

  // CHECK: @BasicIntmoduleInstances
  firrtl.module @BasicIntmoduleInstances(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter5
    // CHECK: firrtl.int.generic "circt.sizeof"
    firrtl.matchingconnect %i1, %clk : !firrtl.clock
    firrtl.matchingconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = firrtl.instance "" @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter6
    // CHECK: firrtl.int.generic "circt.isX"
    firrtl.matchingconnect %i2, %clk : !firrtl.clock
    firrtl.matchingconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = firrtl.instance "" @NameDoesNotMatter7(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter7
    // CHECK: firrtl.int.generic "circt.plusargs.test"
    firrtl.matchingconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = firrtl.instance "" @NameDoesNotMatter8(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter8
    // CHECK: firrtl.int.generic "circt.plusargs.value" <FORMAT: none = "foo"> : () -> !firrtl.bundle<found: uint<1>, result: uint<5>>
    firrtl.matchingconnect %io3, %found4 : !firrtl.uint<1>
    firrtl.matchingconnect %io4, %result1 : !firrtl.uint<5>
  }
}

// -----

// CHECK-LABEL: "ProbeIntrinsicTest"
firrtl.circuit "ProbeIntrinsicTest" {
  // CHECK-NOT: firrtl.intmodule private @FPGAProbeIntrinsic
  firrtl.intmodule private @FPGAProbeIntrinsic(in data: !firrtl.uint, in clock: !firrtl.clock) attributes {intrinsic = "circt_fpga_probe"}

  // CHECK-LABEL: firrtl.module private @ProbeIntrinsicTest
  firrtl.module private @ProbeIntrinsicTest(in %clock : !firrtl.clock, in %data : !firrtl.uint<32>) {
    // CHECK:      [[DATA:%.+]] = firrtl.wire : !firrtl.uint
    // CHECK-NEXT: [[CLOCK:%.+]] = firrtl.wire : !firrtl.clock
    // CHECK-NEXT: firrtl.int.generic "circt_fpga_probe" [[DATA]], [[CLOCK]] : (!firrtl.uint, !firrtl.clock) -> ()
    // CHECK-NEXT: firrtl.matchingconnect [[CLOCK]], %clock : !firrtl.clock
    // CHECK-NEXT: firrtl.connect [[DATA]], %data : !firrtl.uint, !firrtl.uint<32>
    %mod_data, %mod_clock = firrtl.instance mod @FPGAProbeIntrinsic(in data: !firrtl.uint, in clock: !firrtl.clock)
    firrtl.matchingconnect %mod_clock, %clock : !firrtl.clock
    firrtl.connect %mod_data, %data : !firrtl.uint, !firrtl.uint<32>
  }
}
