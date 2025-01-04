// RUN: circt-opt %s | circt-opt | FileCheck %s
// Basic MLIR operation parser round-tripping

firrtl.circuit "Basic" attributes {
  // CHECK: firrtl.specialization_disable = #firrtl<layerspecialization disable>
  firrtl.specialization_disable = #firrtl<layerspecialization disable>,
  // CHECK: firrtl.specialization_enable = #firrtl<layerspecialization enable>
  firrtl.specialization_enable = #firrtl<layerspecialization enable>
  } {
firrtl.extmodule @Basic()

// CHECK: firrtl.module @Top(in %arg0: !firrtl.uint<1>) attributes {portNames = [""]}
firrtl.module @Top(in %arg0: !firrtl.uint<1>) attributes {portNames = [""]} {}

// CHECK-LABEL: firrtl.module @Intrinsics
firrtl.module @Intrinsics(in %ui : !firrtl.uint, in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
  // CHECK-NEXT: firrtl.int.sizeof %ui : (!firrtl.uint) -> !firrtl.uint<32>
  %size = firrtl.int.sizeof %ui : (!firrtl.uint) -> !firrtl.uint<32>

  // CHECK-NEXT: firrtl.int.isX %ui : !firrtl.uint
  %isx = firrtl.int.isX %ui : !firrtl.uint

  // CHECK-NEXT: firrtl.int.plusargs.test "foo"
  // CHECK-NEXT: firrtl.int.plusargs.value "bar" : !firrtl.uint<5>
  %foo_found = firrtl.int.plusargs.test "foo"
  %bar_found, %bar_value = firrtl.int.plusargs.value "bar" : !firrtl.uint<5>

  // CHECK-NEXT: firrtl.int.clock_gate %clock, %ui1
  // CHECK-NEXT: firrtl.int.clock_gate %clock, %ui1, %ui1
  %cg0 = firrtl.int.clock_gate %clock, %ui1
  %cg1 = firrtl.int.clock_gate %clock, %ui1, %ui1

  // CHECK-NEXT: firrtl.int.generic "clock_gate" %clock, %ui1 : (!firrtl.clock, !firrtl.uint<1>)
  // CHECK-NEXT: firrtl.int.generic "noargs" : () -> !firrtl.uint<32>
  // CHECK-NEXT: firrtl.int.generic "params" <FORMAT: none = "foobar"> : () -> !firrtl.bundle<x: uint<1>>
  // CHECK-NEXT: firrtl.int.generic "params_and_operand" <X: i64 = 123> %ui1 : (!firrtl.uint<1>) -> !firrtl.clock
  // CHECK-NEXT: firrtl.int.generic "inputs" %clock, %ui1, %clock : (!firrtl.clock, !firrtl.uint<1>, !firrtl.clock) -> ()
  %cg2 = firrtl.int.generic "clock_gate" %clock, %ui1 : (!firrtl.clock, !firrtl.uint<1>) -> !firrtl.clock
  %noargs = firrtl.int.generic "noargs" : () -> !firrtl.uint<32>
  %p = firrtl.int.generic "params" <FORMAT: none = "foobar"> : () -> !firrtl.bundle<x: uint<1>>
  %po = firrtl.int.generic "params_and_operand" <X: i64 = 123> %ui1 : (!firrtl.uint<1>) -> !firrtl.clock
  firrtl.int.generic "inputs" %clock, %ui1, %clock : (!firrtl.clock, !firrtl.uint<1>, !firrtl.clock) -> ()

  %probe = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK: firrtl.view "View"
  // CHECK-SAME: <{
  // CHECK-SAME:     elements = [
  // CHECK-SAME:       {
  // CHECK-SAME:         class = "sifive.enterprise.grandcentral.AugmentedGroundType",
  // CHECK-SAME:         id = 0 : i64,
  // CHECK-SAME:         name = "baz"
  // CHECK-SAME:       },
  // CHECK-SAME:       {
  // CHECK-SAME:         class = "sifive.enterprise.grandcentral.AugmentedGroundType",
  // CHECK-SAME:         id = 0 : i64,
  // CHECK-SAME:         name = "qux"
  // CHECK-SAME:       }
  // CHECK-SAME:     ]
  // CHECK-SAME: }>, %probe, %probe : !firrtl.probe<uint<1>>, !firrtl.probe<uint<1>>
  firrtl.view "View", <{
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    defName = "Bar",
    elements = [
      {
        class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 0 : i64,
        name = "baz"
      },
      {
        class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 0 : i64,
        name = "qux"
      }
    ]
  }>, %probe, %probe : !firrtl.probe<uint<1>>, !firrtl.probe<uint<1>>
}

// CHECK-LABEL: firrtl.module @FPGAProbe
firrtl.module @FPGAProbe(
  in %clock: !firrtl.clock,
  in %reset: !firrtl.uint<1>,
  in %in: !firrtl.uint<8>
) {
  // CHECK: firrtl.int.fpga_probe %clock, %in : !firrtl.uint<8>
  firrtl.int.fpga_probe %clock, %in : !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.option @Platform
firrtl.option @Platform {
  // CHECK:firrtl.option_case @FPGA
  firrtl.option_case @FPGA
  // CHECK:firrtl.option_case @ASIC
  firrtl.option_case @ASIC
}

firrtl.module private @DefaultTarget(in %clock: !firrtl.clock) {}
firrtl.module private @FPGATarget(in %clock: !firrtl.clock) {}
firrtl.module private @ASICTarget(in %clock: !firrtl.clock) {}

// CHECK-LABEL: firrtl.module @Foo
firrtl.module @Foo(in %clock: !firrtl.clock) {
  // CHECK: %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTarget alternatives @Platform
  // CHECK-SAME: { @FPGA -> @FPGATarget, @ASIC -> @ASICTarget } (in clock: !firrtl.clock)
  %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTarget alternatives @Platform
    { @FPGA -> @FPGATarget, @ASIC -> @ASICTarget } (in clock: !firrtl.clock)
  firrtl.matchingconnect %inst_clock, %clock : !firrtl.clock
}

// CHECK-LABEL: firrtl.layer @LayerA bind
// CHECK-NEXT:    firrtl.layer @LayerB inline
firrtl.layer @LayerA bind {
  firrtl.layer @LayerB inline {}
}

// CHECK-LABEL: firrtl.module @Layers
// CHECK-SAME:    out %a: !firrtl.probe<uint<1>, @LayerA>
// CHECK-SAME:    out %b: !firrtl.rwprobe<uint<1>, @LayerA::@LayerB>
firrtl.module @Layers(
  out %a: !firrtl.probe<uint<1>, @LayerA>,
  out %b: !firrtl.rwprobe<uint<1>, @LayerA::@LayerB>
) {}

// CHECK-LABEL: firrtl.module @LayersEnabled
// CHECK-SAME:    layers = [@LayerA]
firrtl.module @LayersEnabled() attributes {layers = [@LayerA]} {
}

// CHECK-LABEL: firrtl.module @PropertyArithmetic
firrtl.module @PropertyArithmetic() {
  %0 = firrtl.integer 1
  %1 = firrtl.integer 2

  // CHECK: firrtl.integer.add %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %2 = firrtl.integer.add %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

  // CHECK: firrtl.integer.mul %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %3 = firrtl.integer.mul %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

  // CHECK: firrtl.integer.shr %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %4 = firrtl.integer.shr %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

  // CHECK: firrtl.integer.shl %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %5 = firrtl.integer.shl %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
}

// CHECK-LABEL: firrtl.module @PropertyListOps
firrtl.module @PropertyListOps() {
  %0 = firrtl.integer 0
  %1 = firrtl.integer 1
  %2 = firrtl.integer 2

  // CHECK: [[L0:%.+]] = firrtl.list.create %0, %1
  %l0 = firrtl.list.create %0, %1 : !firrtl.list<integer>

  // CHECK: [[L1:%.+]] = firrtl.list.create %2
  %l1 = firrtl.list.create %2 : !firrtl.list<integer>

  // CHECK: firrtl.list.concat [[L0]], [[L1]] : !firrtl.list<integer>
  %concat = firrtl.list.concat %l0, %l1 : !firrtl.list<integer>
}

// CHECK: firrtl.formal @myTestA, @Top {}
firrtl.formal @myTestA, @Top {}
// CHECK: firrtl.formal @myTestB, @Top {bound = 42 : i19}
firrtl.formal @myTestB, @Top {bound = 42 : i19}
// CHECK: firrtl.formal @myTestC, @Top {} attributes {foo}
firrtl.formal @myTestC, @Top {} attributes {foo}

}
