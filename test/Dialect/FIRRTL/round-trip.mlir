// RUN: circt-opt --verify-roundtrip %s | FileCheck %s
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

  %val = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.view "View"
  // CHECK-SAME: <{
  // CHECK-SAME:     elements = [
  // CHECK-SAME:       {
  // CHECK-SAME:         class = "sifive.enterprise.grandcentral.AugmentedGroundType",
  // CHECK-SAME:         name = "baz"
  // CHECK-SAME:       },
  // CHECK-SAME:       {
  // CHECK-SAME:         class = "sifive.enterprise.grandcentral.AugmentedGroundType",
  // CHECK-SAME:         name = "qux"
  // CHECK-SAME:       }
  // CHECK-SAME:     ]
  // CHECK-SAME: }>, %val, %val : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.view "View", <{
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    defName = "Bar",
    elements = [
      {
        class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        name = "baz"
      },
      {
        class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        name = "qux"
      }
    ]
  }>, %val, %val : !firrtl.uint<1>, !firrtl.uint<1>
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

firrtl.formal @myFormalTestA, @Top {}
firrtl.formal @myFormalTestB, @Top {bound = 42 : i19}
firrtl.formal @myFormalTestC, @Top {} attributes {foo}

firrtl.simulation @mySimulationTestA, @SimulationTop {}
firrtl.simulation @mySimulationTestB, @SimulationTop {bound = 42 : i19}
firrtl.simulation @mySimulationTestC, @SimulationTop {} attributes {foo}

firrtl.extmodule @SimulationTop(
  in clock: !firrtl.clock,
  in init: !firrtl.uint<1>,
  out done: !firrtl.uint<1>,
  out success: !firrtl.uint<1>
)

firrtl.module @Contracts(in %a: !firrtl.uint<42>, in %b: !firrtl.bundle<x: uint<1337>>) {
  firrtl.contract {}
  firrtl.contract %a, %b : !firrtl.uint<42>, !firrtl.bundle<x: uint<1337>> {
  ^bb0(%arg0: !firrtl.uint<42>, %arg1: !firrtl.bundle<x: uint<1337>>):
  }
}

// Format string support
// CHECK-LABEL: firrtl.module @FormatString
firrtl.module @FormatString() {

  // CHECK-NEXT: %time = firrtl.fstring.time : !firrtl.fstring
  %time = firrtl.fstring.time : !firrtl.fstring

}

// CHECK-LABEL: firrtl.module @Fprintf
firrtl.module @Fprintf(
  in %clock : !firrtl.clock,
  in %reset : !firrtl.reset,
  in %a : !firrtl.uint<1>
) {
  // CHECK-NEXT: firrtl.fprintf %clock, %a, "test%d.txt"(%a), "%x, %b"(%a, %reset) {name = "foo"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.reset
  firrtl.fprintf %clock, %a, "test%d.txt"(%a), "%x, %b"(%a, %reset) {name = "foo"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.reset
}

// CHECK-LABEL: firrtl.domain @ClockDomain
firrtl.domain @ClockDomain

// CHECK-LABEL: firrtl.domain @PowerDomain [
// CHECK-SAME:    #firrtl.domain.field<"name", !firrtl.string>
// CHECK-SAME:    #firrtl.domain.field<"voltage", !firrtl.integer>
// CHECK-SAME:    #firrtl.domain.field<"alwaysOn", !firrtl.bool>
// CHECK-SAME:  ]
firrtl.domain @PowerDomain [
  #firrtl.domain.field<"name", !firrtl.string>,
  #firrtl.domain.field<"voltage", !firrtl.integer>,
  #firrtl.domain.field<"alwaysOn", !firrtl.bool>
]

firrtl.module @DomainsSubmodule(
  in %A: !firrtl.domain of @ClockDomain,
  in %a: !firrtl.uint<1> domains [%A]
) {}

// CHECK-LABEL: firrtl.module @Domains(
// CHECK-SAME:    in %A: !firrtl.domain
// CHECK-SAME:    in %B: !firrtl.domain
// CHECK-SAME:    in %a: !firrtl.uint<1> domains [%A]
// CHECK-SAME:    out %b: !firrtl.uint<1> domains [%B]
firrtl.module @Domains(
  in %A: !firrtl.domain of @ClockDomain,
  in %B: !firrtl.domain of @ClockDomain,
  in %a: !firrtl.uint<1> domains [%A],
  out %b: !firrtl.uint<1> domains [%B]
) {
  // CHECK: %0 = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>
  %0 = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>
  firrtl.matchingconnect %b, %0 : !firrtl.uint<1>

  // CHECK:      %foo_A, %foo_a = firrtl.instance foo @DomainsSubmodule(
  // CHECK-SAME:   in A: !firrtl.domain of @ClockDomain
  // CHECK-SAME:   in a: !firrtl.uint<1> domains [A]
  %foo_A, %foo_a = firrtl.instance foo @DomainsSubmodule(
    in A: !firrtl.domain of @ClockDomain,
    in a: !firrtl.uint<1> domains [A]
  )
}

// CHECK-LABEL: firrtl.module @AnonymousDomains
// CHECK-SAME:    in %arg0: !firrtl.domain
// CHECK-SAME:    in %a: !firrtl.uint<1> domains [%arg0]
// CHECK-SAME:    portNames = ["", "a"]
firrtl.module @AnonymousDomains(
  in %arg0: !firrtl.domain of @ClockDomain,
  in %a: !firrtl.uint<1> domains [%arg0]
) attributes {
  portNames = ["", "a"]
} {
  // CHECK: %0 = firrtl.unsafe_domain_cast %a domains %arg0 : !firrtl.uint<1>
  %0 = firrtl.unsafe_domain_cast %a domains %arg0 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @DomainDefine
firrtl.module @DomainDefine(
  in  %x : !firrtl.domain of @ClockDomain,
  out %y : !firrtl.domain of @ClockDomain
) {
  // CHECK: firrtl.domain.define %y, %x
  firrtl.domain.define %y, %x
}

firrtl.module private @DefaultTargetWithDomain(in %A: !firrtl.domain of @ClockDomain, in %clock: !firrtl.clock domains [%A]) {}
firrtl.module private @FPGATargetWithDomain(in %A: !firrtl.domain of @ClockDomain, in %clock: !firrtl.clock domains [%A]) {}
firrtl.module private @ASICTargetWithDomain(in %A: !firrtl.domain of @ClockDomain, in %clock: !firrtl.clock domains [%A]) {}

// CHECK-LABEL: firrtl.module @InstanceChoiceWithDomain
firrtl.module @InstanceChoiceWithDomain(in %A: !firrtl.domain of @ClockDomain, in %clock: !firrtl.clock domains [%A]) {
  // CHECK:      %inst_A, %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTargetWithDomain alternatives @Platform
  // CHECK-SAME:   { @FPGA -> @FPGATargetWithDomain, @ASIC -> @ASICTargetWithDomain } (in A: !firrtl.domain of @ClockDomain, in clock: !firrtl.clock domains [A])
  %inst_A, %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTargetWithDomain alternatives @Platform
    { @FPGA -> @FPGATargetWithDomain, @ASIC -> @ASICTargetWithDomain } (in A: !firrtl.domain of @ClockDomain, in clock: !firrtl.clock domains [A])
  firrtl.matchingconnect %inst_clock, %clock : !firrtl.clock
  firrtl.domain.define %inst_A, %A
}

}
