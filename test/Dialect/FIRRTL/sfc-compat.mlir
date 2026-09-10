// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-sfc-compat)))' --verify-diagnostics --split-input-file %s | FileCheck %s

firrtl.circuit "SFCCompatTests" {

  firrtl.module @SFCCompatTests() {}

  // An invalidated regreset should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidValue
  firrtl.module @InvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    // CHECK-NOT: invalid
    %invalid_ui1_dead = firrtl.invalidvalue : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %invalid_ui1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through a wire should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidThroughWire
  firrtl.module @InvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %inv  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through wires with aggregate types should be
  // converted to a reg.
  //
  // CHECK-LABEL: firrtl.module @AggregateInvalidThroughWire
  firrtl.module @AggregateInvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.vector<bundle<a: uint<1>>, 2>, out %q: !firrtl.vector<bundle<a: uint<1>>, 2>, in %foo: !firrtl.vector<uint<1>, 1>) {
    %inv = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %inv_a = firrtl.subfield %inv[a] : !firrtl.bundle<a: uint<1>>
    %invalid = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.matchingconnect %inv_a, %invalid : !firrtl.uint<1>

    %inv1 = firrtl.wire : !firrtl.vector<bundle<a: uint<1>>, 2>
    %inv1_0 = firrtl.subindex %inv1[0] : !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.matchingconnect %inv1_0, %inv : !firrtl.bundle<a: uint<1>>
    %inv1_1 = firrtl.subindex %inv1[0] : !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.matchingconnect %inv1_1, %inv : !firrtl.bundle<a: uint<1>>

    // CHECK: firrtl.reg %clock : !firrtl.clock, !firrtl.vector<bundle<a: uint<1>>, 2>
    %r = firrtl.regreset %clock, %reset, %inv1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<bundle<a: uint<1>>, 2>, !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.matchingconnect %r, %d : !firrtl.vector<bundle<a: uint<1>>, 2>
    firrtl.matchingconnect %q, %r : !firrtl.vector<bundle<a: uint<1>>, 2>

    %bar = firrtl.wire : !firrtl.vector<vector<uint<1>, 1>, 1>
    %1 = firrtl.subindex %bar[0] : !firrtl.vector<vector<uint<1>, 1>, 1>
    %2 = firrtl.subindex %foo[0] : !firrtl.vector<uint<1>, 1>
    %3 = firrtl.subindex %1[0] : !firrtl.vector<uint<1>, 1>
    firrtl.matchingconnect %3, %2 : !firrtl.uint<1>
    // Check that firrtl.regreset is not transformed into reg op if wire is not invalid
    // CHECK: firrtl.regreset
    %x = firrtl.regreset %clock, %reset, %bar : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<vector<uint<1>, 1>, 1>, !firrtl.vector<vector<uint<1>, 1>, 1>
    %4 = firrtl.subindex %x[0] : !firrtl.vector<vector<uint<1>, 1>, 1>
    %5 = firrtl.subindex %4[0] : !firrtl.vector<uint<1>, 1>
    firrtl.matchingconnect %5, %5 : !firrtl.uint<1>
  }

  // A regreset invalidated via an output port should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidPort
  firrtl.module @InvalidPort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>, out %x: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %x  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidate via an instance input port should be converted to a
  // reg.
  //
  // CHECK-LABEL: @InvalidInstancePort
  firrtl.module @InvalidInstancePort_Submodule(in %inv: !firrtl.uint<1>) {}
  firrtl.module @InvalidInstancePort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %submodule_inv = firrtl.instance submodule  @InvalidInstancePort_Submodule(in inv: !firrtl.uint<1>)
    firrtl.connect %submodule_inv, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %submodule_inv  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A primitive operation should block invalid propagation.
  firrtl.module @InvalidPrimop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %0 = firrtl.not %invalid_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: firrtl.regreset %clock
    %r = firrtl.regreset %clock, %reset, %0  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalid value should propagate through a node.
  // Change from SFC behavior.
  firrtl.module @InvalidNode(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    %inv = firrtl.wire  : !firrtl.uint<8>
    %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    firrtl.connect %inv, %invalid_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
    %_T = firrtl.node %inv  : !firrtl.uint<8>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %_T  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %r, %d : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %q, %r : !firrtl.uint<8>, !firrtl.uint<8>
  }


  // Check that we properly lower aggregate invalids.
  firrtl.module @AggregateInvalid() {
    %bundle = firrtl.wire : !firrtl.bundle<a:uint<1>>
    %invalid_bundle = firrtl.invalidvalue : !firrtl.bundle<a:uint<1>>
    firrtl.connect %bundle, %invalid_bundle : !firrtl.bundle<a:uint<1>>, !firrtl.bundle<a:uint<1>>
    // CHECK: [[CAST:%.+]] = firrtl.bitcast %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>>
    // CHECK: connect %bundle, [[CAST]]

    %vector = firrtl.wire : !firrtl.vector<uint<1>, 10>
    %invalid_vector = firrtl.invalidvalue : !firrtl.vector<uint<1>, 10>
    firrtl.connect %vector, %invalid_vector : !firrtl.vector<uint<1>, 10>, !firrtl.vector<uint<1>, 10>
    // CHECK: [[CAST:%.+]] = firrtl.bitcast %c0_ui10 : (!firrtl.uint<10>) -> !firrtl.vector<uint<1>, 10>
    // CHECK: connect %vector, [[CAST]]

    %enum = firrtl.wire : !firrtl.enum<a: uint<1>, b: uint<1>>
    %invalid_enum = firrtl.invalidvalue : !firrtl.enum<a: uint<1>, b: uint<1>>
    firrtl.connect %enum, %invalid_enum : !firrtl.enum<a: uint<1>, b: uint<1>>, !firrtl.enum<a: uint<1>, b: uint<1>>
    // CHECK: [[CAST:%.+]] = firrtl.bitcast %c0_ui2 : (!firrtl.uint<2>) -> !firrtl.enum<a: uint<1>, b: uint<1>>
    // CHECK: connect %enum, [[CAST]]
  }

  // All of these should not error as the register is initialzed to a constant
  // reset value while looking through constructs that the SFC allows.  This is
  // testing the following cases:
  //
  //   1. A wire marked don't touch driven to a constant.
  //   2. A node driven to a constant.
  //   3. A wire driven to an invalid.
  //   4. A constant that passes through SFC-approved primops.
  //
  // CHECK-LABEL: firrtl.module @ConstantAsyncReset
  firrtl.module @ConstantAsyncReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %r0_init = firrtl.wire sym @r0_init : !firrtl.uint<1>
    firrtl.matchingconnect %r0_init, %c0_ui1 : !firrtl.uint<1>
    %r0 = firrtl.regreset %clock, %reset, %r0_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %r1_init = firrtl.node %c0_ui1 : !firrtl.uint<1>
    %r1 = firrtl.regreset %clock, %reset, %r1_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %inv_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %r2_init = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %r2_init, %inv_ui1 : !firrtl.uint<1>
    %r2 = firrtl.regreset %clock, %reset, %r2_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %c0_si1 = firrtl.asSInt %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.sint<1>
    %c0_clock = firrtl.asClock %c0_si1 : (!firrtl.sint<1>) -> !firrtl.clock
    %c0_asyncreset = firrtl.asAsyncReset %c0_clock : (!firrtl.clock) -> !firrtl.asyncreset
    %r3_init = firrtl.asUInt %c0_asyncreset : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %r3 = firrtl.regreset %clock, %reset, %r3_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %agg_const = firrtl.aggregateconstant [1, 2, 1] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<1>>
    %subfield = firrtl.subfield %agg_const[c] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<1>>
    %r4 = firrtl.regreset %clock, %reset, %subfield : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @TailPrimOp
  firrtl.module @TailPrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.pad %c0_ui1, 3 : (!firrtl.uint<1>) -> !firrtl.uint<3>
    %1 = firrtl.tail %0, 2 : (!firrtl.uint<3>) -> !firrtl.uint<1>
    %r0_init = firrtl.wire sym @r0_init : !firrtl.uint<1>
    firrtl.matchingconnect %r0_init, %1: !firrtl.uint<1>
    %r0 = firrtl.regreset %clock, %reset, %r0_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Port" {
  // expected-note @below {{reset driver is "x"}}
  firrtl.module @NonConstantAsyncReset_Port(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.uint<1>) {
    // expected-error @below {{register "r0" has an async reset, but its reset value "x" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %x : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_PrimOp" {
  firrtl.module @NonConstantAsyncReset_PrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-note @+1 {{reset driver is here}}
    %c1_ui1 = firrtl.not %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{register "r0" has an async reset, but its reset value is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Aggregate0" {
  // expected-note @below {{reset driver is "x"}}
  firrtl.module @NonConstantAsyncReset_Aggregate0(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x : !firrtl.vector<uint<1>, 2>) {
    %value = firrtl.wire : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %value, %x : !firrtl.vector<uint<1>, 2>
    // expected-error @below {{register "r0" has an async reset, but its reset value "value" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Aggregate1" {
  // expected-note @below {{reset driver is "x[0].y"}}
  firrtl.module @NonConstantAsyncReset_Aggregate1(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x : !firrtl.vector<bundle<y: uint<1>>, 1>) {

    // Aggregate wire used for the reset value.
    %value = firrtl.wire : !firrtl.vector<uint<1>, 2>

    // Connect a constant 0 to value[0].
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %value_0 = firrtl.subindex %value[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %value_0, %c0_ui1 : !firrtl.uint<1>

    // Connect a complex chain of operations leading to the port to value[1].
    %subindex = firrtl.subindex %x[0] : !firrtl.vector<bundle<y : uint<1>>, 1>
    %node = firrtl.node %subindex : !firrtl.bundle<y : uint<1>>
    %subfield = firrtl.subfield %node[y] : !firrtl.bundle<y : uint<1>>
    %value_1 = firrtl.subindex %value[1] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %value_1, %subfield : !firrtl.uint<1>

    // expected-error @below {{register "r0" has an async reset, but its reset value "value[1]" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
}

// -----

// CHECK-LABEL: "WalksNestedRegions"
firrtl.circuit "WalksNestedRegions" {
  firrtl.module @WalksNestedRegions(in %a: !firrtl.uint<1>) {
    // CHECK: firrtl.when
    firrtl.when %a : !firrtl.uint<1> {
      // CHECK-NOT: firrtl.invalidvalue
      // CHECK-NEXT: %[[zero:[_A-Za-z0-9]+]] = firrtl.constant 0
      %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
      // CHECK-NEXT: %0 = firrtl.node %[[zero]]
      %0 = firrtl.node %invalid_ui1 : !firrtl.uint<1>
    }
  }
}
