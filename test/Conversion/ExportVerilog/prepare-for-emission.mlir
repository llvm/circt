// RUN: circt-opt %s --pass-pipeline='builtin.module(any(prepare-for-emission))' --split-input-file -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-verif-to-sv -export-verilog -split-input-file

// CHECK: @namehint_variadic
hw.module @namehint_variadic(in %a: i3, out b: i3) {
  // CHECK-NEXT: %0 = comb.add %a, %a : i3
  // CHECK-NEXT: %1 = comb.add %a, %0 {sv.namehint = "bar"} : i3
  // CHECK-NEXT: hw.output %1
  %0 = comb.add %a, %a, %a { sv.namehint = "bar" } : i3
  hw.output %0 : i3
}

// CHECK-LABEL:  hw.module @outOfOrderInoutOperations
hw.module @outOfOrderInoutOperations(in %a: i4, out c: i4) {
  // CHECK: %wire = sv.wire
  // CHECK-NEXT: %0 = sv.array_index_inout %wire[%false]
  // CHECK-NEXT: %1 = sv.array_index_inout %0[%false]
  // CHECK-NEXT: %2 = sv.array_index_inout %1[%false]
  // CHECK-NEXT: %3 = sv.read_inout %2
  %false = hw.constant false
  %0 = sv.read_inout %3 : !hw.inout<i4>
  %3 = sv.array_index_inout %2[%false] : !hw.inout<array<1xi4>>, i1
  %2 = sv.array_index_inout %1[%false] : !hw.inout<array<1xarray<1xi4>>>, i1
  %1 = sv.array_index_inout %wire[%false] : !hw.inout<array<1xarray<1xarray<1xi4>>>>, i1
  %wire = sv.wire  : !hw.inout<array<1xarray<1xarray<1xi4>>>>
  hw.output %0: i4
}

// CHECK-LABEL: @twoState_variadic
hw.module @twoState_variadic(in %a: i1, in %b: i1, in %c: i1, out d:i1){
  // CHECK:      %0 = comb.or %b, %c : i1
  // CHECK-NEXT: %1 = comb.or %a, %0 : i1
  // CHECK-NEXT: hw.output %1 : i1
  %0 = comb.or %a, %b, %c: i1
  hw.output %0: i1
}

// CHECK-LABEL: @carryOverWireAttrs
hw.module @carryOverWireAttrs(in %a: i1, out b: i1){
  // CHECK-NEXT: %foo = sv.wire {magic, sv.attributes = []} : !hw.inout<i1>
  // CHECK-NEXT: sv.assign %foo, %a
  // CHECK-NEXT: [[TMP:%.+]] = sv.read_inout %foo
  // CHECK-NEXT: hw.output [[TMP]] : i1
  %foo = hw.wire %a {magic, sv.attributes = []} : i1
  hw.output %foo : i1
}

// -----

module {
  // CHECK-LABEL:  hw.module @SpillTemporaryInProceduralRegion
  hw.module @SpillTemporaryInProceduralRegion(in %a: i4, in %b: i4, in %fd: i32) {
    // CHECK-NEXT: %r = sv.reg
    // CHECK-NEXT: sv.initial {
    // CHECK-NEXT:   %0 = sv.logic
    // CHECK-NEXT:   %1 = comb.add %a, %b
    // CHECK-NEXT:   sv.bpassign %0, %1
    // CHECK-NEXT:   %2 = sv.read_inout %0
    // CHECK-NEXT:   %3 = comb.extract %2 from 3
    // CHECK-NEXT:   sv.passign %r, %3
    // CHECK-NEXT: }
    // CHECK-NEXT: hw.output
    %r = sv.reg : !hw.inout<i1>
    sv.initial {
      %0 = comb.add %a, %b : i4
      %1 = comb.extract %0 from 3 : (i4) -> i1
      sv.passign %r, %1: i1
    }
  }
}

// -----

module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  // CHECK: @test_hoist
  hw.module @test_hoist(in %a: i3) {
    // CHECK-NEXT: %reg = sv.reg
    %reg = sv.reg : !hw.inout<i3>
    // CHECK-NEXT: %0 = comb.add
    // CHECK-NEXT: sv.initial
    sv.initial {
      %0 = comb.add %a, %a : i3
      sv.passign %reg, %0 : i3
    }
  }

  // CHECK-LABEL:  hw.module @SpillTemporary
  hw.module @SpillTemporary(in %a: i4, in %b: i4, out c: i1) {
    // CHECK-NEXT:  %0 = comb.add %a, %b
    // CHECK-NEXT:  %[[GEN:.+]] = sv.wire
    // CHECK-NEXT:  sv.assign %1, %[[GEN:.+]]
    // CHECK-NEXT:  %2 = sv.read_inout %1
    // CHECK-NEXT:  %3 = comb.extract %2 from 3
    // CHECK-NEXT:  hw.output %3
    %0 = comb.add %a, %b : i4
    %1 = comb.extract %0 from 3 : (i4) -> i1
    hw.output %1 : i1
  }

  // CHECK-LABEL:  hw.module @SpillTemporaryInProceduralRegion
  hw.module @SpillTemporaryInProceduralRegion(in %a: i4, in %b: i4, in %fd: i32) {
    // CHECK-NEXT: %r = sv.reg
    // CHECK-NEXT: %[[VAL:.+]] = comb.add %a, %b
    // CHECK-NEXT: %[[GEN:.+]] = sv.wire
    // CHECK-NEXT: sv.assign %[[GEN]], %[[VAL]]
    // CHECK-NEXT: %2 = sv.read_inout %[[GEN]]
    // CHECK-NEXT: %3 = comb.extract %2 from 3
    // CHECK-NEXT: sv.initial {
    // CHECK-NEXT:   sv.passign %r, %3
    // CHECK-NEXT: }
    // CHECK-NEXT: hw.output
    %r = sv.reg : !hw.inout<i1>
    sv.initial {
      %0 = comb.add %a, %b : i4
      %1 = comb.extract %0 from 3 : (i4) -> i1
      sv.passign %r, %1: i1
    }
  }

  // CHECK-LABEL: @SpillTemporaryWireForMultipleUseExpression
  hw.module @SpillTemporaryWireForMultipleUseExpression(in %a: i4, in %b: i4, out c: i4, out d: i4) {
    // CHECK-NEXT: %[[VAL:.+]] = comb.add %a, %b
    // CHECK-NEXT: %[[GEN:bar]] = sv.wire
    // CHECK-NEXT: sv.assign %[[GEN]], %[[VAL]]
    // CHECK-NEXT: %1 = sv.read_inout %[[GEN]]
    // CHECK-NEXT: %2 = sv.read_inout %[[GEN]]
    // CHECK-NEXT: hw.output %2, %1
    %0 = comb.add %a, %b {sv.namehint = "bar"}: i4
    hw.output %0, %0 : i4, i4
  }
}

// -----

module attributes {circt.loweringOptions = "disallowExpressionInliningInPorts"} {
 hw.module.extern @MyExtModule(in %in: i8)
 // CHECK-LABEL: @MoveInstances
 hw.module @MoveInstances(in %a_in: i8){
  // CHECK-NEXT: %_xyz3_in = sv.wire
  // CHECK-NEXT: %0 = comb.add %a_in, %a_in
  // CHECK-NEXT: %1 = sv.read_inout %_xyz3_in
  // CHECK-NEXT: sv.assign %_xyz3_in, %0
  // CHECK-NEXT: hw.instance "xyz3" @MyExtModule(in: %1: i8) -> ()
  %0 = comb.add %a_in, %a_in : i8
  hw.instance "xyz3" @MyExtModule(in: %0: i8) -> ()
 }
}

// -----

module attributes {circt.loweringOptions = "disallowExpressionInliningInPorts"} {
  // CHECK-LABEL: @ClockExpr(
  hw.module @ClockExpr(in %clk: i1, in %a: i1, in %b: i1) {
    %clk_xor_b = comb.xor %clk, %b : i1

    // CHECK: %[[XOR:.+]] = comb.xor
    // CHECK: %[[WIRE:.+]] = sv.wire
    // CHECK: sv.assign %[[WIRE]], %[[XOR]]
    // CHECK: %[[READ:.+]] = sv.read_inout %[[WIRE]]
    // CHECK: ltl.clock %{{.+}} posedge %[[READ]]
    %i0 = ltl.implication %a, %b : i1, i1
    %k0 = ltl.clock %i0, posedge %clk_xor_b : !ltl.property

    verif.assert %k0 : !ltl.property
  }
}

// -----
module attributes {circt.loweringOptions =
                  "wireSpillingHeuristic=spillLargeTermsWithNamehints,wireSpillingNamehintTermLimit=3"} {
  // CHECK-LABEL: namehints
  hw.module @namehints(in %a: i8, out b: i8) {
    // "foo" should be spilled because it has a meaningful name.
    // CHECK: %foo = sv.wire
    %0 = comb.add %a, %a {sv.namehint = "foo" } : i8
    // "_foo" should not be spilled because it has a name which starts with "_" and the term size is 2.
    // CHECK-NOT: %_foo = sv.wire
    %3 = comb.add %a, %a {sv.namehint = "_foo" } : i8
    // "_bar" should be spilled because the term size is 3.
    // CHECK: %_bar = sv.wire
    %1 = comb.add %a, %a, %a {sv.namehint = "_bar" } : i8
    %2 = comb.add %0, %1, %3 : i8
    hw.output %2 : i8
  }
}

// -----
module attributes {circt.loweringOptions =
                  "disallowMuxInlining"} {
  // CHECK-LABEL: mux
  hw.module @mux(in %c: i1, in %b: i8, in %a: i8, out d: i8, out e: i8) {
    // CHECK:      %use_for_mux = sv.wire
    // CHECK-NEXT: sv.assign %use_for_mux, %0 : i8
    // CHECK-NEXT: %[[read:.+]] = sv.read_inout %use_for_mux : !hw.inout<i8>
    // CHECK-NEXT: %[[add:.+]] = comb.add %[[read]], %a : i8
    %0 = comb.mux %c, %a, %b : i8
    %use_for_mux = sv.wire : !hw.inout<i8>
    sv.assign %use_for_mux, %0 : i8
    %1 = comb.add %0, %a : i8
    // CHECK: %[[mux2:.+]] = comb.mux
    %2 = comb.mux %c, %a, %b : i8
    // CHECK: hw.output %[[add]], %[[mux2]]
    hw.output %1, %2 : i8, i8
  }
}

// -----
// CHECK: "wireSpillingHeuristic=spillLargeTermsWithNamehints,disallowMuxInlining"
module attributes {circt.loweringOptions =
                  "wireSpillingHeuristic=spillLargeTermsWithNamehints,disallowMuxInlining"} {
  hw.module @combine(in %c: i1, in %b: i8, in %a: i8, out d: i8) {
    // Meaningful names should be spilled
    // CHECK: %foo = sv.wire
    // Mux should be spilled
    // CHECK: sv.wire
    %0 = comb.add %a, %a {sv.namehint = "foo" } : i8
    %1 = comb.mux %c, %0, %b : i8
    %2 = comb.add %1, %a : i8
    hw.output %2 : i8
  }
}

// -----
module attributes {circt.loweringOptions = "maximumNumberOfTermsPerExpression=2"} {
  // CHECK-NOT: sv.wire
  hw.module @Foo(in %in_0: i4, in %in_1: i4, in %in_2: i4, in %in_3: i4, out out: !hw.array<4xi4>) {
    %0 = comb.concat %in_0, %in_1, %in_2, %in_3 : i4, i4, i4, i4
    %1 = hw.bitcast %0 : (i16) -> !hw.array<4xi4>
    hw.output %1 : !hw.array<4xi4>
  }
}

// -----

// CHECK:   hw.module @packed_struct_assignment(in %[[VAL_0:.*]] : i32, out out : !hw.struct<a: i32>, out out2 : !hw.struct<a: i32>, out out3 : !hw.struct<a: i32>) {
// CHECK:           %[[VAL_1:.*]] = sv.wire
// CHECK-NEXT:      %[[VAL_2:.*]] = sv.struct_field_inout %[[VAL_1]]["a"]
// CHECK-NEXT:      sv.assign %[[VAL_2]], %[[VAL_0]]
// CHECK-NEXT:      %[[VAL_3:.*]] = sv.read_inout %[[VAL_1]]
// CHECK-NEXT:      %[[VAL_4:.*]] = sv.read_inout %[[VAL_1]]
// CHECK:           %[[VAL_5:.*]] = sv.wire
// CHECK-NEXT:      %[[VAL_6:.*]] = sv.struct_field_inout %[[VAL_5]]["a"]
// CHECK-NEXT:      sv.assign %[[VAL_6]], %c1_i32
// CHECK-NEXT:      %[[VAL_7:.*]] = sv.read_inout %[[VAL_5]]
// CHECK-NEXT:      hw.output %[[VAL_4]], %[[VAL_3]], %[[VAL_7]]
// CHECK:         }
!T = !hw.struct<a: i32>
module attributes { circt.loweringOptions = "disallowPackedStructAssignments"} {
  hw.module @packed_struct_assignment(in %in : i32, out out: !T, out out2: !T, out out3: !T)  {
      %0 = hw.struct_create (%in) : !T
      %1 = hw.aggregate_constant [1: i32] : !T
      hw.output %0, %0, %1 : !T, !T, !T
  }
}

// -----
// LTL expressions that are used before being defined should not be spilled to
// wires, where they crash the PrepareForEmission pass. They are always emitted
// inline, so no need to restructure the IR.
// CHECK-LABEL: hw.module @Issue5613
hw.module @Issue5613(in %a: i1, in %b: i1) {
  verif.assert %2 : !ltl.sequence
  %0 = ltl.implication %2, %1 : !ltl.sequence, !ltl.property
  %1 = ltl.or %b, %3 : i1, !ltl.property
  %2 = ltl.and %b, %4 : i1, !ltl.sequence
  %3 = ltl.not %b : i1
  %4 = ltl.delay %a, 42 : i1
  hw.output
}

// -----

// If an operation is duplicated because it is always inline, make sure that the
// recursive inlining of its operands (by splitting) may also duplicate an
// operand which now needs to be spilled to a wire.
//
// See: https://github.com/llvm/circt/issues/5605
// CHECK-LABEL: hw.module @Issue5605
hw.module @Issue5605(in %a: i1, in %b: i1, in %clock: i1, in %reset: i1) {
  %0 = comb.concat %a, %b : i1, i1
  // CHECK:      %1 = sv.wire
  // CHECK-NEXT: sv.assign %1, %0
  %1 = sv.system.sampled %0 : i2
  // CHECK-NEXT: %2 = sv.read_inout %1
  // CHECK-NEXT: %3 = sv.system.sampled %2
  // CHECK-NEXT: sv.assert.concurrent {{.*}}(%3)
  sv.assert.concurrent posedge %clock, %reset label "assert_0" message "foo"(%1) : i2
  // CHECK-NEXT: %4 = sv.read_inout %1
  // CHECK-NEXT: %5 = sv.system.sampled %4
  // CHECK-NEXT: sv.assert.concurrent {{.*}}(%5)
  sv.assert.concurrent posedge %clock, %reset label "assert_1" message "bar"(%1) : i2
  hw.output
}

// -----

// The following use of `sv.xmr.ref` inside a procedural block would trigger an
// assertion in `PrepareForEmission`, since the op was not marked as side-effect
// free.
//
// CHECK-LABEL: hw.module @Foo
module attributes {circt.loweringOptions = "disallowLocalVariables"} {
  hw.module @Foo(in %a: i1) {
    hw.wire %a sym @a : i1
    // CHECK: sv.alwayscomb
    sv.alwayscomb {
      // CHECK-NEXT: sv.xmr.ref
      %0 = sv.xmr.ref @xmr : !hw.inout<i1>
      sv.verbatim "{{0}}" (%0) : !hw.inout<i1>
    }
  }
  hw.hierpath @xmr [@Foo::@a]
}


// -----

// CHECK-LABEL: @constantInitRegWithBackEdge
hw.module @constantInitRegWithBackEdge() {
  // CHECK: %reg = sv.reg init %false : !hw.inout<i1>
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %[[VAL_0:.*]] = sv.read_inout %reg : !hw.inout<i1>
  // CHECK-NEXT: %[[VAL_1:.*]] = comb.or %false, %[[VAL_0]] : i1
  %false = hw.constant false
  %0 = comb.or %false, %1 : i1
  %reg = sv.reg init %false : !hw.inout<i1>
  %1 = sv.read_inout %reg : !hw.inout<i1>
}

// -----

// CHECK-LABEL: @temporaryWireForReg
hw.module @temporaryWireForReg() {
  // CHECK: %[[WIRE:.*]] = sv.wire : !hw.inout<i1>
  // CHECK-NEXT: %[[VAL_0:.*]] = sv.read_inout %[[WIRE]]  : !hw.inout<i1>
  // CHECK-NEXT: %b = sv.reg init %[[VAL_0]] : !hw.inout<i1>
  // CHECK-NEXT: %[[VAL_1:.*]] = sv.read_inout %b : !hw.inout<i1>
  // CHECK-NEXT: %a = sv.reg init %[[VAL_1]] : !hw.inout<i1>
  // CHECK-NEXT: %[[VAL_2:.*]] = sv.read_inout %a : !hw.inout<i1>
  // CHECK-NEXT: sv.assign %[[WIRE]], %[[VAL_2]] : i1
  %0 = sv.read_inout %a : !hw.inout<i1>
  %1 = sv.read_inout %b : !hw.inout<i1>
  %b = sv.reg init %0 : !hw.inout<i1>
  %a = sv.reg init %1 : !hw.inout<i1>
}
