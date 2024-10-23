// RUN: circt-opt %s | circt-opt | FileCheck %s

%true = hw.constant true
%s = unrealized_conversion_cast to !ltl.sequence
%p = unrealized_conversion_cast to !ltl.property

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

// CHECK: verif.assert {{%.+}} : i1
// CHECK: verif.assert {{%.+}} label "foo1" : i1
// CHECK: verif.assert {{%.+}} : !ltl.sequence
// CHECK: verif.assert {{%.+}} : !ltl.property
verif.assert %true : i1
verif.assert %true label "foo1" : i1
verif.assert %s : !ltl.sequence
verif.assert %p : !ltl.property

// CHECK: verif.assume {{%.+}} : i1
// CHECK: verif.assume {{%.+}} label "foo2" : i1
// CHECK: verif.assume {{%.+}} : !ltl.sequence
// CHECK: verif.assume {{%.+}} : !ltl.property
verif.assume %true : i1
verif.assume %true label "foo2" : i1
verif.assume %s : !ltl.sequence
verif.assume %p : !ltl.property

// CHECK: verif.cover {{%.+}} : i1
// CHECK: verif.cover {{%.+}} label "foo3" : i1
// CHECK: verif.cover {{%.+}} : !ltl.sequence
// CHECK: verif.cover {{%.+}} : !ltl.property
verif.cover %true : i1
verif.cover %true label "foo3" : i1
verif.cover %s : !ltl.sequence
verif.cover %p : !ltl.property

//===----------------------------------------------------------------------===//
// Formal Test
//===----------------------------------------------------------------------===//

// CHECK-LABEL: verif.formal @EmptyFormalTest
verif.formal @EmptyFormalTest {
}

// CHECK-LABEL: verif.formal @FormalTestWithAttrs
verif.formal @FormalTestWithAttrs attributes {
  // CHECK-SAME: a,
  a,
  // CHECK-SAME: b = "hello"
  b = "hello",
  // CHECK-SAME: c = 42 : i64
  c = 42 : i64,
  // CHECK-SAME: d = ["x", "y"]
  d = ["x", "y"],
  // CHECK-SAME: e = {u, v}
  e = {u, v}
} {
}

// CHECK-LABEL: verif.formal @FormalTestBody
verif.formal @FormalTestBody {
  // CHECK: {{%.+}} = verif.symbolic_value : i42
  %0 = verif.symbolic_value : i42
}

//===----------------------------------------------------------------------===//
// Contracts
//===----------------------------------------------------------------------===//

verif.formal @CheckMul9 {
  %c3_i42 = hw.constant 3 : i42
  %0 = verif.symbolic_value : i42
  %1 = comb.shl %0, %c3_i42 : i42  // 8*x
  %2 = comb.add %0, %1 : i42 // x + 8*x
  %3 = verif.contract %2 : i42 {
    %c9_i42 = hw.constant 9 : i42
    %4 = comb.mul %0, %c9_i42 : i42 // 9*x
    %5 = comb.icmp eq %2, %4 : i42 // 9*x == x + 8*x
    verif.ensure %5
    verif.yield %4
  }
}

// Example module that computes `z = 9*a` using `shl` and `add`.
hw.module @Mul9(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42    // 8*a
  %1 = comb.add %a, %0 : i42         // a + 8*a
  %2 = verif.contract %1 : i42 {
    %3 = comb.mul %a, %c9_i42 : i42  // 9*a
    verif.ensure_equal %2, %3        // 9*a == a + 8*a
    verif.yield %3
  }
  hw.output %2 : i42
}

// Using the contract by assuming it holds:
//   ensure -> assume
//   contract assumed to hold and yield %3 for %2
//   contract is inlined where it is with %2 -> %3 replacement
hw.module @Mul9WithContractApplied(in %a: i42, out z: i42) {
  %c3_i42 = hw.constant 3 : i42
  %c9_i42 = hw.constant 9 : i42
  %0 = comb.shl %a, %c3_i42 : i42  // 8*a
  %1 = comb.add %a, %0 : i42       // a + 8*a
  %3 = comb.mul %a, %c9_i42 : i42  // 9*a
  verif.assume_equal %3, %3 : i42  // 9*a == a + 8*a
  // assume(%3 == %3) aka assume(true) is a no-op and can be DCE'd
  hw.output %3 : i42
}
hw.module @Mul9WithContractAppliedAfterDCE(in %a: i42, out z: i42) {
  %c9_i42 = hw.constant 9 : i42
  %3 = comb.mul %a, %c9_i42 : i42
  hw.output %3 : i42
}

// Proving the contract:
//   ensure -> assert
//   contract assumed to just forward %1 to %2
//   contract is inlined into a formal test with %2 -> %1 replacement
verif.formal @Mul9ContractTest {
  %a = verif.symbolic_value : i42
  %c3_i42 = hw.constant 3 : i42
  %0 = comb.shl %a, %c3_i42 : i42  // 8*a
  %1 = comb.add %a, %0 : i42       // a + 8*a
  %3 = comb.mul %a, %c9_i42 : i42  // 9*a
  verif.assert_equal %1, %3 : i42  // 9*a == a + 8*a
}

// A module that takes 3 input values and produces 2 output values that sum up
// to the same value as the inputs. Instead of just using add it uses a
// bit-parallel full adder that takes each 3-tuple of bits in the 3 inputs, runs
// them through a full adder, and treats the resulting sum and carry as the 2
// corresponding bits for its 2 output values.
hw.module @CarrySaveCompress3to2(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42  // sum bits of FA (a0^a1^a2)
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42          // carry bits of FA (a0&a1 | a2&(a0|a1))
  %5 = comb.shl %4, %c1_i42 : i42    // %5 = carry << 1
  // At this point, %0+%5 is the same as %a0+%a1+%a2, but without creating a
  // long ripple-carry chain.

  // Contract to check that we output _some_ two numbers that sum up to the same
  // value as the sum of the three inputs. We don't say which exact numbers.
  %z0, %z1 = verif.contract %0, %5 {
    // The contract promises that its outputs will sum up to the same value as
    // the sum of the module inputs.
    %inputSum = comb.add %a0, %a1, %a2 : i42
    %outputSum = comb.add %z0, %z1 : i42
    verif.ensure_equal %inputSum, %outputSum : i42

    // We don't want this contract to give guarantees about what the exact
    // values of its outputs are going to be. Instead, we only want to guarantee
    // that they sum up to the right number. To express this, we pick two
    // symbolic values and constrain them to sum up to that number, and yield
    // those from the contract. This says "you'll get any two numbers that sum
    // up to the sum of the inputs".
    %any0 = verif.symbolic_value : i42
    %any1 = verif.symbolic_value : i42
    %anySum = comb.add %any0, %any1 : i42
    verif.assume_equal %anySum, %inputSum : i42
    verif.yield %any0, %any1 : i42
  }
  hw.output %z0, %z1 : i42, i42
}

// A module that takes 5 input values and sums them up using a carry save adder.
hw.module @CarrySaveAdder5(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  // Each stage takes 3 of the terms and compresses them to 2.
  // terms: [a0, a1, a2, a3, a4]
  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  // terms: [b0, b1, a3, a4]
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  // terms: [b0, c0, c1]
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  // terms: [d0, d1]
  %e = comb.add %d0, %d1 : i42
  // terms: [e]

  // Contract to check that the output is the sum of all inputs.
  %z = verif.contract %e {
    %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
    verif.ensure_equal %z, %inputSum : i42
    verif.yield %inputSum : i42
  }
  hw.output %z : i42
}

// with contracts lowered:

hw.module @CarrySaveCompress3to2_AssumeContract(
  in %a0: i42, in %a1: i42, in %a2: i42,
  out z0: i42, out z1: i42
) {
  // Actual implementation becomes unused after contract inlining and can be
  // removed by dead code elimination.

  // Contract inlined with ensure -> assume, (%z0, %z1) -> (%any0, %any1).
  // Redundant assumes eliminated. They can also be left in.
  %any0 = verif.symbolic_value : i42
  %any1 = verif.symbolic_value : i42
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %any0, %any1 : i42
  verif.assume_equal %inputSum, %outputSum : i42

  hw.output %any0, %any1 : i42, i42
}

verif.formal @CarrySaveCompress3to2_AssertContract {
  %a0 = verif.symbolic_value : i42
  %a1 = verif.symbolic_value : i42
  %a2 = verif.symbolic_value : i42

  %c1_i42 = hw.constant 1 : i42
  %0 = comb.xor %a0, %a1, %a2 : i42
  %1 = comb.and %a0, %a1 : i42
  %2 = comb.or %a0, %a1 : i42
  %3 = comb.and %2, %a2 : i42
  %4 = comb.or %1, %3 : i42
  %5 = comb.shl %4, %c1_i42 : i42

  // Contract inlined with ensure -> assert, (%z0, %z1) -> (%0, %5).
  // Symbolic values omitted.
  %inputSum = comb.add %a0, %a1, %a2 : i42
  %outputSum = comb.add %0, %5 : i42
  verif.assert_equal %inputSum, %outputSum : i42
}

// -----

// with contracts lowered:

hw.module @CarrySaveAdder5_AssumeContract(
  in %a0: i42, in %a1: i42, in %a2: i42, in %a3: i42, in %a4: i42,
  out z: i42
) {
  // Actual implementation becomes unused after contract inlining and can be
  // removed by dead code elimination.

  // Contract inlined with ensure -> assume, %z -> %inputSum.
  // Trivial assume(a == a), which is a noop, can be removed.
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42

  hw.output %inputSum : i42
}

verif.formal @CarrySaveAdder5_AssertContract {
  %a0 = verif.symbolic_value : i42
  %a1 = verif.symbolic_value : i42
  %a2 = verif.symbolic_value : i42
  %a3 = verif.symbolic_value : i42
  %a4 = verif.symbolic_value : i42

  %b0, %b1 = hw.instance "comp0" @CarrySaveCompress3to2_AssumeContract(a0: %a0: i42, a1: %a1: i42, a2: %a2: i42) -> (z0: i42, z1: i42)
  %c0, %c1 = hw.instance "comp1" @CarrySaveCompress3to2_AssumeContract(a0: %b1: i42, a1: %a3: i42, a2: %a4: i42) -> (z0: i42, z1: i42)
  %d0, %d1 = hw.instance "comp2" @CarrySaveCompress3to2_AssumeContract(a0: %b0: i42, a1: %c0: i42, a2: %c1: i42) -> (z0: i42, z1: i42)
  %e = comb.add %d0, %d1 : i42

  // Contract inlined with ensure -> assert, %z -> %e.
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
  verif.assert_equal %e, %inputSum : i42
}

// -----

verif.formal @CarrySaveAdder5_AssertContract_Flat {
  %a0 = verif.symbolic_value : i42
  %a1 = verif.symbolic_value : i42
  %a2 = verif.symbolic_value : i42
  %a3 = verif.symbolic_value : i42
  %a4 = verif.symbolic_value : i42

  %b0 = verif.symbolic_value : i42
  %b1 = verif.symbolic_value : i42
  %inputSum0 = comb.add %a0, %a1, %a2 : i42
  %outputSum0 = comb.add %b0, %b1 : i42
  verif.assume_equal %inputSum0, %outputSum0 : i42

  %c0 = verif.symbolic_value : i42
  %c1 = verif.symbolic_value : i42
  %inputSum1 = comb.add %b1, %a3, %a4 : i42
  %outputSum1 = comb.add %c0, %c1 : i42
  verif.assume_equal %inputSum1, %outputSum1 : i42

  %d0 = verif.symbolic_value : i42
  %d1 = verif.symbolic_value : i42
  %inputSum2 = comb.add %b0, %c0, %c1 : i42
  %outputSum2 = comb.add %d0, %d1 : i42
  verif.assume_equal %inputSum2, %outputSum2 : i42

  %e = comb.add %d0, %d1 : i42

  // Contract inlined with ensure -> assert, %z -> %e.
  %inputSum = comb.add %a0, %a1, %a2, %a3, %a4, %a5 : i42
  verif.assert_equal %e, %inputSum : i42
}


// CHECK-LABEL: hw.module @Bar
hw.module @Bar(in %foo : i8, out "" : i8, out "1" : i8) { 
  // CHECK: %[[C1:.+]] = hw.constant
  %c1_8 = hw.constant 1 : i8
  // CHECK: %[[O1:.+]] = comb.add
  %to0 = comb.add bin %foo, %c1_8 : i8
  // CHECK: %[[O2:.+]] = comb.sub
  %to1 = comb.sub bin %foo, %c1_8 : i8

  // CHECK: %[[OUT:.+]]:2 = verif.contract(%[[O1]], %[[O2]]) : (i8, i8) -> (i8, i8) {
  %o0, %o1 = verif.contract (%to0, %to1) : (i8, i8) -> (i8, i8) {
    // CHECK: ^bb0(%[[BAR0:.+]]: i8, %[[BAR1:.+]]: i8):
    ^bb0(%bar.0 : i8, %bar.1 : i8): 
      // CHECK: %[[C0:.+]] = hw.constant 0 : i8
      %c0_8 = hw.constant 0 : i8 
      // CHECK: %[[PREC:.+]] = comb.icmp bin ugt %foo, %[[C0]] : i8
      %prec = comb.icmp bin ugt %foo, %c0_8 : i8
      // CHECK: verif.require %[[PREC]] : i1
      verif.require %prec : i1

      // CHECK: %[[P0:.+]] = comb.icmp bin ugt %[[BAR0]], %foo : i8
      %post = comb.icmp bin ugt %bar.0, %foo : i8
      // CHECK: %[[P1:.+]] = comb.icmp bin ult %[[BAR1]], %foo : i8
      %post1 = comb.icmp bin ult %bar.1, %foo : i8
      // CHECK: verif.ensure %[[P0]] : i1
      verif.ensure %post : i1
      // CHECK: verif.ensure %[[P1]] : i1
      verif.ensure %post1 : i1
      // CHECK: verif.yield %[[BAR0]], %[[BAR1]] : i8, i8
      verif.yield %bar.0, %bar.1 : i8, i8
  } 
  
  // CHECK-LABEL: hw.output
  hw.output %o0, %o1 : i8, i8
}

//===----------------------------------------------------------------------===//
// Print-related
// Must be inside hw.module to ensure that the dialect is loaded.
//===----------------------------------------------------------------------===//

hw.module @foo() {
// CHECK:    %false = hw.constant false
// CHECK:    %[[FSTR:.*]] = verif.format_verilog_string "Hi %x\0A"(%false) : i1
// CHECK:    verif.print %[[FSTR]]
  %false = hw.constant false
  %fstr = verif.format_verilog_string "Hi %x\0A" (%false) : i1
  verif.print %fstr
}

// CHECK-LABEL: hw.module @HasBeenReset
hw.module @HasBeenReset(in %clock: i1, in %reset: i1) {
  // CHECK-NEXT: verif.has_been_reset %clock, async %reset
  // CHECK-NEXT: verif.has_been_reset %clock, sync %reset
  %hbr0 = verif.has_been_reset %clock, async %reset
  %hbr1 = verif.has_been_reset %clock, sync %reset
}

//===----------------------------------------------------------------------===//
// Logical Equivalence Checking related operations
//===----------------------------------------------------------------------===//

// CHECK: verif.lec first {
// CHECK: } second {
// CHECK: }
verif.lec first {
} second {
}

// CHECK: verif.lec {verif.some_attr} first {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: } second {
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: i32):
// CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32 {verif.some_attr}
// CHECK: }
verif.lec {verif.some_attr} first {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
} second {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0, %arg1 : i32, i32 {verif.some_attr}
}

//===----------------------------------------------------------------------===//
// Bounded Model Checking related operations
//===----------------------------------------------------------------------===//

// CHECK: verif.bmc bound 10 num_regs 0 attributes {verif.some_attr} init {
// CHECK: } loop {
// CHECK: } circuit {
// CHECK: ^bb0(%{{.*}}):
// CHECK: verif.yield %{{.*}} : i32
// CHECK: }
verif.bmc bound 10 num_regs 0 attributes {verif.some_attr} init {
} loop {
} circuit {
^bb0(%arg0: i32):
  %false = hw.constant false
  // Arbitrary assertion so op verifies
  verif.assert %false : i1
  verif.yield %arg0 : i32
}

//CHECK: verif.bmc bound 10 num_regs 1 attributes {verif.some_attr}
//CHECK: init {
//CHECK:   %{{.*}} = hw.constant false
//CHECK:   %{{.*}} = seq.to_clock %{{.*}}
//CHECK:   verif.yield %{{.*}}, %{{.*}} : !seq.clock, i1
//CHECK: }
//CHECK: loop {
//CHECK:   ^bb0(%{{.*}}: !seq.clock, %{{.*}}: i1):
//CHECK:   %{{.*}} = seq.from_clock %{{.*}}
//CHECK:   %{{.*}} = hw.constant true
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i1
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i1
//CHECK:   %{{.*}} = seq.to_clock %{{.*}}
//CHECK:   verif.yield %{{.*}}, %{{.*}} : !seq.clock, i1
//CHECK: }
//CHECK: circuit {
//CHECK: ^bb0(%{{.*}}: !seq.clock, %{{.*}}: i32, %{{.*}}: i32):
//CHECK:   %{{.*}} = hw.constant -1 : i32
//CHECK:   %{{.*}} = comb.add %{{.*}}, %{{.*}} : i32
//CHECK:   %{{.*}} = comb.xor %{{.*}}, %{{.*}} : i32
//CHECK:   verif.yield %{{.*}}, %{{.*}} : i32, i32
//CHECK: }
verif.bmc bound 10 num_regs 1 attributes {verif.some_attr}
init {
  %c0_i1 = hw.constant 0 : i1
  %clk = seq.to_clock %c0_i1
  verif.yield %clk, %c0_i1 : !seq.clock, i1
}
loop {
  ^bb0(%clk: !seq.clock, %stateArg: i1):
  %from_clock = seq.from_clock %clk
  %c-1_i1 = hw.constant -1 : i1
  %neg_clock = comb.xor %from_clock, %c-1_i1 : i1
  %newStateArg = comb.xor %stateArg, %c-1_i1 : i1
  %newclk = seq.to_clock %neg_clock
  verif.yield %newclk, %newStateArg : !seq.clock, i1
}
circuit {
^bb0(%clk: !seq.clock, %arg0: i32, %state0: i32):
  %c-1_i32 = hw.constant -1 : i32
  %0 = comb.add %arg0, %state0 : i32
  // %state0 is the result of a seq.compreg taking %0 as input
  %2 = comb.xor %state0, %c-1_i32 : i32
  %false = hw.constant false
  // Arbitrary assertion so op verifies
  verif.assert %false : i1
  verif.yield %2, %0 : i32, i32
}
