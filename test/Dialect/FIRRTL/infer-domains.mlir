// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains))' %s --split-input-file | FileCheck %s

// Test case 1: Legal domain usage - no crossing
firrtl.circuit "LegalDomains" {
  firrtl.domain @ClockDomain {}
  firrtl.module @LegalDomains(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %b: !firrtl.uint<1> domains [%A]
  ) {
    // Connecting within the same domain is legal.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "LegalDomains"

// -----

// Test case 2: Domain inference through connections
firrtl.circuit "DomainInference" {
  firrtl.domain @ClockDomain {}
  firrtl.module @DomainInference(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %c: !firrtl.uint<1>
  ) {
    %b = firrtl.wire : !firrtl.uint<1>  // No explicit domain

    // This should infer that %b is in domain %A.
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>

    // This should be legal since %b is now inferred to be in domain %A.
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "DomainInference"
// CHECK: out %c: !firrtl.uint<1> domains [%A]

// -----

// Test case 3: Unsafe domain cast
firrtl.circuit "UnsafeDomainCast" {
  firrtl.domain @ClockDomain {}
  firrtl.module @UnsafeDomainCast(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A],
    out %c: !firrtl.uint<1> domains [%B]
  ) {
    // Unsafe cast from domain A to domain B.
    %b = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>

    // This should be legal since we explicitly cast.
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "UnsafeDomainCast"

// -----

// Test case 4: Domain sequence matching - legal case
firrtl.circuit "LegalSequences" {
  firrtl.domain @ClockDomain {}
  firrtl.module @LegalSequences(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A, %B],
    out %b: !firrtl.uint<1> domains [%A, %B]
  ) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "LegalSequences"

// -----

// Test case 5: Domain sequence order equivalence - should be legal
firrtl.circuit "SequenceOrderEquivalence" {
  firrtl.domain @ClockDomain {}
  firrtl.module @SequenceOrderEquivalence(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A, %B],
    out %b: !firrtl.uint<1> domains [%B, %A]
  ) {
    // This should be legal since domain order doesn't matter in canonical representation
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "SequenceOrderEquivalence"

// -----

// Test case 6: Domain sequence inference
firrtl.circuit "SequenceInference" {
  firrtl.domain @ClockDomain {}
  firrtl.module @SequenceInference(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A, %B],
    out %d: !firrtl.uint<1>
  ) {
    %c = firrtl.wire : !firrtl.uint<1>

    // %c should infer domain sequence [%A, %B]
    firrtl.matchingconnect %c, %a : !firrtl.uint<1>

    // This should be legal since %c has inferred [%A, %B]
    firrtl.matchingconnect %d, %c : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "SequenceInference"
// CHECK: out %d: !firrtl.uint<1> domains [%A, %B]

// -----

// Test case 7: Domain duplicate equivalence - should be legal
firrtl.circuit "DuplicateDomainEquivalence" {
  firrtl.domain @ClockDomain {}
  firrtl.module @DuplicateDomainEquivalence(
    in %A: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A, %A],
    out %b: !firrtl.uint<1> domains [%A]
  ) {
    // This should be legal since duplicate domains are canonicalized
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "DuplicateDomainEquivalence"

// -----

// Test case 8: Unsafe domain cast with sequences
firrtl.circuit "UnsafeSequenceCast" {
  firrtl.domain @ClockDomain {}
  firrtl.module @UnsafeSequenceCast(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %C: !firrtl.domain of @ClockDomain,
    in %a: !firrtl.uint<1> domains [%A, %B],
    out %c: !firrtl.uint<1> domains [%C]
  ) {
    %0 = firrtl.unsafe_domain_cast %a domains %C : !firrtl.uint<1>
    firrtl.matchingconnect %c, %0 : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "UnsafeSequenceCast"
// CHECK: out %c: !firrtl.uint<1> domains [%C]

// -----

// Test case 9: Multiple port domain inference
firrtl.circuit "MultiplePortInference" {
  firrtl.domain @ClockDomain {}
  firrtl.module @MultiplePortInference(
    in %A: !firrtl.domain of @ClockDomain,
    in %B: !firrtl.domain of @ClockDomain,
    in %inputA: !firrtl.uint<1> domains [%A],
    in %inputB: !firrtl.uint<1> domains [%B],
    in %inputAB: !firrtl.uint<1> domains [%A, %B],
    out %outputA: !firrtl.uint<1>,
    out %outputB: !firrtl.uint<1>,
    out %outputAB: !firrtl.uint<1>
  ) {
    firrtl.matchingconnect %outputA, %inputA : !firrtl.uint<1>
    firrtl.matchingconnect %outputB, %inputB : !firrtl.uint<1>
    firrtl.matchingconnect %outputAB, %inputAB : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "MultiplePortInference"
// CHECK: out %outputA: !firrtl.uint<1> domains [%A]
// CHECK: out %outputB: !firrtl.uint<1> domains [%B]
// CHECK: out %outputAB: !firrtl.uint<1> domains [%A, %B]

// -----

// Test case 10: Different port types domain inference
firrtl.circuit "DifferentPortTypes" {
  firrtl.domain @ClockDomain {}
  firrtl.module @DifferentPortTypes(
    in %A: !firrtl.domain of @ClockDomain,
    in %uint_input: !firrtl.uint<8> domains [%A],
    in %sint_input: !firrtl.sint<4> domains [%A],
    out %uint_output: !firrtl.uint<8>,
    out %sint_output: !firrtl.sint<4>
  ) {
    firrtl.matchingconnect %uint_output, %uint_input : !firrtl.uint<8>
    firrtl.matchingconnect %sint_output, %sint_input : !firrtl.sint<4>
  }
}
// CHECK-LABEL: firrtl.circuit "DifferentPortTypes"
// CHECK: out %uint_output: !firrtl.uint<8> domains [%A]
// CHECK: out %sint_output: !firrtl.sint<4> domains [%A]

// -----

// Test case 11: Domain inference through wires
firrtl.circuit "DomainInferenceThroughWires" {
  firrtl.domain @ClockDomain {}
  firrtl.module @DomainInferenceThroughWires(
    in %A: !firrtl.domain of @ClockDomain,
    in %input: !firrtl.uint<1> domains [%A],
    out %output: !firrtl.uint<1>
  ) {
    %wire1 = firrtl.wire : !firrtl.uint<1>
    %wire2 = firrtl.wire : !firrtl.uint<1>

    firrtl.matchingconnect %wire1, %input : !firrtl.uint<1>
    firrtl.matchingconnect %wire2, %wire1 : !firrtl.uint<1>
    firrtl.matchingconnect %output, %wire2 : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "DomainInferenceThroughWires"
// CHECK: out %output: !firrtl.uint<1> domains [%A]

// -----

// Test case 12: Register inference
firrtl.circuit "RegisterInference" {
  firrtl.domain @ClockDomain {}
  firrtl.module @RegisterInference(
    in %A: !firrtl.domain of @ClockDomain,
    in %clock: !firrtl.clock domains [%A],
    in %d: !firrtl.uint<1>,
    out %q: !firrtl.uint<1>
  ) {
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %r, %d : !firrtl.uint<1>
    firrtl.matchingconnect %q, %r : !firrtl.uint<1>
  }
}
// CHECK-LABEL: firrtl.circuit "RegisterInference"
// CHECK: in %d: !firrtl.uint<1> domains [%A]
// CHECK: out %q: !firrtl.uint<1> domains [%A]
