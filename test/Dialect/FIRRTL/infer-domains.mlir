// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-domains))' %s --verify-diagnostics --split-input-file

//===----------------------------------------------------------------------===//
// PASSING TESTS - Expected to succeed without errors
//===----------------------------------------------------------------------===//

// Test case 1: Legal domain usage - no crossing
firrtl.circuit "LegalDomains" {
  firrtl.module @LegalDomains(
    in %A: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A : !firrtl.uint<1>

    // This should be legal - connecting within the same domain
    %a2 = firrtl.wire domains %A : !firrtl.uint<1>
    firrtl.matchingconnect %a2, %a : !firrtl.uint<1>
  }
}

// -----

// Test case 2: Domain inference through connections
firrtl.circuit "DomainInference" {
  firrtl.module @DomainInference(
    in %A: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<1>  // No explicit domain

    // This should infer that %b is in domain %A
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>

    // This should be legal since %b is now inferred to be in domain %A
    %c = firrtl.wire domains %A : !firrtl.uint<1>
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>
  }
}

// -----

// Test case 3: Unsafe domain cast
firrtl.circuit "UnsafeDomainCast" {
  firrtl.module @UnsafeDomainCast(
    in %A: !firrtl.domain,
    in %B: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A : !firrtl.uint<1>

    // Unsafe cast from domain A to domain B
    %b = firrtl.unsafe_domain_cast %a domains %B : !firrtl.uint<1>

    // This should be legal since we explicitly cast
    %c = firrtl.wire domains %B : !firrtl.uint<1>
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>
  }
}

// -----

// Test case 4: Domain sequence matching - legal case
firrtl.circuit "LegalSequences" {
  firrtl.module @LegalSequences(
    in %A: !firrtl.domain,
    in %B: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A, %B : !firrtl.uint<1>
    %b = firrtl.wire domains %A, %B : !firrtl.uint<1>

    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// -----

// Test case 5: Domain sequence order equivalence - should be legal
firrtl.circuit "SequenceOrderEquivalence" {
  firrtl.module @SequenceOrderEquivalence(
    in %A: !firrtl.domain,
    in %B: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A, %B : !firrtl.uint<1>
    %b = firrtl.wire domains %B, %A : !firrtl.uint<1>

    // This should be legal since domain order doesn't matter in canonical representation
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// -----

// Test case 6: Domain sequence inference
firrtl.circuit "SequenceInference" {
  firrtl.module @SequenceInference(
    in %A: !firrtl.domain,
    in %B: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A, %B : !firrtl.uint<1>
    %c = firrtl.wire : !firrtl.uint<1>

    // %c should infer domain sequence [%A, %B]
    firrtl.matchingconnect %c, %a : !firrtl.uint<1>

    %d = firrtl.wire domains %A, %B : !firrtl.uint<1>
    // This should be legal since %c has inferred [%A, %B]
    firrtl.matchingconnect %d, %c : !firrtl.uint<1>
  }
}

// -----

// Test case 7: Domain duplicate equivalence - should be legal
firrtl.circuit "DuplicateDomainEquivalence" {
  firrtl.module @DuplicateDomainEquivalence(
    in %A: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A, %A : !firrtl.uint<1>
    %b = firrtl.wire domains %A : !firrtl.uint<1>

    // This should be legal since duplicate domains are canonicalized
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}

// -----

// Test case 8: Unsafe domain cast with sequences
firrtl.circuit "UnsafeSequenceCast" {
  firrtl.module @UnsafeSequenceCast(
    in %A: !firrtl.domain,
    in %B: !firrtl.domain,
    in %C: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A, %B : !firrtl.uint<1>
    %0 = firrtl.unsafe_domain_cast %a domains %C : !firrtl.uint<1>
    %c = firrtl.wire domains %C : !firrtl.uint<1>
    firrtl.matchingconnect %c, %0 : !firrtl.uint<1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// FAILING TESTS - Expected to fail with errors
//===----------------------------------------------------------------------===//

// Test case 9: Illegal domain crossing - both matchingconnect and connect should fail
firrtl.circuit "IllegalDomainCrossing" {
  firrtl.module @IllegalDomainCrossing(
    // expected-note@below {{source is in domain defined here}}
    in %A: !firrtl.domain,
    // expected-note@below {{destination is in domain defined here}}
    in %B: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A : !firrtl.uint<1>
    %b = firrtl.wire domains %B : !firrtl.uint<1>

    // expected-error @below {{illegal domain crossing in connect operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>

    // expected-error @below {{illegal domain crossing in connect operation}}
    firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Test case 10: Multiple domain crossings
firrtl.circuit "MultipleDomainCrossings" {
  firrtl.module @MultipleDomainCrossings(
    // expected-note@below {{source is in domain defined here}}
    in %A: !firrtl.domain,
    // expected-note@below {{destination is in domain defined here}}
    // expected-note@below {{source is in domain defined here}}
    in %B: !firrtl.domain,
    // expected-note@below {{destination is in domain defined here}}
    in %C: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A : !firrtl.uint<1>
    %b = firrtl.wire domains %B : !firrtl.uint<1>
    %c = firrtl.wire domains %C : !firrtl.uint<1>

    // expected-error@below {{illegal domain crossing in connect operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>

    // expected-error@below {{illegal domain crossing in connect operation}}
    firrtl.matchingconnect %c, %b : !firrtl.uint<1>
  }
}

// -----

// Test case 11: Domain sequence mismatch - different lengths
firrtl.circuit "SequenceLengthMismatch" {
  firrtl.module @SequenceLengthMismatch(
    // expected-note@below {{source (domain 1 of 2) is in domain defined here}}
    in %A: !firrtl.domain,
    // expected-note@below {{destination is in domain defined here}}
    // expected-note@below {{source (domain 2 of 2) is in domain defined here}}
    in %B: !firrtl.domain
  ) {
    %a = firrtl.wire domains %A, %B : !firrtl.uint<1>
    %b = firrtl.wire domains %B : !firrtl.uint<1>

    // expected-error@below {{illegal domain crossing in connect operation}}
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
}
