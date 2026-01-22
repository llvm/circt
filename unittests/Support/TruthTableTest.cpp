//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/TruthTable.h"
#include "llvm/ADT/APInt.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace llvm;

TEST(BinaryTruthTableTest, BasicConstruction) {
  // Test default constructor
  BinaryTruthTable defaultTT;
  EXPECT_EQ(defaultTT.numInputs, 0u);
  EXPECT_EQ(defaultTT.numOutputs, 0u);

  // Test parameterized constructor with zero initialization
  BinaryTruthTable zeroTT(2, 1);
  EXPECT_EQ(zeroTT.numInputs, 2u);
  EXPECT_EQ(zeroTT.numOutputs, 1u);
  EXPECT_EQ(zeroTT.table.getBitWidth(), 4u); // 2^2 * 1
  EXPECT_EQ(zeroTT.table.getZExtValue(), 0u);
}

TEST(BinaryTruthTableTest, GetSetOutput) {
  BinaryTruthTable tt(2, 1);

  // Test setting and getting output for different inputs
  APInt input00(2, 0); // 00
  APInt input01(2, 1); // 01
  APInt input10(2, 2); // 10
  APInt input11(2, 3); // 11
  APInt output0(1, 0);
  APInt output1(1, 1);

  // Set outputs for AND gate truth table
  tt.setOutput(input00, output0);
  tt.setOutput(input01, output0);
  tt.setOutput(input10, output0);
  tt.setOutput(input11, output1);

  // Verify outputs
  EXPECT_EQ(tt.getOutput(input00), output0);
  EXPECT_EQ(tt.getOutput(input01), output0);
  EXPECT_EQ(tt.getOutput(input10), output0);
  EXPECT_EQ(tt.getOutput(input11), output1);
}

TEST(BinaryTruthTableTest, Permutation) {
  // Create a truth table for a 2-input function: f(a,b) = a & !b
  BinaryTruthTable original(2, 1);
  APInt output0(1, 0);
  APInt output1(1, 1);

  // Set truth table: f(0,0)=0, f(0,1)=0, f(1,0)=1, f(1,1)=0
  original.setOutput(APInt(2, 0), output0); // f(0,0) = 0
  original.setOutput(APInt(2, 1), output0); // f(0,1) = 0
  original.setOutput(APInt(2, 2), output1); // f(1,0) = 1
  original.setOutput(APInt(2, 3), output0); // f(1,1) = 0

  // Apply permutation [1, 0] (swap inputs)
  SmallVector<unsigned> permutation = {1, 0};
  BinaryTruthTable permuted = original.applyPermutation(permutation);

  // After permutation, the function becomes f(b,a) = b & !a
  // So: f(0,0)=0, f(0,1)=1, f(1,0)=0, f(1,1)=0
  EXPECT_EQ(permuted.getOutput(APInt(2, 0)), output0); // f(0,0) = 0
  EXPECT_EQ(permuted.getOutput(APInt(2, 1)), output1); // f(0,1) = 1
  EXPECT_EQ(permuted.getOutput(APInt(2, 2)), output0); // f(1,0) = 0
  EXPECT_EQ(permuted.getOutput(APInt(2, 3)), output0); // f(1,1) = 0
}

TEST(BinaryTruthTableTest, InputNegation) {
  // Create a truth table for AND gate: f(a,b) = a & b
  BinaryTruthTable original(2, 1);
  APInt output0(1, 0);
  APInt output1(1, 1);

  // Set AND truth table
  original.setOutput(APInt(2, 0), output0); // f(0,0) = 0
  original.setOutput(APInt(2, 1), output0); // f(0,1) = 0
  original.setOutput(APInt(2, 2), output0); // f(1,0) = 0
  original.setOutput(APInt(2, 3), output1); // f(1,1) = 1

  // Apply input negation mask 1 (negate the second)
  BinaryTruthTable negated = original.applyInputNegation(1);

  APInt result00 = negated.getOutput(APInt(2, 0));
  APInt result01 = negated.getOutput(APInt(2, 1));
  APInt result10 = negated.getOutput(APInt(2, 2));
  APInt result11 = negated.getOutput(APInt(2, 3));

  // For input negation mask 1 (negates the last input), the mapping is:
  // clang-format off
  // g(0,0) gets the output from f(0,!0) = f(0,1) = 0
  // g(0,1) gets the output from f(0,!1) = f(0,0) = 0
  // g(1,0) gets the output from f(1,!0) = f(1,1) = 1
  // g(1,1) gets the output from f(1,!1) = f(1,0) = 0
  // clang-format on
  EXPECT_EQ(result00, output0); // g(0,0) = 0
  EXPECT_EQ(result01, output0); // g(0,1) = 0
  EXPECT_EQ(result10, output1); // g(1,0) = 1
  EXPECT_EQ(result11, output0); // g(1,1) = 0
}

TEST(BinaryTruthTableTest, OutputNegation) {
  // Create a truth table for AND gate
  BinaryTruthTable original(2, 1);
  APInt output0(1, 0);
  APInt output1(1, 1);

  original.setOutput(APInt(2, 0), output0);
  original.setOutput(APInt(2, 1), output0);
  original.setOutput(APInt(2, 2), output0);
  original.setOutput(APInt(2, 3), output1);

  // Apply output negation (creates NAND gate)
  BinaryTruthTable negated = original.applyOutputNegation(1);

  EXPECT_EQ(negated.getOutput(APInt(2, 0)), output1);
  EXPECT_EQ(negated.getOutput(APInt(2, 1)), output1);
  EXPECT_EQ(negated.getOutput(APInt(2, 2)), output1);
  EXPECT_EQ(negated.getOutput(APInt(2, 3)), output0);
}

TEST(BinaryTruthTableTest, Equality) {
  BinaryTruthTable tt1(2, 1);
  BinaryTruthTable tt2(2, 1);
  BinaryTruthTable tt3(3, 1); // Different size

  // Both empty truth tables should be equal
  EXPECT_TRUE(tt1 == tt2);
  EXPECT_FALSE(tt1 == tt3);

  // Set same values
  tt1.setOutput(APInt(2, 3), APInt(1, 1));
  tt2.setOutput(APInt(2, 3), APInt(1, 1));
  EXPECT_TRUE(tt1 == tt2);

  // Set different values
  tt2.setOutput(APInt(2, 2), APInt(1, 1));
  EXPECT_FALSE(tt1 == tt2);
}

TEST(NPNClassTest, BasicConstruction) {
  // Test default constructor
  NPNClass defaultNPN;
  EXPECT_EQ(defaultNPN.inputNegation, 0u);
  EXPECT_EQ(defaultNPN.outputNegation, 0u);
  EXPECT_TRUE(defaultNPN.inputPermutation.empty());

  // Test constructor from truth table
  BinaryTruthTable tt(2, 1);
  NPNClass npn(tt);
  EXPECT_EQ(npn.truthTable, tt);
  EXPECT_EQ(npn.inputNegation, 0u);
  EXPECT_EQ(npn.outputNegation, 0u);
}

TEST(NPNClassTest, CanonicalFormSimple) {
  // Create truth table for XOR: f(a,b) = a ^ b
  BinaryTruthTable xorTT(2, 1);
  xorTT.setOutput(APInt(2, 0), APInt(1, 0)); // 0 ^ 0 = 0
  xorTT.setOutput(APInt(2, 1), APInt(1, 1)); // 0 ^ 1 = 1
  xorTT.setOutput(APInt(2, 2), APInt(1, 1)); // 1 ^ 0 = 1
  xorTT.setOutput(APInt(2, 3), APInt(1, 0)); // 1 ^ 1 = 0

  NPNClass canonical = NPNClass::computeNPNCanonicalForm(xorTT);

  // XOR is symmetric, so canonical form should be predictable
  EXPECT_EQ(canonical.truthTable.numInputs, 2u);
  EXPECT_EQ(canonical.truthTable.numOutputs, 1u);
  EXPECT_EQ(canonical.inputPermutation.size(), 2u);
}

TEST(NPNClassTest, EquivalenceOtherThanPermutation) {
  BinaryTruthTable tt1(2, 1);
  BinaryTruthTable tt2(2, 1);

  NPNClass npn1(tt1, {0, 1}, 0, 0);
  NPNClass npn2(tt2, {1, 0}, 0, 0); // Different permutation
  NPNClass npn3(tt1, {0, 1}, 1, 0); // Different input negation

  EXPECT_TRUE(npn1.equivalentOtherThanPermutation(npn2));
  EXPECT_FALSE(npn1.equivalentOtherThanPermutation(npn3));
}

TEST(NPNClassTest, InputMapping) {
  BinaryTruthTable tt(3, 1);
  NPNClass npn1(tt, {2, 0, 1}, 0, 0); // Permutation [2,0,1]
  NPNClass npn2(tt, {1, 2, 0}, 0, 0); // Permutation [1,2,0]

  SmallVector<unsigned> permutation;
  npn1.getInputPermutation(npn2, permutation);
  // Verify the mapping is correct
  EXPECT_EQ(permutation.size(), 3u);

  // For each target input position i, mapping[i] should give us the input
  // position in npn1 that corresponds to the same canonical position
  for (unsigned i = 0; i < 3; ++i) {
    // Target input i maps to canonical position npn2.inputPermutation[i]
    unsigned targetCanonicalPos = npn2.inputPermutation[i];
    // npn1's input mapping[i] should map to the same canonical position
    unsigned npn1CanonicalPos = npn1.inputPermutation[permutation[i]];
    EXPECT_EQ(targetCanonicalPos, npn1CanonicalPos);
  }
}

TEST(NPNClassTest, LexicographicalOrdering) {
  BinaryTruthTable tt1(2, 1, APInt(4, 0x5)); // 0101
  BinaryTruthTable tt2(2, 1, APInt(4, 0x6)); // 0110

  NPNClass npn1(tt1);
  NPNClass npn2(tt2);

  // 0x5 < 0x6, so npn1 should be lexicographically smaller
  EXPECT_TRUE(npn1.isLexicographicallySmaller(npn2));
  EXPECT_FALSE(npn2.isLexicographicallySmaller(npn1));
}

TEST(NPNClassTest, Commutativity) {
  // Create truth table for AND gate
  BinaryTruthTable andTT(2, 1);
  andTT.setOutput(APInt(2, 0), APInt(1, 0));
  andTT.setOutput(APInt(2, 1), APInt(1, 0));
  andTT.setOutput(APInt(2, 2), APInt(1, 0));
  andTT.setOutput(APInt(2, 3), APInt(1, 1));

  // Create swapped version (should be same canonical form since AND is
  // commutative)
  BinaryTruthTable andTTSwapped(2, 1);
  andTTSwapped.setOutput(APInt(2, 0), APInt(1, 0)); // f(0,0) = 0
  andTTSwapped.setOutput(APInt(2, 1), APInt(1, 0)); // f(0,1) = 0
  andTTSwapped.setOutput(APInt(2, 2), APInt(1, 0)); // f(1,0) = 0
  andTTSwapped.setOutput(APInt(2, 3), APInt(1, 1)); // f(1,1) = 1

  NPNClass canonical1 = NPNClass::computeNPNCanonicalForm(andTT);
  NPNClass canonical2 = NPNClass::computeNPNCanonicalForm(andTTSwapped);

  // Should have same canonical form
  EXPECT_TRUE(canonical1.equivalentOtherThanPermutation(canonical2));
}

TEST(BinaryTruthTableTest, MultiBitOutput) {
  // Test 2-input, 2-output function: f(a,b) = (a&b, a|b)
  BinaryTruthTable tt(2, 2);

  // Set outputs for each input combination
  // f(0,0) = (0,0)
  tt.setOutput(APInt(2, 0), APInt(2, 0));
  // f(0,1) = (0,1)
  tt.setOutput(APInt(2, 1), APInt(2, 1));
  // f(1,0) = (0,1)
  tt.setOutput(APInt(2, 2), APInt(2, 1));
  // f(1,1) = (1,1)
  tt.setOutput(APInt(2, 3), APInt(2, 3));

  // Verify outputs
  EXPECT_EQ(tt.getOutput(APInt(2, 0)), APInt(2, 0));
  EXPECT_EQ(tt.getOutput(APInt(2, 1)), APInt(2, 1));
  EXPECT_EQ(tt.getOutput(APInt(2, 2)), APInt(2, 1));
  EXPECT_EQ(tt.getOutput(APInt(2, 3)), APInt(2, 3));

  // Test table bit width
  EXPECT_EQ(tt.table.getBitWidth(), 8u); // 2^2 * 2 = 8 bits
}

TEST(BinaryTruthTableTest, MultiBitOutputPermutation) {
  // Create 2-input, 2-output function: f(a,b) = (a&b, a^b)
  BinaryTruthTable original(2, 2);

  // f(0,0) = (0,0)
  original.setOutput(APInt(2, 0), APInt(2, 0));
  // f(0,1) = (0,1)
  original.setOutput(APInt(2, 1), APInt(2, 1));
  // f(1,0) = (0,1)
  original.setOutput(APInt(2, 2), APInt(2, 1));
  // f(1,1) = (1,0)
  original.setOutput(APInt(2, 3), APInt(2, 2));

  // Apply permutation [1,0] (swap inputs)
  SmallVector<unsigned> permutation = {1, 0};
  BinaryTruthTable permuted = original.applyPermutation(permutation);

  // After swapping inputs, f(b,a) = (b&a, b^a)
  // f(0,0) = (0,0)
  EXPECT_EQ(permuted.getOutput(APInt(2, 0)), APInt(2, 0));
  // f(0,1) = (0,1)
  EXPECT_EQ(permuted.getOutput(APInt(2, 1)), APInt(2, 1));
  // f(1,0) = (0,1)
  EXPECT_EQ(permuted.getOutput(APInt(2, 2)), APInt(2, 1));
  // f(1,1) = (1,0)
  EXPECT_EQ(permuted.getOutput(APInt(2, 3)), APInt(2, 2));
}

TEST(BinaryTruthTableTest, MultiBitOutputNegation) {
  // Create 2-input, 2-output function
  BinaryTruthTable original(2, 2);

  // Set some non-trivial outputs
  original.setOutput(APInt(2, 0), APInt(2, 0)); // 00
  original.setOutput(APInt(2, 1), APInt(2, 1)); // 01
  original.setOutput(APInt(2, 2), APInt(2, 2)); // 10
  original.setOutput(APInt(2, 3), APInt(2, 3)); // 11

  // Apply output negation mask 1 (negate first output bit)
  BinaryTruthTable negated = original.applyOutputNegation(1);

  // Check that only the first bit of each output is negated
  EXPECT_EQ(negated.getOutput(APInt(2, 0)), APInt(2, 1)); // 00 -> 01
  EXPECT_EQ(negated.getOutput(APInt(2, 1)), APInt(2, 0)); // 01 -> 00
  EXPECT_EQ(negated.getOutput(APInt(2, 2)), APInt(2, 3)); // 10 -> 11
  EXPECT_EQ(negated.getOutput(APInt(2, 3)), APInt(2, 2)); // 11 -> 10

  // Apply output negation mask 2 (negate second output bit)
  BinaryTruthTable negated2 = original.applyOutputNegation(2);

  EXPECT_EQ(negated2.getOutput(APInt(2, 0)), APInt(2, 2)); // 00 -> 10
  EXPECT_EQ(negated2.getOutput(APInt(2, 1)), APInt(2, 3)); // 01 -> 11
  EXPECT_EQ(negated2.getOutput(APInt(2, 2)), APInt(2, 0)); // 10 -> 00
  EXPECT_EQ(negated2.getOutput(APInt(2, 3)), APInt(2, 1)); // 11 -> 01

  // Apply output negation mask 3 (negate both output bits)
  BinaryTruthTable negated3 = original.applyOutputNegation(3);

  EXPECT_EQ(negated3.getOutput(APInt(2, 0)), APInt(2, 3)); // 00 -> 11
  EXPECT_EQ(negated3.getOutput(APInt(2, 1)), APInt(2, 2)); // 01 -> 10
  EXPECT_EQ(negated3.getOutput(APInt(2, 2)), APInt(2, 1)); // 10 -> 01
  EXPECT_EQ(negated3.getOutput(APInt(2, 3)), APInt(2, 0)); // 11 -> 00
}

TEST(NPNClassTest, MultiBitOutputCanonical) {
  // Create a 2-input, 2-output function: f(a,b) = (a&b, a|b)
  BinaryTruthTable tt(2, 2);

  tt.setOutput(APInt(2, 0), APInt(2, 0)); // f(0,0) = (0,0)
  tt.setOutput(APInt(2, 1), APInt(2, 1)); // f(0,1) = (0,1)
  tt.setOutput(APInt(2, 2), APInt(2, 1)); // f(1,0) = (0,1)
  tt.setOutput(APInt(2, 3), APInt(2, 3)); // f(1,1) = (1,1)

  NPNClass canonical = NPNClass::computeNPNCanonicalForm(tt);

  // Should have correct dimensions
  EXPECT_EQ(canonical.truthTable.numInputs, 2u);
  EXPECT_EQ(canonical.truthTable.numOutputs, 2u);
  EXPECT_EQ(canonical.inputPermutation.size(), 2u);

  // The canonical form should be well-defined and reproducible
  NPNClass canonical2 = NPNClass::computeNPNCanonicalForm(tt);
  EXPECT_TRUE(canonical.equivalentOtherThanPermutation(canonical2));
}

TEST(NPNClassTest, MultiBitOutputEquivalence) {
  // Create two equivalent 2-input, 2-output functions with different input
  // orderings
  BinaryTruthTable tt1(2, 2);
  BinaryTruthTable tt2(2, 2);

  // tt1: f(a,b) = (a&b, a^b)
  tt1.setOutput(APInt(2, 0), APInt(2, 0)); // f(0,0) = (0,0)
  tt1.setOutput(APInt(2, 1), APInt(2, 1)); // f(0,1) = (0,1)
  tt1.setOutput(APInt(2, 2), APInt(2, 1)); // f(1,0) = (0,1)
  tt1.setOutput(APInt(2, 3), APInt(2, 2)); // f(1,1) = (1,0)

  // tt2: g(a,b) = f(b,a) = (b&a, b^a) - same function with swapped inputs
  tt2.setOutput(APInt(2, 0), APInt(2, 0)); // g(0,0) = (0,0)
  tt2.setOutput(APInt(2, 1), APInt(2, 1)); // g(0,1) = (0,1)
  tt2.setOutput(APInt(2, 2), APInt(2, 1)); // g(1,0) = (0,1)
  tt2.setOutput(APInt(2, 3), APInt(2, 2)); // g(1,1) = (1,0)

  NPNClass canonical1 = NPNClass::computeNPNCanonicalForm(tt1);
  NPNClass canonical2 = NPNClass::computeNPNCanonicalForm(tt2);

  // Since the functions are equivalent under permutation, their canonical forms
  // should be equivalent (though permutations might differ)
  EXPECT_TRUE(canonical1.equivalentOtherThanPermutation(canonical2));
}

TEST(NPNClassTest, MultiBitOutputMapping) {
  // Test input mapping with multi-bit outputs
  BinaryTruthTable tt(2, 3); // 2 inputs, 3 outputs

  NPNClass npn1(tt, {0, 1}, 0, 0); // Identity permutation
  NPNClass npn2(tt, {1, 0}, 0, 0); // Swapped permutation

  SmallVector<unsigned> permutation;
  npn1.getInputPermutation(npn2, permutation);
  EXPECT_EQ(permutation.size(), 2u);

  // Verify the mapping relationship
  for (unsigned i = 0; i < 2; ++i) {
    unsigned targetCanonicalPos = npn2.inputPermutation[i];
    unsigned npn1CanonicalPos = npn1.inputPermutation[permutation[i]];
    EXPECT_EQ(targetCanonicalPos, npn1CanonicalPos);
  }
}

TEST(NPNClassTest, MultiBitOutputLexicographical) {
  // Test lexicographical ordering with multi-bit outputs
  BinaryTruthTable tt1(2, 2, APInt(8, 0x12)); // 00010010
  BinaryTruthTable tt2(2, 2, APInt(8, 0x34)); // 00110100

  NPNClass npn1(tt1);
  NPNClass npn2(tt2);

  // 0x12 < 0x34, so npn1 should be lexicographically smaller
  EXPECT_TRUE(npn1.isLexicographicallySmaller(npn2));
  EXPECT_FALSE(npn2.isLexicographicallySmaller(npn1));
}

//===----------------------------------------------------------------------===//
// Variable Mask Tests
//===----------------------------------------------------------------------===//

TEST(VarMaskTest, TwoVariables) {
  // Test masks for 2 variables (4-bit truth table)
  // Variable ordering: a is MSB (var 1), b is LSB (var 0)

  // Mask for b=0 (LSB): positions where b=0 are 00,10 -> 0b0101
  APInt maskB0 = createVarMask(2, 0, false);
  EXPECT_EQ(maskB0, APInt(4, 0b0101));

  // Mask for b=1 (LSB): positions where b=1 are 01,11 -> 0b1010
  APInt maskB1 = createVarMask(2, 0, true);
  EXPECT_EQ(maskB1, APInt(4, 0b1010));

  // Mask for a=0 (MSB): positions where a=0 are 00,01 -> 0b0011
  APInt maskA0 = createVarMask(2, 1, false);
  EXPECT_EQ(maskA0, APInt(4, 0b0011));
  // Mask for a=1 (MSB): positions where a=1 are 10,11 -> 0b1100
  APInt maskA1 = createVarMask(2, 1, true);
  EXPECT_EQ(maskA1, APInt(4, 0b1100));
}

TEST(VarMaskTest, ThreeVariables) {
  // Test masks for 3 variables (8-bit truth table)
  // Variable ordering: a is bit 2, b is bit 1, c is bit 0 (LSB)

  // Mask for c=1: 0b10101010
  APInt maskC1 = createVarMask(3, 0, true);
  EXPECT_EQ(maskC1, APInt(8, 0b10101010));

  // Mask for b=1: 0b11001100
  APInt maskB1 = createVarMask(3, 1, true);
  EXPECT_EQ(maskB1, APInt(8, 0b11001100));

  // Mask for a=1: 0b11110000
  APInt maskA1 = createVarMask(3, 2, true);
  EXPECT_EQ(maskA1, APInt(8, 0b11110000));

  // Verify complementary property: mask0 | mask1 should be all 1s
  APInt maskC0 = createVarMask(3, 0, false);
  EXPECT_EQ(maskC0 | maskC1, APInt(8, 0b11111111));
}

TEST(VarMaskTest, MaskProperties) {
  // Test that positive and negative masks are complementary
  for (unsigned numVars : {2, 3, 5, 8}) {
    for (unsigned var = 0; var < numVars; ++var) {
      APInt mask0 = createVarMask(numVars, var, false);
      APInt mask1 = createVarMask(numVars, var, true);

      // Masks should be disjoint
      EXPECT_TRUE((mask0 & mask1).isZero());

      // Masks should cover all bits
      APInt allOnes = APInt::getAllOnes(1u << numVars);
      EXPECT_EQ(mask0 | mask1, allOnes);
    }
  }
}

//===----------------------------------------------------------------------===//
// Cofactor Tests
//===----------------------------------------------------------------------===//

TEST(CofactorTest, BasicIdentity) {
  // Test f(a,b) = a (truth table: 0b1100)
  APInt tt(4, 0b1100);

  // Cofactor with respect to b (var 0): should not affect result
  auto [cofB0, cofB1] = computeCofactors(tt, 2, 0);
  EXPECT_EQ(cofB0, cofB1); // Function doesn't depend on b
  EXPECT_EQ(cofB0, tt);    // Should match original

  // Cofactor with respect to a (var 1): should differ
  auto [cofA0, cofA1] = computeCofactors(tt, 2, 1);
  // When a=0, all outputs are 0. When a=1, all outputs are 1
  EXPECT_EQ(cofA0, APInt(4, 0b0000)); // All zeros
  EXPECT_EQ(cofA1, APInt(4, 0b1111)); // All ones
}

TEST(CofactorTest, AndFunction) {
  // Test f(a,b) = a & b (truth table: 0b1000)
  APInt tt(4, 0b1000);

  auto [cofA0, cofA1] = computeCofactors(tt, 2, 1);
  // f|a=0 = 0, f|a=1 = b
  EXPECT_EQ(cofA0, APInt(4, 0));
  EXPECT_EQ(cofA1, createVarMask(2, 0, true)); // Equals b
}

TEST(CofactorTest, XorFunction) {
  // Test f(a,b) = a XOR b (truth table: 0b0110)
  APInt tt(4, 0b0110);

  auto [cofA0, cofA1] = computeCofactors(tt, 2, 1);
  // f|a=0 = b, f|a=1 = !b
  EXPECT_EQ(cofA0, createVarMask(2, 0, true));  // Equals b
  EXPECT_EQ(cofA1, createVarMask(2, 0, false)); // Equals !b
}

TEST(CofactorTest, ShannonExpansionProperty) {
  // Test Shannon expansion property: f = !x * f|x=0 + x * f|x=1
  // for various fixed truth tables

  struct TestCase {
    unsigned numVars;
    std::vector<uint64_t> ttData;
  };

  std::vector<TestCase> testCases = {
      // 4 variables: f = a & b & c & d (all inputs AND)
      {4, {0x8000}},

      // 5 variables: f = parity function (XOR of all inputs)
      {5, {0x69969669}},

      // 6 variables: majority function (at least 4 inputs are 1)
      {6, {0xFFFFFFFE80000000ULL}},

      // 7 variables: f = x0 | x6 (LSB OR MSB)
      {7, {0xAAAAAAAAAAAAAAAAULL, 0xFFFFFFFFFFFFFFFFULL}},

      // 8 variables: f = (x0 & x7) | (x1 & x6)
      {8,
       {0x8080808080808080ULL, 0x8080808080808080ULL, 0x8080808080808080ULL,
        0x8080808080808080ULL}}};

  for (const auto &tc : testCases) {
    unsigned ttSize = 1u << tc.numVars;
    APInt tt(ttSize, ArrayRef<uint64_t>(tc.ttData));

    // Test all variables
    for (unsigned varToTest = 0; varToTest < tc.numVars; ++varToTest) {
      auto [cof0, cof1] = computeCofactors(tt, tc.numVars, varToTest);
      APInt mask0 = createVarMask(tc.numVars, varToTest, false);
      APInt mask1 = createVarMask(tc.numVars, varToTest, true);

      // Verify Shannon expansion
      APInt reconstructed = (mask0 & cof0) | (mask1 & cof1);
      EXPECT_EQ(reconstructed, tt)
          << "Failed for " << tc.numVars << " vars, var " << varToTest;
    }
  }
}
