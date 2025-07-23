//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines NPN (Negation-Permutation-Negation) canonical forms
// and binary truth tables for boolean function representation and equivalence
// checking in combinational logic optimization.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_NPNCLASS_H
#define CIRCT_SUPPORT_NPNCLASS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {

/// Represents a boolean function as a truth table.
///
/// A truth table stores the output values for all possible input combinations
/// of a boolean function. For a function with n inputs and m outputs, the
/// truth table contains 2^n entries, each with m output bits.
///
/// Example: For a 2-input AND gate:
/// - Input 00 -> Output 0
/// - Input 01 -> Output 0
/// - Input 10 -> Output 0
/// - Input 11 -> Output 1
/// This would be stored as the bit pattern 0001 in the truth table.
struct BinaryTruthTable {
  unsigned numInputs;  ///< Number of inputs for this boolean function
  unsigned numOutputs; ///< Number of outputs for this boolean function
  llvm::APInt table;   ///< Truth table data as a packed bit vector

  /// Default constructor creates an empty truth table.
  BinaryTruthTable() : numInputs(0), numOutputs(0), table(1, 0) {}

  /// Constructor for a truth table with given dimensions and evaluation data.
  BinaryTruthTable(unsigned numInputs, unsigned numOutputs,
                   const llvm::APInt &table)
      : numInputs(numInputs), numOutputs(numOutputs), table(table) {
    assert(table.getBitWidth() == (1u << numInputs) * numOutputs &&
           "Truth table size mismatch");
  }

  /// Constructor for a truth table with given dimensions, initialized to zero.
  BinaryTruthTable(unsigned numInputs, unsigned numOutputs)
      : numInputs(numInputs), numOutputs(numOutputs),
        table((1u << numInputs) * numOutputs, 0) {}

  /// Get the output value for a given input combination.
  llvm::APInt getOutput(const llvm::APInt &input) const;

  /// Set the output value for a given input combination.
  void setOutput(const llvm::APInt &input, const llvm::APInt &output);

  /// Apply input permutation to create a new truth table.
  /// This reorders the input variables according to the given permutation.
  BinaryTruthTable applyPermutation(ArrayRef<unsigned> permutation) const;

  /// Apply input negation to create a new truth table.
  /// This negates selected input variables based on the mask.
  BinaryTruthTable applyInputNegation(unsigned mask) const;

  /// Apply output negation to create a new truth table.
  /// This negates selected output variables based on the mask.
  BinaryTruthTable applyOutputNegation(unsigned negation) const;

  /// Check if this truth table is lexicographically smaller than another.
  /// Used for canonical ordering of truth tables.
  bool isLexicographicallySmaller(const BinaryTruthTable &other) const;

  /// Equality comparison for truth tables.
  bool operator==(const BinaryTruthTable &other) const;

  /// Debug dump method for truth tables.
  void dump(llvm::raw_ostream &os = llvm::errs()) const;
};

/// Represents the canonical form of a boolean function under NPN equivalence.
///
/// NPN (Negation-Permutation-Negation) equivalence considers two boolean
/// functions equivalent if one can be obtained from the other by:
/// 1. Negating some inputs (pre-negation)
/// 2. Permuting the inputs
/// 3. Negating some outputs (post-negation)
///
/// Example: The function ab+c is NPN equivalent to !a + b(!c) since we can
/// transform the first function into the second by negating inputs 'a' and 'c',
/// then reordering the inputs appropriately.
///
/// This canonical form is used to efficiently match cuts against library
/// patterns, as functions in the same NPN class can be implemented by the
/// same circuit with appropriate input/output inversions.
struct NPNClass {
  BinaryTruthTable truthTable;                  ///< Canonical truth table
  llvm::SmallVector<unsigned> inputPermutation; ///< Input permutation applied
  unsigned inputNegation = 0;                   ///< Input negation mask
  unsigned outputNegation = 0;                  ///< Output negation mask

  /// Default constructor creates an empty NPN class.
  NPNClass() = default;

  /// Constructor from a truth table.
  NPNClass(const BinaryTruthTable &tt) : truthTable(tt) {}

  NPNClass(const BinaryTruthTable &tt, llvm::SmallVector<unsigned> inputPerm,
           unsigned inputNeg, unsigned outputNeg)
      : truthTable(tt), inputPermutation(std::move(inputPerm)),
        inputNegation(inputNeg), outputNegation(outputNeg) {}

  /// Compute the canonical NPN form for a given truth table.
  ///
  /// This method exhaustively tries all possible input permutations and
  /// negations to find the lexicographically smallest canonical form.
  ///
  /// FIXME: Currently we are using exact canonicalization which doesn't scale
  /// well. For larger truth tables, semi-canonical forms should be used
  /// instead.
  static NPNClass computeNPNCanonicalForm(const BinaryTruthTable &tt);

  /// Get input permutation from this NPN class to another equivalent NPN class.
  ///
  /// When two NPN classes are equivalent, they may have different input
  /// permutations. This function computes a permutation that allows
  /// transforming input indices from the target NPN class to input indices of
  /// this NPN class.
  ///
  /// Returns a permutation vector where result[i] gives the input index in this
  /// NPN class that corresponds to input i in the target NPN class.
  ///
  /// Example: If this has permutation [2,0,1] and target has [1,2,0],
  /// the mapping allows connecting target inputs to this inputs correctly.
  void getInputPermutation(const NPNClass &targetNPN,
                           llvm::SmallVectorImpl<unsigned> &permutation) const;

  /// Equality comparison for NPN classes.
  bool equivalentOtherThanPermutation(const NPNClass &other) const {
    return truthTable == other.truthTable &&
           inputNegation == other.inputNegation &&
           outputNegation == other.outputNegation;
  }

  bool isLexicographicallySmaller(const NPNClass &other) const {
    if (truthTable.table != other.truthTable.table)
      return truthTable.isLexicographicallySmaller(other.truthTable);
    if (inputNegation != other.inputNegation)
      return inputNegation < other.inputNegation;
    return outputNegation < other.outputNegation;
  }

  /// Debug dump method for NPN classes.
  void dump(llvm::raw_ostream &os = llvm::errs()) const;
};

} // namespace circt

#endif // CIRCT_SUPPORT_NPNCLASS_H
