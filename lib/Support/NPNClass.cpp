//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements NPN (Negation-Permutation-Negation) canonical forms
// and binary truth tables for boolean function representation and equivalence
// checking in combinational logic optimization.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/NPNClass.h"

#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cassert>

using namespace circt;

//===----------------------------------------------------------------------===//
// BinaryTruthTable
//===----------------------------------------------------------------------===//

llvm::APInt BinaryTruthTable::getOutput(const llvm::APInt &input) const {
  assert(input.getBitWidth() == numInputs && "Input width mismatch");
  return table.extractBits(numOutputs, input.getZExtValue() * numOutputs);
}

void BinaryTruthTable::setOutput(const llvm::APInt &input,
                                 const llvm::APInt &output) {
  assert(input.getBitWidth() == numInputs && "Input width mismatch");
  assert(output.getBitWidth() == numOutputs && "Output width mismatch");
  unsigned offset = input.getZExtValue() * numOutputs;
  for (unsigned i = 0; i < numOutputs; ++i)
    table.setBitVal(offset + i, output[i]);
}

BinaryTruthTable
BinaryTruthTable::applyPermutation(ArrayRef<unsigned> permutation) const {
  assert(permutation.size() == numInputs && "Permutation size mismatch");
  BinaryTruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);
    llvm::APInt permutedInput(numInputs, 0);

    // Apply permutation
    for (unsigned j = 0; j < numInputs; ++j)
      permutedInput.setBitVal(j, input[permutation[j]]);

    result.setOutput(permutedInput, getOutput(input));
  }

  return result;
}

BinaryTruthTable BinaryTruthTable::applyInputNegation(unsigned mask) const {
  BinaryTruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);
    llvm::APInt negatedInput(numInputs, 0);

    // Apply negation
    for (unsigned j = 0; j < numInputs; ++j)
      negatedInput.setBitVal(j, (mask & (1u << j)) ? !input[j] : input[j]);

    result.setOutput(negatedInput, getOutput(input));
  }

  return result;
}

BinaryTruthTable
BinaryTruthTable::applyOutputNegation(unsigned negation) const {
  BinaryTruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);
    llvm::APInt output = getOutput(input);
    llvm::APInt negatedOutput(numOutputs, 0);

    // Apply negation
    for (unsigned j = 0; j < numOutputs; ++j)
      negatedOutput.setBitVal(j,
                              (negation & (1u << j)) ? !output[j] : output[j]);

    result.setOutput(input, negatedOutput);
  }

  return result;
}

bool BinaryTruthTable::isLexicographicallySmaller(
    const BinaryTruthTable &other) const {
  assert(numInputs == other.numInputs && numOutputs == other.numOutputs);
  return table.ult(other.table);
}

bool BinaryTruthTable::operator==(const BinaryTruthTable &other) const {
  return numInputs == other.numInputs && numOutputs == other.numOutputs &&
         table == other.table;
}

void BinaryTruthTable::dump(llvm::raw_ostream &os) const {
  os << "TruthTable(" << numInputs << " inputs, " << numOutputs << " outputs, "
     << "value=";
  os << table << ")\n";

  // Print header
  for (unsigned i = 0; i < numInputs; ++i) {
    os << "i" << i << " ";
  }
  for (unsigned i = 0; i < numOutputs; ++i) {
    os << "o" << i << " ";
  }
  os << "\n";

  // Print separator
  for (unsigned i = 0; i < numInputs + numOutputs; ++i) {
    os << "---";
  }
  os << "\n";

  // Print truth table rows
  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    // Print input values
    for (unsigned j = 0; j < numInputs; ++j) {
      os << ((i >> j) & 1) << "  ";
    }

    // Print output values
    llvm::APInt input(numInputs, i);
    llvm::APInt output = getOutput(input);
    os << "|";
    for (unsigned j = 0; j < numOutputs; ++j) {
      os << output[j] << "  ";
    }
    os << "\n";
  }
}

//===----------------------------------------------------------------------===//
// NPNClass
//===----------------------------------------------------------------------===//

namespace {
// Helper functions for permutation manipulation - kept as implementation
// details

/// Create an identity permutation of the given size.
/// Result[i] = i for all i in [0, size).
llvm::SmallVector<unsigned> identityPermutation(unsigned size) {
  llvm::SmallVector<unsigned> identity(size);
  for (unsigned i = 0; i < size; ++i)
    identity[i] = i;
  return identity;
}

/// Apply a permutation to a negation mask.
/// Given a negation mask and a permutation, returns a new mask where
/// the negation bits are reordered according to the permutation.
unsigned permuteNegationMask(unsigned negationMask,
                             ArrayRef<unsigned> permutation) {
  unsigned result = 0;
  for (unsigned i = 0; i < permutation.size(); ++i) {
    if (negationMask & (1u << i)) {
      result |= (1u << permutation[i]);
    }
  }
  return result;
}

/// Create the inverse of a permutation.
/// If permutation[i] = j, then inverse[j] = i.
llvm::SmallVector<unsigned> invertPermutation(ArrayRef<unsigned> permutation) {
  llvm::SmallVector<unsigned> inverse(permutation.size());
  for (unsigned i = 0; i < permutation.size(); ++i) {
    inverse[permutation[i]] = i;
  }
  return inverse;
}

} // anonymous namespace

llvm::SmallVector<unsigned>
NPNClass::getInputMappingTo(const NPNClass &targetNPN) const {
  assert(inputPermutation.size() == targetNPN.inputPermutation.size() &&
         "NPN classes must have the same number of inputs");
  assert(equivalentOtherThanPermutation(targetNPN) &&
         "NPN classes must be equivalent for input mapping");

  // Create inverse permutation for this NPN class
  auto thisInverse = invertPermutation(inputPermutation);

  // For each input position in the target NPN class, find the corresponding
  // input position in this NPN class
  llvm::SmallVector<unsigned> mapping(targetNPN.inputPermutation.size());
  for (unsigned i = 0; i < targetNPN.inputPermutation.size(); ++i) {
    // Target input i maps to canonical position targetNPN.inputPermutation[i]
    // We need the input in this NPN class that maps to the same canonical
    // position
    unsigned canonicalPos = targetNPN.inputPermutation[i];
    mapping[i] = thisInverse[canonicalPos];
  }

  return mapping;
}

NPNClass NPNClass::computeNPNCanonicalForm(const BinaryTruthTable &tt) {
  NPNClass canonical(tt);
  // Initialize permutation with identity
  canonical.inputPermutation = identityPermutation(tt.numInputs);
  assert(tt.numInputs <= 8 && "Too many inputs for input negation mask");
  // Try all possible tables and pick the lexicographically smallest.
  // FIXME: The time complexity is O(n! * 2^(n + m)) where n is the number
  // of inputs and m is the number of outputs. This is not scalable so
  // semi-canonical forms should be used instead.
  for (uint32_t negMask = 0; negMask < (1u << tt.numInputs); ++negMask) {
    BinaryTruthTable negatedTT = tt.applyInputNegation(negMask);

    // Try all possible permutations
    auto permutation = identityPermutation(tt.numInputs);

    do {
      BinaryTruthTable permutedTT = negatedTT.applyPermutation(permutation);

      // Permute the negation mask according to the permutation
      unsigned currentNegMask = permuteNegationMask(negMask, permutation);

      // Try all negation masks for the output
      for (unsigned outputNegMask = 0; outputNegMask < (1u << tt.numOutputs);
           ++outputNegMask) {
        // Apply output negation
        BinaryTruthTable candidateTT =
            permutedTT.applyOutputNegation(outputNegMask);

        // Create a new NPN class for the candidate
        NPNClass candidate(candidateTT, permutation, currentNegMask,
                           outputNegMask);

        // Check if this candidate is lexicographically smaller than the
        // current canonical form
        if (candidate.isLexicographicallySmaller(canonical))
          canonical = std::move(candidate);
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }

  return canonical;
}

void NPNClass::dump(llvm::raw_ostream &os) const {
  os << "NPNClass:\n";
  os << "  Input permutation: [";
  for (unsigned i = 0; i < inputPermutation.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << inputPermutation[i];
  }
  os << "]\n";
  os << "  Input negation: 0b";
  for (int i = inputPermutation.size() - 1; i >= 0; --i) {
    os << ((inputNegation >> i) & 1);
  }
  os << "\n";
  os << "  Output negation: 0b";
  for (int i = truthTable.numOutputs - 1; i >= 0; --i) {
    os << ((outputNegation >> i) & 1);
  }
  os << "\n";
  os << "  Canonical truth table:\n";
  truthTable.dump(os);
}
