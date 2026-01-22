//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements truth table utilities.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/TruthTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>

using namespace circt;
using llvm::APInt;

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
  assert(input.getBitWidth() < 32 && "Input width too large");
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
    llvm::APInt permutedInput = input;

    // Apply permutation
    for (unsigned j = 0; j < numInputs; ++j) {
      if (permutation[j] != j)
        permutedInput.setBitVal(j, input[permutation[j]]);
    }

    result.setOutput(permutedInput, getOutput(input));
  }

  return result;
}

BinaryTruthTable BinaryTruthTable::applyInputNegation(unsigned mask) const {
  BinaryTruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);

    // Apply negation using bitwise XOR
    llvm::APInt negatedInput = input ^ llvm::APInt(numInputs, mask);

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

    // Apply negation using bitwise XOR
    llvm::APInt negatedOutput = output ^ llvm::APInt(numOutputs, negation);

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

void NPNClass::getInputPermutation(
    const NPNClass &targetNPN,
    llvm::SmallVectorImpl<unsigned> &permutation) const {
  assert(inputPermutation.size() == targetNPN.inputPermutation.size() &&
         "NPN classes must have the same number of inputs");
  assert(equivalentOtherThanPermutation(targetNPN) &&
         "NPN classes must be equivalent for input mapping");

  // Create inverse permutation for this NPN class
  auto thisInverse = invertPermutation(inputPermutation);

  // For each input position in the target NPN class, find the corresponding
  // input position in this NPN class
  permutation.reserve(targetNPN.inputPermutation.size());
  for (unsigned i = 0; i < targetNPN.inputPermutation.size(); ++i) {
    // Target input i maps to canonical position targetNPN.inputPermutation[i]
    // We need the input in this NPN class that maps to the same canonical
    // position
    unsigned canonicalPos = targetNPN.inputPermutation[i];
    permutation.push_back(thisInverse[canonicalPos]);
  }
}

NPNClass NPNClass::computeNPNCanonicalForm(const BinaryTruthTable &tt) {
  NPNClass canonical(tt);
  // Initialize permutation with identity
  canonical.inputPermutation = identityPermutation(tt.numInputs);
  assert(tt.numInputs <= 8 && "Inputs are too large");
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
  for (int i = truthTable.numInputs - 1; i >= 0; --i) {
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

/// Precomputed masks for variables in truth tables up to 6 variables (64 bits).
///
/// In a truth table, bit position i represents minterm i, where the binary
/// representation of i gives the variable values. For example, with 3 variables
/// (a,b,c), bit 5 (binary 101) represents minterm a=1, b=0, c=1.
///
/// These masks identify which minterms have a particular variable value:
/// - Masks[0][var] = minterms where var=0 (for negative literal !var)
/// - Masks[1][var] = minterms where var=1 (for positive literal var)
static constexpr uint64_t kVarMasks[2][6] = {
    {0x5555555555555555ULL, 0x3333333333333333ULL, 0x0F0F0F0F0F0F0F0FULL,
     0x00FF00FF00FF00FFULL, 0x0000FFFF0000FFFFULL,
     0x00000000FFFFFFFFULL}, // var=0 masks
    {0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
     0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL,
     0xFFFFFFFF00000000ULL}, // var=1 masks
};

/// Create a mask for a variable in the truth table.
/// For positive=true: mask has 1s where var=1 in the truth table encoding
/// For positive=false: mask has 1s where var=0 in the truth table encoding
template <bool positive>
static APInt createVarMaskImpl(unsigned numVars, unsigned varIndex) {
  assert(numVars <= 64 && "Number of variables too large");
  assert(varIndex < numVars && "Variable index out of bounds");
  uint64_t numBits = 1u << numVars;

  // Use precomputed table for small cases (up to 6 variables = 64 bits)
  if (numVars <= 6) {
    assert(varIndex < 6);
    uint64_t maskValue = kVarMasks[positive][varIndex];
    // Mask off bits beyond numBits
    if (numBits < 64)
      maskValue &= (1ULL << numBits) - 1;
    return APInt(numBits, maskValue);
  }

  // For larger cases, use getSplat to create repeating pattern
  // Pattern width is 2^(var+1) bits
  uint64_t patternWidth = 1u << (varIndex + 1);
  APInt pattern(patternWidth, 0);

  // Fill the appropriate half of the pattern
  uint64_t shift = 1u << varIndex;
  if constexpr (positive) {
    // Set upper half: bits [shift, patternWidth)
    pattern.setBits(shift, patternWidth);
  } else {
    // Set lower half: bits [0, shift)
    pattern.setBits(0, shift);
  }

  return APInt::getSplat(numBits, pattern);
}

APInt circt::createVarMask(unsigned numVars, unsigned varIndex, bool positive) {
  if (positive)
    return createVarMaskImpl<true>(numVars, varIndex);
  return createVarMaskImpl<false>(numVars, varIndex);
}

/// Compute cofactor of a Boolean function for a given variable.
///
/// A cofactor of a function f with respect to variable x is the function
/// obtained by fixing x to a constant value:
///   - Negative cofactor f|x=0 : f with variable x set to 0
///   - Positive cofactor f|x=1 : f with variable x set to 1
///
/// Cofactors are fundamental in Boolean function decomposition and the
/// Shannon expansion: f = x'*f|x=0 + x*f|x=1
///
/// In truth table representation, cofactors are computed by selecting the
/// subset of minterms where the variable has the specified value, then
/// replicating that pattern across the full truth table width.
///
/// Returns: (negative cofactor, positive cofactor)
std::pair<APInt, APInt>
circt::computeCofactors(const APInt &f, unsigned numVars, unsigned var) {
  assert(numVars <= 64 && "Number of variables too large");
  assert(var < numVars && "Variable index out of bounds");
  assert(f.getBitWidth() == (1u << numVars) && "Truth table size mismatch");
  uint64_t numBits = 1u << numVars;
  uint64_t shift = 1u << var;

  // Build masks using getSplat to replicate bit patterns
  // For var at position k, we need blocks of size 2^k where:
  // - mask0 selects lower 2^k bits of each 2^(k+1)-bit block (var=0)
  // - mask1 selects upper 2^k bits of each 2^(k+1)-bit block (var=1)
  APInt blockPattern = APInt::getLowBitsSet(2 * shift, shift);
  APInt mask0 = APInt::getSplat(numBits, blockPattern);
  APInt mask1 = mask0.shl(shift);

  // Extract bits for each cofactor
  APInt selected0 = f & mask0;
  APInt selected1 = f & mask1;

  // Replicate the selected bits across the full truth table using getSplat
  APInt cof0 = selected0 | selected0.shl(shift);
  APInt cof1 = selected1 | selected1.lshr(shift);

  return {cof0, cof1};
}
