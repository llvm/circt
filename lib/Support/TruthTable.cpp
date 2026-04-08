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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <cassert>

using namespace circt;
using llvm::APInt;

//===----------------------------------------------------------------------===//
// BinaryTruthTable
//===----------------------------------------------------------------------===//

namespace {

static llvm::APInt
expandTruthTableToInputSpaceSmall(const llvm::APInt &tt,
                                  ArrayRef<unsigned> mapping,
                                  unsigned numExpandedInputs) {
  assert(numExpandedInputs <= 6 && "small fast path requires <= 6 inputs");

  unsigned numOrigInputs = mapping.size();
  unsigned expandedSize = 1U << numExpandedInputs;
  // `kVarMasks[i]` marks rows where expanded input `i` is 1. For example,
  // `kVarMasks[0] = 0b1010...` and `kVarMasks[1] = 0b1100...`, matching the
  // standard packed truth-table row order where input 0 is the LSB.
  static constexpr uint64_t kVarMasks[6] = {
      0xAAAAAAAAAAAAAAAAULL, // input 0: 1010...
      0xCCCCCCCCCCCCCCCCULL, // input 1: 1100...
      0xF0F0F0F0F0F0F0F0ULL, // input 2: 11110000...
      0xFF00FF00FF00FF00ULL, // input 3: 8 zeros, 8 ones
      0xFFFF0000FFFF0000ULL, // input 4: 16 zeros, 16 ones
      0xFFFFFFFF00000000ULL  // input 5: 32 zeros, 32 ones
  };

  uint64_t sizeMask = llvm::maskTrailingOnes<uint64_t>(expandedSize);
  unsigned origSize = 1U << numOrigInputs;
  uint64_t origMask = llvm::maskTrailingOnes<uint64_t>(origSize);
  uint64_t origTT = tt.getZExtValue() & origMask;

  // `constrainMasks[i][b]` is the set of expanded rows where original input
  // `i`, after remapping into the expanded space, observes bit value `b`.
  std::array<std::array<uint64_t, 2>, 6> constrainMasks{};
  for (unsigned i = 0; i < numOrigInputs; ++i) {
    uint64_t posMask = kVarMasks[mapping[i]] & sizeMask;
    constrainMasks[i][0] = (~posMask) & sizeMask;
    constrainMasks[i][1] = posMask;
  }

  uint64_t activePatterns = origTT;
  bool useComplementResult = false;
  uint64_t result = 0;
  // If the original truth table has more 1s than 0s, it's more efficient to
  // iterate over the 0 patterns.
  if (static_cast<unsigned>(llvm::popcount(origTT)) > (origSize / 2)) {
    activePatterns = (~origTT) & origMask;
    useComplementResult = true;
    result = sizeMask;
  }

  while (activePatterns) {
    unsigned origIdx = llvm::countr_zero(activePatterns);
    activePatterns &= activePatterns - 1;

    uint64_t pattern = sizeMask;
    for (unsigned i = 0; i < numOrigInputs; ++i)
      pattern &= constrainMasks[i][(origIdx >> i) & 1U];

    if (useComplementResult)
      result &= ~pattern;
    else
      result |= pattern;
  }

  return llvm::APInt(expandedSize, result);
}

static llvm::APInt
expandTruthTableToInputSpaceGeneric(const llvm::APInt &tt,
                                    ArrayRef<unsigned> mapping,
                                    unsigned numExpandedInputs) {
  unsigned numOrigInputs = mapping.size();
  unsigned expandedSize = 1U << numExpandedInputs;

  llvm::APInt result = llvm::APInt::getZero(expandedSize);
  for (unsigned expandedIdx = 0; expandedIdx < expandedSize; ++expandedIdx) {
    unsigned origIdx = 0;
    for (unsigned i = 0; i < numOrigInputs; ++i)
      if ((expandedIdx >> mapping[i]) & 1U)
        origIdx |= 1U << i;
    if (tt[origIdx])
      result.setBit(expandedIdx);
  }

  return result;
}

} // namespace

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

llvm::APInt
circt::detail::expandTruthTableToInputSpace(const llvm::APInt &tt,
                                            ArrayRef<unsigned> mapping,
                                            unsigned numExpandedInputs) {
  unsigned numOrigInputs = mapping.size();
  unsigned expandedSize = 1U << numExpandedInputs;

  if (numOrigInputs == numExpandedInputs) {
    bool isIdentity = true;
    for (unsigned i = 0; i < numOrigInputs && isIdentity; ++i)
      isIdentity = mapping[i] == i;
    if (isIdentity)
      return tt.zext(expandedSize);
  }

  if (tt.isZero())
    return llvm::APInt::getZero(expandedSize);
  if (tt.isAllOnes())
    return llvm::APInt::getAllOnes(expandedSize);

  if (numExpandedInputs <= 6)
    return expandTruthTableToInputSpaceSmall(tt, mapping, numExpandedInputs);
  return expandTruthTableToInputSpaceGeneric(tt, mapping, numExpandedInputs);
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
  for (unsigned i = 0; i < permutation.size(); ++i)
    if (negationMask & (1u << permutation[i]))
      result |= (1u << i);
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

llvm::SmallVector<unsigned>
expandInputPermutation(const std::array<uint8_t, 4> &permutation) {
  llvm::SmallVector<unsigned> result;
  result.reserve(permutation.size());
  for (uint8_t index : permutation)
    result.push_back(index);
  return result;
}

struct NPNTransform4 {
  // Maps each output minterm in the transformed table back to the minterm to
  // read from the source table.
  std::array<uint8_t, 16> outputToSource = {};
  // Inverse mapping used when reconstructing an original table from a chosen
  // canonical representative.
  std::array<uint8_t, 16> inverseOutputToSource = {};
  std::array<uint8_t, 4> inputPermutation = {};
  uint8_t inputNegation = 0;
  bool outputNegation = false;
};

uint16_t applyNPNTransform4(uint16_t truthTable,
                            const std::array<uint8_t, 16> &outputToSource,
                            bool outputNegation) {
  uint16_t result = 0;
  for (unsigned output = 0; output != 16; ++output) {
    unsigned bit = (truthTable >> outputToSource[output]) & 1u;
    if (outputNegation)
      bit ^= 1u;
    result |= static_cast<uint16_t>(bit << output);
  }
  return result;
}

void buildCanonicalOrderNPNTransforms4(
    llvm::SmallVectorImpl<NPNTransform4> &transforms) {
  transforms.clear();
  transforms.reserve(24 * 16 * 2);

  // Enumerate the full 4-input NPN group in a deterministic order so table
  // construction picks stable representatives and encodings.
  for (unsigned negMask = 0; negMask != 16; ++negMask) {
    std::array<unsigned, 4> permutation = {0, 1, 2, 3};
    do {
      std::array<unsigned, 4> inversePermutation = {};
      for (unsigned i = 0; i != 4; ++i)
        inversePermutation[permutation[i]] = i;

      uint8_t currentNegMask = permuteNegationMask(negMask, permutation);
      for (unsigned outputNegation = 0; outputNegation != 2; ++outputNegation) {
        NPNTransform4 transform;
        transform.inputNegation = currentNegMask;
        transform.outputNegation = outputNegation;
        for (unsigned i = 0; i != 4; ++i)
          transform.inputPermutation[i] = permutation[i];

        for (unsigned output = 0; output != 16; ++output) {
          unsigned source = 0;
          for (unsigned input = 0; input != 4; ++input) {
            unsigned bit = (output >> inversePermutation[input]) & 1u;
            bit ^= (negMask >> input) & 1u;
            source |= bit << input;
          }
          transform.outputToSource[output] = source;
          transform.inverseOutputToSource[source] = output;
        }
        transforms.push_back(transform);
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
}

void collectNPN4Representatives(ArrayRef<NPNTransform4> transforms,
                                llvm::SmallVectorImpl<uint16_t> &reps) {
  llvm::BitVector seen(1u << 16, false);
  reps.clear();

  for (unsigned seed = 0; seed != (1u << 16); ++seed) {
    if (seen.test(seed))
      continue;

    // Walk the full NPN orbit of this truth table and pick the numerically
    // smallest member as the canonical representative.
    uint16_t representative = seed;
    for (const auto &transform : transforms) {
      uint16_t member = applyNPNTransform4(seed, transform.outputToSource,
                                           transform.outputNegation);
      seen.set(member);
      representative = std::min(representative, member);
    }
    reps.push_back(representative);
  }
}

} // anonymous namespace

void circt::collectCanonicalNPN4Representatives(
    llvm::SmallVectorImpl<uint16_t> &representatives) {
  llvm::SmallVector<NPNTransform4, 24 * 16 * 2> transforms;
  buildCanonicalOrderNPNTransforms4(transforms);
  collectNPN4Representatives(transforms, representatives);
}

NPNTable::NPNTable() {
  llvm::SmallVector<uint16_t, 222> representatives;
  collectCanonicalNPN4Representatives(representatives);

  llvm::SmallVector<NPNTransform4, 24 * 16 * 2> transforms;
  buildCanonicalOrderNPNTransforms4(transforms);

  llvm::BitVector initialized(entries4.size(), false);
  auto isBetterEntry = [&](const Entry4 &candidate, const Entry4 &current) {
    // Multiple transforms can map the same function to the same
    // representative. Pick a deterministic encoding for the stored witness.
    if (candidate.representative != current.representative)
      return candidate.representative < current.representative;
    if (candidate.inputNegation != current.inputNegation)
      return candidate.inputNegation < current.inputNegation;
    return candidate.outputNegation < current.outputNegation;
  };

  for (uint16_t representative : representatives) {
    for (const auto &transform : transforms) {
      // Starting from the canonical representative, populate every equivalent
      // member with the transform needed to recover the representative.
      uint16_t member =
          applyNPNTransform4(representative, transform.inverseOutputToSource,
                             transform.outputNegation);

      Entry4 candidate;
      candidate.representative = representative;
      candidate.inputPermutation = transform.inputPermutation;
      candidate.inputNegation = transform.inputNegation;
      candidate.outputNegation = transform.outputNegation;

      if (!initialized.test(member) ||
          isBetterEntry(candidate, entries4[member])) {
        entries4[member] = candidate;
        initialized.set(member);
      }
    }
  }

  assert(initialized.all() && "expected to populate all 4-input NPN entries");
}

bool NPNTable::lookup(const BinaryTruthTable &tt, NPNClass &result) const {
  if (tt.numInputs != 4 || tt.numOutputs != 1)
    return false;

  const auto &entry = entries4[tt.table.getZExtValue()];
  result = NPNClass(BinaryTruthTable(4, 1, APInt(16, entry.representative)),
                    expandInputPermutation(entry.inputPermutation),
                    entry.inputNegation, entry.outputNegation);
  return true;
}

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

  // Replicate the selected bits across the full truth table
  APInt cof0 = selected0 | selected0.shl(shift);
  APInt cof1 = selected1 | selected1.lshr(shift);

  return {cof0, cof1};
}

//===----------------------------------------------------------------------===//
// SOPForm
//===----------------------------------------------------------------------===//

APInt SOPForm::computeTruthTable() const {
  APInt tt(1 << numVars, 0);
  for (const auto &cube : cubes) {
    APInt cubeTT = ~APInt(1 << numVars, 0);
    for (unsigned i = 0; i < numVars; ++i) {
      if (cube.hasLiteral(i))
        cubeTT &= createVarMask(numVars, i, !cube.isLiteralInverted(i));
    }
    tt |= cubeTT;
  }
  return tt;
}

#ifndef NDEBUG
void SOPForm::dump(llvm::raw_ostream &os) const {
  os << "SOPForm: " << numVars << " vars, " << cubes.size() << " cubes\n";
  for (const auto &cube : cubes) {
    os << "  (";
    for (unsigned i = 0; i < numVars; ++i) {
      if (cube.mask & (1ULL << i)) {
        os << ((cube.inverted & (1ULL << i)) ? "!" : "");
        os << "x" << i << " ";
      }
    }
    os << ")\n";
  }
}
#endif
//===----------------------------------------------------------------------===//
// ISOP Extraction
//===----------------------------------------------------------------------===//

namespace {

/// Minato-Morreale ISOP algorithm.
///
/// Computes an Irredundant Sum-of-Products (ISOP) cover for a Boolean function.
///
/// References:
/// - Minato, "Fast Generation of Irredundant Sum-of-Products Forms from Binary
///   Decision Diagrams", SASIMI 1992.
/// - Morreale, "Recursive Operators for Prime Implicant and Irredundant Normal
///   Form Determination", IEEE TC 1970.
///
/// Implementation inspired by lsils/kitty library:
/// https://github.com/lsils/kitty/blob/master/include/kitty/isop.hpp
/// distributed under MIT License.
///
/// The algorithm recursively decomposes the function using Shannon expansion:
///   f = !x * f|x=0 + x * f|x=1
///
/// For ISOP, we partition minterms into three groups:
/// - Minterms only in negative cofactor: must use !x literal
/// - Minterms only in positive cofactor: must use x literal
/// - Minterms in both cofactors: can omit x from the cube
///
/// Parameters:
///   tt: The ON-set (minterms that must be covered)
///   dc: The don't-care set (minterms that can optionally be covered)
///       Invariant: tt is a subset of dc (ON-set is subset of care set)
///   numVars: Total number of variables in the function
///   varIndex: Current variable index to start search from (counts down)
///   result: Output SOP form (cubes are accumulated here)
///
/// Returns: The actual cover computed (subset of dc that covers tt)
///
/// The maximum recursion depth is equal to the number of variables (one level
/// per variable). For typical use cases with TruthTable (up to 6-8 variables),
/// this is not a concern. Since truth tables require 2^numVars bits, the
/// recursion depth is not a limiting factor.
APInt isopImpl(const APInt &tt, const APInt &dc, unsigned numVars,
               unsigned varIndex, SOPForm &result) {
  assert((tt & ~dc).isZero() && "tt must be subset of dc");

  // Base case: nothing to cover
  if (tt.isZero())
    return tt;

  // Base case: function is tautology (all don't-cares)
  if (dc.isAllOnes()) {
    result.cubes.emplace_back();
    return dc;
  }

  // Find highest variable in support (top-down from varIndex).
  // NOTE: It is well known that the order of variable selection largely
  // affects the size of the resulting ISOP. There are numerous studies on
  // implementing heuristics for variable selection that could improve results,
  // albeit at the cost of runtime. We may consider implementing such
  // heuristics in the future.
  int var = -1;
  APInt negCof, posCof, negDC, posDC;
  for (int i = varIndex - 1; i >= 0; --i) {
    std::tie(negCof, posCof) = computeCofactors(tt, numVars, i);
    std::tie(negDC, posDC) = computeCofactors(dc, numVars, i);
    if (negCof != posCof || negDC != posDC) {
      var = i;
      break;
    }
  }
  assert(var >= 0 && "No variable found in support");

  // Recurse on minterms unique to negative cofactor (will get !var literal)
  size_t negBegin = result.cubes.size();
  APInt negCover = isopImpl(negCof & ~posDC, negDC, numVars, var, result);
  size_t negEnd = result.cubes.size();

  // Recurse on minterms unique to positive cofactor (will get var literal)
  APInt posCover = isopImpl(posCof & ~negDC, posDC, numVars, var, result);
  size_t posEnd = result.cubes.size();

  // Recurse on shared minterms (no literal for this variable)
  APInt remaining = (negCof & ~negCover) | (posCof & ~posCover);
  APInt sharedCover = isopImpl(remaining, negDC & posDC, numVars, var, result);

  // Compute total cover by restricting each sub-cover to its domain
  APInt negMask = createVarMaskImpl<false>(numVars, var);
  APInt totalCover = sharedCover | (negCover & negMask) | (posCover & ~negMask);

  // Add literals to cubes from recursions
  for (size_t i = negBegin; i < negEnd; ++i)
    result.cubes[i].setLiteral(var, true);

  for (size_t i = negEnd; i < posEnd; ++i)
    result.cubes[i].setLiteral(var, false);

  return totalCover;
}

} // namespace

SOPForm circt::extractISOP(const APInt &truthTable, unsigned numVars) {
  assert((1u << numVars) == truthTable.getBitWidth() &&
         "Truth table size must match 2^numVars");
  SOPForm sop(numVars);

  if (numVars == 0 || truthTable.isZero())
    return sop;

  (void)isopImpl(truthTable, truthTable, numVars, numVars, sop);

  return sop;
}
