//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeseqUtils.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace circt;
using namespace llhd;
using namespace deseq;
using llvm::SmallMapVector;

//===----------------------------------------------------------------------===//
// Disjunctive Normal Form
//===----------------------------------------------------------------------===//

llvm::raw_ostream &deseq::operator<<(llvm::raw_ostream &os,
                                     const DNFTerm &term) {
  if (term.isTrue()) {
    os << "(true)";
  } else if (term.isFalse()) {
    os << "(false)";
  } else {
    bool needsSeparator = false;
    uint64_t rest = term.andTerms;
    while (rest != 0) {
      auto bit = llvm::countr_zero(rest);
      rest &= ~(1UL << bit);
      auto termIdx = bit >> 1;
      auto negative = bit & 1;
      if (needsSeparator)
        os << "&";
      if (negative)
        os << "!";
      if (termIdx == 0)
        os << "x";
      else
        os << "a" << termIdx;
      needsSeparator = true;
    }
  }
  return os;
}

llvm::raw_ostream &deseq::operator<<(llvm::raw_ostream &os, const DNF &dnf) {
  if (dnf.isPoison())
    os << "poison";
  else if (dnf.isTrue())
    os << "true";
  else if (dnf.isFalse())
    os << "false";
  else
    llvm::interleave(dnf.orTerms, os, " | ");
  return os;
}

//===----------------------------------------------------------------------===//
// Truth Table
//===----------------------------------------------------------------------===//

TruthTable TruthTable::operator~() const {
  auto z = *this;
  z.invert();
  return z;
}

TruthTable TruthTable::operator&(const TruthTable &other) const {
  auto z = *this;
  z &= other;
  return z;
}

TruthTable TruthTable::operator|(const TruthTable &other) const {
  auto z = *this;
  z |= other;
  return z;
}

TruthTable TruthTable::operator^(const TruthTable &other) const {
  auto z = *this;
  z ^= other;
  return z;
}

TruthTable &TruthTable::invert() {
  if (!isPoison()) {
    bits.flipAllBits();
    fixupUnknown();
  }
  return *this;
}

TruthTable &TruthTable::operator&=(const TruthTable &other) {
  if (!isPoison()) {
    if (other.isPoison())
      *this = getPoison();
    else
      bits &= other.bits;
  }
  return *this;
}

TruthTable &TruthTable::operator|=(const TruthTable &other) {
  if (!isPoison()) {
    if (other.isPoison())
      *this = getPoison();
    else
      bits |= other.bits;
  }
  return *this;
}

TruthTable &TruthTable::operator^=(const TruthTable &other) {
  if (isPoison()) {
    // no-op
  } else if (other.isPoison()) {
    *this = getPoison();
  } else if (isTrue()) {
    *this = other;
    invert();
  } else if (isFalse()) {
    *this = other;
  } else if (other.isTrue()) {
    invert();
  } else if (other.isFalse()) {
    // no-op
  } else {
    auto rhs = ~*this;
    rhs &= other;    // ~a & b
    *this &= ~other; // a & ~b
    *this |= rhs;    // (a & ~b) | (~a & b)
  }
  return *this;
}

DNF TruthTable::canonicalize() const {
  // Handle the trivial cases.
  if (isPoison())
    return DNF{{DNFTerm{UINT32_MAX}}};
  if (isTrue())
    return DNF{{DNFTerm{0}}};
  if (isFalse())
    return DNF{{}};

  // Otherwise add terms to the DNF iteratively until there are no more ones
  // remaining in the truth table.
  unsigned numTerms = getNumTerms();
  DNF dnf;
  auto remainingBits = bits;
  while (!remainingBits.isZero()) {
    // Find the index of the next 1 in the truth table. Coincidentally, the
    // binary digits of this index correspond to the assignment of 0s and 1s
    // to the expression terms at this row of the truth table.
    unsigned nextIdx = remainingBits.countTrailingZeros();

    // We want to generate a DNF with the fewest terms possible. To do so,
    // iterate over all possible combinations of which expression terms to
    // add. We do this by iterating the `keeps` variable through all possible
    // bit combinations where each bit corresponds to an expression term being
    // present or not. 0b0000 would not add any terms, 0b0011 would add the
    // terms `t0&t1`, 0b1111 would add the terms `t0&t1&t2&t3`. The fewer
    // terms we add, the more 1s in the truth table are covered. We try to
    // find the largest number of such 1s that are a subset of our truth
    // table, and we remember the exact combination of kept terms for it.
    unsigned bestKeeps = -1;
    auto bestMask = APInt::getZero(bits.getBitWidth());
    for (unsigned keeps = 0; keeps < bits.getBitWidth(); ++keeps) {
      auto mask = APInt::getAllOnes(bits.getBitWidth());
      for (unsigned termIdx = 0; termIdx < numTerms; ++termIdx)
        if (keeps & (1 << termIdx))
          mask &= (nextIdx & (1 << termIdx)) ? getTermMask(termIdx)
                                             : ~getTermMask(termIdx);
      if ((bits & mask) == mask &&
          llvm::popcount(keeps) < llvm::popcount(bestKeeps)) {
        bestKeeps = keeps;
        bestMask = mask;
      }
    }

    // At this point, `bestKeeps` corresponds to a bit mask indicating which
    // of the expression terms to incorporate in the AND operation we are
    // about to add to the DNF. `bestMask` corresponds to the 1s in the truth
    // table covered by these terms. Now populate an AND operation with these
    // terms.
    DNFTerm orTerm;
    for (unsigned termIdx = 0; termIdx < numTerms; ++termIdx) {
      if (~bestKeeps & (1 << termIdx))
        continue;
      bool invert = (~nextIdx & (1 << termIdx));
      orTerm.andTerms |= (1 << (termIdx * 2 + invert));
    }
    dnf.orTerms.push_back(orTerm);

    // Clear the bits that we have covered with this term. This causes the
    // next iteration to work towards covering the next 1 in the truth table
    // that has not been covered by any term thus far.
    remainingBits &= ~bestMask;
  }
  return dnf;
}

void TruthTable::fixupUnknown() {
  auto mask = bits ^ (bits << 1);
  mask &= getTermMask(0);
  bits |= mask;
  mask.lshrInPlace(1);
  mask.flipAllBits();
  bits &= mask;
}

llvm::raw_ostream &deseq::operator<<(llvm::raw_ostream &os,
                                     const TruthTable &table) {
  return os << table.canonicalize();
}

//===----------------------------------------------------------------------===//
// Value Table
//===----------------------------------------------------------------------===//

void ValueEntry::merge(ValueEntry other) {
  if (isPoison() || other.isPoison()) {
    *this = getPoison();
    return;
  }
  if (value == other.value)
    return;
  if (!isUnknown() && !other.isUnknown()) {
    auto result = dyn_cast<OpResult>(value);
    auto otherResult = dyn_cast<OpResult>(other.value);
    if (result && otherResult &&
        result.getResultNumber() == otherResult.getResultNumber() &&
        mlir::OperationEquivalence::isEquivalentTo(
            result.getOwner(), otherResult.getOwner(),
            mlir::OperationEquivalence::Flags::IgnoreLocations))
      return;
  }
  *this = getUnknown();
}

void ValueTable::addCondition(TruthTable condition) {
  for (auto &entry : entries)
    entry.first &= condition;
  minimize();
}

void ValueTable::merge(const ValueTable &other) {
  entries.append(other.entries.begin(), other.entries.end());
  minimize();
}

void ValueTable::minimize() {
  if (entries.empty())
    return;
  auto seenBits = APInt::getZero(entries[0].first.bits.getBitWidth());

  for (unsigned idx = 0; idx < entries.size(); ++idx) {
    // Remove entries where the condition is trivially false.
    auto &condition = entries[idx].first;
    if (condition.isFalse()) {
      if (idx != entries.size() - 1)
        std::swap(entries[idx], entries.back());
      entries.pop_back();
      --idx;
      continue;
    }

    // Check if the condition of this entry overlaps with an earlier one.
    auto bits = condition.bits;
    if ((seenBits & bits).isZero()) {
      seenBits |= bits;
      continue;
    }

    // Find the earliest entry in the table that overlaps.
    unsigned mergeIdx = 0;
    for (; mergeIdx < entries.size(); ++mergeIdx)
      if (!(entries[mergeIdx].first.bits & bits).isZero())
        break;
    assert(mergeIdx < idx && "should have found an overlapping entry");
    bits |= entries[mergeIdx].first.bits;

    // Merge all overlapping entries into this first one.
    for (unsigned otherIdx = mergeIdx + 1; otherIdx <= idx; ++otherIdx) {
      auto &otherBits = entries[otherIdx].first.bits;
      if ((otherBits & bits).isZero())
        continue;
      bits |= otherBits;
      entries[mergeIdx].first |= entries[otherIdx].first;
      entries[mergeIdx].second.merge(entries[otherIdx].second);
      if (otherIdx != entries.size() - 1)
        std::swap(entries[otherIdx], entries.back());
      entries.pop_back();
      --otherIdx;
      --idx;
    }
  }
}

llvm::raw_ostream &deseq::operator<<(llvm::raw_ostream &os,
                                     const ValueEntry &entry) {
  if (entry.isPoison())
    os << "poison";
  else if (entry.isUnknown())
    os << "x";
  else
    entry.value.printAsOperand(os, mlir::OpPrintingFlags());
  return os;
}

llvm::raw_ostream &
deseq::operator<<(llvm::raw_ostream &os,
                  const std::pair<TruthTable, ValueEntry> &pair) {
  return os << pair.first << " -> " << pair.second;
}

llvm::raw_ostream &deseq::operator<<(llvm::raw_ostream &os,
                                     const ValueTable &table) {
  os << "ValueTable(";
  llvm::interleaveComma(table.entries, os);
  os << ")";
  return os;
}
