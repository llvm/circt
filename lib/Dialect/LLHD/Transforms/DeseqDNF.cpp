//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeseqDNF.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-deseq"

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallDenseSet;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Insert an item into something like a vector, maintaining the vector in
/// sorted order and removing any duplicates.
template <typename Container, typename Item>
static bool insertUniqued(Container &container, Item item) {
  auto it = llvm::lower_bound(container, item);
  if (it != container.end() && *it == item)
    return false;
  container.insert(it, std::move(item));
  return true;
}

/// Sort and uniquify a container.
template <typename Container>
static void uniquify(Container &container) {
  llvm::sort(container);
  container.erase(llvm::unique(container), container.end());
}

/// Check whether a container is sorted and contains only unique items.
template <typename Container>
static bool isUniqued(Container &container) {
  if (container.size() < 2)
    return true;
  auto it = container.begin();
  auto prev = it++;
  auto end = container.end();
  for (; it != end; ++it, ++prev)
    if (!(*prev < *it))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// AndTerm
//===----------------------------------------------------------------------===//

int AndTerm::compare(const AndTerm other) const {
  if (value.getAsOpaquePointer() < other.value.getAsOpaquePointer())
    return -1;
  if (value.getAsOpaquePointer() > other.value.getAsOpaquePointer())
    return 1;
  if (uses.to_ulong() < other.uses.to_ulong())
    return -1;
  if (uses.to_ulong() > other.uses.to_ulong())
    return 1;
  return 0;
}

bool AndTerm::isSubsetOf(const AndTerm &other) const {
  if (value != other.value)
    return false;
  return (uses & other.uses) == other.uses;
}

unsigned AndTerm::isSubsetOfModuloSingleFlip(const AndTerm &other,
                                             Use &flippedUse) const {
  if (value != other.value)
    return 2; // not a subset

  // Determine which bits are present in this and the other term. For this to be
  // a subset of the other term, all other terms must be present in this term.
  auto thisMask = (uses | (uses << 2) | (uses >> 2));
  auto otherMask = (other.uses | (other.uses << 2) | (other.uses >> 2));
  auto mask = thisMask & otherMask;
  if (mask != otherMask)
    return 2; // not a subset

  // Determine the terms that are different. If the uses differ only by a
  // single inverted term, the resulting bit set will have those two bits set:
  //   this  = 0011  (Past|Id)
  //   other = 0110  (Past|NotId)
  //   diff  = 0101  (Id|NotId)
  //   which = 0001  (Id)
  auto diff = (uses & mask) ^ (other.uses & mask);
  if (diff.none())
    return 0; // identical terms, no flips
  auto which = diff & (diff >> 2);
  if (llvm::popcount(diff.to_ulong()) != 2 ||
      llvm::popcount(which.to_ulong()) != 1)
    return 2; // more than one flipped term
  flippedUse = Use(llvm::countr_zero(which.to_ulong()));
  return 1; // exactly one flipped term
}

void AndTerm::print(llvm::raw_ostream &os,
                    llvm::function_ref<void(Value)> printValue) const {
  // Default value printer giving each value and anonymous `vN` name.
  SmallMapVector<Value, unsigned, 8> valueIds;
  auto defaultPrintValue = [&](Value value) {
    os << "v" << valueIds.insert({value, valueIds.size()}).first->second;
  };
  if (!printValue)
    printValue = defaultPrintValue;

  // Duplicate the term, handle posedge/negedge separately, and then print the
  // remaining terms.
  auto that = *this;
  bool needsSeparator = false;
  if (that.hasAllUses(PosEdgeUses)) {
    that.removeUses(PosEdgeUses);
    os << "/";
    printValue(value);
    needsSeparator = true;
  }
  if (that.hasAllUses(NegEdgeUses)) {
    that.removeUses(NegEdgeUses);
    if (needsSeparator)
      os << "&";
    os << "\\";
    printValue(value);
    needsSeparator = true;
  }
  for (unsigned i = 0; i < 4; ++i) {
    auto use = Use(i);
    if (that.hasUse(use)) {
      if (needsSeparator)
        os << "&";
      if (use & Not)
        os << "!";
      if (use == Past || use == NotPast)
        os << "@";
      printValue(value);
      needsSeparator = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// OrTerm
//===----------------------------------------------------------------------===//

bool OrTerm::contains(const AndTerm &andTerm) const {
  auto it = llvm::lower_bound(andTerms, andTerm);
  return it != andTerms.end() && *it == andTerm;
}

int OrTerm::compare(const OrTerm &other) const {
  auto &a = *this;
  auto &b = other;
  if (a.andTerms.size() < b.andTerms.size())
    return -1;
  if (a.andTerms.size() > b.andTerms.size())
    return 1;
  for (auto [aTerm, bTerm] : llvm::zip(a.andTerms, b.andTerms))
    if (auto order = aTerm.compare(bTerm); order != 0)
      return order;
  return 0;
}

bool OrTerm::isSubsetOfModuloSingleFlip(const OrTerm &other,
                                        unsigned &flippedIdx,
                                        AndTerm::Use &flippedUse) const {

  // For `this` to be a subset of `other`, all terms of `other` must be
  // present in `this`, with `this` potentially containing other terms. So
  // `this` must at least have as many terms as `other`.
  if (andTerms.size() < other.andTerms.size())
    return false;

  // Iterate through the terms in `this` and `other`, and make sure every term
  // in `other` is present in `this`, and skip terms in `this` that are not
  // present in `other`.
  unsigned thisIdx = 0;
  unsigned otherIdx = 0;
  bool flipFound = false;
  while (thisIdx < andTerms.size() && otherIdx < other.andTerms.size()) {
    auto thisTerm = andTerms[thisIdx];
    auto otherTerm = other.andTerms[otherIdx];
    unsigned flips = thisTerm.isSubsetOfModuloSingleFlip(otherTerm, flippedUse);
    if (flips == 0) {
      ++thisIdx;
      ++otherIdx;
    } else if (flips == 1) {
      // Catch the case where this is the second flip we find.
      if (flipFound)
        return false;
      flipFound = true;
      flippedIdx = thisIdx;
      ++thisIdx;
      ++otherIdx;
    } else if (thisTerm < otherTerm) {
      ++thisIdx;
    } else {
      return false;
    }
  }
  return otherIdx == other.andTerms.size() && flipFound;
}

bool OrTerm::isSubsetOf(const OrTerm &other) const {
  // For `this` to be a subset of `other`, all terms of `other` must be
  // present in `this`, with `this` potentially containing other terms. So
  // `this` must at least have as many terms as `other`.
  if (andTerms.size() < other.andTerms.size())
    return false;

  // Iterate through the terms in `this` and `other`, and make sure every term
  // in `other` is present in `this`, and skip terms in `this` that are not
  // present in `other`.
  unsigned thisIdx = 0;
  unsigned otherIdx = 0;
  while (thisIdx < andTerms.size() && otherIdx < other.andTerms.size()) {
    auto thisTerm = andTerms[thisIdx];
    auto otherTerm = other.andTerms[otherIdx];
    if (thisTerm.isSubsetOf(otherTerm)) {
      ++thisIdx;
      ++otherIdx;
    } else if (thisTerm < otherTerm) {
      ++thisIdx;
    } else {
      return false;
    }
  }
  return otherIdx == other.andTerms.size();
}

llvm::hash_code OrTerm::flipInvariantHash() const {
  auto range = llvm::map_range(andTerms, [](AndTerm andTerm) {
    // Enable all complementary uses in the term to make the hash invariant
    // under term flipping.
    andTerm.uses |= andTerm.uses >> 2;
    andTerm.uses |= andTerm.uses << 2;
    return llvm::hash_combine(andTerm.value, andTerm.uses.to_ulong());
  });
  return llvm::hash_combine_range(range.begin(), range.end());
}

static bool compareValueOnly(const AndTerm &a, const AndTerm &b) {
  return a.value.getAsOpaquePointer() < b.value.getAsOpaquePointer();
}

bool OrTerm::addTerm(AndTerm term) {
  // Try to merge into an existing term with the same value.
  auto it = llvm::lower_bound(andTerms, term, compareValueOnly);
  if (it != andTerms.end() && it->value == term.value) {
    it->addUses(term.uses);
    return !it->isFalse();
  }

  // Otherwise insert.
  andTerms.insert(it, std::move(term));
  return true;
}

bool OrTerm::addTerms(ArrayRef<AndTerm> terms) {
  for (auto term : terms)
    if (!addTerm(term))
      return false;
  return true;
}

bool OrTerm::isSortedAndUnique() const {
  for (unsigned idx = 0; idx + 1 < andTerms.size(); ++idx)
    if (andTerms[idx] >= andTerms[idx + 1])
      return false;
  for (auto &andTerm : andTerms)
    if (andTerm.isFalse() || andTerm.isTrue())
      return false;
  return true;
}

std::optional<bool> OrTerm::evaluate(
    llvm::function_ref<std::optional<bool>(Value, bool)> evaluateTerm) {
  bool allTrue = true;
  for (auto &andTerm : andTerms) {
    for (unsigned use = 0; use < 4; ++use) {
      if (!andTerm.hasUse(AndTerm::Use(use)))
        continue;
      auto result =
          evaluateTerm(andTerm.value, (use & ~AndTerm::Not) == AndTerm::Past);
      if (!result) {
        allTrue = false;
        continue;
      }
      if (use & AndTerm::Not)
        *result = !*result;
      if (!*result)
        return {false};
    }
  }
  if (allTrue)
    return {true};
  return {};
}

void OrTerm::print(llvm::raw_ostream &os,
                   llvm::function_ref<void(Value)> printValue) const {
  // Handle trivial true.
  if (isTrue()) {
    os << "true";
    return;
  }

  // Default value printer giving each value and anonymous `vN` name.
  SmallMapVector<Value, unsigned, 8> valueIds;
  auto defaultPrintValue = [&](Value value) {
    os << "v" << valueIds.insert({value, valueIds.size()}).first->second;
  };
  if (!printValue)
    printValue = defaultPrintValue;

  // Print terms.
  llvm::interleave(
      andTerms, os, [&](auto andTerm) { andTerm.print(os, printValue); }, "&");
}

//===----------------------------------------------------------------------===//
// DNF
//===----------------------------------------------------------------------===//

bool DNF::contains(const OrTerm &orTerm) const {
  auto it = llvm::lower_bound(orTerms, orTerm);
  return it != orTerms.end() && *it == orTerm;
}

bool DNF::contains(const AndTerm &andTerm) const {
  for (auto &orTerm : orTerms)
    if (orTerm.contains(andTerm))
      return true;
  return false;
}

int DNF::compare(const DNF &other) const {
  auto &a = *this;
  auto &b = other;
  if (a.orTerms.size() < b.orTerms.size())
    return -1;
  if (a.orTerms.size() > b.orTerms.size())
    return 1;
  for (auto [aTerm, bTerm] : llvm::zip(a.orTerms, b.orTerms))
    if (auto order = aTerm.compare(bTerm); order != 0)
      return order;
  return 0;
}

bool DNF::isSortedAndUnique() const {
  for (auto &orTerm : orTerms)
    if (!orTerm.isSortedAndUnique())
      return false;
  for (unsigned idx = 0; idx + 1 < orTerms.size(); ++idx)
    if (orTerms[idx] >= orTerms[idx + 1])
      return false;
  return true;
}

void DNF::optimize() {
  OrTerms optimizedTerms;
  OrTerms extraTerms;
  SmallDenseSet<unsigned, 4> eraseTerms;

  // The terms are ordered from shortest to longest. Reverse that such that we
  // are processing long terms first. This is useful since our two
  // optimizations will generally shorten terms. This allows us to just delete
  // the optimized term and add it to the `extraTerms` queue to be re-inserted
  // later.
  std::reverse(orTerms.begin(), orTerms.end());

  unsigned newIdx = 0;
  while (newIdx < orTerms.size() || !extraTerms.empty()) {
    // Pick the next term to add.
    OrTerm nextTerm;
    if (extraTerms.empty()) {
      nextTerm = std::move(orTerms[newIdx++]);
    } else if (newIdx == orTerms.size()) {
      nextTerm = extraTerms.pop_back_val();
    } else {
      auto order = orTerms[newIdx].compare(extraTerms.back());
      if (order < 0)
        nextTerm = extraTerms.pop_back_val();
      else if (order > 0)
        nextTerm = std::move(orTerms[newIdx++]);
      else {
        extraTerms.pop_back();
        continue; // don't add duplicates
      }
    }

    // Check if this is a superset of a term we already have, in which case we
    // can delete the old term.
    for (unsigned idx = 0; idx < optimizedTerms.size(); ++idx)
      if (!eraseTerms.contains(idx))
        if (optimizedTerms[idx].isSubsetOf(nextTerm))
          eraseTerms.insert(idx);

    // Check if this is a superset of a term we already have, but with a single
    // term flipped, in which case we can remove the flipped term from the old
    // term.
    bool nextTermNeeded = true;
    for (unsigned idx = 0; idx < optimizedTerms.size(); ++idx) {
      if (eraseTerms.contains(idx))
        continue;
      unsigned flippedIdx;
      AndTerm::Use flippedUse;
      if (!optimizedTerms[idx].isSubsetOfModuloSingleFlip(nextTerm, flippedIdx,
                                                          flippedUse))
        continue;
      auto extraTerm = optimizedTerms[idx];
      auto &flippedTerm = extraTerm.andTerms[flippedIdx];
      flippedTerm.removeUse(flippedUse);
      flippedTerm.removeUse(AndTerm::Use(flippedUse | AndTerm::Not));
      if (!extraTerm.isTrue())
        extraTerm.andTerms.erase(extraTerm.andTerms.begin() + flippedIdx);
      insertUniqued(extraTerms, std::move(extraTerm));
      assert(isUniqued(extraTerms));
      eraseTerms.insert(idx);

      // If the two terms were identical except for the flipped term, e.g.
      // `a&b&c` and `!a&b&c`, there is no need to add the next term `!a&b&c`,
      // since it will be subsumed by the extra term `b&c`.
      if (optimizedTerms[idx].andTerms.size() == nextTerm.andTerms.size())
        nextTermNeeded = false;
    }
    if (nextTermNeeded)
      optimizedTerms.push_back(std::move(nextTerm));
  }

  // Remove terms marked for deletion. We do this with a two-index method,
  // where we scan ahead through the terms, moving them from the read index
  // position to the write index position. If we keep the term, advance both
  // indices; if we skip the term, only advance the read index without moving
  // anything.
  unsigned writeIdx = 0;
  for (unsigned readIdx = 0; readIdx < optimizedTerms.size(); ++readIdx)
    if (!eraseTerms.contains(readIdx))
      optimizedTerms[writeIdx++] = std::move(optimizedTerms[readIdx]);
  optimizedTerms.resize(writeIdx);

  // Bring the terms back into shortest to longest order.
  std::reverse(optimizedTerms.begin(), optimizedTerms.end());

  orTerms = std::move(optimizedTerms);
}

// static SmallString<16> bin(const APInt &input) {
//   SmallString<16> buffer;
//   input.toStringUnsigned(buffer, 2);
//   std::reverse(buffer.begin(), buffer.end());
//   while (buffer.size() < input.getBitWidth())
//     buffer.push_back('0');
//   std::reverse(buffer.begin(), buffer.end());
//   return buffer;
// }

void DNF::optimize2() {
  // llvm::errs() << "Optimizing ";
  // print(llvm::errs());
  // llvm::errs() << "\n";

  // Count the number of distinct input terms.
  SmallMapVector<std::pair<Value, AndTerm::Use>, uint8_t, 8> terms;
  for (auto &orTerm : orTerms) {
    for (auto &andTerm : orTerm.andTerms) {
      if (andTerm.hasAnyUses(1 << AndTerm::Id | 1 << AndTerm::NotId))
        terms.insert({{andTerm.value, AndTerm::Id}, uint8_t(terms.size())});
      if (andTerm.hasAnyUses(1 << AndTerm::Past | 1 << AndTerm::NotPast))
        terms.insert({{andTerm.value, AndTerm::Past}, uint8_t(terms.size())});
      if (terms.size() > 16)
        return;
    }
  }
  // llvm::errs() << "- " << terms.size() << " inputs\n";
  unsigned tableSize = 1 << terms.size();
  // llvm::errs() << "- Truth table has " << tableSize << " entries\n";

  // Compute the truth table.
  auto table = APInt::getZero(tableSize);
  auto getCheckerMask = [&](unsigned idx) {
    return APInt::getSplat(tableSize,
                           APInt::getHighBitsSet(2 << idx, 1 << idx));
  };
  auto getTermMask = [&](Value value, AndTerm::Use use) {
    return getCheckerMask(terms.lookup({value, use}));
  };
  for (auto &orTerm : orTerms) {
    auto orTable = APInt::getAllOnes(tableSize);
    for (auto &andTerm : orTerm.andTerms) {
      if (andTerm.hasUse(AndTerm::Id))
        orTable &= getTermMask(andTerm.value, AndTerm::Id);
      if (andTerm.hasUse(AndTerm::NotId))
        orTable &= ~getTermMask(andTerm.value, AndTerm::Id);
      if (andTerm.hasUse(AndTerm::Past))
        orTable &= getTermMask(andTerm.value, AndTerm::Past);
      if (andTerm.hasUse(AndTerm::NotPast))
        orTable &= ~getTermMask(andTerm.value, AndTerm::Past);
    }
    // llvm::errs() << "- Subtable " << bin(orTable) << "\n";
    table |= orTable;
  }
  // llvm::errs() << "- Table " << bin(table) << "\n";

  // Handle the trivial cases.
  if (table.isZero()) {
    orTerms.clear();
    return;
  }
  if (table.isAllOnes()) {
    orTerms.clear();
    orTerms.push_back(OrTerm());
    return;
  }

  // Convert the truth table back to the corresponding AND and OR terms.
  // auto sortedTerms = terms.takeVector();
  OrTerms newTerms;
  // auto coveredBits = APInt::getZero(tableSize);
  auto remainingBits = table;
  for (unsigned i = 0; i < 10 && !remainingBits.isZero(); ++i) {
    // llvm::errs() << "- Remaining " << bin(remainingBits) << "\n";
    unsigned nextIdx = remainingBits.countTrailingZeros();
    // llvm::errs() << "  - Targeting " << nextIdx << "\n";
    unsigned bestKeeps = -1;
    auto bestMask = APInt::getZero(tableSize);
    for (unsigned keeps = 0; keeps < tableSize; ++keeps) {
      auto mask = APInt::getAllOnes(tableSize);
      for (unsigned termIdx = 0; termIdx < terms.size(); ++termIdx)
        if (keeps & (1 << termIdx))
          mask &= (nextIdx & (1 << termIdx)) ? getCheckerMask(termIdx)
                                             : ~getCheckerMask(termIdx);
      if ((table & mask) == mask &&
          llvm::popcount(keeps) < llvm::popcount(bestKeeps)) {
        bestKeeps = keeps;
        bestMask = mask;
      }
    }
    // llvm::errs() << "  - Best keep " << bestKeeps << ", covers "
    //              << bin(bestMask) << "\n";
    OrTerm::AndTerms andTerms;
    for (auto [term, termIdx] : terms) {
      if (~bestKeeps & (1 << termIdx))
        continue;
      unsigned use = term.second;
      if (~nextIdx & (1 << termIdx))
        use |= AndTerm::Not;
      if (!andTerms.empty() && andTerms.back().value == term.first)
        andTerms.back().addUse(AndTerm::Use(use));
      else
        andTerms.push_back(AndTerm(term.first, 1 << use));
    }
    llvm::sort(andTerms);
    newTerms.push_back(OrTerm(std::move(andTerms)));
    remainingBits &= ~bestMask;
  }

  llvm::sort(newTerms);
  orTerms = std::move(newTerms);
  // llvm::errs() << "- Final DNF: ";
  // print(llvm::errs());
  // llvm::errs() << "\n";
}

std::optional<bool> DNF::evaluate(
    llvm::function_ref<std::optional<bool>(Value, bool)> evaluateTerm) {
  bool allFalse = true;
  for (auto &orTerm : orTerms) {
    auto result = orTerm.evaluate(evaluateTerm);
    if (!result) {
      allFalse = false;
      continue;
    }
    if (*result)
      return {true};
  }
  if (allFalse)
    return {false};
  return {};
}

void DNF::print(llvm::raw_ostream &os,
                llvm::function_ref<void(Value)> printValue) const {
  // Handle trivial cases.
  if (isNull()) {
    os << "null";
    return;
  }
  if (isFalse()) {
    os << "false";
    return;
  }
  if (isTrue()) {
    os << "true";
    return;
  }

  // Default value printer giving each value and anonymous `vN` name.
  SmallMapVector<Value, unsigned, 8> valueIds;
  auto defaultPrintValue = [&](Value value) {
    os << "v" << valueIds.insert({value, valueIds.size()}).first->second;
  };
  if (!printValue)
    printValue = defaultPrintValue;

  // Print the terms.
  llvm::interleave(
      orTerms, os, [&](auto &orTerm) { orTerm.print(os, printValue); }, " | ");
}

void DNF::printWithValues(llvm::raw_ostream &os) const {
  os << "DNF(";

  // Print the DNF itself.
  SmallMapVector<Value, unsigned, 8> valueIds;
  print(os, [&](Value value) {
    auto id = valueIds.insert({value, valueIds.size()}).first->second;
    os << "v" << id;
  });

  // Print the values used.
  if (!valueIds.empty()) {
    os << " with ";
    llvm::interleaveComma(valueIds, os, [&](auto valueAndId) {
      os << "v" << valueAndId.second << " = ";
      valueAndId.first.printAsOperand(os, OpPrintingFlags());
    });
  }

  os << ")";
}

//===----------------------------------------------------------------------===//
// OR Operation
//===----------------------------------------------------------------------===//

DNF DNF::operator|(const DNF &other) const {
  if (isNull() || other.isNull())
    return DNF();

  // Handle `true | a -> true`.
  if (isTrue())
    return DNF(true);

  // Handle `a | true -> true`.
  if (other.isTrue())
    return DNF(true);

  // Handle `false | a -> a`.
  if (isFalse())
    return other;

  // Handle `a | false -> a`.
  if (other.isFalse())
    return *this;

  // Combine the or terms of the two DNFs, deduplicating them along the way. We
  // assume that the terms in both DNFs are ordered. That allows us to use a
  // two-index method to linearly scan through the terms and pick the lower ones
  // first, ensuring that the merged terms are again ordered.
  auto result = DNF(false);
  unsigned thisIdx = 0;
  unsigned otherIdx = 0;
  while (thisIdx < orTerms.size() && otherIdx < other.orTerms.size()) {
    OrTerm nextTerm;
    auto order = orTerms[thisIdx].compare(other.orTerms[otherIdx]);
    if (order < 0) {
      nextTerm = std::move(orTerms[thisIdx++]);
    } else if (order > 0) {
      nextTerm = other.orTerms[otherIdx++];
    } else {
      ++otherIdx;
      continue; // don't add duplicates
    }
    result.orTerms.push_back(std::move(nextTerm));
  }
  for (; thisIdx < orTerms.size(); ++thisIdx)
    result.orTerms.push_back(std::move(orTerms[thisIdx]));
  for (; otherIdx < other.orTerms.size(); ++otherIdx)
    result.orTerms.push_back(other.orTerms[otherIdx]);

  result.optimize2();
  return result;
}

//===----------------------------------------------------------------------===//
// AND Operation
//===----------------------------------------------------------------------===//

DNF DNF::operator&(const DNF &other) const {
  if (isNull() || other.isNull())
    return DNF();

  // Handle `false & a -> false`.
  if (isFalse())
    return DNF(false);

  // Handle `a & false -> false`.
  if (other.isFalse())
    return DNF(false);

  // Handle `true & a -> a`.
  if (isTrue())
    return other;

  // Handle `a & true -> a`.
  if (other.isTrue())
    return *this;

  // Handle the general case.
  auto result = DNF(false);
  for (auto &thisTerm : orTerms) {
    for (auto &otherTerm : other.orTerms) {
      auto newTerm = thisTerm;
      if (newTerm.addTerms(otherTerm.andTerms))
        insertUniqued(result.orTerms, newTerm);
    }
  }
  assert(result.isSortedAndUnique());
  result.optimize2();
  return result;
}

//===----------------------------------------------------------------------===//
// XOR Operation
//===----------------------------------------------------------------------===//

DNF DNF::operator^(const DNF &other) const {
  if (isNull() || other.isNull())
    return DNF();

  // Handle `a ^ false -> a`.
  if (isFalse())
    return other;

  // Handle `false ^ a -> a`.
  if (other.isFalse())
    return *this;

  // Handle `a ^ true -> ~a`.
  if (isTrue())
    return ~other;

  // Handle `true ^ a -> ~a`.
  if (other.isTrue())
    return ~(*this);

  // Otherwise compute a ^ b as (!a&b) | (a&!b).
  return ~(*this) & other | *this & ~other;
}

//===----------------------------------------------------------------------===//
// NOT Operation
//===----------------------------------------------------------------------===//

DNF DNF::operator~() const {
  // Handle the trivial cases.
  if (isNull())
    return DNF();
  if (isTrue())
    return DNF(false);
  if (isFalse())
    return DNF(true);

  // Otherwise go through each of the AND terms, negate them individually, and
  // then AND them into the result. For example:
  //   !(a&b | c&d | e&f)
  //   !(a&b) & !(c&d) & !(e&f)
  //   (!a | !b) & (!c & !d) & (!e & !f)
  auto result = DNF(true);
  auto inverted = DNF(false);
  for (auto &orTerm : orTerms) {
    // Map `!(a&b&c)` to `(!a | !b | !c)`.
    for (auto &andTerm : orTerm.andTerms)
      for (unsigned i = 0; i < 4; ++i)
        if (andTerm.hasUse(AndTerm::Use(i)))
          inverted.orTerms.push_back(
              OrTerm(AndTerm(andTerm.value, 1 << (i ^ AndTerm::Not))));
    llvm::sort(inverted.orTerms);
    result &= inverted;
    inverted.orTerms.clear();
  }
  assert(result.isSortedAndUnique());
  return result;
}
