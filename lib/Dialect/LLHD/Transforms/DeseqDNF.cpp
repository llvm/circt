//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeseqDNF.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-deseq"

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallDenseSet;
using llvm::SmallMapVector;

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

bool AndTerm::hasSingleFlippedTerm(const AndTerm &other,
                                   Use &flippedUse) const {
  if (value != other.value)
    return false;

  // Determine the terms that are different. If the uses differ only by an
  // single inverted term, the resulting bit set will have those two bits set:
  //   this  = 0011  (Past|Id)
  //   other = 0110  (Past|NotId)
  //   diff  = 0101  (Id|NotId)
  //   which = 0001  (Id)
  auto diff = uses ^ other.uses;
  auto which = diff & (diff >> 2);
  if (llvm::popcount(diff.to_ulong()) != 2 ||
      llvm::popcount(which.to_ulong()) != 1)
    return false;
  flippedUse = Use(llvm::countr_zero(which.to_ulong()));
  return true;
}

void AndTerm::print(llvm::raw_ostream &os,
                    llvm::function_ref<void(Value)> printValue) const {
  // Default value printer giving each value and anonymous `vN` name.
  SmallMapVector<Value, unsigned, 8> valueIds;
  if (!printValue)
    printValue = [&](Value value) {
      os << "v" << valueIds.insert({value, valueIds.size()}).first->second;
    };

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

bool OrTerm::hasSingleFlippedTerm(const OrTerm &other, unsigned &flippedIdx,
                                  AndTerm::Use &flippedUse) const {
  auto &a = *this;
  auto &b = other;
  if (a.andTerms.size() != b.andTerms.size())
    return false;

  bool flipFound = false;
  for (unsigned idx = 0; idx < a.andTerms.size(); ++idx) {
    auto aTerm = a.andTerms[idx];
    auto bTerm = b.andTerms[idx];
    if (aTerm == bTerm)
      continue;
    if (aTerm.hasSingleFlippedTerm(bTerm, flippedUse)) {
      // Catch the case where this is the second flip we find.
      if (flipFound)
        return false;
      flipFound = true;
      flippedIdx = idx;
    }
  }
  return flipFound;
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
    } else if (thisTerm > otherTerm) {
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

void OrTerm::print(llvm::raw_ostream &os,
                   llvm::function_ref<void(Value)> printValue) const {
  // Handle trivial true.
  if (isTrue()) {
    os << "true";
    return;
  }

  // Default value printer giving each value and anonymous `vN` name.
  SmallMapVector<Value, unsigned, 8> valueIds;
  if (!printValue)
    printValue = [&](Value value) {
      os << "v" << valueIds.insert({value, valueIds.size()}).first->second;
    };

  // Print terms.
  llvm::interleave(
      andTerms, os, [&](auto andTerm) { andTerm.print(os, printValue); }, "&");
}

//===----------------------------------------------------------------------===//
// DNF
//===----------------------------------------------------------------------===//

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
  SmallDenseMap<llvm::hash_code, SmallVector<unsigned, 1>> similarTerms;

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

    // Check if any other terms are identical to this term, modulo inversion of
    // individual elements. If we find another term who has exactly one element
    // inverted compared to us, we can remove that element and keep only one of
    // the terms. Since this shortens the term, and we are processing the terms
    // from longest to shortest, simply mark the existing term for deletion and
    // add the new one to the `extraTerms` queue, such that it gets inserted
    // later at the right position.
    auto &similar = similarTerms[nextTerm.flipInvariantHash()];
    bool foundMatchingTerms = false;
    for (auto similarIdx : similar) {
      unsigned flippedIdx;
      AndTerm::Use flippedUse;
      if (!nextTerm.hasSingleFlippedTerm(optimizedTerms[similarIdx], flippedIdx,
                                         flippedUse))
        continue;
      auto extraTerm = optimizedTerms[similarIdx];
      auto &flippedTerm = extraTerm.andTerms[flippedIdx];
      flippedTerm.removeUse(flippedUse);
      flippedTerm.removeUse(AndTerm::Use(flippedUse | AndTerm::Not));
      if (!extraTerm.isTrue())
        extraTerm.andTerms.erase(extraTerm.andTerms.begin() + flippedIdx);
      insertUniqued(extraTerms, std::move(extraTerm));
      assert(isUniqued(extraTerms));
      eraseTerms.insert(similarIdx);
      foundMatchingTerms = true;
    }
    if (!foundMatchingTerms) {
      similar.push_back(optimizedTerms.size());
      optimizedTerms.push_back(std::move(nextTerm));
    }
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

void DNF::print(llvm::raw_ostream &os,
                llvm::function_ref<void(Value)> printValue) const {
  // Handle trivial cases.
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
  if (!printValue)
    printValue = [&](Value value) {
      os << "v" << valueIds.insert({value, valueIds.size()}).first->second;
    };

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
      valueAndId.first.print(os, OpPrintingFlags().skipRegions());
    });
  }

  os << ")";
}

//===----------------------------------------------------------------------===//
// OR Operation
//===----------------------------------------------------------------------===//

DNF DNF::operator|(const DNF &other) const {
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
  auto result = DNF();
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

  result.optimize();
  return result;
}

//===----------------------------------------------------------------------===//
// AND Operation
//===----------------------------------------------------------------------===//

DNF DNF::operator&(const DNF &other) const {
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
  auto result = DNF();
  for (auto &thisTerm : orTerms) {
    for (auto &otherTerm : other.orTerms) {
      auto newTerm = thisTerm;
      if (newTerm.addTerms(otherTerm.andTerms))
        insertUniqued(result.orTerms, newTerm);
    }
  }
  assert(result.isSortedAndUnique());
  result.optimize();
  return result;
}

//===----------------------------------------------------------------------===//
// XOR Operation
//===----------------------------------------------------------------------===//

DNF DNF::operator^(const DNF &other) const {
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
  auto inverted = DNF();
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
