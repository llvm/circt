//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <bitset>

namespace circt {
namespace llhd {

/// An individual term of an AND expression in a DNF. This struct represents all
/// terms of the same `Value`.
struct AndTerm {
  /// An integer indicating one use of the value in this term. There are four
  /// different uses, which means there are a total of 2^4 different
  /// combinations each of which has its own bit in the `uses` mask.
  enum Use {
    /// The plain value `x`.
    Id = 0b00,
    /// The past value `@x`.
    Past = 0b01,
    /// The bit that indicates inversion.
    Not = 0b10,
    /// The inverted value `!x`.
    NotId = Not | Id,
    /// The inverted past value `!@x`.
    NotPast = Not | Past,
  };
  /// The set of uses of this value as a bit mask.
  using Uses = std::bitset<4>;

  /// The set of uses that indicates a posedge.
  static constexpr Uses PosEdgeUses = (1 << Id) | (1 << NotPast);
  /// The set of uses that indicates a negedge.
  static constexpr Uses NegEdgeUses = (1 << NotId) | (1 << Past);

  /// The value.
  Value value;
  /// The different uses of this value.
  Uses uses;

  /// Create a null term.
  AndTerm() {}
  /// Create a term with a value and a set of uses.
  AndTerm(Value value, Uses uses) : value(value), uses(uses) {}
  /// Create a term with a plain value.
  static AndTerm id(Value value) { return AndTerm(value, 1 << Id); }
  /// Create a term with a past value.
  static AndTerm past(Value value) { return AndTerm(value, 1 << Past); }
  /// Create a term representing a posedge on a value.
  static AndTerm posEdge(Value value) { return AndTerm(value, PosEdgeUses); }
  /// Create a term representing a negedge on a value.
  static AndTerm negEdge(Value value) { return AndTerm(value, NegEdgeUses); }

  /// Check if this term is trivially false. This is the case if it contains
  /// complementary uses, such as `Id & !Id`.
  bool isFalse() const { return (uses & (uses >> 2)).any(); }
  /// Check if this term is trivially true. This is the case if there are no
  /// uses.
  bool isTrue() const { return uses.none(); }

  /// Compare against another term.
  int compare(const AndTerm other) const;
  bool operator==(const AndTerm other) const { return compare(other) == 0; }
  bool operator!=(const AndTerm other) const { return compare(other) != 0; }
  bool operator<(const AndTerm other) const { return compare(other) < 0; }
  bool operator>(const AndTerm other) const { return compare(other) > 0; }
  bool operator<=(const AndTerm other) const { return compare(other) <= 0; }
  bool operator>=(const AndTerm other) const { return compare(other) >= 0; }

  /// Add a single use.
  void addUse(Use u) { uses.set(u); }
  /// Add multiple uses.
  void addUses(Uses u) { uses |= u; }
  /// Remove a single use.
  void removeUse(Use u) { uses.reset(u); }
  /// Remove multiple uses.
  void removeUses(Uses u) { uses &= ~u; }

  /// Check if this term has a specific use of the term's value.
  bool hasUse(Use u) const { return uses.test(u); }
  /// Check if this term has all of the specified uses of the term's value.
  bool hasAllUses(Uses u) const { return (uses & u) == u; }
  /// Check if this term has any of the specified uses of the term's value.
  bool hasAnyUses(Uses u) const { return (uses & u).any(); }

  /// Check if this term is a subset of another term. For example, `a & @a` is
  /// considered a subset of `a`.
  bool isSubsetOf(const AndTerm &other) const;

  /// Check if this term differs from another term by only a single flipped
  /// term. If it does, `flippedUse` is set to indicate the flipped term.
  bool hasSingleFlippedTerm(const AndTerm &other, Use &flippedUse) const;

  /// Print this term to `os`, using the given callback to print the value.
  void print(llvm::raw_ostream &os,
             llvm::function_ref<void(Value)> printValue = {}) const;
};

/// An individual term of an OR expression in a DNF. Consists of multiple terms
/// AND-ed together. A `true` value is represented as an empty list of AND
/// terms, since the neutral element of the AND operation is `true`.
struct OrTerm {
  using AndTerms = SmallVector<AndTerm, 1>;
  AndTerms andTerms;

  /// Create a constant `true` term.
  OrTerm() {}
  /// Create a term with a single element.
  OrTerm(AndTerm singleTerm) { andTerms.push_back(singleTerm); }
  /// Create a term with multiple elements. The terms must be sorted.
  OrTerm(AndTerms andTerms) : andTerms(andTerms) {
    assert(isSortedAndUnique());
  }

  /// Check if this term is trivially true.
  bool isTrue() const { return andTerms.empty(); }
  /// Check if this term is trivially false.
  bool isFalse() const {
    return llvm::any_of(andTerms, [](auto term) { return term.isFalse(); });
  }

  /// Compare against another term.
  int compare(const OrTerm &other) const;
  bool operator==(const OrTerm &other) const { return compare(other) == 0; }
  bool operator!=(const OrTerm &other) const { return compare(other) != 0; }
  bool operator<(const OrTerm &other) const { return compare(other) < 0; }
  bool operator>(const OrTerm &other) const { return compare(other) > 0; }
  bool operator<=(const OrTerm &other) const { return compare(other) <= 0; }
  bool operator>=(const OrTerm &other) const { return compare(other) >= 0; }

  /// Check if this term is a subset of another term. `a & b & c` is considered
  /// a subset of `a & b`.
  bool isSubsetOf(const OrTerm &other) const;

  /// Check if this term differs from another term by only a single flipped
  /// term. If it does, `flippedIdx` and `flippedUse` are set to indicate the
  /// flipped term.
  bool hasSingleFlippedTerm(const OrTerm &other, unsigned &flippedIdx,
                            AndTerm::Use &flippedUse) const;

  /// Return a hash code for this term that is invariant to inversion of
  /// individual terms.
  llvm::hash_code flipInvariantHash() const;

  /// Add an `AndTerm`. Returns false if adding the `AndTerm` has made this
  /// `OrTerm` trivially false.
  bool addTerm(AndTerm term);
  /// Add multiple `AndTerm`s. Returns false if adding the terms has made this
  /// `OrTerm` trivially false.
  bool addTerms(ArrayRef<AndTerm> terms);

  /// Check that all terms in this `OrTerm` are sorted. This is an invariant we
  /// want to check in builds with asserts enabled.
  bool isSortedAndUnique() const;

  /// Print this term to `os`, using the given callback to print the value.
  void print(llvm::raw_ostream &os,
             llvm::function_ref<void(Value)> printValue) const;
};

struct DNF {
  using OrTerms = SmallVector<OrTerm, 1>;
  OrTerms orTerms;

  /// Construct a constant `false`.
  DNF() {}
  /// Construct a constant `true` or `false`.
  explicit DNF(bool value) {
    if (value)
      orTerms.push_back(OrTerm());
  }
  /// Construct a DNF with a single opaque `Value`.
  explicit DNF(Value value) : DNF(AndTerm::id(value)) {}
  /// Construct a DNF with a given AND term inside a single OR term.
  explicit DNF(AndTerm value) : DNF(OrTerm(value)) {}
  /// Construct a DNF with a given OR term.
  explicit DNF(OrTerm value) { orTerms.push_back(value); }
  /// Construct a DNF with a multiple OR terms. The terms must be sorted.
  explicit DNF(OrTerms terms) : orTerms(terms) { assert(isSortedAndUnique()); }
  /// Construct a DNF with a single opaque past `Value`.
  static DNF withPastValue(Value value) { return DNF(AndTerm::past(value)); }

  /// Check whether this DNF is trivially false.
  bool isFalse() const { return orTerms.empty(); }
  /// Check whether this DNF is trivially true.
  bool isTrue() const { return orTerms.size() == 1 && orTerms[0].isTrue(); }

  /// Compare against another DNF.
  int compare(const DNF &other) const;
  bool operator==(const DNF &other) const { return compare(other) == 0; }
  bool operator!=(const DNF &other) const { return compare(other) != 0; }
  bool operator<(const DNF &other) const { return compare(other) < 0; }
  bool operator>(const DNF &other) const { return compare(other) > 0; }
  bool operator>=(const DNF &other) const { return compare(other) >= 0; }
  bool operator<=(const DNF &other) const { return compare(other) <= 0; }

  /// Compute the boolean OR of this and another DNF.
  DNF operator|(const DNF &other) const;
  /// Compute the boolean AND of this and another DNF.
  DNF operator&(const DNF &other) const;
  /// Compute the boolean XOR of this and another DNF.
  DNF operator^(const DNF &other) const;
  /// Compute the boolean NOT of this DNF.
  DNF operator~() const;

  DNF &operator|=(const DNF &other) { return *this = *this | other; }
  DNF &operator&=(const DNF &other) { return *this = *this & other; }
  DNF &operator^=(const DNF &other) { return *this = *this ^ other; }
  DNF &negate() { return *this = ~*this; }

  /// Check that all terms in the DNF are sorted. This is an invariant we want
  /// to check in builds with asserts enabled.
  bool isSortedAndUnique() const;

  /// Removes redundant terms as follows:
  /// - a&x | !a&x -> x  (eliminate complements)
  /// - a&x | x -> x     (eliminate supersets)
  void optimize();

  /// Print this DNF to `os`, using the given callback to print the value.
  void print(llvm::raw_ostream &os,
             llvm::function_ref<void(Value)> printValue) const;
  /// Print this DNF to `os`, followed by a list of the concrete values used.
  void printWithValues(llvm::raw_ostream &os) const;
};

} // namespace llhd
} // namespace circt
