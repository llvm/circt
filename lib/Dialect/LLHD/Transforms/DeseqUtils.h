//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace llhd {
namespace deseq {

//===----------------------------------------------------------------------===//
// ValueField
//===----------------------------------------------------------------------===//

/// Identify a specific subfield (or the whole) of an SSA value using the HW
/// field ID scheme. `fieldID == 0` means the whole value.
struct ValueField {
  /// The root SSA value being accessed (e.g. the full struct or array).
  Value value;
  /// The HW field ID describing which subfield is referenced. `0` means the
  /// whole value.
  uint64_t fieldID = 0;
  /// Optional bit/slice projection within the selected field. `0` means the
  /// whole field, otherwise this is `lowBit + 1` of a `comb.extract` applied to
  /// the field (and can be chained by addition).
  uint64_t bitID = 0;
  /// Width (in bits) of the final extracted slice. `0` means no explicit slice
  /// width is tracked.
  uint64_t bitWidth = 0;
  /// An optional SSA value that already materializes this specific subfield.
  /// For non-projection values, this is identical to `value`.
  Value projection;

  ValueField() = default;
  ValueField(Value value, uint64_t fieldID, Value projection = {},
             uint64_t bitID = 0, uint64_t bitWidth = 0)
      : value(value), fieldID(fieldID), bitID(bitID), bitWidth(bitWidth),
        projection(projection ? projection : value) {}

  Value getProjected() const { return projection ? projection : value; }

  bool operator==(const ValueField &other) const {
    return value == other.value && fieldID == other.fieldID &&
           bitID == other.bitID && bitWidth == other.bitWidth;
  }
  bool operator!=(const ValueField &other) const { return !(*this == other); }
  operator bool() const { return static_cast<bool>(value); }
};

//===----------------------------------------------------------------------===//
// Disjunctive Normal Form
//===----------------------------------------------------------------------===//

/// A single AND operation within a DNF.
struct DNFTerm {
  /// A mask with two bits for each possible term that may be present. Even bits
  /// indicate a term is present, odd bits indicate a term's negation is
  /// present.
  uint32_t andTerms = 0;

  bool isTrue() const { return andTerms == 0; }
  bool isFalse() const {
    // If any term has both the even and odd bits set, the term is false.
    return (andTerms & 0x55555555UL) & (andTerms >> 1);
  }

  bool operator==(DNFTerm other) const { return andTerms == other.andTerms; }
  bool operator!=(DNFTerm other) const { return !(*this == other); }
};

/// A boolean function expressed in canonical disjunctive normal form. Supports
/// functions with up to 32 terms. Consists of AND operations nested inside an
/// OR operation.
struct DNF {
  SmallVector<DNFTerm, 2> orTerms;
  bool isPoison() const { return orTerms.size() == 1 && orTerms[0].isFalse(); }
  bool isTrue() const { return orTerms.size() == 1 && orTerms[0].isTrue(); }
  bool isFalse() const { return orTerms.empty(); }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DNFTerm &term);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DNF &dnf);

//===----------------------------------------------------------------------===//
// Truth Table
//===----------------------------------------------------------------------===//

/// A boolean function expressed as a truth table. The first term is treated as
/// a special "unknown" value marker with special semantics. Truth tables have
/// exponential memory requirements. They are only useful to track a small
/// number of terms, say up to 16, which already requires 64 kB of memory.
struct TruthTable {
  /// The value of the boolean function for each possible combination of input
  /// term assignments. If the boolean function has N input terms, this `APInt`
  /// lists all `2**N` possible combinations. The special "poison" value is
  /// represented as a zero-width `bits`.
  APInt bits;

  /// Get the number of terms in the truth table. Since the width of `bits`
  /// corresponds to `2**numInputs`, we count the number of trailing zeros in
  /// the width of `bits` to determine `numInputs`.
  unsigned getNumTerms() const {
    if (isPoison())
      return 0;
    return llvm::countr_zero(bits.getBitWidth());
  }

  static TruthTable getPoison() { return TruthTable{APInt::getZero(0)}; }
  bool isPoison() const { return bits.getBitWidth() == 0; }
  bool isTrue() const { return bits.isAllOnes(); }
  bool isFalse() const { return bits.isZero(); }

  /// Create a boolean expression with a constant true or false value.
  static TruthTable getConst(unsigned numTerms, bool value) {
    assert(numTerms <= 16 && "excessive truth table");
    return TruthTable{value ? APInt::getAllOnes(1 << numTerms)
                            : APInt::getZero(1 << numTerms)};
  }

  /// Create a boolean expression consisting of a single term.
  static TruthTable getTerm(unsigned numTerms, unsigned term) {
    assert(numTerms <= 16 && "excessive truth table");
    assert(term < numTerms);
    return TruthTable{getTermMask(1 << numTerms, term)};
  }

  // Boolean operations.
  TruthTable operator~() const;
  TruthTable operator&(const TruthTable &other) const;
  TruthTable operator|(const TruthTable &other) const;
  TruthTable operator^(const TruthTable &other) const;

  TruthTable &invert();
  TruthTable &operator&=(const TruthTable &other);
  TruthTable &operator|=(const TruthTable &other);
  TruthTable &operator^=(const TruthTable &other);

  bool operator==(const TruthTable &other) const { return bits == other.bits; }
  bool operator!=(const TruthTable &other) const { return !(*this == other); }

  /// Convert the truth table into its canonical disjunctive normal form.
  DNF canonicalize() const;

private:
  /// Return a mask that has a 1 in all truth table rows where `term` is 1,
  /// and a 0 otherwise. This is equivalent to the truth table of a boolean
  /// expression consisting of a single term.
  static APInt getTermMask(unsigned bitWidth, unsigned term) {
    return APInt::getSplat(bitWidth,
                           APInt::getHighBitsSet(2 << term, 1 << term));
  }
  APInt getTermMask(unsigned term) const {
    return getTermMask(bits.getBitWidth(), term);
  }

  // Adjust any `!x` terms to be `x` terms.
  void fixupUnknown();
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TruthTable &table);

//===----------------------------------------------------------------------===//
// Value Table
//===----------------------------------------------------------------------===//

/// A single entry in a value table. Tracks the condition under which a value
/// appears. Can be set to a special "poison" and "unknown" marker.
struct ValueEntry {
  Value value;

  ValueEntry() = default;
  ValueEntry(Value value) : value(value) {}

  bool isPoison() const { return *this == getPoison(); }
  bool isUnknown() const { return *this == getUnknown(); }
  static ValueEntry getPoison() {
    return Value::getFromOpaquePointer((void *)(1));
  }
  static ValueEntry getUnknown() { return Value(); }

  void merge(ValueEntry other);

  bool operator==(ValueEntry other) const { return value == other.value; }
  bool operator!=(ValueEntry other) const { return value != other.value; }
};

/// A table of SSA values and the conditions under which they appear. This
/// struct can be used to track the various concrete values an SSA value may
/// assume depending on how control flow reaches it.
struct ValueTable {
  SmallVector<std::pair<TruthTable, ValueEntry>, 1> entries;

  ValueTable() = default;
  ValueTable(TruthTable condition, ValueEntry entry)
      : entries({{condition, entry}}) {}

  void addCondition(TruthTable condition);
  void merge(const ValueTable &other);
  void minimize();
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ValueEntry &entry);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const std::pair<TruthTable, ValueEntry> &pair);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ValueTable &table);

//===----------------------------------------------------------------------===//
// Clock and Reset Analysis
//===----------------------------------------------------------------------===//

/// A single reset extracted from a process during trigger analysis.
struct ResetInfo {
  /// The value acting as the reset, causing the register to be set to `value`
  /// when triggered.
  Value reset;
  /// The value the register is reset to.
  Value value;
  /// Whether the reset is active when high.
  bool activeHigh;

  /// Check if this reset info is null.
  operator bool() const { return bool(reset); }
};

/// A single clock extracted from a process during trigger analysis.
struct ClockInfo {
  /// The value acting as the clock, causing the register to be set to a value
  /// in `valueTable` when triggered.
  Value clock;
  /// The value the register is set to when the clock is triggered.
  Value value;
  /// Whether the clock is sensitive to a rising or falling edge.
  bool risingEdge;
  /// The optional value acting as an enable.
  Value enable;

  /// Check if this clock info is null.
  operator bool() const { return bool(clock); }
};

/// A drive op and the clock and reset that resulted from trigger analysis. A
/// process may describe multiple clock and reset triggers, but since the
/// registers we lower to only allow a single clock and a single reset, this
/// struct tracks a single clock and reset, respectively. Processes describing
/// multiple clocks or resets are skipped.
struct DriveInfo {
  /// The drive operation.
  DriveOp op;
  /// The clock that triggers a change to the driven value. Guaranteed to be
  /// non-null.
  ClockInfo clock;
  /// The optional reset that triggers a change of the driven value to a fixed
  /// reset value. Null if no reset was detected.
  ResetInfo reset;

  DriveInfo() {}
  explicit DriveInfo(DriveOp op) : op(op) {}
};

//===----------------------------------------------------------------------===//
// Value Assignments for Process Specialization
//===----------------------------------------------------------------------===//

/// A single `i1` value that is fixed to a given value in the past and the
/// present.
struct FixedValue {
  /// The IR value being fixed.
  Value value;
  /// The assigned value in the past, as transported into the presented via a
  /// destination operand of a process' wait op.
  bool past;
  /// The assigned value in the present.
  bool present;

  bool operator==(const FixedValue &other) const {
    return value == other.value && past == other.past &&
           present == other.present;
  }

  operator bool() const { return bool(value); }
};

/// A list of `i1` values that are fixed to a given value. These are used when
/// specializing a process to compute the value and enable condition for a drive
/// when a trigger occurs.
using FixedValues = SmallVector<FixedValue, 2>;

static inline llvm::hash_code hash_value(const FixedValue &arg) {
  return llvm::hash_combine(arg.value, arg.past, arg.present);
}

static inline llvm::hash_code hash_value(const FixedValues &arg) {
  return llvm::hash_combine_range(arg.begin(), arg.end());
}

} // namespace deseq
} // namespace llhd
} // namespace circt

namespace llvm {

// Allow `FixedValue` to be used as hash map key.
template <>
struct DenseMapInfo<circt::llhd::deseq::FixedValue> {
  using Value = mlir::Value;
  using FixedValue = circt::llhd::deseq::FixedValue;
  static inline FixedValue getEmptyKey() {
    return FixedValue{DenseMapInfo<Value>::getEmptyKey(), false, false};
  }
  static inline FixedValue getTombstoneKey() {
    return FixedValue{DenseMapInfo<Value>::getTombstoneKey(), false, false};
  }
  static unsigned getHashValue(const FixedValue &key) {
    return hash_value(key);
  }
  static bool isEqual(const FixedValue &a, const FixedValue &b) {
    return a == b;
  }
};

// Allow `FixedValues` to be used as hash map key.
template <>
struct DenseMapInfo<circt::llhd::deseq::FixedValues> {
  using FixedValue = circt::llhd::deseq::FixedValue;
  using FixedValues = circt::llhd::deseq::FixedValues;
  static inline FixedValues getEmptyKey() {
    return {DenseMapInfo<FixedValue>::getEmptyKey()};
  }
  static inline FixedValues getTombstoneKey() {
    return {DenseMapInfo<FixedValue>::getTombstoneKey()};
  }
  static unsigned getHashValue(const FixedValues &key) {
    return hash_value(key);
  }
  static bool isEqual(const FixedValues &a, const FixedValues &b) {
    return a == b;
  }
};

// Allow `ValueField` to be used as a DenseMap key.
template <>
struct DenseMapInfo<circt::llhd::deseq::ValueField> {
  using VF = circt::llhd::deseq::ValueField;
  static inline VF getEmptyKey() {
    return VF(DenseMapInfo<mlir::Value>::getEmptyKey(), ~0ULL);
  }
  static inline VF getTombstoneKey() {
    return VF(DenseMapInfo<mlir::Value>::getTombstoneKey(), ~0ULL - 1);
  }
  static unsigned getHashValue(const VF &key) {
    return DenseMapInfo<mlir::Value>::getHashValue(key.value) ^
           DenseMapInfo<uint64_t>::getHashValue(key.fieldID) ^
           DenseMapInfo<uint64_t>::getHashValue(key.bitID) ^
           DenseMapInfo<uint64_t>::getHashValue(key.bitWidth);
  }
  static bool isEqual(const VF &a, const VF &b) { return a == b; }
};

} // namespace llvm
