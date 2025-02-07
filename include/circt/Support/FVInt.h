//===- FVInt.h - Four-valued integer ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a class to represent arbitrary precision integers where
// each bit can be one of four values. This corresponds to SystemVerilog's
// four-valued `logic` type (originally defined in IEEE 1364, later merged into
// IEEE 1800).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_FVINT_H
#define CIRCT_SUPPORT_FVINT_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
template <typename T, typename Enable>
struct DenseMapInfo;
} // namespace llvm

namespace circt {

/// Four-valued arbitrary precision integers.
///
/// Each bit of the integer can be 0, 1, X, or Z. Internally the bits are stored
/// in a pair of `APInt`s, one of which specifies the value of each bit (0/X or
/// 1/Z), and the other whether the bit is unknown (X or Z).
class FVInt {
public:
  /// Default constructor that creates an zero-bit zero value.
  explicit FVInt() : FVInt(0, 0) {}

  /// Construct an `FVInt` from a 64-bit value. The result has no X or Z bits.
  FVInt(unsigned numBits, uint64_t value, bool isSigned = false)
      : FVInt(APInt(numBits, value, isSigned)) {}

  /// Construct an `FVInt` from an `APInt`. The result has no X or Z bits.
  FVInt(const APInt &value)
      : value(value), unknown(APInt::getZero(value.getBitWidth())) {}

  /// Construct an `FVInt` from two `APInt`s used internally to store the bit
  /// data. The first argument specifies whether each bit is 0/X or 1/Z. The
  /// second argument specifies whether each bit is 0/1 or X/Z. Both `APInt`s
  /// must have the same bitwidth. The two arguments correspond to the results
  /// of `getRawValue()` and `getRawUnknown()`.
  FVInt(APInt &&rawValue, APInt &&rawUnknown)
      : value(rawValue), unknown(rawUnknown) {
    assert(rawValue.getBitWidth() == rawUnknown.getBitWidth());
  }

  /// Construct an `FVInt` with all bits set to 0.
  static FVInt getZero(unsigned numBits) {
    return FVInt(APInt::getZero(numBits));
  }

  /// Construct an `FVInt` with all bits set to 1.
  static FVInt getAllOnes(unsigned numBits) {
    return FVInt(APInt::getAllOnes(numBits));
  }

  /// Construct an `FVInt` with all bits set to X.
  static FVInt getAllX(unsigned numBits) {
    return FVInt(APInt::getZero(numBits), APInt::getAllOnes(numBits));
  }

  /// Construct an `FVInt` with all bits set to Z.
  static FVInt getAllZ(unsigned numBits) {
    return FVInt(APInt::getAllOnes(numBits), APInt::getAllOnes(numBits));
  }

  /// Return the number of bits this integer has.
  unsigned getBitWidth() const { return value.getBitWidth(); }

  /// Compute the number of active bits in the value. This is the smallest bit
  /// width to which the value can be truncated without losing information in
  /// the most significant bits. Or put differently, the value truncated to its
  /// active bits and zero-extended back to its original width produces the
  /// original value.
  unsigned getActiveBits() const {
    return std::max(value.getActiveBits(), unknown.getActiveBits());
  }

  /// Compute the minimum bit width necessary to accurately represent this
  /// integer's value and sign. This is the smallest bit width to which the
  /// value can be truncated without losing information in the most significant
  /// bits and without flipping from negative to positive or vice versa. Or put
  /// differently, the value truncated to its significant bits and sign-extended
  /// back to its original width produces the original value.
  unsigned getSignificantBits() const {
    return std::max(value.getSignificantBits(), unknown.getSignificantBits());
  }

  /// Return the underlying `APInt` used to store whether a bit is 0/X or 1/Z.
  const APInt &getRawValue() const { return value; }

  /// Return the underlying `APInt` used to store whether a bit is unknown (X or
  /// Z).
  const APInt &getRawUnknown() const { return unknown; }

  /// Convert the four-valued `FVInt` to a two-valued `APInt` by mapping X and Z
  /// bits to either 0 or 1.
  APInt toAPInt(bool unknownBitMapping) const {
    auto v = value;
    if (unknownBitMapping)
      v |= unknown; // set unknown bits to 1
    else
      v &= ~unknown; // set unknown bits to 0
    return v;
  }

  //===--------------------------------------------------------------------===//
  // Resizing
  //===--------------------------------------------------------------------===//

  /// Truncate the integer to a smaller bit width. This simply discards the
  /// high-order bits. If the integer is truncated to a bit width less than its
  /// "active bits", information will be lost and the resulting integer will
  /// have a different value.
  FVInt trunc(unsigned bitWidth) const {
    assert(bitWidth <= getBitWidth());
    return FVInt(value.trunc(bitWidth), unknown.trunc(bitWidth));
  }

  /// Zero-extend the integer to a new bit width. The additional high-order bits
  /// are filled in with zero.
  FVInt zext(unsigned bitWidth) const {
    assert(bitWidth >= getBitWidth());
    return FVInt(value.zext(bitWidth), unknown.zext(bitWidth));
  }

  /// Sign-extend the integer to a new bit width. The additional high-order bits
  /// are filled in with the sign bit (top-most bit) of the original number,
  /// also when that sign bit is X or Z. Zero-width integers are extended with
  /// zeros.
  FVInt sext(unsigned bitWidth) const {
    assert(bitWidth >= getBitWidth());
    return FVInt(value.sext(bitWidth), unknown.sext(bitWidth));
  }

  /// Truncate or zero-extend to a target bit width.
  FVInt zextOrTrunc(unsigned bitWidth) const {
    return bitWidth > getBitWidth() ? zext(bitWidth) : trunc(bitWidth);
  }

  /// Truncate or sign-extend to a target bit width.
  FVInt sextOrTrunc(unsigned bitWidth) const {
    return bitWidth > getBitWidth() ? sext(bitWidth) : trunc(bitWidth);
  }

  //===--------------------------------------------------------------------===//
  // Value Tests
  //===--------------------------------------------------------------------===//

  /// Determine if any bits are X or Z.
  bool hasUnknown() const { return !unknown.isZero(); }

  /// Determine if all bits are 0. This is true for zero-width values.
  bool isZero() const { return value.isZero() && unknown.isZero(); }

  /// Determine if all bits are 1. This is true for zero-width values.
  bool isAllOnes() const { return value.isAllOnes() && unknown.isZero(); }

  /// Determine if all bits are X. This is true for zero-width values.
  bool isAllX() const { return value.isZero() && unknown.isAllOnes(); }

  /// Determine if all bits are Z. This is true for zero-width values.
  bool isAllZ() const { return value.isAllOnes() && unknown.isAllOnes(); }

  /// Determine whether the integer interpreted as a signed number would be
  /// negative. Returns true if the sign bit is 1, and false if it is 0, X, or
  /// Z.
  bool isNegative() const {
    auto idx = getBitWidth() - 1;
    return value[idx] && !unknown[idx];
  }

  //===--------------------------------------------------------------------===//
  // Bit Manipulation
  //===--------------------------------------------------------------------===//

  /// The value of an individual bit. Can be 0, 1, X, or Z.
  enum Bit { V0 = 0b00, V1 = 0b01, X = 0b10, Z = 0b11 };

  /// Get the value of an individual bit.
  Bit getBit(unsigned index) const {
    return static_cast<Bit>(value[index] | unknown[index] << 1);
  }

  /// Set the value of an individual bit.
  void setBit(unsigned index, Bit bit) {
    value.setBitVal(index, (bit >> 0) & 1);
    unknown.setBitVal(index, (bit >> 1) & 1);
  }

  void setBit(unsigned index, bool val) {
    setBit(index, static_cast<Bit>(val));
  }

  /// Compute a mask of all the 0 bits in this integer.
  APInt getZeroBits() const { return ~value & ~unknown; }

  /// Compute a mask of all the 1 bits in this integer.
  APInt getOneBits() const { return value & ~unknown; }

  /// Compute a mask of all the X bits in this integer.
  APInt getXBits() const { return ~value & unknown; }

  /// Compute a mask of all the Z bits in this integer.
  APInt getZBits() const { return value & unknown; }

  /// Compute a mask of all the X and Z bits in this integer.
  APInt getUnknownBits() const { return unknown; }

  /// Set the value of all bits in the mask to 0.
  template <typename T>
  void setZeroBits(const T &mask) {
    value &= ~mask;
    unknown &= ~mask;
  }

  /// Set the value of all bits in the mask to 1.
  template <typename T>
  void setOneBits(const T &mask) {
    value |= mask;
    unknown &= ~mask;
  }

  /// Set the value of all bits in the mask to X.
  template <typename T>
  void setXBits(const T &mask) {
    value &= ~mask;
    unknown |= mask;
  }

  /// Set the value of all bits in the mask to Z.
  template <typename T>
  void setZBits(const T &mask) {
    value |= mask;
    unknown |= mask;
  }

  /// Set all bits to 0.
  void setAllZero() {
    value.clearAllBits();
    unknown.clearAllBits();
  }

  /// Set all bits to 1.
  void setAllOne() {
    value.setAllBits();
    unknown.clearAllBits();
  }

  /// Set all bits to X.
  void setAllX() {
    value.clearAllBits();
    unknown.setAllBits();
  }

  /// Set all bits to Z.
  void setAllZ() {
    value.setAllBits();
    unknown.setAllBits();
  }

  /// Replace all Z bits with X. This is useful since most logic operations will
  /// treat X and Z bits the same way and produce an X bit in the output. By
  /// mapping Z bits to X, these operations can then just handle 0, 1, and X
  /// bits.
  void replaceZWithX() {
    // Z bits have value and unknown set to 1. X bits have value set to 0 and
    // unknown set to 1. To convert between the two, make sure that value is 0
    // wherever unknown is 1.
    value &= ~unknown;
  }

  /// If any bits are X or Z, set the entire integer to X.
  void setAllXIfAnyUnknown() {
    if (hasUnknown())
      setAllX();
  }

  /// If any bits in this integer or another integer are X or Z, set the entire
  /// integer to X. This is useful for binary operators which want to set their
  /// result to X if either of the two inputs contained an X or Z bit.
  void setAllXIfAnyUnknown(const FVInt &other) {
    if (hasUnknown() || other.hasUnknown())
      setAllX();
  }

  //===--------------------------------------------------------------------===//
  // Shift Operators
  //===--------------------------------------------------------------------===//

  /// Perform a logical left-shift. If any bits in the shift amount are unknown,
  /// the entire result is X.
  FVInt &operator<<=(const FVInt &amount) {
    if (amount.hasUnknown()) {
      setAllX();
    } else {
      value <<= amount.value;
      unknown <<= amount.value;
    }
    return *this;
  }

  /// Perform a logical left-shift by a two-valued amount.
  template <typename T>
  FVInt &operator<<=(const T &amount) {
    value <<= amount;
    unknown <<= amount;
    return *this;
  }

  //===--------------------------------------------------------------------===//
  // Logic Operators
  //===--------------------------------------------------------------------===//

  /// Compute the logical NOT of this integer. This implements the following
  /// bit-wise truth table:
  /// ```
  /// 0 | 1
  /// 1 | 0
  /// X | X
  /// Z | X
  /// ```
  void flipAllBits() {
    value = ~value;
    replaceZWithX();
  }

  /// Compute the logical NOT.
  FVInt operator~() const {
    auto v = *this;
    v.flipAllBits();
    return v;
  }

  /// Compute the logical AND of this integer and another. This implements the
  /// following bit-wise truth table:
  /// ```
  ///     0 1 X Z
  ///   +--------
  /// 0 | 0 0 0 0
  /// 1 | 0 1 X X
  /// X | 0 X X X
  /// Z | 0 X X X
  /// ```
  FVInt &operator&=(const FVInt &other) {
    auto zeros = getZeroBits() | other.getZeroBits();
    value &= other.value;
    unknown |= other.unknown;
    unknown &= ~zeros;
    replaceZWithX();
    return *this;
  }

  /// Compute the logical AND of this integer and a two-valued integer.
  template <typename T>
  FVInt &operator&=(T other) {
    value &= other;
    unknown &= other; // make 0 bits known
    replaceZWithX();
    return *this;
  }

  /// Compute the logical AND.
  template <typename T>
  FVInt operator&(const T &other) const {
    auto v = *this;
    v &= other;
    return v;
  }

  /// Compute the logical OR of this integer and another. This implements the
  /// following bit-wise truth table:
  /// ```
  ///     0 1 X Z
  ///   +--------
  /// 0 | 0 1 X X
  /// 1 | 1 1 1 1
  /// X | X 1 X X
  /// Z | X 1 X X
  /// ```
  FVInt &operator|=(const FVInt &other) {
    auto ones = getOneBits() | other.getOneBits();
    value |= other.value;
    unknown |= other.unknown;
    unknown &= ~ones;
    replaceZWithX();
    return *this;
  }

  /// Compute the logical OR of this integer and a two-valued integer.
  template <typename T>
  FVInt &operator|=(T other) {
    value |= other;
    unknown &= ~other; // make 1 bits known
    replaceZWithX();
    return *this;
  }

  /// Compute the logical OR.
  template <typename T>
  FVInt operator|(const T &other) const {
    auto v = *this;
    v |= other;
    return v;
  }

  /// Compute the logical XOR of this integer and another. This implements the
  /// following bit-wise truth table:
  /// ```
  ///     0 1 X Z
  ///   +--------
  /// 0 | 0 1 X X
  /// 1 | 1 0 X X
  /// X | X X X X
  /// Z | X X X X
  /// ```
  FVInt &operator^=(const FVInt &other) {
    value ^= other.value;
    unknown |= other.unknown;
    replaceZWithX();
    return *this;
  }

  /// Compute the logical XOR of this integer and a two-valued integer.
  template <typename T>
  FVInt &operator^=(const T &other) {
    value ^= other;
    replaceZWithX();
    return *this;
  }

  /// Compute the logical XOR.
  template <typename T>
  FVInt operator^(const T &other) const {
    auto v = *this;
    v ^= other;
    return v;
  }

  //===--------------------------------------------------------------------===//
  // Arithmetic Operators
  //===--------------------------------------------------------------------===//

  /// Compute the negation of this integer. If any bits are unknown, the entire
  /// result is X.
  void negate() {
    value.negate();
    setAllXIfAnyUnknown();
  }

  /// Compute the negation of this integer.
  FVInt operator-() const {
    auto v = *this;
    v.negate();
    return v;
  }

  /// Compute the addition of this integer and another. If any bits in either
  /// integer are unknown, the entire result is X.
  FVInt &operator+=(const FVInt &other) {
    value += other.value;
    setAllXIfAnyUnknown(other);
    return *this;
  }

  /// Compute the addition of this integer and a two-valued integer. If any bit
  /// in the integer is unknown, the entire result is X.
  template <typename T>
  FVInt &operator+=(const T &other) {
    value += other;
    setAllXIfAnyUnknown();
    return *this;
  }

  /// Compute an addition.
  template <typename T>
  FVInt operator+(const T &other) const {
    auto v = *this;
    v += other;
    return v;
  }

  /// Compute the subtraction of this integer and another. If any bits in either
  /// integer are unknown, the entire result is X.
  FVInt &operator-=(const FVInt &other) {
    value -= other.value;
    setAllXIfAnyUnknown(other);
    return *this;
  }

  /// Compute the subtraction of this integer and a two-valued integer. If any
  /// bit in the integer is unknown, the entire result is X.
  template <typename T>
  FVInt &operator-=(const T &other) {
    value -= other;
    setAllXIfAnyUnknown();
    return *this;
  }

  /// Compute an subtraction.
  template <typename T>
  FVInt operator-(const T &other) const {
    auto v = *this;
    v -= other;
    return v;
  }

  /// Compute the multiplication of this integer and another. If any bits in
  /// either integer are unknown, the entire result is X.
  FVInt &operator*=(const FVInt &other) {
    value *= other.value;
    setAllXIfAnyUnknown(other);
    return *this;
  }

  /// Compute the multiplication of this integer and a two-valued integer. If
  /// any bit in the integer is unknown, the entire result is X.
  template <typename T>
  FVInt &operator*=(const T &other) {
    value *= other;
    setAllXIfAnyUnknown();
    return *this;
  }

  /// Compute a multiplication.
  template <typename T>
  FVInt operator*(const T &other) const {
    auto v = *this;
    v *= other;
    return v;
  }

  //===--------------------------------------------------------------------===//
  // Comparison
  //===--------------------------------------------------------------------===//

  /// Determine whether this integer is equal to another. Note that this
  /// corresponds to SystemVerilog's `===` operator. Returns false if the two
  /// integers have different bit width.
  bool operator==(const FVInt &other) const {
    if (getBitWidth() != other.getBitWidth())
      return false;
    return value == other.value && unknown == other.unknown;
  }

  /// Determine whether this integer is equal to a two-valued integer. Note that
  /// this corresponds to SystemVerilog's `===` operator.
  template <typename T>
  bool operator==(const T &other) const {
    return value == other && !hasUnknown();
  }

  /// Determine whether this integer is not equal to another. Returns true if
  /// the two integers have different bit width.
  bool operator!=(const FVInt &other) const { return !((*this) == other); }

  /// Determine whether this integer is not equal to a two-valued integer.
  template <typename T>
  bool operator!=(const T &other) const {
    return !((*this) == other);
  }

  //===--------------------------------------------------------------------===//
  // String Conversion
  //===--------------------------------------------------------------------===//

  /// Convert a string into an `FVInt`.
  ///
  /// The radix can be 2, 8, 10, or 16. For radix 2, the input string may
  /// contain the characters `x` or `X` to indicate an unknown X bit, and `z` or
  /// `Z` to indicate an unknown Z bit. For radix 8, each X or Z counts as 3
  /// bits. For radix 16, each X and Z counts as 4 bits. When radix is 10 the
  /// input cannot contain any X or Z.
  ///
  /// Returns the parsed integer if the string is non-empty and a well-formed
  /// number, otherwise returns none.
  static std::optional<FVInt> tryFromString(StringRef str, unsigned radix = 10);

  /// Convert a string into an `FVInt`. Same as `tryFromString`, but aborts if
  /// the string is malformed.
  static FVInt fromString(StringRef str, unsigned radix = 10) {
    auto v = tryFromString(str, radix);
    assert(v.has_value() && "string is not a well-formed FVInt");
    return *v;
  }

  /// Convert an `FVInt` to a string.
  ///
  /// The radix can be 2, 8, 10, or 16. For radix 8 or 16, the integer can only
  /// contain unknown bits in groups of 3 or 4, respectively, such that a `X` or
  /// `Z` can be printed for the entire group of bits. For radix 10, the integer
  /// cannot contain any unknown bits. In case the output contains letters,
  /// `uppercase` specifies whether they are printed as uppercase letters.
  ///
  /// Appends the output characters to `str` and returns true if the integer
  /// could be printed with the given configuration. Otherwise returns false and
  /// leaves `str` in its original state. Always succeeds for radix 2.
  bool tryToString(SmallVectorImpl<char> &str, unsigned radix = 10,
                   bool uppercase = true) const;

  /// Convert an `FVInt` to a string. Same as `tryToString`, but directly
  /// returns the string and aborts if the conversion is unsuccessful.
  SmallString<16> toString(unsigned radix = 10, bool uppercase = true) const {
    SmallString<16> str;
    bool success = tryToString(str, radix, uppercase);
    (void)success;
    assert(success && "radix cannot represent FVInt");
    return str;
  }

  /// Print an `FVInt` to an output stream.
  void print(raw_ostream &os) const;

private:
  APInt value;
  APInt unknown;
};

inline FVInt operator&(uint64_t a, const FVInt &b) { return b & a; }
inline FVInt operator|(uint64_t a, const FVInt &b) { return b | a; }
inline FVInt operator^(uint64_t a, const FVInt &b) { return b ^ a; }
inline FVInt operator+(uint64_t a, const FVInt &b) { return b + a; }
inline FVInt operator*(uint64_t a, const FVInt &b) { return b * a; }

inline FVInt operator&(const APInt &a, const FVInt &b) { return b & a; }
inline FVInt operator|(const APInt &a, const FVInt &b) { return b | a; }
inline FVInt operator^(const APInt &a, const FVInt &b) { return b ^ a; }
inline FVInt operator+(const APInt &a, const FVInt &b) { return b + a; }
inline FVInt operator*(const APInt &a, const FVInt &b) { return b * a; }

inline FVInt operator-(uint64_t a, const FVInt &b) {
  return FVInt(b.getBitWidth(), a) - b;
}

inline FVInt operator-(const APInt &a, const FVInt &b) { return FVInt(a) - b; }

inline bool operator==(uint64_t a, const FVInt &b) { return b == a; }
inline bool operator!=(uint64_t a, const FVInt &b) { return b != a; }

inline raw_ostream &operator<<(raw_ostream &os, const FVInt &value) {
  value.print(os);
  return os;
}

llvm::hash_code hash_value(const FVInt &a);

/// Print a four-valued integer usign an `AsmPrinter`. This produces the
/// following output formats:
///
/// - Decimal notation if the integer has no unknown bits. The sign bit is used
///   to determine whether the value is printed as negative number or not.
/// - Hexadecimal notation with a leading `h` if the integer the bits in each
///   hex digit are either all known, all X, or all Z.
/// - Binary notation with a leading `b` in all other cases.
void printFVInt(AsmPrinter &p, const FVInt &value);

/// Parse a four-valued integer using an `AsmParser`. This accepts the following
/// formats:
///
/// - `42`/`-42`: positive or negative integer in decimal notation. The sign bit
///   of the result indicates whether the value was negative. Cannot contain
///   unknown X or Z digits.
/// - `h123456789ABCDEF0XZ`: signless integer in hexadecimal notation. Can
///   contain unknown X or Z digits.
/// - `b10XZ`: signless integer in binary notation. Can contain unknown X or Z
///   digits.
///
/// The result has enough bits to fully represent the parsed integer, and to
/// have the sign bit encode whether the integer was written as a negative
/// number in the input. The result's bit width may be larger than the minimum
/// number of bits required to represent its value.
ParseResult parseFVInt(AsmParser &p, FVInt &result);

} // namespace circt

namespace llvm {
/// Provide DenseMapInfo for FVInt.
template <>
struct DenseMapInfo<circt::FVInt, void> {
  static inline circt::FVInt getEmptyKey() {
    return circt::FVInt(DenseMapInfo<APInt>::getEmptyKey());
  }

  static inline circt::FVInt getTombstoneKey() {
    return circt::FVInt(DenseMapInfo<APInt>::getTombstoneKey());
  }

  static unsigned getHashValue(const circt::FVInt &Key);

  static bool isEqual(const circt::FVInt &LHS, const circt::FVInt &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // CIRCT_SUPPORT_FVINT_H
