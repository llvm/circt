//===- values.h - ESI value system -------------------------------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI arbitrary width bitvector and integer types.
// These types are not meant to be highly optimized. Rather, its a simple
// implementation to support arbitrary bit widths for ESI runtime values.
//
//===----------------------------------------------------------------------===//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_VALUES_H
#define ESI_VALUES_H

#include <cstdint>
#include <memory> // (may be removable later)
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace esi {

/// A lightweight, non-owning (but optionally owning) bit vector view backed by
/// a byte array. Supports bit-level access, logical width-shrinking shifts, and
/// bitwise boolean operations. Bit 0 is the least-significant bit (LSB) of the
/// first addressed bit in the underlying span. The first addressed bit may be
/// offset into the first byte by `bitIndex` (0-7).
class BitVector {
public:
  using byte = uint8_t;

  /// Construct from an existing span. Width defaults to the number of bits in
  /// the span (size * 8). The BitVector does not take ownership unless an
  /// owning constructor is used.
  BitVector(std::span<const byte> bytes,
            std::optional<size_t> width = std::nullopt, uint8_t bitIndex = 0);

  /// Owning constructor from an rvalue vector (must move in).
  BitVector(std::vector<byte> &&bytes,
            std::optional<size_t> width = std::nullopt, uint8_t bitIndex = 0);

  /// Owning, zero-initialized constructor of a given width.
  explicit BitVector(size_t width);

  BitVector() = default;

  // Copy constructor: if owning, duplicate storage and rebind span.
  BitVector(const BitVector &other);

  // Move constructor: transfer ownership and rebind span if owning.
  BitVector(BitVector &&other) noexcept;

  BitVector &operator=(const BitVector &other);

  BitVector &operator=(BitVector &&other) noexcept;

  size_t width() const { return bitWidth; }

  /// Return the i-th bit (0 = least-significant) as boolean.
  bool getBit(size_t i) const;

  /// Set the i-th bit.
  void setBit(size_t i, bool v);

  /// Return a handle to the underlying span. Throws if the current bit index
  /// is not 0 (since a non-zero bit offset breaks raw byte alignment).
  std::span<const byte> getSpan() const {
    if (bitIndex != 0)
      throw std::runtime_error("Cannot get data span with non-zero bit index");
    return data;
  }

  /// Logical right shift that drops the least-significant n bits by advancing
  /// the start pointer / bitIndex and reducing width. Does not modify the
  /// underlying storage contents.
  BitVector &operator>>=(size_t n);
  BitVector operator>>(size_t n) const;

  /// Logical left shift that drops the most-significant n bits (width
  /// shrink).
  BitVector &operator<<=(size_t n);
  BitVector operator<<(size_t n) const;
  BitVector &operator<<=(const BitVector &other); // concatenate
  friend BitVector operator<<(const BitVector &a,
                              const BitVector &b); // concatenate copy

  enum class BitwiseKind { And, Or, Xor };
  friend BitVector operator&(const BitVector &a, const BitVector &b) {
    return bitwiseOp(a, b, BitwiseKind::And);
  }
  friend BitVector operator|(const BitVector &a, const BitVector &b) {
    return bitwiseOp(a, b, BitwiseKind::Or);
  }
  friend BitVector operator^(const BitVector &a, const BitVector &b) {
    return bitwiseOp(a, b, BitwiseKind::Xor);
  }

  std::string toString(unsigned base = 16) const;

  bool operator==(const BitVector &rhs) const;
  bool operator!=(const BitVector &rhs) const { return !(*this == rhs); }

  // Trucate to new width (must be less than or equal to current width).
  // Returns a non-owning view into the same underlying storage.
  /// Create an owning copy of a contiguous bit slice [offset,
  /// offset+sliceWidth). The returned BitVector always has bitIndex=0 and
  /// owns its storage so the bits are densely packed starting at LSB. Throws
  /// if the requested slice exceeds the current width.
  BitVector slice(size_t offset, size_t sliceWidth) const;

protected:
  static BitVector bitwiseOp(const BitVector &a, const BitVector &b,
                             BitwiseKind kind);

  // Optional ownership of storage. If present, this enables modification
  // functionalities on this BitVector.
  std::optional<std::vector<byte>> owner;

  // Underlying storage view. const, to allow for non-owning immutable views.
  std::span<const byte> data{};
  size_t bitWidth = 0;  // Number of valid bits.
  uint8_t bitIndex = 0; // Starting bit offset in first byte.
};

std::ostream &operator<<(std::ostream &os, const BitVector &bv);

/// Signed arbitrary precision integer (two's complement) built atop
/// BitVector.
class Int : public BitVector {
public:
  Int() = default;
  Int(int64_t value, size_t width) : BitVector(width) { assign(value); }
  Int(const BitVector &bv) : BitVector(bv) {}
  Int(BitVector &&bv) : BitVector(bv) {}

  // cstdint conversion operators.
  operator int64_t() const { return toSigned64(); }

  // Signed comparisons (numeric, with sign extension as needed between
  // widths).
  friend bool operator==(const Int &a, const Int &b) {
    return compare(a, b) == 0;
  }
  friend bool operator!=(const Int &a, const Int &b) {
    return compare(a, b) != 0;
  }
  friend bool operator<(const Int &a, const Int &b) {
    return compare(a, b) < 0;
  }
  friend bool operator<=(const Int &a, const Int &b) {
    return compare(a, b) <= 0;
  }
  friend bool operator>(const Int &a, const Int &b) {
    return compare(a, b) > 0;
  }
  friend bool operator>=(const Int &a, const Int &b) {
    return compare(a, b) >= 0;
  }
  operator uint64_t() const { return toUnsigned64(); }

  friend Int operator+(const Int &a, const Int &b) { return add(a, b); }
  friend Int operator-(const Int &a, const Int &b) { return add(a, negate(b)); }
  friend Int operator*(const Int &a, const Int &b) { return multiply(a, b); }

  Int &operator+=(const Int &o) { return *this = *this + o; }
  Int &operator-=(const Int &o) { return *this = *this - o; }
  Int &operator*=(const Int &o) { return *this = *this * o; }

private:
  void assign(int64_t value);
  int64_t toSigned64() const;
  uint64_t toUnsigned64() const;
  static Int add(const Int &a, const Int &b);
  static Int negate(const Int &v);
  static Int multiply(const Int &a, const Int &b);
  static int compare(const Int &a, const Int &b); // -1,0,1
};

/// Unsigned arbitrary precision integer built atop BitVector.
class UInt : public BitVector {
public:
  UInt() = default;
  UInt(uint64_t value, size_t width) : BitVector(width) { assign(value); }
  UInt(const BitVector &bv) : BitVector(bv) {}
  UInt(BitVector &&bv) : BitVector(bv) {}

  operator uint64_t() const { return toUnsigned64(); }

  // Unsigned comparisons (zero extend differing widths).
  friend bool operator==(const UInt &a, const UInt &b) {
    return compare(a, b) == 0;
  }
  friend bool operator!=(const UInt &a, const UInt &b) {
    return compare(a, b) != 0;
  }
  friend bool operator<(const UInt &a, const UInt &b) {
    return compare(a, b) < 0;
  }
  friend bool operator<=(const UInt &a, const UInt &b) {
    return compare(a, b) <= 0;
  }
  friend bool operator>(const UInt &a, const UInt &b) {
    return compare(a, b) > 0;
  }
  friend bool operator>=(const UInt &a, const UInt &b) {
    return compare(a, b) >= 0;
  }

  friend UInt operator+(const UInt &a, const UInt &b) { return add(a, b); }
  friend UInt operator-(const UInt &a, const UInt &b) { return subtract(a, b); }
  friend UInt operator*(const UInt &a, const UInt &b) { return multiply(a, b); }

  UInt &operator+=(const UInt &o) { return *this = *this + o; }
  UInt &operator-=(const UInt &o) { return *this = *this - o; }
  UInt &operator*=(const UInt &o) { return *this = *this * o; }

private:
  void assign(uint64_t value);
  uint64_t toUnsigned64() const;
  static UInt add(const UInt &a, const UInt &b);
  static UInt subtract(const UInt &a, const UInt &b);
  static UInt multiply(const UInt &a, const UInt &b);
  static int compare(const UInt &a, const UInt &b); // -1,0,1
};

} // namespace esi

#endif // ESI_VALUES_H