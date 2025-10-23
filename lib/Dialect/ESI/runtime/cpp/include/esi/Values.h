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
#include <format>
#include <memory> // (may be removable later)
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace esi {

class MutableBitVector;

/// A lightweight, non-owning bit vector view backed by a byte array span.
/// BitVector is immutable and provides read-only access to bits.
/// It supports bit-level access and returns new views for operations.
/// Bit 0 is the least-significant bit (LSB) of the first addressed bit in the
/// underlying span. The first addressed bit may be offset into the first byte
/// by `bitIndex` (0-7).
class BitVector {
public:
  using byte = uint8_t;

  /// Construct from an existing span. Width defaults to the number of bits in
  /// the span (size * 8). The BitVector does not take ownership.
  BitVector(std::span<const byte> bytes,
            std::optional<size_t> width = std::nullopt, uint8_t bitIndex = 0);

  BitVector() = default;

  // Copy constructor.
  BitVector(const BitVector &other);
  BitVector &operator=(const BitVector &other);

  size_t width() const { return bitWidth; }
  size_t size() const { return width(); }

  /// Return the i-th bit (0 = least-significant) as boolean.
  bool getBit(size_t i) const;

  /// Return a handle to the underlying span. Throws if the current bit index
  /// is not 0 (since a non-zero bit offset breaks raw byte alignment).
  std::span<const byte> getSpan() const {
    if (bitIndex != 0)
      throw std::runtime_error("Cannot get data span with non-zero bit index");
    return data;
  }

  /// Logical right shift that drops the least-significant n bits by advancing
  /// the start pointer / bitIndex and reducing width. Returns a new immutable
  /// view. Does not modify the underlying storage contents.
  BitVector operator>>(size_t n) const;
  BitVector &operator>>=(size_t n);

  /// Create a new immutable view of a contiguous bit slice [offset,
  /// offset+sliceWidth). The returned BitVector is a view (not an owning copy)
  /// into the same underlying span. Throws if the requested slice exceeds the
  /// current width.
  BitVector slice(size_t offset, size_t sliceWidth) const;

  /// Return a view of the N least-significant bits.
  BitVector lsb(size_t n) const { return slice(0, n); }

  /// Return a view of the N most-significant bits.
  BitVector msb(size_t n) const {
    if (n > bitWidth)
      throw std::invalid_argument("msb width exceeds bit width");
    return slice(bitWidth - n, n);
  }

  std::string toString(unsigned base = 16) const;

  bool operator==(const BitVector &rhs) const;
  bool operator!=(const BitVector &rhs) const { return !(*this == rhs); }

  /// Bitwise AND: creates a new MutableBitVector with the result.
  friend MutableBitVector operator&(const BitVector &a, const BitVector &b);

  /// Bitwise OR: creates a new MutableBitVector with the result.
  friend MutableBitVector operator|(const BitVector &a, const BitVector &b);

  /// Bitwise XOR: creates a new MutableBitVector with the result.
  friend MutableBitVector operator^(const BitVector &a, const BitVector &b);

  /// Forward iterator for iterating over bits from LSB (index 0) to MSB.
  class bit_iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = bool;
    using pointer = const bool *;
    using reference = bool;
    using iterator_category = std::forward_iterator_tag;

    /// Construct an iterator at the given bit position.
    bit_iterator(const BitVector *bv, size_t pos = 0)
        : bitVector(bv), position(pos) {}

    /// Dereference: returns the bit value at the current position.
    bool operator*() const {
      if (bitVector == nullptr || position >= bitVector->bitWidth)
        throw std::out_of_range("bit_iterator dereference out of range");
      return bitVector->getBit(position);
    }

    /// Pre-increment: move to next bit.
    bit_iterator &operator++() {
      ++position;
      return *this;
    }

    /// Post-increment: move to next bit.
    bit_iterator operator++(int) {
      bit_iterator tmp = *this;
      ++position;
      return tmp;
    }

    /// Equality comparison.
    bool operator==(const bit_iterator &other) const {
      return bitVector == other.bitVector && position == other.position;
    }

    /// Inequality comparison.
    bool operator!=(const bit_iterator &other) const {
      return !(*this == other);
    }

    /// Less-than comparison (for ranges support).
    bool operator<(const bit_iterator &other) const {
      return bitVector == other.bitVector && position < other.position;
    }

  private:
    const BitVector *bitVector = nullptr;
    size_t position = 0;
  };

  /// Return an iterator to the first bit (LSB).
  bit_iterator begin() const { return bit_iterator(this, 0); }

  /// Return an iterator past the last bit.
  bit_iterator end() const { return bit_iterator(this, bitWidth); }

protected:
  // Underlying storage view. const, to allow for non-owning immutable views.
  std::span<const byte> data{};
  size_t bitWidth = 0;  // Number of valid bits.
  uint8_t bitIndex = 0; // Starting bit offset in first byte.
};

/// A mutable bit vector that owns its underlying storage.
/// MutableBitVector always has bitIndex == 0 since it owns the storage.
/// It supports in-place modifications and mutable operations.
class MutableBitVector : public BitVector {
public:
  /// Owning, zero-initialized constructor of a given width.
  explicit MutableBitVector(size_t width);

  /// Owning constructor from an rvalue vector (must move in).
  MutableBitVector(std::vector<byte> &&bytes,
                   std::optional<size_t> width = std::nullopt);

  MutableBitVector() = default;

  // Copy constructor: duplicate storage.
  MutableBitVector(const MutableBitVector &other);

  // Copy constructor from immutable BitVector: creates owning copy.
  MutableBitVector(const BitVector &other);

  // Move constructor: transfer ownership.
  MutableBitVector(MutableBitVector &&other) noexcept;

  // Move constructor from immutable BitVector: creates owning copy.
  MutableBitVector(BitVector &&other);

  MutableBitVector &operator=(const MutableBitVector &other);

  MutableBitVector &operator=(MutableBitVector &&other) noexcept;

  /// Set the i-th bit.
  void setBit(size_t i, bool v);

  /// Return a handle to the underlying span (always aligned since bitIndex=0).
  std::span<const byte> getSpan() const { return data; }

  /// Return and transfer ownership of the underlying storage.
  std::vector<uint8_t> takeStorage() { return std::move(owner); }

  /// In-place logical right shift that drops the least-significant n bits.
  /// Reduces width and updates internal state. Does not modify underlying
  /// storage.
  MutableBitVector &operator>>=(size_t n);

  /// In-place logical left shift that drops the most-significant n bits (width
  /// shrink).
  MutableBitVector &operator<<=(size_t n);

  /// In-place concatenate: appends bits from other to this.
  MutableBitVector &operator<<=(const MutableBitVector &other);

  MutableBitVector &operator|=(const MutableBitVector &other);
  MutableBitVector &operator&=(const MutableBitVector &other);
  MutableBitVector &operator^=(const MutableBitVector &other);
  MutableBitVector operator~() const;
  MutableBitVector operator|(const MutableBitVector &other) const;
  MutableBitVector operator&(const MutableBitVector &other) const;
  MutableBitVector operator^(const MutableBitVector &other) const;

private:
  // Storage owned by this MutableBitVector.
  std::vector<byte> owner;
};

std::ostream &operator<<(std::ostream &os, const BitVector &bv);

class Int : public MutableBitVector {
public:
  using MutableBitVector::MutableBitVector;
  Int() = default;
  Int(int64_t v, unsigned width = 64);
  operator int64_t() const { return toI64(); }
  operator int32_t() const { return toInt<int32_t>(); }
  operator int16_t() const { return toInt<int16_t>(); }
  operator int8_t() const { return toInt<int8_t>(); }

private:
  template <typename T>
  T toInt() const {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "T must be a signed integral type");
    int64_t v = toI64();
    fits(v, sizeof(T) * 8);
    return static_cast<T>(v);
  }

  // Convert the bitvector to a signed intN_t, throwing if the value doesn't
  // fit.
  int64_t toI64() const {
    if (this->bitWidth == 0)
      return 0;
    uint64_t u = 0;
    size_t limit = this->bitWidth < 64 ? this->bitWidth : 64;
    for (size_t i = 0; i < limit; ++i)
      if (this->getBit(i))
        u |= (1ULL << i);
    bool signBit = this->getBit(this->bitWidth - 1);
    if (this->bitWidth < 64) {
      if (signBit) {
        for (size_t i = this->bitWidth; i < 64; ++i)
          u |= (1ULL << i);
      }
      return static_cast<int64_t>(u);
    }
    for (size_t i = 64; i < this->bitWidth; ++i) {
      if (this->getBit(i) != signBit)
        throw std::overflow_error("Int does not fit in int64_t");
    }
    return static_cast<int64_t>(u);
  }

  // Check if the given value fits in the specified bit width.
  static void fits(int64_t v, unsigned n) {
    if (n == 0 || n > 64)
      throw std::invalid_argument(std::format("Invalid bit width: {}", n));
    int64_t min = -(1LL << (n - 1));
    int64_t max = (1LL << (n - 1)) - 1;
    if (v < min || v > max)
      throw std::overflow_error(
          std::format("Value {} does not fit in int{}_t", v, n));
  }
};

class UInt : public MutableBitVector {
public:
  using MutableBitVector::MutableBitVector;
  UInt() = default;
  UInt(uint64_t v, unsigned width = 64);
  operator uint64_t() const { return toUI64(); }
  operator uint32_t() const { return toUInt<uint32_t>(); }
  operator uint16_t() const { return toUInt<uint16_t>(); }
  operator uint8_t() const { return toUInt<uint8_t>(); }

private:
  uint64_t toUI64() const {
    if (this->bitWidth > 64) {
      for (size_t i = 64; i < this->bitWidth; ++i)
        if (this->getBit(i))
          throw std::overflow_error("Int does not fit in uint64_t");
    }
    uint64_t u = 0;
    size_t limit = this->bitWidth < 64 ? this->bitWidth : 64;
    for (size_t i = 0; i < limit; ++i)
      if (this->getBit(i))
        u |= (1ULL << i);
    return u;
  }

  static void fits(uint64_t v, unsigned n) {
    if (n == 0 || n > 64)
      throw std::invalid_argument(std::format("Invalid bit width: {}", n));
    uint64_t max = (n == 64) ? UINT64_MAX : ((1ULL << n) - 1);
    if (v > max)
      throw std::overflow_error(
          std::format("Value {} does not fit in uint{}_t", v, n));
  }

  template <typename T>
  T toUInt() const {
    static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value,
                  "T must be an unsigned integral type");
    uint64_t v = toUI64();
    fits(v, sizeof(T) * 8);
    return static_cast<T>(v);
  }
};

} // namespace esi

// Enable BitVector and MutableBitVector to work with std::ranges algorithms
template <>
inline constexpr bool std::ranges::enable_borrowed_range<esi::BitVector> = true;

template <>
inline constexpr bool
    std::ranges::enable_borrowed_range<esi::MutableBitVector> = true;

#endif // ESI_VALUES_H