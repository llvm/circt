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
/// BitVector is immutable wrt. modifying the underlying bits, and provides
/// read-only access to bits. It supports bit-level access and returns new views
/// for operations.
///
/// Lifetime: `BitVector`, `IntView`, and `UIntView` (defined just below)
/// are non-owning views, in the same family as `std::span` /
/// `std::string_view`. They store only a span pointer + width + bitIndex
/// and do not extend the lifetime of the underlying buffer. A view
/// returned from an accessor of a temporary dangles once the temporary
/// is destroyed; bind the parent to a named local, or construct an
/// owning `Int` / `UInt` / `MutableBitVector` from the view if the value
/// needs to outlive its source.
class BitVector {
public:
  using byte = uint8_t;

  /// Construct from an existing span. Width defaults to the number of bits in
  /// the span (size * 8). The BitVector does not take ownership.
  BitVector(std::span<const byte> bytes,
            std::optional<size_t> width = std::nullopt, uint8_t bitIndex = 0);
  BitVector() = default;
  BitVector(const BitVector &other);
  BitVector &operator=(const BitVector &other);

  size_t width() const { return bitWidth; }
  size_t size() const { return width(); }

  /// Return the i-th bit (0 = LSB) as boolean.
  bool getBit(size_t i) const;

  /// Return a handle to the underlying span. Throws if the current bit index
  /// is not 0 (since a non-zero bit offset breaks raw byte alignment).
  std::span<const byte> getSpan() const {
    if (bitIndex != 0)
      throw std::runtime_error("Cannot get data span with non-zero bit index");
    return data;
  }

  /// Logical right shift that drops the least-significant n bits by advancing
  /// the byte/bit index and reducing width. Returns a new immutable
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

    /// Default constructor.
    bit_iterator() = default;

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

    /// Sentinel-compatible equality (for ranges support).
    bool operator==(std::default_sentinel_t) const {
      return bitVector == nullptr || position >= bitVector->bitWidth;
    }

    /// Sentinel-compatible inequality.
    bool operator!=(std::default_sentinel_t sent) const {
      return !(*this == sent);
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

  // Scan bits [lowExclusive, width()-1] in byte-sized chunks and throw
  // `std::overflow_error(std::vformat(fmt, std::make_format_args(target)))`
  // if any of them differ from `expectedByte` (0x00 to check
  // "all-zero" for the unsigned narrow-conversion case; 0xFF to check
  // "all-one" for the signed narrow-conversion case with the sign bit
  // set).
  void checkHighBytesEqual(unsigned lowExclusive, uint8_t expectedByte,
                           const char *fmt, unsigned target) const;
};

/// Non-owning view of an unsigned bit vector with `toUI64()` and implicit
/// conversions to unsigned scalar types. Adds only static-type tagging and
/// conversion methods on top of `BitVector`. See the lifetime note on
/// `BitVector`.
class UIntView : public BitVector {
public:
  using BitVector::BitVector;
  UIntView() = default;
  /// Adopt an existing view as unsigned. Cheap (no copy of bytes); the
  /// resulting UIntView aliases the same buffer.
  UIntView(const BitVector &v) : BitVector(v) {}

  /// Convert to a `uint64_t`, throwing if the value does not fit.
  uint64_t toUI64() const;
  operator uint64_t() const { return toUI64(); }
  operator uint32_t() const { return toUInt<uint32_t>(); }
  operator uint16_t() const { return toUInt<uint16_t>(); }
  operator uint8_t() const { return toUInt<uint8_t>(); }

private:
  template <typename T>
  T toUInt() const {
    static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value,
                  "T must be an unsigned integral type");
    constexpr unsigned N = sizeof(T) * 8;
    // Pre-conversion range check: bits [N, width-1] must all be zero
    // for the value to fit in unsigned `T`. We have to do this
    // *before* `toUI64()` because `toUI64()` would itself throw
    // "does not fit in uint64_t" for any wide value where bits 64+
    // are set -- even if the low N bits would have fit in `T`.
    if (this->width() > N)
      this->checkHighBytesEqual(N, /*expectedByte=*/uint8_t{0},
                                "UInt does not fit in uint{}_t", N);
    return static_cast<T>(toUI64());
  }
};

/// Non-owning view of a signed bit vector with `toI64()` and implicit
/// conversions to signed scalar types. See the lifetime note on `BitVector`.
class IntView : public BitVector {
public:
  using BitVector::BitVector;
  IntView() = default;
  /// Adopt an existing view as signed. Cheap (no copy of bytes); the
  /// resulting IntView aliases the same buffer.
  IntView(const BitVector &v) : BitVector(v) {}

  /// Convert to an `int64_t`, sign-extending from the high bit and throwing
  /// if the value does not fit.
  int64_t toI64() const;
  operator int64_t() const { return toI64(); }
  operator int32_t() const { return toInt<int32_t>(); }
  operator int16_t() const { return toInt<int16_t>(); }
  operator int8_t() const { return toInt<int8_t>(); }

private:
  template <typename T>
  T toInt() const {
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                  "T must be a signed integral type");
    constexpr unsigned N = sizeof(T) * 8;
    // Pre-conversion range check: bits [N, width-1] must all equal
    // the sign bit (bit N-1) for the value to fit in signed `T`. We
    // have to do this *before* `toI64()` because `toI64()` would
    // itself throw "does not fit in int64_t" for any wide value
    // whose bits 64+ don't sign-extend bit 63 -- even if the actual
    // problem is that the value doesn't fit in `T`.
    if (this->width() > N) {
      const uint8_t expected = this->getBit(N - 1) ? uint8_t{0xFF} : uint8_t{0};
      this->checkHighBytesEqual(N, expected, "Int does not fit in int{}_t", N);
    }
    return static_cast<T>(toI64());
  }
};

/// A mutable bit vector that owns its underlying storage.
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

  /// In-place logical left shift shifts in n zero bits at LSB, shifting
  /// existing bits upward.
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

// Arbitrary width signed integer type built on MutableBitVector. The
// scalar-conversion operators all delegate to `IntView` so the
// canonical width-check + i64 conversion lives in one place.
class Int : public MutableBitVector {
public:
  using MutableBitVector::MutableBitVector;
  Int() = default;
  Int(int64_t v, unsigned width = 64);
  operator int64_t() const { return IntView(*this); }
  operator int32_t() const { return IntView(*this); }
  operator int16_t() const { return IntView(*this); }
  operator int8_t() const { return IntView(*this); }
};

// Arbitrary width unsigned integer type built on MutableBitVector. The
// scalar-conversion operators all delegate to `UIntView`.
class UInt : public MutableBitVector {
public:
  using MutableBitVector::MutableBitVector;
  UInt() = default;
  UInt(uint64_t v, unsigned width = 64);
  operator uint64_t() const { return UIntView(*this); }
  operator uint32_t() const { return UIntView(*this); }
  operator uint16_t() const { return UIntView(*this); }
  operator uint8_t() const { return UIntView(*this); }
};

} // namespace esi

// Enable BitVector and MutableBitVector to work with std::ranges algorithms
template <>
inline constexpr bool std::ranges::enable_borrowed_range<esi::BitVector> = true;

template <>
inline constexpr bool
    std::ranges::enable_borrowed_range<esi::MutableBitVector> = true;

#endif // ESI_VALUES_H
