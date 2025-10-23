//===- Values.cpp - ESI value system impl ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Values.h"
#include <cstring>
#include <format>

using namespace esi;

BitVector::BitVector(std::span<const byte> bytes, std::optional<size_t> width,
                     uint8_t bitIndex)
    : bitWidth(width ? *width : bytes.size() * 8), bitIndex(bitIndex) {
  data = bytes;
  if (bitIndex > 7)
    throw std::invalid_argument("bitIndex must be <= 7");
  size_t totalBitsAvail = bytes.size() * 8 - bitIndex;
  if (bitWidth > totalBitsAvail)
    throw std::invalid_argument(
        std::format("Width of {} bits exceeds provided storage of {} bits",
                    bitWidth, totalBitsAvail));
}

BitVector::BitVector(const BitVector &other)
    : data(other.data), bitWidth(other.bitWidth), bitIndex(other.bitIndex) {}

BitVector &BitVector::operator=(const BitVector &other) {
  if (this == &other)
    return *this;
  data = other.data;
  bitWidth = other.bitWidth;
  bitIndex = other.bitIndex;
  return *this;
}

bool BitVector::getBit(size_t i) const {
  if (i >= bitWidth)
    throw std::out_of_range("Bit index out of range");
  size_t absoluteBit = bitIndex + i;
  size_t byteIdx = absoluteBit / 8;
  uint8_t bitOff = absoluteBit % 8;
  return (data[byteIdx] >> bitOff) & 0x1;
}

BitVector BitVector::operator>>(size_t n) const {
  // Shifting by more than the current logical width is invalid.
  if (n > bitWidth)
    throw std::out_of_range("Right shift exceeds bit width");
  // Shifting by exactly the width yields an empty (zero-width) vector.
  if (n == bitWidth) {
    BitVector empty;
    return empty;
  }

  // Create a new view with adjusted bitIndex and width
  BitVector result = *this;
  result.bitWidth -= n;

  // Compute new bitIndex and byte offset
  size_t totalBitOffset = bitIndex + n;
  size_t byteOffset = totalBitOffset / 8;
  result.bitIndex = static_cast<uint8_t>(totalBitOffset % 8);

  // Skip the appropriate number of bytes
  if (byteOffset > 0 && byteOffset < data.size())
    result.data = data.subspan(byteOffset);
  else if (byteOffset >= data.size())
    result.data = data.subspan(data.size());

  return result;
}

BitVector &BitVector::operator>>=(size_t n) {
  *this = *this >> n;
  return *this;
}

BitVector BitVector::slice(size_t offset, size_t sliceWidth) const {
  if (offset > bitWidth || sliceWidth > bitWidth - offset)
    throw std::invalid_argument("slice range exceeds current width");
  // Return a new view with adjusted bitIndex
  BitVector result;
  result.bitWidth = sliceWidth;
  result.bitIndex = static_cast<uint8_t>(bitIndex + (offset % 8));
  size_t byteOffset = offset / 8;
  result.data = data.subspan(byteOffset);
  if (result.bitIndex >= 8) {
    result.bitIndex -= 8;
    result.data = result.data.subspan(1);
  }
  return result;
}

MutableBitVector::MutableBitVector(size_t width) : owner((width + 7) / 8, 0) {
  bitWidth = width;
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
}

MutableBitVector::MutableBitVector(std::vector<byte> &&bytes,
                                   std::optional<size_t> width)
    : owner(std::move(bytes)) {
  bitWidth = width ? *width : owner.size() * 8;
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
  if (bitWidth > data.size() * 8)
    throw std::invalid_argument("Width exceeds provided storage");
}

MutableBitVector::MutableBitVector(const MutableBitVector &other)
    : BitVector(), owner(other.owner) {
  bitWidth = other.bitWidth;
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
}

MutableBitVector::MutableBitVector(const BitVector &other)
    : BitVector(), owner(std::vector<byte>((other.width() + 7) / 8, 0)) {
  bitWidth = other.width();
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
  // Copy bits from source
  for (size_t i = 0; i < bitWidth; ++i)
    if (other.getBit(i))
      setBit(i, true);
}

MutableBitVector::MutableBitVector(BitVector &&other)
    : BitVector(), owner(std::vector<byte>((other.width() + 7) / 8, 0)) {
  bitWidth = other.width();
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
  // Copy bits from source
  for (size_t i = 0; i < bitWidth; ++i)
    if (other.getBit(i))
      setBit(i, true);
}

MutableBitVector::MutableBitVector(MutableBitVector &&other) noexcept
    : BitVector(), owner(std::move(other.owner)) {
  bitWidth = other.bitWidth;
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
  other.data = {};
  other.bitWidth = 0;
  other.bitIndex = 0;
}

MutableBitVector &MutableBitVector::operator=(const MutableBitVector &other) {
  if (this == &other)
    return *this;
  owner = other.owner;
  bitWidth = other.bitWidth;
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
  return *this;
}

MutableBitVector &
MutableBitVector::operator=(MutableBitVector &&other) noexcept {
  if (this == &other)
    return *this;
  owner = std::move(other.owner);
  bitWidth = other.bitWidth;
  bitIndex = 0;
  data = std::span<const byte>(owner.data(), owner.size());
  other.data = {};
  other.bitWidth = 0;
  other.bitIndex = 0;
  return *this;
}

void MutableBitVector::setBit(size_t i, bool v) {
  if (i >= bitWidth)
    throw std::out_of_range("Bit index out of range");
  size_t byteIdx = i / 8;
  uint8_t bitOff = i % 8;
  uint8_t mask = static_cast<uint8_t>(1u << bitOff);
  byte &target = owner[byteIdx];
  if (v)
    target |= mask;
  else
    target &= static_cast<uint8_t>(~mask);
}

MutableBitVector &MutableBitVector::operator>>=(size_t n) {
  // Shifting by more than the current logical width is invalid.
  if (n > bitWidth)
    throw std::out_of_range("Right shift exceeds bit width");
  // Shifting by exactly the width yields an empty (zero-width) vector.
  if (n == bitWidth) {
    bitWidth = 0;
    return *this;
  }
  bitWidth -= n;
  bitIndex = static_cast<uint8_t>(bitIndex + n);
  while (bitIndex >= 8) {
    bitIndex -= 8;
    if (data.size() == 0)
      break;
    data = data.subspan(1);
  }
  return *this;
}

MutableBitVector &MutableBitVector::operator<<=(size_t n) {
  // Extend width by n bits, shifting existing bits upward by n and inserting
  // n zero bits at the least-significant positions.
  if (n == 0)
    return *this;
  size_t oldWidth = bitWidth;
  size_t newWidth = oldWidth + n;
  // Always produce an owning, tightly packed representation with bitIndex=0.
  std::vector<byte> newStorage((newWidth + 7) / 8, 0);
  // Copy old bits to new positions.
  for (size_t i = 0; i < oldWidth; ++i)
    if (getBit(i)) {
      size_t newPos = i + n; // shifted upward
      newStorage[newPos / 8] |= static_cast<byte>(1u << (newPos % 8));
    }
  owner = std::move(newStorage);
  data = std::span<const byte>(owner.data(), owner.size());
  bitWidth = newWidth;
  bitIndex = 0;
  return *this;
}

MutableBitVector &MutableBitVector::operator<<=(const MutableBitVector &other) {
  // Concatenation: append bits from other (must create new owning storage)
  size_t oldWidth = bitWidth;
  size_t otherWidth = other.width();
  size_t newWidth = oldWidth + otherWidth;
  std::vector<byte> newStorage((newWidth + 7) / 8, 0);
  // Copy this's bits
  for (size_t i = 0; i < oldWidth; ++i)
    if (getBit(i)) {
      newStorage[i / 8] |= static_cast<byte>(1u << (i % 8));
    }
  // Copy other's bits offset by oldWidth
  for (size_t i = 0; i < otherWidth; ++i)
    if (other.getBit(i)) {
      size_t newPos = i + oldWidth;
      newStorage[newPos / 8] |= static_cast<byte>(1u << (newPos % 8));
    }
  owner = std::move(newStorage);
  data = std::span<const byte>(owner.data(), owner.size());
  bitWidth = newWidth;
  bitIndex = 0;
  return *this;
}

MutableBitVector esi::operator&(const BitVector &a, const BitVector &b) {
  if (a.width() != b.width())
    throw std::invalid_argument("Bitwise & require equal widths");
  MutableBitVector result(a.width());
  for (size_t i = 0; i < a.width(); ++i)
    if (a.getBit(i) && b.getBit(i))
      result.setBit(i, true);
  return result;
}

MutableBitVector esi::operator|(const BitVector &a, const BitVector &b) {
  if (a.width() != b.width())
    throw std::invalid_argument("Bitwise ops require equal widths");
  MutableBitVector result(a.width());
  for (size_t i = 0; i < a.width(); ++i)
    if (a.getBit(i) || b.getBit(i))
      result.setBit(i, true);
  return result;
}

MutableBitVector esi::operator^(const BitVector &a, const BitVector &b) {
  if (a.width() != b.width())
    throw std::invalid_argument("Bitwise ops require equal widths");
  MutableBitVector result(a.width());
  for (size_t i = 0; i < a.width(); ++i)
    if (a.getBit(i) != b.getBit(i))
      result.setBit(i, true);
  return result;
}

// Helper: convert to hexadecimal / octal (power-of-two bases) without
// allocating large temporaries. bitsPerDigit must be 1/3/4.
static std::string toPowerOfTwoString(const BitVector &bv,
                                      unsigned bitsPerDigit, bool uppercase) {
  if (bv.width() == 0)
    return "0";
  unsigned groups = (bv.width() + bitsPerDigit - 1) / bitsPerDigit;
  std::string out;
  out.reserve(groups);
  auto hexDigit = [&](unsigned v) -> char {
    if (v < 10)
      return static_cast<char>('0' + v);
    return static_cast<char>((uppercase ? 'A' : 'a') + (v - 10));
  };
  for (int g = static_cast<int>(groups) - 1; g >= 0; --g) {
    unsigned value = 0;
    for (unsigned j = 0; j < bitsPerDigit; ++j) {
      unsigned bitIdx = g * bitsPerDigit + j;
      if (bitIdx < bv.width() && bv.getBit(bitIdx))
        value |= (1u << j); // LSB-first inside digit
    }
    if (out.empty() && value == 0 && g != 0)
      continue; // skip leading zeros
    if (bitsPerDigit == 4)
      out.push_back(hexDigit(value));
    else // octal (3 bits) or binary (1 bit)
      out.push_back(static_cast<char>('0' + value));
  }
  if (out.empty())
    return "0"; // all zeros
  return out;
}

// Helper: decimal conversion via base 1e9 limbs (little-endian).
static std::string toDecimalString(const BitVector &bv) {
  if (bv.width() == 0)
    return "0";
  constexpr uint32_t BASE = 1000000000u; // 1e9
  std::vector<uint32_t> limbs(1, 0);
  // Iterate MSB->LSB so we can perform: acc = acc*2 + bit
  for (size_t idx = 0; idx < bv.width(); ++idx) {
    size_t i = bv.width() - 1 - idx;
    // acc *= 2
    uint64_t carry = 0;
    for (auto &d : limbs) {
      uint64_t v = static_cast<uint64_t>(d) * 2 + carry;
      d = static_cast<uint32_t>(v % BASE);
      carry = v / BASE;
    }
    if (carry)
      limbs.push_back(static_cast<uint32_t>(carry));
    // acc += bit
    if (bv.getBit(i)) {
      uint64_t c = 1;
      for (auto &d : limbs) {
        uint64_t v = static_cast<uint64_t>(d) + c;
        d = static_cast<uint32_t>(v % BASE);
        c = v / BASE;
        if (!c)
          break;
      }
      if (c)
        limbs.push_back(static_cast<uint32_t>(c));
    }
  }
  // Convert limbs to string.
  std::string out = std::to_string(limbs.back());
  for (int i = static_cast<int>(limbs.size()) - 2; i >= 0; --i) {
    std::string chunk = std::to_string(limbs[i]);
    out.append(9 - chunk.size(), '0');
    out += chunk;
  }
  return out;
}

std::string BitVector::toString(unsigned base) const {
  switch (base) {
  case 2: {
    // Binary (MSB -> LSB) representation.
    if (bitWidth == 0)
      return std::string("0");
    std::string s;
    s.reserve(bitWidth);
    for (size_t i = 0; i < bitWidth; ++i)
      s.push_back(getBit(bitWidth - 1 - i) ? '1' : '0');
    return s;
  }
  case 8:
    return toPowerOfTwoString(*this, 3, false);
  case 16:
    return toPowerOfTwoString(*this, 4, false);
  case 10:
    return toDecimalString(*this);
  default:
    throw std::invalid_argument(
        std::format("Unsupported base '{}' for BitVector::toString", base));
  }
}

std::ostream &esi::operator<<(std::ostream &os, const BitVector &bv) {
  using std::ios_base;
  ios_base::fmtflags basefmt = os.flags() & ios_base::basefield;
  bool showBase = (os.flags() & ios_base::showbase) != 0;
  bool upper = (os.flags() & ios_base::uppercase) != 0;
  std::string body;
  switch (basefmt) {
  case ios_base::hex: {
    body = toPowerOfTwoString(bv, 4, upper);
    if (showBase)
      os << (upper ? "0X" : "0x");
    os << body;
    return os;
  }
  case ios_base::oct: {
    body = toPowerOfTwoString(bv, 3, false);
    if (showBase)
      os << '0';
    os << body;
    return os;
  }
  case ios_base::dec: {
    body = toDecimalString(bv);
    os << body; // showbase ignored for dec (matches standard)
    return os;
  }
  default: { // Fallback: binary string (no standard manipulator for this)
    body = bv.toString(2);
    os << body;
    return os;
  }
  }
}

bool BitVector::operator==(const BitVector &rhs) const {
  if (bitWidth != rhs.bitWidth)
    return false;
  return std::ranges::equal(*this, rhs);
}

UInt::UInt(uint64_t v, unsigned width) : MutableBitVector(width) {
  if (width > 0 && width < 64 && (v >> width) != 0)
    throw std::overflow_error(
        std::format("Value {} does not fit in {} bits", v, width));
  for (size_t i = 0; i < width; ++i)
    if ((v >> i) & 1)
      setBit(i, true);
}

Int::Int(int64_t v, unsigned width) : MutableBitVector(width) {
  if (width > 0 && width < 64) {
    int64_t maxVal = (1LL << (width - 1)) - 1;
    int64_t minVal = -(1LL << (width - 1));
    if (v < minVal || v > maxVal)
      throw std::overflow_error(
          std::format("Value {} does not fit in {} bits", v, width));
  }
  for (size_t i = 0; i < width; ++i)
    if ((v >> i) & 1)
      setBit(i, true);
}

MutableBitVector &MutableBitVector::operator&=(const MutableBitVector &other) {
  if (bitWidth != other.bitWidth)
    throw std::invalid_argument("Bitwise &= requires equal widths");
  for (size_t i = 0; i < bitWidth; ++i)
    setBit(i, getBit(i) && other.getBit(i));
  return *this;
}

MutableBitVector &MutableBitVector::operator|=(const MutableBitVector &other) {
  if (bitWidth < other.bitWidth)
    throw std::invalid_argument("Bitwise |= requires <= widths");
  for (size_t i = 0; i < other.bitWidth; ++i)
    setBit(i, getBit(i) || other.getBit(i));
  return *this;
}

MutableBitVector &MutableBitVector::operator^=(const MutableBitVector &other) {
  if (bitWidth != other.bitWidth)
    throw std::invalid_argument("Bitwise ^= requires equal widths");
  for (size_t i = 0; i < bitWidth; ++i)
    setBit(i, getBit(i) != other.getBit(i));
  return *this;
}
MutableBitVector MutableBitVector::operator~() const {
  MutableBitVector res(bitWidth);
  for (size_t i = 0; i < bitWidth; ++i)
    res.setBit(i, !getBit(i));
  return res;
}
MutableBitVector
MutableBitVector::operator|(const MutableBitVector &other) const {
  MutableBitVector result(static_cast<const BitVector &>(*this));
  result |= other;
  return result;
}

MutableBitVector
MutableBitVector::operator&(const MutableBitVector &other) const {
  MutableBitVector result(static_cast<const BitVector &>(*this));
  result &= other;
  return result;
}

MutableBitVector
MutableBitVector::operator^(const MutableBitVector &other) const {
  MutableBitVector result(static_cast<const BitVector &>(*this));
  result ^= other;
  return result;
}

int64_t Int::toI64() const {
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

void Int::fits(int64_t v, unsigned n) {
  if (n == 0 || n > 64)
    throw std::invalid_argument(std::format("Invalid bit width: {}", n));
  int64_t min = -(1LL << (n - 1));
  int64_t max = (1LL << (n - 1)) - 1;
  if (v < min || v > max)
    throw std::overflow_error(
        std::format("Value {} does not fit in int{}_t", v, n));
}

uint64_t UInt::toUI64() const {
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

void UInt::fits(uint64_t v, unsigned n) {
  if (n == 0 || n > 64)
    throw std::invalid_argument(std::format("Invalid bit width: {}", n));
  uint64_t max = (n == 64) ? UINT64_MAX : ((1ULL << n) - 1);
  if (v > max)
    throw std::overflow_error(
        std::format("Value {} does not fit in uint{}_t", v, n));
}
