//===- Values.cpp - ESI value system impl ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Values.h"
#include <cstring>

using namespace esi;

namespace {

// Core add with carry. If signedMode=true, extend bits beyond operand width
// with the operand's sign bit. If false, extend with zero.
BitVector addCore(const BitVector &a, const BitVector &b, bool signedMode) {
  size_t w = a.width() > b.width() ? a.width() : b.width();

  // Concatenate another BitVector by shifting existing bits upward by the
  // other's width and inserting the other's bits at the new LSB positions.
  BitVector r(w + 1); // Provide one extra bit for carry / sign.
  bool carry = false;
  for (size_t i = 0; i < r.width(); ++i) {
    bool abit;
    bool bbit;
    if (i < a.width())
      abit = a.getBit(i);
    else if (signedMode && a.width() > 0)
      abit = a.getBit(a.width() - 1); // sign extend
    else
      abit = false;
    if (i < b.width())
      bbit = b.getBit(i);
    else if (signedMode && b.width() > 0)
      bbit = b.getBit(b.width() - 1);
    else
      bbit = false;
    bool sum = abit ^ bbit ^ carry;
    carry = (abit & bbit) | (abit & carry) | (bbit & carry);
    r.setBit(i, sum);
  }
  return r;
}

// BitVector negate (two's complement) for signed widths.
BitVector negateTwoComplement(const BitVector &v) {
  // Fixed-width two's complement negate: invert and add one modulo width.
  size_t w = v.width();
  BitVector r(w);
  bool carry = true; // adding one
  for (size_t i = 0; i < w; ++i) {
    bool bit = !v.getBit(i);
    bool sum = bit ^ carry;
    carry = bit & carry; // carry only if bit was 1 and carry was 1
    r.setBit(i, sum);
  }
  return r; // overflow beyond width is discarded (mod 2^w).
}

// Unsigned multiply using simple shift-add algorithm.
BitVector mulUnsigned(const BitVector &a, const BitVector &b) {
  BitVector acc(a.width() + b.width());
  for (size_t i = 0; i < b.width(); ++i) {
    if (!b.getBit(i))
      continue;
    BitVector shifted(acc.width());
    for (size_t j = 0; j < a.width(); ++j)
      if (a.getBit(j) && (j + i) < shifted.width())
        shifted.setBit(j + i, true);
    acc = addCore(acc, shifted, false);
  }
  return acc;
}
} // namespace

//===----------------------------------------------------------------------===//
// BitVector implementation outline
//===----------------------------------------------------------------------===//

BitVector::BitVector(std::span<const byte> bytes, std::optional<size_t> width,
                     uint8_t bitIndex)
    : bitWidth(width ? *width : bytes.size() * 8), bitIndex(bitIndex) {
  data = bytes;
  if (bitIndex > 7)
    throw std::invalid_argument("bitIndex must be <= 7");
  size_t totalBitsAvail = bytes.size() * 8 - bitIndex;
  if (bitWidth > totalBitsAvail)
    throw std::invalid_argument("Width exceeds provided storage");
}

BitVector::BitVector(std::vector<byte> &&bytes, std::optional<size_t> width,
                     uint8_t bitIndex)
    : owner(std::move(bytes)), bitWidth(width ? *width : owner->size() * 8),
      bitIndex(bitIndex) {
  data = std::span<const byte>(owner->data(), owner->size());
  if (bitIndex > 7)
    throw std::invalid_argument("bitIndex must be <= 7");
  size_t totalBitsAvail = data.size() * 8 - bitIndex;
  if (bitWidth > totalBitsAvail)
    throw std::invalid_argument("Width exceeds provided storage");
}

BitVector::BitVector(size_t width)
    : owner(std::vector<byte>((width + 7) / 8, 0)), bitWidth(width),
      bitIndex(0) {
  data = std::span<const byte>(owner->data(), owner->size());
}

BitVector::BitVector(const BitVector &other)
    : owner(other.owner), data(other.data), bitWidth(other.bitWidth),
      bitIndex(other.bitIndex) {
  if (owner)
    data = std::span<const byte>(owner->data(), owner->size());
}

BitVector::BitVector(BitVector &&other) noexcept
    : owner(std::move(other.owner)), data(other.data), bitWidth(other.bitWidth),
      bitIndex(other.bitIndex) {
  if (owner)
    data = std::span<const byte>(owner->data(), owner->size());
  other.data = {};
  other.bitWidth = 0;
  other.bitIndex = 0;
}

BitVector &BitVector::operator=(const BitVector &other) {
  if (this == &other)
    return *this;
  owner = other.owner;
  data = other.data;
  bitWidth = other.bitWidth;
  bitIndex = other.bitIndex;
  if (owner)
    data = std::span<const byte>(owner->data(), owner->size());
  return *this;
}

BitVector &BitVector::operator=(BitVector &&other) noexcept {
  if (this == &other)
    return *this;
  owner = std::move(other.owner);
  data = other.data;
  bitWidth = other.bitWidth;
  bitIndex = other.bitIndex;
  if (owner)
    data = std::span<const byte>(owner->data(), owner->size());
  other.data = {};
  other.bitWidth = 0;
  other.bitIndex = 0;
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

void BitVector::setBit(size_t i, bool v) {
  if (i >= bitWidth)
    throw std::out_of_range("Bit index out of range");
  if (!owner)
    throw std::runtime_error("Cannot modify non-owning BitVector");
  size_t absoluteBit = bitIndex + i;
  size_t byteIdx = absoluteBit / 8;
  uint8_t bitOff = absoluteBit % 8;
  uint8_t mask = static_cast<uint8_t>(1u << bitOff);
  // Compute base offset of current data span within owner storage.
  ptrdiff_t baseOffset = data.data() - owner->data();
  byte &target = (*owner)[baseOffset + byteIdx];
  if (v)
    target |= mask;
  else
    target &= static_cast<uint8_t>(~mask);
}

BitVector &BitVector::operator>>=(size_t n) {
  // Zero-width: any non-zero shift is invalid (cannot drop bits that don't
  // exist).
  if (bitWidth == 0) {
    if (n == 0)
      return *this; // no-op
    throw std::out_of_range("Right shift on zero-width BitVector");
  }
  // Shifting by more than the current logical width is invalid.
  if (n > bitWidth)
    throw std::out_of_range("Right shift exceeds bit width");
  // Shifting by exactly the width yields an empty (zero-width) vector.
  if (n == bitWidth) {
    bitWidth = 0;
    bitIndex = 0;
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

BitVector BitVector::operator>>(size_t n) const {
  BitVector tmp = *this;
  tmp >>= n;
  return tmp;
}

BitVector &BitVector::operator<<=(size_t n) {
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
  data = std::span<const byte>(owner->data(), owner->size());
  bitWidth = newWidth;
  bitIndex = 0;
  return *this;
}

BitVector BitVector::operator<<(size_t n) const {
  BitVector tmp = *this; // copy
  tmp <<= n;             // will allocate new storage
  return tmp;
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
    return toString(2);
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
  for (size_t i = 0; i < bitWidth; ++i)
    if (getBit(i) != rhs.getBit(i))
      return false;
  return true;
}

BitVector BitVector::slice(size_t offset, size_t sliceWidth) const {
  if (offset > bitWidth || sliceWidth > bitWidth - offset)
    throw std::invalid_argument("slice range exceeds current width");
  BitVector out(sliceWidth);
  if (sliceWidth == 0)
    return out;
  size_t startBit = bitIndex + offset;
  bool aligned = (startBit % 8 == 0);
  if (aligned) {
    size_t byteStart = startBit / 8;
    size_t bytesToCopy = (sliceWidth + 7) / 8;
    // Direct byte copy.
    std::memcpy(out.owner->data(), data.data() + byteStart, bytesToCopy);
    // We intentionally do not clear high bits in the final byte (unspecified
    // tail).
    return out;
  }
  // Fallback per-bit extraction.
  for (size_t i = 0; i < sliceWidth; ++i)
    if (getBit(offset + i))
      out.setBit(i, true);
  return out;
}

BitVector BitVector::bitwiseOp(const BitVector &a, const BitVector &b,
                               BitwiseKind kind) {
  if (a.bitWidth != b.bitWidth)
    throw std::invalid_argument("Bitwise ops require equal widths");
  size_t width = a.bitWidth;
  BitVector res(width);
  if (width == 0)
    return res;

  // Fast path: byte-aligned, same bitIndex. We intentionally do NOT mask off
  // high bits of a partial last byte; bits above logical width may contain
  // arbitrary values and must not be relied upon by callers.
  bool aligned = (a.bitIndex == 0) && (b.bitIndex == 0);
  if (aligned) {
    auto *ra = a.data.data();
    auto *rb = b.data.data();
    // Cast away const on result storage (it is owned / mutable here).
    auto *rr = const_cast<uint8_t *>(res.data.data());
    size_t fullBytes = width / 8;
    size_t tailBits = width % 8;
    // Process complete bytes.
    for (size_t i = 0; i < fullBytes; ++i) {
      switch (kind) {
      case BitwiseKind::And:
        rr[i] = static_cast<uint8_t>(ra[i] & rb[i]);
        break;
      case BitwiseKind::Or:
        rr[i] = static_cast<uint8_t>(ra[i] | rb[i]);
        break;
      case BitwiseKind::Xor:
        rr[i] = static_cast<uint8_t>(ra[i] ^ rb[i]);
        break;
      }
    }
    if (tailBits) {
      // Operate on entire last byte unmasked (high bits beyond width are
      // unspecified).
      switch (kind) {
      case BitwiseKind::And:
        rr[fullBytes] = static_cast<uint8_t>(ra[fullBytes] & rb[fullBytes]);
        break;
      case BitwiseKind::Or:
        rr[fullBytes] = static_cast<uint8_t>(ra[fullBytes] | rb[fullBytes]);
        break;
      case BitwiseKind::Xor:
        rr[fullBytes] = static_cast<uint8_t>(ra[fullBytes] ^ rb[fullBytes]);
        break;
      }
    }
    return res;
  }

  // Fallback slow path: misaligned or differing bitIndex.
  for (size_t i = 0; i < width; ++i) {
    bool abit = a.getBit(i);
    bool bbit = b.getBit(i);
    bool out = false;
    switch (kind) {
    case BitwiseKind::And:
      out = abit & bbit;
      break;
    case BitwiseKind::Or:
      out = abit | bbit;
      break;
    case BitwiseKind::Xor:
      out = abit ^ bbit;
      break;
    }
    res.setBit(i, out);
  }
  return res;
}

//===----------------------------------------------------------------------===//
// Int implementation
//===----------------------------------------------------------------------===//

void Int::assign(int64_t value) {
  uint64_t pattern = static_cast<uint64_t>(value); // two's complement pattern
  size_t limit = bitWidth < 64 ? bitWidth : 64;
  for (size_t i = 0; i < limit; ++i)
    setBit(i, (pattern >> i) & 1ULL);
  if (bitWidth > 64) {
    bool signBit = (pattern >> 63) & 1ULL;
    for (size_t i = 64; i < bitWidth; ++i)
      setBit(i, signBit);
  }
}

int64_t Int::toSigned64() const {
  if (bitWidth == 0)
    return 0;
  uint64_t u = 0;
  size_t limit = bitWidth < 64 ? bitWidth : 64;
  for (size_t i = 0; i < limit; ++i)
    if (getBit(i))
      u |= (1ULL << i);
  bool signBit = getBit(bitWidth - 1);
  if (bitWidth < 64) {
    if (signBit) {
      for (size_t i = bitWidth; i < 64; ++i)
        u |= (1ULL << i);
    }
    return static_cast<int64_t>(u);
  }
  for (size_t i = 64; i < bitWidth; ++i) {
    if (getBit(i) != signBit)
      throw std::overflow_error("Int does not fit in int64_t");
  }
  return static_cast<int64_t>(u);
}

uint64_t Int::toUnsigned64() const {
  if (bitWidth > 64) {
    for (size_t i = 64; i < bitWidth; ++i)
      if (getBit(i))
        throw std::overflow_error("Int does not fit in uint64_t");
  }
  uint64_t u = 0;
  size_t limit = bitWidth < 64 ? bitWidth : 64;
  for (size_t i = 0; i < limit; ++i)
    if (getBit(i))
      u |= (1ULL << i);
  return u;
}

Int Int::add(const Int &a, const Int &b) { return Int(addCore(a, b, true)); }

Int Int::negate(const Int &v) { return Int(negateTwoComplement(v)); }

Int Int::multiply(const Int &a, const Int &b) {
  if (a.bitWidth == 0 || b.bitWidth == 0)
    return Int(0, 0);
  bool aNeg = a.getBit(a.bitWidth - 1);
  bool bNeg = b.getBit(b.bitWidth - 1);
  BitVector absA =
      aNeg ? negateTwoComplement(a) : static_cast<const BitVector &>(a);
  BitVector absB =
      bNeg ? negateTwoComplement(b) : static_cast<const BitVector &>(b);
  BitVector prod = mulUnsigned(absA, absB);
  if (aNeg ^ bNeg)
    prod = negateTwoComplement(prod);
  return Int(prod);
}

int Int::compare(const Int &a, const Int &b) {
  size_t w = a.bitWidth > b.bitWidth ? a.bitWidth : b.bitWidth;
  if (w == 0)
    return 0;
  auto getSE = [&](const Int &v, size_t i) -> bool {
    if (i < v.bitWidth)
      return v.getBit(i);
    if (v.bitWidth == 0)
      return false;
    return v.getBit(v.bitWidth - 1);
  };
  bool signA = getSE(a, w - 1);
  bool signB = getSE(b, w - 1);
  if (signA != signB)
    return signA ? -1 : 1; // 1 means negative (signA) < positive
  // Same sign: unsigned lexicographic compare works for both positive and
  // negative.
  for (size_t idx = 0; idx < w; ++idx) {
    size_t i = w - 1 - idx;
    bool abit = getSE(a, i);
    bool bbit = getSE(b, i);
    if (abit != bbit)
      return abit < bbit ? -1 : 1;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// UInt implementation
//===----------------------------------------------------------------------===//

void UInt::assign(uint64_t value) {
  for (size_t i = 0; i < bitWidth; ++i)
    setBit(i, (value >> i) & 1ULL);
  for (size_t i = 64; i < bitWidth; ++i)
    setBit(i, false);
}

uint64_t UInt::toUnsigned64() const {
  if (bitWidth > 64) {
    for (size_t i = 64; i < bitWidth; ++i)
      if (getBit(i))
        throw std::overflow_error("UInt does not fit in uint64_t");
  }
  uint64_t u = 0;
  size_t limit = bitWidth < 64 ? bitWidth : 64;
  for (size_t i = 0; i < limit; ++i)
    if (getBit(i))
      u |= (1ULL << i);
  return u;
}

UInt UInt::add(const UInt &a, const UInt &b) {
  return UInt(addCore(a, b, false));
}

UInt UInt::subtract(const UInt &a, const UInt &b) {
  size_t w = a.bitWidth > b.bitWidth ? a.bitWidth : b.bitWidth;
  UInt r(0, w);
  bool borrow = false;
  for (size_t i = 0; i < w; ++i) {
    bool abit = (i < a.bitWidth) ? a.getBit(i) : false;
    bool bbit = (i < b.bitWidth) ? b.getBit(i) : false;
    bool diff = abit ^ bbit ^ borrow;
    bool newBorrow = (!abit & (bbit | borrow)) | (bbit & borrow);
    r.setBit(i, diff);
    borrow = newBorrow;
  }
  return r; // Wrap semantics modulo 2^w.
}

UInt UInt::multiply(const UInt &a, const UInt &b) {
  return UInt(mulUnsigned(a, b));
}

int UInt::compare(const UInt &a, const UInt &b) {
  size_t w = a.bitWidth > b.bitWidth ? a.bitWidth : b.bitWidth;
  for (size_t idx = 0; idx < w; ++idx) {
    size_t i = w - 1 - idx;
    bool abit = (i < a.bitWidth) ? a.getBit(i) : false;
    bool bbit = (i < b.bitWidth) ? b.getBit(i) : false;
    if (abit != bbit)
      return abit < bbit ? -1 : 1;
  }
  return 0;
}
