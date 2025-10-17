//===- ESIRuntimeValuesTest.cpp - ESI Runtime Values System Tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Values.h"
#include "gtest/gtest.h"
#include <any>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>


using namespace esi;

namespace {

TEST(BitVectorTest, BasicConstructionAndAccess) {
  std::vector<uint8_t> data{0xAA,
                            0x0F}; // 0b10101010 00001111 (LSB first per byte)
  BitVector bv(data);              // width = 16
  EXPECT_EQ(bv.width(), 16u);
  // Check a few bits: byte layout little-endian bit ordering.
  EXPECT_EQ(bv.getBit(0), false); // 0xAA LSB is 0
  EXPECT_EQ(bv.getBit(1), true);
  EXPECT_EQ(bv.getBit(2), false);
  EXPECT_EQ(bv.getBit(3), true);
  EXPECT_EQ(bv.getBit(7), true); // MSB of 0xAA
  // Bits from second byte (0x0F): lower 4 bits 1111
  EXPECT_TRUE(bv.getBit(8));
  EXPECT_TRUE(bv.getBit(9));
  EXPECT_TRUE(bv.getBit(10));
  EXPECT_TRUE(bv.getBit(11));
  EXPECT_FALSE(bv.getBit(12));
}

TEST(BitVectorTest, ShiftRightShrinksWidth) {
  BitVector bv(16);
  // Set pattern 0b...0001111 (low 4 bits set)
  for (int i = 0; i < 4; ++i)
    bv.setBit(i, true);
  bv >>= 2; // Drop two LSBs
  EXPECT_EQ(bv.width(), 14u);
  // New LSB should be original bit 2.
  EXPECT_TRUE(bv.getBit(0));
  EXPECT_TRUE(bv.getBit(1));
  // Overshift beyond remaining width should throw (width currently 14).
  EXPECT_THROW(bv >>= 20, std::out_of_range);
}

TEST(BitVectorTest, ShiftLeftShrinksWidth) {
  BitVector bv(16);
  for (int i = 0; i < 8; ++i)
    bv.setBit(i, true); // lower byte = 0xFF
  bv <<= 4;             // Drop 4 MSBs
  EXPECT_EQ(bv.width(), 12u);
  // Highest remaining bit index is 11; bit 11 was original bit 11.
  EXPECT_TRUE(bv.getBit(0));
}

TEST(BitVectorTest, BitwiseOps) {
  BitVector a(8), b(8);
  // a = 0b10101010, b = 0b11001100
  for (int i = 0; i < 8; ++i) {
    a.setBit(i, (i % 2) == 1); // set odd bits
    b.setBit(i, (i / 2) % 2 ==
                    0); // pattern 0011 blocks -> reversed little-endian logic
  }
  auto c_and = a & b;
  auto c_or = a | b;
  auto c_xor = a ^ b;
  EXPECT_EQ(c_and.width(), 8u);
  // Validate a few bits
  for (int i = 0; i < 8; ++i) {
    bool abit = a.getBit(i);
    bool bbit = b.getBit(i);
    EXPECT_EQ(c_and.getBit(i), (abit & bbit));
    EXPECT_EQ(c_or.getBit(i), (abit | bbit));
    EXPECT_EQ(c_xor.getBit(i), (abit ^ bbit));
  }
}

TEST(BitVectorTest, EqualityAlignedAndMisaligned) {
  // Aligned equality with two views of the same underlying storage (aliasing)
  // -- non-owning.
  {
    std::vector<uint8_t> raw{0xAA, 0x0F};
    BitVector a0(raw, 16);
    BitVector b0(raw, 16);
    EXPECT_TRUE(a0 == b0);
  }
  // Now create two owning buffers with identical contents to ensure equality
  // does not depend on pointer identity AND we can mutate one.
  BitVector a(std::vector<uint8_t>{0xAA, 0x0F}, 16);
  BitVector b(std::vector<uint8_t>{0xAA, 0x0F}, 16);
  EXPECT_TRUE(a == b);
  BitVector aShift = a >> 3; // width 13, misaligned view
  BitVector bShift = b >> 3; // width 13
  EXPECT_TRUE(aShift == bShift);
  // Mutate only bShift's underlying storage (its owner).
  bShift.setBit(0, !bShift.getBit(0));
  EXPECT_TRUE(aShift != bShift);
}

TEST(BitVectorTest, ZeroWidthAndBitwise) {
  // Construct zero-width via explicit width 0.
  BitVector a(0); // width 0
  BitVector b(0);
  EXPECT_EQ(a.width(), 0u);
  EXPECT_EQ((a & b).width(), 0u);
  EXPECT_EQ((a | b).width(), 0u);
  EXPECT_EQ((a ^ b).width(), 0u);
}

TEST(BitVectorTest, ZeroWidthShiftBehavior) {
  BitVector z(0);
  // Right shifting any positive amount should throw now.
  EXPECT_THROW(z >>= 1, std::out_of_range);
  // Right shifting by 0 is a no-op.
  EXPECT_NO_THROW(z >>= 0);
  // Left shifting zero-width is permitted (remains zero width per
  // implementation).
  EXPECT_NO_THROW(z <<= 5);
  EXPECT_EQ(z.width(), 0u);
}

TEST(BitVectorTest, CrossByteAccessAndShift) {
  std::vector<uint8_t> data{0xF0, 0x55, 0xCC};
  BitVector bv(data, 24);
  bv >>= 3; // misalign
  EXPECT_EQ(bv.width(), 21u);
  // Original byte 0 (0xF0) bits LSB->MSB: 0 0 0 0 1 1 1 1. After dropping 3
  // LSBs, new bit0 is old bit3 (0).
  EXPECT_FALSE(bv.getBit(0));
  // Spot check a cross-byte bit: choose bit 8 post-shift.
  (void)bv.getBit(8); // Ensure no exceptions.
}

TEST(BitVectorTest, MisalignedBitwiseFallback) {
  std::vector<uint8_t> da{0xAA, 0xCC};
  std::vector<uint8_t> db{0x0F, 0xF0};
  BitVector a(da, 16);
  BitVector b(db, 16);
  a >>= 3; // width 13, bitIndex=3
  b >>= 3; // width 13, bitIndex=3
  auto r_and = a & b;
  auto r_or = a | b;
  auto r_xor = a ^ b;
  for (size_t i = 0; i < a.width(); ++i) {
    bool abit = a.getBit(i);
    bool bbit = b.getBit(i);
    EXPECT_EQ(r_and.getBit(i), (abit & bbit));
    EXPECT_EQ(r_or.getBit(i), (abit | bbit));
    EXPECT_EQ(r_xor.getBit(i), (abit ^ bbit));
  }
}

TEST(BitVectorTest, TailByteBitwiseNoMasking) {
  // Width 13 -> final byte partially used; ensure operations stay in-bounds and
  // correct for used bits.
  std::vector<uint8_t> da{0b10101111, 0b00000001};
  std::vector<uint8_t> db{0b11000011, 0b00000001};
  BitVector a(da, 13);
  BitVector b(db, 13);
  auto r = a ^ b;
  for (size_t i = 0; i < 13; ++i)
    EXPECT_EQ(r.getBit(i), a.getBit(i) ^ b.getBit(i));
}

TEST(BitVectorTest, ConstructorInvalidBitIndex) {
  std::vector<uint8_t> raw{0xAA, 0xBB};
  EXPECT_THROW(BitVector(raw, 16, 8), std::invalid_argument); // bitIndex > 7
}

TEST(BitVectorTest, ConstructorWidthExceedsStorage) {
  std::vector<uint8_t> raw{0xAA}; // 8 bits available
  EXPECT_THROW(BitVector(raw, 16, 0), std::invalid_argument); // request 16
}

TEST(BitVectorTest, GetBitOutOfRange) {
  BitVector bv(8);
  EXPECT_THROW(bv.getBit(8), std::out_of_range);
}

TEST(BitVectorTest, SetBitOutOfRange) {
  BitVector bv(4);
  EXPECT_THROW(bv.setBit(4, true), std::out_of_range);
}

TEST(BitVectorTest, ZeroWidthGetBitThrows) {
  BitVector z(0);
  EXPECT_THROW(z.getBit(0), std::out_of_range);
}

TEST(BitVectorTest, BitwiseWidthMismatchThrows) {
  BitVector a(8);
  BitVector b(7);
  EXPECT_THROW((void)(a & b), std::invalid_argument);
  EXPECT_THROW((void)(a | b), std::invalid_argument);
  EXPECT_THROW((void)(a ^ b), std::invalid_argument);
}

TEST(BitVectorTest, NonOwningModificationThrows) {
  std::vector<uint8_t> raw{0x00, 0x00};
  BitVector view(std::span<const uint8_t>(raw.data(), raw.size()),
                 16); // non-owning
  EXPECT_THROW(view.setBit(0, true), std::runtime_error);
  // Owning instance should allow modification.
  BitVector owned(std::vector<uint8_t>(2, 0), 16);
  EXPECT_NO_THROW(owned.setBit(0, true));
  EXPECT_TRUE(owned.getBit(0));
  // Truncated view of owning BitVector retains ownership, so modification
  // allowed.
  BitVector trunc = owned.slice(0, 8);
  EXPECT_NO_THROW(trunc.setBit(1, true));
  EXPECT_TRUE(trunc.getBit(1));
}

TEST(BitVectorTest, PackedSerializationRoundTrip) {
  // Same field layout as before, but pack via shifts and masks (no setBit):
  //  b1:1, u7:7, b2:1, s5:5, b3:1, u9:9, u3:3 => 27 bits total.
  bool b1 = true;
  uint64_t u7 = 0x55; // 0b1010101
  bool b2 = false;
  Int s5(-5, 5); // two's complement 5-bit
  bool b3 = true;
  uint64_t u9 = 0x1A5; // 9 bits
  uint64_t u3 = 0x5;   // 3 bits
  size_t totalWidth = 27;

  // Gather s5 raw bit pattern.
  uint64_t s5bits = 0;
  for (size_t i = 0; i < s5.width(); ++i)
    if (s5.getBit(i))
      s5bits |= (1ULL << i);

  // Pack little-endian (LSB-first) by shifting the accumulator.
  uint64_t acc = 0;
  size_t offset = 0;
  auto packField = [&](uint64_t value, size_t width) {
    uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1ULL);
    acc |= ((value & mask) << offset);
    offset += width;
  };

  packField(b1 ? 1 : 0, 1);
  packField(u7, 7);
  packField(b2 ? 1 : 0, 1);
  packField(s5bits, 5);
  packField(b3 ? 1 : 0, 1);
  packField(u9, 9);
  packField(u3, 3);
  ASSERT_EQ(offset, totalWidth);

  // Materialize BitVector storage from accumulator bytes.
  std::vector<uint8_t> bytes((totalWidth + 7) / 8, 0);
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<uint8_t>((acc >> (8 * i)) & 0xFF);
  BitVector packed(bytes, totalWidth);

  // Deserialize by masking then truncating.
  auto extract = [](BitVector &v, size_t width) -> uint64_t {
    // Build mask bytes efficiently.
    BitVector mask(width); // mask view width 'width'
    // Fill mask bits to 1.
    size_t fullBytes = width / 8;
    size_t tail = width % 8;
    for (size_t i = 0; i < fullBytes * 8; ++i)
      mask.setBit(i, true);
    if (tail) {
      for (size_t i = fullBytes * 8; i < fullBytes * 8 + tail; ++i)
        mask.setBit(i, true);
    }
    // Align mask to stream width by expanding via bitwise OR with zero-extended
    // padding. We need a mask BitVector of the same width as 'v'. Construct
    // temporary of v.width().
    BitVector fullMask(v.width());
    for (size_t i = 0; i < width; ++i)
      fullMask.setBit(i, mask.getBit(i));
    BitVector masked = v & fullMask;
    // Truncate masked bits to requested width for conversion.
    BitVector truncated = masked.slice(0, width);
    UInt val(truncated);
    uint64_t out = static_cast<uint64_t>(val);
    v >>= width; // consume bits
    return out;
  };
  auto signExtend = [](uint64_t val, size_t width) -> int64_t {
    if (width == 0)
      return 0;
    if (val & (1ULL << (width - 1))) {
      uint64_t mask = ~0ULL << width;
      val |= mask;
    }
    return static_cast<int64_t>(val);
  };

  BitVector stream = packed; // copy for reading
  uint64_t rb1 = extract(stream, 1);
  uint64_t ru7 = extract(stream, 7);
  uint64_t rb2 = extract(stream, 1);
  uint64_t rs5raw = extract(stream, 5);
  uint64_t rb3 = extract(stream, 1);
  uint64_t ru9 = extract(stream, 9);
  uint64_t ru3 = extract(stream, 3);
  EXPECT_EQ(stream.width(), 0u);

  EXPECT_EQ(rb1, b1 ? 1u : 0u);
  EXPECT_EQ(ru7, u7 & 0x7FULL);
  EXPECT_EQ(rb2, b2 ? 1u : 0u);
  EXPECT_EQ(signExtend(rs5raw, 5), -5);
  EXPECT_EQ(rb3, b3 ? 1u : 0u);
  EXPECT_EQ(ru9, u9 & 0x1FFULL);
  EXPECT_EQ(ru3, u3 & 0x7ULL);
}

TEST(IntTest, ConstructionAndSignExtension) {
  Int v(-5, 8); // Expect 0xFB pattern (11111011 MSB->LSB)
  EXPECT_EQ(static_cast<int64_t>(v), -5);
  // LSB-first expected bits for 0xFB: 1 1 0 1 1 1 1 1
  bool expected[8] = {1, 1, 0, 1, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(v.getBit(i), expected[i]) << "bit " << i;
}

TEST(IntTest, PositiveFitsUnsigned) {
  Int v(42, 16);
  EXPECT_EQ(static_cast<uint64_t>(v), 42u);
  EXPECT_EQ(static_cast<int64_t>(v), 42);
}

TEST(IntTest, SignExtendOnNarrowTo64) {
  Int v(-1, 12); // -1 in 12 bits should still convert to -1 in 64
  EXPECT_EQ(static_cast<int64_t>(v), -1);
  // Check high (top) bit is 1.
  EXPECT_TRUE(v.getBit(11));
}

TEST(IntTest, OverflowSigned) {
  // Create a value that cannot fit in signed 64: width 70 with pattern not
  // sign-extended.
  Int big(0, 70);
  // Set bits 0..63 = 0, bit 64 = 1, others 0 => positive value > 2^63-1 OR
  // inconsistent sign.
  big.setBit(64, true);
  EXPECT_THROW((void)static_cast<int64_t>(big), std::overflow_error);
}

TEST(IntTest, OverflowUnsigned) {
  Int big(0, 70);
  big.setBit(65, true);
  EXPECT_THROW((void)static_cast<uint64_t>(big), std::overflow_error);
}

TEST(IntTest, Addition) {
  Int a(13, 8);  // 13
  Int b(-5, 8);  // -5
  Int c = a + b; // Expect 8 (with possible extended width)
  EXPECT_EQ(static_cast<int64_t>(c), 8);
}

TEST(IntTest, Subtraction) {
  Int a(10, 8);
  Int b(3, 8);
  Int c = a - b; // 7
  EXPECT_EQ(static_cast<int64_t>(c), 7);
}

TEST(IntTest, Multiplication) {
  Int a(-3, 8);
  Int b(7, 8);
  Int c = a * b; // -21
  EXPECT_EQ(static_cast<int64_t>(c), -21);
}

TEST(IntTest, NegativeSubtractionResult) {
  Int a(3, 8);
  Int b(10, 8);
  Int c = a - b; // -7
  EXPECT_EQ(static_cast<int64_t>(c), -7);
}

TEST(IntComparison, DifferentWidthsSignExtendedEquality) {
  Int a(-1, 8);  // 0xFF
  Int b(-1, 20); // sign-extended
  EXPECT_TRUE(a == b);
  Int c(1, 20);
  EXPECT_TRUE(c > a);
  EXPECT_TRUE(a < c);
}

TEST(IntComparison, OrderingPositiveNegative) {
  Int pos(5, 8);
  Int neg(-2, 8);
  EXPECT_TRUE(neg < pos);
  EXPECT_TRUE(!(pos < neg));
  EXPECT_TRUE(pos > neg);
}

TEST(IntComparison, WideVsNarrow) {
  Int narrow(127, 8); // 0x7F
  Int wide(127, 64);  // sign-extended positive
  Int bigger(128, 9); // 0x80 (positive since sign bit cleared in 9-bit?)
                      // Actually sign bit=bit8 -> 128 positive
  EXPECT_TRUE(narrow == wide);
  EXPECT_TRUE(bigger > narrow);
}

TEST(IntTest, WidthOneValues) {
  Int z(0, 1);
  Int neg(-1, 1);
  EXPECT_EQ(static_cast<int64_t>(z), 0);
  EXPECT_EQ(static_cast<int64_t>(neg), -1);
}

TEST(IntTest, AdditionCarryAcrossBytes) {
  Int a(0x00FF, 16);
  Int b(0x0001, 16);
  Int c = a + b; // expect 256, width widened
  EXPECT_GE(c.width(), 17u);
  EXPECT_EQ(static_cast<uint64_t>(c), 256u);
}

TEST(IntTest, MultiplyNegativePairs) {
  Int a(-7, 8), b(-3, 8);
  Int c = a * b; // 21
  EXPECT_EQ(static_cast<int64_t>(c), 21);
}

TEST(IntTest, LargeWidthSignExtendedConversions) {
  Int negAll(-1, 130);
  EXPECT_EQ(static_cast<int64_t>(negAll), -1);
  Int posSmall(5, 130);
  EXPECT_EQ(static_cast<int64_t>(posSmall), 5);
}

TEST(IntTest, LargeWidthBadSignExtensionOverflow) {
  Int v(5, 130);       // positive, high bits zero
  v.setBit(100, true); // flip a bit above 64 but below sign bit
  EXPECT_THROW((void)static_cast<int64_t>(v), std::overflow_error);
}

TEST(UIntTest, BasicConstruction) {
  UInt u(123u, 16); // 16-bit unsigned
  EXPECT_EQ(static_cast<uint64_t>(u), 123u);
  EXPECT_EQ(u.width(), 16u);
}

TEST(UIntTest, Addition) {
  UInt a(200u, 8); // 8-bit width -> stores 200 mod 256
  UInt b(100u, 8);
  UInt c = a + b; // Width could expand to 9 bits per implementation
  EXPECT_EQ(static_cast<uint64_t>(c) & 0x1FFu, 300u); // Accept modulo semantics
}

TEST(UIntTest, SubtractionWrap) {
  UInt a(5u, 8);
  UInt b(10u, 8);
  UInt c = a - b; // Wraps modulo 256
  EXPECT_EQ(static_cast<uint64_t>(c), (uint64_t)((5u - 10u) & 0xFFu));
}

TEST(UIntTest, Multiplication) {
  UInt a(13u, 8);
  UInt b(17u, 8);
  UInt c = a * b; // 221
  EXPECT_EQ(static_cast<uint64_t>(c) & 0xFFFFu, 221u);
}

TEST(UIntComparison, DifferentWidthsZeroExtended) {
  UInt a(255u, 8);
  UInt b(255u, 16);
  EXPECT_TRUE(a == b);
  UInt c(256u, 16);
  EXPECT_TRUE(c > b);
  EXPECT_TRUE(b < c);
}

TEST(UIntComparison, BasicOrdering) {
  UInt a(5u, 8);
  UInt b(2u, 4);
  EXPECT_TRUE(a > b);
  EXPECT_TRUE(!(b > a));
  EXPECT_TRUE(b < a);
}

TEST(UIntTest, SubtractWrapCrossByte) {
  UInt a(0x0100, 17); // 256
  UInt b(0x0001, 17); // 1
  UInt c = a - b;     // 255
  EXPECT_EQ(static_cast<uint64_t>(c), 255u);
}

TEST(UIntTest, MultiplyWiden) {
  UInt a(0x00FF, 16);
  UInt b(0x0002, 16);
  UInt c = a * b; // 510
  EXPECT_EQ(static_cast<uint64_t>(c) & 0xFFFFu, 510u);
}

TEST(UIntTest, LargeWidthArithmetic) {
  UInt a(1, 129);
  a.setBit(128, true); // set top bit
  UInt b(3, 129);
  UInt sum = a + b;
  EXPECT_TRUE(sum.getBit(128));
  EXPECT_EQ(sum.getBit(0), ((1 + 3) & 1));
}

TEST(UIntTest, BitwiseFastVsFallbackConsistency) {
  std::vector<uint8_t> raw1{0xFF, 0x0F, 0xAA};
  std::vector<uint8_t> raw2{0x0F, 0xF0, 0x55};
  BitVector aAligned(raw1, 24);
  BitVector bAligned(raw2, 24);
  BitVector alignedAnd = aAligned & bAligned; // fast path
  BitVector aMis = aAligned >> 3;             // fallback path candidate
  BitVector bMis = bAligned >> 3;
  BitVector misAnd = aMis & bMis;
  for (size_t i = 0; i < aMis.width(); ++i)
    EXPECT_EQ(misAnd.getBit(i), (aMis.getBit(i) && bMis.getBit(i)));
  for (size_t i = 0; i < aAligned.width(); ++i)
    EXPECT_EQ(alignedAnd.getBit(i), (aAligned.getBit(i) && bAligned.getBit(i)));
}

TEST(UIntTest, OverflowUnsignedConversion) {
  UInt big(0, 70); // 70-bit zero (fits)
  EXPECT_NO_THROW((void)static_cast<uint64_t>(big));
  big.setBit(65, true); // Set bit above 64
  EXPECT_THROW((void)static_cast<uint64_t>(big), std::overflow_error);
}

// New comprehensive formatting tests for BitVector string/stream output.
TEST(BitVectorFormatTest, ToStringAndStreamBases) {
  // Value: 0x1F3 = 499 dec = 0o763, choose width 12 to exercise leading zeros
  // in binary.
  const uint64_t val = 0x1F3ULL;
  const size_t width =
      12; // ensures three leading zero bits in binary representation
  std::vector<uint8_t> bytes((width + 7) / 8, 0);
  for (size_t i = 0; i < width; ++i)
    if (val & (1ULL << i))
      bytes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  BitVector bv(bytes, width);

  // Direct base conversions.
  EXPECT_EQ(bv.toString(), std::string("1f3")); // default base=16
  EXPECT_EQ(bv.toString(16), std::string("1f3"));
  EXPECT_EQ(bv.toString(10), std::string("499"));
  EXPECT_EQ(bv.toString(8), std::string("763"));
  EXPECT_EQ(bv.toString(2), std::string("000111110011")); // width-length binary
  // Unsupported base falls back to binary.
  EXPECT_EQ(bv.toString(3), std::string("000111110011"));

  // Stream formatting: dec (default), hex, oct, showbase, uppercase.
  {
    std::ostringstream oss;
    oss << bv; // default dec
    EXPECT_EQ(oss.str(), "499");
  }
  {
    std::ostringstream oss;
    oss << std::dec << std::showbase << bv; // showbase ignored for dec
    EXPECT_EQ(oss.str(), "499");
  }
  {
    std::ostringstream oss;
    oss << std::hex << bv;
    EXPECT_EQ(oss.str(), "1f3");
  }
  {
    std::ostringstream oss;
    oss << std::showbase << std::hex << bv;
    EXPECT_EQ(oss.str(), "0x1f3");
  }
  {
    std::ostringstream oss;
    oss << std::uppercase << std::showbase << std::hex << bv;
    EXPECT_EQ(oss.str(), "0X1F3");
  }
  {
    std::ostringstream oss;
    oss << std::oct << bv;
    EXPECT_EQ(oss.str(), "763");
  }
  {
    std::ostringstream oss;
    oss << std::showbase << std::oct << bv;
    EXPECT_EQ(oss.str(), "0763");
  }

  // Zero-width value formatting -> "0" in all bases.
  {
    BitVector z(0);
    std::ostringstream ossHex, ossDec, ossOct;
    ossHex << std::hex << z;
    ossDec << std::dec << z;
    ossOct << std::oct << z;
    EXPECT_EQ(ossHex.str(), "0");
    EXPECT_EQ(ossDec.str(), "0");
    EXPECT_EQ(ossOct.str(), "0");
    EXPECT_EQ(z.toString(2), "0");
  }
}

} // namespace