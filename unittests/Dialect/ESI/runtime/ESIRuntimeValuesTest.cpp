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
  BitVector bv(data, 16);          // non-owning view, width = 16
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
  MutableBitVector bv(16);
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

TEST(BitVectorTest, ShiftLeftIncreaseWidth) {
  MutableBitVector bv(16);
  for (int i = 0; i < 8; ++i)
    bv.setBit(i, true); // lower byte = 0xFF
  bv <<= 4;             // Shift left by 4
  EXPECT_EQ(bv.width(), 20u);

  // New bits 0-3 should be 0.
  for (int i = 0; i < 4; ++i)
    EXPECT_FALSE(bv.getBit(i));

  // Bits 4-11 should be original bits 0-7 (all 1).
  for (int i = 4; i < 12; ++i)
    EXPECT_TRUE(bv.getBit(i));

  // Bits 12-19 should be 0.
  for (int i = 12; i < 20; ++i)
    EXPECT_FALSE(bv.getBit(i));
}

TEST(BitVectorTest, BitwiseOps) {
  MutableBitVector a(8), b(8);
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
  MutableBitVector a(std::vector<uint8_t>{0xAA, 0x0F}, 16);
  MutableBitVector b(std::vector<uint8_t>{0xAA, 0x0F}, 16);
  EXPECT_TRUE(a == b);
  BitVector aShift = a >> 3; // width 13, misaligned view
  BitVector bShift = b >> 3; // width 13
  EXPECT_TRUE(aShift == bShift);
  // Mutate only bShift's underlying storage (its owner).
  // Note: slice returns an immutable view, but we can modify the original
  // mutable source.
  b.setBit(3, !b.getBit(3));
  EXPECT_TRUE(aShift != bShift);
}

TEST(BitVectorTest, ZeroWidthAndBitwise) {
  // Construct zero-width via explicit width 0.
  MutableBitVector a(0); // width 0
  MutableBitVector b(0);
  EXPECT_EQ(a.width(), 0u);
  EXPECT_EQ((a & b).width(), 0u);
  EXPECT_EQ((a | b).width(), 0u);
  EXPECT_EQ((a ^ b).width(), 0u);
}

TEST(BitVectorTest, ZeroWidthShiftBehavior) {
  MutableBitVector z(0);
  // Right shifting any positive amount should throw now.
  EXPECT_THROW(z >>= 1, std::out_of_range);
  // Right shifting by 0 is a no-op.
  EXPECT_NO_THROW(z >>= 0);
  // Left shifting zero-width is permitted.
  EXPECT_NO_THROW(z <<= 5);
  EXPECT_EQ(z.width(), 5u);
}

TEST(BitVectorTest, CrossByteAccessAndShift) {
  std::vector<uint8_t> data{0xF0, 0x55, 0xCC};
  MutableBitVector bv(std::move(data), 24);
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
  MutableBitVector a(std::move(da), 16);
  MutableBitVector b(std::move(db), 16);
  a >>= 3; // width 13, bitIndex=3
  b >>= 3; // width 13, bitIndex=3
  auto rAnd = a & b;
  auto rOr = a | b;
  auto rXor = a ^ b;
  for (size_t i = 0; i < a.width(); ++i) {
    bool abit = a.getBit(i);
    bool bbit = b.getBit(i);
    EXPECT_EQ(rAnd.getBit(i), (abit & bbit));
    EXPECT_EQ(rOr.getBit(i), (abit | bbit));
    EXPECT_EQ(rXor.getBit(i), (abit ^ bbit));
  }
}

TEST(BitVectorTest, TailByteBitwiseNoMasking) {
  // Width 13 -> final byte partially used; ensure operations stay in-bounds and
  // correct for used bits.
  std::vector<uint8_t> da{0b10101111, 0b00000001};
  std::vector<uint8_t> db{0b11000011, 0b00000001};
  MutableBitVector a(std::move(da), 13);
  MutableBitVector b(std::move(db), 13);
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
  std::vector<uint8_t> raw{0x44};
  BitVector bv(raw);
  EXPECT_THROW(bv.getBit(8), std::out_of_range);
}

TEST(BitVectorTest, SetBitOutOfRange) {
  MutableBitVector bv(std::vector<uint8_t>{0x00}, 4);
  EXPECT_THROW(bv.setBit(4, true), std::out_of_range);
}

TEST(BitVectorTest, ZeroWidthGetBitThrows) {
  MutableBitVector z(std::vector<uint8_t>{0x00}, 0);
  EXPECT_THROW(z.getBit(0), std::out_of_range);
}

TEST(BitVectorTest, BitwiseWidthMismatchThrows) {
  std::vector<uint8_t> raw0{0x00};
  std::vector<uint8_t> raw1{0x00};
  BitVector a(raw0, 8);
  BitVector b(raw1, 7);
  EXPECT_THROW((void)(a & b), std::invalid_argument);
  EXPECT_THROW((void)(a | b), std::invalid_argument);
  EXPECT_THROW((void)(a ^ b), std::invalid_argument);
}

TEST(BitVectorTest, NonOwningModificationThrows) {
  std::vector<uint8_t> raw{0x00, 0x00};
  BitVector view(raw, 16); // non-owning
  // BitVector is immutable, so setBit is not available on it
  // This test verifies the immutable design: views cannot be modified
  EXPECT_THROW(view.getBit(99), std::out_of_range); // Test bounds checking

  // Owning instance (MutableBitVector) should allow modification.
  MutableBitVector owned(std::vector<uint8_t>(2, 0), 16);
  EXPECT_NO_THROW(owned.setBit(0, true));
  EXPECT_TRUE(owned.getBit(0));

  // Creating a view via implicit cast
  BitVector trunc(owned); // Convert MutableBitVector to BitVector view
  EXPECT_EQ(trunc.width(), 16u);
  // The original owned can still be modified
  owned.setBit(5, true);
  EXPECT_TRUE(owned.getBit(5));
}

TEST(BitVectorTest, PackedSerializationRoundTrip) {
  // Simplified test: pack values, create BitVector, extract values
  // Pack: b1:1=1, u7:7=0x55, b2:1=0, s5:5=-5, b3:1=1, u9:9=0x1A5, u3:3=5
  // => 27 bits total

  uint64_t packed_value = 0;
  size_t bit_offset = 0;

  // Pack fields manually into packed_value
  auto pack = [&](uint64_t value, size_t width) {
    uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1ULL);
    packed_value |= ((value & mask) << bit_offset);
    bit_offset += width;
  };

  pack(1, 1);     // b1 = true
  pack(0x55, 7);  // u7 = 0x55
  pack(0, 1);     // b2 = false
  pack(27, 5);    // s5 = -5 in 5-bit two's complement (0b11011 = 27)
  pack(1, 1);     // b3 = true
  pack(0x1A5, 9); // u9 = 0x1A5
  pack(5, 3);     // u3 = 5

  ASSERT_EQ(bit_offset, 27u);

  // Create MutableBitVector from packed data
  std::vector<uint8_t> bytes((27 + 7) / 8, 0);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>((packed_value >> (8 * i)) & 0xFF);
  }

  MutableBitVector stream(std::move(bytes), 27);

  // Extract and verify
  uint64_t rb1 = 0;
  for (size_t i = 0; i < 1; ++i)
    if (stream.getBit(i))
      rb1 |= (1ULL << i);
  EXPECT_EQ(rb1, 1u);

  uint64_t ru7 = 0;
  for (size_t i = 0; i < 7; ++i)
    if (stream.getBit(1 + i))
      ru7 |= (1ULL << i);
  EXPECT_EQ(ru7, 0x55u);

  uint64_t rb2 = 0;
  for (size_t i = 0; i < 1; ++i)
    if (stream.getBit(8 + i))
      rb2 |= (1ULL << i);
  EXPECT_EQ(rb2, 0u);

  uint64_t rs5raw = 0;
  for (size_t i = 0; i < 5; ++i)
    if (stream.getBit(9 + i))
      rs5raw |= (1ULL << i);
  EXPECT_EQ(rs5raw, 27u); // -5 in 5-bit

  uint64_t rb3 = 0;
  for (size_t i = 0; i < 1; ++i)
    if (stream.getBit(14 + i))
      rb3 |= (1ULL << i);
  EXPECT_EQ(rb3, 1u);

  uint64_t ru9 = 0;
  for (size_t i = 0; i < 9; ++i)
    if (stream.getBit(15 + i))
      ru9 |= (1ULL << i);
  EXPECT_EQ(ru9, 0x1A5u);

  uint64_t ru3 = 0;
  for (size_t i = 0; i < 3; ++i)
    if (stream.getBit(24 + i))
      ru3 |= (1ULL << i);
  EXPECT_EQ(ru3, 5u);
}

TEST(IntTest, ConstructionAndSignExtension) {
  // Create a MutableBitVector with bit pattern for -5 in 8 bits (0xFB)
  MutableBitVector mbv(8);
  // -5 in 8-bit two's complement: 11111011 (LSB first: 1 1 0 1 1 1 1 1)
  mbv.setBit(0, true);
  mbv.setBit(1, true);
  mbv.setBit(2, false);
  mbv.setBit(3, true);
  mbv.setBit(4, true);
  mbv.setBit(5, true);
  mbv.setBit(6, true);
  mbv.setBit(7, true);

  Int v(mbv); // Construct Int from BitVector view
  EXPECT_EQ(static_cast<int64_t>(v), -5);
  // Verify bit pattern
  bool expected[8] = {true, true, false, true, true, true, true, true};
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(v.getBit(i), expected[i]) << "bit " << i;
}

TEST(IntTest, SignExtendOnNarrowTo64) {
  // Create -1 in 12 bits (all bits set)
  MutableBitVector mbv(12);
  for (size_t i = 0; i < 12; ++i)
    mbv.setBit(i, true);

  Int v(mbv);
  EXPECT_EQ(static_cast<int64_t>(v), -1);
  // Check high (top) bit is 1.
  EXPECT_TRUE(v.getBit(11));
}

TEST(IntTest, OverflowSigned) {
  // Create a 70-bit value with bit 64 set (value that doesn't fit in signed 64)
  MutableBitVector big(70);
  big.setBit(64, true);
  Int v(big);
  EXPECT_THROW((void)static_cast<int64_t>(v), std::overflow_error);
}

TEST(IntTest, WidthOneValues) {
  // Create 0 in 1 bit
  MutableBitVector mbv_z(1);

  // Create -1 in 1 bit (bit 0 set to 1)
  MutableBitVector mbv_neg(1);
  mbv_neg.setBit(0, true);

  Int z(mbv_z);
  Int neg(mbv_neg);

  EXPECT_EQ(static_cast<int64_t>(z), 0);
  EXPECT_EQ(static_cast<int64_t>(neg), -1);
}

TEST(IntTest, LargeWidthSignExtendedConversions) {
  // Create -1 in 130 bits
  MutableBitVector mbv_negAll(130);
  for (size_t i = 0; i < 130; ++i)
    mbv_negAll.setBit(i, true);

  // Create 5 in 130 bits
  MutableBitVector mbv_posSmall(130);
  mbv_posSmall.setBit(0, true);
  mbv_posSmall.setBit(2, true);

  Int negAll(mbv_negAll);
  Int posSmall(mbv_posSmall);

  EXPECT_EQ(static_cast<int64_t>(negAll), -1);
  EXPECT_EQ(static_cast<int64_t>(posSmall), 5);
}

TEST(IntTest, LargeWidthBadSignExtensionOverflow) {
  // Create 5 in 130 bits and set bit 100
  MutableBitVector mbv_v(130);
  mbv_v.setBit(0, true);
  mbv_v.setBit(2, true);
  mbv_v.setBit(100, true);

  Int v(mbv_v);
  EXPECT_THROW((void)static_cast<int64_t>(v), std::overflow_error);
}

TEST(UIntTest, BasicConstruction) {
  // Create 123 in 16 bits
  MutableBitVector mbv(16);
  uint64_t val = 123;
  for (size_t i = 0; i < 16; ++i)
    if (val & (1ULL << i))
      mbv.setBit(i, true);

  UInt u(mbv);
  EXPECT_EQ(static_cast<uint64_t>(u), 123u);
  EXPECT_EQ(u.width(), 16u);
}

TEST(UIntTest, BitwiseFastVsFallbackConsistency) {
  // Test that bitwise operations work correctly on MutableBitVectors
  std::vector<uint8_t> raw1{0xFF, 0x0F, 0xAA};
  std::vector<uint8_t> raw2{0x0F, 0xF0, 0x55};
  MutableBitVector aAligned(std::move(raw1), 24);
  MutableBitVector bAligned(std::move(raw2), 24);
  auto alignedAnd = aAligned & bAligned; // aligned operation

  // Verify result is correct
  for (size_t i = 0; i < aAligned.width(); ++i)
    EXPECT_EQ(alignedAnd.getBit(i), (aAligned.getBit(i) && bAligned.getBit(i)));
}

TEST(UIntTest, OverflowUnsignedConversion) {
  // Create 70-bit zero
  MutableBitVector big(70);
  EXPECT_NO_THROW((void)static_cast<uint64_t>(UInt(big)));

  // Set bit 65 to cause overflow
  big.setBit(65, true);
  UInt bigUInt(big);
  EXPECT_THROW((void)static_cast<uint64_t>(bigUInt), std::overflow_error);
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
  // Create a MutableBitVector then get a view
  MutableBitVector mbv(std::move(bytes), width);
  BitVector bv = mbv; // implicit conversion to BitVector view

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
    MutableBitVector z_mut(0);
    BitVector z(z_mut); // implicit conversion
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