//===- ESIRuntimeTest.cpp - ESI Runtime Type System Tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Context.h"
#include "esi/Types.h"
#include "esi/Values.h"
#include "gtest/gtest.h"
#include <any>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace esi;

namespace {

// Test VoidType serialization and deserialization
TEST(ESITypesTest, VoidTypeSerialization) {
  VoidType voidType("void");

  // Test valid void value (empty std::any)
  std::any voidValue;
  EXPECT_NO_THROW(voidType.ensureValid(voidValue));

  // Test valid void value (nullptr)
  std::any nullptrValue = std::any(nullptr);
  EXPECT_NO_THROW(voidType.ensureValid(nullptrValue));

  // Test invalid void value
  std::any invalidValue = std::any(42);
  EXPECT_THROW(voidType.ensureValid(invalidValue), std::runtime_error);

  // Test serialization
  MessageData serialized(voidType.serialize(voidValue).getSpan());
  EXPECT_EQ(serialized.getSize(), 1UL)
      << "VoidType serialization should produce exactly 1 byte";
  EXPECT_EQ(serialized.getData()[0], 0)
      << "VoidType serialization should produce a zero byte";

  // Test deserialization
  BitVector v(serialized.getData());
  auto deserialized = voidType.deserialize(v);
  EXPECT_FALSE(deserialized.has_value())
      << "VoidType deserialization should not have a value";
  EXPECT_EQ(v.width(), 0UL)
      << "VoidType deserialization should consume all data";
}

// Test BitsType serialization and deserialization
TEST(ESITypesTest, BitsTypeSerialization) {
  BitsType bitsType("bits8", 8);

  // Test valid bits value
  std::vector<uint8_t> bitsValue = {0xAB};
  std::any validBits = std::any(bitsValue);
  EXPECT_NO_THROW(bitsType.ensureValid(validBits));

  // Test invalid size
  std::vector<uint8_t> wrongSize = {0xAB, 0xCD};
  std::any invalidBits = std::any(wrongSize);
  EXPECT_THROW(bitsType.ensureValid(invalidBits), std::runtime_error);

  // Test serialization
  MessageData serialized(bitsType.serialize(validBits).takeStorage());
  EXPECT_EQ(serialized.getSize(), 1UL)
      << "BitsType(8) serialization should produce exactly 1 byte";
  EXPECT_EQ(serialized.getData()[0], 0xAB)
      << "BitsType serialization should preserve the input byte value";

  // Test deserialization
  BitVector serializedBits(serialized.getData());
  auto deserialized = bitsType.deserialize(serializedBits);
  auto deserializedBits = std::any_cast<std::vector<uint8_t>>(deserialized);
  EXPECT_EQ(deserializedBits.size(), 1UL)
      << "BitsType(8) deserialization should produce 1 byte";
  EXPECT_EQ(deserializedBits[0], 0xAB)
      << "BitsType deserialization should preserve the original value";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "BitsType deserialization should consume all data";
}

// Test UIntType serialization and deserialization
TEST(ESITypesTest, UIntTypeSerialization) {
  UIntType uintType("uint16", 16);

  // Test valid uint value
  uint64_t uintValue = 0x1234;
  std::any validUInt = std::any(uintValue);
  EXPECT_NO_THROW(uintType.ensureValid(validUInt));

  // Test out of range value
  uint64_t outOfRange = 0x10000; // Too big for 16-bit
  std::any invalidUInt = std::any(outOfRange);
  EXPECT_THROW(uintType.ensureValid(invalidUInt), std::runtime_error);

  // Test serialization (little-endian)
  MessageData serialized(uintType.serialize(validUInt).takeStorage());
  EXPECT_EQ(serialized.getSize(), 2UL)
      << "UIntType(16) serialization should produce exactly 2 bytes";
  EXPECT_EQ(serialized.getData()[0], 0x34)
      << "UIntType serialization low byte should be 0x34 (little-endian)";
  EXPECT_EQ(serialized.getData()[1], 0x12)
      << "UIntType serialization high byte should be 0x12 (little-endian)";

  // Test deserialization
  BitVector serializedBits(serialized.getData());
  auto deserialized = uintType.deserialize(serializedBits);
  auto deserializedUInt =
      static_cast<uint16_t>(std::any_cast<UInt>(deserialized));
  EXPECT_EQ(deserializedUInt, 0x1234)
      << "UIntType deserialization should reconstruct original value 0x1234";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "UIntType deserialization should consume all data";

  // Test that different input types work (uint8_t, uint16_t, uint32_t)
  uint8_t smallVal = 42;
  uint16_t mediumVal = 1000;
  uint32_t largeVal = 50000;

  EXPECT_NO_THROW(uintType.ensureValid(std::any(smallVal)));
  EXPECT_NO_THROW(uintType.ensureValid(std::any(mediumVal)));
  EXPECT_NO_THROW(uintType.ensureValid(std::any(largeVal)));
}

// Test SIntType serialization and deserialization
TEST(ESITypesTest, SIntTypeSerialization) {
  SIntType sintType("sint16", 16);

  // Test valid positive sint value
  int64_t positiveValue = 0x1234;
  std::any validSInt = std::any(positiveValue);
  EXPECT_NO_THROW(sintType.ensureValid(validSInt));

  // Test valid negative sint value
  int64_t negativeValue = -1000;
  std::any validNegSInt = std::any(negativeValue);
  EXPECT_NO_THROW(sintType.ensureValid(validNegSInt));

  // Test serialization of positive value
  MessageData serialized(sintType.serialize(validSInt).takeStorage());
  EXPECT_EQ(serialized.getSize(), 2UL)
      << "SIntType(16) serialization should produce exactly 2 bytes";
  EXPECT_EQ(serialized.getData()[0], 0x34)
      << "SIntType serialization low byte should be 0x34 (little-endian)";
  EXPECT_EQ(serialized.getData()[1], 0x12)
      << "SIntType serialization high byte should be 0x12 (little-endian)";

  // Test deserialization
  BitVector serializedBits(serialized.getData());
  auto deserialized = sintType.deserialize(serializedBits);
  auto deserializedSInt =
      static_cast<int16_t>(std::any_cast<Int>(deserialized));
  EXPECT_EQ(deserializedSInt, 0x1234)
      << "SIntType deserialization should reconstruct original positive value";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "SIntType deserialization should consume all data";

  // Test negative value serialization/deserialization
  MessageData negSerialized(sintType.serialize(validNegSInt).takeStorage());
  EXPECT_EQ(negSerialized.getSize(), 2UL)
      << "SIntType(16) negative value should serialize to 2 bytes";
  // -1000 in 16-bit two's complement: 0xFC18
  EXPECT_EQ(negSerialized.getData()[0], 0x18)
      << "SIntType negative serialization low byte should be 0x18 "
         "(little-endian)";
  EXPECT_EQ(negSerialized.getData()[1], 0xFC)
      << "SIntType negative serialization high byte should be 0xFC "
         "(little-endian)";

  BitVector negSerializedBits(negSerialized.getData());
  auto negDeserialized = sintType.deserialize(negSerializedBits);
  auto deserializedNegSInt =
      static_cast<int16_t>(std::any_cast<Int>(negDeserialized));
  EXPECT_EQ(deserializedNegSInt, -1000)
      << "SIntType deserialization should reconstruct original negative value "
         "(-1000)";

  // Test that different input types work (int8_t, int16_t, int32_t)
  int8_t smallVal = -42;
  int16_t mediumVal = -1000;
  int32_t largeVal = -30000;

  EXPECT_NO_THROW(sintType.ensureValid(std::any(smallVal)));
  EXPECT_NO_THROW(sintType.ensureValid(std::any(mediumVal)));
  EXPECT_NO_THROW(sintType.ensureValid(std::any(largeVal)));
}

// Test SIntType sign extension logic comprehensively
TEST(ESITypesTest, SIntTypeSignExtension) {
  // Test 8-bit signed integers
  SIntType sint8("sint8", 8);

  // Test -1 (all bits set in 8-bit: 0xFF)
  int64_t minusOne = -1;
  MessageData serializedMinusOne(
      sint8.serialize(std::any(minusOne)).takeStorage());
  EXPECT_EQ(serializedMinusOne.getSize(), 1UL)
      << "SIntType(8) serialization of -1 should produce 1 byte";
  EXPECT_EQ(serializedMinusOne.getData()[0], 0xFF)
      << "SIntType(8) serialization of -1 should be 0xFF (all bits set)";

  BitVector serializedMinusOneBits(serializedMinusOne.getData());
  auto deserializedMinusOne = sint8.deserialize(serializedMinusOneBits);
  auto resultMinusOne =
      static_cast<int8_t>(std::any_cast<Int>(deserializedMinusOne));
  EXPECT_EQ(resultMinusOne, -1)
      << "SIntType(8) deserialization of 0xFF should reconstruct -1";

  // Test maximum negative value for 8-bit (-128 = 0x80)
  int64_t maxNeg8 = -128;
  MessageData serializedMaxNeg(
      sint8.serialize(std::any(maxNeg8)).takeStorage());
  EXPECT_EQ(serializedMaxNeg.getSize(), 1UL)
      << "SIntType(8) serialization of -128 should produce 1 byte";
  EXPECT_EQ(serializedMaxNeg.getData()[0], 0x80)
      << "SIntType(8) serialization of -128 should be 0x80";

  BitVector serializedMaxNegBits(serializedMaxNeg.getData());
  auto deserializedMaxNeg = sint8.deserialize(serializedMaxNegBits);
  auto resultMaxNeg =
      static_cast<int8_t>(std::any_cast<Int>(deserializedMaxNeg));
  EXPECT_EQ(resultMaxNeg, -128) << "SIntType(8) deserialization should "
                                   "reconstruct maximum negative value (-128)";

  // Test maximum positive value for 8-bit (127 = 0x7F)
  int64_t maxPos8 = 127;
  MessageData serializedMaxPos(
      sint8.serialize(std::any(maxPos8)).takeStorage());
  EXPECT_EQ(serializedMaxPos.getSize(), 1UL)
      << "SIntType(8) serialization of 127 should produce 1 byte";
  EXPECT_EQ(serializedMaxPos.getData()[0], 0x7F)
      << "SIntType(8) serialization of 127 should be 0x7F";

  BitVector serializedMaxPosBits(serializedMaxPos.getData());
  auto deserializedMaxPos = sint8.deserialize(serializedMaxPosBits);
  auto resultMaxPos =
      static_cast<int8_t>(std::any_cast<Int>(deserializedMaxPos));
  EXPECT_EQ(resultMaxPos, 127) << "SIntType(8) deserialization should "
                                  "reconstruct maximum positive value (127)";

  // Test 4-bit signed integers (edge case for smaller widths)
  SIntType sint4("sint4", 4);

  // Test -1 in 4-bit (0x0F in the lower nibble, should sign extend to all 1s)
  int64_t minus1w4bit = -1;
  MessageData serialized4bit(
      sint4.serialize(std::any(minus1w4bit)).takeStorage());
  EXPECT_EQ(serialized4bit.getSize(), 1UL)
      << "SIntType(4) serialization should produce 1 byte";
  EXPECT_EQ(serialized4bit.getData()[0] & 0x0F, 0x0F)
      << "SIntType(4) serialization of -1 should have lower 4 bits set to "
         "1111";

  BitVector serialized4bitBits(serialized4bit.getData());
  auto deserialized4bit = sint4.deserialize(serialized4bitBits);
  auto result4bit = static_cast<int8_t>(std::any_cast<Int>(deserialized4bit));
  EXPECT_EQ(result4bit, -1)
      << "SIntType(4) deserialization should sign-extend -1 correctly";

  // Test maximum negative for 4-bit (-8 = 0x8 in 4 bits)
  int64_t maxNeg4 = -8;
  MessageData serializedMaxNeg4(
      sint4.serialize(std::any(maxNeg4)).takeStorage());
  BitVector serializedMaxNeg4Bits(serializedMaxNeg4.getData());
  auto deserializedMaxNeg4 = sint4.deserialize(serializedMaxNeg4Bits);
  auto resultMaxNeg4 =
      static_cast<int8_t>(std::any_cast<Int>(deserializedMaxNeg4));
  EXPECT_EQ(resultMaxNeg4, -8)
      << "SIntType(4) should handle maximum negative value (-8) correctly";

  // Test maximum positive for 4-bit (7 = 0x7 in 4 bits)
  int64_t maxPos4 = 7;
  MessageData serializedMaxPos4(
      sint4.serialize(std::any(maxPos4)).takeStorage());
  BitVector serializedMaxPos4Bits(serializedMaxPos4.getData());
  auto deserializedMaxPos4 = sint4.deserialize(serializedMaxPos4Bits);
  auto resultMaxPos4 =
      static_cast<int8_t>(std::any_cast<Int>(deserializedMaxPos4));
  EXPECT_EQ(resultMaxPos4, 7)
      << "SIntType(4) should handle maximum positive value (7) correctly";

  // Test 12-bit signed integers (non-byte-aligned case)
  SIntType sint12("sint12", 12);

  // Test -1 in 12-bit (should be 0xFFF in lower 12 bits)
  int64_t minus1w12bit = -1;
  MessageData serialized12bit(
      sint12.serialize(std::any(minus1w12bit)).takeStorage());
  EXPECT_EQ(serialized12bit.getSize(), 2UL)
      << "SIntType(12) serialization should produce 2 bytes (12 bits = 2 "
         "bytes)";
  EXPECT_EQ(serialized12bit.getData()[0], 0xFF)
      << "SIntType(12) serialization of -1 should have lower byte 0xFF";
  EXPECT_EQ(serialized12bit.getData()[1] & 0x0F, 0x0F)
      << "SIntType(12) serialization of -1 should have upper nibble 0x0F";

  BitVector serialized12bitBits(serialized12bit.getData());
  auto deserialized12bit = sint12.deserialize(serialized12bitBits);
  auto result12bit =
      static_cast<int16_t>(std::any_cast<Int>(deserialized12bit));
  EXPECT_EQ(result12bit, -1)
      << "SIntType(12) deserialization should sign-extend -1 correctly";

  // Test a value that requires sign extension: -100 in 12-bit
  int64_t neg100w12bit = -100;
  MessageData serializedNeg100(
      sint12.serialize(std::any(neg100w12bit)).takeStorage());
  BitVector serializedNeg100Bits(serializedNeg100.getData());
  auto deserializedNeg100 = sint12.deserialize(serializedNeg100Bits);
  auto resultNeg100 =
      static_cast<int16_t>(std::any_cast<Int>(deserializedNeg100));
  EXPECT_EQ(resultNeg100, -100)
      << "SIntType(12) should correctly handle sign extension for -100";
}

// Test boundary conditions for sign extension
TEST(ESITypesTest, SIntTypeSignExtensionBoundaries) {
  // Test various bit widths to ensure sign extension works correctly
  for (int width = 1; width <= 16; ++width) {
    SIntType sintType("sint" + std::to_string(width), width);

    // Calculate the range for this bit width
    int64_t maxVal = (width == 64) ? INT64_MAX : ((1LL << (width - 1)) - 1);
    int64_t minVal = (width == 64) ? INT64_MIN : (-(1LL << (width - 1)));

    // Test maximum positive value
    MessageData serializedMax(
        sintType.serialize(std::any(maxVal)).takeStorage());
    BitVector serializedMaxBits(serializedMax.getData());
    auto deserializedMax = sintType.deserialize(serializedMaxBits);

    // Cast to appropriate type based on width
    if (width <= 8) {
      auto resultMax = static_cast<int8_t>(std::any_cast<Int>(deserializedMax));
      EXPECT_EQ(resultMax, static_cast<int8_t>(maxVal))
          << "Failed for width " << width << " max value";
    } else if (width <= 16) {
      auto resultMax =
          static_cast<int16_t>(std::any_cast<Int>(deserializedMax));
      EXPECT_EQ(resultMax, static_cast<int16_t>(maxVal))
          << "Failed for width " << width << " max value";
    } else if (width <= 32) {
      auto resultMax =
          static_cast<int32_t>(std::any_cast<Int>(deserializedMax));
      EXPECT_EQ(resultMax, static_cast<int32_t>(maxVal))
          << "Failed for width " << width << " max value";
    } else {
      auto resultMax =
          static_cast<int64_t>(std::any_cast<Int>(deserializedMax));
      EXPECT_EQ(resultMax, maxVal)
          << "Failed for width " << width << " max value";
    }

    // Test maximum negative value
    MessageData serializedMin(
        sintType.serialize(std::any(minVal)).takeStorage());
    BitVector serializedMinBits(serializedMin.getData());
    auto deserializedMin = sintType.deserialize(serializedMinBits);

    if (width <= 8) {
      auto resultMin = static_cast<int8_t>(std::any_cast<Int>(deserializedMin));
      EXPECT_EQ(resultMin, static_cast<int8_t>(minVal))
          << "Failed for width " << width << " min value";
    } else if (width <= 16) {
      auto resultMin =
          static_cast<int16_t>(std::any_cast<Int>(deserializedMin));
      EXPECT_EQ(resultMin, static_cast<int16_t>(minVal))
          << "Failed for width " << width << " min value";
    } else if (width <= 32) {
      auto resultMin =
          static_cast<int32_t>(std::any_cast<Int>(deserializedMin));
      EXPECT_EQ(resultMin, static_cast<int32_t>(minVal))
          << "Failed for width " << width << " min value";
    } else {
      auto resultMin =
          static_cast<int64_t>(std::any_cast<Int>(deserializedMin));
      EXPECT_EQ(resultMin, minVal)
          << "Failed for width " << width << " min value";
    }

    // Test -1 (all bits set case)
    MessageData serializedMinusOne(
        sintType.serialize(std::any(static_cast<int64_t>(-1))).takeStorage());
    BitVector serializedMinusOneBits(serializedMinusOne.getData());
    auto deserializedMinusOne = sintType.deserialize(serializedMinusOneBits);

    if (width <= 8) {
      auto resultMinusOne =
          static_cast<int8_t>(std::any_cast<Int>(deserializedMinusOne));
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    } else if (width <= 16) {
      auto resultMinusOne =
          static_cast<int16_t>(std::any_cast<Int>(deserializedMinusOne));
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    } else if (width <= 32) {
      auto resultMinusOne =
          static_cast<int32_t>(std::any_cast<Int>(deserializedMinusOne));
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    } else {
      auto resultMinusOne =
          static_cast<int64_t>(std::any_cast<Int>(deserializedMinusOne));
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    }
  }
}

// Test wide UIntType serialization and deserialization (>64 bits)
TEST(ESITypesTest, WideUIntTypeSerialization) {
  // Test 128-bit unsigned integer
  UIntType uint128("uint128", 128);

  // Test a specific 128-bit value: 0x123456789ABCDEF0FEDCBA9876543210
  // This will be represented as a BitVector and serialized/deserialized
  uint64_t lowPart = 0xFEDCBA9876543210ULL;
  uint64_t highPart = 0x123456789ABCDEF0ULL;

  // Construct the 128-bit value by creating bytes in little-endian order
  std::vector<uint8_t> bytes(16, 0);
  for (size_t i = 0; i < 8; ++i) {
    bytes[i] = static_cast<uint8_t>((lowPart >> (8 * i)) & 0xFF);
    bytes[i + 8] = static_cast<uint8_t>((highPart >> (8 * i)) & 0xFF);
  }

  // Create a UInt from the bytes.
  std::any uint128Value = std::any(UInt(std::move(bytes)));

  // Test validation
  EXPECT_NO_THROW(uint128.ensureValid(uint128Value));

  // Test serialization
  MessageData serialized(uint128.serialize(uint128Value).takeStorage());
  EXPECT_EQ(serialized.getSize(), 16UL)
      << "UIntType(128) serialization should produce exactly 16 bytes";

  // Verify byte values in little-endian order
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(serialized.getData()[i],
              static_cast<uint8_t>((lowPart >> (8 * i)) & 0xFF))
        << "Low part byte " << i << " mismatch";
    EXPECT_EQ(serialized.getData()[i + 8],
              static_cast<uint8_t>((highPart >> (8 * i)) & 0xFF))
        << "High part byte " << i << " mismatch";
  }

  // Test deserialization
  BitVector serializedBits(serialized.getData());
  auto deserialized = uint128.deserialize(serializedBits);
  auto deserializedUInt = std::any_cast<UInt>(deserialized);
  EXPECT_EQ(deserializedUInt.width(), 128u)
      << "Deserialized UInt(128) should have width 128";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "UIntType(128) deserialization should consume all data";

  // Test 80-bit value (non-power-of-2)
  UIntType uint80("uint80", 80);
  std::vector<uint8_t> bytes80(10, 0);
  uint64_t val80 = 0x123456789ABCDEFULL;
  for (size_t i = 0; i < 10; ++i) {
    bytes80[i] = static_cast<uint8_t>((val80 >> (8 * i)) & 0xFF);
  }

  std::any uint80Value = std::any(UInt(std::move(bytes80), 80));

  EXPECT_NO_THROW(uint80.ensureValid(uint80Value));

  MessageData serialized80(uint80.serialize(uint80Value).takeStorage());
  EXPECT_EQ(serialized80.getSize(), 10UL)
      << "UIntType(80) serialization should produce exactly 10 bytes";

  BitVector serialized80Bits(serialized80.getData());
  auto deserialized80 = uint80.deserialize(serialized80Bits);
  auto deserializedUInt80 = std::any_cast<UInt>(deserialized80);
  EXPECT_EQ(deserializedUInt80.width(), 80u)
      << "Deserialized UInt(80) should have width 80";
  EXPECT_EQ(serialized80Bits.size(), 0UL)
      << "UIntType(80) deserialization should consume all data";
}

// Test wide SIntType serialization and deserialization (>64 bits)
TEST(ESITypesTest, WideSIntTypeSerialization) {
  // Test 128-bit signed integer with positive value
  SIntType sint128("sint128", 128);

  // Create a positive 128-bit value: 0x123456789ABCDEF0 (upper bits 0)
  std::vector<uint8_t> bytes(16, 0);
  uint64_t val = 0x123456789ABCDEF0ULL;
  for (size_t i = 0; i < 8; ++i) {
    bytes[i] = static_cast<uint8_t>((val >> (8 * i)) & 0xFF);
  }

  std::any sint128Value = std::any(Int(std::move(bytes), 128));

  // Test validation
  EXPECT_NO_THROW(sint128.ensureValid(sint128Value));

  // Test serialization
  MessageData serialized(sint128.serialize(sint128Value).takeStorage());
  EXPECT_EQ(serialized.getSize(), 16UL)
      << "SIntType(128) serialization should produce exactly 16 bytes";

  // Test deserialization
  BitVector serializedBits(serialized.getData());
  auto deserialized = sint128.deserialize(serializedBits);
  auto deserializedSInt = std::any_cast<Int>(deserialized);
  EXPECT_EQ(deserializedSInt.width(), 128u)
      << "Deserialized SInt(128) should have width 128";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "SIntType(128) deserialization should consume all data";

  // Test 128-bit signed integer with negative value: -1 (all bits set)
  std::vector<uint8_t> bytesNegOne(16, 0xFF);
  std::any sint128NegOne = std::any(Int(std::move(bytesNegOne), 128));

  EXPECT_NO_THROW(sint128.ensureValid(sint128NegOne));

  MessageData serializedNegOne(sint128.serialize(sint128NegOne).takeStorage());
  EXPECT_EQ(serializedNegOne.getSize(), 16UL)
      << "SIntType(128) serialization of -1 should produce 16 bytes";

  // Verify all bytes are 0xFF
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(serializedNegOne.getData()[i], 0xFF)
        << "All bytes in -1 should be 0xFF";
  }

  // Test deserialization of -1
  BitVector serializedNegOneBits(serializedNegOne.getData());
  auto deserializedNegOne = sint128.deserialize(serializedNegOneBits);
  auto deserializedSIntNegOne = std::any_cast<Int>(deserializedNegOne);
  EXPECT_EQ(deserializedSIntNegOne.width(), 128u)
      << "Deserialized SInt(128) of -1 should have width 128";

  // Test 192-bit signed integer (3 bytes x 8 bits = 24 bytes x 8 bits)
  SIntType sint192("sint192", 192);
  std::vector<uint8_t> bytes192(24, 0);
  // Set a pattern in the lower 16 bytes
  uint64_t val192Low = 0xDEADBEEFCAFEBABEULL;
  uint64_t val192Mid = 0x0123456789ABCDEFULL;
  for (size_t i = 0; i < 8; ++i) {
    bytes192[i] = static_cast<uint8_t>((val192Low >> (8 * i)) & 0xFF);
    bytes192[i + 8] = static_cast<uint8_t>((val192Mid >> (8 * i)) & 0xFF);
  }

  std::any sint192Value = std::any(Int(std::move(bytes192), 192));

  EXPECT_NO_THROW(sint192.ensureValid(sint192Value));

  MessageData serialized192(sint192.serialize(sint192Value).takeStorage());
  EXPECT_EQ(serialized192.getSize(), 24UL)
      << "SIntType(192) serialization should produce exactly 24 bytes";

  BitVector serialized192Bits(serialized192.getData());
  auto deserialized192 = sint192.deserialize(serialized192Bits);
  auto deserializedSInt192 = std::any_cast<Int>(deserialized192);
  EXPECT_EQ(deserializedSInt192.width(), 192u)
      << "Deserialized SInt(192) should have width 192";
  EXPECT_EQ(serialized192Bits.size(), 0UL)
      << "SIntType(192) deserialization should consume all data";

  // Test 72-bit signed integer (non-byte-aligned, non-power-of-2)
  SIntType sint72("sint72", 72);
  std::vector<uint8_t> bytes72(9, 0);
  uint64_t val72 = 0x0123456789ABCDEFULL;
  for (size_t i = 0; i < 8; ++i) {
    bytes72[i] = static_cast<uint8_t>((val72 >> (8 * i)) & 0xFF);
  }
  bytes72[8] = 0x01; // High byte with bit 72 representing sign bit position

  std::any sint72Value = std::any(Int(std::move(bytes72), 72));

  EXPECT_NO_THROW(sint72.ensureValid(sint72Value));

  MessageData serialized72(sint72.serialize(sint72Value).takeStorage());
  EXPECT_EQ(serialized72.getSize(), 9UL)
      << "SIntType(72) serialization should produce exactly 9 bytes";

  BitVector serialized72Bits(serialized72.getData());
  auto deserialized72 = sint72.deserialize(serialized72Bits);
  auto deserializedSInt72 = std::any_cast<Int>(deserialized72);
  EXPECT_EQ(deserializedSInt72.width(), 72u)
      << "Deserialized SInt(72) should have width 72";
  EXPECT_EQ(serialized72Bits.size(), 0UL)
      << "SIntType(72) deserialization should consume all data";
}

// Test StructType serialization and deserialization
TEST(ESITypesTest, StructTypeSerialization) {
  // Create field types
  auto uintType = std::make_unique<UIntType>("uint8", 8);
  auto sintType = std::make_unique<SIntType>("sint8", 8);

  // Create struct type with fields
  StructType::FieldVector fields = {{"field1", uintType.get()},
                                    {"field2", sintType.get()}};
  StructType structType("testStruct", fields);

  // Test valid struct value
  std::map<std::string, std::any> structValue = {
      {"field1", std::any(static_cast<uint64_t>(42))},
      {"field2", std::any(static_cast<int64_t>(-10))}};
  std::any validStruct = std::any(structValue);
  EXPECT_NO_THROW(structType.ensureValid(validStruct));

  // Test missing field
  std::map<std::string, std::any> incompleteStruct = {
      {"field1", std::any(static_cast<uint64_t>(42))}};
  std::any invalidStruct = std::any(incompleteStruct);
  EXPECT_THROW(structType.ensureValid(invalidStruct), std::runtime_error);

  // Test field not in type.
  std::map<std::string, std::any> incompatibleStruct = {
      {"UnknownField", std::any(static_cast<uint64_t>(0xdeadbeef))}};
  std::any incompatibleStructAny = std::any(incompatibleStruct);
  EXPECT_THROW(structType.ensureValid(incompatibleStructAny),
               std::runtime_error);

  // Test serialization
  MessageData serialized(structType.serialize(validStruct).takeStorage());
  EXPECT_EQ(serialized.getSize(), 2UL)
      << "StructType with uint8 + sint8 fields should serialize to 2 bytes";

  // Test deserialization
  BitVector serializedBits(serialized.getData());
  auto deserialized = structType.deserialize(serializedBits);
  auto deserializedStruct =
      std::any_cast<std::map<std::string, std::any>>(deserialized);
  EXPECT_EQ(deserializedStruct.size(), 2UL)
      << "Deserialized struct should contain exactly 2 fields";
  EXPECT_TRUE(deserializedStruct.find("field1") != deserializedStruct.end())
      << "Deserialized struct should contain field1";
  EXPECT_TRUE(deserializedStruct.find("field2") != deserializedStruct.end())
      << "Deserialized struct should contain field2";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "StructType deserialization should consume all data";

  // Verify field values
  auto field1Val =
      static_cast<uint8_t>(std::any_cast<UInt>(deserializedStruct["field1"]));
  auto field2Val =
      static_cast<int8_t>(std::any_cast<Int>(deserializedStruct["field2"]));
  EXPECT_EQ(field1Val, 42) << "Deserialized field1 should have value 42";
  EXPECT_EQ(field2Val, -10) << "Deserialized field2 should have value -10";

  // Test struct with non-byte aligned field. Should succeed.
  auto oddUintType = std::make_unique<UIntType>("uint6", 6);
  auto boolType = std::make_unique<BitsType>("bool", 1);
  StructType::FieldVector oddFields = {{"field1", uintType.get()},
                                       {"bool1", boolType.get()},
                                       {"field2", oddUintType.get()},
                                       {"bool2", boolType.get()}};
  StructType oddStruct("oddStruct", oddFields);

  std::map<std::string, std::any> oddStructValue = {
      {"field1", std::any(static_cast<uint64_t>(1))},
      {"bool1", std::any(std::vector<uint8_t>{1})},
      {"field2", std::any(static_cast<uint64_t>(2))},
      {"bool2", std::any(std::vector<uint8_t>{0})},
  };

  std::any validOddStruct = std::any(oddStructValue);
  EXPECT_NO_THROW(oddStruct.ensureValid(validOddStruct));
  MessageData oddSerialized(oddStruct.serialize(validOddStruct).takeStorage());
  // Expect 2 bytes (round up from 14 bits)
  EXPECT_EQ(oddSerialized.size(), 2UL);

  BitVector oddSerializedBits(oddSerialized.getData());
  auto oddDeserialized = oddStruct.deserialize(oddSerializedBits);
  auto deserializedOddStruct =
      std::any_cast<std::map<std::string, std::any>>(oddDeserialized);
  EXPECT_EQ(deserializedOddStruct.size(), 4UL)
      << "Deserialized odd struct should contain exactly 4 fields";
  EXPECT_EQ(oddSerializedBits.size(), 0UL)
      << "Odd StructType deserialization should consume all data";
  // Verify field values
  auto oddField1Val = static_cast<uint8_t>(
      std::any_cast<UInt>(deserializedOddStruct["field1"]));
  auto bool1Val =
      std::any_cast<std::vector<uint8_t>>(deserializedOddStruct["bool1"]);
  auto oddField2Val = static_cast<uint8_t>(
      std::any_cast<UInt>(deserializedOddStruct["field2"]));
  auto bool2Val =
      std::any_cast<std::vector<uint8_t>>(deserializedOddStruct["bool2"]);

  EXPECT_EQ(oddField1Val, 1) << "Deserialized odd field1 should have value 1";
  EXPECT_EQ(bool1Val.size(), 1ULL)
      << "Deserialized odd bool1 should have size 1";
  EXPECT_EQ(bool1Val[0], 1) << "Deserialized odd bool1 should have value true";
  EXPECT_EQ(oddField2Val, 2) << "Deserialized odd field2 should have value 2";
  EXPECT_EQ(bool2Val.size(), 1ULL)
      << "Deserialized odd bool2 should have size 1";
  EXPECT_EQ(bool2Val[0], 0) << "Deserialized odd bool2 should have value false";
}

// Test ArrayType serialization and deserialization
TEST(ESITypesTest, ArrayTypeSerialization) {
  // Create element type
  auto uintType = std::make_unique<UIntType>("uint8", 8);

  // Create array type
  ArrayType arrayType("uint8Array", uintType.get(), 3);

  // Test valid array value
  std::vector<std::any> arrayValue = {std::any(static_cast<uint64_t>(10)),
                                      std::any(static_cast<uint64_t>(20)),
                                      std::any(static_cast<uint64_t>(30))};
  std::any validArray = std::any(arrayValue);
  EXPECT_NO_THROW(arrayType.ensureValid(validArray));

  // Test wrong size array
  std::vector<std::any> wrongSizeArray = {std::any(static_cast<uint64_t>(10)),
                                          std::any(static_cast<uint64_t>(20))};
  std::any invalidArray = std::any(wrongSizeArray);
  EXPECT_THROW(arrayType.ensureValid(invalidArray), std::runtime_error);

  // Test serialization
  MessageData serialized(arrayType.serialize(validArray).takeStorage());
  EXPECT_EQ(serialized.getSize(), 3UL)
      << "ArrayType of 3 uint8 elements should serialize to 3 bytes";
  EXPECT_EQ(serialized.getData()[0], 30)
      << "First array element should serialize to 30 but got "
      << static_cast<uint32_t>(serialized.getData()[0]);
  EXPECT_EQ(serialized.getData()[1], 20)
      << "Second array element should serialize to 20 but got "
      << static_cast<uint32_t>(serialized.getData()[1]);
  EXPECT_EQ(serialized.getData()[2], 10)
      << "Third array element should serialize to 10 but got "
      << static_cast<uint32_t>(serialized.getData()[2]);

  // Test deserialization
  BitVector serializedBits(serialized.getData());
  auto deserialized = arrayType.deserialize(serializedBits);
  auto deserializedArray = std::any_cast<std::vector<std::any>>(deserialized);
  EXPECT_EQ(deserializedArray.size(), 3UL)
      << "Deserialized array should contain exactly 3 elements";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "ArrayType deserialization should consume all data";

  // Verify element values
  auto elem0 = static_cast<uint8_t>(std::any_cast<UInt>(deserializedArray[0]));
  auto elem1 = static_cast<uint8_t>(std::any_cast<UInt>(deserializedArray[1]));
  auto elem2 = static_cast<uint8_t>(std::any_cast<UInt>(deserializedArray[2]));
  EXPECT_EQ(elem0, 10) << "First array element should have value 10";
  EXPECT_EQ(elem1, 20) << "Second array element should have value 20";
  EXPECT_EQ(elem2, 30) << "Third array element should have value 30";
}

// Test bit width calculations
TEST(ESITypesTest, BitWidthCalculations) {
  VoidType voidType("void");
  EXPECT_EQ(voidType.getBitWidth(), 1) << "VoidType should have bit width of 1";

  BitsType bitsType("bits16", 16);
  EXPECT_EQ(bitsType.getBitWidth(), 16)
      << "BitsType(16) should have bit width of 16";

  UIntType uintType("uint32", 32);
  EXPECT_EQ(uintType.getBitWidth(), 32)
      << "UIntType(32) should have bit width of 32";

  SIntType sintType("sint64", 64);
  EXPECT_EQ(sintType.getBitWidth(), 64)
      << "SIntType(64) should have bit width of 64";

  // Test struct bit width
  auto uintType8 = std::make_unique<UIntType>("uint8", 8);
  auto sintType16 = std::make_unique<SIntType>("sint16", 16);
  StructType::FieldVector fields = {{"field1", uintType8.get()},
                                    {"field2", sintType16.get()}};
  StructType structType("testStruct", fields);
  EXPECT_EQ(structType.getBitWidth(), 24)
      << "StructType with uint8 + sint16 should have bit width of 24 (8 + "
         "16)";

  // Test array bit width
  ArrayType arrayType("uint8Array", uintType8.get(), 5);
  EXPECT_EQ(arrayType.getBitWidth(), 40)
      << "ArrayType of 5 uint8 elements should have bit width of 40 (8 * 5)";
}

//===----------------------------------------------------------------------===//
// Type::serializeType / Type::deserializeType round-trip tests
//===----------------------------------------------------------------------===//

class TypeSerializationTest : public ::testing::Test {
protected:
  Context ctx;
};

// Test VoidType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, VoidTypeRoundTrip) {
  VoidType voidType("void");

  std::string serialized = Type::serializeType(&voidType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedVoid = dynamic_cast<const VoidType *>(deserialized);
  ASSERT_NE(deserializedVoid, nullptr)
      << "Deserialized type should be VoidType";
  EXPECT_EQ(deserializedVoid->getID(), "void")
      << "Deserialized VoidType should have correct ID";
}

// Test BitsType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, BitsTypeRoundTrip) {
  BitsType bitsType("bits32", 32);

  std::string serialized = Type::serializeType(&bitsType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedBits = dynamic_cast<const BitsType *>(deserialized);
  ASSERT_NE(deserializedBits, nullptr)
      << "Deserialized type should be BitsType";
  EXPECT_EQ(deserializedBits->getWidth(), 32u)
      << "Deserialized BitsType should have width 32";
}

// Test UIntType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, UIntTypeRoundTrip) {
  UIntType uintType("u64", 64);

  std::string serialized = Type::serializeType(&uintType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedUInt = dynamic_cast<const UIntType *>(deserialized);
  ASSERT_NE(deserializedUInt, nullptr)
      << "Deserialized type should be UIntType";
  EXPECT_EQ(deserializedUInt->getWidth(), 64u)
      << "Deserialized UIntType should have width 64";
}

// Test SIntType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, SIntTypeRoundTrip) {
  SIntType sintType("s32", 32);

  std::string serialized = Type::serializeType(&sintType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedSInt = dynamic_cast<const SIntType *>(deserialized);
  ASSERT_NE(deserializedSInt, nullptr)
      << "Deserialized type should be SIntType";
  EXPECT_EQ(deserializedSInt->getWidth(), 32u)
      << "Deserialized SIntType should have width 32";
}

// Test AnyType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, AnyTypeRoundTrip) {
  AnyType anyType("any");

  std::string serialized = Type::serializeType(&anyType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedAny = dynamic_cast<const AnyType *>(deserialized);
  ASSERT_NE(deserializedAny, nullptr) << "Deserialized type should be AnyType";
  EXPECT_EQ(deserializedAny->getID(), "any")
      << "Deserialized AnyType should have correct ID";
}

// Test StructType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, StructTypeRoundTrip) {
  // Create field types and register them with context
  auto *uintType = new UIntType("u16", 16);
  auto *sintType = new SIntType("s8", 8);
  ctx.registerType(uintType);
  ctx.registerType(sintType);

  StructType::FieldVector fields = {{"field1", uintType}, {"field2", sintType}};
  StructType structType("myStruct", fields);

  std::string serialized = Type::serializeType(&structType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedStruct = dynamic_cast<const StructType *>(deserialized);
  ASSERT_NE(deserializedStruct, nullptr)
      << "Deserialized type should be StructType";
  EXPECT_EQ(deserializedStruct->getID(), "myStruct")
      << "Deserialized StructType should have correct ID";

  const auto &deserializedFields = deserializedStruct->getFields();
  ASSERT_EQ(deserializedFields.size(), 2u)
      << "Deserialized struct should have 2 fields";
  EXPECT_EQ(deserializedFields[0].first, "field1")
      << "First field name should be 'field1'";
  EXPECT_EQ(deserializedFields[1].first, "field2")
      << "Second field name should be 'field2'";

  // Check field types
  auto *field1Type =
      dynamic_cast<const UIntType *>(deserializedFields[0].second);
  ASSERT_NE(field1Type, nullptr) << "First field should be UIntType";
  EXPECT_EQ(field1Type->getWidth(), 16u)
      << "First field type should have width 16";

  auto *field2Type =
      dynamic_cast<const SIntType *>(deserializedFields[1].second);
  ASSERT_NE(field2Type, nullptr) << "Second field should be SIntType";
  EXPECT_EQ(field2Type->getWidth(), 8u)
      << "Second field type should have width 8";
}

// Test ArrayType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, ArrayTypeRoundTrip) {
  auto *elementType = new UIntType("u8", 8);
  ctx.registerType(elementType);

  ArrayType arrayType("myArray", elementType, 10);

  std::string serialized = Type::serializeType(&arrayType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedArray = dynamic_cast<const ArrayType *>(deserialized);
  ASSERT_NE(deserializedArray, nullptr)
      << "Deserialized type should be ArrayType";
  EXPECT_EQ(deserializedArray->getSize(), 10u)
      << "Deserialized ArrayType should have size 10";

  auto *deserializedElementType =
      dynamic_cast<const UIntType *>(deserializedArray->getElementType());
  ASSERT_NE(deserializedElementType, nullptr)
      << "Deserialized element type should be UIntType";
  EXPECT_EQ(deserializedElementType->getWidth(), 8u)
      << "Deserialized element type should have width 8";
}

// Test ChannelType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, ChannelTypeRoundTrip) {
  auto *innerType = new UIntType("u32", 32);
  ctx.registerType(innerType);

  ChannelType channelType("myChannel", innerType);

  std::string serialized = Type::serializeType(&channelType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedChannel = dynamic_cast<const ChannelType *>(deserialized);
  ASSERT_NE(deserializedChannel, nullptr)
      << "Deserialized type should be ChannelType";

  auto *deserializedInner =
      dynamic_cast<const UIntType *>(deserializedChannel->getInner());
  ASSERT_NE(deserializedInner, nullptr)
      << "Deserialized inner type should be UIntType";
  EXPECT_EQ(deserializedInner->getWidth(), 32u)
      << "Deserialized inner type should have width 32";
}

// Test BundleType serialization/deserialization round-trip
TEST_F(TypeSerializationTest, BundleTypeRoundTrip) {
  auto *toType = new UIntType("u16", 16);
  auto *fromType = new SIntType("s32", 32);
  ctx.registerType(toType);
  ctx.registerType(fromType);

  BundleType::ChannelVector channels = {
      {"input", BundleType::Direction::To, toType},
      {"output", BundleType::Direction::From, fromType}};
  BundleType bundleType("myBundle", channels);

  std::string serialized = Type::serializeType(&bundleType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedBundle = dynamic_cast<const BundleType *>(deserialized);
  ASSERT_NE(deserializedBundle, nullptr)
      << "Deserialized type should be BundleType";
  EXPECT_EQ(deserializedBundle->getID(), "myBundle")
      << "Deserialized BundleType should have correct ID";

  const auto &deserializedChannels = deserializedBundle->getChannels();
  ASSERT_EQ(deserializedChannels.size(), 2u)
      << "Deserialized bundle should have 2 channels";

  // Check first channel (input, To)
  EXPECT_EQ(std::get<0>(deserializedChannels[0]), "input")
      << "First channel name should be 'input'";
  EXPECT_EQ(std::get<1>(deserializedChannels[0]), BundleType::Direction::To)
      << "First channel direction should be To";

  // Check second channel (output, From)
  EXPECT_EQ(std::get<0>(deserializedChannels[1]), "output")
      << "Second channel name should be 'output'";
  EXPECT_EQ(std::get<1>(deserializedChannels[1]), BundleType::Direction::From)
      << "Second channel direction should be From";
}

// Test nested types (struct containing array)
TEST_F(TypeSerializationTest, NestedStructWithArrayRoundTrip) {
  auto *elementType = new UIntType("u8", 8);
  ctx.registerType(elementType);

  auto *arrayType = new ArrayType("u8Array", elementType, 4);
  ctx.registerType(arrayType);

  auto *sintType = new SIntType("s16", 16);
  ctx.registerType(sintType);

  StructType::FieldVector fields = {{"data", arrayType}, {"count", sintType}};
  StructType structType("nestedStruct", fields);

  std::string serialized = Type::serializeType(&structType);
  EXPECT_FALSE(serialized.empty()) << "Serialization should produce output";

  const Type *deserialized = Type::deserializeType(ctx, serialized);
  ASSERT_NE(deserialized, nullptr) << "Deserialization should succeed";

  auto *deserializedStruct = dynamic_cast<const StructType *>(deserialized);
  ASSERT_NE(deserializedStruct, nullptr)
      << "Deserialized type should be StructType";

  const auto &deserializedFields = deserializedStruct->getFields();
  ASSERT_EQ(deserializedFields.size(), 2u)
      << "Deserialized struct should have 2 fields";

  // Check array field
  auto *dataField =
      dynamic_cast<const ArrayType *>(deserializedFields[0].second);
  ASSERT_NE(dataField, nullptr) << "First field should be ArrayType";
  EXPECT_EQ(dataField->getSize(), 4u) << "Array should have size 4";

  auto *arrayElementType =
      dynamic_cast<const UIntType *>(dataField->getElementType());
  ASSERT_NE(arrayElementType, nullptr)
      << "Array element type should be UIntType";
  EXPECT_EQ(arrayElementType->getWidth(), 8u)
      << "Array element width should be 8";

  // Check sint field
  auto *countField =
      dynamic_cast<const SIntType *>(deserializedFields[1].second);
  ASSERT_NE(countField, nullptr) << "Second field should be SIntType";
  EXPECT_EQ(countField->getWidth(), 16u) << "Second field width should be 16";
}

// Test type caching - deserializing the same type twice should return same ptr
TEST_F(TypeSerializationTest, TypeCachingBehavior) {
  UIntType uintType("u32", 32);
  std::string serialized = Type::serializeType(&uintType);

  const Type *first = Type::deserializeType(ctx, serialized);
  const Type *second = Type::deserializeType(ctx, serialized);

  EXPECT_EQ(first, second)
      << "Deserializing same type twice should return same cached instance";
}

// Test error handling for null type
TEST_F(TypeSerializationTest, SerializeNullTypeThrows) {
  EXPECT_THROW(Type::serializeType(nullptr), std::runtime_error)
      << "Serializing null type should throw";
}

// Test error handling for invalid JSON
TEST_F(TypeSerializationTest, DeserializeInvalidJsonThrows) {
  EXPECT_THROW(Type::deserializeType(ctx, "not valid json"), std::exception)
      << "Deserializing invalid JSON should throw";
}

// Test error handling for unknown mnemonic
TEST_F(TypeSerializationTest, DeserializeUnknownMnemonicThrows) {
  std::string badJson = R"({"mnemonic": "unknown_type", "id": "bad"})";
  EXPECT_THROW(Type::deserializeType(ctx, badJson), std::runtime_error)
      << "Deserializing unknown mnemonic should throw";
}

// Test various bit widths for integer types
TEST_F(TypeSerializationTest, VariousBitWidths) {
  // Test small widths
  for (uint64_t width : {1, 4, 7, 8, 12, 16, 24, 32, 48, 64, 128}) {
    UIntType uintType("u" + std::to_string(width), width);
    std::string serialized = Type::serializeType(&uintType);

    // Create new context to avoid caching
    Context localCtx;
    const Type *deserialized = Type::deserializeType(localCtx, serialized);
    ASSERT_NE(deserialized, nullptr)
        << "Deserialization should succeed for width " << width;

    auto *deserializedUInt = dynamic_cast<const UIntType *>(deserialized);
    ASSERT_NE(deserializedUInt, nullptr)
        << "Deserialized type should be UIntType for width " << width;
    EXPECT_EQ(deserializedUInt->getWidth(), width)
        << "Deserialized width should match for width " << width;
  }
}

} // namespace
