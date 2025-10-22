//===- ESIRuntimeTest.cpp - ESI Runtime Type System Tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Types.h"
#include "esi/Values.h"
#include "gtest/gtest.h"
#include <any>
#include <cstdint>
#include <map>
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
  std::any validUInt = std::any(MutableUInt(uintValue, 16));
  EXPECT_NO_THROW(uintType.ensureValid(validUInt));

  // Test out of range value
  uint64_t outOfRange = 0x10000; // Too big for 16-bit
  std::any invalidUInt = std::any(MutableUInt(outOfRange, 16));
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
  auto deserializedUInt = std::any_cast<uint16_t>(deserialized);
  EXPECT_EQ(deserializedUInt, 0x1234)
      << "UIntType deserialization should reconstruct original value 0x1234";
  EXPECT_EQ(serializedBits.size(), 0UL)
      << "UIntType deserialization should consume all data";

  // Test that different input types work (uint8_t, uint16_t, uint32_t)
  uint8_t smallVal = 42;
  uint16_t mediumVal = 1000;
  uint32_t largeVal = 50000;

  EXPECT_NO_THROW(uintType.ensureValid(std::any(MutableUInt(smallVal, 16))));
  EXPECT_NO_THROW(uintType.ensureValid(std::any(MutableUInt(mediumVal, 16))));
  EXPECT_NO_THROW(uintType.ensureValid(std::any(MutableUInt(largeVal, 16))));
}

// Test SIntType serialization and deserialization
TEST(ESITypesTest, SIntTypeSerialization) {
  SIntType sintType("sint16", 16);

  // Test valid positive sint value
  int64_t positiveValue = 0x1234;
  std::any validSInt = std::any(MutableInt(positiveValue, 16));
  EXPECT_NO_THROW(sintType.ensureValid(validSInt));

  // Test valid negative sint value
  int64_t negativeValue = -1000;
  std::any validNegSInt = std::any(MutableInt(negativeValue, 16));
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
  auto deserializedSInt = std::any_cast<int16_t>(deserialized);
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
  auto deserializedNegSInt = std::any_cast<int16_t>(negDeserialized);
  EXPECT_EQ(deserializedNegSInt, -1000)
      << "SIntType deserialization should reconstruct original negative value "
         "(-1000)";

  // Test that different input types work (int8_t, int16_t, int32_t)
  int8_t smallVal = -42;
  int16_t mediumVal = -1000;
  int32_t largeVal = -30000;

  EXPECT_NO_THROW(sintType.ensureValid(std::any(MutableInt(smallVal, 16))));
  EXPECT_NO_THROW(sintType.ensureValid(std::any(MutableInt(mediumVal, 16))));
  EXPECT_NO_THROW(sintType.ensureValid(std::any(MutableInt(largeVal, 16))));
}

// Test SIntType sign extension logic comprehensively
TEST(ESITypesTest, SIntTypeSignExtension) {
  // Test 8-bit signed integers
  SIntType sint8("sint8", 8);

  // Test -1 (all bits set in 8-bit: 0xFF)
  int64_t minusOne = -1;
  MessageData serializedMinusOne(
      sint8.serialize(std::any(MutableInt(minusOne, 8))).takeStorage());
  EXPECT_EQ(serializedMinusOne.getSize(), 1UL)
      << "SIntType(8) serialization of -1 should produce 1 byte";
  EXPECT_EQ(serializedMinusOne.getData()[0], 0xFF)
      << "SIntType(8) serialization of -1 should be 0xFF (all bits set)";

  BitVector serializedMinusOneBits(serializedMinusOne.getData());
  auto deserializedMinusOne = sint8.deserialize(serializedMinusOneBits);
  auto resultMinusOne = std::any_cast<int8_t>(deserializedMinusOne);
  EXPECT_EQ(resultMinusOne, -1)
      << "SIntType(8) deserialization of 0xFF should reconstruct -1";

  // Test maximum negative value for 8-bit (-128 = 0x80)
  int64_t maxNeg8 = -128;
  MessageData serializedMaxNeg(
      sint8.serialize(std::any(MutableInt(maxNeg8, 8))).takeStorage());
  EXPECT_EQ(serializedMaxNeg.getSize(), 1UL)
      << "SIntType(8) serialization of -128 should produce 1 byte";
  EXPECT_EQ(serializedMaxNeg.getData()[0], 0x80)
      << "SIntType(8) serialization of -128 should be 0x80";

  BitVector serializedMaxNegBits(serializedMaxNeg.getData());
  auto deserializedMaxNeg = sint8.deserialize(serializedMaxNegBits);
  auto resultMaxNeg = std::any_cast<int8_t>(deserializedMaxNeg);
  EXPECT_EQ(resultMaxNeg, -128) << "SIntType(8) deserialization should "
                                   "reconstruct maximum negative value (-128)";

  // Test maximum positive value for 8-bit (127 = 0x7F)
  int64_t maxPos8 = 127;
  MessageData serializedMaxPos(
      sint8.serialize(std::any(MutableInt(maxPos8, 8))).takeStorage());
  EXPECT_EQ(serializedMaxPos.getSize(), 1UL)
      << "SIntType(8) serialization of 127 should produce 1 byte";
  EXPECT_EQ(serializedMaxPos.getData()[0], 0x7F)
      << "SIntType(8) serialization of 127 should be 0x7F";

  BitVector serializedMaxPosBits(serializedMaxPos.getData());
  auto deserializedMaxPos = sint8.deserialize(serializedMaxPosBits);
  auto resultMaxPos = std::any_cast<int8_t>(deserializedMaxPos);
  EXPECT_EQ(resultMaxPos, 127) << "SIntType(8) deserialization should "
                                  "reconstruct maximum positive value (127)";

  // Test 4-bit signed integers (edge case for smaller widths)
  SIntType sint4("sint4", 4);

  // Test -1 in 4-bit (0x0F in the lower nibble, should sign extend to all 1s)
  int64_t minus1w4bit = -1;
  MessageData serialized4bit(
      sint4.serialize(std::any(MutableInt(minus1w4bit, 4))).takeStorage());
  EXPECT_EQ(serialized4bit.getSize(), 1UL)
      << "SIntType(4) serialization should produce 1 byte";
  EXPECT_EQ(serialized4bit.getData()[0] & 0x0F, 0x0F)
      << "SIntType(4) serialization of -1 should have lower 4 bits set to 1111";

  BitVector serialized4bitBits(serialized4bit.getData());
  auto deserialized4bit = sint4.deserialize(serialized4bitBits);
  auto result4bit = std::any_cast<int8_t>(deserialized4bit);
  EXPECT_EQ(result4bit, -1)
      << "SIntType(4) deserialization should sign-extend -1 correctly";

  // Test maximum negative for 4-bit (-8 = 0x8 in 4 bits)
  int64_t maxNeg4 = -8;
  MessageData serializedMaxNeg4(
      sint4.serialize(std::any(MutableInt(maxNeg4, 4))).takeStorage());
  BitVector serializedMaxNeg4Bits(serializedMaxNeg4.getData());
  auto deserializedMaxNeg4 = sint4.deserialize(serializedMaxNeg4Bits);
  auto resultMaxNeg4 = std::any_cast<int8_t>(deserializedMaxNeg4);
  EXPECT_EQ(resultMaxNeg4, -8)
      << "SIntType(4) should handle maximum negative value (-8) correctly";

  // Test maximum positive for 4-bit (7 = 0x7 in 4 bits)
  int64_t maxPos4 = 7;
  MessageData serializedMaxPos4(
      sint4.serialize(std::any(MutableInt(maxPos4, 4))).takeStorage());
  BitVector serializedMaxPos4Bits(serializedMaxPos4.getData());
  auto deserializedMaxPos4 = sint4.deserialize(serializedMaxPos4Bits);
  auto resultMaxPos4 = std::any_cast<int8_t>(deserializedMaxPos4);
  EXPECT_EQ(resultMaxPos4, 7)
      << "SIntType(4) should handle maximum positive value (7) correctly";

  // Test 12-bit signed integers (non-byte-aligned case)
  SIntType sint12("sint12", 12);

  // Test -1 in 12-bit (should be 0xFFF in lower 12 bits)
  int64_t minus1w12bit = -1;
  MessageData serialized12bit(
      sint12.serialize(std::any(MutableInt(minus1w12bit, 12))).takeStorage());
  EXPECT_EQ(serialized12bit.getSize(), 2UL)
      << "SIntType(12) serialization should produce 2 bytes (12 bits = 2 "
         "bytes)";
  EXPECT_EQ(serialized12bit.getData()[0], 0xFF)
      << "SIntType(12) serialization of -1 should have lower byte 0xFF";
  EXPECT_EQ(serialized12bit.getData()[1] & 0x0F, 0x0F)
      << "SIntType(12) serialization of -1 should have upper nibble 0x0F";

  BitVector serialized12bitBits(serialized12bit.getData());
  auto deserialized12bit = sint12.deserialize(serialized12bitBits);
  auto result12bit = std::any_cast<int16_t>(deserialized12bit);
  EXPECT_EQ(result12bit, -1)
      << "SIntType(12) deserialization should sign-extend -1 correctly";

  // Test a value that requires sign extension: -100 in 12-bit
  int64_t neg100w12bit = -100;
  MessageData serializedNeg100(
      sint12.serialize(std::any(MutableInt(neg100w12bit, 12))).takeStorage());
  BitVector serializedNeg100Bits(serializedNeg100.getData());
  auto deserializedNeg100 = sint12.deserialize(serializedNeg100Bits);
  auto resultNeg100 = std::any_cast<int16_t>(deserializedNeg100);
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
        sintType.serialize(std::any(MutableInt(maxVal, width))).takeStorage());
    BitVector serializedMaxBits(serializedMax.getData());
    auto deserializedMax = sintType.deserialize(serializedMaxBits);

    // Cast to appropriate type based on width
    if (width <= 8) {
      auto resultMax = std::any_cast<int8_t>(deserializedMax);
      EXPECT_EQ(resultMax, static_cast<int8_t>(maxVal))
          << "Failed for width " << width << " max value";
    } else if (width <= 16) {
      auto resultMax = std::any_cast<int16_t>(deserializedMax);
      EXPECT_EQ(resultMax, static_cast<int16_t>(maxVal))
          << "Failed for width " << width << " max value";
    } else if (width <= 32) {
      auto resultMax = std::any_cast<int32_t>(deserializedMax);
      EXPECT_EQ(resultMax, static_cast<int32_t>(maxVal))
          << "Failed for width " << width << " max value";
    } else {
      auto resultMax = std::any_cast<int64_t>(deserializedMax);
      EXPECT_EQ(resultMax, maxVal)
          << "Failed for width " << width << " max value";
    }

    // Test maximum negative value
    MessageData serializedMin(
        sintType.serialize(std::any(MutableInt(minVal, width))).takeStorage());
    BitVector serializedMinBits(serializedMin.getData());
    auto deserializedMin = sintType.deserialize(serializedMinBits);

    if (width <= 8) {
      auto resultMin = std::any_cast<int8_t>(deserializedMin);
      EXPECT_EQ(resultMin, static_cast<int8_t>(minVal))
          << "Failed for width " << width << " min value";
    } else if (width <= 16) {
      auto resultMin = std::any_cast<int16_t>(deserializedMin);
      EXPECT_EQ(resultMin, static_cast<int16_t>(minVal))
          << "Failed for width " << width << " min value";
    } else if (width <= 32) {
      auto resultMin = std::any_cast<int32_t>(deserializedMin);
      EXPECT_EQ(resultMin, static_cast<int32_t>(minVal))
          << "Failed for width " << width << " min value";
    } else {
      auto resultMin = std::any_cast<int64_t>(deserializedMin);
      EXPECT_EQ(resultMin, minVal)
          << "Failed for width " << width << " min value";
    }

    // Test -1 (all bits set case)
    MessageData serializedMinusOne(
        sintType
            .serialize(std::any(MutableInt(static_cast<int64_t>(-1), width)))
            .takeStorage());
    BitVector serializedMinusOneBits(serializedMinusOne.getData());
    auto deserializedMinusOne = sintType.deserialize(serializedMinusOneBits);

    if (width <= 8) {
      auto resultMinusOne = std::any_cast<int8_t>(deserializedMinusOne);
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    } else if (width <= 16) {
      auto resultMinusOne = std::any_cast<int16_t>(deserializedMinusOne);
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    } else if (width <= 32) {
      auto resultMinusOne = std::any_cast<int32_t>(deserializedMinusOne);
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    } else {
      auto resultMinusOne = std::any_cast<int64_t>(deserializedMinusOne);
      EXPECT_EQ(resultMinusOne, -1)
          << "Failed for width " << width << " value -1";
    }
  }
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
  auto field1Val = std::any_cast<uint8_t>(deserializedStruct["field1"]);
  auto field2Val = std::any_cast<int8_t>(deserializedStruct["field2"]);
  EXPECT_EQ(field1Val, 42) << "Deserialized field1 should have value 42";
  EXPECT_EQ(field2Val, -10) << "Deserialized field2 should have value -10";

  // Test struct with non-byte aligned field. Should fail.
  auto oddUintType = std::make_unique<UIntType>("uint6", 6);
  StructType::FieldVector oddFields = {{"field1", uintType.get()},
                                       {"field2", oddUintType.get()}};
  StructType nonSerializeableStruct("oddStruct", oddFields);

  std::map<std::string, std::any> oddStructValue = {
      {"field1", std::any(static_cast<uint64_t>(1))},
      {"field2", std::any(static_cast<uint64_t>(2))}};

  std::any oddStructAny = std::any(oddStructValue);
  EXPECT_THROW(nonSerializeableStruct.ensureValid(oddStructAny),
               std::runtime_error);
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
  auto elem0 = std::any_cast<uint8_t>(deserializedArray[0]);
  auto elem1 = std::any_cast<uint8_t>(deserializedArray[1]);
  auto elem2 = std::any_cast<uint8_t>(deserializedArray[2]);
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
      << "StructType with uint8 + sint16 should have bit width of 24 (8 + 16)";

  // Test array bit width
  ArrayType arrayType("uint8Array", uintType8.get(), 5);
  EXPECT_EQ(arrayType.getBitWidth(), 40)
      << "ArrayType of 5 uint8 elements should have bit width of 40 (8 * 5)";
}
} // namespace