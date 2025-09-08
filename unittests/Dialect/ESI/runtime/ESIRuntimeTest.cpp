//===- ESIRuntimeTest.cpp - ESI Runtime Type System Tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Types.h"
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
  MessageData serialized = voidType.serialize(voidValue);
  EXPECT_EQ(serialized.getSize(), 1UL)
      << "VoidType serialization should produce exactly 1 byte";
  EXPECT_EQ(serialized.getData()[0], 0)
      << "VoidType serialization should produce a zero byte";

  // Test deserialization
  auto [deserialized, remaining] = voidType.deserialize(serialized);
  EXPECT_FALSE(deserialized.has_value())
      << "VoidType deserialization should not have a value";
  EXPECT_EQ(remaining.size(), 0UL)
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
  MessageData serialized = bitsType.serialize(validBits);
  EXPECT_EQ(serialized.getSize(), 1UL)
      << "BitsType(8) serialization should produce exactly 1 byte";
  EXPECT_EQ(serialized.getData()[0], 0xAB)
      << "BitsType serialization should preserve the input byte value";

  // Test deserialization
  auto [deserialized, remaining] = bitsType.deserialize(serialized);
  auto deserializedBits = std::any_cast<std::vector<uint8_t>>(deserialized);
  EXPECT_EQ(deserializedBits.size(), 1UL)
      << "BitsType(8) deserialization should produce 1 byte";
  EXPECT_EQ(deserializedBits[0], 0xAB)
      << "BitsType deserialization should preserve the original value";
  EXPECT_EQ(remaining.size(), 0UL)
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
  MessageData serialized = uintType.serialize(validUInt);
  EXPECT_EQ(serialized.getSize(), 2UL)
      << "UIntType(16) serialization should produce exactly 2 bytes";
  EXPECT_EQ(serialized.getData()[0], 0x34)
      << "UIntType serialization low byte should be 0x34 (little-endian)";
  EXPECT_EQ(serialized.getData()[1], 0x12)
      << "UIntType serialization high byte should be 0x12 (little-endian)";

  // Test deserialization
  auto [deserialized, remaining] = uintType.deserialize(serialized);
  auto deserializedUInt = std::any_cast<uint64_t>(deserialized);
  EXPECT_EQ(deserializedUInt, 0x1234UL)
      << "UIntType deserialization should reconstruct original value 0x1234";
  EXPECT_EQ(remaining.size(), 0UL)
      << "UIntType deserialization should consume all data";
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
  MessageData serialized = sintType.serialize(validSInt);
  EXPECT_EQ(serialized.getSize(), 2UL)
      << "SIntType(16) serialization should produce exactly 2 bytes";
  EXPECT_EQ(serialized.getData()[0], 0x34)
      << "SIntType serialization low byte should be 0x34 (little-endian)";
  EXPECT_EQ(serialized.getData()[1], 0x12)
      << "SIntType serialization high byte should be 0x12 (little-endian)";

  // Test deserialization
  auto [deserialized, remaining] = sintType.deserialize(serialized);
  auto deserializedSInt = std::any_cast<int64_t>(deserialized);
  EXPECT_EQ(deserializedSInt, 0x1234)
      << "SIntType deserialization should reconstruct original positive value";
  EXPECT_EQ(remaining.size(), 0UL)
      << "SIntType deserialization should consume all data";

  // Test negative value serialization/deserialization
  MessageData negSerialized = sintType.serialize(validNegSInt);
  EXPECT_EQ(negSerialized.getSize(), 2UL)
      << "SIntType(16) negative value should serialize to 2 bytes";
  // -1000 in 16-bit two's complement: 0xFC18
  EXPECT_EQ(negSerialized.getData()[0], 0x18)
      << "SIntType negative serialization low byte should be 0x18 "
         "(little-endian)";
  EXPECT_EQ(negSerialized.getData()[1], 0xFC)
      << "SIntType negative serialization high byte should be 0xFC "
         "(little-endian)";

  auto [negDeserialized, negRemaining] = sintType.deserialize(negSerialized);
  auto deserializedNegSInt = std::any_cast<int64_t>(negDeserialized);
  EXPECT_EQ(deserializedNegSInt, -1000)
      << "SIntType deserialization should reconstruct original negative value "
         "(-1000)";
}

// Test SIntType sign extension logic comprehensively
TEST(ESITypesTest, SIntTypeSignExtension) {
  // Test 8-bit signed integers
  SIntType sint8("sint8", 8);

  // Test -1 (all bits set in 8-bit: 0xFF)
  int64_t minusOne = -1;
  MessageData serializedMinusOne = sint8.serialize(std::any(minusOne));
  EXPECT_EQ(serializedMinusOne.getSize(), 1UL)
      << "SIntType(8) serialization of -1 should produce 1 byte";
  EXPECT_EQ(serializedMinusOne.getData()[0], 0xFF)
      << "SIntType(8) serialization of -1 should be 0xFF (all bits set)";

  auto [deserializedMinusOne, remaining1] =
      sint8.deserialize(serializedMinusOne);
  auto resultMinusOne = std::any_cast<int64_t>(deserializedMinusOne);
  EXPECT_EQ(resultMinusOne, -1)
      << "SIntType(8) deserialization of 0xFF should reconstruct -1";

  // Test maximum negative value for 8-bit (-128 = 0x80)
  int64_t maxNeg8 = -128;
  MessageData serializedMaxNeg = sint8.serialize(std::any(maxNeg8));
  EXPECT_EQ(serializedMaxNeg.getSize(), 1UL)
      << "SIntType(8) serialization of -128 should produce 1 byte";
  EXPECT_EQ(serializedMaxNeg.getData()[0], 0x80)
      << "SIntType(8) serialization of -128 should be 0x80";

  auto [deserializedMaxNeg, remaining2] = sint8.deserialize(serializedMaxNeg);
  auto resultMaxNeg = std::any_cast<int64_t>(deserializedMaxNeg);
  EXPECT_EQ(resultMaxNeg, -128) << "SIntType(8) deserialization should "
                                   "reconstruct maximum negative value (-128)";

  // Test maximum positive value for 8-bit (127 = 0x7F)
  int64_t maxPos8 = 127;
  MessageData serializedMaxPos = sint8.serialize(std::any(maxPos8));
  EXPECT_EQ(serializedMaxPos.getSize(), 1UL)
      << "SIntType(8) serialization of 127 should produce 1 byte";
  EXPECT_EQ(serializedMaxPos.getData()[0], 0x7F)
      << "SIntType(8) serialization of 127 should be 0x7F";

  auto [deserializedMaxPos, remaining3] = sint8.deserialize(serializedMaxPos);
  auto resultMaxPos = std::any_cast<int64_t>(deserializedMaxPos);
  EXPECT_EQ(resultMaxPos, 127) << "SIntType(8) deserialization should "
                                  "reconstruct maximum positive value (127)";

  // Test 4-bit signed integers (edge case for smaller widths)
  SIntType sint4("sint4", 4);

  // Test -1 in 4-bit (0x0F in the lower nibble, should sign extend to all 1s)
  int64_t minus1w4bit = -1;
  MessageData serialized4bit = sint4.serialize(std::any(minus1w4bit));
  EXPECT_EQ(serialized4bit.getSize(), 1UL)
      << "SIntType(4) serialization should produce 1 byte";
  EXPECT_EQ(serialized4bit.getData()[0] & 0x0F, 0x0F)
      << "SIntType(4) serialization of -1 should have lower 4 bits set to 1111";

  auto [deserialized4bit, remaining4] = sint4.deserialize(serialized4bit);
  auto result4bit = std::any_cast<int64_t>(deserialized4bit);
  EXPECT_EQ(result4bit, -1)
      << "SIntType(4) deserialization should sign-extend -1 correctly";

  // Test maximum negative for 4-bit (-8 = 0x8 in 4 bits)
  int64_t maxNeg4 = -8;
  MessageData serializedMaxNeg4 = sint4.serialize(std::any(maxNeg4));
  auto [deserializedMaxNeg4, remaining5] = sint4.deserialize(serializedMaxNeg4);
  auto resultMaxNeg4 = std::any_cast<int64_t>(deserializedMaxNeg4);
  EXPECT_EQ(resultMaxNeg4, -8)
      << "SIntType(4) should handle maximum negative value (-8) correctly";

  // Test maximum positive for 4-bit (7 = 0x7 in 4 bits)
  int64_t maxPos4 = 7;
  MessageData serializedMaxPos4 = sint4.serialize(std::any(maxPos4));
  auto [deserializedMaxPos4, remaining6] = sint4.deserialize(serializedMaxPos4);
  auto resultMaxPos4 = std::any_cast<int64_t>(deserializedMaxPos4);
  EXPECT_EQ(resultMaxPos4, 7)
      << "SIntType(4) should handle maximum positive value (7) correctly";

  // Test 12-bit signed integers (non-byte-aligned case)
  SIntType sint12("sint12", 12);

  // Test -1 in 12-bit (should be 0xFFF in lower 12 bits)
  int64_t minus1w12bit = -1;
  MessageData serialized12bit = sint12.serialize(std::any(minus1w12bit));
  EXPECT_EQ(serialized12bit.getSize(), 2UL)
      << "SIntType(12) serialization should produce 2 bytes (12 bits = 2 "
         "bytes)";
  EXPECT_EQ(serialized12bit.getData()[0], 0xFF)
      << "SIntType(12) serialization of -1 should have lower byte 0xFF";
  EXPECT_EQ(serialized12bit.getData()[1] & 0x0F, 0x0F)
      << "SIntType(12) serialization of -1 should have upper nibble 0x0F";

  auto [deserialized12bit, remaining7] = sint12.deserialize(serialized12bit);
  auto result12bit = std::any_cast<int64_t>(deserialized12bit);
  EXPECT_EQ(result12bit, -1)
      << "SIntType(12) deserialization should sign-extend -1 correctly";

  // Test a value that requires sign extension: -100 in 12-bit
  int64_t neg100w12bit = -100;
  MessageData serializedNeg100 = sint12.serialize(std::any(neg100w12bit));
  auto [deserializedNeg100, remaining8] = sint12.deserialize(serializedNeg100);
  auto resultNeg100 = std::any_cast<int64_t>(deserializedNeg100);
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
    MessageData serializedMax = sintType.serialize(std::any(maxVal));
    auto [deserializedMax, remainingMax] = sintType.deserialize(serializedMax);
    auto resultMax = std::any_cast<int64_t>(deserializedMax);
    EXPECT_EQ(resultMax, maxVal)
        << "Failed for width " << width << " max value";

    // Test maximum negative value
    MessageData serializedMin = sintType.serialize(std::any(minVal));
    auto [deserializedMin, remainingMin] = sintType.deserialize(serializedMin);
    auto resultMin = std::any_cast<int64_t>(deserializedMin);
    EXPECT_EQ(resultMin, minVal)
        << "Failed for width " << width << " min value";

    // Test -1 (all bits set case)
    MessageData serializedMinusOne =
        sintType.serialize(std::any(static_cast<int64_t>(-1)));
    auto [deserializedMinusOne, remainingMinusOne] =
        sintType.deserialize(serializedMinusOne);
    auto resultMinusOne = std::any_cast<int64_t>(deserializedMinusOne);
    EXPECT_EQ(resultMinusOne, -1)
        << "Failed for width " << width << " value -1";
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
  MessageData serialized = structType.serialize(validStruct);
  EXPECT_EQ(serialized.getSize(), 2UL)
      << "StructType with uint8 + sint8 fields should serialize to 2 bytes";

  // Test deserialization
  auto [deserialized, remaining] = structType.deserialize(serialized);
  auto deserializedStruct =
      std::any_cast<std::map<std::string, std::any>>(deserialized);
  EXPECT_EQ(deserializedStruct.size(), 2UL)
      << "Deserialized struct should contain exactly 2 fields";
  EXPECT_TRUE(deserializedStruct.find("field1") != deserializedStruct.end())
      << "Deserialized struct should contain field1";
  EXPECT_TRUE(deserializedStruct.find("field2") != deserializedStruct.end())
      << "Deserialized struct should contain field2";
  EXPECT_EQ(remaining.size(), 0UL)
      << "StructType deserialization should consume all data";

  // Verify field values
  auto field1Val = std::any_cast<uint64_t>(deserializedStruct["field1"]);
  auto field2Val = std::any_cast<int64_t>(deserializedStruct["field2"]);
  EXPECT_EQ(field1Val, 42UL) << "Deserialized field1 should have value 42";
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
  MessageData serialized = arrayType.serialize(validArray);
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
  auto [deserialized, remaining] = arrayType.deserialize(serialized);
  auto deserializedArray = std::any_cast<std::vector<std::any>>(deserialized);
  EXPECT_EQ(deserializedArray.size(), 3UL)
      << "Deserialized array should contain exactly 3 elements";
  EXPECT_EQ(remaining.size(), 0UL)
      << "ArrayType deserialization should consume all data";

  // Verify element values
  auto elem0 = std::any_cast<uint64_t>(deserializedArray[0]);
  auto elem1 = std::any_cast<uint64_t>(deserializedArray[1]);
  auto elem2 = std::any_cast<uint64_t>(deserializedArray[2]);
  EXPECT_EQ(elem0, 10UL) << "First array element should have value 10";
  EXPECT_EQ(elem1, 20UL) << "Second array element should have value 20";
  EXPECT_EQ(elem2, 30UL) << "Third array element should have value 30";
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

// Test UIntType with various integer types (getUIntLikeFromAny functionality)
TEST(ESITypesTest, UIntTypeExtendedSerialization) {
  UIntType uint16Type("uint16", 16);
  UIntType uint8Type("uint8", 8);

  // Test uint8_t input
  uint8_t uint8Value = 42;
  std::any uint8Any = std::any(uint8Value);
  EXPECT_NO_THROW(uint8Type.ensureValid(uint8Any));
  MessageData serialized8 = uint8Type.serialize(uint8Any);
  EXPECT_EQ(serialized8.getSize(), 1UL);
  EXPECT_EQ(serialized8.getData()[0], 42);

  // Test uint16_t input
  uint16_t uint16Value = 0x1234;
  std::any uint16Any = std::any(uint16Value);
  EXPECT_NO_THROW(uint16Type.ensureValid(uint16Any));
  MessageData serialized16 = uint16Type.serialize(uint16Any);
  EXPECT_EQ(serialized16.getSize(), 2UL);
  EXPECT_EQ(serialized16.getData()[0], 0x34); // little-endian
  EXPECT_EQ(serialized16.getData()[1], 0x12);

  // Test uint32_t input
  uint32_t uint32Value = 0x5678;
  std::any uint32Any = std::any(uint32Value);
  EXPECT_NO_THROW(uint16Type.ensureValid(uint32Any));
  MessageData serialized32 = uint16Type.serialize(uint32Any);
  EXPECT_EQ(serialized32.getSize(), 2UL);
  EXPECT_EQ(serialized32.getData()[0], 0x78); // little-endian
  EXPECT_EQ(serialized32.getData()[1], 0x56);

  // Test uint64_t input (original functionality)
  uint64_t uint64Value = 0x9ABC;
  std::any uint64Any = std::any(uint64Value);
  EXPECT_NO_THROW(uint16Type.ensureValid(uint64Any));
  MessageData serialized64 = uint16Type.serialize(uint64Any);
  EXPECT_EQ(serialized64.getSize(), 2UL);
  EXPECT_EQ(serialized64.getData()[0], 0xBC); // little-endian
  EXPECT_EQ(serialized64.getData()[1], 0x9A);

  // Test range validation with smaller types
  uint16_t outOfRange16 = 0xFFFF;
  std::any outOfRange16Any = std::any(outOfRange16);
  EXPECT_NO_THROW(uint16Type.ensureValid(
      outOfRange16Any)); // Should be valid for 16-bit type

  uint32_t outOfRange32 = 0x10000; // Too big for 16-bit
  std::any outOfRange32Any = std::any(outOfRange32);
  EXPECT_THROW(uint16Type.ensureValid(outOfRange32Any), std::runtime_error);

  // Test invalid type
  std::string invalidType = "not an integer";
  std::any invalidAny = std::any(invalidType);
  EXPECT_THROW(uint16Type.ensureValid(invalidAny), std::runtime_error);
}

// Test SIntType with various integer types (getIntLikeFromAny functionality)
TEST(ESITypesTest, SIntTypeExtendedSerialization) {
  SIntType sint16Type("sint16", 16);
  SIntType sint8Type("sint8", 8);

  // Test int8_t input
  int8_t int8Value = -42;
  std::any int8Any = std::any(int8Value);
  EXPECT_NO_THROW(sint8Type.ensureValid(int8Any));
  MessageData serialized8 = sint8Type.serialize(int8Any);
  EXPECT_EQ(serialized8.getSize(), 1UL);
  EXPECT_EQ(serialized8.getData()[0],
            static_cast<uint8_t>(-42)); // Two's complement

  // Test int16_t input
  int16_t int16Value = -1000;
  std::any int16Any = std::any(int16Value);
  EXPECT_NO_THROW(sint16Type.ensureValid(int16Any));
  MessageData serialized16 = sint16Type.serialize(int16Any);
  EXPECT_EQ(serialized16.getSize(), 2UL);
  // -1000 in 16-bit two's complement: 0xFC18
  EXPECT_EQ(serialized16.getData()[0], 0x18); // little-endian
  EXPECT_EQ(serialized16.getData()[1], 0xFC);

  // Test int32_t input (positive value)
  int32_t int32Value = 12345;
  std::any int32Any = std::any(int32Value);
  EXPECT_NO_THROW(sint16Type.ensureValid(int32Any));
  MessageData serialized32 = sint16Type.serialize(int32Any);
  EXPECT_EQ(serialized32.getSize(), 2UL);
  EXPECT_EQ(serialized32.getData()[0], 0x39); // 12345 = 0x3039, little-endian
  EXPECT_EQ(serialized32.getData()[1], 0x30);

  // Test int64_t input (original functionality)
  int64_t int64Value = -5000;
  std::any int64Any = std::any(int64Value);
  EXPECT_NO_THROW(sint16Type.ensureValid(int64Any));
  MessageData serialized64 = sint16Type.serialize(int64Any);
  EXPECT_EQ(serialized64.getSize(), 2UL);
  // -5000 in 16-bit two's complement: 0xEC78
  EXPECT_EQ(serialized64.getData()[0], 0x78); // little-endian
  EXPECT_EQ(serialized64.getData()[1], 0xEC);

  // Test range validation with smaller types
  int8_t validInt8 = 127;
  std::any validInt8Any = std::any(validInt8);
  EXPECT_NO_THROW(sint8Type.ensureValid(validInt8Any));

  int16_t validInt16 = 32767;
  std::any validInt16Any = std::any(validInt16);
  EXPECT_NO_THROW(sint16Type.ensureValid(validInt16Any));

  // Test out of range values
  int32_t outOfRange32 = 40000; // Too big for 16-bit signed
  std::any outOfRange32Any = std::any(outOfRange32);
  EXPECT_THROW(sint16Type.ensureValid(outOfRange32Any), std::runtime_error);

  int32_t outOfRangeNeg32 = -40000; // Too small for 16-bit signed
  std::any outOfRangeNeg32Any = std::any(outOfRangeNeg32);
  EXPECT_THROW(sint16Type.ensureValid(outOfRangeNeg32Any), std::runtime_error);

  // Test invalid type
  std::string invalidType = "not an integer";
  std::any invalidAny = std::any(invalidType);
  EXPECT_THROW(sint16Type.ensureValid(invalidAny), std::runtime_error);
}

} // namespace
