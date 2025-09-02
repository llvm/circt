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
  EXPECT_NO_THROW(voidType.isValid(voidValue));

  // Test valid void value (nullptr)
  std::any nullptrValue = std::any(nullptr);
  EXPECT_NO_THROW(voidType.isValid(nullptrValue));

  // Test invalid void value
  std::any invalidValue = std::any(42);
  EXPECT_THROW(voidType.ensureValid(invalidValue), std::runtime_error);

  // Test serialization
  MessageData serialized = voidType.serialize(voidValue);
  EXPECT_EQ(serialized.getSize(), 1UL);
  EXPECT_EQ(serialized.getData()[0], 0);

  // Test deserialization
  auto [deserialized, remaining] = voidType.deserialize(serialized);
  EXPECT_FALSE(deserialized.has_value());
  EXPECT_EQ(remaining.size(), 0UL);
}

// Test BitsType serialization and deserialization
TEST(ESITypesTest, BitsTypeSerialization) {
  BitsType bitsType("bits8", 8);

  // Test valid bits value
  std::vector<uint8_t> bitsValue = {0xAB};
  std::any validBits = std::any(bitsValue);
  EXPECT_NO_THROW(bitsType.isValid(validBits));

  // Test invalid size
  std::vector<uint8_t> wrongSize = {0xAB, 0xCD};
  std::any invalidBits = std::any(wrongSize);
  EXPECT_THROW(bitsType.ensureValid(invalidBits), std::runtime_error);

  // Test serialization
  MessageData serialized = bitsType.serialize(validBits);
  EXPECT_EQ(serialized.getSize(), 1UL);
  EXPECT_EQ(serialized.getData()[0], 0xAB);

  // Test deserialization
  auto [deserialized, remaining] = bitsType.deserialize(serialized);
  auto deserializedBits = std::any_cast<std::vector<uint8_t>>(deserialized);
  EXPECT_EQ(deserializedBits.size(), 1UL);
  EXPECT_EQ(deserializedBits[0], 0xAB);
  EXPECT_EQ(remaining.size(), 0UL);
}

// Test UIntType serialization and deserialization
TEST(ESITypesTest, UIntTypeSerialization) {
  UIntType uintType("uint16", 16);

  // Test valid uint value
  uint64_t uintValue = 0x1234;
  std::any validUInt = std::any(uintValue);
  EXPECT_NO_THROW(uintType.isValid(validUInt));

  // Test out of range value
  uint64_t outOfRange = 0x10000; // Too big for 16-bit
  std::any invalidUInt = std::any(outOfRange);
  EXPECT_THROW(uintType.ensureValid(invalidUInt), std::runtime_error);

  // Test serialization (little-endian)
  MessageData serialized = uintType.serialize(validUInt);
  EXPECT_EQ(serialized.getSize(), 2UL);
  EXPECT_EQ(serialized.getData()[0], 0x34); // Little-endian
  EXPECT_EQ(serialized.getData()[1], 0x12);

  // Test deserialization
  auto [deserialized, remaining] = uintType.deserialize(serialized);
  auto deserializedUInt = std::any_cast<uint64_t>(deserialized);
  EXPECT_EQ(deserializedUInt, 0x1234UL);
  EXPECT_EQ(remaining.size(), 0UL);
}

// Test SIntType serialization and deserialization
TEST(ESITypesTest, SIntTypeSerialization) {
  SIntType sintType("sint16", 16);

  // Test valid positive sint value
  int64_t positiveValue = 0x1234;
  std::any validSInt = std::any(positiveValue);
  EXPECT_NO_THROW(sintType.isValid(validSInt));

  // Test valid negative sint value
  int64_t negativeValue = -1000;
  std::any validNegSInt = std::any(negativeValue);
  EXPECT_NO_THROW(sintType.isValid(validNegSInt));

  // Test serialization of positive value
  MessageData serialized = sintType.serialize(validSInt);
  EXPECT_EQ(serialized.getSize(), 2UL);
  EXPECT_EQ(serialized.getData()[0], 0x34); // Little-endian
  EXPECT_EQ(serialized.getData()[1], 0x12);

  // Test deserialization
  auto [deserialized, remaining] = sintType.deserialize(serialized);
  auto deserializedSInt = std::any_cast<int64_t>(deserialized);
  EXPECT_EQ(deserializedSInt, 0x1234);
  EXPECT_EQ(remaining.size(), 0UL);

  // Test negative value serialization/deserialization
  MessageData negSerialized = sintType.serialize(validNegSInt);
  auto [negDeserialized, negRemaining] = sintType.deserialize(negSerialized);
  auto deserializedNegSInt = std::any_cast<int64_t>(negDeserialized);
  EXPECT_EQ(deserializedNegSInt, -1000);
}

// Test SIntType sign extension logic comprehensively
TEST(ESITypesTest, SIntTypeSignExtension) {
  // Test 8-bit signed integers
  SIntType sint8("sint8", 8);

  // Test -1 (all bits set in 8-bit: 0xFF)
  int64_t minusOne = -1;
  MessageData serializedMinusOne = sint8.serialize(std::any(minusOne));
  EXPECT_EQ(serializedMinusOne.getSize(), 1UL);
  EXPECT_EQ(serializedMinusOne.getData()[0], 0xFF);

  auto [deserializedMinusOne, remaining1] =
      sint8.deserialize(serializedMinusOne);
  auto resultMinusOne = std::any_cast<int64_t>(deserializedMinusOne);
  EXPECT_EQ(resultMinusOne, -1);

  // Test maximum negative value for 8-bit (-128 = 0x80)
  int64_t maxNeg8 = -128;
  MessageData serializedMaxNeg = sint8.serialize(std::any(maxNeg8));
  EXPECT_EQ(serializedMaxNeg.getSize(), 1UL);
  EXPECT_EQ(serializedMaxNeg.getData()[0], 0x80);

  auto [deserializedMaxNeg, remaining2] = sint8.deserialize(serializedMaxNeg);
  auto resultMaxNeg = std::any_cast<int64_t>(deserializedMaxNeg);
  EXPECT_EQ(resultMaxNeg, -128);

  // Test maximum positive value for 8-bit (127 = 0x7F)
  int64_t maxPos8 = 127;
  MessageData serializedMaxPos = sint8.serialize(std::any(maxPos8));
  EXPECT_EQ(serializedMaxPos.getSize(), 1UL);
  EXPECT_EQ(serializedMaxPos.getData()[0], 0x7F);

  auto [deserializedMaxPos, remaining3] = sint8.deserialize(serializedMaxPos);
  auto resultMaxPos = std::any_cast<int64_t>(deserializedMaxPos);
  EXPECT_EQ(resultMaxPos, 127);

  // Test 4-bit signed integers (edge case for smaller widths)
  SIntType sint4("sint4", 4);

  // Test -1 in 4-bit (0x0F in the lower nibble, should sign extend to all 1s)
  int64_t minus1w4bit = -1;
  MessageData serialized4bit = sint4.serialize(std::any(minus1w4bit));
  EXPECT_EQ(serialized4bit.getSize(), 1UL);
  EXPECT_EQ(serialized4bit.getData()[0] & 0x0F,
            0x0F); // Lower 4 bits should be 1111

  auto [deserialized4bit, remaining4] = sint4.deserialize(serialized4bit);
  auto result4bit = std::any_cast<int64_t>(deserialized4bit);
  EXPECT_EQ(result4bit, -1);

  // Test maximum negative for 4-bit (-8 = 0x8 in 4 bits)
  int64_t maxNeg4 = -8;
  MessageData serializedMaxNeg4 = sint4.serialize(std::any(maxNeg4));
  auto [deserializedMaxNeg4, remaining5] = sint4.deserialize(serializedMaxNeg4);
  auto resultMaxNeg4 = std::any_cast<int64_t>(deserializedMaxNeg4);
  EXPECT_EQ(resultMaxNeg4, -8);

  // Test maximum positive for 4-bit (7 = 0x7 in 4 bits)
  int64_t maxPos4 = 7;
  MessageData serializedMaxPos4 = sint4.serialize(std::any(maxPos4));
  auto [deserializedMaxPos4, remaining6] = sint4.deserialize(serializedMaxPos4);
  auto resultMaxPos4 = std::any_cast<int64_t>(deserializedMaxPos4);
  EXPECT_EQ(resultMaxPos4, 7);

  // Test 12-bit signed integers (non-byte-aligned case)
  SIntType sint12("sint12", 12);

  // Test -1 in 12-bit (should be 0xFFF in lower 12 bits)
  int64_t minus1w12bit = -1;
  MessageData serialized12bit = sint12.serialize(std::any(minus1w12bit));
  EXPECT_EQ(serialized12bit.getSize(), 2UL); // 12 bits = 2 bytes

  auto [deserialized12bit, remaining7] = sint12.deserialize(serialized12bit);
  auto result12bit = std::any_cast<int64_t>(deserialized12bit);
  EXPECT_EQ(result12bit, -1);

  // Test a value that requires sign extension: -100 in 12-bit
  int64_t neg100w12bit = -100;
  MessageData serializedNeg100 = sint12.serialize(std::any(neg100w12bit));
  auto [deserializedNeg100, remaining8] = sint12.deserialize(serializedNeg100);
  auto resultNeg100 = std::any_cast<int64_t>(deserializedNeg100);
  EXPECT_EQ(resultNeg100, -100);
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
    std::span<const uint8_t> spanMax(serializedMax.getData().data(),
                                     serializedMax.getSize());
    auto [deserializedMax, remainingMax] = sintType.deserialize(spanMax);
    auto resultMax = std::any_cast<int64_t>(deserializedMax);
    EXPECT_EQ(resultMax, maxVal)
        << "Failed for width " << width << " max value";

    // Test maximum negative value
    MessageData serializedMin = sintType.serialize(std::any(minVal));
    std::span<const uint8_t> spanMin(serializedMin.getData().data(),
                                     serializedMin.getSize());
    auto [deserializedMin, remainingMin] = sintType.deserialize(spanMin);
    auto resultMin = std::any_cast<int64_t>(deserializedMin);
    EXPECT_EQ(resultMin, minVal)
        << "Failed for width " << width << " min value";

    // Test -1 (all bits set case)
    MessageData serializedMinusOne =
        sintType.serialize(std::any(static_cast<int64_t>(-1)));
    std::span<const uint8_t> spanMinusOne(serializedMinusOne.getData().data(),
                                          serializedMinusOne.getSize());
    auto [deserializedMinusOne, remainingMinusOne] =
        sintType.deserialize(spanMinusOne);
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
  EXPECT_NO_THROW(structType.isValid(validStruct));

  // Test missing field
  std::map<std::string, std::any> incompleteStruct = {
      {"field1", std::any(static_cast<uint64_t>(42))}};
  std::any invalidStruct = std::any(incompleteStruct);
  EXPECT_THROW(structType.ensureValid(invalidStruct), std::runtime_error);

  // Test serialization
  MessageData serialized = structType.serialize(validStruct);
  EXPECT_EQ(serialized.getSize(), 2UL); // 1 byte for each field

  // Test deserialization
  auto [deserialized, remaining] = structType.deserialize(serialized);
  auto deserializedStruct =
      std::any_cast<std::map<std::string, std::any>>(deserialized);
  EXPECT_EQ(deserializedStruct.size(), 2UL);
  EXPECT_TRUE(deserializedStruct.find("field1") != deserializedStruct.end());
  EXPECT_TRUE(deserializedStruct.find("field2") != deserializedStruct.end());
  EXPECT_EQ(remaining.size(), 0UL);

  // Verify field values
  auto field1Val = std::any_cast<uint64_t>(deserializedStruct["field1"]);
  auto field2Val = std::any_cast<int64_t>(deserializedStruct["field2"]);
  EXPECT_EQ(field1Val, 42UL);
  EXPECT_EQ(field2Val, -10);
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
  EXPECT_NO_THROW(arrayType.isValid(validArray));

  // Test wrong size array
  std::vector<std::any> wrongSizeArray = {std::any(static_cast<uint64_t>(10)),
                                          std::any(static_cast<uint64_t>(20))};
  std::any invalidArray = std::any(wrongSizeArray);
  EXPECT_THROW(arrayType.ensureValid(invalidArray), std::runtime_error);

  // Test serialization
  MessageData serialized = arrayType.serialize(validArray);
  EXPECT_EQ(serialized.getSize(), 3UL); // 1 byte per element

  // Test deserialization
  auto [deserialized, remaining] = arrayType.deserialize(serialized);
  auto deserializedArray = std::any_cast<std::vector<std::any>>(deserialized);
  EXPECT_EQ(deserializedArray.size(), 3UL);
  EXPECT_EQ(remaining.size(), 0UL);

  // Verify element values
  auto elem0 = std::any_cast<uint64_t>(deserializedArray[0]);
  auto elem1 = std::any_cast<uint64_t>(deserializedArray[1]);
  auto elem2 = std::any_cast<uint64_t>(deserializedArray[2]);
  EXPECT_EQ(elem0, 10UL);
  EXPECT_EQ(elem1, 20UL);
  EXPECT_EQ(elem2, 30UL);
}

// Test bit width calculations
TEST(ESITypesTest, BitWidthCalculations) {
  VoidType voidType("void");
  EXPECT_EQ(voidType.getBitWidth(), 1);

  BitsType bitsType("bits16", 16);
  EXPECT_EQ(bitsType.getBitWidth(), 16);

  UIntType uintType("uint32", 32);

  EXPECT_EQ(uintType.getBitWidth(), 32);

  SIntType sintType("sint64", 64);
  EXPECT_EQ(sintType.getBitWidth(), 64);

  // Test struct bit width
  auto uintType8 = std::make_unique<UIntType>("uint8", 8);
  auto sintType16 = std::make_unique<SIntType>("sint16", 16);
  StructType::FieldVector fields = {{"field1", uintType8.get()},
                                    {"field2", sintType16.get()}};
  StructType structType("testStruct", fields);
  EXPECT_EQ(structType.getBitWidth(), 24); // 8 + 16

  // Test array bit width
  ArrayType arrayType("uint8Array", uintType8.get(), 5);
  EXPECT_EQ(arrayType.getBitWidth(), 40); // 8 * 5
}

// Test round-trip serialization/deserialization
TEST(ESITypesTest, RoundTripSerialization) {
  // Test with a complex nested structure
  auto uintType = std::make_unique<UIntType>("uint16", 16);
  auto sintType = std::make_unique<SIntType>("sint8", 8);

  // Create inner struct
  StructType::FieldVector innerFields = {{"inner_uint", uintType.get()},
                                         {"inner_sint", sintType.get()}};
  auto innerStructType =
      std::make_unique<StructType>("innerStruct", innerFields);

  // Create array of inner structs
  auto arrayType =
      std::make_unique<ArrayType>("structArray", innerStructType.get(), 2);

  // Create outer struct
  StructType::FieldVector outerFields = {{"array_field", arrayType.get()},
                                         {"simple_field", uintType.get()}};
  StructType outerStructType("outerStruct", outerFields);

  // Create test data
  std::map<std::string, std::any> innerStruct1 = {
      {"inner_uint", std::any(static_cast<uint64_t>(0x1234))},
      {"inner_sint", std::any(static_cast<int64_t>(-50))}};
  std::map<std::string, std::any> innerStruct2 = {
      {"inner_uint", std::any(static_cast<uint64_t>(0x5678))},
      {"inner_sint", std::any(static_cast<int64_t>(100))}};
  std::vector<std::any> arrayValue = {std::any(innerStruct1),
                                      std::any(innerStruct2)};
  std::map<std::string, std::any> outerStruct = {
      {"array_field", std::any(arrayValue)},
      {"simple_field", std::any(static_cast<uint64_t>(0xABCD))}};

  std::any originalValue = std::any(outerStruct);

  // Test validation
  EXPECT_NO_THROW(outerStructType.isValid(originalValue));

  // Serialize
  MessageData serialized = outerStructType.serialize(originalValue);
  EXPECT_GT(serialized.getSize(), 0UL);

  // Deserialize
  auto [deserialized, remaining] = outerStructType.deserialize(serialized);
  EXPECT_EQ(remaining.size(), 0UL);

  // Verify the deserialized data matches the original
  auto deserializedOuter =
      std::any_cast<std::map<std::string, std::any>>(deserialized);
  EXPECT_EQ(deserializedOuter.size(), 2UL);

  // Check simple field
  auto simpleField = std::any_cast<uint64_t>(deserializedOuter["simple_field"]);
  EXPECT_EQ(simpleField, 0xABCDUL);

  // Check array field
  auto arrayField =
      std::any_cast<std::vector<std::any>>(deserializedOuter["array_field"]);
  EXPECT_EQ(arrayField.size(), 2UL);

  // Check first array element
  auto firstElement =
      std::any_cast<std::map<std::string, std::any>>(arrayField[0]);
  auto innerUint1 = std::any_cast<uint64_t>(firstElement["inner_uint"]);
  auto innerSint1 = std::any_cast<int64_t>(firstElement["inner_sint"]);
  EXPECT_EQ(innerUint1, 0x1234UL);
  EXPECT_EQ(innerSint1, -50);

  // Check second array element
  auto secondElement =
      std::any_cast<std::map<std::string, std::any>>(arrayField[1]);
  auto innerUint2 = std::any_cast<uint64_t>(secondElement["inner_uint"]);
  auto innerSint2 = std::any_cast<int64_t>(secondElement["inner_sint"]);
  EXPECT_EQ(innerUint2, 0x5678UL);
  EXPECT_EQ(innerSint2, 100);
}

} // namespace
