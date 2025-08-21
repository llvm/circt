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
  auto [valid, reason] = voidType.isValid(voidValue);
  EXPECT_TRUE(valid);
  EXPECT_TRUE(reason.empty());

  // Test valid void value (nullptr)
  std::any nullptrValue = std::any(nullptr);
  auto [valid2, reason2] = voidType.isValid(nullptrValue);
  EXPECT_TRUE(valid2);
  EXPECT_TRUE(reason2.empty());

  // Test invalid void value
  std::any invalidValue = std::any(42);
  auto [valid3, reason3] = voidType.isValid(invalidValue);
  EXPECT_FALSE(valid3);
  EXPECT_FALSE(reason3.empty());

  // Test serialization
  MessageData serialized = voidType.serialize(voidValue);
  EXPECT_EQ(serialized.getSize(), 1UL);
  EXPECT_EQ(serialized.getData()[0], 0);

  // Test deserialization
  auto [deserialized, remaining] = voidType.deserialize(serialized);
  EXPECT_FALSE(deserialized.has_value());
  EXPECT_EQ(remaining.getSize(), 0UL);
}

// Test BitsType serialization and deserialization
TEST(ESITypesTest, BitsTypeSerialization) {
  BitsType bitsType("bits8", 8);

  // Test valid bits value
  std::vector<uint8_t> bitsValue = {0xAB};
  std::any validBits = std::any(bitsValue);
  auto [valid, reason] = bitsType.isValid(validBits);
  EXPECT_TRUE(valid);
  EXPECT_TRUE(reason.empty());

  // Test invalid size
  std::vector<uint8_t> wrongSize = {0xAB, 0xCD};
  std::any invalidBits = std::any(wrongSize);
  auto [valid2, reason2] = bitsType.isValid(invalidBits);
  EXPECT_FALSE(valid2);
  EXPECT_FALSE(reason2.empty());

  // Test serialization
  MessageData serialized = bitsType.serialize(validBits);
  EXPECT_EQ(serialized.getSize(), 1UL);
  EXPECT_EQ(serialized.getData()[0], 0xAB);

  // Test deserialization
  auto [deserialized, remaining] = bitsType.deserialize(serialized);
  auto deserializedBits = std::any_cast<std::vector<uint8_t>>(deserialized);
  EXPECT_EQ(deserializedBits.size(), 1UL);
  EXPECT_EQ(deserializedBits[0], 0xAB);
  EXPECT_EQ(remaining.getSize(), 0UL);
}

// Test UIntType serialization and deserialization
TEST(ESITypesTest, UIntTypeSerialization) {
  UIntType uintType("uint16", 16);

  // Test valid uint value
  uint64_t uintValue = 0x1234;
  std::any validUInt = std::any(uintValue);
  auto [valid, reason] = uintType.isValid(validUInt);
  EXPECT_TRUE(valid);
  EXPECT_TRUE(reason.empty());

  // Test out of range value
  uint64_t outOfRange = 0x10000; // Too big for 16-bit
  std::any invalidUInt = std::any(outOfRange);
  auto [valid2, reason2] = uintType.isValid(invalidUInt);
  EXPECT_FALSE(valid2);
  EXPECT_FALSE(reason2.empty());

  // Test serialization (little-endian)
  MessageData serialized = uintType.serialize(validUInt);
  EXPECT_EQ(serialized.getSize(), 2UL);
  EXPECT_EQ(serialized.getData()[0], 0x34); // Little-endian
  EXPECT_EQ(serialized.getData()[1], 0x12);

  // Test deserialization
  auto [deserialized, remaining] = uintType.deserialize(serialized);
  auto deserializedUInt = std::any_cast<uint64_t>(deserialized);
  EXPECT_EQ(deserializedUInt, 0x1234UL);
  EXPECT_EQ(remaining.getSize(), 0UL);
}

// Test SIntType serialization and deserialization
TEST(ESITypesTest, SIntTypeSerialization) {
  SIntType sintType("sint16", 16);

  // Test valid positive sint value
  int64_t positiveValue = 0x1234;
  std::any validSInt = std::any(positiveValue);
  auto [valid, reason] = sintType.isValid(validSInt);
  EXPECT_TRUE(valid);
  EXPECT_TRUE(reason.empty());

  // Test valid negative sint value
  int64_t negativeValue = -1000;
  std::any validNegSInt = std::any(negativeValue);
  auto [valid2, reason2] = sintType.isValid(validNegSInt);
  EXPECT_TRUE(valid2);
  EXPECT_TRUE(reason2.empty());

  // Test serialization of positive value
  MessageData serialized = sintType.serialize(validSInt);
  EXPECT_EQ(serialized.getSize(), 2UL);
  EXPECT_EQ(serialized.getData()[0], 0x34); // Little-endian
  EXPECT_EQ(serialized.getData()[1], 0x12);

  // Test deserialization
  auto [deserialized, remaining] = sintType.deserialize(serialized);
  auto deserializedSInt = std::any_cast<int64_t>(deserialized);
  EXPECT_EQ(deserializedSInt, 0x1234);
  EXPECT_EQ(remaining.getSize(), 0UL);

  // Test negative value serialization/deserialization
  MessageData negSerialized = sintType.serialize(validNegSInt);
  auto [negDeserialized, negRemaining] = sintType.deserialize(negSerialized);
  auto deserializedNegSInt = std::any_cast<int64_t>(negDeserialized);
  EXPECT_EQ(deserializedNegSInt, -1000);
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
  auto [valid, reason] = structType.isValid(validStruct);
  EXPECT_TRUE(valid);
  EXPECT_TRUE(reason.empty());

  // Test missing field
  std::map<std::string, std::any> incompleteStruct = {
      {"field1", std::any(static_cast<uint64_t>(42))}};
  std::any invalidStruct = std::any(incompleteStruct);
  auto [valid2, reason2] = structType.isValid(invalidStruct);
  EXPECT_FALSE(valid2);
  EXPECT_FALSE(reason2.empty());

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
  EXPECT_EQ(remaining.getSize(), 0UL);

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
  auto [valid, reason] = arrayType.isValid(validArray);
  EXPECT_TRUE(valid);
  EXPECT_TRUE(reason.empty());

  // Test wrong size array
  std::vector<std::any> wrongSizeArray = {std::any(static_cast<uint64_t>(10)),
                                          std::any(static_cast<uint64_t>(20))};
  std::any invalidArray = std::any(wrongSizeArray);
  auto [valid2, reason2] = arrayType.isValid(invalidArray);
  EXPECT_FALSE(valid2);
  EXPECT_FALSE(reason2.empty());

  // Test serialization
  MessageData serialized = arrayType.serialize(validArray);
  EXPECT_EQ(serialized.getSize(), 3UL); // 1 byte per element

  // Test deserialization
  auto [deserialized, remaining] = arrayType.deserialize(serialized);
  auto deserializedArray = std::any_cast<std::vector<std::any>>(deserialized);
  EXPECT_EQ(deserializedArray.size(), 3UL);
  EXPECT_EQ(remaining.getSize(), 0UL);

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
  auto [valid, reason] = outerStructType.isValid(originalValue);
  EXPECT_TRUE(valid) << "Validation failed: " << reason;

  // Serialize
  MessageData serialized = outerStructType.serialize(originalValue);
  EXPECT_GT(serialized.getSize(), 0UL);

  // Deserialize
  auto [deserialized, remaining] = outerStructType.deserialize(serialized);
  EXPECT_EQ(remaining.getSize(), 0UL);

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
