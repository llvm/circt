//===- ESIRuntimeTypedPortsTest.cpp - Typed ESI port tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/TypedPorts.h"
#include "gtest/gtest.h"

using namespace esi;

namespace {

//===----------------------------------------------------------------------===//
// verifyTypeCompatibility tests
//===----------------------------------------------------------------------===//

TEST(TypedPortsTest, VoidTypeCompatibility) {
  VoidType voidType("void");
  EXPECT_NO_THROW(verifyTypeCompatibility<void>(&voidType));

  // Non-void types should fail.
  UIntType uint1("ui1", 1);
  EXPECT_THROW(verifyTypeCompatibility<void>(&uint1), std::runtime_error);

  SIntType sint32("si32", 32);
  EXPECT_THROW(verifyTypeCompatibility<void>(&sint32), std::runtime_error);
}

TEST(TypedPortsTest, BoolTypeCompatibility) {
  BitsType bits1("i1", 1);
  EXPECT_NO_THROW(verifyTypeCompatibility<bool>(&bits1));

  // Width > 1 should fail.
  BitsType bits8("i8", 8);
  EXPECT_THROW(verifyTypeCompatibility<bool>(&bits8), std::runtime_error);

  // Wrong type entirely should fail.
  SIntType sint1("si1", 1);
  EXPECT_THROW(verifyTypeCompatibility<bool>(&sint1), std::runtime_error);
}

TEST(TypedPortsTest, SignedIntTypeCompatibility) {
  // int32_t can hold si31 (width 31 < 32).
  SIntType sint31("si31", 31);
  EXPECT_NO_THROW(verifyTypeCompatibility<int32_t>(&sint31));

  // si32 has width 32, which fits exactly in int32_t. Should pass.
  SIntType sint32("si32", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<int32_t>(&sint32));

  // si33 has width 33, which exceeds int32_t. Should fail.
  SIntType sint33("si33", 33);
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(&sint33), std::runtime_error);

  // UIntType should fail for signed C++ type.
  UIntType uint31("ui31", 31);
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(&uint31), std::runtime_error);

  // int64_t can hold si63.
  SIntType sint63("si63", 63);
  EXPECT_NO_THROW(verifyTypeCompatibility<int64_t>(&sint63));

  // si64 fits exactly in int64_t. Should pass.
  SIntType sint64("si64", 64);
  EXPECT_NO_THROW(verifyTypeCompatibility<int64_t>(&sint64));

  // si65 exceeds int64_t. Should fail.
  SIntType sint65("si65", 65);
  EXPECT_THROW(verifyTypeCompatibility<int64_t>(&sint65), std::runtime_error);
}

TEST(TypedPortsTest, UnsignedIntTypeCompatibility) {
  // uint32_t can hold ui31 (width 31 < 32).
  UIntType uint31("ui31", 31);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&uint31));

  // ui32 has width 32, which fits exactly in uint32_t. Should pass.
  UIntType uint32_t_("ui32", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&uint32_t_));

  // ui33 exceeds uint32_t. Should fail.
  UIntType uint33("ui33", 33);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&uint33), std::runtime_error);

  // BitsType (signless iM) should also be accepted for unsigned.
  BitsType bits31("i31", 31);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&bits31));

  // BitsType with width 32 fits in uint32_t. Should pass.
  BitsType bits32("i32", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&bits32));

  // BitsType with width 33 exceeds uint32_t. Should fail.
  BitsType bits33("i33", 33);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&bits33), std::runtime_error);

  // uint64_t with ui63.
  UIntType uint63("ui63", 63);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint64_t>(&uint63));

  // uint64_t with ui64 fits exactly. Should pass.
  UIntType uint64_t_("ui64", 64);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint64_t>(&uint64_t_));

  // uint64_t with ui65 exceeds. Should fail.
  UIntType uint65("ui65", 65);
  EXPECT_THROW(verifyTypeCompatibility<uint64_t>(&uint65), std::runtime_error);

  // SIntType should fail for unsigned C++ type.
  SIntType sint31("si31", 31);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&sint31), std::runtime_error);
}

// Test struct with _ESI_ID.
struct TestStruct {
  static constexpr std::string_view _ESI_ID = "MyModule.TestStruct";
  uint32_t field1;
  uint16_t field2;
};

TEST(TypedPortsTest, ESIIDTypeCompatibility) {
  // Matching ID should pass.
  StructType matchType("MyModule.TestStruct", {});
  EXPECT_NO_THROW(verifyTypeCompatibility<TestStruct>(&matchType));

  // Mismatched ID should fail.
  StructType mismatchType("OtherModule.OtherStruct", {});
  EXPECT_THROW(verifyTypeCompatibility<TestStruct>(&mismatchType),
               std::runtime_error);

  // Even a non-struct type with matching ID should pass (ID comparison only).
  UIntType uintWithMatchingID("MyModule.TestStruct", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<TestStruct>(&uintWithMatchingID));
}

TEST(TypedPortsTest, NullPortTypeThrows) {
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(nullptr), std::runtime_error);
  EXPECT_THROW(verifyTypeCompatibility<void>(nullptr), std::runtime_error);
}

// A type that is not integral and has no _ESI_ID — should hit fallback.
struct UnknownCppType {
  double x;
};

TEST(TypedPortsTest, FallbackThrows) {
  UIntType uint32("ui32", 32);
  EXPECT_THROW(verifyTypeCompatibility<UnknownCppType>(&uint32),
               std::runtime_error);
}

//===----------------------------------------------------------------------===//
// TypedWritePort round-trip tests (verify MessageData encoding)
//===----------------------------------------------------------------------===//

// A minimal concrete WriteChannelPort for testing. Captures the last written
// MessageData instead of sending it anywhere.
class MockWritePort : public WriteChannelPort {
public:
  MockWritePort(const Type *type) : WriteChannelPort(type) {}

  void connect(const ConnectOptions &opts = {}) override {
    connectImpl(opts);
    connected = true;
  }
  void disconnect() override { connected = false; }
  bool isConnected() const override { return connected; }

  MessageData lastWritten;

protected:
  void writeImpl(const MessageData &data) override { lastWritten = data; }
  bool tryWriteImpl(const MessageData &data) override {
    lastWritten = data;
    return true;
  }

private:
  bool connected = false;
};

TEST(TypedPortsTest, TypedWritePortConnectThrowsOnMismatch) {
  UIntType uint32("ui32", 32);
  MockWritePort mock(&uint32);
  TypedWritePort<int32_t> typed(mock); // int32_t expects SIntType
  EXPECT_THROW(typed.connect(), std::runtime_error);
}

TEST(TypedPortsTest, TypedWritePortConnectSucceeds) {
  SIntType sint31("si31", 31);
  MockWritePort mock(&sint31);
  TypedWritePort<int32_t> typed(mock);
  EXPECT_NO_THROW(typed.connect());
  EXPECT_TRUE(typed.isConnected());
}

TEST(TypedPortsTest, TypedWritePortRoundTrip) {
  SIntType sint15("si15", 15);
  MockWritePort mock(&sint15);
  TypedWritePort<int32_t> typed(mock);
  typed.connect();

  int32_t val = 12345;
  typed.write(val);

  // Verify the written data matches the raw bytes of val.
  ASSERT_EQ(mock.lastWritten.getSize(), sizeof(int32_t));
  EXPECT_EQ(*mock.lastWritten.as<int32_t>(), 12345);
}

TEST(TypedPortsTest, TypedWritePortVoid) {
  VoidType voidType("void");
  MockWritePort mock(&voidType);
  TypedWritePort<void> typed(mock);
  EXPECT_NO_THROW(typed.connect());

  typed.write();
  ASSERT_EQ(mock.lastWritten.getSize(), 1u);
  EXPECT_EQ(mock.lastWritten.getData()[0], 0);
}

//===----------------------------------------------------------------------===//
// MockReadPort for TypedFunction testing
//===----------------------------------------------------------------------===//

// A minimal concrete ReadChannelPort that returns a preset response.
class MockReadPort : public ReadChannelPort {
public:
  MockReadPort(const Type *type) : ReadChannelPort(type) {}

  void connect(std::function<bool(MessageData)>,
               const ConnectOptions & = {}) override {
    mode = Mode::Callback;
  }
  void connect(const ConnectOptions & = {}) override { mode = Mode::Polling; }

  void read(MessageData &outData) override { outData = nextResponse; }
  std::future<MessageData> readAsync() override {
    std::promise<MessageData> p;
    p.set_value(nextResponse);
    return p.get_future();
  }

  MessageData nextResponse;
};

//===----------------------------------------------------------------------===//
// TypedFunction tests
//===----------------------------------------------------------------------===//

TEST(TypedPortsTest, TypedFunctionNullThrows) {
  EXPECT_THROW((TypedFunction<uint32_t, uint16_t>(nullptr)),
               std::runtime_error);
}

TEST(TypedPortsTest, TypedFunctionConnectVerifiesTypes) {
  // Create channel types matching si24 arg and ui16 result.
  SIntType argInner("si24", 24);
  ChannelType argChanType("channel<si24>", &argInner);
  UIntType resultInner("ui16", 15);
  ChannelType resultChanType("channel<ui16>", &resultInner);

  BundleType::ChannelVector channels = {
      {"arg", BundleType::Direction::To, &argChanType},
      {"result", BundleType::Direction::From, &resultChanType},
  };
  BundleType bundleType("func_bundle", channels);

  MockWritePort mockWrite(&argInner);
  MockReadPort mockRead(&resultInner);

  auto *func = services::FuncService::Function::get(AppID("test"), &bundleType,
                                                    mockWrite, mockRead);

  // int32_t arg (signed) against si24 — should pass.
  // uint16_t result (unsigned) against ui15 — should pass.
  TypedFunction<int32_t, uint16_t> typed(func);
  EXPECT_NO_THROW(typed.connect());
  delete func;
}

TEST(TypedPortsTest, TypedFunctionConnectRejectsArgMismatch) {
  UIntType argInner("ui24", 24);
  ChannelType argChanType("channel<ui24>", &argInner);
  UIntType resultInner("ui16", 15);
  ChannelType resultChanType("channel<ui16>", &resultInner);

  BundleType::ChannelVector channels = {
      {"arg", BundleType::Direction::To, &argChanType},
      {"result", BundleType::Direction::From, &resultChanType},
  };
  BundleType bundleType("func_bundle", channels);

  MockWritePort mockWrite(&argInner);
  MockReadPort mockRead(&resultInner);

  auto *func = services::FuncService::Function::get(AppID("test"), &bundleType,
                                                    mockWrite, mockRead);

  // int32_t (signed) against UIntType — should fail at connect.
  TypedFunction<int32_t, uint16_t> typed(func);
  EXPECT_THROW(typed.connect(), std::runtime_error);
  delete func;
}

TEST(TypedPortsTest, TypedFunctionCallRoundTrip) {
  SIntType argInner("si24", 24);
  ChannelType argChanType("channel<si24>", &argInner);
  UIntType resultInner("ui16", 15);
  ChannelType resultChanType("channel<ui16>", &resultInner);

  BundleType::ChannelVector channels = {
      {"arg", BundleType::Direction::To, &argChanType},
      {"result", BundleType::Direction::From, &resultChanType},
  };
  BundleType bundleType("func_bundle", channels);

  MockWritePort mockWrite(&argInner);
  MockReadPort mockRead(&resultInner);

  // Set up mock read to return a known uint16_t value.
  uint16_t expected = 42;
  mockRead.nextResponse = MessageData::from(expected);

  auto *func = services::FuncService::Function::get(AppID("test"), &bundleType,
                                                    mockWrite, mockRead);

  TypedFunction<int32_t, uint16_t> typed(func);
  typed.connect();

  int32_t arg = 100;
  uint16_t result = typed.call(arg).get();
  EXPECT_EQ(result, 42);

  // Verify the written arg matches.
  ASSERT_EQ(mockWrite.lastWritten.getSize(), sizeof(int32_t));
  EXPECT_EQ(*mockWrite.lastWritten.as<int32_t>(), 100);
  delete func;
}

} // namespace
