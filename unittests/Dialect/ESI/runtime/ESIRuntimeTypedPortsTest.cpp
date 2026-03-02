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
  EXPECT_THROW(verifyTypeCompatibility<void>(&uint1), AcceleratorMismatchError);

  SIntType sint32("si32", 32);
  EXPECT_THROW(verifyTypeCompatibility<void>(&sint32),
               AcceleratorMismatchError);
}

TEST(TypedPortsTest, BoolTypeCompatibility) {
  BitsType bits1("i1", 1);
  EXPECT_NO_THROW(verifyTypeCompatibility<bool>(&bits1));

  // Width > 1 should fail.
  BitsType bits8("i8", 8);
  EXPECT_THROW(verifyTypeCompatibility<bool>(&bits8), AcceleratorMismatchError);

  // Wrong type entirely should fail.
  SIntType sint1("si1", 1);
  EXPECT_THROW(verifyTypeCompatibility<bool>(&sint1), AcceleratorMismatchError);
}

TEST(TypedPortsTest, SignedIntTypeCompatibility) {
  // int32_t can hold si17 (width 17, in range (16,32]).
  SIntType sint17("si17", 17);
  EXPECT_NO_THROW(verifyTypeCompatibility<int32_t>(&sint17));

  // si32 has width 32, which fits exactly in int32_t. Should pass.
  SIntType sint32("si32", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<int32_t>(&sint32));

  // si33 has width 33, which exceeds int32_t. Should fail.
  SIntType sint33("si33", 33);
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(&sint33),
               AcceleratorMismatchError);

  // si16 fits in int32_t but a smaller type (int16_t) would suffice. Reject.
  SIntType sint16("si16", 16);
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(&sint16),
               AcceleratorMismatchError);

  // si8 is even smaller — also reject for int32_t.
  SIntType sint8("si8", 8);
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(&sint8),
               AcceleratorMismatchError);

  // But si8 should be fine for int8_t (closest match).
  EXPECT_NO_THROW(verifyTypeCompatibility<int8_t>(&sint8));

  // UIntType should fail for signed C++ type.
  UIntType uint31("ui31", 31);
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(&uint31),
               AcceleratorMismatchError);

  // int64_t can hold si33 (width 33, in range (32,64]).
  SIntType sint33b("si33", 33);
  EXPECT_NO_THROW(verifyTypeCompatibility<int64_t>(&sint33b));

  // si64 fits exactly in int64_t. Should pass.
  SIntType sint64("si64", 64);
  EXPECT_NO_THROW(verifyTypeCompatibility<int64_t>(&sint64));

  // si65 exceeds int64_t. Should fail.
  SIntType sint65("si65", 65);
  EXPECT_THROW(verifyTypeCompatibility<int64_t>(&sint65),
               AcceleratorMismatchError);

  // si32 fits in int64_t but int32_t would suffice. Reject.
  EXPECT_THROW(verifyTypeCompatibility<int64_t>(&sint32),
               AcceleratorMismatchError);
}

TEST(TypedPortsTest, UnsignedIntTypeCompatibility) {
  // uint32_t can hold ui17 (width 17, in range (16,32]).
  UIntType uint17("ui17", 17);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&uint17));

  // ui32 has width 32, which fits exactly in uint32_t. Should pass.
  UIntType uint32_t_("ui32", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&uint32_t_));

  // ui33 exceeds uint32_t. Should fail.
  UIntType uint33("ui33", 33);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&uint33),
               AcceleratorMismatchError);

  // ui16 fits but uint16_t would suffice. Reject for uint32_t.
  UIntType uint16_("ui16", 16);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&uint16_),
               AcceleratorMismatchError);

  // But ui16 should be fine for uint16_t.
  EXPECT_NO_THROW(verifyTypeCompatibility<uint16_t>(&uint16_));

  // BitsType (signless iM) should also be accepted for unsigned.
  BitsType bits17("i17", 17);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&bits17));

  // BitsType with width 32 fits in uint32_t. Should pass.
  BitsType bits32("i32", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint32_t>(&bits32));

  // BitsType with width 33 exceeds uint32_t. Should fail.
  BitsType bits33("i33", 33);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&bits33),
               AcceleratorMismatchError);

  // BitsType width 8 should be rejected for uint32_t (uint8_t suffices).
  BitsType bits8("i8", 8);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&bits8),
               AcceleratorMismatchError);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint8_t>(&bits8));

  // uint64_t with ui33 (in range (32,64]).
  UIntType uint33b("ui33", 33);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint64_t>(&uint33b));

  // uint64_t with ui64 fits exactly. Should pass.
  UIntType uint64_t_("ui64", 64);
  EXPECT_NO_THROW(verifyTypeCompatibility<uint64_t>(&uint64_t_));

  // uint64_t with ui65 exceeds. Should fail.
  UIntType uint65("ui65", 65);
  EXPECT_THROW(verifyTypeCompatibility<uint64_t>(&uint65),
               AcceleratorMismatchError);

  // uint64_t with ui32 — uint32_t would suffice. Reject.
  EXPECT_THROW(verifyTypeCompatibility<uint64_t>(&uint32_t_),
               AcceleratorMismatchError);

  // SIntType should fail for unsigned C++ type.
  SIntType sint31("si31", 31);
  EXPECT_THROW(verifyTypeCompatibility<uint32_t>(&sint31),
               AcceleratorMismatchError);
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
               AcceleratorMismatchError);

  // Even a non-struct type with matching ID should pass (ID comparison only).
  UIntType uintWithMatchingID("MyModule.TestStruct", 32);
  EXPECT_NO_THROW(verifyTypeCompatibility<TestStruct>(&uintWithMatchingID));
}

TEST(TypedPortsTest, NullPortTypeThrows) {
  EXPECT_THROW(verifyTypeCompatibility<int32_t>(nullptr),
               AcceleratorMismatchError);
  EXPECT_THROW(verifyTypeCompatibility<void>(nullptr),
               AcceleratorMismatchError);
}

// A type that is not integral and has no _ESI_ID — should hit fallback.
struct UnknownCppType {
  double x;
};

TEST(TypedPortsTest, FallbackThrows) {
  UIntType uint32("ui32", 32);
  EXPECT_THROW(verifyTypeCompatibility<UnknownCppType>(&uint32),
               AcceleratorMismatchError);
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
  EXPECT_THROW(typed.connect(), AcceleratorMismatchError);
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
  TypedWritePort<int16_t> typed(mock);
  typed.connect();

  int16_t val = 12345;
  typed.write(val);

  // Wire size for si15 is 2 bytes ((15+7)/8).
  ASSERT_EQ(mock.lastWritten.getSize(), 2u);
}

TEST(TypedPortsTest, SignExtensionNonByteAligned) {
  // Test fromMessageData sign extension for non-byte-aligned widths.
  // si4 value -1 is wire 0x0F (4 bits: 1111). Sign bit is bit 3.
  {
    SIntType si4("si4", 4);
    WireInfo wi = getWireInfo(&si4);
    EXPECT_EQ(wi.bytes, 1u);
    EXPECT_EQ(wi.bitWidth, 4u);
    uint8_t wire = 0x0F; // -1 in si4
    MessageData msg(&wire, 1);
    int32_t val = fromMessageData<int32_t>(msg, wi);
    EXPECT_EQ(val, -1);
  }
  // si4 value 7 is wire 0x07 (4 bits: 0111). Positive, no sign extension.
  {
    SIntType si4("si4", 4);
    WireInfo wi = getWireInfo(&si4);
    uint8_t wire = 0x07;
    MessageData msg(&wire, 1);
    int32_t val = fromMessageData<int32_t>(msg, wi);
    EXPECT_EQ(val, 7);
  }
  // si22 value -1 is wire {0xFF, 0xFF, 0x3F} (22 bits all 1s).
  // Sign bit is bit 21 = bit 5 of byte 2, mask 0x20.
  {
    SIntType si22("si22", 22);
    WireInfo wi = getWireInfo(&si22);
    EXPECT_EQ(wi.bytes, 3u);
    uint8_t wire[3] = {0xFF, 0xFF, 0x3F};
    MessageData msg(wire, 3);
    int32_t val = fromMessageData<int32_t>(msg, wi);
    EXPECT_EQ(val, -1);
  }
  // si22 positive value: 0x1FFFFF (all data bits 1, sign bit 0)
  {
    SIntType si22("si22", 22);
    WireInfo wi = getWireInfo(&si22);
    uint8_t wire[3] = {0xFF, 0xFF, 0x1F}; // bit 21 = 0
    MessageData msg(wire, 3);
    int32_t val = fromMessageData<int32_t>(msg, wi);
    EXPECT_EQ(val, 0x1FFFFF); // 2097151
  }
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

TEST(TypedPortsTest, TypedFunctionNullThrowsAtConnect) {
  // Null is accepted at construction but throws at connect().
  TypedFunction<uint32_t, uint16_t> typed(nullptr);
  EXPECT_THROW(typed.connect(), AcceleratorMismatchError);
}

TEST(TypedPortsTest, TypedFunctionConnectVerifiesTypes) {
  // Create channel types matching si24 arg and ui16 result.
  SIntType argInner("si24", 24);
  ChannelType argChanType("channel<si24>", &argInner);
  UIntType resultInner("ui15", 15);
  ChannelType resultChanType("channel<ui15>", &resultInner);

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
  UIntType resultInner("ui15", 15);
  ChannelType resultChanType("channel<ui15>", &resultInner);

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
  EXPECT_THROW(typed.connect(), AcceleratorMismatchError);
  delete func;
}

TEST(TypedPortsTest, TypedFunctionCallRoundTrip) {
  SIntType argInner("si24", 24);
  ChannelType argChanType("channel<si24>", &argInner);
  UIntType resultInner("ui15", 15);
  ChannelType resultChanType("channel<ui15>", &resultInner);

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

  // Verify the written arg matches — si24 wire size is 3 bytes.
  ASSERT_EQ(mockWrite.lastWritten.getSize(), 3u);
  delete func;
}

} // namespace
