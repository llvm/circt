//===- ESIRuntimeTypedPortsTest.cpp - Typed ESI port tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/TypedPorts.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>

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

struct DeserializerWithESIID {
  static constexpr std::string_view _ESI_ID = "MyModule.DeserializedStruct";

  class TypeDeserializer
      : public QueuedDecodeTypeDeserializer<DeserializerWithESIID> {
  public:
    using Base = QueuedDecodeTypeDeserializer<DeserializerWithESIID>;
    using OutputCallback = Base::OutputCallback;
    using DecodedOutputs = Base::DecodedOutputs;

    explicit TypeDeserializer(OutputCallback output)
        : Base(std::move(output)) {}

  private:
    DecodedOutputs decode(std::unique_ptr<SegmentedMessageData> &msg) override {
      msg.reset();
      return {};
    }
  };
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

TEST(TypedPortsTest, TypedReadPortCustomDeserializerVerifiesESIID) {
  StructType matchType("MyModule.DeserializedStruct", {});
  MockReadPort matching(&matchType);
  TypedReadPort<DeserializerWithESIID> ok(matching);
  EXPECT_NO_THROW(ok.connect());

  StructType mismatchType("OtherModule.OtherStruct", {});
  MockReadPort mismatch(&mismatchType);
  TypedReadPort<DeserializerWithESIID> bad(mismatch);
  EXPECT_THROW(bad.connect(), AcceleratorMismatchError);
}

class CallbackDrivenMockReadPort : public ReadChannelPort {
public:
  CallbackDrivenMockReadPort(const Type *type) : ReadChannelPort(type) {}

  bool deliver(std::unique_ptr<SegmentedMessageData> msg) {
    ++deliveryCount;
    pending = std::move(msg);
    return retryPending();
  }

  bool retryPending() {
    if (!pending)
      throw std::runtime_error(
          "CallbackDrivenMockReadPort::retryPending with no message");
    if (!invokeCallback(pending))
      return false;
    pending.reset();
    return true;
  }

  bool hasPending() const { return static_cast<bool>(pending); }
  size_t numActiveCallbacks() const { return activeCallbacks; }

  size_t deliveryCount = 0;

private:
  std::unique_ptr<SegmentedMessageData> pending;
};

class ThrowOnCopyReadCallback {
public:
  explicit ThrowOnCopyReadCallback(std::shared_ptr<bool> shouldThrow)
      : shouldThrow(std::move(shouldThrow)) {}

  ThrowOnCopyReadCallback(const ThrowOnCopyReadCallback &other)
      : shouldThrow(other.shouldThrow) {
    if (*shouldThrow)
      throw std::runtime_error("ThrowOnCopyReadCallback copy failure");
  }

  ThrowOnCopyReadCallback(ThrowOnCopyReadCallback &&) = default;
  ThrowOnCopyReadCallback &operator=(const ThrowOnCopyReadCallback &) = default;
  ThrowOnCopyReadCallback &operator=(ThrowOnCopyReadCallback &&) = default;

  bool operator()(std::unique_ptr<SegmentedMessageData> &) const {
    return true;
  }

private:
  std::shared_ptr<bool> shouldThrow;
};

static MessageData packUint32Words(std::initializer_list<uint32_t> values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(uint32_t));
  size_t offset = 0;
  for (uint32_t value : values) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
    offset += sizeof(value);
  }
  return MessageData(std::move(bytes));
}

struct BufferedSequence {
  std::vector<uint32_t> values;

  class TypeDeserializer
      : public QueuedDecodeTypeDeserializer<BufferedSequence> {
  public:
    using Base = QueuedDecodeTypeDeserializer<BufferedSequence>;
    using OutputCallback = Base::OutputCallback;
    using DecodedOutputs = Base::DecodedOutputs;

    explicit TypeDeserializer(OutputCallback output)
        : Base(std::move(output)) {}

  private:
    DecodedOutputs decode(std::unique_ptr<SegmentedMessageData> &msg) override {
      MessageData scratch;
      const MessageData &flat =
          detail::getMessageDataRef<BufferedSequence>(*msg, scratch);
      if (flat.getSize() % sizeof(uint32_t) != 0)
        throw std::runtime_error(
            "BufferedSequence::TypeDeserializer: truncated word payload");

      DecodedOutputs decoded;
      for (size_t offset = 0; offset < flat.getSize();
           offset += sizeof(uint32_t)) {
        uint32_t value = 0;
        std::memcpy(&value, flat.getBytes() + offset, sizeof(value));
        auto sequence = std::make_unique<BufferedSequence>();
        sequence->values.push_back(value);
        decoded.push_back(std::move(sequence));
      }
      msg.reset();
      return decoded;
    }
  };
};

TEST(TypedPortsTest, TypedReadPortPODBackpressuresAfterOneBufferedOutput) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  TypedReadPort<uint32_t> typed(mock);
  typed.connect();
  typed.setMaxDataQueueMsgs(1);

  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({11}))));
  // The second raw message is consumed into the POD deserializer's single-slot
  // typed buffer even though the polling queue is full. Backpressure shows up
  // on the next raw message boundary.
  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({22}))));
  EXPECT_FALSE(mock.hasPending());
  EXPECT_FALSE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({33}))));
  EXPECT_TRUE(mock.hasPending());

  std::unique_ptr<uint32_t> first = typed.read();
  ASSERT_TRUE(first);
  EXPECT_EQ(*first, 11u);
  EXPECT_TRUE(mock.retryPending());
  EXPECT_FALSE(mock.hasPending());
  std::unique_ptr<uint32_t> second = typed.read();
  ASSERT_TRUE(second);
  EXPECT_EQ(*second, 22u);
  std::unique_ptr<uint32_t> third = typed.read();
  ASSERT_TRUE(third);
  EXPECT_EQ(*third, 33u);
}

TEST(TypedPortsTest, TypedReadPortPODRetriesSameOwnedObjectOnLaterPush) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  TypedReadPort<uint32_t> typed(mock);
  const uint32_t *firstObject = nullptr;
  size_t callbackAttempts = 0;

  typed.connect([&](std::unique_ptr<uint32_t> &value) {
    ++callbackAttempts;
    EXPECT_TRUE(value);
    if (callbackAttempts == 1) {
      EXPECT_EQ(*value, 11u);
      firstObject = value.get();
      return false;
    }
    if (callbackAttempts == 2) {
      EXPECT_EQ(*value, 11u);
      EXPECT_EQ(value.get(), firstObject);
      return true;
    }
    EXPECT_EQ(*value, 22u);
    return true;
  });

  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({11}))));
  EXPECT_FALSE(mock.hasPending());
  // A later raw push first retries the buffered typed value, then handles the
  // new message once that buffered value is accepted.
  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({22}))));
  EXPECT_FALSE(mock.hasPending());
  EXPECT_EQ(callbackAttempts, 3u);
}

TEST(TypedPortsTest, TypedReadPortCustomDeserializerPokesBlockedOutput) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  TypedReadPort<BufferedSequence> typed(mock);
  typed.connect();
  typed.setMaxDataQueueMsgs(1);

  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({10, 20}))));
  EXPECT_EQ(mock.deliveryCount, 1u);

  std::unique_ptr<BufferedSequence> first = typed.read();
  ASSERT_TRUE(first);
  ASSERT_EQ(first->values.size(), 1u);
  EXPECT_EQ(first->values[0], 10u);

  std::unique_ptr<BufferedSequence> second = typed.read();
  ASSERT_TRUE(second);
  ASSERT_EQ(second->values.size(), 1u);
  EXPECT_EQ(second->values[0], 20u);
  EXPECT_EQ(mock.deliveryCount, 1u);
}

TEST(TypedPortsTest,
     TypedReadPortCustomDeserializerConsumesMultipleFramesPerRawMessage) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  TypedReadPort<BufferedSequence> typed(mock);
  typed.connect();

  std::future<std::unique_ptr<BufferedSequence>> first = typed.readAsync();
  std::future<std::unique_ptr<BufferedSequence>> second = typed.readAsync();
  std::future<std::unique_ptr<BufferedSequence>> third = typed.readAsync();

  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({1, 2, 3}))));

  std::unique_ptr<BufferedSequence> firstValue = first.get();
  ASSERT_TRUE(firstValue);
  EXPECT_EQ(firstValue->values[0], 1u);

  std::unique_ptr<BufferedSequence> secondValue = second.get();
  ASSERT_TRUE(secondValue);
  EXPECT_EQ(secondValue->values[0], 2u);

  std::unique_ptr<BufferedSequence> thirdValue = third.get();
  ASSERT_TRUE(thirdValue);
  EXPECT_EQ(thirdValue->values[0], 3u);
}

TEST(TypedPortsTest,
     TypedReadPortCustomDeserializerQueuesMultiplePendingOutputs) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  TypedReadPort<BufferedSequence> typed(mock);
  typed.connect();
  typed.setMaxDataQueueMsgs(1);

  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({7, 8, 9}))));
  EXPECT_EQ(mock.deliveryCount, 1u);

  std::unique_ptr<BufferedSequence> first = typed.read();
  ASSERT_TRUE(first);
  EXPECT_EQ(first->values[0], 7u);

  std::unique_ptr<BufferedSequence> second = typed.read();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->values[0], 8u);

  std::unique_ptr<BufferedSequence> third = typed.read();
  ASSERT_TRUE(third);
  EXPECT_EQ(third->values[0], 9u);
  EXPECT_EQ(mock.deliveryCount, 1u);
}

struct FragmentedCoord {
  uint32_t y;
  uint32_t x;
};
static_assert(sizeof(FragmentedCoord) == 8, "Size mismatch");

static std::array<uint8_t, sizeof(FragmentedCoord)> packCoordBytes(uint32_t y,
                                                                   uint32_t x) {
  FragmentedCoord coord{y, x};
  std::array<uint8_t, sizeof(FragmentedCoord)> bytes{};
  std::memcpy(bytes.data(), &coord, sizeof(coord));
  return bytes;
}

struct FragmentedCoordBatch {
  std::vector<FragmentedCoord> coords;

  class TypeDeserializer
      : public QueuedDecodeTypeDeserializer<FragmentedCoordBatch> {
  public:
    using Base = QueuedDecodeTypeDeserializer<FragmentedCoordBatch>;
    using OutputCallback = Base::OutputCallback;
    using DecodedOutputs = Base::DecodedOutputs;

    explicit TypeDeserializer(OutputCallback output)
        : Base(std::move(output)) {}

  private:
    DecodedOutputs decode(std::unique_ptr<SegmentedMessageData> &msg) override {
      MessageData scratch;
      const MessageData &flat =
          detail::getMessageDataRef<FragmentedCoordBatch>(*msg, scratch);

      DecodedOutputs decoded;
      const uint8_t *bytes = flat.getBytes();
      size_t offset = 0;
      while (offset < flat.getSize()) {
        size_t needed = sizeof(FragmentedCoord) - partialFrameBytes.size();
        size_t chunkSize = std::min(needed, flat.getSize() - offset);
        partialFrameBytes.insert(partialFrameBytes.end(), bytes + offset,
                                 bytes + offset + chunkSize);
        offset += chunkSize;

        if (partialFrameBytes.size() != sizeof(FragmentedCoord))
          break;

        FragmentedCoord coord;
        std::memcpy(&coord, partialFrameBytes.data(), sizeof(coord));
        partialFrameBytes.clear();

        auto batch = std::make_unique<FragmentedCoordBatch>();
        batch->coords.push_back(coord);
        decoded.push_back(std::move(batch));
      }

      msg.reset();
      return decoded;
    }

    std::vector<uint8_t> partialFrameBytes;
  };
};

TEST(TypedPortsTest,
     TypedReadPortCustomDeserializerConsumesSplitFramesAcrossRawMessages) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  TypedReadPort<FragmentedCoordBatch> typed(mock);
  typed.connect();

  std::future<std::unique_ptr<FragmentedCoordBatch>> first = typed.readAsync();
  std::future<std::unique_ptr<FragmentedCoordBatch>> second = typed.readAsync();

  std::array<uint8_t, sizeof(FragmentedCoord)> coordA = packCoordBytes(10, 20);
  std::array<uint8_t, sizeof(FragmentedCoord)> coordB = packCoordBytes(30, 40);

  std::vector<uint8_t> firstChunk(coordA.begin(), coordA.begin() + 6);
  EXPECT_TRUE(mock.deliver(
      std::make_unique<MessageData>(MessageData(std::move(firstChunk)))));

  std::vector<uint8_t> secondChunk;
  secondChunk.insert(secondChunk.end(), coordA.begin() + 6, coordA.end());
  secondChunk.insert(secondChunk.end(), coordB.begin(), coordB.end());
  EXPECT_TRUE(mock.deliver(
      std::make_unique<MessageData>(MessageData(std::move(secondChunk)))));

  std::unique_ptr<FragmentedCoordBatch> firstBatch = first.get();
  ASSERT_TRUE(firstBatch);
  ASSERT_EQ(firstBatch->coords.size(), 1u);
  EXPECT_EQ(firstBatch->coords[0].y, 10u);
  EXPECT_EQ(firstBatch->coords[0].x, 20u);

  std::unique_ptr<FragmentedCoordBatch> secondBatch = second.get();
  ASSERT_TRUE(secondBatch);
  ASSERT_EQ(secondBatch->coords.size(), 1u);
  EXPECT_EQ(secondBatch->coords[0].y, 30u);
  EXPECT_EQ(secondBatch->coords[0].x, 40u);
}
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

//===----------------------------------------------------------------------===//
// SegmentedMessageData tests via TypedWritePort
//===----------------------------------------------------------------------===//

/// A minimal SegmentedMessageData for testing.
struct TestSegmented : public SegmentedMessageData {
#pragma pack(push, 1)
  struct Header {
    uint32_t a;
    uint16_t b;
  };
#pragma pack(pop)

  Header header;
  std::vector<uint32_t> items;

  size_t numSegments() const override { return 2; }
  Segment segment(size_t idx) const override {
    if (idx == 0)
      return {reinterpret_cast<const uint8_t *>(&header), sizeof(Header)};
    return {reinterpret_cast<const uint8_t *>(items.data()),
            items.size() * sizeof(uint32_t)};
  }
};

TEST(TypedPortsTest, ReadChannelPortSegmentedCallbackRetriesSameMessageObject) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  uint32_t expected = 0x12345678;
  const uint8_t *firstBytes = nullptr;
  size_t callbackCalls = 0;

  mock.connect([&](std::unique_ptr<SegmentedMessageData> &msg) -> bool {
    ++callbackCalls;
    EXPECT_EQ(msg->numSegments(), 1u);

    auto *flat = dynamic_cast<MessageData *>(msg.get());
    EXPECT_NE(flat, nullptr);
    if (!flat)
      return false;
    EXPECT_EQ(*flat->as<uint32_t>(), expected);

    if (callbackCalls == 1)
      firstBytes = flat->getBytes();
    else
      EXPECT_EQ(flat->getBytes(), firstBytes);

    return callbackCalls == 2;
  });

  EXPECT_FALSE(
      mock.deliver(std::make_unique<MessageData>(MessageData::from(expected))));
  EXPECT_TRUE(mock.hasPending());
  EXPECT_TRUE(mock.retryPending());
  EXPECT_FALSE(mock.hasPending());
  EXPECT_EQ(callbackCalls, 2u);
}

TEST(TypedPortsTest, ReadChannelPortFlatCallbackFlattensSegmentedMessageRetry) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  TestSegmented input;
  input.header.a = 0x12345678;
  input.header.b = 0xABCD;
  input.items = {1, 2, 3};
  MessageData expected = input.toMessageData();

  size_t callbackCalls = 0;
  mock.connect([&](MessageData data) {
    ++callbackCalls;
    EXPECT_EQ(data.getData(), expected.getData());
    return callbackCalls == 2;
  });

  EXPECT_FALSE(mock.deliver(std::make_unique<TestSegmented>(input)));
  EXPECT_TRUE(mock.hasPending());
  EXPECT_TRUE(mock.retryPending());
  EXPECT_FALSE(mock.hasPending());
  EXPECT_EQ(callbackCalls, 2u);
}

TEST(TypedPortsTest, ReadChannelPortPollingRetriesFlattenedSegmentedMessage) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);
  mock.connect();
  mock.setMaxDataQueueMsgs(1);

  TestSegmented firstInput;
  firstInput.header.a = 0xAAAA5555;
  firstInput.header.b = 0x1357;
  firstInput.items = {10};
  MessageData firstExpected = firstInput.toMessageData();

  TestSegmented secondInput;
  secondInput.header.a = 0xDEADBEEF;
  secondInput.header.b = 0x2468;
  secondInput.items = {20, 30};
  MessageData secondExpected = secondInput.toMessageData();

  EXPECT_TRUE(mock.deliver(std::make_unique<TestSegmented>(firstInput)));
  EXPECT_FALSE(mock.deliver(std::make_unique<TestSegmented>(secondInput)));
  EXPECT_TRUE(mock.hasPending());

  MessageData firstOut;
  mock.read(firstOut);
  EXPECT_EQ(firstOut.getData(), firstExpected.getData());

  EXPECT_TRUE(mock.retryPending());
  EXPECT_FALSE(mock.hasPending());

  MessageData secondOut;
  mock.read(secondOut);
  EXPECT_EQ(secondOut.getData(), secondExpected.getData());
}

TEST(TypedPortsTest, ReadChannelPortPollingReadAsyncThrowsWhenDisconnected) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  EXPECT_THROW(mock.readAsync(), std::runtime_error);

  mock.connect();
  std::future<MessageData> pending = mock.readAsync();
  mock.disconnect();

  EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(0)),
            std::future_status::ready);
  EXPECT_THROW(pending.get(), std::future_error);
  EXPECT_THROW(mock.readAsync(), std::runtime_error);

  mock.connect();
  EXPECT_TRUE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({11}))));
  MessageData out;
  mock.read(out);
  EXPECT_EQ(*out.as<uint32_t>(), 11u);
}

TEST(TypedPortsTest, ReadChannelPortPollingConnectRejectsReconnect) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  mock.connect();
  EXPECT_THROW(mock.connect(), std::runtime_error);
}

TEST(TypedPortsTest, ReadChannelPortDisconnectRevokesCallback) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  mock.connect();
  mock.disconnect();

  EXPECT_FALSE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({11}))));
  EXPECT_TRUE(mock.hasPending());

  mock.connect();
  EXPECT_TRUE(mock.retryPending());
  EXPECT_FALSE(mock.hasPending());

  MessageData out;
  mock.read(out);
  EXPECT_EQ(*out.as<uint32_t>(), 11u);
}

TEST(TypedPortsTest, TypedReadPortDestructorDisconnectsRawPort) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);

  {
    TypedReadPort<uint32_t> typed(mock);
    typed.connect();
    EXPECT_TRUE(mock.isConnected());
  }

  EXPECT_FALSE(mock.isConnected());
  EXPECT_FALSE(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({11}))));
  EXPECT_TRUE(mock.hasPending());

  TypedReadPort<uint32_t> reconnected(mock);
  reconnected.connect();
  EXPECT_TRUE(mock.retryPending());
  EXPECT_FALSE(mock.hasPending());

  std::unique_ptr<uint32_t> out = reconnected.read();
  ASSERT_TRUE(out);
  EXPECT_EQ(*out, 11u);
}

TEST(TypedPortsTest,
     ReadChannelPortInvokeCallbackMaintainsCountOnCallbackCopyFailure) {
  UIntType uint32("ui32", 32);
  CallbackDrivenMockReadPort mock(&uint32);
  auto shouldThrow = std::make_shared<bool>(false);

  mock.connect(ThrowOnCopyReadCallback(shouldThrow));
  *shouldThrow = true;

  EXPECT_THROW(
      mock.deliver(std::make_unique<MessageData>(packUint32Words({11}))),
      std::runtime_error);
  EXPECT_EQ(mock.numActiveCallbacks(), 0u);
}

TEST(TypedPortsTest, TypedWritePortSegmentedMessageData) {
  // Use any type — SegmentedMessageData skips type checks.
  UIntType uint32("ui32", 32);
  MockWritePort mock(&uint32);

  TypedWritePort<TestSegmented, true> typed(mock);
  typed.connect();

  TestSegmented msg;
  msg.header.a = 0x12345678;
  msg.header.b = 0xABCD;
  msg.items = {1, 2, 3};

  // write(const T&) should flatten via toMessageData().
  typed.write(msg);

  // Expected: 6 bytes header + 12 bytes data = 18 bytes.
  EXPECT_EQ(mock.lastWritten.getSize(), 18u);

  // Verify header bytes.
  const uint8_t *bytes = mock.lastWritten.getBytes();
  uint32_t gotA;
  uint16_t gotB;
  std::memcpy(&gotA, bytes, 4);
  std::memcpy(&gotB, bytes + 4, 2);
  EXPECT_EQ(gotA, 0x12345678u);
  EXPECT_EQ(gotB, 0xABCDu);

  // Verify item bytes.
  uint32_t gotItems[3];
  std::memcpy(gotItems, bytes + 6, 12);
  EXPECT_EQ(gotItems[0], 1u);
  EXPECT_EQ(gotItems[1], 2u);
  EXPECT_EQ(gotItems[2], 3u);
}

TEST(TypedPortsTest, TypedWritePortSegmentedNoTypeCheck) {
  // SegmentedMessageData type against a mismatched port type — works because
  // SkipTypeCheck=true bypasses verifyTypeCompatibility entirely.
  SIntType sint8("si8", 8);
  MockWritePort mock(&sint8);

  TypedWritePort<TestSegmented, true> typed(mock);
  EXPECT_NO_THROW(typed.connect());

  TestSegmented msg;
  msg.header.a = 42;
  msg.header.b = 7;
  msg.items = {100};

  typed.write(msg);
  // 6 bytes header + 4 bytes data = 10 bytes.
  EXPECT_EQ(mock.lastWritten.getSize(), 10u);
}

TEST(TypedPortsTest, TypedFunctionSegmentedArg) {
  // Arg type is SegmentedMessageData — type check is skipped for it.
  // Use an arbitrary inner type for the arg channel since it won't be checked.
  UIntType argInner("ui32", 32);
  ChannelType argChanType("channel<ui32>", &argInner);
  UIntType resultInner("ui16", 16);
  ChannelType resultChanType("channel<ui16>", &resultInner);

  BundleType::ChannelVector channels = {
      {"arg", BundleType::Direction::To, &argChanType},
      {"result", BundleType::Direction::From, &resultChanType},
  };
  BundleType bundleType("func_bundle", channels);

  MockWritePort mockWrite(&argInner);
  MockReadPort mockRead(&resultInner);

  // Set up mock read to return a known uint16_t value.
  uint16_t expected = 99;
  mockRead.nextResponse = MessageData::from(expected);

  auto *func = services::FuncService::Function::get(AppID("test"), &bundleType,
                                                    mockWrite, mockRead);

  TypedFunction<TestSegmented, uint16_t, true> typed(func);
  typed.connect();

  TestSegmented arg;
  arg.header.a = 0xDEAD;
  arg.header.b = 0xBE;
  arg.items = {10, 20};

  uint16_t result = typed.call(arg).get();
  EXPECT_EQ(result, 99u);

  // Verify the flattened arg: 6 bytes header + 8 bytes items = 14 bytes.
  EXPECT_EQ(mockWrite.lastWritten.getSize(), 14u);

  const uint8_t *bytes = mockWrite.lastWritten.getBytes();
  uint32_t gotA;
  std::memcpy(&gotA, bytes, 4);
  EXPECT_EQ(gotA, 0xDEADu);

  uint32_t gotItem0, gotItem1;
  std::memcpy(&gotItem0, bytes + 6, 4);
  std::memcpy(&gotItem1, bytes + 10, 4);
  EXPECT_EQ(gotItem0, 10u);
  EXPECT_EQ(gotItem1, 20u);

  delete func;
}

// A plain struct without _ESI_ID — not integral, not void, not bool.
struct UnrecognizedCppType {
  double x;
  double y;
};

TEST(TypedPortsTest, VerifyTypeCompatibilityThrowsForUnsupportedType) {
  UIntType uint32("ui32", 32);
  EXPECT_THROW(verifyTypeCompatibility<UnrecognizedCppType>(&uint32),
               AcceleratorMismatchError);

  SIntType sint16("si16", 16);
  EXPECT_THROW(verifyTypeCompatibility<UnrecognizedCppType>(&sint16),
               AcceleratorMismatchError);
}

} // namespace
