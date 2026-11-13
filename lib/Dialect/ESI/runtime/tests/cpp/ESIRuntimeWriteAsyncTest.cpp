//===- ESIRuntimeWriteAsyncTest.cpp - WriteChannelPort callback mode -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Accelerator.h"
#include "esi/Ports.h"
#include "esi/TypedPorts.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace esi;

namespace {

/// A WriteChannelPort which records accepted messages. tryWriteImpl() can be
/// made to backpressure or throw.
class RecordingWritePort : public WriteChannelPort {
public:
  using WriteChannelPort::WriteChannelPort;

  std::vector<std::vector<uint8_t>> written() {
    std::lock_guard<std::mutex> lock(mutex);
    return writtenMsgs;
  }

  /// Reject (backpressure) this many tryWriteImpl() calls before accepting.
  std::atomic<int> rejectCount = 0;
  /// Reject all tryWriteImpl() calls.
  std::atomic<bool> rejectAll = false;
  /// Accept this many tryWriteImpl() calls, then throw on the next one.
  std::atomic<int> throwAfter = -1;
  /// If non-negative, accept this many more tryWriteImpl() calls, then reject.
  std::atomic<int> acceptBudget = -1;
  /// Reject every other tryWriteImpl() call.
  std::atomic<bool> alternate = false;
  /// Number of pollWriteImpl() calls.
  std::atomic<int> pollImplCount = 0;

protected:
  void writeImpl(const MessageData &data) override {
    while (!tryWriteImpl(data))
      std::this_thread::yield();
  }
  bool tryWriteImpl(const MessageData &data) override {
    if (rejectAll || acceptBudget == 0 || (alternate && calls++ % 2))
      return false;
    if (acceptBudget > 0)
      --acceptBudget;
    if (rejectCount > 0) {
      --rejectCount;
      return false;
    }
    if (throwAfter >= 0 && throwAfter-- == 0)
      throw std::out_of_range("injected failure");
    std::lock_guard<std::mutex> lock(mutex);
    writtenMsgs.push_back(data.getData());
    return true;
  }
  bool pollWriteImpl() override {
    ++pollImplCount;
    return false;
  }

private:
  std::atomic<unsigned> calls = 0;
  std::mutex mutex;
  std::vector<std::vector<uint8_t>> writtenMsgs;
};

/// A data source for callback mode: a queue of messages which the port fetches
/// from, plus a record of the errors reported for them.
struct Source {
  std::mutex mutex;
  std::deque<std::unique_ptr<SegmentedMessageData>> queue;
  std::vector<std::exception_ptr> errors;
  size_t fetches = 0;
  /// Optional extra behavior, invoked at the start of each fetch with its
  /// index, and with each error's index.
  std::function<void(size_t)> onFetch, onError;

  void push(std::vector<uint8_t> bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push_back(std::make_unique<MessageData>(std::move(bytes)));
  }
  void push(std::unique_ptr<SegmentedMessageData> msg) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push_back(std::move(msg));
  }

  WriteChannelPort::DataFetchCallback fetch() {
    return [this]() -> std::unique_ptr<SegmentedMessageData> {
      size_t idx;
      {
        std::lock_guard<std::mutex> lock(mutex);
        idx = fetches++;
      }
      if (onFetch)
        onFetch(idx);
      std::lock_guard<std::mutex> lock(mutex);
      if (queue.empty())
        return nullptr;
      std::unique_ptr<SegmentedMessageData> msg = std::move(queue.front());
      queue.pop_front();
      return msg;
    };
  }
  WriteChannelPort::WriteErrorCallback error() {
    return [this](std::exception_ptr e) {
      size_t idx;
      {
        std::lock_guard<std::mutex> lock(mutex);
        idx = errors.size();
        errors.push_back(e);
      }
      if (onError)
        onError(idx);
    };
  }

  size_t numFetches() {
    std::lock_guard<std::mutex> lock(mutex);
    return fetches;
  }
  size_t numErrors() {
    std::lock_guard<std::mutex> lock(mutex);
    return errors.size();
  }
  std::exception_ptr operator[](size_t idx) {
    std::lock_guard<std::mutex> lock(mutex);
    return errors.at(idx);
  }
  size_t queued() {
    std::lock_guard<std::mutex> lock(mutex);
    return queue.size();
  }
};

/// A two-segment message.
struct TwoSegmentMessage : public SegmentedMessageData {
  std::vector<uint8_t> header = {1, 2};
  std::vector<uint8_t> body = {3, 4, 5};
  size_t numSegments() const override { return 2; }
  Segment segment(size_t idx) const override {
    const std::vector<uint8_t> &seg = idx == 0 ? header : body;
    return {seg.data(), seg.size()};
  }
};

std::unique_ptr<SegmentedMessageData> makeMsg(std::vector<uint8_t> bytes) {
  return std::make_unique<MessageData>(std::move(bytes));
}

using Frames = std::vector<std::vector<uint8_t>>;

/// Rethrow `error` and return its message, checking that it is of type `E`.
template <typename E>
std::string errorMessage(std::exception_ptr error) {
  EXPECT_TRUE(error);
  if (!error)
    return "";
  try {
    std::rethrow_exception(error);
  } catch (const E &e) {
    return e.what();
  } catch (...) {
    ADD_FAILURE() << "unexpected exception type";
  }
  return "";
}

/// A two-frame window over a struct of two ui16s.
struct TwoFrameWindow {
  UIntType ui16{"ui16", 16};
  StructType into{"into", {{"a", &ui16}, {"b", &ui16}}, /*reverse=*/false};
  BitsType lowered{"lowered", 16};
  WindowType window{
      "window", "window", &into, &lowered, {{"f0", {{"a"}}}, {"f1", {{"b"}}}}};
};

/// A three-frame window over a struct of three ui16s.
struct ThreeFrameWindow {
  UIntType ui16{"ui16", 16};
  StructType into{"into",
                  {{"a", &ui16}, {"b", &ui16}, {"c", &ui16}},
                  /*reverse=*/false};
  BitsType lowered{"lowered", 16};
  WindowType window{"window",
                    "window",
                    &into,
                    &lowered,
                    {{"f0", {{"a"}}}, {"f1", {{"b"}}}, {"f2", {{"c"}}}}};
};

/// Feed the frames a write port produced into a translating read port of the
/// same type, as the receiving end of the channel would, and return the
/// messages it assembles.
std::vector<std::vector<uint8_t>>
receive(const Type *type, const std::vector<std::vector<uint8_t>> &frames) {
  struct LoopbackReadPort : public ReadChannelPort {
    using ReadChannelPort::ReadChannelPort;
    void deliver(const std::vector<uint8_t> &frame) {
      std::unique_ptr<SegmentedMessageData> msg =
          std::make_unique<MessageData>(frame);
      EXPECT_TRUE(invokeCallback(msg));
    }
  };
  std::vector<std::vector<uint8_t>> received;
  LoopbackReadPort reader(type);
  reader.connect(ReadChannelPort::ReadCallback(
                     [&](std::unique_ptr<SegmentedMessageData> &msg) {
                       received.push_back(msg->toMessageData().getData());
                       return true;
                     }),
                 ChannelPort::ConnectOptions(std::nullopt, true));
  for (const auto &frame : frames)
    reader.deliver(frame);
  reader.disconnect();
  return received;
}

TEST(WriteCallbackModeTest, NotifyWritesImmediately) {
  UIntType ui24("ui24", 24);
  Source source;
  RecordingWritePort port(&ui24);
  port.connect(source.fetch(), source.error());

  // The data is written on the calling thread before the notification returns,
  // then the source is drained until it returns nullptr.
  source.push({1, 2, 3});
  source.push({4, 5, 6});
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3}, {4, 5, 6}}));
  EXPECT_EQ(source.numFetches(), 3u);

  // After returning nullptr, the source isn't called again until notified.
  source.push({7, 8, 9});
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(source.numFetches(), 3u);
  EXPECT_EQ(port.pollImplCount, 1);
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written().size(), 3u);
  EXPECT_EQ(source.numErrors(), 0u);
}

TEST(WriteCallbackModeTest, FirstPollFetches) {
  UIntType ui8("ui8", 8);
  Source source;
  source.push({1});
  RecordingWritePort port(&ui8);
  port.connect(source.fetch(), source.error());
  // Connecting doesn't call the source; the first poll() does.
  EXPECT_EQ(source.numFetches(), 0u);
  EXPECT_TRUE(port.poll());
  EXPECT_EQ(port.written(), (Frames{{1}}));
}

TEST(WriteCallbackModeTest, BackpressureHoldsOneMessage) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  port.connect(source.fetch(), source.error());
  for (uint8_t i = 0; i < 10; ++i)
    source.push({i});

  // Under backpressure, only one message is taken from the source.
  port.rejectAll = true;
  port.notifyAsyncDataAvailable();
  port.notifyAsyncDataAvailable();
  EXPECT_FALSE(port.poll());
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(source.numFetches(), 1u);
  EXPECT_EQ(source.queued(), 9u);
  EXPECT_TRUE(port.written().empty());

  // Once the backend accepts it, the rest are fetched and written in order.
  port.rejectAll = false;
  EXPECT_TRUE(port.poll());
  Frames expected;
  for (uint8_t i = 0; i < 10; ++i)
    expected.push_back({i});
  EXPECT_EQ(port.written(), expected);
  EXPECT_EQ(source.numErrors(), 0u);
}

TEST(WriteCallbackModeTest, NotificationDuringFetchIsNotLost) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  // While the port is fetching (and about to be told there's no data), data
  // arrives and the port is notified.
  source.onFetch = [&](size_t idx) {
    if (idx == 0) {
      port.notifyAsyncDataAvailable();
      EXPECT_EQ(source.numFetches(), 1u) << "notify must not recurse";
    }
  };
  port.connect(source.fetch(), source.error());
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(source.numFetches(), 2u);

  source.onFetch = [&](size_t idx) {
    if (idx == 2) {
      source.push({1});
      port.notifyAsyncDataAvailable();
    }
  };
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{1}}));
}

TEST(WriteCallbackModeTest, LongBurstDoesNotRecurse) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  constexpr size_t numMsgs = 100000;
  for (size_t i = 0; i < numMsgs; ++i)
    source.push({static_cast<uint8_t>(i)});
  port.connect(source.fetch(), source.error());
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written().size(), numMsgs);
}

TEST(WriteCallbackModeTest, WriteErrorReportedAndMessageDropped) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  port.connect(source.fetch(), source.error());
  source.push({1});
  source.push({2});
  port.throwAfter = 0;
  port.notifyAsyncDataAvailable();

  // {1} fails and is dropped; writing continues with {2}.
  ASSERT_EQ(source.numErrors(), 1u);
  EXPECT_EQ(errorMessage<std::out_of_range>(source[0]), "injected failure");
  EXPECT_EQ(port.written(), (Frames{{2}}));
}

TEST(WriteCallbackModeTest, FetchErrorReported) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  source.onFetch = [](size_t idx) {
    if (idx == 0)
      throw std::logic_error("fetch failure");
  };
  port.connect(source.fetch(), source.error());
  source.push({1});
  port.notifyAsyncDataAvailable();
  ASSERT_EQ(source.numErrors(), 1u);
  EXPECT_EQ(errorMessage<std::logic_error>(source[0]), "fetch failure");
  // Like returning nullptr, the port doesn't fetch again until notified.
  EXPECT_TRUE(port.written().empty());
  port.poll();
  EXPECT_EQ(source.numFetches(), 1u);
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{1}}));
}

TEST(WriteCallbackModeTest, ThrowingErrorCallbackIsContained) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  source.onError = [](size_t) { throw std::runtime_error("callback failure"); };
  port.connect(source.fetch(), source.error());
  source.push({1});
  source.push({2});
  port.throwAfter = 0;
  EXPECT_NO_THROW(port.notifyAsyncDataAvailable());
  EXPECT_EQ(source.numErrors(), 1u);
  EXPECT_EQ(port.written(), (Frames{{2}}));
}

TEST(WriteCallbackModeTest, SegmentedMessageIsFlattened) {
  BitsType bits40("i40", 40);
  Source source;
  RecordingWritePort port(&bits40);
  port.connect(source.fetch(), source.error());
  source.push(std::make_unique<TwoSegmentMessage>());
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3, 4, 5}}));
}

TEST(WriteCallbackModeTest, UsesBackendSegmentedTryWrite) {
  // A backend with scatter-gather support, which takes segmented messages
  // as-is.
  struct ScatterGatherPort : public RecordingWritePort {
    using RecordingWritePort::RecordingWritePort;
    using WriteChannelPort::tryWrite;
    bool tryWrite(std::unique_ptr<SegmentedMessageData> &msg) override {
      if (rejectAll)
        return false;
      received.push_back(std::move(msg));
      return true;
    }
    std::vector<std::unique_ptr<SegmentedMessageData>> received;
  };

  BitsType bits40("i40", 40);
  Source source;
  ScatterGatherPort port(&bits40);
  port.connect(source.fetch(), source.error());
  auto msg = std::make_unique<TwoSegmentMessage>();
  SegmentedMessageData *original = msg.get();
  source.push(std::move(msg));
  port.rejectAll = true;
  port.notifyAsyncDataAvailable();
  EXPECT_TRUE(port.received.empty());
  port.rejectAll = false;
  port.poll();
  // The original message was handed over without being flattened.
  ASSERT_EQ(port.received.size(), 1u);
  EXPECT_EQ(port.received[0].get(), original);
  EXPECT_TRUE(port.written().empty());
}

TEST(WriteCallbackModeTest, VoidTypeIsPadded) {
  VoidType voidType("void");
  Source source;
  RecordingWritePort port(&voidType);
  port.connect(source.fetch(), source.error());
  source.push(std::make_unique<MessageData>());
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{0}}));
}

TEST(WriteCallbackModeTest, InvalidCalls) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  // Notifying a port which isn't in callback mode is a no-op.
  EXPECT_NO_THROW(port.notifyAsyncDataAvailable());
  EXPECT_THROW(port.connect(nullptr, source.error()), std::invalid_argument);
  EXPECT_THROW(port.connect(source.fetch(), nullptr), std::invalid_argument);
  EXPECT_FALSE(port.isConnected());
  port.connect();
  EXPECT_NO_THROW(port.notifyAsyncDataAvailable());
  EXPECT_EQ(source.numFetches(), 0u);
}

TEST(WriteCallbackModeTest, ModesAreExclusive) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  MessageData data(std::vector<uint8_t>{1});

  // Sync mode rejects connecting with callbacks.
  port.connect();
  EXPECT_THROW(port.connect(source.fetch(), source.error()),
               std::runtime_error);
  port.write(data);
  EXPECT_TRUE(port.tryWrite(data));
  EXPECT_TRUE(port.flush());
  port.disconnect();

  // Callback mode rejects the sync APIs and reconnecting.
  port.connect(source.fetch(), source.error());
  EXPECT_THROW(port.write(data), std::runtime_error);
  EXPECT_THROW(port.write(makeMsg({1})), std::runtime_error);
  EXPECT_THROW(port.tryWrite(data), std::runtime_error);
  std::unique_ptr<SegmentedMessageData> segmented = makeMsg({1});
  EXPECT_THROW(port.tryWrite(segmented), std::runtime_error);
  EXPECT_TRUE(segmented);
  EXPECT_THROW(port.flush(), std::runtime_error);
  EXPECT_THROW(port.connect(), std::runtime_error);
  EXPECT_THROW(port.connect(source.fetch(), source.error()),
               std::runtime_error);
  port.disconnect();

  // After disconnecting, sync mode works again.
  port.connect();
  port.write(data);
  EXPECT_EQ(port.written().size(), 3u);
}

/// Count how many of the sync write APIs throw when called on `port`.
int countRejectedSyncWrites(WriteChannelPort &port) {
  MessageData data(std::vector<uint8_t>{9, 9});
  int rejected = 0;
  auto expectThrow = [&](auto &&fn) {
    try {
      fn();
    } catch (const std::runtime_error &) {
      ++rejected;
    }
  };
  expectThrow([&] { port.write(data); });
  expectThrow([&] { port.write(makeMsg({9, 9})); });
  expectThrow([&] { port.tryWrite(data); });
  expectThrow([&] {
    std::unique_ptr<SegmentedMessageData> msg = makeMsg({9, 9});
    port.tryWrite(msg);
  });
  expectThrow([&] { port.flush(); });
  return rejected;
}

TEST(WriteCallbackModeTest, SyncWritesThrowInCallbacks) {
  UIntType ui16("ui16", 16);
  Source source;
  RecordingWritePort port(&ui16);
  int rejected = 0;
  source.onFetch = [&](size_t) { rejected += countRejectedSyncWrites(port); };
  source.onError = [&](size_t) { rejected += countRejectedSyncWrites(port); };
  port.connect(source.fetch(), source.error());
  source.push({1, 2});
  source.push({3, 4});
  port.throwAfter = 0;
  port.notifyAsyncDataAvailable();
  // Three fetches and one error.
  EXPECT_EQ(rejected, 4 * 5);
  EXPECT_EQ(port.written(), (Frames{{3, 4}}));
}

//===----------------------------------------------------------------------===//
// Windowed messages, whose frames may be sent across several calls.
//===----------------------------------------------------------------------===//

TEST(WriteCallbackModeTest, WindowedFramesFinishedBeforeNextFetch) {
  TwoFrameWindow types;
  Source source;
  RecordingWritePort port(&types.window);
  std::vector<size_t> framesWrittenAtFetch;
  source.onFetch = [&](size_t) {
    framesWrittenAtFetch.push_back(port.written().size());
  };
  port.connect(source.fetch(), source.error(),
               ChannelPort::ConnectOptions(std::nullopt, true));
  source.push({1, 2, 3, 4});
  source.push({5, 6, 7, 8});

  // The first frame is accepted, the second isn't.
  port.acceptBudget = 1;
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written().size(), 1u);
  EXPECT_EQ(source.numFetches(), 1u);
  EXPECT_FALSE(port.poll());

  port.acceptBudget = -1;
  EXPECT_TRUE(port.poll());
  // Each message was fetched only once the previous one's frames were sent.
  EXPECT_EQ(framesWrittenAtFetch, (std::vector<size_t>{0, 2, 4}));
  EXPECT_EQ(receive(&types.window, port.written()),
            (Frames{{1, 2, 3, 4}, {5, 6, 7, 8}}));
}

TEST(WriteCallbackModeTest, PollReportsProgressOnRemainingFrames) {
  ThreeFrameWindow types;
  Source source;
  RecordingWritePort port(&types.window);
  port.connect(source.fetch(), source.error(),
               ChannelPort::ConnectOptions(std::nullopt, true));
  source.push({1, 2, 3, 4, 5, 6});
  port.acceptBudget = 1;
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written().size(), 1u);

  // Sending the second frame is progress, even though the third is stuck.
  port.acceptBudget = 1;
  EXPECT_TRUE(port.poll());
  EXPECT_EQ(port.written().size(), 2u);
  EXPECT_FALSE(port.poll());

  port.acceptBudget = -1;
  EXPECT_TRUE(port.poll());
  EXPECT_EQ(receive(&types.window, port.written()),
            (Frames{{1, 2, 3, 4, 5, 6}}));
}

TEST(WriteCallbackModeTest, WindowedFrameFailureReported) {
  TwoFrameWindow types;
  Source source;
  RecordingWritePort port(&types.window);
  port.connect(source.fetch(), source.error(),
               ChannelPort::ConnectOptions(std::nullopt, true));
  source.push({1, 2, 3, 4});
  source.push({5, 6, 7, 8});
  // The first frame is sent, then the second fails.
  port.throwAfter = 1;
  port.notifyAsyncDataAvailable();
  ASSERT_EQ(source.numErrors(), 1u);
  EXPECT_EQ(errorMessage<std::out_of_range>(source[0]), "injected failure");
  // Its remaining frame is dropped, and writing continues.
  EXPECT_EQ(port.written(), (Frames{{1, 2}, {5, 6}, {7, 8}}));
}

//===----------------------------------------------------------------------===//
// Disconnection and destruction.
//===----------------------------------------------------------------------===//

TEST(WriteCallbackModeTest, DisconnectReportsHeldMessage) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  port.connect(source.fetch(), source.error());
  source.push({1});
  source.push({2});
  port.rejectAll = true;
  port.notifyAsyncDataAvailable();

  port.disconnect();
  ASSERT_EQ(source.numErrors(), 1u);
  EXPECT_NE(errorMessage<std::runtime_error>(source[0]).find("disconnected"),
            std::string::npos);
  // Only the held message was fetched; the other stays with the source.
  EXPECT_EQ(source.queued(), 1u);
  EXPECT_FALSE(port.poll());

  // The port can be reconnected and used again.
  port.rejectAll = false;
  Source source2;
  port.connect(source2.fetch(), source2.error());
  source2.push({3});
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{3}}));
  EXPECT_EQ(source.numErrors(), 1u);
}

TEST(WriteCallbackModeTest, DisconnectWithNothingHeldReportsNothing) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  port.connect(source.fetch(), source.error());
  source.push({1});
  port.notifyAsyncDataAvailable();
  port.disconnect();
  EXPECT_EQ(source.numErrors(), 0u);
}

TEST(WriteCallbackModeTest, DisconnectReportsWindowedRemainder) {
  TwoFrameWindow types;
  Source source;
  RecordingWritePort port(&types.window);
  port.connect(source.fetch(), source.error(),
               ChannelPort::ConnectOptions(std::nullopt, true));
  source.push({1, 2, 3, 4});
  port.acceptBudget = 1;
  port.notifyAsyncDataAvailable();
  port.disconnect();
  ASSERT_EQ(source.numErrors(), 1u);
  EXPECT_EQ(port.written(), (Frames{{1, 2}}));
}

TEST(WriteCallbackModeTest, DisconnectFromFetchReportsFetchedMessage) {
  UIntType ui8("ui8", 8);
  Source source;
  RecordingWritePort port(&ui8);
  source.onFetch = [&](size_t) { port.disconnect(); };
  port.connect(source.fetch(), source.error());
  source.push({1});
  port.notifyAsyncDataAvailable();
  // The message returned after disconnecting isn't written, but reported.
  EXPECT_FALSE(port.isConnected());
  EXPECT_TRUE(port.written().empty());
  ASSERT_EQ(source.numErrors(), 1u);
  EXPECT_EQ(port.pollImplCount, 0);
}

TEST(WriteCallbackModeTest, ReconnectFromErrorCallback) {
  UIntType ui8("ui8", 8);
  Source oldSource, newSource;
  RecordingWritePort port(&ui8);
  // Recover from the disconnection by reconnecting with new callbacks.
  oldSource.onError = [&](size_t) {
    port.disconnect();
    port.connect(newSource.fetch(), newSource.error());
  };
  port.connect(oldSource.fetch(), oldSource.error());
  oldSource.push({1});
  port.rejectAll = true;
  port.notifyAsyncDataAvailable();

  port.disconnect();
  // The new connection survives the outer disconnect().
  EXPECT_EQ(oldSource.numErrors(), 1u);
  EXPECT_TRUE(port.isConnected());
  port.rejectAll = false;
  newSource.push({2});
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{2}}));
  EXPECT_EQ(newSource.numErrors(), 0u);
}

TEST(WriteCallbackModeTest, ConnectWaitsForWriterInCallback) {
  UIntType ui8("ui8", 8);
  Source oldSource, newSource;
  RecordingWritePort port(&ui8);
  std::atomic<bool> connected = false;
  std::thread other;
  // A callback disconnects the port, then another thread reconnects it while
  // the callback is still running. The reconnect must wait for the writer.
  oldSource.onFetch = [&](size_t idx) {
    if (idx != 0)
      return;
    port.disconnect();
    other = std::thread([&] {
      port.connect(newSource.fetch(), newSource.error());
      connected = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(connected);
  };
  port.connect(oldSource.fetch(), oldSource.error());
  oldSource.push({1});
  port.notifyAsyncDataAvailable();
  other.join();
  EXPECT_TRUE(connected);
  // The message fetched by the disconnected writer is reported, not written.
  EXPECT_EQ(oldSource.numErrors(), 1u);
  EXPECT_TRUE(port.written().empty());
  newSource.push({2});
  port.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{2}}));
}

TEST(WriteCallbackModeTest, DestructorReportsHeldMessage) {
  UIntType ui8("ui8", 8);
  Source source;
  {
    RecordingWritePort port(&ui8);
    port.connect(source.fetch(), source.error());
    source.push({1});
    port.rejectAll = true;
    port.notifyAsyncDataAvailable();
  }
  ASSERT_EQ(source.numErrors(), 1u);
  EXPECT_NE(errorMessage<std::runtime_error>(source[0]).find("destroyed"),
            std::string::npos);
}

TEST(WriteCallbackModeTest, DestructorCallbacksCannotUsePort) {
  UIntType ui8("ui8", 8);
  Source source;
  WriteChannelPort *port = nullptr;
  bool sawConnected = false;
  int rejected = 0;
  source.onError = [&](size_t) {
    // The derived port has already been destroyed, so nothing may reach the
    // backend, even after an explicit disconnect().
    sawConnected |= port->isConnected();
    port->disconnect();
    try {
      port->connect(source.fetch(), source.error());
    } catch (const std::runtime_error &) {
      ++rejected;
    }
    port->notifyAsyncDataAvailable();
    rejected += countRejectedSyncWrites(*port);
  };
  {
    RecordingWritePort recordingPort(&ui8);
    port = &recordingPort;
    recordingPort.connect(source.fetch(), source.error());
    source.push({1});
    recordingPort.rejectAll = true;
    recordingPort.notifyAsyncDataAvailable();
  }
  EXPECT_EQ(source.numErrors(), 1u);
  EXPECT_FALSE(sawConnected);
  EXPECT_EQ(rejected, 6);
}

//===----------------------------------------------------------------------===//
// Concurrency.
//===----------------------------------------------------------------------===//

TEST(WriteCallbackModeTest, ProducerThreadsWithServiceThread) {
  TwoFrameWindow types;
  Source source;
  RecordingWritePort port(&types.window);
  port.connect(source.fetch(), source.error(),
               ChannelPort::ConnectOptions(std::nullopt, true));
  // Backpressure every other frame, so messages are written both by the
  // producers' notifications and by the service thread's polls.
  port.alternate = true;

  AcceleratorServiceThread serviceThread;
  serviceThread.addTask([&port]() { port.poll(); });
  constexpr size_t numProducers = 4, numMsgs = 1000;
  std::vector<std::thread> producers;
  for (size_t p = 0; p < numProducers; ++p)
    producers.emplace_back([&, p] {
      for (size_t i = 0; i < numMsgs; ++i) {
        source.push({static_cast<uint8_t>(p), static_cast<uint8_t>(i),
                     static_cast<uint8_t>(i >> 8), 0});
        port.notifyAsyncDataAvailable();
      }
    });
  for (auto &producer : producers)
    producer.join();

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (port.written().size() < 2 * numProducers * numMsgs &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  serviceThread.stop();
  EXPECT_EQ(source.numErrors(), 0u);

  // Every message arrives intact, and each producer's in order.
  Frames received = receive(&types.window, port.written());
  ASSERT_EQ(received.size(), numProducers * numMsgs);
  std::vector<size_t> next(numProducers, 0);
  for (const auto &msg : received) {
    size_t p = msg[0], i = msg[1] | (msg[2] << 8);
    EXPECT_EQ(i, next[p]++);
  }
}

//===----------------------------------------------------------------------===//
// TypedWritePort callback mode.
//===----------------------------------------------------------------------===//

TEST(WriteCallbackModeTest, TypedWritePortValue) {
  UIntType ui24("ui24", 24);
  RecordingWritePort port(&ui24);
  TypedWritePort<uint32_t> typed(port);
  std::deque<uint32_t> values = {0x030201u, 0x060504u};
  typed.connect(
      [&]() -> std::unique_ptr<uint32_t> {
        if (values.empty())
          return nullptr;
        auto value = std::make_unique<uint32_t>(values.front());
        values.pop_front();
        return value;
      },
      [](std::exception_ptr) { ADD_FAILURE(); });
  typed.notifyAsyncDataAvailable();
  // Serialized to the 3-byte wire width.
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3}, {4, 5, 6}}));
}

TEST(WriteCallbackModeTest, TypedWritePortTypeCheck) {
  UIntType ui32("ui32", 32);
  Source source;
  RecordingWritePort port(&ui32);
  TypedWritePort<int32_t> typed(port); // int32_t expects SIntType
  EXPECT_THROW(
      typed.connect([] { return std::unique_ptr<int32_t>(); }, source.error()),
      AcceleratorMismatchError);
  EXPECT_FALSE(port.isConnected());
}

TEST(WriteCallbackModeTest, TypedWritePortSegmented) {
  BitsType bits40("i40", 40);
  RecordingWritePort port(&bits40);
  TypedWritePort<TwoSegmentMessage, /*SkipTypeCheck=*/true> typed(port);
  bool sent = false;
  typed.connect(
      [&]() -> std::unique_ptr<TwoSegmentMessage> {
        if (sent)
          return nullptr;
        sent = true;
        return std::make_unique<TwoSegmentMessage>();
      },
      [](std::exception_ptr) { ADD_FAILURE(); });
  typed.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3, 4, 5}}));
}

TEST(WriteCallbackModeTest, TypedWritePortVoid) {
  VoidType voidType("void");
  RecordingWritePort port(&voidType);
  TypedWritePort<void> typed(port);
  int remaining = 2;
  typed.connect([&] { return remaining-- > 0; },
                [](std::exception_ptr) { ADD_FAILURE(); });
  typed.notifyAsyncDataAvailable();
  EXPECT_EQ(port.written(), (Frames{{0}, {0}}));
}

} // namespace
