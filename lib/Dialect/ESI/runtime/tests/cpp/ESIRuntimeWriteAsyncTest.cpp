//===- ESIRuntimeWriteAsyncTest.cpp - WriteChannelPort::tryWriteAsync ---===//
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

/// Records the completions of tryWriteAsync() calls, in order.
struct Recorder {
  std::mutex mutex;
  std::vector<std::exception_ptr> results;
  /// Optional extra behavior, invoked with the result index.
  std::function<void(size_t)> onResult;

  WriteChannelPort::WriteCallback callback() {
    return [this](std::exception_ptr e) {
      size_t idx;
      {
        std::lock_guard<std::mutex> lock(mutex);
        idx = results.size();
        results.push_back(e);
      }
      if (onResult)
        onResult(idx);
    };
  }
  size_t size() {
    std::lock_guard<std::mutex> lock(mutex);
    return results.size();
  }
  std::exception_ptr operator[](size_t idx) {
    std::lock_guard<std::mutex> lock(mutex);
    return results.at(idx);
  }
  /// True if exactly `n` writes completed, all successfully.
  bool allSucceeded(size_t n) {
    std::lock_guard<std::mutex> lock(mutex);
    return results.size() == n &&
           std::all_of(results.begin(), results.end(),
                       [](std::exception_ptr e) { return !e; });
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

/// tryWriteAsync() a new message holding `bytes`.
bool tryAsync(WriteChannelPort &port, std::vector<uint8_t> bytes) {
  std::unique_ptr<SegmentedMessageData> msg = makeMsg(std::move(bytes));
  return port.tryWriteAsync(msg);
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

TEST(TryWriteAsyncTest, CompletesImmediately) {
  UIntType ui24("ui24", 24);
  RecordingWritePort port(&ui24);
  auto callerThread = std::this_thread::get_id();
  std::thread::id callbackThread;
  Recorder recorder;
  recorder.onResult = [&](size_t) {
    callbackThread = std::this_thread::get_id();
  };
  port.connect(recorder.callback());

  // The message is written, and the callback invoked on the calling thread,
  // before tryWriteAsync() returns.
  std::unique_ptr<SegmentedMessageData> msg = makeMsg({1, 2, 3});
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_FALSE(msg);
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(callbackThread, callerThread);
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3}}));

  // Nothing left for poll() to do, but the backend is still polled.
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(port.pollImplCount, 1);
  EXPECT_EQ(recorder.size(), 1u);
}

TEST(TryWriteAsyncTest, BackpressureReturnsFalseWithoutQueueing) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  port.connect(recorder.callback());
  port.rejectAll = true;

  // Nothing is queued: the caller keeps the message and retries it.
  std::unique_ptr<SegmentedMessageData> msg = makeMsg({1});
  SegmentedMessageData *original = msg.get();
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(port.tryWriteAsync(msg));
    EXPECT_EQ(msg.get(), original);
  }
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_TRUE(port.written().empty());

  port.rejectAll = false;
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (Frames{{1}}));
}

TEST(TryWriteAsyncTest, FailureThrowsAndKeepsMessage) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  port.connect(recorder.callback());
  port.throwAfter = 0;

  std::unique_ptr<SegmentedMessageData> msg = makeMsg({1});
  EXPECT_THROW(port.tryWriteAsync(msg), std::out_of_range);
  EXPECT_TRUE(msg);
  EXPECT_EQ(recorder.size(), 0u);

  // The port remains usable.
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (Frames{{1}}));
}

TEST(TryWriteAsyncTest, SegmentedMessageIsFlattened) {
  BitsType bits40("i40", 40);
  RecordingWritePort port(&bits40);
  Recorder recorder;
  port.connect(recorder.callback());

  std::unique_ptr<SegmentedMessageData> msg =
      std::make_unique<TwoSegmentMessage>();
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3, 4, 5}}));
}

TEST(TryWriteAsyncTest, UsesBackendSegmentedTryWrite) {
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
  ScatterGatherPort port(&bits40);
  Recorder recorder;
  port.connect(recorder.callback());

  std::unique_ptr<SegmentedMessageData> msg =
      std::make_unique<TwoSegmentMessage>();
  SegmentedMessageData *original = msg.get();
  port.rejectAll = true;
  EXPECT_FALSE(port.tryWriteAsync(msg));
  EXPECT_EQ(msg.get(), original);
  port.rejectAll = false;
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_TRUE(recorder.allSucceeded(1));
  // The original message was handed over without being flattened.
  ASSERT_EQ(port.received.size(), 1u);
  EXPECT_EQ(port.received[0].get(), original);
  EXPECT_TRUE(port.written().empty());
}

TEST(TryWriteAsyncTest, InvalidCallsThrow) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  EXPECT_THROW(tryAsync(port, {1}), std::runtime_error);
  EXPECT_THROW(port.connect(WriteChannelPort::WriteCallback()),
               std::invalid_argument);
  EXPECT_FALSE(port.isConnected());

  Recorder recorder;
  port.connect(recorder.callback());
  std::unique_ptr<SegmentedMessageData> null;
  EXPECT_THROW(port.tryWriteAsync(null), std::invalid_argument);
}

TEST(TryWriteAsyncTest, ModesAreExclusive) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  MessageData data(std::vector<uint8_t>{1});

  // Sync mode rejects tryWriteAsync() and connecting with a callback.
  port.connect();
  EXPECT_THROW(tryAsync(port, {1}), std::runtime_error);
  EXPECT_THROW(port.connect(recorder.callback()), std::runtime_error);
  port.write(data);
  EXPECT_TRUE(port.tryWrite(data));
  EXPECT_TRUE(port.flush());
  port.disconnect();

  // Callback mode rejects the sync APIs and reconnecting without a callback.
  port.connect(recorder.callback());
  EXPECT_THROW(port.write(data), std::runtime_error);
  EXPECT_THROW(port.write(makeMsg({1})), std::runtime_error);
  EXPECT_THROW(port.tryWrite(data), std::runtime_error);
  std::unique_ptr<SegmentedMessageData> segmented = makeMsg({1});
  EXPECT_THROW(port.tryWrite(segmented), std::runtime_error);
  EXPECT_TRUE(segmented);
  EXPECT_THROW(port.flush(), std::runtime_error);
  EXPECT_THROW(port.connect(), std::runtime_error);
  EXPECT_THROW(port.connect(recorder.callback()), std::runtime_error);
  port.disconnect();

  // After disconnecting, sync mode works again.
  port.connect();
  port.write(data);
  EXPECT_EQ(port.written().size(), 3u);
  EXPECT_EQ(recorder.size(), 0u);
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

TEST(TryWriteAsyncTest, SyncWritesThrowInCallbacks) {
  // A callback invoked by tryWriteAsync() itself.
  {
    UIntType ui16("ui16", 16);
    RecordingWritePort port(&ui16);
    int rejected = 0;
    Recorder recorder;
    recorder.onResult = [&](size_t) {
      rejected += countRejectedSyncWrites(port);
    };
    port.connect(recorder.callback());
    EXPECT_TRUE(tryAsync(port, {1, 2}));
    EXPECT_EQ(rejected, 5);
    EXPECT_EQ(port.written(), (Frames{{1, 2}}));
  }
  // A callback invoked by poll(), once a windowed message's remaining frames
  // are sent.
  {
    TwoFrameWindow types;
    RecordingWritePort port(&types.window);
    int rejected = 0;
    Recorder recorder;
    recorder.onResult = [&](size_t) {
      rejected += countRejectedSyncWrites(port);
    };
    port.connect(recorder.callback(),
                 ChannelPort::ConnectOptions(std::nullopt, true));
    port.acceptBudget = 1;
    EXPECT_TRUE(tryAsync(port, {1, 2, 3, 4}));
    port.acceptBudget = -1;
    port.poll();
    EXPECT_EQ(rejected, 5);
    EXPECT_TRUE(recorder.allSucceeded(1));
    EXPECT_EQ(port.written(), (Frames{{1, 2}, {3, 4}}));
  }
}

TEST(TryWriteAsyncTest, CallbackChainDoesNotRecurse) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  constexpr size_t numMsgs = 100000;
  size_t depth = 0, maxDepth = 0;
  Recorder recorder;
  recorder.onResult = [&](size_t idx) {
    ++depth;
    maxDepth = std::max(maxDepth, depth);
    // Write the next message upon completion of the previous one.
    if (idx + 1 < numMsgs)
      EXPECT_TRUE(tryAsync(port, {static_cast<uint8_t>(idx + 1)}));
    --depth;
  };
  port.connect(recorder.callback());

  EXPECT_TRUE(tryAsync(port, {0}));
  EXPECT_TRUE(recorder.allSucceeded(numMsgs));
  EXPECT_EQ(port.written().size(), numMsgs);
  EXPECT_EQ(maxDepth, 1u);
}

TEST(TryWriteAsyncTest, CallbackMayDisconnect) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  recorder.onResult = [&](size_t) { port.disconnect(); };
  port.connect(recorder.callback());

  EXPECT_TRUE(tryAsync(port, {1}));
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_FALSE(port.isConnected());
  EXPECT_THROW(tryAsync(port, {2}), std::runtime_error);
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(port.pollImplCount, 0);
}

TEST(TryWriteAsyncTest, ThrowingCallbackIsContained) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  recorder.onResult = [](size_t idx) {
    if (idx == 0)
      throw std::runtime_error("callback failure");
  };
  port.connect(recorder.callback());

  EXPECT_NO_THROW(tryAsync(port, {1}));
  EXPECT_TRUE(tryAsync(port, {2}));
  EXPECT_TRUE(recorder.allSucceeded(2));
  EXPECT_EQ(port.written().size(), 2u);
}

TEST(TryWriteAsyncTest, VoidTypeIsPadded) {
  VoidType voidType("void");
  RecordingWritePort port(&voidType);
  Recorder recorder;
  port.connect(recorder.callback());

  std::unique_ptr<SegmentedMessageData> msg = std::make_unique<MessageData>();
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (Frames{{0}}));
}

//===----------------------------------------------------------------------===//
// Windowed messages, whose frames may be sent across several calls.
//===----------------------------------------------------------------------===//

TEST(TryWriteAsyncTest, WindowedMessageCompletesAfterAllFrames) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  port.connect(recorder.callback(),
               ChannelPort::ConnectOptions(std::nullopt, true));

  // The first frame is accepted, so the message is, but the second isn't.
  port.acceptBudget = 1;
  std::unique_ptr<SegmentedMessageData> msg = makeMsg({1, 2, 3, 4});
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_FALSE(msg);
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_EQ(port.written().size(), 1u);
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(recorder.size(), 0u);

  // The callback is invoked once poll() sends the last frame.
  port.acceptBudget = -1;
  EXPECT_TRUE(port.poll());
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(receive(&types.window, port.written()), (Frames{{1, 2, 3, 4}}));
}

TEST(TryWriteAsyncTest, WindowedPendingFramesBlockNextMessage) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  std::vector<size_t> framesWrittenAtCompletion;
  Recorder recorder;
  recorder.onResult = [&](size_t) {
    framesWrittenAtCompletion.push_back(port.written().size());
  };
  port.connect(recorder.callback(),
               ChannelPort::ConnectOptions(std::nullopt, true));

  port.acceptBudget = 1;
  EXPECT_TRUE(tryAsync(port, {1, 2, 3, 4}));
  // While the first message has frames left, the next isn't accepted.
  std::unique_ptr<SegmentedMessageData> msg = makeMsg({5, 6, 7, 8});
  EXPECT_FALSE(port.tryWriteAsync(msg));
  EXPECT_TRUE(msg);
  EXPECT_EQ(recorder.size(), 0u);

  // Once the backend is ready, tryWriteAsync() finishes the first message
  // (invoking its callback) before writing the second.
  port.acceptBudget = -1;
  EXPECT_TRUE(port.tryWriteAsync(msg));
  EXPECT_TRUE(recorder.allSucceeded(2));
  EXPECT_EQ(framesWrittenAtCompletion, (std::vector<size_t>{2, 4}));
  EXPECT_EQ(receive(&types.window, port.written()),
            (Frames{{1, 2, 3, 4}, {5, 6, 7, 8}}));
}

TEST(TryWriteAsyncTest, PollReportsProgressOnRemainingFrames) {
  ThreeFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  port.connect(recorder.callback(),
               ChannelPort::ConnectOptions(std::nullopt, true));

  port.acceptBudget = 1;
  EXPECT_TRUE(tryAsync(port, {1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(port.written().size(), 1u);

  // Sending the second frame is progress, even though the third is stuck.
  port.acceptBudget = 1;
  EXPECT_TRUE(port.poll());
  EXPECT_EQ(port.written().size(), 2u);
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_FALSE(port.poll());

  port.acceptBudget = -1;
  EXPECT_TRUE(port.poll());
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(receive(&types.window, port.written()),
            (Frames{{1, 2, 3, 4, 5, 6}}));
}

TEST(TryWriteAsyncTest, WindowedFrameFailureReportedViaCallback) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  port.connect(recorder.callback(),
               ChannelPort::ConnectOptions(std::nullopt, true));

  port.acceptBudget = 1;
  EXPECT_TRUE(tryAsync(port, {1, 2, 3, 4}));
  // Sending the accepted message's last frame fails.
  port.acceptBudget = -1;
  port.throwAfter = 0;
  EXPECT_TRUE(port.poll());
  ASSERT_EQ(recorder.size(), 1u);
  EXPECT_EQ(errorMessage<std::out_of_range>(recorder[0]), "injected failure");

  // The port remains usable.
  EXPECT_TRUE(tryAsync(port, {5, 6, 7, 8}));
  ASSERT_EQ(recorder.size(), 2u);
  EXPECT_FALSE(recorder[1]);
  EXPECT_EQ(port.written(), (Frames{{1, 2}, {5, 6}, {7, 8}}));
}

TEST(TryWriteAsyncTest, WindowedFailureInTryWriteAsyncThrows) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  port.connect(recorder.callback(),
               ChannelPort::ConnectOptions(std::nullopt, true));

  // The first frame is sent, then the second fails. As with tryWrite(), the
  // error is thrown and the caller keeps the message.
  port.throwAfter = 1;
  std::unique_ptr<SegmentedMessageData> msg = makeMsg({1, 2, 3, 4});
  EXPECT_THROW(port.tryWriteAsync(msg), std::out_of_range);
  EXPECT_TRUE(msg);
  EXPECT_EQ(recorder.size(), 0u);
  // Its remaining frame was dropped rather than left for poll() to send.
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_EQ(port.written(), (Frames{{1, 2}}));

  EXPECT_TRUE(tryAsync(port, {5, 6, 7, 8}));
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (Frames{{1, 2}, {5, 6}, {7, 8}}));
}

TEST(TryWriteAsyncTest, DisconnectFailsPendingWrite) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  ChannelPort::ConnectOptions opts(std::nullopt, true);
  port.connect(recorder.callback(), opts);

  port.acceptBudget = 1;
  EXPECT_TRUE(tryAsync(port, {1, 2, 3, 4}));
  port.disconnect();
  ASSERT_EQ(recorder.size(), 1u);
  EXPECT_NE(errorMessage<std::runtime_error>(recorder[0]).find("disconnected"),
            std::string::npos);

  // The port can be reconnected with a new callback and used again.
  port.acceptBudget = -1;
  Recorder recorder2;
  port.connect(recorder2.callback(), opts);
  EXPECT_TRUE(tryAsync(port, {5, 6, 7, 8}));
  EXPECT_TRUE(recorder2.allSucceeded(1));
  EXPECT_EQ(recorder.size(), 1u);
  EXPECT_EQ(port.written(), (Frames{{1, 2}, {5, 6}, {7, 8}}));
}

TEST(TryWriteAsyncTest, ReconnectFromCompletionCallback) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  ChannelPort::ConnectOptions opts(std::nullopt, true);
  Recorder oldRecorder, newRecorder;
  // Recover from the cancellation by reconnecting with a new callback.
  oldRecorder.onResult = [&](size_t) {
    port.disconnect();
    port.connect(newRecorder.callback(), opts);
  };
  port.connect(oldRecorder.callback(), opts);
  port.acceptBudget = 1;
  EXPECT_TRUE(tryAsync(port, {1, 2, 3, 4}));

  port.disconnect();
  // The cancellation is reported to the old callback, and the new connection
  // survives the outer disconnect().
  ASSERT_EQ(oldRecorder.size(), 1u);
  EXPECT_TRUE(oldRecorder[0]);
  EXPECT_TRUE(port.isConnected());
  port.acceptBudget = -1;
  EXPECT_TRUE(tryAsync(port, {5, 6, 7, 8}));
  EXPECT_TRUE(newRecorder.allSucceeded(1));
  EXPECT_EQ(oldRecorder.size(), 1u);
}

TEST(TryWriteAsyncTest, DestructorFailsPendingWrite) {
  TwoFrameWindow types;
  Recorder recorder;
  {
    RecordingWritePort port(&types.window);
    port.connect(recorder.callback(),
                 ChannelPort::ConnectOptions(std::nullopt, true));
    port.acceptBudget = 1;
    EXPECT_TRUE(tryAsync(port, {1, 2, 3, 4}));
  }
  ASSERT_EQ(recorder.size(), 1u);
  EXPECT_NE(errorMessage<std::runtime_error>(recorder[0]).find("destroyed"),
            std::string::npos);
}

TEST(TryWriteAsyncTest, DestructorCallbacksCannotUsePort) {
  TwoFrameWindow types;
  WriteChannelPort *port = nullptr;
  bool sawConnected = false;
  int rejected = 0;
  Recorder recorder;
  recorder.onResult = [&](size_t) {
    // The derived port has already been destroyed, so nothing may reach the
    // backend, even after an explicit disconnect().
    sawConnected |= port->isConnected();
    port->disconnect();
    try {
      port->connect([](std::exception_ptr) {});
    } catch (const std::runtime_error &) {
      ++rejected;
    }
    try {
      tryAsync(*port, {5, 6, 7, 8});
    } catch (const std::runtime_error &) {
      ++rejected;
    }
    rejected += countRejectedSyncWrites(*port);
  };
  {
    RecordingWritePort recordingPort(&types.window);
    port = &recordingPort;
    recordingPort.connect(recorder.callback(),
                          ChannelPort::ConnectOptions(std::nullopt, true));
    recordingPort.acceptBudget = 1;
    EXPECT_TRUE(tryAsync(recordingPort, {1, 2, 3, 4}));
  }
  EXPECT_EQ(recorder.size(), 1u);
  EXPECT_FALSE(sawConnected);
  EXPECT_EQ(rejected, 7);
}

TEST(TryWriteAsyncTest, DrivenByServiceThread) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  constexpr size_t numMsgs = 1000;
  std::promise<void> allDone;
  std::atomic<size_t> completed = 0;
  port.connect(
      [&](std::exception_ptr e) {
        EXPECT_FALSE(e);
        if (++completed == numMsgs)
          allDone.set_value();
      },
      ChannelPort::ConnectOptions(std::nullopt, true));
  // Backpressure every other frame, so messages are completed both by
  // tryWriteAsync() on this thread and by poll() on the service thread.
  port.alternate = true;

  AcceleratorServiceThread serviceThread;
  serviceThread.addTask([&port]() { port.poll(); });
  Frames expected;
  for (size_t i = 0; i < numMsgs; ++i) {
    std::vector<uint8_t> bytes = {static_cast<uint8_t>(i), 1,
                                  static_cast<uint8_t>(i >> 8), 2};
    expected.push_back(bytes);
    std::unique_ptr<SegmentedMessageData> msg = makeMsg(bytes);
    while (!port.tryWriteAsync(msg))
      std::this_thread::yield();
  }

  auto future = allDone.get_future();
  ASSERT_EQ(future.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  serviceThread.stop();
  EXPECT_EQ(receive(&types.window, port.written()), expected);
}

//===----------------------------------------------------------------------===//
// TypedWritePort::tryWriteAsync
//===----------------------------------------------------------------------===//

TEST(TryWriteAsyncTest, TypedWritePortValue) {
  UIntType ui24("ui24", 24);
  RecordingWritePort port(&ui24);
  TypedWritePort<uint32_t> typed(port);
  Recorder recorder;
  typed.connect(recorder.callback());

  EXPECT_TRUE(typed.tryWriteAsync(0x030201u));
  auto ptr = std::make_unique<uint32_t>(0x060504u);
  port.rejectAll = true;
  EXPECT_FALSE(typed.tryWriteAsync(ptr));
  EXPECT_TRUE(ptr);
  port.rejectAll = false;
  EXPECT_TRUE(typed.tryWriteAsync(ptr));
  EXPECT_FALSE(ptr);
  EXPECT_TRUE(recorder.allSucceeded(2));
  // Serialized to the 3-byte wire width.
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3}, {4, 5, 6}}));
}

TEST(TryWriteAsyncTest, TypedWritePortTypeCheck) {
  UIntType ui32("ui32", 32);
  RecordingWritePort port(&ui32);
  TypedWritePort<int32_t> typed(port); // int32_t expects SIntType
  Recorder recorder;
  EXPECT_THROW(typed.connect(recorder.callback()), AcceleratorMismatchError);
  EXPECT_FALSE(port.isConnected());
}

TEST(TryWriteAsyncTest, TypedWritePortSegmented) {
  BitsType bits40("i40", 40);
  RecordingWritePort port(&bits40);
  TypedWritePort<TwoSegmentMessage, /*SkipTypeCheck=*/true> typed(port);
  Recorder recorder;
  typed.connect(recorder.callback());

  auto msg = std::make_unique<TwoSegmentMessage>();
  TwoSegmentMessage *original = msg.get();
  port.rejectAll = true;
  EXPECT_FALSE(typed.tryWriteAsync(msg));
  EXPECT_EQ(msg.get(), original);
  port.throwAfter = 0;
  port.rejectAll = false;
  EXPECT_THROW(typed.tryWriteAsync(msg), std::out_of_range);
  EXPECT_EQ(msg.get(), original);
  EXPECT_TRUE(typed.tryWriteAsync(msg));
  EXPECT_FALSE(msg);
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (Frames{{1, 2, 3, 4, 5}}));
}

TEST(TryWriteAsyncTest, TypedWritePortVoid) {
  VoidType voidType("void");
  RecordingWritePort port(&voidType);
  TypedWritePort<void> typed(port);
  Recorder recorder;
  typed.connect(recorder.callback());

  EXPECT_TRUE(typed.tryWriteAsync());
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (Frames{{0}}));
}

} // namespace
