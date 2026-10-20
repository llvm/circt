//===- ESIRuntimeWriteAsyncTest.cpp - WriteChannelPort::writeAsync tests --===//
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
  /// Number of pollImpl() calls.
  std::atomic<int> pollImplCount = 0;

protected:
  void writeImpl(const MessageData &data) override {
    while (!tryWriteImpl(data))
      std::this_thread::yield();
  }
  bool tryWriteImpl(const MessageData &data) override {
    if (rejectAll || acceptBudget == 0)
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
  bool pollImpl() override {
    ++pollImplCount;
    return false;
  }

private:
  std::mutex mutex;
  std::vector<std::vector<uint8_t>> writtenMsgs;
};

/// Records the results of writeAsync() calls, in completion order.
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

std::unique_ptr<MessageData> makeMsg(std::vector<uint8_t> bytes) {
  return std::make_unique<MessageData>(std::move(bytes));
}

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

TEST(WriteAsyncTest, WritesImmediatelyWhenIdle) {
  UIntType ui24("ui24", 24);
  RecordingWritePort port(&ui24);
  auto callerThread = std::this_thread::get_id();
  std::thread::id callbackThread;
  Recorder recorder;
  recorder.onResult = [&](size_t) {
    callbackThread = std::this_thread::get_id();
  };
  port.connect(recorder.callback());

  // With nothing queued, the message is written (and the callback invoked) on
  // the calling thread before writeAsync() returns.
  port.writeAsync(makeMsg({1, 2, 3}));
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(callbackThread, callerThread);
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{1, 2, 3}}));

  // Nothing left for poll() to do, but the backend is still polled.
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(port.pollImplCount, 1);
  EXPECT_EQ(recorder.size(), 1u);
}

TEST(WriteAsyncTest, BackpressuredWriteCompletesOnPoll) {
  UIntType ui24("ui24", 24);
  RecordingWritePort port(&ui24);
  Recorder recorder;
  port.connect(recorder.callback());
  port.rejectAll = true;

  port.writeAsync(makeMsg({1, 2, 3}));
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_TRUE(port.written().empty());
  EXPECT_FALSE(port.poll());

  port.rejectAll = false;
  EXPECT_TRUE(port.poll());
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{1, 2, 3}}));
}

TEST(WriteAsyncTest, FastPathPreservesOrder) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  port.connect(recorder.callback());

  // {1} is backpressured and stays in flight, so {2} must not bypass it even
  // once the backend is ready again.
  port.rejectCount = 1;
  port.writeAsync(makeMsg({1}));
  port.writeAsync(makeMsg({2}));
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_TRUE(port.written().empty());

  port.poll();
  EXPECT_TRUE(recorder.allSucceeded(2));
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{1}, {2}}));

  // Idle again: the fast path is used.
  port.writeAsync(makeMsg({3}));
  EXPECT_TRUE(recorder.allSucceeded(3));
}

TEST(WriteAsyncTest, CallbackChainDoesNotRecurse) {
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
      port.writeAsync(makeMsg({static_cast<uint8_t>(idx + 1)}));
    --depth;
  };
  port.connect(recorder.callback());

  port.writeAsync(makeMsg({0}));
  EXPECT_TRUE(recorder.allSucceeded(numMsgs));
  EXPECT_EQ(port.written().size(), numMsgs);
  EXPECT_EQ(maxDepth, 1u);
}

TEST(WriteAsyncTest, SegmentedMessageIsFlattened) {
  BitsType bits40("i40", 40);
  RecordingWritePort port(&bits40);
  Recorder recorder;
  port.connect(recorder.callback());

  port.writeAsync(std::make_unique<TwoSegmentMessage>());
  port.poll();
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(),
            (std::vector<std::vector<uint8_t>>{{1, 2, 3, 4, 5}}));
}

TEST(WriteAsyncTest, FifoOrderAndBackpressure) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  port.connect(recorder.callback());

  // While backpressured, nothing completes.
  port.rejectAll = true;
  for (uint8_t i = 0; i < 8; ++i)
    port.writeAsync(makeMsg({i}));
  EXPECT_FALSE(port.poll());
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(recorder.size(), 0u);

  port.rejectAll = false;
  port.rejectCount = 5;
  for (int i = 0; i < 20; ++i)
    port.poll();
  EXPECT_TRUE(recorder.allSucceeded(8));
  auto written = port.written();
  ASSERT_EQ(written.size(), 8u);
  for (uint8_t i = 0; i < 8; ++i)
    EXPECT_EQ(written[i], std::vector<uint8_t>{i});
}

TEST(WriteAsyncTest, FailurePassesException) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  port.connect(recorder.callback());
  port.throwAfter = 0;

  port.writeAsync(makeMsg({1}));
  port.writeAsync(makeMsg({2}));
  port.poll();

  ASSERT_EQ(recorder.size(), 2u);
  EXPECT_EQ(errorMessage<std::out_of_range>(recorder[0]), "injected failure");
  // Subsequent writes are unaffected.
  EXPECT_FALSE(recorder[1]);
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{2}}));
}

TEST(WriteAsyncTest, InvalidCallsThrow) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  EXPECT_THROW(port.writeAsync(makeMsg({1})), std::runtime_error);
  EXPECT_THROW(port.connect(WriteChannelPort::WriteCallback()),
               std::invalid_argument);
  EXPECT_FALSE(port.isConnected());

  Recorder recorder;
  port.connect(recorder.callback());
  EXPECT_THROW(port.writeAsync(nullptr), std::invalid_argument);
}

TEST(WriteAsyncTest, ModesAreExclusive) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  MessageData data(std::vector<uint8_t>{1});

  // Sync mode rejects writeAsync() and connecting with a callback.
  port.connect();
  EXPECT_THROW(port.writeAsync(makeMsg({1})), std::runtime_error);
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

TEST(WriteAsyncTest, DisconnectFailsPendingWrites) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  port.connect(recorder.callback());
  port.rejectAll = true;

  for (uint8_t i = 0; i < 3; ++i)
    port.writeAsync(makeMsg({i}));
  port.poll();

  port.disconnect();
  ASSERT_EQ(recorder.size(), 3u);
  for (size_t i = 0; i < 3; ++i)
    EXPECT_NE(
        errorMessage<std::runtime_error>(recorder[i]).find("disconnected"),
        std::string::npos);
  EXPECT_TRUE(port.written().empty());

  // The port can be reconnected with a new callback and used again.
  port.rejectAll = false;
  Recorder recorder2;
  port.connect(recorder2.callback());
  port.writeAsync(makeMsg({4}));
  port.poll();
  EXPECT_TRUE(recorder2.allSucceeded(1));
  EXPECT_EQ(recorder.size(), 3u);
  EXPECT_EQ(port.written().size(), 1u);
}

TEST(WriteAsyncTest, DestructorFailsPendingWrites) {
  UIntType ui8("ui8", 8);
  Recorder recorder;
  {
    RecordingWritePort port(&ui8);
    port.connect(recorder.callback());
    port.rejectAll = true;
    port.writeAsync(makeMsg({1}));
  }
  ASSERT_EQ(recorder.size(), 1u);
  EXPECT_NE(errorMessage<std::runtime_error>(recorder[0]).find("destroyed"),
            std::string::npos);
}

TEST(WriteAsyncTest, DestructorCallbacksCannotStartWrites) {
  UIntType ui8("ui8", 8);
  WriteChannelPort *port = nullptr;
  bool sawConnected = false;
  int retriesRejected = 0;
  Recorder recorder;
  recorder.onResult = [&](size_t) {
    // The derived port has already been destroyed, so a retry must not reach
    // the backend.
    sawConnected |= port->isConnected();
    try {
      port->writeAsync(makeMsg({2}));
    } catch (const std::runtime_error &) {
      ++retriesRejected;
    }
  };
  {
    RecordingWritePort recordingPort(&ui8);
    port = &recordingPort;
    recordingPort.connect(recorder.callback());
    recordingPort.rejectAll = true;
    recordingPort.writeAsync(makeMsg({1}));
  }
  EXPECT_EQ(recorder.size(), 1u);
  EXPECT_FALSE(sawConnected);
  EXPECT_EQ(retriesRejected, 1);
}

TEST(WriteAsyncTest, DisconnectFromCancellationCallbackReportsAll) {
  UIntType ui8("ui8", 8);
  for (bool viaDestructor : {false, true}) {
    SCOPED_TRACE(viaDestructor ? "destructor" : "disconnect");
    WriteChannelPort *port = nullptr;
    Recorder recorder;
    // Disconnecting from the first cancellation callback must not suppress
    // the rest.
    recorder.onResult = [&](size_t) { port->disconnect(); };
    {
      RecordingWritePort recordingPort(&ui8);
      port = &recordingPort;
      recordingPort.connect(recorder.callback());
      recordingPort.rejectAll = true;
      for (uint8_t i = 0; i < 3; ++i)
        recordingPort.writeAsync(makeMsg({i}));
      if (!viaDestructor)
        recordingPort.disconnect();
    }
    ASSERT_EQ(recorder.size(), 3u);
    for (size_t i = 0; i < 3; ++i)
      EXPECT_TRUE(recorder[i]);
  }
}

TEST(WriteAsyncTest, CallbackMayWriteAndDisconnect) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  recorder.onResult = [&](size_t idx) {
    if (idx == 0)
      port.writeAsync(makeMsg({2}));
    // Disconnecting from a callback fails the remaining writes without
    // deadlocking.
    else if (idx == 1)
      port.disconnect();
  };
  port.connect(recorder.callback());

  port.rejectAll = true;
  port.writeAsync(makeMsg({1}));
  port.writeAsync(makeMsg({3}));
  port.writeAsync(makeMsg({4}));
  port.rejectAll = false;
  port.poll();

  EXPECT_FALSE(port.isConnected());
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{1}, {3}}));
  ASSERT_EQ(recorder.size(), 4u);
  EXPECT_FALSE(recorder[0]);
  EXPECT_FALSE(recorder[1]);
  EXPECT_TRUE(recorder[2]);
  EXPECT_TRUE(recorder[3]);
  // pollImpl() isn't called on a port which was disconnected.
  EXPECT_EQ(port.pollImplCount, 0);
}

TEST(WriteAsyncTest, ThrowingCallbackIsContained) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  Recorder recorder;
  recorder.onResult = [](size_t idx) {
    if (idx == 0)
      throw std::runtime_error("callback failure");
  };
  port.connect(recorder.callback());

  port.writeAsync(makeMsg({1}));
  port.writeAsync(makeMsg({2}));
  EXPECT_NO_THROW(port.poll());
  EXPECT_TRUE(recorder.allSucceeded(2));
  EXPECT_EQ(port.written().size(), 2u);
}

TEST(WriteAsyncTest, VoidTypeIsPadded) {
  VoidType voidType("void");
  RecordingWritePort port(&voidType);
  Recorder recorder;
  port.connect(recorder.callback());

  port.writeAsync(std::make_unique<MessageData>());
  port.poll();
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{0}}));
}

TEST(WriteAsyncTest, WindowedMessageCompletesAfterAllFrames) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  port.connect(recorder.callback(),
               ChannelPort::ConnectOptions(std::nullopt, true));

  // Accept the first frame, then backpressure.
  port.acceptBudget = 1;
  port.writeAsync(makeMsg({1, 2, 3, 4}));
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_EQ(port.written().size(), 1u);
  EXPECT_FALSE(port.poll());
  EXPECT_EQ(recorder.size(), 0u);

  // The callback fires only once the last frame is sent.
  port.acceptBudget = -1;
  EXPECT_TRUE(port.poll());
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(),
            (std::vector<std::vector<uint8_t>>{{1, 2}, {3, 4}}));
  EXPECT_EQ(receive(&types.window, port.written()),
            (std::vector<std::vector<uint8_t>>{{1, 2, 3, 4}}));
}

TEST(WriteAsyncTest, WindowedFailureBeforeAnyFrameIsRecoverable) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  port.connect(recorder.callback(),
               ChannelPort::ConnectOptions(std::nullopt, true));
  port.throwAfter = 0;

  // The first frame fails, so nothing of the message reached the receiver and
  // the next message can be sent.
  port.writeAsync(makeMsg({1, 2, 3, 4}));
  port.writeAsync(makeMsg({5, 6, 7, 8}));
  ASSERT_EQ(recorder.size(), 2u);
  EXPECT_EQ(errorMessage<std::out_of_range>(recorder[0]), "injected failure");
  EXPECT_FALSE(recorder[1]);
  EXPECT_EQ(receive(&types.window, port.written()),
            (std::vector<std::vector<uint8_t>>{{5, 6, 7, 8}}));
}

TEST(WriteAsyncTest, WindowedFailureMidMessageFaultsStream) {
  TwoFrameWindow types;
  RecordingWritePort port(&types.window);
  Recorder recorder;
  ChannelPort::ConnectOptions opts(std::nullopt, true);
  port.connect(recorder.callback(), opts);
  // The first frame is sent, then the second fails.
  port.throwAfter = 1;
  port.rejectAll = true;
  port.writeAsync(makeMsg({1, 2, 3, 4}));
  port.writeAsync(makeMsg({5, 6, 7, 8}));
  port.rejectAll = false;
  port.poll();

  ASSERT_EQ(recorder.size(), 2u);
  EXPECT_EQ(errorMessage<std::out_of_range>(recorder[0]), "injected failure");
  // Sending {5, 6, 7, 8} would complete the receiver's partial message with
  // the wrong data, so it fails instead.
  EXPECT_NE(
      errorMessage<std::runtime_error>(recorder[1]).find("framing is corrupt"),
      std::string::npos);
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{1, 2}}));
  EXPECT_TRUE(receive(&types.window, port.written()).empty());

  // Later writes fail too (promptly, via the fast path).
  port.writeAsync(makeMsg({9, 10, 11, 12}));
  ASSERT_EQ(recorder.size(), 3u);
  EXPECT_TRUE(recorder[2]);
  EXPECT_EQ(port.written().size(), 1u);

  // Reconnecting clears the fault.
  port.disconnect();
  port.connect(recorder.callback(), opts);
  port.writeAsync(makeMsg({5, 6, 7, 8}));
  ASSERT_EQ(recorder.size(), 4u);
  EXPECT_FALSE(recorder[3]);
  EXPECT_EQ(port.written(),
            (std::vector<std::vector<uint8_t>>{{1, 2}, {5, 6}, {7, 8}}));
}

TEST(WriteAsyncTest, DrivenByServiceThread) {
  UIntType ui8("ui8", 8);
  RecordingWritePort port(&ui8);
  constexpr size_t numMsgs = 100;
  std::promise<void> allDone;
  std::atomic<size_t> completed = 0;
  std::atomic<bool> onCallerThread = false;
  auto callerThread = std::this_thread::get_id();
  port.connect([&](std::exception_ptr e) {
    EXPECT_FALSE(e);
    if (std::this_thread::get_id() == callerThread)
      onCallerThread = true;
    if (++completed == numMsgs)
      allDone.set_value();
  });
  // Backpressure while queueing so that all the writes are completed by the
  // service thread.
  port.rejectAll = true;

  AcceleratorServiceThread serviceThread;
  serviceThread.addTask([&port]() { port.poll(); });
  for (size_t i = 0; i < numMsgs; ++i)
    port.writeAsync(makeMsg({static_cast<uint8_t>(i)}));
  port.rejectCount = 50;
  port.rejectAll = false;

  auto future = allDone.get_future();
  ASSERT_EQ(future.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  serviceThread.stop();
  // Callbacks run on the service thread.
  EXPECT_FALSE(onCallerThread);

  auto written = port.written();
  ASSERT_EQ(written.size(), numMsgs);
  for (size_t i = 0; i < numMsgs; ++i)
    EXPECT_EQ(written[i], std::vector<uint8_t>{static_cast<uint8_t>(i)});
}

TEST(WriteAsyncTest, FastPathRacesWithServiceThread) {
  UIntType ui16("ui16", 16);
  RecordingWritePort port(&ui16);
  constexpr size_t numMsgs = 5000;
  std::promise<void> allDone;
  std::atomic<size_t> completed = 0;
  port.connect([&](std::exception_ptr e) {
    EXPECT_FALSE(e);
    if (++completed == numMsgs)
      allDone.set_value();
  });
  // Intermittent backpressure so that writes alternate between the fast path
  // and being completed by the service thread.
  port.rejectCount = 1000;

  AcceleratorServiceThread serviceThread;
  serviceThread.addTask([&port]() { port.poll(); });
  for (size_t i = 0; i < numMsgs; ++i) {
    port.writeAsync(makeMsg(
        {static_cast<uint8_t>(i & 0xFF), static_cast<uint8_t>(i >> 8)}));
    if (i % 500 == 0)
      port.rejectCount = 3;
  }

  auto future = allDone.get_future();
  ASSERT_EQ(future.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  serviceThread.stop();

  auto written = port.written();
  ASSERT_EQ(written.size(), numMsgs);
  for (size_t i = 0; i < numMsgs; ++i)
    EXPECT_EQ(written[i], (std::vector<uint8_t>{static_cast<uint8_t>(i & 0xFF),
                                                static_cast<uint8_t>(i >> 8)}));
}

//===----------------------------------------------------------------------===//
// TypedWritePort::writeAsync
//===----------------------------------------------------------------------===//

TEST(WriteAsyncTest, TypedWritePortValue) {
  UIntType ui24("ui24", 24);
  RecordingWritePort port(&ui24);
  TypedWritePort<uint32_t> typed(port);
  Recorder recorder;
  typed.connect(recorder.callback());

  typed.writeAsync(0x030201u);
  typed.writeAsync(std::make_unique<uint32_t>(0x060504u));
  port.poll();
  EXPECT_TRUE(recorder.allSucceeded(2));
  // Serialized to the 3-byte wire width.
  EXPECT_EQ(port.written(),
            (std::vector<std::vector<uint8_t>>{{1, 2, 3}, {4, 5, 6}}));
}

TEST(WriteAsyncTest, TypedWritePortTypeCheck) {
  UIntType ui32("ui32", 32);
  RecordingWritePort port(&ui32);
  TypedWritePort<int32_t> typed(port); // int32_t expects SIntType
  Recorder recorder;
  EXPECT_THROW(typed.connect(recorder.callback()), AcceleratorMismatchError);
  EXPECT_FALSE(port.isConnected());
}

TEST(WriteAsyncTest, TypedWritePortSegmented) {
  BitsType bits40("i40", 40);
  RecordingWritePort port(&bits40);
  TypedWritePort<TwoSegmentMessage, /*SkipTypeCheck=*/true> typed(port);
  Recorder recorder;
  typed.connect(recorder.callback());

  typed.writeAsync(std::make_unique<TwoSegmentMessage>());
  port.poll();
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(),
            (std::vector<std::vector<uint8_t>>{{1, 2, 3, 4, 5}}));
}

TEST(WriteAsyncTest, TypedWritePortVoid) {
  VoidType voidType("void");
  RecordingWritePort port(&voidType);
  TypedWritePort<void> typed(port);
  Recorder recorder;
  typed.connect(recorder.callback());

  typed.writeAsync();
  port.poll();
  EXPECT_TRUE(recorder.allSucceeded(1));
  EXPECT_EQ(port.written(), (std::vector<std::vector<uint8_t>>{{0}}));
}

} // namespace
