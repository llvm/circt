//===- ESIRuntimeCosimRpcTest.cpp - Cosim RPC protocol tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs the cosim RpcServer and RpcClient in-process (no simulator) to check
// the protocol's behavior.
//
//===----------------------------------------------------------------------===//

#include "esi/Context.h"
#include "esi/Logging.h"
#include "esi/Ports.h"
#include "esi/backends/RpcClient.h"
#include "esi/backends/RpcServer.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace esi;
using esi::backends::cosim::RpcClient;
using esi::cosim::RpcServer;
using namespace std::chrono_literals;

namespace {

MessageData msg(uint32_t v) { return MessageData::from(v); }

/// Poll `cond` until it holds or a generous deadline passes.
bool waitFor(const std::function<bool()> &cond,
             std::chrono::milliseconds timeout = 10s) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!cond()) {
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

class CosimRpcTest : public ::testing::Test {
protected:
  void SetUp() override {
    // RpcServer writes `cosim.cfg` into the working directory.
    oldCwd = std::filesystem::current_path();
    tmpDir = std::filesystem::temp_directory_path() /
             ("esi_cosim_rpc_test_" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(tmpDir);
    std::filesystem::current_path(tmpDir);

    ctxt = Context::withLogger<NullLogger>();
    server = std::make_unique<RpcServer>(*ctxt);
    // The server picks a free port by probing, then binds it; another process
    // can take it in between. Retry, since tests run in parallel.
    for (int attempt = 0;; ++attempt) {
      try {
        server->run(0);
        break;
      } catch (const std::runtime_error &) {
        if (attempt == 4)
          throw;
      }
    }
    toServer = &server->registerReadPort("in", "i32");
    // Registered, but left for each test to connect (or not).
    lateToServer = &server->registerReadPort("late", "i32");
    toClient = &server->registerWritePort("out", "i32");
    toServer->connect();
    toClient->connect();
    // Non-empty: IXWebSocket's base64 encoder crashes on empty input.
    server->setManifest(0, {1, 2, 3, 4});
  }

  void TearDown() override {
    client.reset();
    if (server)
      server->stop();
    server.reset();
    std::filesystem::current_path(oldCwd);
    std::error_code ec;
    std::filesystem::remove_all(tmpDir, ec);
  }

  void connectClient() {
    client = std::make_unique<RpcClient>(ctxt->getLogger(), "127.0.0.1",
                                         server->getPort());
  }

  /// Read one message from the to-server port as the simulator would, and
  /// return its credit.
  uint32_t consumeOne() { return consumeFrom(*toServer); }

  std::filesystem::path oldCwd, tmpDir;
  std::unique_ptr<Context> ctxt;
  std::unique_ptr<RpcServer> server;
  ReadChannelPort *toServer = nullptr;
  ReadChannelPort *lateToServer = nullptr;
  WriteChannelPort *toClient = nullptr;
  std::unique_ptr<RpcClient> client;

  /// Read one message from `port` as the simulator would, and return its
  /// credit. Fails the test (and returns an arbitrary value) rather than
  /// hanging if none arrives.
  uint32_t consumeFrom(ReadChannelPort &port) {
    std::future<MessageData> f = port.readAsync();
    if (f.wait_for(10s) != std::future_status::ready) {
      ADD_FAILURE() << "timed out waiting for a message";
      return ~0u;
    }
    MessageData m = f.get();
    server->messageConsumed(port);
    return *m.as<uint32_t>();
  }
};

TEST_F(CosimRpcTest, ToServerTryWriteBackpressure) {
  connectClient();
  client->setToServerMaxInFlight("in", 4);

  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_TRUE(client->tryWriteToServer("in", {msg(i)}));
  // At the in-flight limit: nothing consumed yet.
  EXPECT_FALSE(client->tryWriteToServer("in", {msg(4)}));

  // Consuming one returns one credit.
  EXPECT_EQ(consumeOne(), 0u);
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("in", {msg(4)}); }));
  EXPECT_FALSE(client->tryWriteToServer("in", {msg(5)}));

  for (uint32_t i = 1; i <= 4; ++i)
    EXPECT_EQ(consumeOne(), i);

  // A multi-frame message is admitted on a single credit.
  ASSERT_TRUE(waitFor([&] {
    return client->tryWriteToServer(
        "in", {msg(10), msg(11), msg(12), msg(13), msg(14)});
  }));
  EXPECT_FALSE(client->tryWriteToServer("in", {msg(15)}));
  for (uint32_t i = 10; i <= 14; ++i)
    EXPECT_EQ(consumeOne(), i);

  // With no in-flight limit, writes are never refused.
  client->setToServerMaxInFlight("in", 0);
  for (uint32_t i = 0; i < 100; ++i)
    EXPECT_TRUE(client->tryWriteToServer("in", {msg(i)}));
  for (uint32_t i = 0; i < 100; ++i)
    EXPECT_EQ(consumeOne(), i);
}

TEST_F(CosimRpcTest, ToServerForcedWriteIgnoresLimit) {
  connectClient();
  client->setToServerMaxInFlight("in", 1);
  EXPECT_TRUE(client->tryWriteToServer("in", {msg(0)}));
  EXPECT_FALSE(client->tryWriteToServer("in", {msg(1)}));
  // The rest of an admitted message is sent regardless of the limit, and
  // counts as in flight.
  EXPECT_TRUE(client->tryWriteToServer("in", {msg(1)}, /*force=*/true));
  EXPECT_EQ(consumeOne(), 0u);
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(client->tryWriteToServer("in", {msg(2)}));
  EXPECT_EQ(consumeOne(), 1u);
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("in", {msg(2)}); }));
  EXPECT_EQ(consumeOne(), 2u);
}

TEST_F(CosimRpcTest, ToServerBlockingWrite) {
  connectClient();
  client->setToServerMaxInFlight("in", 2);
  client->writeToServer("in", msg(0));
  client->writeToServer("in", msg(1));

  std::atomic<bool> done = false;
  std::thread writer([&] {
    client->writeToServer("in", msg(2));
    done = true;
  });
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(done.load()) << "write() should block at the in-flight limit";

  EXPECT_EQ(consumeOne(), 0u);
  EXPECT_TRUE(waitFor([&] { return done.load(); }));
  writer.join();
  EXPECT_EQ(consumeOne(), 1u);
  EXPECT_EQ(consumeOne(), 2u);
}

TEST_F(CosimRpcTest, ToServerBlockedWriterThrowsOnDisconnect) {
  connectClient();
  // Use a port the simulator side never connects: frames held for it stay in
  // flight even when the server stops. (Stopping disconnects connected ports,
  // which hands back the in-flight count of frames they held, legitimately
  // unblocking the writer.)
  client->setToServerMaxInFlight("late", 1);
  client->writeToServer("late", msg(0));

  std::promise<bool> threw;
  std::thread writer([&] {
    try {
      client->writeToServer("late", msg(1));
      threw.set_value(false);
    } catch (const std::exception &) {
      threw.set_value(true);
    }
  });
  std::this_thread::sleep_for(50ms);
  server->stop();
  server.reset();
  auto f = threw.get_future();
  ASSERT_EQ(f.wait_for(10s), std::future_status::ready);
  EXPECT_TRUE(f.get());
  writer.join();
}

TEST_F(CosimRpcTest, ToClientBackpressure) {
  // Without a subscriber, the default in-flight limit bounds the server-side
  // queue.
  uint32_t next = 0;
  while (toClient->tryWrite(msg(next)))
    ASSERT_LT(++next, 1000u) << "server never applied backpressure";
  EXPECT_EQ(next, RpcClient::DefaultToClientMaxInFlight);

  connectClient();
  std::atomic<bool> accepting = true;
  std::mutex receivedMutex;
  std::vector<uint32_t> received;
  auto conn = client->connectClientReceiver(
      "out",
      [&](std::unique_ptr<SegmentedMessageData> &data) {
        if (!accepting)
          return false;
        std::lock_guard<std::mutex> lock(receivedMutex);
        received.push_back(*data->toMessageData().as<uint32_t>());
        return true;
      },
      /*maxInFlight=*/4);
  auto numReceived = [&] {
    std::lock_guard<std::mutex> lock(receivedMutex);
    return received.size();
  };

  // The backlog drains to the client, whose acks make room again.
  ASSERT_TRUE(waitFor([&] { return numReceived() == next; }));
  ASSERT_TRUE(waitFor([&] { return toClient->tryWrite(msg(next)); }));
  ++next;
  ASSERT_TRUE(waitFor([&] { return numReceived() == next; }));

  // While the client refuses messages, at most `maxInFlight` can be in
  // flight.
  accepting = false;
  uint32_t admitted = 0;
  while (toClient->tryWrite(msg(next))) {
    ++next;
    ASSERT_LE(++admitted, 4u);
  }
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(toClient->tryWrite(msg(next)));

  // Once the client consumes again, credits flow back.
  accepting = true;
  ASSERT_TRUE(waitFor([&] { return toClient->tryWrite(msg(next)); }));
  ++next;
  ASSERT_TRUE(waitFor([&] { return numReceived() == next; }));
  std::lock_guard<std::mutex> lock(receivedMutex);
  for (uint32_t i = 0; i < next; ++i)
    EXPECT_EQ(received[i], i);
}

TEST_F(CosimRpcTest, ToClientUnsubscribeResetsInFlight) {
  connectClient();
  {
    std::atomic<bool> seen = false;
    auto conn = client->connectClientReceiver(
        "out",
        [&](std::unique_ptr<SegmentedMessageData> &) {
          seen = true;
          return false;
        },
        /*maxInFlight=*/1);
    EXPECT_TRUE(toClient->tryWrite(msg(0)));
    EXPECT_FALSE(toClient->tryWrite(msg(1)));
    // Make sure it was sent (not merely queued, which the server would keep
    // for the next subscriber) before unsubscribing.
    ASSERT_TRUE(waitFor([&] { return seen.load(); }));
  }
  // Unsubscribing drops the unacknowledged message and restores the default
  // in-flight limit for the (now empty) queue.
  uint32_t admitted = 0;
  ASSERT_TRUE(waitFor([&] { return toClient->tryWrite(msg(0)); }));
  ++admitted;
  while (toClient->tryWrite(msg(admitted)))
    ASSERT_LE(++admitted, RpcClient::DefaultToClientMaxInFlight);
  EXPECT_EQ(admitted, RpcClient::DefaultToClientMaxInFlight);
}

TEST_F(CosimRpcTest, UnconnectedServerPortBackpressures) {
  connectClient();
  // Frames for the unconnected port must be parked and count as in flight,
  // so the client stalls rather than losing data.
  client->setToServerMaxInFlight("late", 4);
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_TRUE(client->tryWriteToServer("late", {msg(i)}));
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(client->tryWriteToServer("late", {msg(4)}));

  // Connecting hands the parked frames over in order.
  lateToServer->connect();
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(consumeFrom(*lateToServer), i);
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("late", {msg(4)}); }));
  EXPECT_EQ(consumeFrom(*lateToServer), 4u);
}

TEST_F(CosimRpcTest, ServerPortDisconnectReturnsInFlight) {
  connectClient();
  lateToServer->connect();
  client->setToServerMaxInFlight("late", 3);
  for (uint32_t i = 0; i < 3; ++i)
    EXPECT_TRUE(client->tryWriteToServer("late", {msg(i)}));
  // Make sure they reached the port's queue before disconnecting: the server
  // handles frames in arrival order, so once a later frame on another
  // channel has been delivered, these have too.
  client->writeToServer("in", msg(99));
  EXPECT_EQ(consumeOne(), 99u);

  // Disconnecting discards the unconsumed frames; their count comes back.
  lateToServer->disconnect();
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("late", {msg(10)}); }));

  // And frames sent while disconnected are parked until reconnect.
  lateToServer->connect();
  EXPECT_EQ(consumeFrom(*lateToServer), 10u);
}

TEST_F(CosimRpcTest, RefusedFramesDontSkewDisconnectCredits) {
  connectClient();
  client->setToServerMaxInFlight("late", 1);
  // A callback-mode consumer which refuses everything: frames are dropped
  // and their credits returned.
  std::atomic<int> offered{0};
  lateToServer->connect([&](MessageData) -> bool {
    ++offered;
    return false;
  });
  EXPECT_TRUE(client->tryWriteToServer("late", {msg(1)}));
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("late", {msg(2)}); }));
  ASSERT_TRUE(waitFor([&] { return offered.load() == 2; }));
  lateToServer->disconnect();

  // Reconnect in polling mode and leave one frame unread.
  lateToServer->connect();
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("late", {msg(3)}); }));
  client->writeToServer("in", msg(99));
  EXPECT_EQ(consumeOne(), 99u);
  EXPECT_FALSE(client->tryWriteToServer("late", {msg(4)}));

  // Disconnecting discards it, which must return its credit.
  lateToServer->disconnect();
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("late", {msg(5)}); }));
}

TEST_F(CosimRpcTest, DepartedClientsCreditsNotGivenToNext) {
  // A client leaves a frame parked on "late" and goes away.
  connectClient();
  client->writeToServer("late", msg(1));
  client->writeToServer("in", msg(99));
  EXPECT_EQ(consumeOne(), 99u);
  client.reset();

  // The next client reaches its in-flight limit on "late".
  connectClient();
  client->setToServerMaxInFlight("late", 1);
  EXPECT_TRUE(client->tryWriteToServer("late", {msg(2)}));
  client->writeToServer("in", msg(98));
  EXPECT_EQ(consumeOne(), 98u);

  // Consuming the departed client's frame returns nothing to this one.
  lateToServer->connect();
  EXPECT_EQ(consumeFrom(*lateToServer), 1u);
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(client->tryWriteToServer("late", {msg(3)}));

  // Consuming its own frame does.
  EXPECT_EQ(consumeFrom(*lateToServer), 2u);
  ASSERT_TRUE(
      waitFor([&] { return client->tryWriteToServer("late", {msg(3)}); }));
  EXPECT_EQ(consumeFrom(*lateToServer), 3u);
}

TEST_F(CosimRpcTest, UnconnectedServerPortHoldsFrames) {
  connectClient();
  // Frames for a port the simulator hasn't connected yet are held, not
  // dropped, and handed over in order once it connects.
  for (uint32_t i = 0; i < 4; ++i)
    client->writeToServer("late", msg(i));
  // Make sure they've arrived: the server handles frames in order, so once a
  // later frame on another channel has been delivered, these have too.
  client->writeToServer("in", msg(99));
  EXPECT_EQ(consumeOne(), 99u);

  lateToServer->connect();
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_EQ(consumeFrom(*lateToServer), i);

  // And again after a disconnect.
  lateToServer->disconnect();
  client->writeToServer("late", msg(10));
  client->writeToServer("in", msg(98));
  EXPECT_EQ(consumeOne(), 98u);
  lateToServer->connect();
  EXPECT_EQ(consumeFrom(*lateToServer), 10u);
}

TEST_F(CosimRpcTest, RejectedClientLeavesActiveOneConnected) {
  connectClient();
  // Shared, since the callback may still be running in `set_value()` once
  // the test has seen the result and is returning.
  auto received = std::make_shared<std::promise<uint32_t>>();
  std::future<uint32_t> f = received->get_future();
  auto conn = client->connectClientReceiver(
      "out", [received](std::unique_ptr<SegmentedMessageData> &data) {
        received->set_value(*data->toMessageData().as<uint32_t>());
        return true;
      });

  // Only one client at a time: a second is turned away...
  EXPECT_ANY_THROW(
      RpcClient(ctxt->getLogger(), "127.0.0.1", server->getPort()));
  // (Let the server see it close.)
  std::this_thread::sleep_for(100ms);

  // ...without disturbing the first.
  toClient->write(msg(7));
  ASSERT_EQ(f.wait_for(10s), std::future_status::ready);
  EXPECT_EQ(f.get(), 7u);
}

TEST_F(CosimRpcTest, RefusedParkedFrameDoesNotBlockTheRest) {
  connectClient();
  for (uint32_t i = 0; i < 3; ++i)
    client->writeToServer("late", msg(i));
  client->writeToServer("in", msg(99));
  EXPECT_EQ(consumeOne(), 99u);

  // A callback-mode consumer which refuses the first frame it's offered.
  std::mutex receivedMutex;
  std::vector<uint32_t> received;
  bool refused = false;
  lateToServer->connect([&](MessageData data) -> bool {
    std::lock_guard<std::mutex> lock(receivedMutex);
    if (!refused) {
      refused = true;
      return false;
    }
    received.push_back(*data.as<uint32_t>());
    return true;
  });
  // The refused frame is dropped, as one arriving while connected would be;
  // the ones behind it still get through.
  {
    std::lock_guard<std::mutex> lock(receivedMutex);
    EXPECT_EQ(received, (std::vector<uint32_t>{1, 2}));
  }
  // The callback refers to locals.
  lateToServer->disconnect();
}

TEST_F(CosimRpcTest, SecondReceiverRejected) {
  connectClient();
  // Shared, since the callback may still be running in `set_value()` once
  // the test has seen the result and is returning.
  auto received = std::make_shared<std::promise<uint32_t>>();
  std::future<uint32_t> f = received->get_future();
  auto conn = client->connectClientReceiver(
      "out", [received](std::unique_ptr<SegmentedMessageData> &data) {
        received->set_value(*data->toMessageData().as<uint32_t>());
        return true;
      });
  EXPECT_THROW(
      client->connectClientReceiver(
          "out", [](std::unique_ptr<SegmentedMessageData> &) { return true; }),
      std::runtime_error);

  // The first receiver is undisturbed.
  toClient->write(msg(5));
  ASSERT_EQ(f.wait_for(10s), std::future_status::ready);
  EXPECT_EQ(f.get(), 5u);
}

TEST_F(CosimRpcTest, ToClientBlockingWrite) {
  connectClient();
  std::atomic<bool> accepting = false;
  std::atomic<int> received = 0;
  auto conn = client->connectClientReceiver(
      "out",
      [&](std::unique_ptr<SegmentedMessageData> &) {
        if (!accepting)
          return false;
        ++received;
        return true;
      },
      /*maxInFlight=*/2);
  toClient->write(msg(0));
  toClient->write(msg(1));

  // At the limit: `write()` blocks until the client consumes something.
  std::atomic<bool> written = false;
  std::thread writer([&] {
    toClient->write(msg(2));
    written = true;
  });
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(written);
  accepting = true;
  ASSERT_TRUE(waitFor([&] { return written.load(); }));
  writer.join();
  ASSERT_TRUE(waitFor([&] { return received == 3; }));
}

TEST_F(CosimRpcTest, ToClientBlockedWriteThrowsOnDisconnect) {
  // Without a subscriber, the default in-flight limit applies.
  uint32_t next = 0;
  while (toClient->tryWrite(msg(next)))
    ++next;
  std::promise<bool> threw;
  std::thread writer([&] {
    try {
      toClient->write(msg(next));
      threw.set_value(false);
    } catch (const std::exception &) {
      threw.set_value(true);
    }
  });
  std::this_thread::sleep_for(50ms);
  toClient->disconnect();
  auto f = threw.get_future();
  ASSERT_EQ(f.wait_for(10s), std::future_status::ready);
  EXPECT_TRUE(f.get());
  writer.join();
}

} // namespace
