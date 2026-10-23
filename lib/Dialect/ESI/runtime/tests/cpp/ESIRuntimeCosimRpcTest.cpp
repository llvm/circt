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
#include <vector>

using namespace esi;
using esi::backends::cosim::RpcClient;
using esi::cosim::RpcServer;
using namespace std::chrono_literals;

namespace {

MessageData msg(uint32_t v) { return MessageData::from(v); }

bool isReady(std::future<void> &f) {
  return f.wait_for(0s) == std::future_status::ready;
}

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
    late2ToServer = &server->registerReadPort("late2", "i32");
    toClient = &server->registerWritePort("out", "i32");
    // Registered, but left for each test to connect (or not).
    lateToClient = &server->registerWritePort("lateOut", "i32");
    toServerClientEnd = toServer->connect();
    toClientClientEnd = toClient->connect();
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
  uint32_t consumeOne() {
    std::future<MessageData> f = toServer->readAsync();
    EXPECT_EQ(f.wait_for(10s), std::future_status::ready);
    MessageData m = f.get();
    server->messageConsumed(*toServer);
    return *m.as<uint32_t>();
  }

  std::filesystem::path oldCwd, tmpDir;
  std::unique_ptr<Context> ctxt;
  std::unique_ptr<RpcServer> server;
  ReadChannelPort *toServer = nullptr;
  ReadChannelPort *lateToServer = nullptr;
  ReadChannelPort *late2ToServer = nullptr;
  WriteChannelPort *lateToClient = nullptr;
  // What the simulator side's connect() returned for "in" and "out".
  std::future<void> toServerClientEnd, toClientClientEnd;
  WriteChannelPort *toClient = nullptr;
  std::unique_ptr<RpcClient> client;

  /// Read one message from `port` as the simulator would, and return its
  /// credit.
  uint32_t consumeFrom(ReadChannelPort &port) {
    std::future<MessageData> f = port.readAsync();
    EXPECT_EQ(f.wait_for(10s), std::future_status::ready);
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
  conn->subscribed().get();
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
    conn->subscribed().get();
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

TEST_F(CosimRpcTest, ConnectWaitsForServerPort) {
  connectClient();
  // Already connected on the server side: ready at once.
  std::future<void> in = client->connectToServer("in");
  ASSERT_EQ(in.wait_for(10s), std::future_status::ready);
  in.get();

  std::future<void> late = client->connectToServer("late");
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(isReady(late))
      << "connectToServer() should wait for the server-side port";

  // Other traffic keeps flowing while the request is parked.
  client->writeToServer("in", msg(7));
  EXPECT_EQ(consumeOne(), 7u);

  // The simulator side's future is ready immediately: the client end is
  // already connected.
  std::future<void> simSide = lateToServer->connect();
  EXPECT_EQ(simSide.wait_for(10s), std::future_status::ready);
  ASSERT_EQ(late.wait_for(10s), std::future_status::ready);
  late.get();

  // Frames sent after the connect completes are delivered.
  client->writeToServer("late", msg(42));
  EXPECT_EQ(consumeFrom(*lateToServer), 42u);
}

TEST_F(CosimRpcTest, ServerReadPortFutureWaitsForClient) {
  // Connected in SetUp, before any client: waits for one to connect its end.
  EXPECT_FALSE(isReady(toServerClientEnd));
  connectClient();
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(isReady(toServerClientEnd))
      << "a client connecting isn't the same as it connecting the channel";
  client->connectToServer("in");
  ASSERT_EQ(toServerClientEnd.wait_for(10s), std::future_status::ready);
}

TEST_F(CosimRpcTest, ConnectFailsOnDisconnect) {
  connectClient();
  std::future<void> late = client->connectToServer("late");
  std::this_thread::sleep_for(50ms);
  server->stop();
  server.reset();
  ASSERT_EQ(late.wait_for(10s), std::future_status::ready);
  EXPECT_THROW(late.get(), std::exception);
}

TEST_F(CosimRpcTest, ConnectReplyNotDeliveredToNextClient) {
  // A client parks a connect request on "late", then goes away before it's
  // answered.
  connectClient();
  std::future<void> stale = client->connectToServer("late");
  std::this_thread::sleep_for(50ms);
  client.reset();
  EXPECT_THROW(stale.get(), std::exception);

  // A new client parks a connect on a different port. Its request id matches
  // the stale one (ids restart per connection: hello, then connect).
  connectClient();
  std::future<void> late2 = client->connectToServer("late2");
  std::this_thread::sleep_for(50ms);

  // Connecting "late" answers only the departed client's request.
  lateToServer->connect();
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(isReady(late2)) << "stale connect reply reached the new client";

  late2ToServer->connect();
  ASSERT_EQ(late2.wait_for(10s), std::future_status::ready);
  late2.get();
}

TEST_F(CosimRpcTest, SubscribeWaitsForServerPort) {
  connectClient();
  std::atomic<int> received = 0;
  auto conn = client->connectClientReceiver(
      "lateOut", [&](std::unique_ptr<SegmentedMessageData> &) {
        ++received;
        return true;
      });
  std::future<void> subscribed = conn->subscribed();
  std::this_thread::sleep_for(100ms);
  EXPECT_FALSE(isReady(subscribed))
      << "subscribe should wait for the server-side port";

  // Connecting the simulator side completes the subscription. Its own future
  // is ready right away, since the client is already waiting.
  std::future<void> simSide = lateToClient->connect();
  ASSERT_EQ(subscribed.wait_for(10s), std::future_status::ready);
  subscribed.get();
  ASSERT_EQ(simSide.wait_for(10s), std::future_status::ready);

  EXPECT_TRUE(lateToClient->tryWrite(msg(5)));
  EXPECT_TRUE(waitFor([&] { return received.load() == 1; }));
}

TEST_F(CosimRpcTest, ServerWritePortFutureWaitsForSubscribe) {
  EXPECT_FALSE(isReady(toClientClientEnd));
  connectClient();
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(isReady(toClientClientEnd));
  auto conn = client->connectClientReceiver(
      "out", [](std::unique_ptr<SegmentedMessageData> &) { return true; });
  ASSERT_EQ(toClientClientEnd.wait_for(10s), std::future_status::ready);
  conn->subscribed().get();
}

TEST_F(CosimRpcTest, ServerFuturesWaitForNextClientAfterDisconnect) {
  connectClient();
  {
    auto conn = client->connectClientReceiver(
        "out", [](std::unique_ptr<SegmentedMessageData> &) { return true; });
    conn->subscribed().get();
  }
  client->connectToServer("in").get();
  ASSERT_EQ(toClientClientEnd.wait_for(10s), std::future_status::ready);
  ASSERT_EQ(toServerClientEnd.wait_for(10s), std::future_status::ready);

  // Once the client is gone, reconnecting the simulator side waits for the
  // next one.
  client.reset();
  std::this_thread::sleep_for(50ms);
  toClient->disconnect();
  toServer->disconnect();
  std::future<void> out = toClient->connect();
  std::future<void> in = toServer->connect();
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(isReady(out));
  EXPECT_FALSE(isReady(in));

  connectClient();
  auto conn = client->connectClientReceiver(
      "out", [](std::unique_ptr<SegmentedMessageData> &) { return true; });
  client->connectToServer("in");
  EXPECT_EQ(out.wait_for(10s), std::future_status::ready);
  EXPECT_EQ(in.wait_for(10s), std::future_status::ready);
}

TEST_F(CosimRpcTest, DisconnectCancelsPendingSubscribe) {
  connectClient();
  auto conn = client->connectClientReceiver(
      "lateOut", [](std::unique_ptr<SegmentedMessageData> &) { return true; });
  std::future<void> subscribed = conn->subscribed();
  std::this_thread::sleep_for(50ms);
  // Disconnecting before the simulator side connects cancels the request.
  conn->disconnect();
  ASSERT_EQ(subscribed.wait_for(10s), std::future_status::ready);
  EXPECT_THROW(subscribed.get(), std::exception);

  // The channel can be subscribed again afterwards.
  auto conn2 = client->connectClientReceiver(
      "lateOut", [](std::unique_ptr<SegmentedMessageData> &) { return true; });
  lateToClient->connect();
  conn2->subscribed().get();
}

TEST_F(CosimRpcTest, UnconnectedServerPortBackpressures) {
  connectClient();
  // Skip connectToServer(): frames for the unconnected port must be parked
  // and count as in flight, so the client stalls rather than losing data.
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

} // namespace
