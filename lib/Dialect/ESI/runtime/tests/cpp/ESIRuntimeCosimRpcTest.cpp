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

  /// Read one message from the to-server port as the simulator would.
  uint32_t consumeOne() { return consumeFrom(*toServer); }

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

  /// Read one message from `port` as the simulator would.
  uint32_t consumeFrom(ReadChannelPort &port) {
    std::future<MessageData> f = port.readAsync();
    EXPECT_EQ(f.wait_for(10s), std::future_status::ready);
    return *f.get().as<uint32_t>();
  }
};

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
