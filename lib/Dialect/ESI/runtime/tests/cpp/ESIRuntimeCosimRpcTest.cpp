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

#include <chrono>
#include <cstdint>
#include <filesystem>
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

  /// Read one message from the to-server port as the simulator would.
  uint32_t consumeOne() { return consumeFrom(*toServer); }

  std::filesystem::path oldCwd, tmpDir;
  std::unique_ptr<Context> ctxt;
  std::unique_ptr<RpcServer> server;
  ReadChannelPort *toServer = nullptr;
  ReadChannelPort *lateToServer = nullptr;
  WriteChannelPort *toClient = nullptr;
  std::unique_ptr<RpcClient> client;

  /// Read one message from `port` as the simulator would. Fails the test
  /// (and returns an arbitrary value) rather than hanging if none arrives.
  uint32_t consumeFrom(ReadChannelPort &port) {
    std::future<MessageData> f = port.readAsync();
    if (f.wait_for(10s) != std::future_status::ready) {
      ADD_FAILURE() << "timed out waiting for a message";
      return ~0u;
    }
    return *f.get().as<uint32_t>();
  }
};

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

} // namespace
