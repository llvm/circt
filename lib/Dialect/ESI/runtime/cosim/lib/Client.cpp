//===- Client.cpp - Cosim RPC client ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CosimDpi.capnp.h"
#include "cosim/CapnpThreads.h"
#include "esi/Types.h"
#include <capnp/ez-rpc.h>

#include <cassert>
#include <condition_variable>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace capnp;
using namespace esi::cosim;

namespace {
class MessageReceiver final : public MessageReceiverInterface::Server {
public:
  MessageReceiver(
      std::function<void(Endpoint::MessageDataPtr)> messageRecvCallback)
      : messageRecvCallback(messageRecvCallback) {}

private:
  std::function<void(Endpoint::MessageDataPtr)> messageRecvCallback;

  kj::Promise<void> receiveMessage(ReceiveMessageContext context) override {
    auto msg = context.getParams().getMsg().asBytes();
    messageRecvCallback(
        std::make_unique<esi::MessageData>(msg.begin(), msg.size()));
    return kj::READY_NOW;
  }
};
} // namespace

/// Internal implementation to hide all the capnp details.
struct esi::cosim::RpcClient::Impl {

  Impl(RpcClient &client, capnp::EzRpcClient &rpcClient)
      : client(client), waitScope(rpcClient.getWaitScope()), cosim(nullptr),
        lowLevel(nullptr) {
    RpcClient::Lock l(client.m);

    // Get the main interface.
    cosim = rpcClient.getMain<CosimInterface>();

    // Grab a reference to the low level interface.
    auto llReq = cosim.openLowLevelRequest();
    auto llPromise = llReq.send();
    lowLevel = llPromise.wait(waitScope).getLowLevel();

    // Get the ESI version and compressed manifest.
    do {
      auto maniResp =
          cosim.getCompressedManifestRequest().send().wait(waitScope);
      capnp::Data::Reader data = maniResp.getCompressedManifest();
      client.esiVersion = maniResp.getVersion();
      client.compressedManifest =
          std::vector<uint8_t>(data.begin(), data.end());
      // If the version is invalid, the manifest is not yet loaded. Probably
      // means the simulation isn't quite ready to roll. Spin.
    } while (client.esiVersion < 0);
  }

  ~Impl() {}

  void disconnectAll() {
    for (auto &[epid, epIface] : endpointsConnected) {
      auto req = epIface.disconnectRequest();
      req.send().ignoreResult().wait(waitScope);
    }
    endpointsConnected.clear();
  }

  void connectSendEndpoint(const std::string &epId, const esi::Type *sendType) {
    if (connected[epId])
      throw std::runtime_error("Endpoint already connected");
    std::optional<std::runtime_error> failure;
    std::mutex m;
    std::unique_lock lk(m);
    std::condition_variable cv;
    ConnectRequest req = {epId, sendType->getID(),
                          [&](std::optional<std::runtime_error> e) {
                            std::unique_lock lk(m);
                            failure = e;
                            cv.notify_all();
                          },
                          nullptr};
    connectReqQueue.push(req);
    cv.wait(lk);
    if (failure)
      throw *failure;
  }

  void connectRecvEndpoint(
      const std::string &epId, const esi::Type *sendType,
      std::function<void(Endpoint::MessageDataPtr)> messageRecvCallback) {
    std::optional<std::runtime_error> failure;
    std::mutex m;
    std::unique_lock lk(m);
    std::condition_variable cv;
    ConnectRequest req = {epId, sendType->getID(),
                          [&](std::optional<std::runtime_error> e) {
                            std::unique_lock lk(m);
                            failure = e;
                            cv.notify_all();
                          },
                          messageRecvCallback};
    connectReqQueue.push(req);
    cv.wait(lk);
    if (failure)
      throw *failure;
  }

  struct ConnectRequest {
    std::string epId;
    std::string type;
    std::function<void(std::optional<std::runtime_error>)>
        connectResultCallback;
    std::function<void(Endpoint::MessageDataPtr)> messageRecvCallback;
  };
  TSQueue<ConnectRequest> connectReqQueue;

  RpcClient &client;
  kj::WaitScope &waitScope;
  CosimInterface::Client cosim;
  LowLevelInterface::Client lowLevel;

  std::map<std::string, ToSimInterface::Client> toSimEndpointMap;
  // std::map<std::string, MessageReceiver> fromSimMap;

  // Must lock on this mutex before accessing any shared member variables.
  std::mutex m;

  /// Shared member vars.
  std::map<std::string, bool> connected;
  std::map<std::string, FromSimInterface::Client> endpointsConnected;

  /// Called from the event loop periodically.
  // TODO: try to reduce work in here. Ideally, eliminate polling altogether
  // though I can't figure out how with libkj's event loop.
  void pollInternal();
};

void esi::cosim::RpcClient::Impl::pollInternal() {

  while (auto connReq = connectReqQueue.pop()) {
    std::optional<EndpointDesc::Reader> desc;
    auto capnpEndpointsResp = cosim.listRequest().send().wait(waitScope);
    for (const auto &capnpEndpoint : capnpEndpointsResp.getIfaces()) {
      if (capnpEndpoint.getId() == connReq->epId) {
        desc = capnpEndpoint;
        break;
      }
    }
    if (!desc)
      connReq->connectResultCallback(
          std::make_optional<std::runtime_error>("Endpoint not found"));
    if (desc->getType() != connReq->type)
      connReq->connectResultCallback(
          std::make_optional<std::runtime_error>("Endpoint type mismatch"));

    auto openReq = cosim.openRequest();
    openReq.setDesc(*desc);
    EndpointInterface::Client capnpEpClient =
        openReq.send().wait(waitScope).getEndpoint();
    std::optional<kj::Promise<void>> sendPromise;
    if (connReq->messageRecvCallback == nullptr) {
      auto toSim = capnpEpClient.castAs<ToSimInterface>();
      toSimEndpointMap.emplace(connReq->epId, toSim);
      auto capnpConnReq = toSim.connectRequest();
      sendPromise = capnpConnReq.send().ignoreResult();
      sendPromise
          ->then([this, connReq]() {
            std::lock_guard<std::mutex> l(m);
            connected[connReq->epId] = true;
            connReq->connectResultCallback(std::nullopt);
          })
          .detach([connReq](kj::Exception &&e) {
            connReq->connectResultCallback(
                std::make_optional<std::runtime_error>(
                    e.getDescription().cStr()));
          });
    } else {
      FromSimInterface::Client fromSimInterface =
          capnpEpClient.castAs<FromSimInterface>();
      auto capnpConnReq = fromSimInterface.connectRequest();
      capnpConnReq.setCallback(MessageReceiverInterface::Client(
          kj::heap<MessageReceiver>(connReq->messageRecvCallback)));
      capnpConnReq.send().ignoreResult().detach([connReq](
                                                    kj::Exception &&e) -> void {
        connReq->connectResultCallback(
            std::make_optional<std::runtime_error>(e.getDescription().cStr()));
      });
      std::lock_guard<std::mutex> l(m);
      connected[connReq->epId] = true;
      connReq->connectResultCallback(std::nullopt);
    }
  }

  // Iterate through the endpoints checking for messages.
  while (auto msgToSend = client.sendQueue.pop()) {
    // Process writes to the simulation.
    std::string epId = msgToSend->first;
    MessageData msg = std::move(msgToSend->second);

    {
      std::lock_guard<std::mutex> l(m);
      if (!connected[epId]) {
        fprintf(stderr, "Endpoint %s not connected.\n", epId.c_str());
        continue;
      }
    }

    assert(toSimEndpointMap.find(epId) != toSimEndpointMap.end() &&
           "Endpoint ID not found");
    auto req = toSimEndpointMap.at(epId).sendMessageRequest();
    req.setMsg(capnp::Data::Reader(msg.getBytes(), msg.getSize()));
    req.send().detach([](kj::Exception &&e) -> void {
      throw std::runtime_error("Error sending message to simulation: " +
                               std::string(e.getDescription().cStr()));
    });
  }

  // Process MMIO read requests.
  if (auto readReq = client.lowLevelBridge.readReqs.pop()) {
    auto req = lowLevel.readMMIORequest();
    req.setAddress(*readReq);
    auto respPromise = req.send();
    respPromise
        .then([&](auto resp) -> void {
          client.lowLevelBridge.readResps.push(
              std::make_pair(resp.getData(), 0));
        })
        .detach([&](kj::Exception &&e) -> void {
          client.lowLevelBridge.readResps.push(std::make_pair(0, 1));
        });
  }

  // Process MMIO write requests.
  if (auto writeReq = client.lowLevelBridge.writeReqs.pop()) {
    auto req = lowLevel.writeMMIORequest();
    req.setAddress(writeReq->first);
    req.setData(writeReq->second);
    req.send()
        .then([&](auto resp) -> void {
          client.lowLevelBridge.writeResps.push(0);
        })
        .detach([&](kj::Exception &&e) -> void {
          client.lowLevelBridge.writeResps.push(1);
        });
  }
}

RpcClient::~RpcClient() {}

void RpcClient::mainLoop(std::string host, uint16_t port) {
  capnp::EzRpcClient rpcClient(host, port);
  kj::WaitScope &waitScope = rpcClient.getWaitScope();
  impl = new Impl(*this, rpcClient);

  // Start the event loop. Does not return until stop() is called.
  loop(waitScope, [&]() { impl.load()->pollInternal(); });
  impl.load()->disconnectAll();
  delete impl;
  impl = nullptr;
}

/// Start the client if not already started.
void RpcClient::run(std::string host, uint16_t port) {
  {
    Lock l(m);
    if (myThread)
      throw std::runtime_error("Cannot Run() RPC client more than once!");
    myThread = new std::thread(&RpcClient::mainLoop, this, host, port);
  }

  // Spin until the capnp thread is started and ready to go.
  while (!impl.load())
    std::this_thread::sleep_for(std::chrono::microseconds(10));
}

void RpcClient::connectSendEndpoint(const std::string &epId,
                                    const esi::Type *sendType) {
  // Spin until the capnp thread is started and ready to go.
  while (!impl.load())
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  impl.load()->connectSendEndpoint(epId, sendType);
}
void RpcClient::connectRecvEndpoint(
    const std::string &epId, const esi::Type *sendType,
    std::function<void(cosim::Endpoint::MessageDataPtr)> messageRecvCallback) {
  // Spin until the capnp thread is started and ready to go.
  while (!impl.load())
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  impl.load()->connectRecvEndpoint(epId, sendType, messageRecvCallback);
}
