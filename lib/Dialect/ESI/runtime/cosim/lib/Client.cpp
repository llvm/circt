//===- Client.cpp - Cosim RPC client ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CosimDpi.capnp.h"
#include "cosim/CapnpThreads.h"
#include <capnp/ez-rpc.h>

#include <cassert>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace capnp;
using namespace esi::cosim;

/// Internal implementation to hide all the capnp details.
struct esi::cosim::RpcClient::Impl {

  Impl(RpcClient &client, capnp::EzRpcClient &rpcClient)
      : client(client), waitScope(rpcClient.getWaitScope()), cosim(nullptr),
        lowLevel(nullptr) {
    // Get the main interface.
    cosim = rpcClient.getMain<CosimDpiServer>();

    // Grab a reference to the low level interface.
    auto llReq = cosim.openLowLevelRequest();
    auto llPromise = llReq.send();
    lowLevel = llPromise.wait(waitScope).getLowLevel();

    // Get the ESI version and compressed manifest.
    auto maniResp = cosim.getCompressedManifestRequest().send().wait(waitScope);
    capnp::Data::Reader data = maniResp.getCompressedManifest();
    client.esiVersion = maniResp.getVersion();
    client.compressedManifest = std::vector<uint8_t>(data.begin(), data.end());

    // Iterate through the endpoints and register them.
    auto capnpEndpointsResp = cosim.listRequest().send().wait(waitScope);
    for (const auto &capnpEndpoint : capnpEndpointsResp.getIfaces()) {
      assert(capnpEndpoint.hasEndpointID() &&
             "Response did not contain endpoint ID not found!");
      std::string fromHostType, toHostType;
      if (capnpEndpoint.hasFromHostType())
        fromHostType = capnpEndpoint.getFromHostType();
      if (capnpEndpoint.hasToHostType())
        toHostType = capnpEndpoint.getToHostType();
      bool rc = client.endpoints.registerEndpoint(capnpEndpoint.getEndpointID(),
                                                  fromHostType, toHostType);
      assert(rc && "Endpoint ID already exists!");
      Endpoint *ep = client.endpoints[capnpEndpoint.getEndpointID()];
      // TODO: delay opening until client calls connect().
      auto openReq = cosim.openRequest();
      openReq.setIface(capnpEndpoint);
      EsiDpiEndpoint::Client dpiEp =
          openReq.send().wait(waitScope).getEndpoint();
      endpointMap.emplace(ep, dpiEp);
    }
  }

  RpcClient &client;
  kj::WaitScope &waitScope;
  CosimDpiServer::Client cosim;
  EsiLowLevel::Client lowLevel;
  std::map<Endpoint *, EsiDpiEndpoint::Client> endpointMap;

  /// Called from the event loop periodically.
  // TODO: try to reduce work in here. Ideally, eliminate polling altogether
  // though I can't figure out how with libkj's event loop.
  void pollInternal();
};

void esi::cosim::RpcClient::Impl::pollInternal() {
  // Iterate through the endpoints checking for messages.
  for (auto &[ep, capnpEp] : endpointMap) {
    // Process writes to the simulation.
    Endpoint::MessageDataPtr msg;
    if (!ep->getSendTypeId().empty() && ep->getMessageToSim(msg)) {
      auto req = capnpEp.sendFromHostRequest();
      req.setMsg(capnp::Data::Reader(msg->getBytes(), msg->getSize()));
      req.send().detach([](kj::Exception &&e) -> void {
        throw std::runtime_error("Error sending message to simulation: " +
                                 std::string(e.getDescription().cStr()));
      });
    }

    // Process reads from the simulation.
    // TODO: polling for a response is horribly slow and inefficient. Rework
    // the capnp protocol to avoid it.
    if (!ep->getRecvTypeId().empty()) {
      auto resp = capnpEp.recvToHostRequest().send().wait(waitScope);
      if (resp.getHasData()) {
        auto data = resp.getResp();
        ep->pushMessageToClient(
            std::make_unique<MessageData>(data.begin(), data.size()));
      }
    }
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

void RpcClient::mainLoop(std::string host, uint16_t port) {
  capnp::EzRpcClient rpcClient(host, port);
  kj::WaitScope &waitScope = rpcClient.getWaitScope();
  Impl impl(*this, rpcClient);

  // Signal that we're good to go.
  started.store(true);

  // Start the event loop. Does not return until stop() is called.
  loop(waitScope, [&]() { impl.pollInternal(); });
}

/// Start the client if not already started.
void RpcClient::run(std::string host, uint16_t port) {
  Lock g(m);
  if (myThread == nullptr) {
    started.store(false);
    myThread = new std::thread(&RpcClient::mainLoop, this, host, port);
    // Spin until the capnp thread is started and ready to go.
    while (!started.load())
      std::this_thread::sleep_for(std::chrono::microseconds(10));
  } else {
    fprintf(stderr, "Warning: cannot Run() RPC client more than once!");
  }
}
