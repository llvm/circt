//===- Server.cpp - Cosim RPC server ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for the RPC server class. Capnp C++ RPC servers are based on
// 'libkj' and its asyncrony model plus the capnp C++ API, both of which feel
// very foreign. In general, both RPC arguments and returns are passed as a C++
// object. In order to return data, the capnp message must be constructed inside
// that object.
//
// A [capnp encoded message](https://capnproto.org/encoding.html) can have
// multiple 'segments', which is a pain to deal with. (See comments below.)
//
//===----------------------------------------------------------------------===//

#include "CosimDpi.capnp.h"
#include "cosim/CapnpThreads.h"

#include <capnp/ez-rpc.h>
#include <kj/async-io.h>

#include <cassert>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace capnp;
using namespace esi::cosim;

namespace {
/// Implements the `EsiDpiEndpoint` interface from the RPC schema. Mostly a
/// wrapper around an `Endpoint` object. Whereas the `Endpoint`s are long-lived
/// (associated with the HW endpoint), this class is constructed/destructed
/// when the client open()s it.
class ToSimServer final : public ToSimInterface::Server {
  /// The wrapped endpoint.
  Endpoint &endpoint;
  /// Signals that this endpoint has been opened by a client and hasn't been
  /// closed by said client.
  bool open;

public:
  ToSimServer(Endpoint &ep);
  /// Release the Endpoint should the client disconnect without properly closing
  /// it.
  ~ToSimServer();
  /// Disallow copying as the 'open' variable needs to track the endpoint.
  ToSimServer(const ToSimServer &) = delete;

  kj::Promise<void> connect(ConnectContext) override;
  kj::Promise<void> disconnect(DisconnectContext) override;

  /// Implement the EsiDpiEndpoint RPC interface.
  kj::Promise<void> sendMessage(SendMessageContext) override;
};

class FromSimServer final : public FromSimInterface::Server {
  /// The wrapped endpoint.
  Endpoint &endpoint;
  kj::AsyncIoProvider &io;

  /// Signals that this endpoint has been opened by a client and hasn't been
  /// closed by said client.
  bool open;

public:
  FromSimServer(Endpoint &ep, kj::AsyncIoProvider &io);
  /// Release the Endpoint should the client disconnect without properly closing
  /// it.
  ~FromSimServer();
  /// Disallow copying as the 'open' variable needs to track the endpoint.
  FromSimServer(const FromSimServer &) = delete;

  /// Implement the EsiDpiEndpoint RPC interface.
  kj::Promise<void> connect(ConnectContext) override;
  // kj::Promise<void> recvToHost(RecvToHostContext) override;
  kj::Promise<void> disconnect(DisconnectContext) override;

private:
  kj::Promise<void> pollRecv(MessageReceiverInterface::Client);
};

/// Implement the low level cosim RPC protocol.
class LowLevelServer final : public LowLevelInterface::Server {
  // Queues to and from the simulation.
  LowLevel &bridge;

  // Functions which poll for responses without blocking the main loop. Polling
  // ain't great, but it's the only way (AFAICT) to do inter-thread
  // communication between a libkj concurrent thread and other threads. There is
  // a non-polling way to do it by setting up a queue over a OS-level pipe
  // (since the libkj event loop uses 'select').
  kj::Promise<void> pollReadResp(ReadMMIOContext context);
  kj::Promise<void> pollWriteResp(WriteMMIOContext context);

public:
  LowLevelServer(LowLevel &bridge);
  /// Release the Endpoint should the client disconnect without properly closing
  /// it.
  ~LowLevelServer();
  /// Disallow copying as the 'open' variable needs to track the endpoint.
  LowLevelServer(const LowLevelServer &) = delete;

  // Implement the protocol methods.
  kj::Promise<void> readMMIO(ReadMMIOContext) override;
  kj::Promise<void> writeMMIO(WriteMMIOContext) override;
};

/// Implements the `CosimDpiServer` interface from the RPC schema.
class CosimServer final : public CosimInterface::Server {
  /// The registry of endpoints. The RpcServer class owns this.
  EndpointRegistry &reg;
  LowLevel &lowLevelBridge;
  const signed int &esiVersion;
  const std::vector<uint8_t> &compressedManifest;

public:
  kj::AsyncIoProvider *io;

  CosimServer(EndpointRegistry &reg, LowLevel &lowLevelBridge,
              const signed int &esiVersion,
              const std::vector<uint8_t> &compressedManifest);

  /// List all the registered interfaces.
  kj::Promise<void> list(ListContext ctxt) override;
  /// Open a specific interface, locking it in the process.
  kj::Promise<void> open(OpenContext ctxt) override;

  kj::Promise<void>
      getCompressedManifest(GetCompressedManifestContext) override;

  kj::Promise<void> openLowLevel(OpenLowLevelContext ctxt) override;
};
} // anonymous namespace

/// ------ EndpointServer definitions.

ToSimServer::ToSimServer(Endpoint &ep) : endpoint(ep), open(false) {
  assert(ep.getDirection() == esi::cosim::Direction::ToSim &&
         "endpoint direction mismatch");
}
ToSimServer::~ToSimServer() {
  if (open)
    endpoint.returnForUse();
  open = false;
}

/// 'Send' is from the client perspective, so this is a message we are
/// recieving. The only way I could figure out to copy the raw message is a
/// double copy. I was have issues getting libkj's arrays to play nice with
/// others.
kj::Promise<void> ToSimServer::sendMessage(SendMessageContext context) {
  KJ_REQUIRE(open, "EndPoint closed");
  KJ_REQUIRE(context.getParams().hasMsg(), "Send request must have a message.");
  kj::ArrayPtr<const kj::byte> data = context.getParams().getMsg().asBytes();
  Endpoint::MessageDataPtr blob = std::make_unique<esi::MessageData>(
      (const uint8_t *)data.begin(), data.size());
  endpoint.pushMessage(std::move(blob));
  return kj::READY_NOW;
}

kj::Promise<void> ToSimServer::connect(ConnectContext context) {
  KJ_LOG(INFO, "Connecting toSim endpoint", endpoint.getId());
  KJ_REQUIRE(!open, "EndPoint connected already");
  open = true;
  auto gotLock = endpoint.setInUse();
  KJ_REQUIRE(gotLock, "Endpoint in use");
  return kj::READY_NOW;
}

kj::Promise<void> ToSimServer::disconnect(DisconnectContext context) {
  KJ_LOG(INFO, "Disconnecting toSim endpoint", endpoint.getId());
  KJ_REQUIRE(open, "EndPoint disconnected already");
  open = false;
  return kj::READY_NOW;
}

FromSimServer::FromSimServer(Endpoint &ep, kj::AsyncIoProvider &io)
    : endpoint(ep), io(io), open(false) {
  assert(ep.getDirection() == esi::cosim::Direction::FromSim &&
         "endpoint direction mismatch");
}
FromSimServer::~FromSimServer() {
  if (open)
    endpoint.returnForUse();
  open = false;
}

kj::Promise<void> FromSimServer::connect(ConnectContext context) {
  KJ_LOG(INFO, "Connecting fromSim endpoint", endpoint.getId());
  KJ_REQUIRE(!open, "EndPoint already open");
  open = true;
  auto gotLock = endpoint.setInUse();
  KJ_REQUIRE(gotLock, "Endpoint in use");
  return pollRecv(context.getParams().getCallback());
}

kj::Promise<void> FromSimServer::disconnect(DisconnectContext context) {
  KJ_LOG(INFO, "Disconnecting fromSim endpoint", endpoint.getId());
  KJ_REQUIRE(open, "EndPoint disconnected already");
  open = false;
  endpoint.returnForUse();
  return kj::READY_NOW;
}

kj::Promise<void>
FromSimServer::pollRecv(MessageReceiverInterface::Client callback) {
  if (!open) {
    KJ_DBG("Endpoint closed", endpoint.getId());
    return kj::READY_NOW;
  }

  Endpoint::MessageDataPtr msg;
  if (endpoint.getMessage(msg)) {
    KJ_DBG("Sending message to client", endpoint.getId());
    auto msgCall = callback.receiveMessageRequest();
    msgCall.setMsg(kj::arrayPtr(msg->getBytes(), msg->getSize()));
    return msgCall.send().ignoreResult().then(
        [this, KJ_CPCAP(callback)]() mutable { return pollRecv(callback); });
  } else {
    return io.getTimer()
        .afterDelay(1 * kj::MILLISECONDS)
        .then(
            [this, KJ_CPCAP(callback)]() mutable { return pollRecv(callback); })
        .eagerlyEvaluate([](kj::Exception &&e) { KJ_LOG(ERROR, e); });
  }
}

/// ------ LowLevelServer definitions.

LowLevelServer::LowLevelServer(LowLevel &bridge) : bridge(bridge) {}
LowLevelServer::~LowLevelServer() {}

kj::Promise<void> LowLevelServer::pollReadResp(ReadMMIOContext context) {
  auto respMaybe = bridge.readResps.pop();
  if (!respMaybe.has_value()) {
    return kj::evalLast(
        [this, KJ_CPCAP(context)]() mutable { return pollReadResp(context); });
  }
  auto resp = respMaybe.value();
  KJ_REQUIRE(resp.second == 0, "Read MMIO register encountered an error");
  context.getResults().setData(resp.first);
  return kj::READY_NOW;
}

kj::Promise<void> LowLevelServer::readMMIO(ReadMMIOContext context) {
  bridge.readReqs.push(context.getParams().getAddress());
  return kj::evalLast(
      [this, KJ_CPCAP(context)]() mutable { return pollReadResp(context); });
}

kj::Promise<void> LowLevelServer::pollWriteResp(WriteMMIOContext context) {
  auto respMaybe = bridge.writeResps.pop();
  if (!respMaybe.has_value()) {
    return kj::evalLast(
        [this, KJ_CPCAP(context)]() mutable { return pollWriteResp(context); });
  }
  auto resp = respMaybe.value();
  KJ_REQUIRE(resp == 0, "write MMIO register encountered an error");
  return kj::READY_NOW;
}

kj::Promise<void> LowLevelServer::writeMMIO(WriteMMIOContext context) {
  bridge.writeReqs.push(context.getParams().getAddress(),
                        context.getParams().getData());
  return kj::evalLast(
      [this, KJ_CPCAP(context)]() mutable { return pollWriteResp(context); });
}

/// ----- CosimServer definitions.

CosimServer::CosimServer(EndpointRegistry &reg, LowLevel &lowLevelBridge,
                         const signed int &esiVersion,
                         const std::vector<uint8_t> &compressedManifest)
    : reg(reg), lowLevelBridge(lowLevelBridge), esiVersion(esiVersion),
      compressedManifest(compressedManifest) {
  printf("version: %d\n", esiVersion);
}

kj::Promise<void> CosimServer::list(ListContext context) {
  auto ifaces = context.getResults().initIfaces((unsigned int)reg.size());
  unsigned int ctr = 0u;
  reg.iterateEndpoints([&](std::string id, const Endpoint &ep) {
    ifaces[ctr].setId(id);
    ifaces[ctr].setType(ep.getTypeId());
    if (ep.getDirection() == esi::cosim::Direction::ToSim)
      ifaces[ctr].setDirection(::Direction::TO_SIM);
    else
      ifaces[ctr].setDirection(::Direction::FROM_SIM);
    ++ctr;
  });
  return kj::READY_NOW;
}

kj::Promise<void> CosimServer::open(OpenContext ctxt) {
  const auto &epDesc = ctxt.getParams().getDesc();
  Endpoint *ep = reg[epDesc.getId()];
  KJ_REQUIRE(ep != nullptr, "Could not find endpoint");

  if (epDesc.getDirection() == ::Direction::TO_SIM)
    ctxt.getResults().setEndpoint(
        EndpointInterface::Client(kj::heap<ToSimServer>(*ep)));
  else
    ctxt.getResults().setEndpoint(
        EndpointInterface::Client(kj::heap<FromSimServer>(*ep, *io)));
  return kj::READY_NOW;
}

kj::Promise<void>
CosimServer::getCompressedManifest(GetCompressedManifestContext ctxt) {
  ctxt.getResults().setVersion(esiVersion);
  ctxt.getResults().setCompressedManifest(
      Data::Reader(compressedManifest.data(), compressedManifest.size()));
  return kj::READY_NOW;
}

kj::Promise<void> CosimServer::openLowLevel(OpenLowLevelContext ctxt) {
  ctxt.getResults().setLowLevel(kj::heap<LowLevelServer>(lowLevelBridge));
  return kj::READY_NOW;
}

/// ----- RpcServer definitions.

/// Write the port number to a file. Necessary when we allow 'EzRpcServer' to
/// select its own port. We can't use stdout/stderr because the flushing
/// semantics are undefined (as in `flush()` doesn't work on all simulators).
static void writePort(uint16_t port) {
  // "cosim.cfg" since we may want to include other info in the future.
  FILE *fd = fopen("cosim.cfg", "w");
  fprintf(fd, "port: %u\n", (unsigned int)port);
  fclose(fd);
}

void RpcServer::mainLoop(uint16_t port) {
  auto server = kj::heap<CosimServer>(endpoints, lowLevelBridge, esiVersion,
                                      compressedManifest);
  auto serverPtr = server.get();
  capnp::EzRpcServer rpcServer(std::move(server),
                               /* bindAddress */ "*", port);
  serverPtr->io = &rpcServer.getIoProvider();
  auto &waitScope = rpcServer.getWaitScope();
  // If port is 0, ExRpcSever selects one and we have to wait to get the port.
  if (port == 0) {
    auto portPromise = rpcServer.getPort();
    port = portPromise.wait(waitScope);
  }
  writePort(port);
  printf("[COSIM] Listening on port: %u\n", (unsigned int)port);
  loop(waitScope, []() {});
}

/// Start the server if not already started.
void RpcServer::run(uint16_t port) {
  Lock g(m);
  kj::_::Debug::setLogLevel(kj::LogSeverity::INFO);
  if (myThread == nullptr) {
    myThread = new std::thread(&RpcServer::mainLoop, this, port);
  } else {
    fprintf(stderr, "Warning: cannot Run() RPC server more than once!");
  }
}

Endpoint *RpcServer::getEndpoint(std::string epId) { return endpoints[epId]; }
