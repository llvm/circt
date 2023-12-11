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

#include "cosim/Server.h"
#include "CosimDpi.capnp.h"
#include <capnp/ez-rpc.h>
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
class EndpointServer final : public EsiDpiEndpoint::Server {
  /// The wrapped endpoint.
  Endpoint &endpoint;
  /// Signals that this endpoint has been opened by a client and hasn't been
  /// closed by said client.
  bool open;

public:
  EndpointServer(Endpoint &ep);
  /// Release the Endpoint should the client disconnect without properly closing
  /// it.
  ~EndpointServer();
  /// Disallow copying as the 'open' variable needs to track the endpoint.
  EndpointServer(const EndpointServer &) = delete;

  /// Implement the EsiDpiEndpoint RPC interface.
  kj::Promise<void> sendFromHost(SendFromHostContext) override;
  kj::Promise<void> recvToHost(RecvToHostContext) override;
  kj::Promise<void> close(CloseContext) override;
};

/// Implement the low level cosim RPC protocol.
class LowLevelServer final : public EsiLowLevel::Server {
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
class CosimServer final : public CosimDpiServer::Server {
  /// The registry of endpoints. The RpcServer class owns this.
  EndpointRegistry &reg;
  LowLevel &lowLevelBridge;
  const unsigned int &esiVersion;
  const std::vector<uint8_t> &compressedManifest;

public:
  CosimServer(EndpointRegistry &reg, LowLevel &lowLevelBridge,
              const unsigned int &esiVersion,
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

EndpointServer::EndpointServer(Endpoint &ep) : endpoint(ep), open(true) {}
EndpointServer::~EndpointServer() {
  if (open)
    endpoint.returnForUse();
}

/// This is the client polling for a message. If one is available, send it.
/// TODO: implement a blocking call with a timeout.
kj::Promise<void> EndpointServer::recvToHost(RecvToHostContext context) {
  KJ_REQUIRE(open, "EndPoint closed already");

  // Try to pop a message.
  Endpoint::BlobPtr blob;
  auto msgPresent = endpoint.getMessageToClient(blob);
  context.getResults().setHasData(msgPresent);
  if (msgPresent) {
    Data::Builder data(blob->data(), blob->size());
    context.getResults().setResp(data.asReader());
  }
  return kj::READY_NOW;
}

/// 'Send' is from the client perspective, so this is a message we are
/// recieving. The only way I could figure out to copy the raw message is a
/// double copy. I was have issues getting libkj's arrays to play nice with
/// others.
kj::Promise<void> EndpointServer::sendFromHost(SendFromHostContext context) {
  KJ_REQUIRE(open, "EndPoint closed already");
  KJ_REQUIRE(context.getParams().hasMsg(), "Send request must have a message.");
  kj::ArrayPtr<const kj::byte> data = context.getParams().getMsg().asBytes();
  Endpoint::BlobPtr blob =
      std::make_unique<Endpoint::Blob>(data.begin(), data.end());
  endpoint.pushMessageToSim(std::move(blob));
  return kj::READY_NOW;
}

kj::Promise<void> EndpointServer::close(CloseContext context) {
  KJ_REQUIRE(open, "EndPoint closed already");
  open = false;
  endpoint.returnForUse();
  return kj::READY_NOW;
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
                         const unsigned int &esiVersion,
                         const std::vector<uint8_t> &compressedManifest)
    : reg(reg), lowLevelBridge(lowLevelBridge), esiVersion(esiVersion),
      compressedManifest(compressedManifest) {
  printf("version: %d\n", esiVersion);
}

kj::Promise<void> CosimServer::list(ListContext context) {
  auto ifaces = context.getResults().initIfaces((unsigned int)reg.size());
  unsigned int ctr = 0u;
  reg.iterateEndpoints([&](std::string id, const Endpoint &ep) {
    ifaces[ctr].setEndpointID(id);
    ifaces[ctr].setFromHostType(ep.getSendTypeId());
    ifaces[ctr].setToHostType(ep.getRecvTypeId());
    ++ctr;
  });
  return kj::READY_NOW;
}

kj::Promise<void> CosimServer::open(OpenContext ctxt) {
  Endpoint *ep = reg[ctxt.getParams().getIface().getEndpointID()];
  KJ_REQUIRE(ep != nullptr, "Could not find endpoint");

  auto gotLock = ep->setInUse();
  KJ_REQUIRE(gotLock, "Endpoint in use");

  ctxt.getResults().setEndpoint(
      EsiDpiEndpoint::Client(kj::heap<EndpointServer>(*ep)));
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

RpcServer::RpcServer() : mainThread(nullptr), stopSig(false) {}
RpcServer::~RpcServer() { stop(); }

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
  capnp::EzRpcServer rpcServer(kj::heap<CosimServer>(endpoints, lowLevelBridge,
                                                     esiVersion,
                                                     compressedManifest),
                               /* bindAddress */ "*", port);
  auto &waitScope = rpcServer.getWaitScope();
  // If port is 0, ExRpcSever selects one and we have to wait to get the port.
  if (port == 0) {
    auto portPromise = rpcServer.getPort();
    port = portPromise.wait(waitScope);
  }
  writePort(port);
  printf("[COSIM] Listening on port: %u\n", (unsigned int)port);

  // OK, this is uber hacky, but it unblocks me and isn't _too_ inefficient. The
  // problem is that I can't figure out how read the stop signal from libkj
  // asyncrony land.
  //
  // IIRC the main libkj wait loop uses `select()` (or something similar on
  // Windows) on its FDs. As a result, any code which checks the stop variable
  // doesn't run until there is some I/O. Probably the right way is to set up a
  // pipe to deliver a shutdown signal.
  //
  // TODO: Figure out how to do this properly, if possible.
  while (!stopSig) {
    waitScope.poll();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

/// Start the server if not already started.
void RpcServer::run(uint16_t port) {
  Lock g(m);
  if (mainThread == nullptr) {
    mainThread = new std::thread(&RpcServer::mainLoop, this, port);
  } else {
    fprintf(stderr, "Warning: cannot Run() RPC server more than once!");
  }
}

/// Signal the RPC server thread to stop. Wait for it to exit.
void RpcServer::stop() {
  Lock g(m);
  if (mainThread == nullptr) {
    fprintf(stderr, "RpcServer not Run()\n");
  } else if (!stopSig) {
    stopSig = true;
    mainThread->join();
  }
}
