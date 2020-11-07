//===- Server.cpp - Cosim RPC server ----------------------------*- C++ -*-===//
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

#include "circt/Dialect/ESI/cosim/Server.h"
#include <capnp/layout.h>
#include <capnp/message.h>
#include <kj/debug.h>
#include <stdexcept>

using namespace capnp;
using namespace circt::esi::cosim;

kj::Promise<void> EndPointServer::close(CloseContext context) {
  KJ_REQUIRE(_Open, "EndPoint closed already");
  _Open = false;
  _EndPoint->ReturnForUse();
  return kj::READY_NOW;
}

/// 'Send' is from the client perspective, so this is a message we are recieving.
/// The only way I could figure out to copy the raw message is a double copy. I
/// was have issues getting libkj's arrays to play nice with others.
kj::Promise<void> EndPointServer::send(SendContext context) {
  KJ_REQUIRE(_Open, "EndPoint closed already");
  auto capnpMsgPointer = context.getParams().getMsg();
  KJ_REQUIRE(capnpMsgPointer.isStruct(),
             "Only messages can go in the 'msg' parameter");

  // Copy the incoming message into a flat, single segment buffer.
  auto msgSize = capnpMsgPointer.targetSize();
  auto builder = std::make_unique<MallocMessageBuilder>(
      msgSize.wordCount + 1, AllocationStrategy::FIXED_SIZE);
  builder->setRoot(capnpMsgPointer);
  auto segments = builder->getSegmentsForOutput();
  KJ_ASSERT(segments.size() == 1);

  // Now copy it into a blob and queue it.
  auto fstSegmentData = segments[0].asBytes();
  auto blob = std::make_shared<EndPoint::Blob>(fstSegmentData.begin(),
                                               fstSegmentData.end());
  _EndPoint->PushMessageToSim(blob);
  return kj::READY_NOW;
}

/// This is the client polling for a message. If one is available, send it.
/// TODO: implement a blocking call with a timeout.
kj::Promise<void> EndPointServer::recv(RecvContext context) {
  KJ_REQUIRE(_Open, "EndPoint closed already");
  KJ_REQUIRE(!context.getParams().getBlock(),
             "Blocking recv() not supported yet");

  // Try to pop a message.
  EndPoint::BlobPtr blob;
  auto msgPresent = _EndPoint->GetMessageToClient(blob);
  context.getResults().setHasData(msgPresent);
  if (msgPresent) {
    KJ_REQUIRE(blob->size() % 8 == 0,
               "Response msg was malformed. Size of response was not a "
               "multiple of 8 bytes.");
    // Copy the blob into a single segment.
    auto segment =
        kj::ArrayPtr<word>((word *)blob->data(), blob->size() / 8).asConst();
    // Create a single-element array of segments.
    auto segments = kj::heapArray({segment});
    // Create an object which will read the segments into a message on send.
    auto msgReader = std::make_unique<SegmentArrayMessageReader>(segments);
    // Send.
    context.getResults().getResp().set(msgReader->getRoot<AnyPointer>());
  }
  return kj::READY_NOW;
}

kj::Promise<void> CosimServer::list(ListContext context) {
  auto ifaces = context.getResults().initIfaces((unsigned int)_Reg->Size());
  unsigned int ctr = 0u;
  _Reg->IterateEndpoints([&](int id, const EndPoint& ep) {
    ifaces[ctr].setEndpointID(id);
    ifaces[ctr].setTypeID(ep.GetEsiTypeId());
    ctr++;
  });
  return kj::READY_NOW;
}

kj::Promise<void> CosimServer::open(OpenContext ctxt) {
  EndPoint* ep;
  bool found = _Reg->Get(ctxt.getParams().getIface().getEndpointID(), ep);
  KJ_REQUIRE(found, "Could not find endpoint");

  auto gotLock = ep->SetInUse();
  KJ_REQUIRE(gotLock, "Endpoint in use");

  ctxt.getResults().setIface(EsiDpiEndpoint<AnyPointer, AnyPointer>::Client(
      kj::heap<EndPointServer>(ep)));
  return kj::READY_NOW;
}

RpcServer::~RpcServer() { Stop(); }

void RpcServer::MainLoop(uint16_t port) {
  _RpcServer = new EzRpcServer(kj::heap<CosimServer>(&EndPoints), "*", port);
  auto &waitScope = _RpcServer->getWaitScope();

  // OK, this is uber hacky, but it unblocks me and isn't too inefficient. The
  // problem is that I can't figure out how read the stop signal from libkj
  // asyncrony land.
  //
  // IIRC the main libkj wait loop uses `select()` (or something similar on
  // Windows) on its FDs. As a result, any code which checks the stop variable
  // doesn't run until there is some I/O. Probably the right way is to set up a
  // pipe to deliver a shutdown signal.
  //
  // TODO: Figure out how to do this properly, if possible.
  while (!_Stop) {
    waitScope.poll();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void RpcServer::Run(uint16_t port) {
  if (_MainThread == nullptr) {
    _MainThread = new std::thread(&RpcServer::MainLoop, this, port);
  } else {
    throw std::runtime_error("Cannot Run() RPC server more than once!");
  }
}

void RpcServer::Stop() {
  if (_MainThread == nullptr) {
    throw std::runtime_error("RpcServer not Run()");
  } else if (!_Stop) {
    _Stop = true;
    _MainThread->join();
  }
}
