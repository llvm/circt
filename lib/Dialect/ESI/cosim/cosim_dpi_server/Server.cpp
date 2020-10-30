// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "circt/Dialect/ESI/cosim/Server.h"
#include <capnp/layout.h>
#include <capnp/message.h>
#include <kj/debug.h>
#include <stdexcept>

using namespace std;
using namespace capnp;

kj::Promise<void> EndPointServer::close(CloseContext context) {
  KJ_REQUIRE(_Open, "EndPoint closed already");
  _Open = false;
  _EndPoint->ReturnForUse();
  return kj::READY_NOW;
}

kj::Promise<void> EndPointServer::send(SendContext context) {
  KJ_REQUIRE(_Open, "EndPoint closed already");
  auto capnpMsgPointer = context.getParams().getMsg();
  KJ_REQUIRE(capnpMsgPointer.isStruct(),
             "Only messages can go in the 'msg' parameter");
  auto msgSize = capnpMsgPointer.targetSize();
  auto builder = make_unique<MallocMessageBuilder>(
      msgSize.wordCount + 1, AllocationStrategy::FIXED_SIZE);
  builder->setRoot(capnpMsgPointer);
  auto segments = builder->getSegmentsForOutput();
  KJ_ASSERT(segments.size() == 1);
  auto fstSegmentData = segments[0].asBytes();

  auto blob = std::make_shared<EndPoint::Blob>(fstSegmentData.begin(),
                                               fstSegmentData.end());
  _EndPoint->PushMessageToSim(blob);
  // for (uint8_t b : *blob) {
  //     printf("%02X ", b);
  // }
  // printf("\n");
  return kj::READY_NOW;
}

kj::Promise<void> EndPointServer::recv(RecvContext context) {
  KJ_REQUIRE(_Open, "EndPoint closed already");
  KJ_REQUIRE(!context.getParams().getBlock(),
             "Blocking recv() not supported yet");

  EndPoint::BlobPtr blob;
  auto msgPresent = _EndPoint->GetMessageToClient(blob);
  context.getResults().setHasData(msgPresent);
  if (msgPresent) {
    KJ_REQUIRE(blob->size() % 8 == 0,
               "Response msg was malformed. Size of response was not a "
               "multiple of 8 bytes.");
    auto segment =
        kj::ArrayPtr<word>((word *)blob->data(), blob->size() / 8).asConst();
    auto segments = kj::heapArray({segment});
    auto msgReader = make_unique<SegmentArrayMessageReader>(segments);
    context.getResults().getResp().set(msgReader->getRoot<AnyPointer>());
  }
  return kj::READY_NOW;
}

kj::Promise<void> CosimServer::list(ListContext context) {
  auto ifaces =
      context.getResults().initIfaces((unsigned int)_Reg->EndPoints.size());
  unsigned int ctr = 0u;
  for (auto i = _Reg->EndPoints.begin(); i != _Reg->EndPoints.end(); i++) {
    ifaces[ctr].setEndpointID(i->first);
    ifaces[ctr].setTypeID(i->second->GetEsiTypeId());
    ctr++;
  }
  return kj::READY_NOW;
}

kj::Promise<void> CosimServer::open(OpenContext ctxt) {
  auto epIter =
      _Reg->EndPoints.find(ctxt.getParams().getIface().getEndpointID());
  KJ_REQUIRE(epIter != _Reg->EndPoints.end(), "Could not find endpoint");

  auto &ep = epIter->second;
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

  // OK, this is hacky as shit, but it unblocks me and isn't too inefficient
  while (!_Stop) {
    waitScope.poll();
    this_thread::sleep_for(chrono::milliseconds(1));
  }
}

void RpcServer::Run(uint16_t port) {
  if (_MainThread == nullptr) {
    _MainThread = new thread(&RpcServer::MainLoop, this, port);
  } else {
    throw runtime_error("Cannot Run() RPC server more than once!");
  }
}

void RpcServer::Stop() {
  if (_MainThread == nullptr) {
    throw runtime_error("RpcServer not Run()");
  } else if (!_Stop) {
    _Stop = true;
    _MainThread->join();
  }
}
