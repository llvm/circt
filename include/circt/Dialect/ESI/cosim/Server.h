

#include "EndPoint.h"
#include "circt/Dialect/ESI/cosim/CosimDpi.capnp.h"
#include <capnp/any.h>
#include <capnp/ez-rpc.h>
#include <iostream>
#include <kj/async.h>
#include <map>
#include <thread>

#ifndef __COSIM_DPI_SERVER_HPP__
#define __COSIM_DPI_SERVER_HPP__

class EndPointServer
    : public EsiDpiEndpoint<capnp::AnyPointer, capnp::AnyPointer>::Server {
  std::unique_ptr<EndPoint> &_EndPoint;
  bool _Open;

public:
  EndPointServer(std::unique_ptr<EndPoint> &ep) : _EndPoint(ep), _Open(true) {}

  virtual ~EndPointServer() {
    if (_Open)
      _EndPoint->ReturnForUse();
  }

  std::unique_ptr<EndPoint> &GetEndPoint() { return _EndPoint; }

  kj::Promise<void> send(SendContext);
  kj::Promise<void> recv(RecvContext);
  kj::Promise<void> close(CloseContext);
};

class CosimServer : public CosimDpiServer::Server {
  EndPointRegistry *_Reg;

public:
  CosimServer(EndPointRegistry *reg) : _Reg(reg) {}
  virtual ~CosimServer() {}

  kj::Promise<void> list(ListContext ctxt);
  kj::Promise<void> open(OpenContext ctxt);
};

class RpcServer {
  capnp::EzRpcServer *_RpcServer;
  std::thread *_MainThread;
  volatile bool _Stop;

  void MainLoop(uint16_t port);

public:
  EndPointRegistry EndPoints;

  RpcServer() : _RpcServer(nullptr), _MainThread(nullptr), _Stop(false) {}

  ~RpcServer();

  void Run(uint16_t port);
  void Stop();
};

#endif
