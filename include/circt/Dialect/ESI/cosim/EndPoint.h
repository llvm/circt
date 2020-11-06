//===- EndPoint.h - Cosim endpoint server ----------------------*- C++ -*-===//
//
// Declare the class which is used to model DPI endpoints.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/cosim/CosimDpi.capnp.h"
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>

#ifndef __ESI_ENDPOINT_HPP__
#define __ESI_ENDPOINT_HPP__

class EndPoint {
public:
  typedef std::vector<uint8_t> Blob;
  typedef std::shared_ptr<Blob> BlobPtr;

private:
  uint64_t _EsiTypeId;
  bool _InUse;

  typedef std::lock_guard<std::mutex> Lock;
  std::mutex _M; // Object-wide mutex
  std::queue<BlobPtr> toCosim;
  std::queue<BlobPtr> toClient;

public:
  EndPoint(uint64_t EsiTypeId, int MaxSize)
      : _EsiTypeId(EsiTypeId), _InUse(false) {}
  virtual ~EndPoint() {}

  uint64_t GetEsiTypeId() { return _EsiTypeId; }

  bool SetInUse() {
    Lock g(_M);
    if (_InUse)
      return false;
    _InUse = true;
    return true;
  }

  void ReturnForUse() {
    Lock g(_M);
    if (!_InUse)
      throw std::runtime_error("Cannot return something not in use!");
    _InUse = false;
  }

  void PushMessageToSim(BlobPtr msg) {
    Lock g(_M);
    toCosim.push(msg);
  }

  bool GetMessageToSim(BlobPtr &msg) {
    Lock g(_M);
    if (toCosim.size() > 0) {
      msg = toCosim.front();
      toCosim.pop();
      return true;
    }
    return false;
  }

  void PushMessageToClient(BlobPtr msg) {
    Lock g(_M);
    toClient.push(msg);
  }

  bool GetMessageToClient(BlobPtr &msg) {
    Lock g(_M);
    if (toClient.size() > 0) {
      msg = toClient.front();
      toClient.pop();
      return true;
    }
    return false;
  }
};

class EndPointRegistry {
  typedef std::lock_guard<std::mutex> Lock;
  std::mutex _M; // Object-wide mutex

public:
  std::map<int, std::unique_ptr<EndPoint>> EndPoints;

  ~EndPointRegistry();

  /// Takes ownership of ep
  void RegisterEndPoint(int ep_id, long long esi_type_id, int type_size);

  std::unique_ptr<EndPoint> &operator[](int ep_id) {
    Lock g(_M);
    auto ep = EndPoints.find(ep_id);
    if (ep == EndPoints.end())
      throw std::runtime_error("Could not find endpoint");
    return ep->second;
  }
};

#endif
