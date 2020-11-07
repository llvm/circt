//===- EndPoint.cpp - Definitions for EndPointRegistry ----------*- C++ -*-===//
//
// Definitions for Cosim EndPoint and EndPointRegistry.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/cosim/EndPoint.h"

using namespace circt::esi::cosim;

EndPoint::EndPoint(uint64_t EsiTypeId, int MaxSize)
    : _EsiTypeId(EsiTypeId), _InUse(false) {}
EndPoint::~EndPoint() {}

bool EndPoint::SetInUse() {
  Lock g(_M);
  if (_InUse)
    return false;
  _InUse = true;
  return true;
}

void EndPoint::ReturnForUse() {
  Lock g(_M);
  if (!_InUse)
    throw std::runtime_error("Cannot return something not in use!");
  _InUse = false;
}

EndPointRegistry::~EndPointRegistry() {
  Lock g(_M);
  EndPoints.clear();
}

void EndPointRegistry::RegisterEndPoint(int ep_id, long long esi_type_id,
                                        int type_size) {
  Lock g(_M);
  if (EndPoints.find(ep_id) != EndPoints.end()) {
    throw std::runtime_error("Endpoint ID already exists!");
  }
  EndPoints.emplace(std::piecewise_construct, std::forward_as_tuple(ep_id),
                    std::forward_as_tuple(esi_type_id, type_size));
}

bool EndPointRegistry::Get(int ep_id, EndPoint *&ep) {
  Lock g(_M);
  auto it = EndPoints.find(ep_id);
  if (it == EndPoints.end())
    return false;
  ep = &it->second;
  return true;
}

EndPoint &EndPointRegistry::operator[](int ep_id) {
  Lock g(_M);
  auto it = EndPoints.find(ep_id);
  if (it == EndPoints.end())
    throw std::runtime_error("Could not locate Endpoint");
  return it->second;
}

void EndPointRegistry::IterateEndpoints(
    std::function<void(int, const EndPoint &)> F) const {
  // This function is logically const, but modification is needed to obtain a
  // lock.
  Lock g(const_cast<EndPointRegistry *>(this)->_M);
  for (const auto &ep : EndPoints) {
    F(ep.first, ep.second);
  }
}

size_t EndPointRegistry::Size() const {
  // This function is logically const, but modification is needed to obtain a
  // lock.
  Lock g(const_cast<EndPointRegistry *>(this)->_M);
  return EndPoints.size();
}
