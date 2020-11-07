//===- EndPoint.cpp - Definitions for EndPointRegistry ----------*- C++ -*-===//
//
// Definitions for Cosim EndPoint and EndPointRegistry.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/cosim/EndPoint.h"

using namespace circt::esi::cosim;

Endpoint::Endpoint(uint64_t EsiTypeId, int MaxSize)
    : esiTypeId(EsiTypeId), inUse(false) {}
Endpoint::~Endpoint() {}

bool Endpoint::setInUse() {
  Lock g(m);
  if (inUse)
    return false;
  inUse = true;
  return true;
}

void Endpoint::returnForUse() {
  Lock g(m);
  if (!inUse)
    throw std::runtime_error("Cannot return something not in use!");
  inUse = false;
}

EndpointRegistry::~EndpointRegistry() {
  Lock g(m);
  endpoints.clear();
}

void EndpointRegistry::registerEndpoint(int ep_id, long long esi_type_id,
                                        int type_size) {
  Lock g(m);
  if (endpoints.find(ep_id) != endpoints.end()) {
    throw std::runtime_error("Endpoint ID already exists!");
  }
  endpoints.emplace(std::piecewise_construct, std::forward_as_tuple(ep_id),
                    std::forward_as_tuple(esi_type_id, type_size));
}

bool EndpointRegistry::get(int ep_id, Endpoint *&ep) {
  Lock g(m);
  auto it = endpoints.find(ep_id);
  if (it == endpoints.end())
    return false;
  ep = &it->second;
  return true;
}

Endpoint &EndpointRegistry::operator[](int ep_id) {
  Lock g(m);
  auto it = endpoints.find(ep_id);
  if (it == endpoints.end())
    throw std::runtime_error("Could not locate Endpoint");
  return it->second;
}

void EndpointRegistry::iterateEndpoints(
    std::function<void(int, const Endpoint &)> F) const {
  // This function is logically const, but modification is needed to obtain a
  // lock.
  Lock g(const_cast<EndpointRegistry *>(this)->m);
  for (const auto &ep : endpoints) {
    F(ep.first, ep.second);
  }
}

size_t EndpointRegistry::size() const {
  // This function is logically const, but modification is needed to obtain a
  // lock.
  Lock g(const_cast<EndpointRegistry *>(this)->m);
  return endpoints.size();
}
