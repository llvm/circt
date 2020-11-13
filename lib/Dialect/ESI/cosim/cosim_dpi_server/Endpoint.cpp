//===- EndPoint.cpp - Definitions for EndPointRegistry ----------*- C++ -*-===//
//
// Definitions for Cosim EndPoint and EndPointRegistry.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/cosim/Endpoint.h"

using namespace circt::esi::cosim;

Endpoint::Endpoint(uint64_t sendTypeId, int sendTypeMaxSize,
                   uint64_t recvTypeId, int recvTypeMaxSize)
    : sendTypeId(sendTypeId), recvTypeId(recvTypeId), inUse(false) {}
Endpoint::~Endpoint() {}

uint64_t Endpoint::getSendTypeId() const { return sendTypeId; }
uint64_t Endpoint::getRecvTypeId() const { return recvTypeId; }

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

void EndpointRegistry::registerEndpoint(int epId, uint64_t sendTypeId,
                                        int sendTypeMaxSize,
                                        uint64_t recvTypeId,
                                        int recvTypeMaxSize) {
  Lock g(m);
  if (endpoints.find(epId) != endpoints.end())
    throw std::runtime_error("Endpoint ID already exists!");
  endpoints.emplace(std::piecewise_construct, std::forward_as_tuple(epId),
                    std::forward_as_tuple(sendTypeId, sendTypeMaxSize,
                                          recvTypeId, recvTypeMaxSize));
}

bool EndpointRegistry::get(int epId, Endpoint *&ep) {
  Lock g(m);
  auto it = endpoints.find(epId);
  if (it == endpoints.end())
    return false;
  ep = &it->second;
  return true;
}

void EndpointRegistry::iterateEndpoints(
    std::function<void(int, const Endpoint &)> f) const {
  // This function is logically const, but modification is needed to obtain a
  // lock.
  Lock g(const_cast<EndpointRegistry *>(this)->m);
  for (const auto &ep : endpoints) {
    f(ep.first, ep.second);
  }
}

size_t EndpointRegistry::size() const {
  // This function is logically const, but modification is needed to obtain a
  // lock.
  Lock g(const_cast<EndpointRegistry *>(this)->m);
  return endpoints.size();
}
