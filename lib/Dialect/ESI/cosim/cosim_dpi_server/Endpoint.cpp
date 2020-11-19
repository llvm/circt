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
    fprintf(stderr, "Warning: Returning an endpoint which was not in use.\n");
  inUse = false;
}

bool EndpointRegistry::registerEndpoint(int epId, uint64_t sendTypeId,
                                        int sendTypeMaxSize,
                                        uint64_t recvTypeId,
                                        int recvTypeMaxSize) {
  Lock g(m);
  if (endpoints.find(epId) != endpoints.end()) {
    fprintf(stderr, "Endpoint ID already exists!\n");
    return false;
  }
  // The following ugliness adds an Endpoint to the map of Endpoints. The
  // Endpoint class has its copy constructor deleted, thus the metaprogramming.
  endpoints.emplace(std::piecewise_construct,
                    // Map key.
                    std::forward_as_tuple(epId),
                    // Endpoint constructor args.
                    std::forward_as_tuple(sendTypeId, sendTypeMaxSize,
                                          recvTypeId, recvTypeMaxSize));
  return false;
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
