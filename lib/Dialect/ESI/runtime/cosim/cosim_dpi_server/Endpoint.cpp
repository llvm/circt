//===- EndPoint.cpp - Definitions for EndPointRegistry ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for Cosim EndPoint and EndPointRegistry.
//
//===----------------------------------------------------------------------===//

#include "cosim/Endpoint.h"

using namespace esi::cosim;

Endpoint::Endpoint(std::string fromHostTypeId, int fromHostTypeMaxSize,
                   std::string toHostTypeId, int toHostTypeMaxSize)
    : fromHostTypeId(fromHostTypeId), toHostTypeId(toHostTypeId), inUse(false) {
}
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

bool EndpointRegistry::registerEndpoint(std::string epId,
                                        std::string fromHostTypeId,
                                        int fromHostTypeMaxSize,
                                        std::string toHostTypeId,
                                        int toHostTypeMaxSize) {
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
                    std::forward_as_tuple(fromHostTypeId, fromHostTypeMaxSize,
                                          toHostTypeId, toHostTypeMaxSize));
  return true;
}

void EndpointRegistry::iterateEndpoints(
    const std::function<void(std::string, const Endpoint &)> &f) const {
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
