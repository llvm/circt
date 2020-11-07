#ifndef CIRCT_DIALECT_ESI_COSIM_ENDPOINT_H
#define CIRCT_DIALECT_ESI_COSIM_ENDPOINT_H

//===- EndPoint.h - Cosim endpoint server ----------------------*- C++ -*-===//
//
// Declare the class which is used to model DPI endpoints.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/cosim/CosimDpi.capnp.h"
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>

#ifndef __ESI_ENDPOINT_HPP__
#define __ESI_ENDPOINT_HPP__

namespace circt {
namespace esi {
namespace cosim {

/// Implements a bi-directional, thread-safe bridge between the RPC server and
/// DPI functions. These methods are mostly inline to make them candidates for
/// inlining for performance reasons.
class Endpoint {
public:
  /// Representing messages as shared pointers to vectors may be a performance
  /// issue in the future but it is the easiest way to ensure memory
  /// correctness.
  using Blob = std::vector<uint8_t>;
  using BlobPtr = std::shared_ptr<Blob>;

private:
  const uint64_t esiTypeId;
  bool inUse;

  using Lock = std::lock_guard<std::mutex>;

  /// This class needs to be thread-safe. All of the mutable member variables
  /// are protected with this object-wide lock. This may be a performance issue
  /// in the future.
  std::mutex m;
  /// Message queue from RPC client to the simulation.
  std::queue<BlobPtr> toCosim;
  /// Message queue to RPC client from the simulation.
  std::queue<BlobPtr> toClient;

public:
  Endpoint(uint64_t esiTypeId, int maxSize);
  virtual ~Endpoint();
  /// Disallow copying. There is only ONE endpoint so copying is almost always a
  /// bug.
  Endpoint(const Endpoint &) = delete;

  uint64_t getEsiTypeId() const { return esiTypeId; }

  bool setInUse();
  void returnForUse();

  /// Queue message to the simulation.
  void pushMessageToSim(BlobPtr msg) {
    Lock g(m);
    toCosim.push(msg);
  }

  /// Pop from the to-simulator queue. Return true if there was a message in the
  /// queue.
  bool getMessageToSim(BlobPtr &msg) {
    Lock g(m);
    if (toCosim.size() > 0) {
      msg = toCosim.front();
      toCosim.pop();
      return true;
    }
    return false;
  }

  /// Queue message to the RPC client.
  void pushMessageToClient(BlobPtr msg) {
    Lock g(m);
    toClient.push(msg);
  }

  /// Pop from the to-RPC-client queue. Return true if there was a message in
  /// the queue.
  bool getMessageToClient(BlobPtr &msg) {
    Lock g(m);
    if (toClient.size() > 0) {
      msg = toClient.front();
      toClient.pop();
      return true;
    }
    return false;
  }
};

/// The Endpoint registry.
class EndpointRegistry {
  using Lock = std::lock_guard<std::mutex>;

public:
  ~EndpointRegistry();

  /// Takes ownership of ep
  void registerEndpoint(int epId, long long esiTypeId, int typeSize);

  /// Get the specified endpoint, if it exists. Return false if it does not.
  bool get(int epId, Endpoint *&);
  /// Get the specified endpoint, throwing an exception if it doesn't exist.
  Endpoint &operator[](int epId);
  /// Iterate over the list of endpoints, calling the provided function for each
  /// endpoint.
  void iterateEndpoints(std::function<void(int id, const Endpoint &)> f) const;
  /// Return the number of endpoints.
  size_t size() const;

private:
  /// This object needs to be thread-safe. An object-wide mutex is sufficient.
  std::mutex m;

  /// Endpoint ID to object pointer mapping.
  std::map<int, Endpoint> endpoints;
};

} // namespace cosim
} // namespace esi
} // namespace circt

#endif

#endif
