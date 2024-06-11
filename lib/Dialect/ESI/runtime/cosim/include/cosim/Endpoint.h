//===- Endpoint.h - Cosim endpoint server -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declare the class which is used to model DPI endpoints.
//
//===----------------------------------------------------------------------===//

#ifndef COSIM_ENDPOINT_H
#define COSIM_ENDPOINT_H

#include "esi/Common.h"

#include "cosim/Utils.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace esi {
namespace cosim {

enum Direction { ToSim, FromSim };

/// Implements a bi-directional, thread-safe bridge between the RPC server and
/// DPI functions.
///
/// Several of the methods below are inline with the declaration to make them
/// candidates for inlining during compilation. This is particularly important
/// on the simulation side since polling happens at each clock and we do not
/// want to slow down the simulation any more than necessary.
class Endpoint {
public:
  /// Representing messages as shared pointers to vectors may be a performance
  /// issue in the future but it is the easiest way to ensure memory
  /// correctness.
  using MessageDataPtr = std::unique_ptr<MessageData>;

  /// Construct an endpoint which knows and the type IDs in both directions.
  Endpoint(std::string id, std::string typeId, Direction dir);
  ~Endpoint();
  /// Disallow copying. There is only ONE endpoint object per logical endpoint
  /// so copying is almost always a bug.
  Endpoint(const Endpoint &) = delete;

  std::string getId() const { return id; }
  std::string getTypeId() const { return typeId; }
  Direction getDirection() const { return dir; }

  /// These two are used to set and unset the inUse flag, to ensure that an open
  /// endpoint is not opened again.
  bool setInUse();
  void returnForUse();

  /// Queue message to the simulation.
  void pushMessage(MessageDataPtr msg) { msgQueue.push(std::move(msg)); }

  /// Pop from the to-simulator queue. Return true if there was a message in the
  /// queue.
  bool getMessage(MessageDataPtr &msgOut) {
    if (messageCurrentlyBeingSent)
      return false;

    if (auto msg = msgQueue.pop()) {
      msgOut = std::move(*msg);
      return true;
    }
    return false;
  }

  /// Used in the rare case where an error is encountered while trying to send
  /// the message.
  void giveBackMessage(MessageDataPtr msg) {
    msgQueue.push_front(std::move(msg));
  }

  void confirmMessageSent() { messageCurrentlyBeingSent.reset(); }
  void failedToSendMessage() {
    if (messageCurrentlyBeingSent)
      // If the message was not sent, put it back in the queue.
      giveBackMessage(std::move(messageCurrentlyBeingSent));
  }

private:
  bool inUse;

  /// Message queue from RPC client to the simulation.
  TSQueue<MessageDataPtr> msgQueue;
  MessageDataPtr messageCurrentlyBeingSent;

  /// Endpoint name.
  std::string id;
  const std::string typeId;
  Direction dir;
};

/// The Endpoint registry is where Endpoints report their existence (register)
/// and they are looked up by RPC clients.
class EndpointRegistry {
public:
  /// Register an Endpoint. Creates the Endpoint object and owns it. Returns
  /// false if unsuccessful.
  bool registerEndpoint(std::string epId, std::string typeId, Direction dir);

  /// Get the specified endpoint. Return nullptr if it does not exist. This
  /// method is defined inline so it can be inlined at compile time. Performance
  /// is important here since this method is used in the polling call from the
  /// simulator. Returns nullptr if the endpoint cannot be found.
  Endpoint *operator[](const std::string &epId) {
    Lock g(m);
    auto it = endpoints.find(epId);
    if (it == endpoints.end())
      return nullptr;
    return &it->second;
  }

  /// Iterate over the list of endpoints, calling the provided function for each
  /// endpoint.
  void iterateEndpoints(
      const std::function<void(std::string id, const Endpoint &)> &f) const;
  /// Return the number of endpoints.
  size_t size() const;

private:
  using Lock = std::lock_guard<std::mutex>;

  /// This object needs to be thread-safe. An object-wide mutex is sufficient.
  mutable std::mutex m;

  /// Endpoint ID to object pointer mapping.
  std::map<std::string, Endpoint> endpoints;
};

} // namespace cosim
} // namespace esi

#endif
