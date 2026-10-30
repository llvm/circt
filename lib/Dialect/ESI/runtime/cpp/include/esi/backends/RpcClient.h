//===- RpcClient.h - ESI Cosim RPC client -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public C++ interface for the cosim RPC client. The on-the-wire protocol is
// WebSocket + JSON; see cosim-protocol.md for the spec. This header exposes
// no transport-specific types, so the implementation (RpcClient.cpp) is free
// to evolve independently.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_BACKENDS_RPCCLIENT_H
#define ESI_BACKENDS_RPCCLIENT_H

#include "esi/Common.h"

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace esi {
class Logger;
namespace backends {
namespace cosim {

/// A client for the cosim RPC server. Hides the WebSocket + JSON transport
/// behind a small C++ API; see cosim-protocol.md for the wire format.
class RpcClient {
public:
  RpcClient(Logger &logger, const std::string &hostname, uint16_t port);
  ~RpcClient();

  // Non-copyable.
  RpcClient(const RpcClient &) = delete;
  RpcClient &operator=(const RpcClient &) = delete;

  /// Get the ESI version from the manifest.
  uint32_t getEsiVersion() const;

  /// Get the compressed manifest from the server.
  std::vector<uint8_t> getCompressedManifest() const;

  /// Channel direction as reported by the server.
  enum class ChannelDirection { ToServer, ToClient };

  /// Description of a channel from the server.
  struct ChannelDesc {
    std::string name;
    std::string type;
    ChannelDirection dir;
  };

  /// Get the channel description for a channel name.
  /// Returns true if the channel was found.
  bool getChannelDesc(const std::string &channelName, ChannelDesc &desc) const;

  /// List all channels available on the server.
  std::vector<ChannelDesc> listChannels() const;

  /// Called once with nullptr when a request completes, or with an exception
  /// if it fails. Runs on the client's network thread (or inline, if the
  /// connection is already lost), so it must be quick and must not block.
  using CompletionCallback = std::function<void(std::exception_ptr)>;

  /// Ask the server to report when the simulation has connected its end of a
  /// server-bound channel. Doesn't block: `onConnected` is called then
  /// (as soon as the server replies, if it already is), or with an exception
  /// if the connection is lost first. Frames sent before then are held by the
  /// server, not dropped.
  void connectToServer(const std::string &channelName,
                       CompletionCallback onConnected);

  /// Send a message to a server-bound channel.
  void writeToServer(const std::string &channelName, const MessageData &data);

  /// Callback type for receiving messages from a client-bound channel.
  /// Return true if the message was consumed, false to retry the same owning
  /// message object.
  using ReadCallback =
      std::function<bool(std::unique_ptr<SegmentedMessageData> &)>;

  /// Abstract handle for a read channel connection.
  /// Destructor disconnects from the channel.
  class ReadChannelConnection {
  public:
    virtual ~ReadChannelConnection() = default;
    virtual void disconnect() = 0;
  };

  /// Connect to a client-bound channel and receive messages via callback.
  /// Doesn't block: `onSubscribed` (if given) is called once the server has
  /// accepted the subscription, i.e. once the simulation side of the channel
  /// is connected, or with an exception if the subscription fails. Returns a
  /// handle that disconnects when destroyed.
  std::unique_ptr<ReadChannelConnection>
  connectClientReceiver(const std::string &channelName, ReadCallback callback,
                        CompletionCallback onSubscribed = nullptr);

  /// Hide the implementation details from this header file.
  class Impl;

private:
  std::unique_ptr<Impl> impl;
};

} // namespace cosim
} // namespace backends
} // namespace esi

#endif // ESI_BACKENDS_RPCCLIENT_H
