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

  /// Default maximum number of frames sent to a server-bound channel which
  /// the simulation has not yet consumed.
  static constexpr unsigned DefaultToServerMaxInFlight = 16;
  /// Default maximum number of frames in flight (queued in the simulator or
  /// sent but not yet consumed by the client) on a client-bound channel.
  static constexpr unsigned DefaultToClientMaxInFlight = 32;

  /// Called once with nullptr when a request completes, or with an exception
  /// if it fails. Runs on the client's network thread (or inline, if the
  /// connection is already lost), so it must be quick and must not block.
  using CompletionCallback = std::function<void(std::exception_ptr)>;

  /// Ask the server to report when the simulation has connected its end of a
  /// server-bound channel. Doesn't block: `onConnected` is called then
  /// (as soon as the server replies, if it already is), or with an exception
  /// if the connection is lost first. Frames sent before then are held by the
  /// server and backpressured, not dropped.
  void connectToServer(const std::string &channelName,
                       CompletionCallback onConnected);

  /// Send a message to a server-bound channel. Blocks while the channel's
  /// in-flight limit is reached. Throws if the connection is lost.
  void writeToServer(const std::string &channelName, const MessageData &data);

  /// Non-blocking send of one message, split into `frames`, to a server-bound
  /// channel. If the channel is below its in-flight limit, sends all of them
  /// (possibly overshooting the limit by `frames.size() - 1`)
  /// and returns true. Otherwise sends nothing and returns false.
  bool tryWriteToServer(const std::string &channelName,
                        const std::vector<MessageData> &frames);

  /// Set the maximum number of unconsumed frames on a server-bound channel.
  /// Zero means unbounded.
  void setToServerMaxInFlight(const std::string &channelName,
                              unsigned maxInFlight);

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
  /// is connected, or with an exception if the subscription fails.
  /// `maxInFlight` is the maximum number of frames the simulator may have in
  /// flight for this channel; zero means unbounded. A frame stops counting
  /// once `callback` returns true for it. Returns a handle that disconnects
  /// when destroyed.
  std::unique_ptr<ReadChannelConnection>
  connectClientReceiver(const std::string &channelName, ReadCallback callback,
                        CompletionCallback onSubscribed = nullptr,
                        unsigned maxInFlight = DefaultToClientMaxInFlight);

  /// Hide the implementation details from this header file.
  class Impl;

private:
  std::unique_ptr<Impl> impl;
};

} // namespace cosim
} // namespace backends
} // namespace esi

#endif // ESI_BACKENDS_RPCCLIENT_H
