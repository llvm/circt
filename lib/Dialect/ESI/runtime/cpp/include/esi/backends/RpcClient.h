//===- RpcClient.h - ESI Cosim RPC client -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the gRPC client implementation for ESI cosimulation.
// It wraps all gRPC/protobuf dependencies so they don't leak into other headers.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp).
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_BACKENDS_RPCCLIENT_H
#define ESI_BACKENDS_RPCCLIENT_H

#include "esi/Common.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace esi {
namespace backends {
namespace cosim {

/// A gRPC client for communicating with the cosimulation server.
/// This class wraps all gRPC/protobuf dependencies.
class RpcClient {
public:
  RpcClient(const std::string &hostname, uint16_t port);
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

  /// Send a message to a server-bound channel.
  void writeToServer(const std::string &channelName, const MessageData &data);

  /// Callback type for receiving messages from a client-bound channel.
  /// Return true if the message was consumed, false to retry.
  using ReadCallback = std::function<bool(const MessageData &)>;

  /// Abstract handle for a read channel connection.
  /// Destructor disconnects from the channel.
  class ReadChannelConnection {
  public:
    virtual ~ReadChannelConnection() = default;
    virtual void disconnect() = 0;
  };

  /// Connect to a client-bound channel and receive messages via callback.
  /// Returns a handle that disconnects when destroyed.
  std::unique_ptr<ReadChannelConnection>
  connectClientReceiver(const std::string &channelName, ReadCallback callback);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace cosim
} // namespace backends
} // namespace esi

#endif // ESI_BACKENDS_RPCCLIENT_H
