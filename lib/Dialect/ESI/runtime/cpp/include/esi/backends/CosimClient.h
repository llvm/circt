//===- CosimClient.h - ESI Cosim gRPC client --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the gRPC client implementation for ESI cosimulation.
// It wraps the gRPC/protobuf dependencies so they don't leak into Cosim.h.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp).
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_BACKENDS_COSIMCLIENT_H
#define ESI_BACKENDS_COSIMCLIENT_H

#include "esi/Accelerator.h"
#include "esi/Ports.h"

#include <memory>
#include <string>

namespace esi {
namespace backends {
namespace cosim {

/// Forward declaration of the internal client implementation.
class CosimClientImpl;

/// A gRPC client for communicating with the cosimulation server.
/// This class wraps all gRPC/protobuf dependencies.
class CosimClient {
public:
  CosimClient(const std::string &hostname, uint16_t port);
  ~CosimClient();

  // Non-copyable.
  CosimClient(const CosimClient &) = delete;
  CosimClient &operator=(const CosimClient &) = delete;

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

  /// Get the internal implementation (for use by channel ports).
  CosimClientImpl *getImpl() const { return impl.get(); }

private:
  std::unique_ptr<CosimClientImpl> impl;
};

/// Cosim client implementation of a write channel port.
class WriteCosimChannelPort : public WriteChannelPort {
public:
  WriteCosimChannelPort(AcceleratorConnection &conn, CosimClient &client,
                        const CosimClient::ChannelDesc &desc, const Type *type,
                        std::string name);
  ~WriteCosimChannelPort();

  void connectImpl(const ChannelPort::ConnectOptions &options) override;

protected:
  void writeImpl(const MessageData &data) override;
  bool tryWriteImpl(const MessageData &data) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/// Cosim client implementation of a read channel port. Since gRPC read protocol
/// streams messages back, this implementation is quite complex.
class ReadCosimChannelPort : public ReadChannelPort {
public:
  ReadCosimChannelPort(AcceleratorConnection &conn, CosimClient &client,
                       const CosimClient::ChannelDesc &desc, const Type *type,
                       std::string name);
  virtual ~ReadCosimChannelPort();

  void connectImpl(const ChannelPort::ConnectOptions &options) override;
  void disconnect() override;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace cosim
} // namespace backends
} // namespace esi

#endif // ESI_BACKENDS_COSIMCLIENT_H
