//===- RpcServer.h - Run a cosim server -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Setup and run a server accepting connections via the 'cosim' RPC protocol.
// Then, one can request ports to and from the clients.
//
// Abstract this out to support multi-party communication in the future.
//
//===----------------------------------------------------------------------===//

#ifndef ESI_COSIM_RPCSERVER_H
#define ESI_COSIM_RPCSERVER_H

#include "esi/Context.h"
#include "esi/Ports.h"

namespace esi {
namespace cosim {

/// TODO: make this a proper backend (as much as possible).
class RpcServer {
public:
  RpcServer(Context &ctxt);
  ~RpcServer();

  /// Get the context.
  Context &getContext() { return ctxt; }

  /// Set the manifest and version. There is a race condition here in that the
  /// RPC server can be started and a connection from the client could happen
  /// before the manifest is set. TODO: rework the DPI API to require that the
  /// manifest gets set first.
  void setManifest(int esiVersion,
                   const std::vector<uint8_t> &compressedManifest);

  /// Register a read or write port which communicates over RPC.
  ReadChannelPort &registerReadPort(const std::string &name,
                                    const std::string &type);
  WriteChannelPort &registerWritePort(const std::string &name,
                                      const std::string &type);

  void stop();

  // Start the RPC server. If no port is provided, the RPC server will let the
  // OS pick a port.
  void run(int port = -1);

  // Return which port the RPC server is executing on.
  int getPort();

  /// Hide the implementation details from this header file.
  class Impl;

private:
  Context &ctxt;
  std::unique_ptr<Impl> impl;
};

} // namespace cosim
} // namespace esi

#endif // ESI_COSIM_RPCSERVER_H
