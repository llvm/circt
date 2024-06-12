//===- Server.h - Run a cosim server ----------------------------*- C++ -*-===//
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

#include "esi/Ports.h"

namespace esi {
namespace cosim {

class RpcServer {
public:
  ~RpcServer();

  void setManifest(int esiVersion, std::vector<uint8_t> compressedManifest);

  ReadChannelPort &registerReadPort(const std::string &name,
                                    const std::string &type);
  WriteChannelPort &registerWritePort(const std::string &name,
                                      const std::string &type);

  void stop();
  void run(int port);

  class Impl;

private:
  Impl *impl;
};

} // namespace cosim
} // namespace esi

#endif // ESI_COSIM_RPCSERVER_H
