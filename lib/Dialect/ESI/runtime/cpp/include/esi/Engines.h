//===- Engines.h - Implement port communication -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//
//
// Engines (as in DMA engine) implement the actual communication between the
// host and the accelerator. They are low level of the ESI runtime API and are
// not intended to be used directly by users.
//
// They are called "engines" rather than "DMA engines" since communication need
// not be implemented via DMA.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_ENGINGES_H
#define ESI_ENGINGES_H

#include "esi/Common.h"
#include "esi/Ports.h"
#include "esi/Services.h"
#include "esi/Utils.h"

#include <cassert>
#include <future>

namespace esi {
class Accelerator;

/// Engines implement the actual channel communication between the host and the
/// accelerator. Engines can support multiple channels. They are low level of
/// the ESI runtime API and are not intended to be used directly by users.
class Engine {
public:
  Engine(AcceleratorConnection &conn) : connected(false), conn(conn) {}
  virtual ~Engine() = default;
  /// Start the engine, if applicable.
  virtual void connect() { connected = true; };
  /// Stop the engine, if applicable.
  virtual void disconnect() { connected = false; };
  /// Get a port for a channel, from the cache if it exists or create it. An
  /// engine may override this method if different behavior is desired.
  virtual ChannelPort &requestPort(AppIDPath idPath,
                                   const std::string &channelName,
                                   BundleType::Direction dir, const Type *type);

protected:
  /// Each engine needs to know how to create a ports. This method is called if
  /// a port doesn't exist in the engine cache.
  virtual std::unique_ptr<ChannelPort>
  createPort(AppIDPath idPath, const std::string &channelName,
             BundleType::Direction dir, const Type *type) = 0;

  bool connected;
  AcceleratorConnection &conn;

private:
  std::map<std::pair<AppIDPath, std::string>, std::unique_ptr<ChannelPort>>
      ownedPorts;
};

/// Since engines can support multiple channels BUT not necessarily all of the
/// channels in a bundle, a mapping from bundle channels to engines is needed.
class BundleEngineMap {
  friend class AcceleratorConnection;

public:
  /// Request ports for all the channels in a bundle. If the engine doesn't
  /// exist for a particular channel, skip said channel.
  PortMap requestPorts(const AppIDPath &idPath,
                       const BundleType *bundleType) const;

private:
  /// Set a particlar engine for a particular channel. Should only be called by
  /// AcceleratorConnection while registering engines.
  void setEngine(const std::string &channelName, Engine *engine);
  std::map<std::string, Engine *> bundleEngineMap;
};

namespace registry {

/// Create an engine by name. This is the primary way to create engines for
/// "normal" backends.
std::unique_ptr<Engine> createEngine(AcceleratorConnection &conn,
                                     const std::string &dmaEngineName,
                                     AppIDPath idPath,
                                     const ServiceImplDetails &details,
                                     const HWClientDetails &clients);

namespace internal {

/// Engines can register themselves for pluggable functionality.
using EngineCreate = std::function<std::unique_ptr<Engine>(
    AcceleratorConnection &conn, AppIDPath idPath,
    const ServiceImplDetails &details, const HWClientDetails &clients)>;
void registerEngine(const std::string &name, EngineCreate create);

/// Helper struct to register engines.
template <typename TEngine>
struct RegisterEngine {
  RegisterEngine(const char *name) { registerEngine(name, &TEngine::create); }
};

#define CONCAT_(prefix, suffix) prefix##suffix
#define CONCAT(prefix, suffix) CONCAT_(prefix, suffix)
#define REGISTER_ENGINE(Name, TEngine)                                         \
  static ::esi::registry::internal::RegisterEngine<TEngine> CONCAT(            \
      __register_engine__, __LINE__)(Name)

} // namespace internal
} // namespace registry

} // namespace esi

#endif // ESI_PORTS_H
