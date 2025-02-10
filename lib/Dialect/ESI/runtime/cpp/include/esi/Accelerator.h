//===- Accelerator.h - Base ESI runtime API ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic ESI APIs. The 'Accelerator' class is the superclass for all accelerator
// backends. It should (usually) provide enough functionality such that users do
// not have to interact with the platform-specific backend implementation with
// the exception of connecting to the accelerator.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_ACCELERATOR_H
#define ESI_ACCELERATOR_H

#include "esi/Context.h"
#include "esi/Design.h"
#include "esi/Engines.h"
#include "esi/Manifest.h"
#include "esi/Ports.h"
#include "esi/Services.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>

namespace esi {
// Forward declarations.
class AcceleratorServiceThread;

//===----------------------------------------------------------------------===//
// Constants used by low-level APIs.
//===----------------------------------------------------------------------===//

constexpr uint32_t MetadataOffset = 8;
constexpr uint64_t MagicNumberLo = 0xE5100E51;
constexpr uint64_t MagicNumberHi = 0x207D98E5;
constexpr uint64_t MagicNumber = MagicNumberLo | (MagicNumberHi << 32);
constexpr uint32_t ExpectedVersionNumber = 0;

//===----------------------------------------------------------------------===//
// Accelerator design hierarchy root.
//===----------------------------------------------------------------------===//

/// Top level accelerator class. Maintains a shared pointer to the manifest,
/// which owns objects used in the design hierarchy owned by this class. Since
/// this class owns the entire design hierarchy, when it gets destroyed the
/// entire design hierarchy gets destroyed so all of the instances, ports, etc.
/// are no longer valid pointers.
class Accelerator : public HWModule {
public:
  Accelerator() = delete;
  Accelerator(const Accelerator &) = delete;
  ~Accelerator() = default;
  Accelerator(std::optional<ModuleInfo> info,
              std::vector<std::unique_ptr<Instance>> children,
              std::vector<services::Service *> services,
              std::vector<std::unique_ptr<BundlePort>> &ports)
      : HWModule(info, std::move(children), services, ports) {}
};

//===----------------------------------------------------------------------===//
// Connection to the accelerator and its services.
//===----------------------------------------------------------------------===//

/// Abstract class representing a connection to an accelerator. Actual
/// connections (e.g. to a co-simulation or actual device) are implemented by
/// subclasses. No methods in here are thread safe.
class AcceleratorConnection {
public:
  AcceleratorConnection(Context &ctxt);
  virtual ~AcceleratorConnection();
  Context &getCtxt() const { return ctxt; }
  Logger &getLogger() const { return ctxt.getLogger(); }

  /// Disconnect from the accelerator cleanly.
  virtual void disconnect();

  // While building the design, keep around a std::map of active services
  // indexed by the service name. When a new service is encountered during
  // descent, add it to the table (perhaps overwriting one). Modifications to
  // the table only apply to the current branch, so copy this and update it at
  // each level of the tree.
  using ServiceTable = std::map<std::string, services::Service *>;

  /// Return a pointer to the accelerator 'service' thread (or threads). If the
  /// thread(s) are not running, they will be started when this method is
  /// called. `std::thread` is used. If users don't want the runtime to spin up
  /// threads, don't call this method. `AcceleratorServiceThread` is owned by
  /// AcceleratorConnection and governed by the lifetime of the this object.
  AcceleratorServiceThread *getServiceThread();

  using Service = services::Service;
  /// Get a typed reference to a particular service type. Caller does *not* take
  /// ownership of the returned pointer -- the Accelerator object owns it.
  /// Pointer lifetime ends with the Accelerator lifetime.
  template <typename ServiceClass>
  ServiceClass *getService(AppIDPath id = {}, std::string implName = {},
                           ServiceImplDetails details = {},
                           HWClientDetails clients = {}) {
    return dynamic_cast<ServiceClass *>(
        getService(typeid(ServiceClass), id, implName, details, clients));
  }
  /// Calls `createService` and caches the result. Subclasses can override if
  /// they want to use their own caching mechanism.
  virtual Service *getService(Service::Type service, AppIDPath id = {},
                              std::string implName = {},
                              ServiceImplDetails details = {},
                              HWClientDetails clients = {});

  /// Assume ownership of an accelerator object. Ties the lifetime of the
  /// accelerator to this connection. Returns a raw pointer to the object.
  Accelerator *takeOwnership(std::unique_ptr<Accelerator> accel);

  /// Create a new engine for channel communication with the accelerator. The
  /// default is to call the global `createEngine` to get an engine which has
  /// registered itself. Individual accelerator connection backends can override
  /// this to customize behavior.
  virtual void createEngine(const std::string &engineTypeName, AppIDPath idPath,
                            const ServiceImplDetails &details,
                            const HWClientDetails &clients);
  virtual const BundleEngineMap &getEngineMapFor(AppIDPath id) {
    return clientEngines[id];
  }

  Accelerator &getAccelerator() {
    if (!ownedAccelerator)
      throw std::runtime_error(
          "AcceleratorConnection does not own an accelerator");
    return *ownedAccelerator;
  }

protected:
  /// If `createEngine` is overridden, this method should be called to register
  /// the engine and all of the channels it services.
  void registerEngine(AppIDPath idPath, std::unique_ptr<Engine> engine,
                      const HWClientDetails &clients);

  /// Called by `getServiceImpl` exclusively. It wraps the pointer returned by
  /// this in a unique_ptr and caches it. Separate this from the
  /// wrapping/caching since wrapping/caching is an implementation detail.
  virtual Service *createService(Service::Type service, AppIDPath idPath,
                                 std::string implName,
                                 const ServiceImplDetails &details,
                                 const HWClientDetails &clients) = 0;

  /// Collection of owned engines.
  std::map<AppIDPath, std::unique_ptr<Engine>> ownedEngines;
  /// Mapping of clients to their servicing engines.
  std::map<AppIDPath, BundleEngineMap> clientEngines;

private:
  /// ESI accelerator context.
  Context &ctxt;

  /// Cache services via a unique_ptr so they get free'd automatically when
  /// Accelerator objects get deconstructed.
  using ServiceCacheKey = std::tuple<const std::type_info *, AppIDPath>;
  std::map<ServiceCacheKey, std::unique_ptr<Service>> serviceCache;

  std::unique_ptr<AcceleratorServiceThread> serviceThread;

  /// Accelerator object owned by this connection.
  std::unique_ptr<Accelerator> ownedAccelerator;
};

namespace registry {

// Connect to an ESI accelerator given a backend name and connection specifier.
// Alternatively, instantiate the backend directly (if you're using C++).
std::unique_ptr<AcceleratorConnection> connect(Context &ctxt,
                                               const std::string &backend,
                                               const std::string &connection);

namespace internal {

/// Backends can register themselves to be connected via a connection string.
using BackendCreate = std::function<std::unique_ptr<AcceleratorConnection>(
    Context &, std::string)>;
void registerBackend(const std::string &name, BackendCreate create);

// Helper struct to
template <typename TAccelerator>
struct RegisterAccelerator {
  RegisterAccelerator(const char *name) {
    registerBackend(name, &TAccelerator::connect);
  }
};

#define REGISTER_ACCELERATOR(Name, TAccelerator)                               \
  static ::esi::registry::internal::RegisterAccelerator<TAccelerator>          \
  __register_accel____LINE__(Name)

} // namespace internal
} // namespace registry

/// Background thread which services various requests. Currently, it listens on
/// ports and calls callbacks for incoming messages on said ports.
class AcceleratorServiceThread {
public:
  AcceleratorServiceThread();
  ~AcceleratorServiceThread();

  /// When there's data on any of the listenPorts, call the callback. Callable
  /// from any thread.
  void
  addListener(std::initializer_list<ReadChannelPort *> listenPorts,
              std::function<void(ReadChannelPort *, MessageData)> callback);

  /// Poll this module.
  void addPoll(HWModule &module);

  /// Instruct the service thread to stop running.
  void stop();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace esi

#endif // ESI_ACCELERATOR_H
