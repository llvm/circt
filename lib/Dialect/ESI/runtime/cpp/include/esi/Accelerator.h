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

#include "esi/Manifest.h"
#include "esi/Ports.h"
#include "esi/Services.h"

#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>

namespace esi {

//===----------------------------------------------------------------------===//
// Constants used by low-level APIs.
//===----------------------------------------------------------------------===//

constexpr uint32_t MagicNumOffset = 16;
constexpr uint32_t MagicNumberLo = 0xE5100E51;
constexpr uint32_t MagicNumberHi = 0x207D98E5;
constexpr uint32_t VersionNumberOffset = MagicNumOffset + 8;
constexpr uint32_t ExpectedVersionNumber = 0;

//===----------------------------------------------------------------------===//
// Accelerator design hierarchy.
//===----------------------------------------------------------------------===//

class Instance;

class HWModule {
public:
  HWModule(std::optional<ModuleInfo> info,
           std::vector<std::unique_ptr<Instance>> children,
           std::vector<services::Service *> services,
           std::vector<BundlePort> ports);

  std::optional<ModuleInfo> getInfo() const { return info; }
  const std::vector<std::unique_ptr<Instance>> &getChildrenOrdered() const {
    return children;
  }
  const std::map<AppID, Instance *> &getChildren() const { return childIndex; }
  const std::vector<BundlePort> &getPortsOrdered() const { return ports; }
  const std::map<AppID, const BundlePort &> &getPorts() const {
    return portIndex;
  }

protected:
  const std::optional<ModuleInfo> info;
  const std::vector<std::unique_ptr<Instance>> children;
  const std::map<AppID, Instance *> childIndex;
  const std::vector<services::Service *> services;
  const std::vector<BundlePort> ports;
  const std::map<AppID, const BundlePort &> portIndex;
};

class Instance : public HWModule {
public:
  Instance() = delete;
  Instance(const Instance &) = delete;
  ~Instance() = default;
  Instance(AppID id, std::optional<ModuleInfo> info,
           std::vector<std::unique_ptr<Instance>> children,
           std::vector<services::Service *> services,
           std::vector<BundlePort> ports)
      : HWModule(info, std::move(children), services, ports), id(id) {}

  const AppID getID() const { return id; }

protected:
  const AppID id;
};

class Accelerator : public HWModule {
public:
  Accelerator() = delete;
  Accelerator(const Accelerator &) = delete;
  ~Accelerator() = default;
  Accelerator(std::optional<ModuleInfo> info,
              std::vector<std::unique_ptr<Instance>> children,
              std::vector<services::Service *> services,
              std::vector<BundlePort> ports,
              std::shared_ptr<Manifest::Impl> manifestImpl)
      : HWModule(info, std::move(children), services, ports),
        manifestImpl(manifestImpl) {}

private:
  std::shared_ptr<Manifest::Impl> manifestImpl;
};

//===----------------------------------------------------------------------===//
// Connection to the accelerator and its services.
//===----------------------------------------------------------------------===//

class AcceleratorConnection {
public:
  virtual ~AcceleratorConnection() = default;

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

protected:
  /// Called by `getServiceImpl` exclusively. It wraps the pointer returned by
  /// this in a unique_ptr and caches it. Separate this from the
  /// wrapping/caching since wrapping/caching is an implementation detail.
  virtual Service *createService(Service::Type service, AppIDPath idPath,
                                 std::string implName,
                                 const ServiceImplDetails &details,
                                 const HWClientDetails &clients) = 0;

private:
  /// Cache services via a unique_ptr so they get free'd automatically when
  /// Accelerator objects get deconstructed.
  using ServiceCacheKey = std::tuple<const std::type_info *, AppIDPath>;
  std::map<ServiceCacheKey, std::unique_ptr<Service>> serviceCache;
};

namespace registry {

// Connect to an ESI accelerator given a backend name and connection specifier.
// Alternatively, instantiate the backend directly (if you're using C++).
std::unique_ptr<AcceleratorConnection> connect(std::string backend,
                                               std::string connection);

namespace internal {

/// Backends can register themselves to be connected via a connection string.
using BackendCreate =
    std::function<std::unique_ptr<AcceleratorConnection>(std::string)>;
void registerBackend(std::string name, BackendCreate create);

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
} // namespace esi

#endif // ESI_ACCELERATOR_H
