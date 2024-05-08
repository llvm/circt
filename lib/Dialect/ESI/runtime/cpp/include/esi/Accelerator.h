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
#include "esi/Manifest.h"
#include "esi/Ports.h"
#include "esi/Services.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>

namespace esi {

//===----------------------------------------------------------------------===//
// Constants used by low-level APIs.
//===----------------------------------------------------------------------===//

constexpr uint32_t MetadataOffset = 8;
constexpr uint32_t MagicNumberLo = 0xE5100E51;
constexpr uint32_t MagicNumberHi = 0x207D98E5;
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
/// subclasses.
class AcceleratorConnection {
public:
  AcceleratorConnection(Context &ctxt) : ctxt(ctxt) {}

  virtual ~AcceleratorConnection() = default;
  Context &getCtxt() const { return ctxt; }

  /// Request the host side channel ports for a particular instance (identified
  /// by the AppID path). For convenience, provide the bundle type.
  virtual std::map<std::string, ChannelPort &>
  requestChannelsFor(AppIDPath, const BundleType *) = 0;

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

  /// ESI accelerator context.
  Context &ctxt;
};

namespace registry {

// Connect to an ESI accelerator given a backend name and connection specifier.
// Alternatively, instantiate the backend directly (if you're using C++).
std::unique_ptr<AcceleratorConnection>
connect(Context &ctxt, std::string backend, std::string connection);

namespace internal {

/// Backends can register themselves to be connected via a connection string.
using BackendCreate = std::function<std::unique_ptr<AcceleratorConnection>(
    Context &, std::string)>;
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
