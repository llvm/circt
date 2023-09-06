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

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>

namespace esi {

constexpr uint32_t MagicNumOffset = 16;
constexpr uint32_t MagicNumberLo = 0xE5100E51;
constexpr uint32_t MagicNumberHi = 0x207D98E5;
constexpr uint32_t VersionNumberOffset = MagicNumOffset + 8;
constexpr uint32_t ExpectedVersionNumber = 0;

namespace services {
/// Parent class of all APIs modeled as 'services'. May or may not map to a
/// hardware side 'service'.
class Service {
public:
  using Type = const std::type_info &;
  virtual ~Service() = default;
};
} // namespace services

/// An ESI accelerator system.
class Accelerator {
public:
  virtual ~Accelerator() = default;

  /// Get a typed reference to a particular service type. Caller does *not* take
  /// ownership of the returned pointer -- the Accelerator object owns it.
  /// Pointer lifetime ends with the Accelerator lifetime.
  template <typename ServiceClass>
  ServiceClass *getService() {
    return dynamic_cast<ServiceClass *>(getServiceImpl(typeid(ServiceClass)));
  }

protected:
  using Service = services::Service;
  /// Called by `getServiceImpl` exclusively. It wraps the pointer returned by
  /// this in a unique_ptr and caches it. Separate this from the
  /// wrapping/caching since wrapping/caching is an implementation detail.
  virtual Service *createService(Service::Type service) = 0;
  /// Calls `createService` and caches the result. Subclasses can override if
  /// they want to use their own caching mechanism.
  virtual Service *getServiceImpl(Service::Type service);

private:
  /// Cache services via a unique_ptr so they get free'd automatically when
  /// Accelerator objects get deconstructed.
  std::map<const std::type_info *, std::unique_ptr<Service>> serviceCache;
};

namespace registry {

// Connect to an ESI accelerator given a backend name and connection specifier.
// Alternatively, instantiate the backend directly (if you're using C++).
std::unique_ptr<Accelerator> connect(std::string backend,
                                     std::string connection);

namespace internal {

/// Backends can register themselves to be connected via a connection string.
using BackendCreate = std::function<std::unique_ptr<Accelerator>(std::string)>;
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
