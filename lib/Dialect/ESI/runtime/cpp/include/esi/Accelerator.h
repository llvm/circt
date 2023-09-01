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
constexpr uint32_t ExpectedVersionNumber = 0;

class SysInfo;

namespace services {
class Service {
public:
  virtual ~Service() = default;
};
} // namespace services

/// An ESI accelerator system.
class Accelerator {
public:
  virtual ~Accelerator() = default;

  virtual const SysInfo &sysInfo() = 0;

  template <typename ServiceClass>
  ServiceClass *getService() {
    return dynamic_cast<ServiceClass *>(getServiceImpl(typeid(ServiceClass)));
  }

protected:
  virtual services::Service *getServiceImpl(const std::type_info &service) = 0;
  std::map<const std::type_info *, services::Service *> serviceCache;
};

/// Information about the Accelerator system.
class SysInfo {
public:
  virtual ~SysInfo() = default;

  /// Get the ESI version number to check version compatibility.
  virtual uint32_t esiVersion() const = 0;

  /// Return the JSON-formatted system manifest.
  virtual std::string rawJsonManifest() const = 0;
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
