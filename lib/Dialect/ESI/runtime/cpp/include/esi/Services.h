//===- StdServices.h - ESI standard services C++ API ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The APIs in this backend are all optionally implemented. The lower level
// ones, however, are strongly recommended. 'Services' here refers to ESI
// services. These are standard APIs into the standard ESI services.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_RUNTIME_SERVICES_H
#define ESI_RUNTIME_SERVICES_H

#include "esi/Common.h"
#include "esi/Ports.h"

#include <cstdint>

namespace esi {
class AcceleratorConnection;
namespace services {

/// Add a custom interface to a service client at a particular point in the
/// design hierarchy.
class ServicePort : public BundlePort {
public:
  using BundlePort::BundlePort;
  virtual ~ServicePort() = default;
};

/// Parent class of all APIs modeled as 'services'. May or may not map to a
/// hardware side 'service'.
class Service {
public:
  using Type = const std::type_info &;
  virtual ~Service() = default;

  virtual std::string getServiceSymbol() const = 0;

  /// Get specialized port for this service to attach to the given appid path.
  /// Null returns mean nothing to attach.
  virtual ServicePort *getPort(AppIDPath id, const BundleType *type,
                               const std::map<std::string, ChannelPort &> &,
                               AcceleratorConnection &) const {
    return nullptr;
  }
};

/// A service for which there are no standard services registered. Requires
/// ports be added to the design hierarchy instead of high level interfaces like
/// the ones in StdServices.h.
class CustomService : public Service {
public:
  CustomService(AppIDPath idPath, const ServiceImplDetails &details,
                const HWClientDetails &clients);
  virtual ~CustomService() = default;

  virtual std::string getServiceSymbol() const override {
    return serviceSymbol;
  }

protected:
  std::string serviceSymbol;
  AppIDPath id;
};

/// Information about the Accelerator system.
class SysInfo : public Service {
public:
  virtual ~SysInfo() = default;

  virtual std::string getServiceSymbol() const override;

  /// Get the ESI version number to check version compatibility.
  virtual uint32_t getEsiVersion() const = 0;

  /// Return the JSON-formatted system manifest.
  virtual std::string getJsonManifest() const;

  /// Return the zlib compressed JSON system manifest.
  virtual std::vector<uint8_t> getCompressedManifest() const = 0;
};

class MMIO : public Service {
public:
  virtual ~MMIO() = default;
  virtual uint32_t read(uint32_t addr) const = 0;
  virtual void write(uint32_t addr, uint32_t data) = 0;
  virtual std::string getServiceSymbol() const override;
};

/// Implement the SysInfo API for a standard MMIO protocol.
class MMIOSysInfo final : public SysInfo {
public:
  MMIOSysInfo(const MMIO *);

  /// Get the ESI version number to check version compatibility.
  uint32_t getEsiVersion() const override;

  /// Return the zlib compressed JSON system manifest.
  virtual std::vector<uint8_t> getCompressedManifest() const override;

private:
  const MMIO *mmio;
};

/// Service for calling functions.
class FuncService : public Service {
public:
  FuncService(AcceleratorConnection *acc, AppIDPath id, std::string implName,
              ServiceImplDetails details, HWClientDetails clients);

  virtual std::string getServiceSymbol() const override;
  virtual ServicePort *getPort(AppIDPath id, const BundleType *type,
                               const std::map<std::string, ChannelPort &> &,
                               AcceleratorConnection &) const override;

  /// A function call which gets attached to a service port.
  class Function : public ServicePort {
    friend class FuncService;
    Function(AppID id, const std::map<std::string, ChannelPort &> &channels);

  public:
    void connect();
    MessageData call(const MessageData &arg);

  private:
    WriteChannelPort &arg;
    ReadChannelPort &result;
  };

private:
  std::string symbol;
};

/// Registry of services which can be instantiated directly by the Accelerator
/// class if the backend doesn't do anything special with a service.
class ServiceRegistry {
public:
  /// Create a service instance from the given details. Returns nullptr if
  /// 'svcType' isn't registered.
  static Service *createService(AcceleratorConnection *acc,
                                Service::Type svcType, AppIDPath id,
                                std::string implName,
                                ServiceImplDetails details,
                                HWClientDetails clients);

  /// Resolve a service type from a string. If the string isn't recognized,
  /// default to CustomService.
  static Service::Type lookupServiceType(const std::string &);
};

} // namespace services
} // namespace esi

#endif // ESI_RUNTIME_SERVICES_H
