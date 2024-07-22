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
  virtual uint64_t read(uint32_t addr) const = 0;
  virtual void write(uint32_t addr, uint64_t data) = 0;
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

class HostMem : public Service {
public:
  virtual ~HostMem() = default;
  virtual std::string getServiceSymbol() const override;

  /// RAII memory region for host memory. Automatically frees the memory when
  /// deconstructed.
  struct HostMemRegion {
    virtual ~HostMemRegion() = default;
    virtual void *getPtr() const = 0;
    operator void *() const { return getPtr(); }
    virtual std::size_t getSize() const = 0;
  };

  /// Options for allocating host memory.
  struct Options {
    bool writeable = false;
    bool useLargePages = false;
  };

  /// Allocate a region of host memory in accelerator accessible address space.
  virtual std::unique_ptr<HostMemRegion> allocate(std::size_t size,
                                                  Options opts) const = 0;

  /// Try to make a region of host memory accessible to the accelerator. Returns
  /// 'false' on failure. It is optional for an accelerator backend to implement
  /// this, so client code needs to have a fallback for when this returns
  /// 'false'. On success, it is the client's responsibility to ensure that the
  /// memory eventually gets unmapped.
  virtual bool mapMemory(void *ptr, std::size_t size, Options opts) const {
    return false;
  }
  /// Unmap memory which was previously mapped with 'mapMemory'. Undefined
  /// behavior when called with a pointer which was not previously mapped.
  virtual void unmapMemory(void *ptr) const {}
};

/// Service for calling functions.
class FuncService : public Service {
public:
  FuncService(AcceleratorConnection *acc, AppIDPath id,
              const std::string &implName, ServiceImplDetails details,
              HWClientDetails clients);

  virtual std::string getServiceSymbol() const override;
  virtual ServicePort *getPort(AppIDPath id, const BundleType *type,
                               const std::map<std::string, ChannelPort &> &,
                               AcceleratorConnection &) const override;

  /// A function call which gets attached to a service port.
  class Function : public ServicePort {
    friend class FuncService;
    Function(AppID id, const std::map<std::string, ChannelPort &> &channels);

  public:
    static Function *get(AppID id, WriteChannelPort &arg,
                         ReadChannelPort &result);

    void connect();
    std::future<MessageData> call(const MessageData &arg);

  private:
    std::mutex callMutex;
    WriteChannelPort &arg;
    ReadChannelPort &result;
  };

private:
  std::string symbol;
};

/// Service for servicing function calls from the accelerator.
class CallService : public Service {
public:
  CallService(AcceleratorConnection *acc, AppIDPath id, std::string implName,
              ServiceImplDetails details, HWClientDetails clients);

  virtual std::string getServiceSymbol() const override;
  virtual ServicePort *getPort(AppIDPath id, const BundleType *type,
                               const std::map<std::string, ChannelPort &> &,
                               AcceleratorConnection &) const override;

  /// A function call which gets attached to a service port.
  class Callback : public ServicePort {
    friend class CallService;
    Callback(AcceleratorConnection &acc, AppID id,
             const std::map<std::string, ChannelPort &> &channels);

  public:
    /// Connect a callback to code which will be executed when the accelerator
    /// invokes the callback. The 'quick' flag indicates that the callback is
    /// sufficiently fast that it could be called in the same thread as the
    /// port callback.
    void connect(std::function<MessageData(const MessageData &)> callback,
                 bool quick = false);

  private:
    ReadChannelPort &arg;
    WriteChannelPort &result;
    AcceleratorConnection &acc;
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
