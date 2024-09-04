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
#include "esi/Context.h"
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
  // Get a description of the service port.
  virtual std::optional<std::string> toString() const { return std::nullopt; }
};

/// Parent class of all APIs modeled as 'services'. May or may not map to a
/// hardware side 'service'.
class Service {
public:
  using Type = const std::type_info &;
  virtual ~Service() = default;

  virtual std::string getServiceSymbol() const = 0;

  /// Create a "child" service of this service. Does not have to be the same
  /// service type, but typically is. Used when a service already exists in the
  /// active services table, but a new one wants to replace it. Useful for cases
  /// where the child service needs to use the parent service. Defaults to
  /// calling the `getService` method on `AcceleratorConnection` to get the
  /// global service, implying that the child service does not need to use the
  /// service it is replacing.
  virtual Service *getChildService(AcceleratorConnection *conn,
                                   Service::Type service, AppIDPath id = {},
                                   std::string implName = {},
                                   ServiceImplDetails details = {},
                                   HWClientDetails clients = {});

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
  static constexpr std::string_view StdName = "esi.service.std.mmio";

  /// Describe a region (slice) of MMIO space.
  struct RegionDescriptor {
    uint32_t base;
    uint32_t size;
  };

  MMIO(Context &ctxt, AppIDPath idPath, std::string implName,
       const ServiceImplDetails &details, const HWClientDetails &clients);
  MMIO() = default;
  virtual ~MMIO() = default;

  /// Read a 64-bit value from the global MMIO space.
  virtual uint64_t read(uint32_t addr) const = 0;
  /// Write a 64-bit value to the global MMIO space.
  virtual void write(uint32_t addr, uint64_t data) = 0;
  /// Get the regions of MMIO space that this service manages. Otherwise known
  /// as the base address table.
  const std::map<AppIDPath, RegionDescriptor> &getRegions() const {
    return regions;
  }

  /// If the service is a MMIO service, return a region of the MMIO space which
  /// peers into ours.
  virtual Service *getChildService(AcceleratorConnection *conn,
                                   Service::Type service, AppIDPath id = {},
                                   std::string implName = {},
                                   ServiceImplDetails details = {},
                                   HWClientDetails clients = {}) override;

  virtual std::string getServiceSymbol() const override;

  /// Get a MMIO region port for a particular region descriptor.
  virtual ServicePort *getPort(AppIDPath id, const BundleType *type,
                               const std::map<std::string, ChannelPort &> &,
                               AcceleratorConnection &) const override;

private:
  /// MMIO base address table.
  std::map<AppIDPath, RegionDescriptor> regions;

public:
  /// A "slice" of some parent MMIO space.
  class MMIORegion : public ServicePort {
    friend class MMIO;
    MMIORegion(AppID id, MMIO *parent, RegionDescriptor desc);

  public:
    /// Get the offset (and size) of the region in the parent (usually global)
    /// MMIO address space.
    virtual RegionDescriptor getDescriptor() const { return desc; };
    /// Read a 64-bit value from this region, not the global address space.
    virtual uint64_t read(uint32_t addr) const;
    /// Write a 64-bit value to this region, not the global address space.
    virtual void write(uint32_t addr, uint64_t data);

    virtual std::optional<std::string> toString() const override {
      return "MMIO region " + toHex(desc.base) + " - " +
             toHex(desc.base + desc.size);
    }

  private:
    MMIO *parent;
    RegionDescriptor desc;
  };
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

    virtual std::optional<std::string> toString() const override {
      const esi::Type *argType =
          dynamic_cast<const ChannelType *>(arg.getType())->getInner();
      const esi::Type *resultType =
          dynamic_cast<const ChannelType *>(result.getType())->getInner();
      return "function " + resultType->getID() + "(" + argType->getID() + ")";
    }

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
