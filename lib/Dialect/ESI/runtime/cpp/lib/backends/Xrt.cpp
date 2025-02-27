//===- Xrt.cpp - ESI XRT device backend -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT
// (lib/dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp).
//
//===----------------------------------------------------------------------===//

#include "esi/backends/Xrt.h"
#include "esi/Services.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_xclbin.h"

#include <fstream>
#include <iostream>
#include <set>

using namespace esi;
using namespace esi::services;
using namespace esi::backends::xrt;

/// Parse the connection std::string and instantiate the accelerator. Connection
/// std::string format:
///  <xclbin>[:<device_id>]
/// wherein <device_id> is in BDF format.
std::unique_ptr<AcceleratorConnection>
XrtAccelerator::connect(Context &ctxt, std::string connectionString) {
  std::string xclbin;
  std::string device_id;

  size_t colon = connectionString.find(':');
  if (colon == std::string::npos) {
    xclbin = connectionString;
  } else {
    xclbin = connectionString.substr(0, colon);
    device_id = connectionString.substr(colon + 1);
  }

  return make_unique<XrtAccelerator>(ctxt, xclbin, device_id);
}

struct esi::backends::xrt::XrtAccelerator::Impl {
  constexpr static char kernel_name[] = "esi_kernel";

  Impl(std::string xclbin, std::string device_id) {
    if (device_id.empty())
      device = ::xrt::device(0);
    else
      device = ::xrt::device(device_id);

    // Find memory group for the host.
    ::xrt::xclbin xcl(xclbin);
    std::optional<::xrt::xclbin::mem> host_mem;
    for (auto mem : xcl.get_mems()) {
      // The host memory is tagged with "HOST[0]". Memory type is wrong --
      // reports as DRAM rather than host memory so we can't filter on that.
      if (mem.get_tag().starts_with("HOST")) {
        if (host_mem.has_value())
          throw std::runtime_error("Multiple host memories found in xclbin");
        else
          host_mem = mem;
      }
    }
    if (!host_mem)
      throw std::runtime_error("No host memory found in xclbin");
    memoryGroup = host_mem->get_index();

    // Load the xclbin and instantiate the IP.
    auto uuid = device.load_xclbin(xcl);
    ip = ::xrt::ip(device, uuid, kernel_name);
  }

  ::xrt::device device;
  ::xrt::ip ip;
  int32_t memoryGroup;
};

/// Construct and connect to a cosim server.
XrtAccelerator::XrtAccelerator(Context &ctxt, std::string xclbin,
                               std::string device_id)
    : AcceleratorConnection(ctxt) {
  impl = make_unique<Impl>(xclbin, device_id);
}
XrtAccelerator::~XrtAccelerator() { disconnect(); }

namespace {
/// Implementation of MMIO for XRT. Uses an indirect register to access a
/// virtual MMIO space since Vivado/Vitis don't support large enough MMIO
/// spaces.
class XrtMMIO : public MMIO {

  // Physical register to write the virtual address.
  constexpr static uint32_t IndirectLocation = 0x18;
  // Physical register to read/write the virtual MMIO address stored at
  // `IndirectLocation`.
  constexpr static uint32_t IndirectMMIOReg = 0x20;

public:
  XrtMMIO(XrtAccelerator &conn, ::xrt::ip &ip, const HWClientDetails &clients)
      : MMIO(conn, clients), ip(ip) {}

  uint64_t read(uint32_t addr) const override {
    std::lock_guard<std::mutex> lock(m);

    // Write the address to the indirect location register.
    xrt_write(IndirectLocation, addr);
    // Read from the indirect register.
    uint64_t ret = xrt_read(IndirectMMIOReg);

    getConnection().getLogger().debug(
        [addr, ret](std::string &subsystem, std::string &msg,
                    std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "xrt_mmio";
          msg = "MMIO[0x" + toHex(addr) + "] = 0x" + toHex(ret);
        });
    return ret;
  }
  void write(uint32_t addr, uint64_t data) override {
    std::lock_guard<std::mutex> lock(m);

    // Write the address to the indirect location register.
    xrt_write(IndirectLocation, addr);
    // Write to the indirect register.
    xrt_write(IndirectMMIOReg, data);

    conn.getLogger().debug(
        [addr,
         data](std::string &subsystem, std::string &msg,
               std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "xrt_mmio";
          msg = "MMIO[0x" + toHex(addr) + "] <- 0x" + toHex(data);
        });
  }

  /// Read a physical register.
  uint64_t xrt_read(uint32_t addr) const {
    auto lo = static_cast<uint64_t>(ip.read_register(addr));
    auto hi = static_cast<uint64_t>(ip.read_register(addr + 0x4));
    return (hi << 32) | lo;
  }
  /// Write a physical register.
  void xrt_write(uint32_t addr, uint64_t data) const {
    ip.write_register(addr, data);
    ip.write_register(addr + 0x4, data >> 32);
  }

private:
  ::xrt::ip &ip;
  mutable std::mutex m;
};
} // namespace

namespace {
/// Host memory service specialized to XRT.
class XrtHostMem : public HostMem {
public:
  XrtHostMem(XrtAccelerator &conn, ::xrt::device &device, int32_t memoryGroup)
      : HostMem(conn), device(device), memoryGroup(memoryGroup){}

  struct XrtHostMemRegion : public HostMemRegion {
    XrtHostMemRegion(::xrt::device &device, std::size_t size,
                     HostMem::Options opts, int32_t memoryGroup) {
      bo = ::xrt::bo(device, size, ::xrt::bo::flags::host_only, memoryGroup);
      // Map the buffer into application memory space so that the application
      // can use it just like any memory -- no need to use bo::write.
      ptr = bo.map();
    }
    virtual void *getPtr() const override { return ptr; }
    /// On XRT platforms, the pointer which the device sees is different from
    /// the pointer the user application sees.
    virtual void *getDevicePtr() const override { return (void *)bo.address(); }
    virtual std::size_t getSize() const override { return bo.size(); }
    /// It is required to use 'sync' to flush the caches before executing any
    /// DMA.
    virtual void flush() override { bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); }

  private:
    ::xrt::bo bo;
    void *ptr;
  };

  std::unique_ptr<HostMemRegion>
  allocate(std::size_t size, HostMem::Options opts) const override {
    return std::unique_ptr<HostMemRegion>(
        new XrtHostMemRegion(device, size, opts, memoryGroup));
  }

private:
  ::xrt::device &device;
  int32_t memoryGroup;
};
} // namespace

Service *XrtAccelerator::createService(Service::Type svcType, AppIDPath id,
                                       std::string implName,
                                       const ServiceImplDetails &details,
                                       const HWClientDetails &clients) {
  if (svcType == typeid(MMIO))
    return new XrtMMIO(*this, impl->ip, clients);
  else if (svcType == typeid(HostMem))
    return new XrtHostMem(*this, impl->device, impl->memoryGroup);
  else if (svcType == typeid(SysInfo))
    return new MMIOSysInfo(getService<MMIO>());
  return nullptr;
}

REGISTER_ACCELERATOR("xrt", backends::xrt::XrtAccelerator);
