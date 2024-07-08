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
  constexpr static char kernel[] = "esi_kernel";

  Impl(std::string xclbin, std::string device_id) {
    if (device_id.empty())
      device = ::xrt::device(0);
    else
      device = ::xrt::device(device_id);

    auto uuid = device.load_xclbin(xclbin);
    ip = ::xrt::ip(device, uuid, kernel);
  }

  std::map<std::string, ChannelPort &> requestChannelsFor(AppIDPath,
                                                          const BundleType *) {
    throw std::runtime_error("XRT does not support channel communication yet");
  }

  ::xrt::device device;
  ::xrt::ip ip;
};

/// Construct and connect to a cosim server.
XrtAccelerator::XrtAccelerator(Context &ctxt, std::string xclbin,
                               std::string device_id)
    : AcceleratorConnection(ctxt) {
  impl = make_unique<Impl>(xclbin, device_id);
}

namespace {
class XrtMMIO : public MMIO {
public:
  XrtMMIO(::xrt::ip &ip) : ip(ip) {}

  uint64_t read(uint32_t addr) const override {
    auto lo = static_cast<uint64_t>(ip.read_register(addr));
    auto hi = static_cast<uint64_t>(ip.read_register(addr + 0x4));
    return (hi << 32) | lo;
  }
  void write(uint32_t addr, uint64_t data) override {
    ip.write_register(addr, data);
    ip.write_register(addr + 0x4, data >> 32);
  }

private:
  ::xrt::ip &ip;
};
} // namespace

std::map<std::string, ChannelPort &>
XrtAccelerator::requestChannelsFor(AppIDPath idPath,
                                   const BundleType *bundleType) {
  return impl->requestChannelsFor(idPath, bundleType);
}

Service *XrtAccelerator::createService(Service::Type svcType, AppIDPath id,
                                       std::string implName,
                                       const ServiceImplDetails &details,
                                       const HWClientDetails &clients) {
  if (svcType == typeid(MMIO))
    return new XrtMMIO(impl->ip);
  else if (svcType == typeid(SysInfo))
    return new MMIOSysInfo(getService<MMIO>());
  return nullptr;
}

REGISTER_ACCELERATOR("xrt", backends::xrt::XrtAccelerator);
