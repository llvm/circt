//===- Design.h - Dynamic accelerator API -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The dynamic API into an accelerator allows access to the accelerator's design
// and communication channels through various stl containers (e.g. std::vector,
// std::map, etc.). This allows runtime reflection against the accelerator and
// can be pybind'd to create a Python API.
//
// The static API, in contrast, is a compile-time API that allows access to the
// design and communication channels symbolically. It will be generated once
// (not here) then compiled into the host software.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_DESIGN_H
#define ESI_DESIGN_H

#include "esi/Accelerator.h"
#include "esi/Manifest.h"

#include <any>
#include <cstdint>
#include <string>

namespace esi {
// Forward declarations.
class Instance;

class ChannelPort {};

class BundlePort {
  friend class services::Service;

public:
  BundlePort(AppID id, std::map<std::string, ChannelPort *> channels);
  bool isConnected() const;

private:
  AppID _id;
  std::map<std::string, ChannelPort *> _channels;
};

class Design {
public:
  Design(std::optional<ModuleInfo> info,
         std::vector<std::unique_ptr<Instance>> children,
         std::vector<services::Service *> services,
         std::vector<BundlePort> ports)
      : _info(info), _children(std::move(children)), _services(services),
        _ports(ports) {}

  std::optional<ModuleInfo> info() const { return _info; }
  const std::vector<std::unique_ptr<Instance>> &children() const {
    return _children;
  }

protected:
  const std::optional<ModuleInfo> _info;
  const std::vector<std::unique_ptr<Instance>> _children;
  const std::vector<services::Service *> _services;
  const std::vector<BundlePort> _ports;
};

class Instance : public Design {
public:
  Instance() = delete;
  Instance(const Instance &) = delete;
  ~Instance() = default;
  Instance(AppID id, std::optional<ModuleInfo> info,
           std::vector<std::unique_ptr<Instance>> children,
           std::vector<services::Service *> services,
           std::vector<BundlePort> ports)
      : Design(info, std::move(children), services, ports), _id(id) {}

  const AppID id() const { return _id; }

protected:
  const AppID _id;
};

} // namespace esi

#endif // ESI_DESIGN_H
