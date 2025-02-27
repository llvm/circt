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
// Note for hardware designers: the "design hierarchy" from the host API
// perspective is not the same as the hardware module instance hierarchy.
// Rather, it is only the relevant parts as defined by the AppID hierarchy --
// levels in the hardware module instance hierarchy get skipped.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_DESIGN_H
#define ESI_DESIGN_H

#include "esi/Manifest.h"
#include "esi/Ports.h"
#include "esi/Services.h"

#include <string>

namespace esi {
// Forward declarations.
class Instance;
namespace services {
class Service;
} // namespace services

/// Represents either the top level or an instance of a hardware module.
class HWModule {
protected:
  HWModule(std::optional<ModuleInfo> info,
           std::vector<std::unique_ptr<Instance>> children,
           std::vector<services::Service *> services,
           std::vector<std::unique_ptr<BundlePort>> &ports);

public:
  virtual ~HWModule() = default;

  /// Access the module's metadata, if any.
  std::optional<ModuleInfo> getInfo() const { return info; }
  /// Get a vector of the module's children in a deterministic order.
  std::vector<const Instance *> getChildrenOrdered() const {
    std::vector<const Instance *> ret;
    for (const auto &c : children)
      ret.push_back(c.get());
    return ret;
  }
  /// Access the module's children by ID.
  const std::map<AppID, Instance *> &getChildren() const { return childIndex; }
  /// Get the module's ports in a deterministic order.
  std::vector<std::reference_wrapper<BundlePort>> getPortsOrdered() const {
    std::vector<std::reference_wrapper<BundlePort>> ret;
    for (const auto &p : ports)
      ret.push_back(*p);
    return ret;
  }
  /// Access the module's ports by ID.
  const std::map<AppID, BundlePort &> &getPorts() const { return portIndex; }
  /// Access the services provided by this module.
  const std::vector<services::Service *> &getServices() const {
    return services;
  }

  /// Master poll method. Calls the `poll` method on all locally owned ports and
  /// the master `poll` method on all of the children. Returns true if any of
  /// the `poll` calls returns true.
  bool poll();

  /// Attempt to resolve a path to a module instance. If a child is not found,
  /// return null and set lastLookup to the path which wasn't found.
  const HWModule *resolveInst(const AppIDPath &path,
                              AppIDPath &lastLookup) const;

  /// Attempt to resolve a path to a port. If a child or port is not found,
  /// return null and set lastLookup to the path which wasn't found.
  BundlePort *resolvePort(const AppIDPath &path, AppIDPath &lastLookup) const;

protected:
  const std::optional<ModuleInfo> info;
  const std::vector<std::unique_ptr<Instance>> children;
  const std::map<AppID, Instance *> childIndex;
  const std::vector<services::Service *> services;
  const std::vector<std::unique_ptr<BundlePort>> ports;
  const std::map<AppID, BundlePort &> portIndex;
};

/// Subclass of `HWModule` which represents a submodule instance. Adds an AppID,
/// which the top level doesn't have or need.
class Instance : public HWModule {
public:
  Instance() = delete;
  Instance(const Instance &) = delete;
  ~Instance() = default;
  Instance(AppID id, std::optional<ModuleInfo> info,
           std::vector<std::unique_ptr<Instance>> children,
           std::vector<services::Service *> services,
           std::vector<std::unique_ptr<BundlePort>> &ports)
      : HWModule(info, std::move(children), services, ports), id(id) {}

  /// Get the instance's ID, which it will always have.
  const AppID getID() const { return id; }

protected:
  const AppID id;
};

} // namespace esi

#endif // ESI_DESIGN_H
