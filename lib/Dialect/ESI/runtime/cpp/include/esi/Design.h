//===- Design.h - Dynamic accelerator API -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
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
namespace manifest {
struct Service;
}
class Instance;

class PortChannel {};

class Design {
public:
  Design(std::optional<ModuleInfo> info,
         std::vector<std::unique_ptr<Instance>> children,
         std::vector<std::unique_ptr<services::Service>> services)
      : _info(info), _children(std::move(children)),
        _services(std::move(services)) {}

  std::optional<ModuleInfo> info() const { return _info; }
  const std::vector<std::unique_ptr<Instance>> &children() const {
    return _children;
  }

protected:
  const std::optional<ModuleInfo> _info;
  const std::vector<std::unique_ptr<Instance>> _children;
  const std::vector<std::unique_ptr<services::Service>> _services;
  // const std::map<AppID, PortChannel &> _ports;
};

class Instance : public Design {
public:
  Instance() = delete;
  Instance(const Instance &) = delete;
  ~Instance() = default;
  Instance(AppID id, std::optional<ModuleInfo> info,
           std::vector<std::unique_ptr<Instance>> children,
           std::vector<std::unique_ptr<services::Service>> services)
      : Design(info, std::move(children), std::move(services)), _id(id) {}

  const AppID id() const { return _id; }

protected:
  const AppID _id;
};

} // namespace esi

std::ostream &operator<<(std::ostream &, const esi::AppID &);

#endif // ESI_DESIGN_H
