//===- Common.h - Commonly used classes w/o dependencies --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_COMMON_H
#define ESI_COMMON_H

#include <any>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace esi {

//===----------------------------------------------------------------------===//
// Common accelerator description types.
//===----------------------------------------------------------------------===//

struct AppID {
  std::string name;
  std::optional<uint32_t> idx;

  AppID(const AppID &) = default;
  AppID(const std::string &name, std::optional<uint32_t> idx = std::nullopt)
      : name(name), idx(idx) {}

  bool operator==(const AppID &other) const {
    return name == other.name && idx == other.idx;
  }
  bool operator!=(const AppID &other) const { return !(*this == other); }
};
bool operator<(const AppID &a, const AppID &b);

class AppIDPath : public std::vector<AppID> {
public:
  using std::vector<AppID>::vector;

  AppIDPath operator+(const AppIDPath &b);
  std::string toStr() const;
};
bool operator<(const AppIDPath &a, const AppIDPath &b);
std::ostream &operator<<(std::ostream &, const esi::AppIDPath &);

struct ModuleInfo {
  const std::optional<std::string> name;
  const std::optional<std::string> summary;
  const std::optional<std::string> version;
  const std::optional<std::string> repo;
  const std::optional<std::string> commitHash;
  const std::map<std::string, std::any> extra;
};

/// A description of a service port. Used pretty exclusively in setting up the
/// design.
struct ServicePortDesc {
  std::string name;
  std::string portName;
};

/// A description of a hardware client. Used pretty exclusively in setting up
/// the design.
struct HWClientDetail {
  AppIDPath relPath;
  ServicePortDesc port;
  std::map<std::string, std::any> implOptions;
};
using HWClientDetails = std::vector<HWClientDetail>;
using ServiceImplDetails = std::map<std::string, std::any>;

} // namespace esi

std::ostream &operator<<(std::ostream &, const esi::ModuleInfo &);
std::ostream &operator<<(std::ostream &, const esi::AppID &);

#endif // ESI_COMMON_H
