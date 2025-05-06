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
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace esi {
class Type;

//===----------------------------------------------------------------------===//
// Common accelerator description types.
//===----------------------------------------------------------------------===//

struct AppID {
  std::string name;
  std::optional<uint32_t> idx;

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

struct Constant {
  std::any value;
  std::optional<const Type *> type;
};

struct ModuleInfo {
  std::optional<std::string> name;
  std::optional<std::string> summary;
  std::optional<std::string> version;
  std::optional<std::string> repo;
  std::optional<std::string> commitHash;
  std::map<std::string, Constant> constants;
  std::map<std::string, std::any> extra;
};

/// A description of a service port. Used pretty exclusively in setting up the
/// design.
struct ServicePortDesc {
  std::string name;
  std::string portName;
};

/// Details about how to connect to a particular channel.
struct ChannelAssignment {
  /// The name of the type of connection. Typically, the name of the DMA engine
  /// or "cosim" if a cosimulation channel is being used.
  std::string type;
  /// Implementation-specific options.
  std::map<std::string, std::any> implOptions;
};
using ChannelAssignments = std::map<std::string, ChannelAssignment>;

/// A description of a hardware client. Used pretty exclusively in setting up
/// the design.
struct HWClientDetail {
  AppIDPath relPath;
  ServicePortDesc port;
  ChannelAssignments channelAssignments;
  std::map<std::string, std::any> implOptions;
};
using HWClientDetails = std::vector<HWClientDetail>;
using ServiceImplDetails = std::map<std::string, std::any>;

/// A logical chunk of data representing serialized data. Currently, just a
/// wrapper for a vector of bytes, which is not efficient in terms of memory
/// copying. This will change in the future as will the API.
class MessageData {
public:
  /// Adopts the data vector buffer.
  MessageData() = default;
  MessageData(std::vector<uint8_t> &data) : data(std::move(data)) {}
  MessageData(const uint8_t *data, size_t size) : data(data, data + size) {}
  ~MessageData() = default;

  const uint8_t *getBytes() const { return data.data(); }
  /// Get the size of the data in bytes.
  size_t getSize() const { return data.size(); }

  /// Cast to a type. Throws if the size of the data does not match the size of
  /// the message. The lifetime of the resulting pointer is tied to the lifetime
  /// of this object.
  template <typename T>
  const T *as() const {
    if (data.size() != sizeof(T))
      throw std::runtime_error("Data size does not match type size. Size is " +
                               std::to_string(data.size()) + ", expected " +
                               std::to_string(sizeof(T)) + ".");
    return reinterpret_cast<const T *>(data.data());
  }

  /// Cast from a type to its raw bytes.
  template <typename T>
  static MessageData from(T &t) {
    return MessageData(reinterpret_cast<const uint8_t *>(&t), sizeof(T));
  }

  /// Convert the data to a hex string.
  std::string toHex() const;

private:
  std::vector<uint8_t> data;
};

} // namespace esi

std::ostream &operator<<(std::ostream &, const esi::ModuleInfo &);
std::ostream &operator<<(std::ostream &, const esi::AppID &);

//===----------------------------------------------------------------------===//
// Functions which should be in the standard library.
//===----------------------------------------------------------------------===//

namespace esi {
std::string toHex(void *val);
std::string toHex(uint64_t val);
} // namespace esi

#endif // ESI_COMMON_H
