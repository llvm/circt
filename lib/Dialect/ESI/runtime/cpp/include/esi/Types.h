//===- Types.h - ESI type system -------------------------------*- C++ -*-===//
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
#ifndef ESI_TYPES_H
#define ESI_TYPES_H

#include <any>
#include <assert.h>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace esi {

/// Root class of the ESI type system.
class Type {
public:
  using ID = std::string;
  Type(const ID &id) : id(id) {}
  virtual ~Type() = default;

  ID getID() { return id; }

protected:
  ID id;
};

/// Bundles represent a collection of channels. Services exclusively expose
/// bundles (sometimes of just one channel). As such, they are the type of
/// accessible ports on an accelerator, from a host API perspective.
/// TODO: Add a good description of direction?
class BundleType : public Type {
public:
  enum Direction { To, From };

  using ChannelVector =
      std::vector<std::tuple<std::string, Direction, const Type &>>;

  BundleType(const ID &id, const ChannelVector &channels)
      : Type(id), channels(channels) {}

  const ChannelVector &getChannels() const { return channels; }

protected:
  ChannelVector channels;
};

} // namespace esi

#endif // ESI_TYPES_H
