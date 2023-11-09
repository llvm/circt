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

class Type {
public:
  using ID = std::string;
  Type(ID id) : _id(id) {}

  ID getID() { return _id; }

private:
  ID _id;
};

class BundleType : public Type {
public:
  enum Direction { To, From };
  BundleType(
      ID id,
      std::vector<std::tuple<std::string, Direction, const Type &>> channels)
      : Type(id), _channels(channels) {}

  const std::vector<std::tuple<std::string, Direction, const Type &>> &
  getChannels() {
    return _channels;
  }

private:
  std::vector<std::tuple<std::string, Direction, const Type &>> _channels;
};

} // namespace esi

#endif // ESI_TYPES_H
