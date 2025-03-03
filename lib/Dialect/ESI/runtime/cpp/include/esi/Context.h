//===- Context.h - Accelerator context --------------------------*- C++ -*-===//
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
#ifndef ESI_CONTEXT_H
#define ESI_CONTEXT_H

#include "esi/Logging.h"
#include "esi/Types.h"

#include <exception>
#include <memory>
#include <optional>

namespace esi {
class AcceleratorConnection;

/// AcceleratorConnections, Accelerators, and Manifests must all share a
/// context. It owns all the types, uniquifying them.
class Context {
public:
  Context() : logger(std::make_unique<ConsoleLogger>(Logger::Level::Warning)) {}
  Context(std::unique_ptr<Logger> logger) : logger(std::move(logger)) {}

  /// Create a context with a specific logger type.
  template <typename T, typename... Args>
  static Context withLogger(Args &&...args) {
    return Context(std::make_unique<T>(args...));
  }

  /// Resolve a type id to the type.
  std::optional<const Type *> getType(Type::ID id) const {
    if (auto f = types.find(id); f != types.end())
      return f->second.get();
    return std::nullopt;
  }

  /// Register a type with the context. Takes ownership of the pointer type.
  void registerType(Type *type);

  /// Connect to an accelerator backend.
  std::unique_ptr<AcceleratorConnection> connect(std::string backend,
                                                 std::string connection);

  /// Register a logger with the accelerator. Assumes ownership of the logger.
  void setLogger(std::unique_ptr<Logger> logger) {
    if (!logger)
      throw std::invalid_argument("logger must not be null");
    this->logger = std::move(logger);
  }
  inline Logger &getLogger() { return *logger; }

private:
  std::unique_ptr<Logger> logger;

private:
  using TypeCache = std::map<Type::ID, std::unique_ptr<Type>>;
  TypeCache types;
};

} // namespace esi

#endif // ESI_CONTEXT_H
