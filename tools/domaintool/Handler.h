//===- Handler.h - Utility for processing domain information ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines utilities related to "handlers" of domain information.
// This defines the base class of a handler which it is expected will be
// implemented by the different handlers that are needed.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_DOMAINTOOL_HANDLER_H
#define CIRCT_TOOLS_DOMAINTOOL_HANDLER_H

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "llvm/ADT/MapVector.h"

namespace circt {

using namespace llvm;
using namespace om;

/// The type that will be passed to each handler.  This is a mapping of domain
/// objects _of one type_ to the associations for that domain.  This is ordered
/// based on the discovered order of output domains on the module of interest.
using ObjectMap =
    llvm::MapVector<om::evaluator::ObjectValue *,
                    SmallVector<om::evaluator::EvaluatorValuePtr>>;

/// Base class of a domain handler.  A handler is used to process domains of one
/// or more kinds.
struct Handler {
  /// Return true if this is a domain that should be processed by this handler.
  virtual bool shouldHandle(Type tpe) = 0;

  /// Given the ObjectMap for a domain that this handler is sensitive to, do
  /// something with it.  The intent is that this "accumulates" the domain
  /// information into internal handler state.
  virtual llvm::LogicalResult handle(const ObjectMap &objectMap) = 0;

  /// Do something with the accumulated domain information and send data to an
  /// output stream.
  ///
  /// TODO: Figure out how to make this make sense for handlers which want to
  /// write files, stream to stdout, and to make this work for multiple
  /// handlers.
  virtual llvm::LogicalResult emit(raw_ostream &os) = 0;

  /// Reset this handler to its initial state.  This is called after `emit`.
  virtual void clear() = 0;

  virtual ~Handler() = default;
};

/// A registry of different domain handlers.  This is used to enable new
/// handlers to be added to `domaintool` without having to tell `domaintool`
/// about them.
struct HandlerRegistry {

  /// Return a singleton handler registroy.
  static HandlerRegistry &get() {
    static HandlerRegistry instance;
    return instance;
  }

  /// Register a handler.
  void registerHandler(std::unique_ptr<Handler> handler) {
    handlers.push_back(std::move(handler));
  }

  /// Return all registered handlers.
  ArrayRef<std::unique_ptr<Handler>> getHandlers() const { return handlers; }

private:
  /// Internal storage of handlers.
  SmallVector<std::unique_ptr<Handler>> handlers;
};

} // namespace circt

#endif // CIRCT_TOOLS_DOMAINTOOL_HANDLER_H
