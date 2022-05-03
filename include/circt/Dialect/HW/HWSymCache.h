//===- HWSymCache.h - Declare Symbol Cache ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a Symbol Cache.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_SYMCACHE_H
#define CIRCT_DIALECT_HW_SYMCACHE_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Casting.h"

namespace circt {
namespace hw {

/// This stores lookup tables to make manipulating and working with the IR more
/// efficient.  There are two phases to this object: the "building" phase in
/// which it is "write only" and then the "using" phase which is read-only (and
/// thus can be used by multiple threads).  The
/// "freeze" method transitions between the two states.
class SymbolCache {
public:
  class Item {
  public:
    Item(mlir::Operation *op) : op(op), port(~0ULL) {}
    Item(mlir::Operation *op, size_t port) : op(op), port(port) {}
    bool hasPort() const { return port != ~0ULL; }
    mlir::Operation *getOp() const { return op; }
    size_t getPort() const { return port; }

  private:
    mlir::Operation *op;
    size_t port;
  };

  /// In the building phase, add symbols.
  void addDefinition(mlir::StringAttr symbol, mlir::Operation *op) {
    assert(!isFrozen && "cannot mutate a frozen cache");
    symbolCache.try_emplace(symbol, op, ~0ULL);
  }

  // Add inner names, which might be ports
  void addDefinition(mlir::StringAttr modSymbol, mlir::StringAttr name,
                     mlir::Operation *op, size_t port = ~0ULL) {
    assert(!isFrozen && "cannot mutate a frozen cache");
    auto key = InnerRefAttr::get(modSymbol, name);
    symbolCache.try_emplace(key, op, port);
  }

  /// Populate the symbol cache with all symbol-defining operations within the
  /// 'top' operation.
  void addDefinitions(mlir::Operation *top) {
    for (auto &region : top->getRegions())
      for (auto &block : region.getBlocks())
        for (mlir::Operation &op : block)
          if (auto symOp = llvm::dyn_cast<mlir::SymbolOpInterface>(op))
            if (auto name = symOp.getNameAttr())
              addDefinition(name, symOp);
  }

  // Add inner names, which might be ports
  void addDefinition(InnerRefAttr name, mlir::Operation *op,
                     size_t port = ~0ULL) {
    assert(!isFrozen && "cannot mutate a frozen cache");
    symbolCache.try_emplace(name, op, port);
  }

  /// Mark the cache as frozen, which allows it to be shared across threads.
  void freeze() { isFrozen = true; }

  mlir::Operation *getDefinition(mlir::StringAttr symbol) const {
    return lookup(symbol);
  }

  mlir::Operation *getDefinition(mlir::FlatSymbolRefAttr symbol) const {
    return lookup(symbol.getAttr());
  }

  Item getDefinition(mlir::StringAttr modSymbol, mlir::StringAttr name) const {
    return lookup(InnerRefAttr::get(modSymbol, name));
  }

  Item getDefinition(InnerRefAttr name) const { return lookup(name); }

protected:
  bool isFrozen = false;

private:
  Item lookup(InnerRefAttr attr) const {
    assert(isFrozen && "cannot read from this cache until it is frozen");
    auto it = symbolCache.find(attr);
    return it == symbolCache.end() ? Item{nullptr, ~0ULL} : it->second;
  }

  mlir::Operation *lookup(mlir::StringAttr attr) const {
    assert(isFrozen && "cannot read from this cache until it is frozen");
    auto it = symbolCache.find(attr);
    if (it == symbolCache.end())
      return nullptr;
    assert(!it->second.hasPort() && "Module names should never be ports");
    return it->second.getOp();
  }

  /// This stores a lookup table from symbol attribute to the operation
  /// (hw.module, hw.instance, etc) that defines it.
  /// TODO: It is super annoying that symbols are *defined* as StringAttr, but
  /// are then referenced as FlatSymbolRefAttr.  Why can't we have nice
  /// pointer uniqued things?? :-(
  llvm::DenseMap<mlir::Attribute, Item> symbolCache;
};

/// Like a SymbolCache, but allows for unfreezing to add new definitions.
/// Unlike SymbolCache, the MutableSymbolCache is not thread safe, and the
/// caller is expected to perform synchronization if used in a multithreaded
/// context.
class MutableSymbolCache : public SymbolCache {
public:
  /// Mark the cache as unfrozen, allowing for mutation. Caller should ensure
  /// that the cache is no longer being read from after unfreezing occurs.
  void unfreeze() { isFrozen = false; }
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_SYMCACHE_H
