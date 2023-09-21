//===- AppID.h - AppID related code -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent the placements of primitives on an FPGA.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_APPID_H
#define CIRCT_DIALECT_MSFT_APPID_H

#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace msft {

/// An index for resolving AppIDPaths to dynamic instances.
class AppIDIndex {
public:
  AppIDIndex(Operation *mlirTop);
  ~AppIDIndex();

  // If invalid, construction failed for some reason (which was emitted via an
  // error). Since we want to be able to call this class as an analysis, all of
  // the index construction occurs in the constructor, which doesn't allow for
  // a LogicalResult return. (This is where exceptions would be useful.)
  bool isValid() const { return valid; }

  // Return an array of AppIDAttrs which are contained in the module.
  ArrayAttr getChildAppIDsOf(hw::HWModuleLike) const;

  /// Return an array of InnerNameRefAttrs representing the relative path to
  /// 'appid' from 'fromMod'.
  FailureOr<ArrayAttr> getAppIDPathAttr(hw::HWModuleLike fromMod,
                                        AppIDAttr appid, Location loc) const;

  /// Get the dynamic instance for a particular appid path. If one doesn't
  /// already exist, if will be created.
  FailureOr<DynamicInstanceOp> getInstance(AppIDPathAttr path,
                                           Location loc) const;

private:
  //===--------------------------------------------------------------------===//
  // Query method helpers.
  //===--------------------------------------------------------------------===//

  /// Resolve the appid path once the root InstanceHierarchy has been resolved.
  FailureOr<DynamicInstanceOp> getSubInstance(hw::HWModuleLike mod,
                                              InstanceHierarchyOp dynInstRoot,
                                              ArrayRef<AppIDAttr> subpath,
                                              Location loc) const;

  /// Get the instance (relative to 'mod') for the AppID component. String a
  /// bunch of calls to this (one call per path component) together to do the
  /// full resolution. Also return the inner named (static) operation to which
  /// the dynamic instance points.
  FailureOr<std::pair<DynamicInstanceOp, hw::InnerSymbolOpInterface>>
  getSubInstance(hw::HWModuleLike mod, Operation *parent, AppIDAttr appid,
                 Location loc) const;

  /// Get or create (if it doesn't exist) the dynamic instance for inner name
  /// 'name' under 'parent'.
  DynamicInstanceOp getOrCreate(Operation *parent, hw::InnerRefAttr name,
                                Location) const;

  // Dynamic instance hierarchy caches to avoid a bunch of linear scans.
  mutable DenseMap<SymbolRefAttr, InstanceHierarchyOp> dynHierRoots;
  using NamedChildren = DenseMap<hw::InnerRefAttr, DynamicInstanceOp>;
  mutable DenseMap<Operation *, NamedChildren> dynInstChildLookup;

  //===--------------------------------------------------------------------===//
  // Index construction and storage.
  //===--------------------------------------------------------------------===//
  class ModuleAppIDs;

  /// Construct the index for a module.
  FailureOr<const ModuleAppIDs *> buildIndexFor(hw::HWModuleLike modToProcess);

  // Map modules to their cached child app ID indexes.
  DenseMap<hw::HWModuleLike, ModuleAppIDs *> containerAppIDs;

  bool valid;
  hw::HWSymbolCache symCache;
  Operation *mlirTop;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_APPID_H
