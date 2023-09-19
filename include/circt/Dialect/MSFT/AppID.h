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

class AppIDIndex {
public:
  AppIDIndex(Operation *mlirTop);
  ~AppIDIndex() {
    for (auto [appId, childAppIDs] : containerAppIDs)
      delete childAppIDs;
  }
  bool isValid() const { return valid; }

  /// Get the dynamic instance for a particular appid path.
  FailureOr<DynamicInstanceOp> getInstance(AppIDPathAttr path,
                                           Location loc) const;

private:
  class ChildAppIDs {
  public:
    ChildAppIDs() {}

    LogicalResult add(AppIDAttr id, Operation *op, bool inherited);
    FailureOr<Operation *> lookup(Operation *, AppIDAttr id,
                                  Location loc) const;
    auto getAppIDs() const { return llvm::make_first_range(childAppIDPaths); }

  private:
    // The operation involved in an appid.
    DenseMap<AppIDAttr, Operation *> childAppIDPaths;
  };

  /// Get the subinstance (relative to 'submod') for the subpath.
  FailureOr<DynamicInstanceOp> getSubInstance(hw::HWModuleLike mod,
                                              Operation *dynInstParent,
                                              ArrayRef<AppIDAttr> subpath,
                                              Location loc) const;
  FailureOr<std::pair<DynamicInstanceOp, hw::InnerSymbolOpInterface>>
  getSubInstance(hw::HWModuleLike mod, Operation *inst, AppIDAttr appid,
                 Location loc) const;
  DynamicInstanceOp getOrCreate(Operation *parent, hw::InnerRefAttr name,
                                Location) const;

  FailureOr<const ChildAppIDs *> process(hw::HWModuleLike modToProcess);

  // Map modules to their cached child app ID indexes.
  DenseMap<hw::HWModuleLike, ChildAppIDs *> containerAppIDs;

  hw::HWSymbolCache symCache;
  bool valid;
  Operation *mlirTop;

  // Caches
  mutable DenseMap<SymbolRefAttr, InstanceHierarchyOp> dynHierRoots;
  using NamedChildren = DenseMap<hw::InnerRefAttr, DynamicInstanceOp>;
  mutable DenseMap<Operation *, NamedChildren> dynInstChildLookup;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_APPID_H
