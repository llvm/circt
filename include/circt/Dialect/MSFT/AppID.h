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
namespace igraph {
class InstanceGraph;
}

namespace msft {

class AppIDIndex {
public:
  AppIDIndex(Operation *mlirTop);

  /// Get the dynamic instance for a particular appid path.
  DynamicInstanceOp getInstance(AppIDPathAttr path);

private:
  class ChildAppIDs {
  public:
    ChildAppIDs() : processed(false) {}

    LogicalResult addChildAppID(AppIDAttr id, Operation *op);
    LogicalResult process(hw::HWModuleLike modToProcess,
                          igraph::InstanceGraph &);

  private:
    hw::HWModuleLike mod;
    bool processed;

    // The operation involved in an appid.
    DenseMap<AppIDAttr, Operation *> childAppIDPaths;
  };

  /// Get the subinstance (relative to 'submod') for the subpath.
  DynamicInstanceOp getSubInstance(hw::HWModuleLike mod,
                                   InstanceHierarchyOp inst,
                                   ArrayRef<AppIDAttr> subpath);

  // The 'top' MLIR module. Not necessarily a `mlir::ModuleOp` since this will
  // eventually be replaced by `hw::DesignOp`.
  Operation *mlirTop;

  // Map modules to their cached child app ID indexes.
  DenseMap<hw::HWModuleLike, ChildAppIDs> containerAppIDs;

  hw::HWSymbolCache symCache;
  DenseMap<SymbolRefAttr, InstanceHierarchyOp> dynHierRoots;
  DenseMap<DynamicInstanceOp, DenseMap<hw::InnerRefAttr, DynamicInstanceOp>>
      childIndex;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_APPID_H
