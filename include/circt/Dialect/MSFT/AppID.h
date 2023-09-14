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
  AppIDIndex(Operation *root);

  /// Get the dynamic instance for a particular appid path.
  DynamicInstanceOp getInstance(AppIDPathAttr path);

private:
  class ChildAppIDs {
    friend class AppIDIndex;
    hw::HWModuleLike mod;

    using InstancePath = SmallVector<hw::InnerRefAttr, 2>;
    // Contains paths to all of our child AppIDs. Necessary because AppIDs can
    // skip levels in the instance hierarchy. InstancePaths are relative to
    // 'mod'.
    DenseMap<AppIDAttr, InstancePath> childAppIDPaths;
  };

  /// Get the index for a module. Reference is valid for the lifetime of this
  /// class instance.
  const ChildAppIDs &lookup(hw::HWModuleLike mod);

  /// Get the subinstance (relative to 'submod') for the subpath.
  DynamicInstanceOp getSubInstance(hw::HWModuleLike submod,
                                   ArrayRef<AppIDAttr> subpath);

  // The 'top' MLIR module. Not necessarily a `mlir::ModuleOp` since this will
  // eventually be replaced by `hw::DesignOp`.
  Operation *root;

  // Map modules to their cached child app ID indexes. Use pointers to make
  // movement within the map cheaper.
  DenseMap<hw::HWModuleLike, std::unique_ptr<ChildAppIDs>> containerAppIDs;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_APPID_H
