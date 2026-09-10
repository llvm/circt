//===- AppID.h - AppID related code -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Application IDs are paths through the instance hierarchy with some
// application-specific meaning. They allow designers and users to avoid some of
// the design's implementation details.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_APPID_H
#define CIRCT_DIALECT_ESI_APPID_H

#include "circt/Dialect/ESI/ESIAttributes.h"
#include "circt/Dialect/HW/HWSymCache.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace esi {

/// Get the AppID of a particular operation. Returns null if the operation does
/// not have one.
AppIDAttr getAppID(Operation *op);

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

  /// Walk the AppID hierarchy rooted at the specified module.
  LogicalResult
  walk(hw::HWModuleLike top,
       function_ref<void(AppIDPathAttr, ArrayRef<Operation *>)> fn) const;
  LogicalResult
  walk(StringRef top,
       function_ref<void(AppIDPathAttr, ArrayRef<Operation *>)> fn) const;

  /// Return an array of InnerNameRefAttrs representing the relative path to
  /// 'appid' from 'fromMod'.
  FailureOr<ArrayAttr> getAppIDPathAttr(hw::HWModuleLike fromMod,
                                        AppIDAttr appid, Location loc) const;

private:
  /// Walk the AppID hierarchy rooted at the specified module.
  LogicalResult
  walk(hw::HWModuleLike top, hw::HWModuleLike current,
       SmallVectorImpl<AppIDAttr> &pathStack,
       SmallVectorImpl<Operation *> &opStack,
       function_ref<void(AppIDPathAttr, ArrayRef<Operation *>)> fn) const;

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

} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_APPID_H
