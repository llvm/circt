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

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace msft {

class AppIDIndex {
public:
  AppIDIndex(Operation *root);

private:
  Operation *root;

  class ChildAppIDs {
    friend class AppIDIndex;

    using InstancePath = SmallVector<hw::InnerRefAttr, 2>;
    DenseMap<AppIDAttr, InstancePath> appIDPaths;
  };
  DenseMap<Operation *, ChildAppIDs> containerAppIDs;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_APPID_H
