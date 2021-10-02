//===- DeviceDB.h - Device database -----------------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_MSFT_DEVICEDB_H
#define CIRCT_DIALECT_MSFT_DEVICEDB_H

#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace msft {

/// A data structure to contain locations of the primitives on the
/// device.
class DeviceDB {
public:
  /// Create a DB treating 'top' as the root module.
  DeviceDB();

  /// Place a primitive at a location.
  LogicalResult addPrimitive(PhysLocationAttr);
  /// Check to see if a primitive exists.
  bool isValidLocation(PhysLocationAttr);

private:
  using DimPrimitiveType = DenseSet<PrimitiveType>;
  using DimNumMap = DenseMap<size_t, DimPrimitiveType>;
  using DimYMap = DenseMap<size_t, DimNumMap>;
  using DimXMap = DenseMap<size_t, DimYMap>;

  /// Get the leaf node. Abstract this out to make it easier to change the
  /// underlying data structure.
  DimPrimitiveType &getLeaf(PhysLocationAttr);

  DimXMap placements;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_DEVICEDB_H
