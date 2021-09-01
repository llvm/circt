//===- DeviceDB.h - Device database -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represent the possible placements and actual placements of primitives on an
// FPGA.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_DEVICEDB_H
#define CIRCT_DIALECT_MSFT_DEVICEDB_H

#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Operation.h"

namespace circt {
namespace msft {

/// A data structure to contain both the locations of the primitives on the
/// device and instance assignments to said primitives locations, aka
/// placements.
///
/// Holds pointers into the IR, which may become invalid as a result of IR
/// transforms. As a result, this class should only be used for analysis and
/// then thrown away. It is permissible to persist it through transformations so
/// long as it is maintained along with the transformations.
class DeviceDB {
public:
  // TODO: Add calls to model the device primitive locations.

  /// In addition to an Operation which is the instance at the level being
  /// modeled in MLIR, the instance path within the MLIR instance is often
  /// necessary as most often the instance is an extern module.
  using PlacedInstance = std::pair<InstanceIDAttr, llvm::StringRef>;

  /// Assign an instance to a primitive. Return false if another instance is
  /// already placed at that location.
  bool addPlacement(PhysLocationAttr, PlacedInstance);
  /// Using the operation attributes, add the proper placements to the database.
  /// Return the number of placements which weren't added due to conflicts.
  size_t addPlacements(mlir::Operation *);
  /// Walk the entire instance hierarchy with 'top' as the root module.
  size_t addDesignPlacements(mlir::Operation *top);

  /// Lookup the instance at a particular location.
  llvm::Optional<PlacedInstance> getInstanceAt(PhysLocationAttr);
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_DEVICEDB_H
