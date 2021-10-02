//===- PlacementDB.h - Placement database -----------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_MSFT_PLACEMENTDB_H
#define CIRCT_DIALECT_MSFT_PLACEMENTDB_H

#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

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
class PlacementDB {
public:
  /// Create a DB treating 'top' as the root module.
  PlacementDB(Operation *top);

  // TODO: Add calls to model the device primitive locations.

  /// In addition to an Operation which is the instance at the level being
  /// modeled in MLIR, the instance path within the MLIR instance is often
  /// necessary as most often the instance is an extern module.
  struct PlacedInstance {
    RootedInstancePathAttr path;
    llvm::StringRef subpath;
    Operation *op;
  };

  /// Assign an instance to a primitive. Return false if another instance is
  /// already placed at that location.
  LogicalResult addPlacement(PhysLocationAttr, PlacedInstance);
  /// Using the operation attributes, add the proper placements to the database.
  /// Return the number of placements which weren't added due to conflicts.
  size_t addPlacements(FlatSymbolRefAttr rootMod, mlir::Operation *);
  /// Walk the entire design adding placements root at the top module.
  size_t addDesignPlacements();

  /// Lookup the instance at a particular location.
  Optional<PlacedInstance> getInstanceAt(PhysLocationAttr);

  /// Walk the placement information in some sort of reasonable order.
  void walkPlacements(function_ref<void(PhysLocationAttr, PlacedInstance)>);

private:
  MLIRContext *ctxt;
  Operation *top;

  using DimDevType = DenseMap<DeviceType, PlacedInstance>;
  using DimNumMap = DenseMap<size_t, DimDevType>;
  using DimYMap = DenseMap<size_t, DimNumMap>;
  using DimXMap = DenseMap<size_t, DimYMap>;

  DimXMap placements;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_PLACEMENTDB_H
