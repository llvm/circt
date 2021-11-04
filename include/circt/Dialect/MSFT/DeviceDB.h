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
class PrimitiveDB {
public:
  /// Create a DB treating 'top' as the root module.
  PrimitiveDB(MLIRContext *);

  /// Place a primitive at a location.
  LogicalResult addPrimitive(PhysLocationAttr);
  /// Check to see if a primitive exists.
  bool isValidLocation(PhysLocationAttr);

  /// Iterate over all the primitive locations, executing 'callback' on each
  /// one.
  void foreach (function_ref<void(PhysLocationAttr)> callback) const;

private:
  using DimPrimitiveType = DenseSet<PrimitiveType>;
  using DimNumMap = DenseMap<size_t, DimPrimitiveType>;
  using DimYMap = DenseMap<size_t, DimNumMap>;
  using DimXMap = DenseMap<size_t, DimYMap>;

  /// Get the leaf node. Abstract this out to make it easier to change the
  /// underlying data structure.
  DimPrimitiveType &getLeaf(PhysLocationAttr);
  // TODO: Create read-only version of getLeaf.

  DimXMap placements;
  MLIRContext *ctxt;
};

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
  PlacementDB(Operation *top, const PrimitiveDB &seed);

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

  /// Find the nearest unoccupied primitive location to 'nearestToY' in
  /// 'column'.
  PhysLocationAttr getNearestFreeInColumn(PrimitiveType prim, uint64_t column,
                                          uint64_t nearestToY);

  /// Walk the placement information in some sort of reasonable order. Bounds
  /// restricts the walk to a rectangle of [xmin, xmax, ymin, ymax] (inclusive),
  /// with -1 meaning unbounded.
  void walkPlacements(function_ref<void(PhysLocationAttr, PlacedInstance)>,
                      std::tuple<int64_t, int64_t, int64_t, int64_t> bounds =
                          std::make_tuple(-1, -1, -1, -1),
                      Optional<PrimitiveType> primType = {});

  /// Helper function to check if the database is empty.
  bool empty();

private:
  MLIRContext *ctxt;
  Operation *top;

  using DimDevType = DenseMap<PrimitiveType, PlacedInstance>;
  using DimNumMap = DenseMap<size_t, DimDevType>;
  using DimYMap = DenseMap<size_t, DimNumMap>;
  using DimXMap = DenseMap<size_t, DimYMap>;

  /// Get the leaf node. Abstract this out to make it easier to change the
  /// underlying data structure.
  Optional<PlacedInstance *> getLeaf(PhysLocationAttr);

  DimXMap placements;
  bool seeded;
};

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_DEVICEDB_H
