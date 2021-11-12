//===- DeviceDB.cpp - Implement a device database -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// PrimitiveDB.
//===----------------------------------------------------------------------===//
// NOTE: Nothing in this implementation is in any way the most optimal
// implementation. We put off deciding what the correct data structure is until
// we have a better handle of the operations it must accelerate. Performance is
// not an immediate goal.
//===----------------------------------------------------------------------===//

PrimitiveDB::PrimitiveDB(MLIRContext *ctxt) : ctxt(ctxt) {}

/// Assign an instance to a primitive. Return false if another instance is
/// already placed at that location.
LogicalResult PrimitiveDB::addPrimitive(PhysLocationAttr loc) {
  DenseSet<PrimitiveType> &primsAtLoc = getLeaf(loc);
  PrimitiveType prim = loc.getPrimitiveType().getValue();
  if (primsAtLoc.contains(prim))
    return failure();
  primsAtLoc.insert(prim);
  return success();
}

/// Assign an instance to a primitive. Return false if another instance is
/// already placed at that location.
/// Check to see if a primitive exists.
bool PrimitiveDB::isValidLocation(PhysLocationAttr loc) {
  DenseSet<PrimitiveType> primsAtLoc = getLeaf(loc);
  return primsAtLoc.contains(loc.getPrimitiveType().getValue());
}

PrimitiveDB::DimPrimitiveType &PrimitiveDB::getLeaf(PhysLocationAttr loc) {
  return placements[loc.getX()][loc.getY()][loc.getNum()];
}

void PrimitiveDB::foreach (
    function_ref<void(PhysLocationAttr)> callback) const {
  for (auto x : placements)
    for (auto y : x.second)
      for (auto n : y.second)
        for (auto p : n.second)
          callback(PhysLocationAttr::get(ctxt, PrimitiveTypeAttr::get(ctxt, p),
                                         x.first, y.first, n.first));
}

//===----------------------------------------------------------------------===//
// PlacementDB.
//===----------------------------------------------------------------------===//
// NOTE: Nothing in this implementation is in any way the most optimal
// implementation. We put off deciding what the correct data structure is until
// we have a better handle of the operations it must accelerate. Performance is
// not an immediate goal.
//===----------------------------------------------------------------------===//

PlacementDB::PlacementDB(Operation *top)
    : ctxt(top->getContext()), top(top), seeded(false) {}
PlacementDB::PlacementDB(Operation *top, const PrimitiveDB &seed)
    : ctxt(top->getContext()), top(top), seeded(false) {

  seed.foreach ([this](PhysLocationAttr loc) { (void)addPlacement(loc, {}); });
  seeded = true;
}

/// Assign an instance to a primitive. Return false if another instance is
/// already placed at that location.
LogicalResult PlacementDB::addPlacement(PhysLocationAttr loc,
                                        PlacedInstance inst) {

  Optional<PlacedInstance *> leaf = getLeaf(loc);
  if (!leaf)
    return inst.op->emitOpError("Could not apply placement. Invalid location: ")
           << loc;
  PlacedInstance *cell = *leaf;
  if (cell->op != nullptr)
    return inst.op->emitOpError("Could not apply placement ")
           << loc << ". Position already occupied by " << cell->op << ".";
  *cell = inst;
  return success();
}

/// Assign an operation to a physical region. Return false on failure.
LogicalResult PlacementDB::addPlacement(PhysicalRegionRefAttr regionRef,
                                        PlacedInstance inst) {
  auto topModule = inst.op->getParentOfType<mlir::ModuleOp>();
  auto physicalRegion =
      topModule.lookupSymbol<PhysicalRegionOp>(regionRef.getPhysicalRegion());
  if (!physicalRegion)
    return inst.op->emitOpError("referenced non-existant PhysicalRegion named ")
           << regionRef.getPhysicalRegion().getValue();

  regionPlacements.emplace_back(regionRef, inst);
  return success();
}

/// Using the operation attributes, add the proper placements to the database.
/// Return the number of placements which weren't added due to conflicts.
size_t PlacementDB::addPlacements(FlatSymbolRefAttr rootMod,
                                  mlir::Operation *op) {
  size_t numFailed = 0;
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef attrName = attr.first;
    llvm::TypeSwitch<Attribute>(attr.second)

        // Handle switch instance.
        .Case([&](SwitchInstanceAttr instSwitch) {
          for (auto caseAttr : instSwitch.getCases()) {
            RootedInstancePathAttr instPath = caseAttr.getInst();

            // Filter out all paths which aren't related to this DB.
            if (instPath.getRootModule() != rootMod)
              continue;

            // If we recognize the type, validate and add it.
            if (auto loc = caseAttr.getAttr().dyn_cast<PhysLocationAttr>()) {
              if (!attrName.startswith("loc:")) {
                op->emitOpError(
                    "PhysLoc attributes must have names starting with 'loc'");
                ++numFailed;
                continue;
              }
              LogicalResult added = addPlacement(
                  loc, PlacedInstance{instPath, attrName.substr(4), op});
              if (failed(added))
                ++numFailed;
            } else if (auto loc = caseAttr.getAttr()
                                      .dyn_cast<PhysicalRegionRefAttr>()) {
              LogicalResult added =
                  addPlacement(loc, PlacedInstance{instPath, StringRef(), op});
              if (failed(added))
                ++numFailed;
            }
          }
        })

        // Physloc outside of a switch instance is not valid.
        .Case([op, &numFailed](PhysLocationAttr) {
          ++numFailed;
          op->emitOpError("PhysLoc attribute must be inside an "
                          "instance switch attribute");
        })

        // Ignore attributes we don't understand.
        .Default([](Attribute) {});
  }
  return numFailed;
}

/// Walk the entire design adding placements.
size_t PlacementDB::addDesignPlacements() {
  size_t failed = 0;
  FlatSymbolRefAttr rootModule = FlatSymbolRefAttr::get(top);
  auto mlirModule = top->getParentOfType<mlir::ModuleOp>();
  mlirModule.walk(
      [&](Operation *op) { failed += addPlacements(rootModule, op); });
  return failed;
}

/// Lookup the instance at a particular location.
Optional<PlacementDB::PlacedInstance>
PlacementDB::getInstanceAt(PhysLocationAttr loc) {
  auto innerMap = placements[loc.getX()][loc.getY()][loc.getNum()];
  auto instF = innerMap.find(loc.getPrimitiveType().getValue());
  if (instF == innerMap.end())
    return {};
  return instF->getSecond();
}

PhysLocationAttr PlacementDB::getNearestFreeInColumn(PrimitiveType prim,
                                                     uint64_t columnNum,
                                                     uint64_t nearestToY) {
  // Simplest possible algorithm.
  PhysLocationAttr nearest = {};
  walkPlacements(
      [&nearest, columnNum](PhysLocationAttr loc, PlacedInstance inst) {
        if (inst.op)
          return;
        if (!nearest) {
          nearest = loc;
          return;
        }
        int64_t curDist =
            std::abs((int64_t)columnNum - (int64_t)nearest.getY());
        int64_t replDist = std::abs((int64_t)columnNum - (int64_t)loc.getY());
        if (replDist < curDist)
          nearest = loc;
      },
      std::make_tuple(columnNum, columnNum, -1, -1), prim);
  return nearest;
}

Optional<PlacementDB::PlacedInstance *>
PlacementDB::getLeaf(PhysLocationAttr loc) {
  PrimitiveType primType = loc.getPrimitiveType().getValue();

  DimNumMap &nums = placements[loc.getX()][loc.getY()];
  if (!seeded)
    return &nums[loc.getNum()][primType];
  if (!nums.count(loc.getNum()))
    return {};

  DimDevType &primitives = nums[loc.getNum()];
  if (primitives.count(primType) == 0)
    return {};
  return &primitives[primType];
}

/// Walker for placements.
void PlacementDB::walkPlacements(
    function_ref<void(PhysLocationAttr, PlacedInstance)> callback,
    std::tuple<int64_t, int64_t, int64_t, int64_t> bounds,
    Optional<PrimitiveType> primType) {
  uint64_t xmin = std::get<0>(bounds) < 0 ? 0 : std::get<0>(bounds);
  uint64_t xmax = std::get<1>(bounds) < 0 ? std::numeric_limits<uint64_t>::max()
                                          : (uint64_t)std::get<1>(bounds);
  uint64_t ymin = std::get<2>(bounds) < 0 ? 0 : std::get<2>(bounds);
  uint64_t ymax = std::get<3>(bounds) < 0 ? std::numeric_limits<uint64_t>::max()
                                          : (uint64_t)std::get<3>(bounds);

  // TODO: Since the data structures we're using aren't sorted, the best we can
  // do is iterate and filter. Once we get to performance, we'll figure out the
  // right data structure.

  // X loop.
  for (auto colF = placements.begin(), colE = placements.end(); colF != colE;
       ++colF) {
    size_t x = colF->getFirst();
    if (x < xmin || x > xmax)
      continue;
    DimYMap yMap = colF->getSecond();

    // Y loop.
    for (auto rowF = yMap.begin(), rowE = yMap.end(); rowF != rowE; ++rowF) {
      size_t y = rowF->getFirst();
      if (y < ymin || y > ymax)
        continue;
      DimNumMap numMap = rowF->getSecond();

      // Num loop.
      for (auto numF = numMap.begin(), numE = numMap.end(); numF != numE;
           ++numF) {
        size_t num = numF->getFirst();
        DimDevType devMap = numF->getSecond();

        // DevType loop.
        for (auto devF = devMap.begin(), devE = devMap.end(); devF != devE;
             ++devF) {
          PrimitiveType devtype = devF->getFirst();
          if (primType && devtype != *primType)
            continue;
          PlacedInstance inst = devF->getSecond();

          // Marshall and run the callback.
          PhysLocationAttr loc = PhysLocationAttr::get(
              ctxt, PrimitiveTypeAttr::get(ctxt, devtype), x, y, num);
          callback(loc, inst);
        }
      }
    }
  }
}

/// Walk the region placement information.
void PlacementDB::walkRegionPlacements(
    function_ref<void(PhysicalRegionRefAttr, PlacedInstance)> callback) {
  for (auto iter = regionPlacements.begin(), end = regionPlacements.end();
       iter != end; ++iter)
    callback(iter->first, iter->second);
}
