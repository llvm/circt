//===- PlacementDB.cpp - Implement a device database ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/PlacementDB.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// NOTE: Nothing in this implementation is in any way the most optimal
// implementation. We put off deciding what the correct data structure is until
// we have a better handle of the operations it must accelerate. Performance is
// not an immediate goal.
//===----------------------------------------------------------------------===//

PlacementDB::PlacementDB(Operation *top)
    : ctxt(top->getContext()), top(top), seeded(false) {}
PlacementDB::PlacementDB(Operation *top, const DeviceDB &seed)
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
    return inst.op->emitOpError("Could not apply placement. Invalid location");
  PlacedInstance *cell = *leaf;
  if (cell->op != nullptr)
    return inst.op->emitOpError("Could not apply placement ")
           << loc << ". Position already occupied by " << cell->op << ".";
  *cell = inst;
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
  walkColumnPlacements(columnNum, [&nearest, columnNum](PhysLocationAttr loc,
                                                        PlacedInstance inst) {
    if (inst.op)
      return;
    if (!nearest) {
      nearest = loc;
      return;
    }
    int64_t curDist = std::abs((int64_t)columnNum - (int64_t)nearest.getY());
    int64_t replDist = std::abs((int64_t)columnNum - (int64_t)loc.getY());
    if (replDist < curDist)
      nearest = loc;
  });
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
    function_ref<void(PhysLocationAttr, PlacedInstance)> callback) {
  // X loop.
  for (auto colF = placements.begin(), colE = placements.end(); colF != colE;
       ++colF) {
    size_t x = colF->getFirst();
    walkColumnPlacements(x, callback);
  }
}

/// Walk the column placements in some sort of reasonable order.
void PlacementDB::walkColumnPlacements(
    uint64_t columnNum,
    function_ref<void(PhysLocationAttr, PlacedInstance)> callback) {

  auto colF = placements.find(columnNum);
  if (colF == placements.end())
    return;
  DimYMap yMap = colF->getSecond();

  // Y loop.
  for (auto rowF = yMap.begin(), rowE = yMap.end(); rowF != rowE; ++rowF) {
    size_t y = rowF->getFirst();
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
        PlacedInstance inst = devF->getSecond();

        // Marshall and run the callback.
        PhysLocationAttr loc = PhysLocationAttr::get(
            ctxt, PrimitiveTypeAttr::get(ctxt, devtype), columnNum, y, num);
        callback(loc, inst);
      }
    }
  }
}
