//===- DeviceDB.cpp - Implement a device database -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/DeviceDB.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// NOTE: Nothing in this implementation is in any way the most optimal
// implementation. We put off deciding what the correct data structure is until
// we have a better handle of the operations it must accelerate. Performance is
// not an immediate goal.
//===----------------------------------------------------------------------===//

DeviceDB::DeviceDB(MLIRContext *ctxt, Operation *top) : ctxt(ctxt), top(top) {}

/// Assign an instance to a primitive. Return false if another instance is
/// already placed at that location.
LogicalResult DeviceDB::addPlacement(PhysLocationAttr loc,
                                     PlacedInstance inst) {
  PlacedInstance &cell = placements[loc.getX()][loc.getY()][loc.getNum()]
                                   [loc.getDevType().getValue()];
  if (cell.op != nullptr)
    return inst.op->emitOpError("Could not apply placement ")
           << loc << ". Position already occupied by " << cell.op << ".";
  cell = inst;
  return success();
}

/// Using the operation attributes, add the proper placements to the database.
/// Return the number of placements which weren't added due to conflicts.
size_t DeviceDB::addPlacements(FlatSymbolRefAttr rootMod, mlir::Operation *op) {
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
size_t DeviceDB::addDesignPlacements() {
  size_t failed = 0;
  FlatSymbolRefAttr rootModule = FlatSymbolRefAttr::get(top);
  auto mlirModule = top->getParentOfType<mlir::ModuleOp>();
  mlirModule.walk(
      [&](Operation *op) { failed += addPlacements(rootModule, op); });
  return failed;
}

/// Lookup the instance at a particular location.
Optional<DeviceDB::PlacedInstance>
DeviceDB::getInstanceAt(PhysLocationAttr loc) {
  auto innerMap = placements[loc.getX()][loc.getY()][loc.getNum()];
  auto instF = innerMap.find(loc.getDevType().getValue());
  if (instF == innerMap.end())
    return {};
  return instF->getSecond();
}

/// Walker for placements.
void DeviceDB::walkPlacements(
    function_ref<void(PhysLocationAttr, PlacedInstance)> callback) {
  // X loop.
  for (auto colF = placements.begin(), colE = placements.end(); colF != colE;
       ++colF) {
    size_t x = colF->getFirst();
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
          DeviceType devtype = devF->getFirst();
          PlacedInstance inst = devF->getSecond();

          // Marshall and run the callback.
          PhysLocationAttr loc = PhysLocationAttr::get(
              ctxt, DeviceTypeAttr::get(ctxt, devtype), x, y, num);
          callback(loc, inst);
        }
      }
    }
  }
}
