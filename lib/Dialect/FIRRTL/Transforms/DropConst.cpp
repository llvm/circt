//===- DropConst.cpp - Check and remove const types -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DropConst pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/SaveAndRestore.h"

#define DEBUG_TYPE "drop-const"

using namespace circt;
using namespace firrtl;

// NOLINTBEGIN(misc-no-recursion)
/// Return this type with a 'const' modifiers dropped
static FIRRTLBaseType getAllConstDroppedType(FIRRTLBaseType type) {
  if (!type.containsConst())
    return type;
  return TypeSwitch<FIRRTLBaseType, FIRRTLBaseType>(type)
      .Case<ClockType, ResetType, AsyncResetType, AnalogType, SIntType,
            UIntType>([](auto type) { return type.getConstType(false); })
      .Case<BundleType>([](BundleType type) {
        SmallVector<BundleType::BundleElement> constDroppedElements(
            llvm::map_range(
                type.getElements(), [](BundleType::BundleElement element) {
                  element.type = getAllConstDroppedType(element.type);
                  return element;
                }));
        return BundleType::get(type.getContext(), constDroppedElements, false);
      })
      .Case<FVectorType>([](FVectorType type) {
        return FVectorType::get(getAllConstDroppedType(type.getElementType()),
                                type.getNumElements(), false);
      })
      .Case<FEnumType>([](FEnumType type) {
        SmallVector<FEnumType::EnumElement> constDroppedElements(
            llvm::map_range(
                type.getElements(), [](FEnumType::EnumElement element) {
                  element.type = getAllConstDroppedType(element.type);
                  return element;
                }));
        return FEnumType::get(type.getContext(), constDroppedElements, false);
        ;
      })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return FIRRTLBaseType();
      });
}
// NOLINTEND(misc-no-recursion)

/// Returns null type if no conversion is needed.
static FIRRTLBaseType convertType(FIRRTLBaseType type) {
  auto nonConstType = getAllConstDroppedType(type);
  return nonConstType != type ? nonConstType : FIRRTLBaseType{};
}

/// Returns null type if no conversion is needed.
static Type convertType(Type type) {
  if (auto base = type.dyn_cast<FIRRTLBaseType>()) {
    return convertType(base);
  }

  if (auto refType = type.dyn_cast<RefType>()) {
    if (auto converted = convertType(refType.getType()))
      return RefType::get(converted, refType.getForceable());
  }

  return {};
}

namespace {
class DropConstPass : public DropConstBase<DropConstPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto fmodule = dyn_cast<FModuleOp>(*module);

    // Convert the module body if present
    if (fmodule) {
      fmodule->walk([](Operation *op) {
        if (auto constCastOp = dyn_cast<ConstCastOp>(op)) {
          // Remove any `ConstCastOp`, replacing results with inputs
          constCastOp.getResult().replaceAllUsesWith(constCastOp.getInput());
          constCastOp->erase();
          return;
        }

        for (auto result : op->getResults()) {
          if (auto convertedType = convertType(result.getType()))
            result.setType(convertedType);
        }
      });
    }

    // Find 'const' ports
    auto portTypes = SmallVector<Attribute>(module.getPortTypes());
    for (size_t portIndex = 0, numPorts = module.getPortTypes().size();
         portIndex < numPorts; ++portIndex) {
      if (auto convertedType = convertType(module.getPortType(portIndex))) {
        // If this is an FModuleOp, register the block argument to drop 'const'
        if (fmodule)
          fmodule.getArgument(portIndex).setType(convertedType);
        portTypes[portIndex] = TypeAttr::get(convertedType);
      }
    }

    // Update the module signature with non-'const' ports
    module->setAttr(FModuleLike::getPortTypesAttrName(),
                    ArrayAttr::get(module.getContext(), portTypes));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDropConstPass() {
  return std::make_unique<DropConstPass>();
}
