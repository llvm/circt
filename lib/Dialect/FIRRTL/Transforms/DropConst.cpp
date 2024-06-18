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

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_DROPCONST
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

/// Returns null type if no conversion is needed.
static FIRRTLBaseType convertType(FIRRTLBaseType type) {
  auto nonConstType = type.getAllConstDroppedType();
  return nonConstType != type ? nonConstType : FIRRTLBaseType{};
}

/// Returns null type if no conversion is needed.
static Type convertType(Type type) {
  if (auto base = type_dyn_cast<FIRRTLBaseType>(type)) {
    return convertType(base);
  }

  if (auto refType = type_dyn_cast<RefType>(type)) {
    if (auto converted = convertType(refType.getType()))
      return RefType::get(converted, refType.getForceable(),
                          refType.getLayer());
  }

  return {};
}

namespace {
class DropConstPass : public circt::firrtl::impl::DropConstBase<DropConstPass> {
  void runOnOperation() override {

    // Update signatures of all module-likes.
    auto moduleLikes = getOperation().getOps<FModuleLike>();
    for (auto mod : moduleLikes) {
      //// Update the module signature with non-'const' ports
      SmallVector<Attribute> portTypes;
      portTypes.reserve(mod.getNumPorts());
      bool convertedAny = false;
      llvm::transform(mod.getPortTypes(), std::back_inserter(portTypes),
                      [&](Attribute type) -> Attribute {
                        if (auto convertedType =
                                convertType(cast<TypeAttr>(type).getValue())) {
                          convertedAny = true;
                          return TypeAttr::get(convertedType);
                        }
                        return type;
                      });
      if (convertedAny)
        mod->setAttr(FModuleLike::getPortTypesAttrName(),
                     ArrayAttr::get(mod.getContext(), portTypes));
    };

    // Rewrite module bodies in parallel.
    // Filter on FModuleOp specifically as there's no "hasBody()".
    mlir::parallelForEach(
        &getContext(),
        llvm::make_filter_range(moduleLikes, llvm::IsaPred<FModuleOp>),
        [](auto module) {
          // Convert the module body if present
          module->walk([](Operation *op) {
            if (auto constCastOp = dyn_cast<ConstCastOp>(op)) {
              // Remove any `ConstCastOp`, replacing results with inputs
              constCastOp.getResult().replaceAllUsesWith(
                  constCastOp.getInput());
              constCastOp->erase();
              return;
            }

            // Convert any block arguments
            for (auto &region : op->getRegions())
              for (auto &block : region.getBlocks())
                for (auto argument : block.getArguments())
                  if (auto convertedType = convertType(argument.getType()))
                    argument.setType(convertedType);

            for (auto result : op->getResults())
              if (auto convertedType = convertType(result.getType()))
                result.setType(convertedType);
          });
        });

    markAnalysesPreserved<InstanceGraph>();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDropConstPass() {
  return std::make_unique<DropConstPass>();
}
