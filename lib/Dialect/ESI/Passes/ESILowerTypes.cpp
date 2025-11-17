//===- ESILowerTypes.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower high-level ESI types to HW conversions and pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_LOWERESITYPES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::esi;

namespace {
/// Lower all "high-level" ESI types on modules to some lower construct.
struct ESILowerTypesPass
    : public circt::esi::impl::LowerESITypesBase<ESILowerTypesPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Materializations and type conversions to lower ESI data windows.
class LowerTypesConverter : public TypeConverter {
public:
  LowerTypesConverter() {
    addConversion([](Type t) { return t; });
    addConversion([](WindowType window) { return window.getLoweredType(); });
    addSourceMaterialization(wrapMaterialization);
    addTargetMaterialization(unwrapMaterialization);
  }

private:
  static mlir::Value wrapMaterialization(OpBuilder &b, WindowType resultType,
                                         ValueRange inputs, Location loc) {
    if (inputs.size() != 1)
      return mlir::Value();
    return b.createOrFold<WrapWindow>(loc, resultType, inputs[0]);
  }

  static mlir::Value unwrapMaterialization(OpBuilder &b, Type resultType,
                                           ValueRange inputs, Location loc) {
    if (inputs.size() != 1 || !isa<WindowType>(inputs[0].getType()))
      return mlir::Value();
    return b.createOrFold<UnwrapWindow>(loc, resultType, inputs[0]);
  }
};
} // namespace

void ESILowerTypesPass::runOnOperation() {
  ConversionTarget target(getContext());

  // We need to lower instances, modules, and outputs with data windows.
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    return TypeSwitch<Operation *, bool>(op)
        .Case([](igraph::InstanceOpInterface inst) {
          return !(
              llvm::any_of(inst->getOperandTypes(), hw::type_isa<WindowType>) ||
              llvm::any_of(inst->getResultTypes(), hw::type_isa<WindowType>));
        })
        .Case([](hw::HWMutableModuleLike mod) {
          auto isWindowPort = [](hw::PortInfo p) {
            return hw::type_isa<WindowType>(p.type);
          };
          return !(llvm::any_of(mod.getPortList(), isWindowPort));
        })
        .Default([](Operation *op) {
          if (op->hasTrait<OpTrait::ReturnLike>())
            return !llvm::any_of(op->getOperandTypes(),
                                 hw::type_isa<WindowType>);
          return true;
        });
  });

  LowerTypesConverter types;
  RewritePatternSet patterns(&getContext());
  patterns.add<TypeConversionPattern>(types, &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  // Now do a canonicalization pass to clean up any unnecessary wrap-unwrap
  // pairs.
  mlir::ConversionConfig config;
  config.foldingMode = mlir::DialectConversionFoldingMode::BeforePatterns;
  ConversionTarget partialCanonicalizedTarget(getContext());
  RewritePatternSet partialPatterns(&getContext());
  partialCanonicalizedTarget.addIllegalOp<WrapWindow, UnwrapWindow>();
  WrapWindow::getCanonicalizationPatterns(partialPatterns, &getContext());
  UnwrapWindow::getCanonicalizationPatterns(partialPatterns, &getContext());
  if (failed(mlir::applyPartialConversion(getOperation(),
                                          partialCanonicalizedTarget,
                                          std::move(partialPatterns), config)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESITypeLoweringPass() {
  return std::make_unique<ESILowerTypesPass>();
}
