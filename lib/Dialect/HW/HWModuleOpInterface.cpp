//===- HWModuleOpInterface.cpp.h - Implement HWModuleLike  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements HWModuleLike related functionality.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// HWModuleLike Signature Conversion
//===----------------------------------------------------------------------===//

static LogicalResult convertModuleOpTypes(HWModuleLike funcOp,
                                          const TypeConverter &typeConverter,
                                          ConversionPatternRewriter &rewriter) {
  ModuleType type = funcOp.getHWModuleType();
  if (!type)
    return failure();

  // Convert the original function types.
  // Update the function signature in-place.
  SmallVector<ModulePort> newPorts;
  TypeConverter::SignatureConversion result(type.getNumInputs());
  unsigned atInput = 0;
  unsigned curInputs = 0;
  for (auto &p : type.getPorts()) {
    if (p.dir == ModulePort::Direction::Output) {
      SmallVector<Type, 1> newResults;
      if (failed(typeConverter.convertType(p.type, newResults)))
        return failure();
      for (auto np : newResults)
        newPorts.push_back({p.name, np, p.dir});
    } else {
      if (failed(typeConverter.convertSignatureArg(
              atInput++,
              p.dir == ModulePort::Direction::Input ? p.type
                                                    : InOutType::get(p.type),
              result)))
        return failure();
      for (auto np : result.getConvertedTypes().drop_front(curInputs))
        newPorts.push_back({p.name, np, p.dir});
      curInputs = result.getConvertedTypes().size();
    }
  }

  if (failed(rewriter.convertRegionTypes(&funcOp->getRegion(0), typeConverter,
                                         &result)))
    return failure();

  auto newType = ModuleType::get(rewriter.getContext(), newPorts);
  rewriter.updateRootInPlace(funcOp, [&] { funcOp.setHWModuleType(newType); });

  return success();
}

/// Create a default conversion pattern that rewrites the type signature of a
/// FunctionOpInterface op. This only supports ops which use FunctionType to
/// represent their type.
namespace {
struct HWModuleLikeSignatureConversion : public ConversionPattern {
  HWModuleLikeSignatureConversion(StringRef moduleLikeOpName, MLIRContext *ctx,
                                  const TypeConverter &converter)
      : ConversionPattern(converter, moduleLikeOpName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    HWModuleLike funcOp = cast<HWModuleLike>(op);
    return convertModuleOpTypes(funcOp, *typeConverter, rewriter);
  }
};
} // namespace

void circt::hw::populateHWModuleLikeTypeConversionPattern(
    StringRef moduleLikeOpName, RewritePatternSet &patterns,
    TypeConverter &converter) {
  patterns.add<HWModuleLikeSignatureConversion>(
      moduleLikeOpName, patterns.getContext(), converter);
}
