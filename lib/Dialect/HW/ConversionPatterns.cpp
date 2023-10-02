//===- ConversionPatterns.cpp - Common Conversion patterns ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/HW/HWTypes.h"

using namespace circt;

// Converts a function type wrt. the given type converter.
static FunctionType convertFunctionType(const TypeConverter &typeConverter,
                                        FunctionType type) {
  // Convert the original function types.
  llvm::SmallVector<Type> res, arg;
  llvm::transform(type.getResults(), std::back_inserter(res),
                  [&](Type t) { return typeConverter.convertType(t); });
  llvm::transform(type.getInputs(), std::back_inserter(arg),
                  [&](Type t) { return typeConverter.convertType(t); });

  return FunctionType::get(type.getContext(), arg, res);
}

// Converts a function type wrt. the given type converter.
static hw::ModuleType convertModuleType(const TypeConverter &typeConverter,
                                        hw::ModuleType type) {
  // Convert the original function types.
  SmallVector<hw::ModulePort> ports(type.getPorts());
  for (auto &p : ports)
    p.type = typeConverter.convertType(p.type);
  return hw::ModuleType::get(type.getContext(), ports);
}

LogicalResult circt::doTypeConversion(Operation *op, ValueRange operands,
                                      ConversionPatternRewriter &rewriter,
                                      const TypeConverter *typeConverter) {
  // Convert the TypeAttrs.
  llvm::SmallVector<NamedAttribute, 4> newAttrs;
  newAttrs.reserve(op->getAttrs().size());
  for (auto attr : op->getAttrs()) {
    if (auto typeAttr = attr.getValue().dyn_cast<TypeAttr>()) {
      auto innerType = typeAttr.getValue();
      // TypeConvert::convertType doesn't handle function types, so we need to
      // handle them manually.
      if (auto funcType = innerType.dyn_cast<FunctionType>())
        innerType = convertFunctionType(*typeConverter, funcType);
      else if (auto modType = innerType.dyn_cast<hw::ModuleType>())
        innerType = convertModuleType(*typeConverter, modType);
      else
        innerType = typeConverter->convertType(innerType);
      newAttrs.emplace_back(attr.getName(), TypeAttr::get(innerType));
    } else {
      newAttrs.push_back(attr);
    }
  }

  // Convert the result types.
  llvm::SmallVector<Type, 4> newResults;
  if (failed(typeConverter->convertTypes(op->getResultTypes(), newResults)))
    return rewriter.notifyMatchFailure(op->getLoc(), "type conversion failed");

  // Build the state for the edited clone.
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, newAttrs, op->getSuccessors());
  for (size_t i = 0, e = op->getNumRegions(); i < e; ++i)
    state.addRegion();

  // Must create the op before running any modifications on the regions so that
  // we don't crash with '-debug' and so we have something to 'root update'.
  Operation *newOp = rewriter.create(state);

  // Move the regions over, converting the signatures as we go.
  rewriter.startRootUpdate(newOp);
  for (size_t i = 0, e = op->getNumRegions(); i < e; ++i) {
    Region &region = op->getRegion(i);
    Region *newRegion = &newOp->getRegion(i);

    // TypeConverter::SignatureConversion drops argument locations, so we need
    // to manually copy them over (a verifier in e.g. HWModule checks this).
    llvm::SmallVector<Location, 4> argLocs;
    for (auto arg : region.getArguments())
      argLocs.push_back(arg.getLoc());

    // Move the region and convert the region args.
    rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    TypeConverter::SignatureConversion result(newRegion->getNumArguments());
    if (failed(typeConverter->convertSignatureArgs(
            newRegion->getArgumentTypes(), result)))
      return rewriter.notifyMatchFailure(op->getLoc(),
                                         "type conversion failed");
    rewriter.applySignatureConversion(newRegion, result, typeConverter);

    // Apply the argument locations.
    for (auto [arg, loc] : llvm::zip(newRegion->getArguments(), argLocs))
      arg.setLoc(loc);
  }
  rewriter.finalizeRootUpdate(newOp);

  rewriter.replaceOp(op, newOp->getResults());
  return success();
}
