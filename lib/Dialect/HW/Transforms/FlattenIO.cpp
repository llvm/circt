//===- FlattenIO.cpp - HW I/O flattening pass -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

static bool isStructType(Type type) {
  return hw::getCanonicalType(type).isa<hw::StructType>();
}

static hw::StructType getStructType(Type type) {
  return hw::getCanonicalType(type).dyn_cast<hw::StructType>();
}

// Legal if no in- or output type is a struct.
static bool isLegalModLikeOp(hw::HWModuleLike moduleLikeOp) {
  return llvm::none_of(moduleLikeOp.getHWModuleType().getPortTypes(),
                       isStructType);
}

static llvm::SmallVector<Type> getInnerTypes(hw::StructType t) {
  llvm::SmallVector<Type> inner;
  t.getInnerTypes(inner);
  for (auto [index, innerType] : llvm::enumerate(inner))
    inner[index] = hw::getCanonicalType(innerType);
  return inner;
}

namespace {

// Replaces an output op with a new output with flattened (exploded) structs.
struct OutputOpConversion : public OpConversionPattern<hw::OutputOp> {
  OutputOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     DenseSet<Operation *> *opVisited)
      : OpConversionPattern(typeConverter, context), opVisited(opVisited) {}

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> convOperands;

    // Flatten the operands.
    for (auto operand : adaptor.getOperands()) {
      if (auto structType = getStructType(operand.getType())) {
        auto explodedStruct = rewriter.create<hw::StructExplodeOp>(
            op.getLoc(), getInnerTypes(structType), operand);
        llvm::copy(explodedStruct.getResults(),
                   std::back_inserter(convOperands));
      } else {
        convOperands.push_back(operand);
      }
    }

    // And replace.
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, convOperands);
    opVisited->insert(op->getParentOp());
    return success();
  }
  DenseSet<Operation *> *opVisited;
};

struct InstanceOpConversion : public OpConversionPattern<hw::InstanceOp> {
  InstanceOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                       DenseSet<hw::InstanceOp> *convertedOps)
      : OpConversionPattern(typeConverter, context),
        convertedOps(convertedOps) {}

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Flatten the operands.
    llvm::SmallVector<Value> convOperands;
    for (auto operand : adaptor.getOperands()) {
      if (auto structType = getStructType(operand.getType())) {
        auto explodedStruct = rewriter.create<hw::StructExplodeOp>(
            loc, getInnerTypes(structType), operand);
        llvm::copy(explodedStruct.getResults(),
                   std::back_inserter(convOperands));
      } else {
        convOperands.push_back(operand);
      }
    }

    // Create the new instance...
    Operation *targetModule = SymbolTable::lookupNearestSymbolFrom(
        op, op.getReferencedModuleNameAttr());
    auto newInstance = rewriter.create<hw::InstanceOp>(
        loc, targetModule, op.getInstanceName(), convOperands);

    // re-create any structs in the result.
    llvm::SmallVector<Value> convResults;
    size_t oldResultCntr = 0;
    for (size_t resIndex = 0; resIndex < newInstance.getNumResults();
         ++resIndex) {
      Type oldResultType = op.getResultTypes()[oldResultCntr];
      if (auto structType = getStructType(oldResultType)) {
        size_t nElements = structType.getElements().size();
        auto implodedStruct = rewriter.create<hw::StructCreateOp>(
            loc, structType,
            newInstance.getResults().slice(resIndex, nElements));
        convResults.push_back(implodedStruct.getResult());
        resIndex += nElements - 1;
      } else
        convResults.push_back(newInstance.getResult(resIndex));

      ++oldResultCntr;
    }
    rewriter.replaceOp(op, convResults);
    convertedOps->insert(newInstance);
    return success();
  }

  DenseSet<hw::InstanceOp> *convertedOps;
};

using IOTypes = std::pair<TypeRange, TypeRange>;

struct IOInfo {
  // A mapping between an arg/res index and the struct type of the given field.
  DenseMap<unsigned, hw::StructType> argStructs, resStructs;

  // Records of the original arg/res types.
  SmallVector<Type> argTypes, resTypes;
};

class FlattenIOTypeConverter : public TypeConverter {
public:
  FlattenIOTypeConverter() {
    addConversion([](Type type, SmallVectorImpl<Type> &results) {
      auto structType = getStructType(type);
      if (!structType)
        results.push_back(type);
      else {
        for (auto field : structType.getElements())
          results.push_back(field.type);
      }
      return success();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::StructType type,
                                ValueRange inputs, Location loc) {
      auto result = builder.create<hw::StructCreateOp>(loc, type, inputs);
      return result.getResult();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::TypeAliasType type,
                                ValueRange inputs, Location loc) {
      auto structType = getStructType(type);
      assert(structType && "expected struct type");
      auto result = builder.create<hw::StructCreateOp>(loc, structType, inputs);
      return result.getResult();
    });
  }
};

} // namespace

template <typename... TOp>
static void addSignatureConversion(DenseMap<Operation *, IOInfo> &ioMap,
                                   ConversionTarget &target,
                                   RewritePatternSet &patterns,
                                   FlattenIOTypeConverter &typeConverter) {
  (hw::populateHWModuleLikeTypeConversionPattern(TOp::getOperationName(),
                                                 patterns, typeConverter),
   ...);

  // Legality is defined by a module having been processed once. This is due to
  // that a pattern cannot be applied multiple times (a 'pattern was already
  // applied' error - a case that would occur for nested structs). Additionally,
  // if a pattern could be applied multiple times, this would complicate
  // updating arg/res names.

  // Instead, we define legality as when a module has had a modification to its
  // top-level i/o. This ensures that only a single level of structs are
  // processed during signature conversion, which then allows us to use the
  // signature conversion in a recursive manner.
  target.addDynamicallyLegalOp<TOp...>([&](hw::HWModuleLike moduleLikeOp) {
    if (isLegalModLikeOp(moduleLikeOp))
      return true;

    // This op is involved in conversion. Check if the signature has changed.
    auto ioInfoIt = ioMap.find(moduleLikeOp);
    if (ioInfoIt == ioMap.end()) {
      // Op wasn't primed in the map. Do the safe thing, assume
      // that it's not considered in this pass, and mark it as legal
      return true;
    }
    auto ioInfo = ioInfoIt->second;

    auto compareTypes = [&](TypeRange oldTypes, TypeRange newTypes) {
      return llvm::any_of(llvm::zip(oldTypes, newTypes), [&](auto typePair) {
        auto oldType = std::get<0>(typePair);
        auto newType = std::get<1>(typePair);
        return oldType != newType;
      });
    };
    auto mtype = moduleLikeOp.getHWModuleType();
    if (compareTypes(mtype.getOutputTypes(), ioInfo.resTypes) ||
        compareTypes(mtype.getInputTypes(), ioInfo.argTypes))
      return true;

    // We're pre-conversion for an op that was primed in the map - it will
    // always be illegal since it has to-be-converted struct types at its I/O.
    return false;
  });
}

template <typename T>
static bool hasUnconvertedOps(mlir::ModuleOp module) {
  return llvm::any_of(module.getBody()->getOps<T>(),
                      [](T op) { return !isLegalModLikeOp(op); });
}

template <typename T>
static DenseMap<Operation *, IOTypes> populateIOMap(mlir::ModuleOp module) {
  DenseMap<Operation *, IOTypes> ioMap;
  for (auto op : module.getOps<T>())
    ioMap[op] = {op.getArgumentTypes(), op.getResultTypes()};
  return ioMap;
}

template <typename ModTy, typename T>
static llvm::SmallVector<Attribute>
updateNameAttribute(ModTy op, StringRef attrName,
                    DenseMap<unsigned, hw::StructType> &structMap, T oldNames) {
  llvm::SmallVector<Attribute> newNames;
  for (auto [i, oldName] : llvm::enumerate(oldNames)) {
    // Was this arg/res index a struct?
    auto it = structMap.find(i);
    if (it == structMap.end()) {
      // No, keep old name.
      newNames.push_back(StringAttr::get(op->getContext(), oldName));
      continue;
    }

    // Yes - create new names from the struct fields and the old name at the
    // index.
    auto structType = it->second;
    for (auto field : structType.getElements())
      newNames.push_back(
          StringAttr::get(op->getContext(), oldName + "." + field.name.str()));
  }
  return newNames;
}

static llvm::SmallVector<Attribute>
updateLocAttribute(DenseMap<unsigned, hw::StructType> &structMap,
                   ArrayAttr oldLocs) {
  llvm::SmallVector<Attribute> newLocs;
  if (!oldLocs)
    return newLocs;
  for (auto [i, oldLoc] : llvm::enumerate(oldLocs.getAsRange<Location>())) {
    // Was this arg/res index a struct?
    auto it = structMap.find(i);
    if (it == structMap.end()) {
      // No, keep old name.
      newLocs.push_back(oldLoc);
      continue;
    }

    auto structType = it->second;
    for (size_t i = 0, e = structType.getElements().size(); i < e; ++i)
      newLocs.push_back(oldLoc);
  }
  return newLocs;
}

/// The conversion framework seems to throw away block argument locations.  We
/// use this function to copy the location from the original argument to the
/// set of flattened arguments.
static void
updateBlockLocations(hw::HWModuleLike op,
                     DenseMap<unsigned, hw::StructType> &structMap) {
  auto locs = op.getInputLocs();
  if (locs.empty() || op.getModuleBody().empty())
    return;
  for (auto [arg, loc] : llvm::zip(op.getBodyBlock()->getArguments(), locs))
    arg.setLoc(loc);
}

template <typename T>
static DenseMap<Operation *, IOInfo> populateIOInfoMap(mlir::ModuleOp module) {
  DenseMap<Operation *, IOInfo> ioInfoMap;
  for (auto op : module.getOps<T>()) {
    IOInfo ioInfo;
    ioInfo.argTypes = op.getInputTypes();
    ioInfo.resTypes = op.getOutputTypes();
    for (auto [i, arg] : llvm::enumerate(ioInfo.argTypes)) {
      if (auto structType = getStructType(arg))
        ioInfo.argStructs[i] = structType;
    }
    for (auto [i, res] : llvm::enumerate(ioInfo.resTypes)) {
      if (auto structType = getStructType(res))
        ioInfo.resStructs[i] = structType;
    }
    ioInfoMap[op] = ioInfo;
  }
  return ioInfoMap;
}

template <typename T>
static LogicalResult flattenOpsOfType(ModuleOp module, bool recursive) {
  auto *ctx = module.getContext();
  FlattenIOTypeConverter typeConverter;

  // Recursively (in case of nested structs) lower the module. We do this one
  // conversion at a time to allow for updating the arg/res names of the
  // module in between flattening each level of structs.
  while (hasUnconvertedOps<T>(module)) {
    ConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);
    target.addLegalDialect<hw::HWDialect>();

    // Record any struct types at the module signature. This will be used
    // post-conversion to update the argument and result names.
    auto ioInfoMap = populateIOInfoMap<T>(module);

    // Record the instances that were converted. We keep these around since we
    // need to update their arg/res attribute names after the modules themselves
    // have been updated.
    llvm::DenseSet<hw::InstanceOp> convertedInstances;

    // Argument conversion for output ops. Similarly to the signature
    // conversion, legality is based on the op having been visited once, due to
    // the possibility of nested structs.
    DenseSet<Operation *> opVisited;
    patterns.add<OutputOpConversion>(typeConverter, ctx, &opVisited);

    patterns.add<InstanceOpConversion>(typeConverter, ctx, &convertedInstances);
    target.addDynamicallyLegalOp<hw::OutputOp>(
        [&](auto op) { return opVisited.contains(op->getParentOp()); });
    target.addDynamicallyLegalOp<hw::InstanceOp>([&](auto op) {
      return llvm::none_of(op->getOperands(), [](auto operand) {
        return isStructType(operand.getType());
      });
    });

    DenseMap<Operation *, ArrayAttr> oldArgNames, oldResNames, oldArgLocs,
        oldResLocs;
    for (auto op : module.getOps<T>()) {
      oldArgNames[op] = ArrayAttr::get(module.getContext(), op.getInputNames());
      oldResNames[op] =
          ArrayAttr::get(module.getContext(), op.getOutputNames());
      oldArgLocs[op] = op.getInputLocsAttr();
      oldResLocs[op] = op.getOutputLocsAttr();
    }

    // Signature conversion and legalization patterns.
    addSignatureConversion<T>(ioInfoMap, target, patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return failure();

    // Update the arg/res names of the module.
    for (auto op : module.getOps<T>()) {
      auto ioInfo = ioInfoMap[op];
      auto newArgNames = updateNameAttribute(
          op, "argNames", ioInfo.argStructs,
          oldArgNames[op].template getAsValueRange<StringAttr>());
      auto newResNames = updateNameAttribute(
          op, "resultNames", ioInfo.resStructs,
          oldResNames[op].template getAsValueRange<StringAttr>());
      newArgNames.append(newResNames.begin(), newResNames.end());
      op.setAllPortNames(newArgNames);
      auto newArgLocs = updateLocAttribute(ioInfo.argStructs, oldArgLocs[op]);
      auto newResLocs = updateLocAttribute(ioInfo.resStructs, oldResLocs[op]);
      newArgLocs.append(newResLocs.begin(), newResLocs.end());
      op.setPortLocsAttr(ArrayAttr::get(op.getContext(), newArgLocs));
      updateBlockLocations(op, ioInfo.argStructs);
    }

    // And likewise with the converted instance ops.
    for (auto instanceOp : convertedInstances) {
      Operation *targetModule = SymbolTable::lookupNearestSymbolFrom(
          instanceOp, instanceOp.getReferencedModuleNameAttr());

      auto ioInfo = ioInfoMap[targetModule];
      instanceOp.setInputNames(ArrayAttr::get(
          instanceOp.getContext(),
          updateNameAttribute(instanceOp, "argNames", ioInfo.argStructs,
                              oldArgNames[targetModule]
                                  .template getAsValueRange<StringAttr>())));
      instanceOp.setOutputNames(ArrayAttr::get(
          instanceOp.getContext(),
          updateNameAttribute(instanceOp, "resultNames", ioInfo.resStructs,
                              oldResNames[targetModule]
                                  .template getAsValueRange<StringAttr>())));
      instanceOp.dump();
    }

    // Break if we've only lowering a single level of structs.
    if (!recursive)
      break;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

template <typename... TOps>
static bool flattenIO(ModuleOp module, bool recursive) {
  return (failed(flattenOpsOfType<TOps>(module, recursive)) || ...);
}

namespace {

class FlattenIOPass : public circt::hw::FlattenIOBase<FlattenIOPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (flattenIO<hw::HWModuleOp, hw::HWModuleExternOp,
                  hw::HWModuleGeneratedOp>(module, recursive))
      signalPassFailure();
  };
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::hw::createFlattenIOPass() {
  return std::make_unique<FlattenIOPass>();
}
