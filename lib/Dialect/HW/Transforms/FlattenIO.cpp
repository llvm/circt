//===- FlattenIO.cpp - HW I/O flattening pass -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_FLATTENIO
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;

static bool isStructType(Type type) {
  return isa<hw::StructType>(hw::getCanonicalType(type));
}

static hw::StructType getStructType(Type type) {
  return dyn_cast<hw::StructType>(hw::getCanonicalType(type));
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

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const auto &vals : values)
    llvm::append_range(result, vals);
  return result;
}

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
    opVisited->insert(op->getParentOp());
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, convOperands);
    return success();
  }

  LogicalResult
  matchAndRewrite(hw::OutputOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> convOperands;

    // Flatten the operands.
    for (auto operand : flattenValues(adaptor.getOperands())) {
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
    opVisited->insert(op->getParentOp());
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, convOperands);
    return success();
  }
  DenseSet<Operation *> *opVisited;
};

struct InstanceOpConversion : public OpConversionPattern<hw::InstanceOp> {
  InstanceOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                       DenseSet<hw::InstanceOp> *convertedOps,
                       const StringSet<> *externModules)
      : OpConversionPattern(typeConverter, context), convertedOps(convertedOps),
        externModules(externModules) {}

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto referencedMod = op.getReferencedModuleNameAttr();
    // If externModules is populated and this is an extern module instance,
    // donot flatten it.
    if (externModules->contains(referencedMod.getValue()))
      return success();

    auto loc = op.getLoc();
    // Flatten the operands.
    llvm::SmallVector<Value> convOperands;
    for (auto operand : flattenValues(adaptor.getOperands())) {
      if (auto structType = getStructType(operand.getType())) {
        auto explodedStruct = rewriter.create<hw::StructExplodeOp>(
            loc, getInnerTypes(structType), operand);
        llvm::copy(explodedStruct.getResults(),
                   std::back_inserter(convOperands));
      } else {
        convOperands.push_back(operand);
      }
    }

    // Get the new module return type.
    llvm::SmallVector<Type> newResultTypes;
    for (auto oldResultType : op.getResultTypes()) {
      if (auto structType = getStructType(oldResultType))
        for (auto t : structType.getElements())
          newResultTypes.push_back(t.type);
      else
        newResultTypes.push_back(oldResultType);
    }

    // Create the new instance with the flattened module, attributes will be
    // adjusted later.
    auto newInstance = rewriter.create<hw::InstanceOp>(
        loc, newResultTypes, op.getInstanceNameAttr(),
        FlatSymbolRefAttr::get(referencedMod), convOperands,
        op.getArgNamesAttr(), op.getResultNamesAttr(), op.getParametersAttr(),
        op.getInnerSymAttr(), op.getDoNotPrintAttr());

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
  const StringSet<> *externModules;
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

    // Materialize !hw.struct<a,b,...> to a, b, ... via. hw.explode. This
    // situation may occur in case of hw.extern_module's with struct outputs.
    addTargetMaterialization([](OpBuilder &builder, TypeRange resultTypes,
                                ValueRange inputs, Location loc) {
      if (inputs.size() != 1 && !isStructType(inputs[0].getType()))
        return ValueRange();

      auto explodeOp = builder.create<hw::StructExplodeOp>(loc, inputs[0]);
      return ValueRange(explodeOp.getResults());
    });
    addTargetMaterialization([](OpBuilder &builder, hw::StructType type,
                                ValueRange inputs, Location loc) {
      auto result = builder.create<hw::StructCreateOp>(loc, type, inputs);
      return result.getResult();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::TypeAliasType type,
                                ValueRange inputs, Location loc) {
      auto result = builder.create<hw::StructCreateOp>(loc, type, inputs);
      return result.getResult();
    });

    // In the presence of hw.extern_module which takes struct arguments, we may
    // have materialized struct explodes for said arguments (say, e.g., if the
    // parent module of the hw.instance had structs in its input, and feeds
    // these structs to the hw.instance).
    // These struct explodes needs to be converted back to the original struct,
    // which persist beyond the conversion.
    addSourceMaterialization([](OpBuilder &builder, hw::StructType type,
                                ValueRange inputs, Location loc) {
      auto result = builder.create<hw::StructCreateOp>(loc, type, inputs);
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
                    DenseMap<unsigned, hw::StructType> &structMap, T oldNames,
                    char joinChar) {
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
      newNames.push_back(StringAttr::get(
          op->getContext(), oldName + Twine(joinChar) + field.name.str()));
  }
  return newNames;
}

template <typename ModTy>
static void updateModulePortNames(ModTy op, hw::ModuleType oldModType,
                                  char joinChar) {
  // Module arg and result port names may not be ordered. So we cannot reuse
  // updateNameAttribute. The arg and result order must be preserved.
  SmallVector<Attribute> newNames;
  SmallVector<hw::ModulePort> oldPorts(oldModType.getPorts().begin(),
                                       oldModType.getPorts().end());
  for (auto oldPort : oldPorts) {
    auto oldName = oldPort.name;
    if (auto structType = getStructType(oldPort.type)) {
      for (auto field : structType.getElements()) {
        newNames.push_back(StringAttr::get(
            op->getContext(),
            oldName.getValue() + Twine(joinChar) + field.name.str()));
      }
    } else
      newNames.push_back(oldName);
  }
  op.setAllPortNames(newNames);
}

static llvm::SmallVector<Location>
updateLocAttribute(DenseMap<unsigned, hw::StructType> &structMap,
                   SmallVectorImpl<Location> &oldLocs) {
  llvm::SmallVector<Location> newLocs;
  for (auto [i, oldLoc] : llvm::enumerate(oldLocs)) {
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

static void setIOInfo(hw::HWModuleLike op, IOInfo &ioInfo) {
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
}

template <typename T>
static DenseMap<Operation *, IOInfo> populateIOInfoMap(mlir::ModuleOp module) {
  DenseMap<Operation *, IOInfo> ioInfoMap;
  for (auto op : module.getOps<T>()) {
    IOInfo ioInfo;
    setIOInfo(op, ioInfo);
    ioInfoMap[op] = ioInfo;
  }
  return ioInfoMap;
}

template <typename T>
static LogicalResult flattenOpsOfType(ModuleOp module, bool recursive,
                                      StringSet<> &externModules,
                                      char joinChar) {
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

    patterns.add<InstanceOpConversion>(typeConverter, ctx, &convertedInstances,
                                       &externModules);
    target.addDynamicallyLegalOp<hw::OutputOp>(
        [&](auto op) { return opVisited.contains(op->getParentOp()); });
    target.addDynamicallyLegalOp<hw::InstanceOp>([&](hw::InstanceOp op) {
      auto refName = op.getReferencedModuleName();
      return externModules.contains(refName) ||
             (llvm::none_of(op->getOperands(),
                            [](auto operand) {
                              return isStructType(operand.getType());
                            }) &&
              llvm::none_of(op->getResultTypes(),
                            [](auto result) { return isStructType(result); }));
    });

    DenseMap<Operation *, ArrayAttr> oldArgNames, oldResNames;
    DenseMap<Operation *, SmallVector<Location>> oldArgLocs, oldResLocs;
    DenseMap<Operation *, hw::ModuleType> oldModTypes;

    for (auto op : module.getOps<T>()) {
      oldModTypes[op] = op.getHWModuleType();
      oldArgNames[op] = ArrayAttr::get(module.getContext(), op.getInputNames());
      oldResNames[op] =
          ArrayAttr::get(module.getContext(), op.getOutputNames());
      oldArgLocs[op] = op.getInputLocs();
      oldResLocs[op] = op.getOutputLocs();
    }

    // Signature conversion and legalization patterns.
    addSignatureConversion<T>(ioInfoMap, target, patterns, typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return failure();

    // Update the arg/res names of the module.
    for (auto op : module.getOps<T>()) {
      auto ioInfo = ioInfoMap[op];
      updateModulePortNames(op, oldModTypes[op], joinChar);
      auto newArgLocs = updateLocAttribute(ioInfo.argStructs, oldArgLocs[op]);
      auto newResLocs = updateLocAttribute(ioInfo.resStructs, oldResLocs[op]);
      newArgLocs.append(newResLocs.begin(), newResLocs.end());
      op.setAllPortLocs(newArgLocs);
      updateBlockLocations(op, ioInfo.argStructs);
    }

    // And likewise with the converted instance ops.
    for (auto instanceOp : convertedInstances) {
      auto targetModule =
          cast<hw::HWModuleLike>(SymbolTable::lookupNearestSymbolFrom(
              instanceOp, instanceOp.getReferencedModuleNameAttr()));

      IOInfo ioInfo;
      if (!ioInfoMap.contains(targetModule)) {
        // If an extern module, then not yet processed, populate the maps.
        setIOInfo(targetModule, ioInfo);
        ioInfoMap[targetModule] = ioInfo;
        oldArgNames[targetModule] =
            ArrayAttr::get(module.getContext(), targetModule.getInputNames());
        oldResNames[targetModule] =
            ArrayAttr::get(module.getContext(), targetModule.getOutputNames());
        oldArgLocs[targetModule] = targetModule.getInputLocs();
        oldResLocs[targetModule] = targetModule.getOutputLocs();
      } else
        ioInfo = ioInfoMap[targetModule];

      instanceOp.setInputNames(ArrayAttr::get(
          instanceOp.getContext(),
          updateNameAttribute(
              instanceOp, "argNames", ioInfo.argStructs,
              oldArgNames[targetModule].template getAsValueRange<StringAttr>(),
              joinChar)));
      instanceOp.setOutputNames(ArrayAttr::get(
          instanceOp.getContext(),
          updateNameAttribute(
              instanceOp, "resultNames", ioInfo.resStructs,
              oldResNames[targetModule].template getAsValueRange<StringAttr>(),
              joinChar)));
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
static bool flattenIO(ModuleOp module, bool recursive,
                      StringSet<> &externModules, char joinChar) {
  return (failed(flattenOpsOfType<TOps>(module, recursive, externModules,
                                        joinChar)) ||
          ...);
}

namespace {

class FlattenIOPass : public circt::hw::impl::FlattenIOBase<FlattenIOPass> {
public:
  FlattenIOPass(bool recursiveFlag, bool flattenExternFlag, char join) {
    recursive = recursiveFlag;
    flattenExtern = flattenExternFlag;
    joinChar = join;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!flattenExtern) {
      // Record the extern modules, do not flatten them.
      for (auto m : module.getOps<hw::HWModuleExternOp>())
        externModules.insert(m.getModuleName());
      if (flattenIO<hw::HWModuleOp, hw::HWModuleGeneratedOp>(
              module, recursive, externModules, joinChar))
        signalPassFailure();
      return;
    }

    if (flattenIO<hw::HWModuleOp, hw::HWModuleExternOp,
                  hw::HWModuleGeneratedOp>(module, recursive, externModules,
                                           joinChar))
      signalPassFailure();
  };

private:
  StringSet<> externModules;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::hw::createFlattenIOPass(bool recursiveFlag,
                                                     bool flattenExternFlag,
                                                     char joinChar) {
  return std::make_unique<FlattenIOPass>(recursiveFlag, flattenExternFlag,
                                         joinChar);
}
