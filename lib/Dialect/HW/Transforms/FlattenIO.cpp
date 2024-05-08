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
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

// Legal if no in- or output type is a struct or array.
static bool isLegalModLikeOp(hw::HWModuleLike moduleLikeOp) {
  return llvm::none_of(moduleLikeOp.getHWModuleType().getPortTypes(),
                       [](auto op) {
                         return hw::type_isa<hw::StructType>(op) ||
                                hw::type_isa<hw::ArrayType>(op);
                       });
}

static llvm::SmallVector<Type> getInnerTypes(hw::StructType t) {
  llvm::SmallVector<Type> inner;
  t.getInnerTypes(inner);
  for (auto [index, innerType] : llvm::enumerate(inner))
    inner[index] = hw::getCanonicalType(innerType);
  return inner;
}

template <typename T>
static llvm::SmallVector<Value>
flattenOperands(T op, typename T::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<Value> convOperands;
  for (auto operand : adaptor.getOperands()) {
    llvm::TypeSwitch<Type>(hw::getCanonicalType(operand.getType()))
        .template Case<hw::StructType>([&](auto st) {
          auto explodedStruct = rewriter.create<hw::StructExplodeOp>(
              op.getLoc(), getInnerTypes(st), operand);
          llvm::copy(explodedStruct.getResults(),
                     std::back_inserter(convOperands));
        })
        .template Case<hw::ArrayType>([&](auto arr) {
          for (auto idx : llvm::seq(arr.getNumElements())) {
            IntegerType idxType = rewriter.getIntegerType(
                llvm::Log2_64_Ceil(arr.getNumElements()));
            Value idxVal =
                rewriter.create<hw::ConstantOp>(op.getLoc(), idxType, idx);
            auto element =
                rewriter.create<hw::ArrayGetOp>(op.getLoc(), operand, idxVal);
            convOperands.push_back(element);
          }
        })
        .Default([&](auto type) { convOperands.push_back(operand); });
  }
  return convOperands;
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
    llvm::SmallVector<Value> convOperands =
        flattenOperands(op, adaptor, rewriter);

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
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto referencedMod = op.getReferencedModuleNameAttr();
    // If externModules is populated and this is an extern module instance,
    // donot flatten it.
    if (externModules->contains(referencedMod.getValue()))
      return success();

    llvm::SmallVector<Value> convOperands =
        flattenOperands(op, adaptor, rewriter);

    // Get the new module return type.
    llvm::SmallVector<Type> newResultTypes;
    for (auto oldResultType : op.getResultTypes()) {
      llvm::TypeSwitch<Type>(hw::getCanonicalType(oldResultType))
          .Case<hw::StructType>([&](auto st) {
            llvm::transform(st.getElements(),
                            std::back_inserter(newResultTypes),
                            [](auto s) { return s.type; });
          })
          .Case<hw::ArrayType>([&](auto arr) {
            newResultTypes.append(arr.getNumElements(), arr.getElementType());
          })
          .Default([&](auto type) { newResultTypes.push_back(oldResultType); });
    }

    // Create the new instance with the flattened module, attributes will be
    // adjusted later.
    auto newInstance = rewriter.create<hw::InstanceOp>(
        op.getLoc(), newResultTypes, op.getInstanceNameAttr(),
        FlatSymbolRefAttr::get(referencedMod), convOperands,
        op.getArgNamesAttr(), op.getResultNamesAttr(), op.getParametersAttr(),
        op.getInnerSymAttr());

    // re-create any structs in the result.
    llvm::SmallVector<Value> convResults;
    size_t resIndex = 0;
    for (auto oldResultType : op.getResultTypes()) {
      llvm::TypeSwitch<Type>(hw::getCanonicalType(oldResultType))
          .Case<hw::StructType>([&](auto st) {
            size_t nElements = st.getElements().size();
            auto implodedStruct = rewriter.create<hw::StructCreateOp>(
                op.getLoc(), st,
                newInstance.getResults().slice(resIndex, nElements));
            convResults.push_back(implodedStruct.getResult());
            resIndex += nElements;
          })
          .Case<hw::ArrayType>([&](auto arr) {
            size_t nElements = arr.getNumElements();
            auto implodedArray = rewriter.create<hw::ArrayCreateOp>(
                op.getLoc(), arr,
                newInstance.getResults().slice(resIndex, nElements));
            convResults.push_back(implodedArray.getResult());
            resIndex += nElements;
          })
          .Default([&](auto type) {
            convResults.push_back(newInstance.getResult(resIndex));
            resIndex += 1;
          });
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
  SmallVector<Type> argTypes, resTypes;
};

class FlattenIOTypeConverter : public TypeConverter {
public:
  FlattenIOTypeConverter() {
    addConversion([](Type type, SmallVectorImpl<Type> &results) {
      llvm::TypeSwitch<Type>(hw::getCanonicalType(type))
          .Case<hw::StructType>([&](auto st) {
            llvm::transform(st.getElements(), std::back_inserter(results),
                            [](auto s) { return s.type; });
          })
          .Case<hw::ArrayType>([&](auto arr) {
            results.append(arr.getNumElements(), arr.getElementType());
          })
          .Default([&](auto type) { results.push_back(type); });

      return success();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::ArrayType type,
                                ValueRange inputs, Location loc) {
      auto result = builder.create<hw::ArrayCreateOp>(loc, type, inputs);
      return result.getResult();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::StructType type,
                                ValueRange inputs, Location loc) {
      auto result = builder.create<hw::StructCreateOp>(loc, type, inputs);
      return result.getResult();
    });

    addTargetMaterialization([](OpBuilder &builder, hw::TypeAliasType type,
                                ValueRange inputs, Location loc) {
      auto structType = hw::type_dyn_cast<hw::StructType>(type);
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
updateNameAttribute(ModTy op, llvm::SmallVectorImpl<Type> &oldTypes, T oldNames,
                    char joinChar) {
  llvm::SmallVector<Attribute> newNames;

  for (auto [oldType, oldName] : llvm::zip(oldTypes, oldNames)) {
    TypeSwitch<Type>(hw::getCanonicalType(oldType))
        .template Case<hw::StructType>([&](auto st) {
          for (auto field : st.getElements())
            newNames.push_back(
                StringAttr::get(op->getContext(),
                                oldName + Twine(joinChar) + field.name.str()));
        })
        .template Case<hw::ArrayType>([&](auto arr) {
          for (auto idx : llvm::seq(arr.getNumElements()))
            newNames.push_back(
                StringAttr::get(op->getContext(), oldName + Twine(joinChar) +
                                                      std::to_string(idx)));
        })
        .Default([&](auto type) {
          newNames.push_back(StringAttr::get(op->getContext(), oldName));
        });
  }
  return newNames;
}

template <typename ModTy>
static void updateModulePortNames(ModTy op, hw::ModuleType oldModType,
                                  char joinChar) {
  // Module arg and result port names may not be ordered. So we cannot reuse
  // updateNameAttribute. The arg and result order must be preserved.
  SmallVector<StringRef> oldNames;
  SmallVector<Type> oldTypes;
  for (auto port : oldModType.getPorts()) {
    oldNames.push_back(port.name);
    oldTypes.push_back(port.type);
  }
  SmallVector<Attribute> newNames =
      updateNameAttribute(op, oldTypes, oldNames, joinChar);
  op.setAllPortNames(newNames);
}

static llvm::SmallVector<Location>
updateLocAttribute(SmallVectorImpl<Type> &oldTypes,
                   SmallVectorImpl<Location> &oldLocs) {
  llvm::SmallVector<Location> newLocs;
  for (auto [oldType, oldLoc] : llvm::zip(oldTypes, oldLocs)) {
    llvm::TypeSwitch<Type>(hw::getCanonicalType(oldType))
        .Case<hw::StructType>(
            [&](auto st) { newLocs.append(st.getElements().size(), oldLoc); })
        .Case<hw::ArrayType>(
            [&](auto arr) { newLocs.append(arr.getNumElements(), oldLoc); })
        .Default([&](auto type) { newLocs.push_back(oldLoc); });
  }
  return newLocs;
}

/// The conversion framework seems to throw away block argument locations.  We
/// use this function to copy the location from the original argument to the
/// set of flattened arguments.
static void updateBlockLocations(hw::HWModuleLike op) {
  auto locs = op.getInputLocs();
  if (locs.empty() || op.getModuleBody().empty())
    return;
  for (auto [arg, loc] : llvm::zip(op.getBodyBlock()->getArguments(), locs))
    arg.setLoc(loc);
}

static void setIOInfo(hw::HWModuleLike op, IOInfo &ioInfo) {
  ioInfo.argTypes = op.getInputTypes();
  ioInfo.resTypes = op.getOutputTypes();
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
             llvm::none_of(op->getOperands(), [](auto operand) {
               return hw::type_isa<hw::StructType>(operand.getType()) ||
                      hw::type_isa<hw::ArrayType>(operand.getType());
             });
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
      auto newArgLocs = updateLocAttribute(ioInfo.argTypes, oldArgLocs[op]);
      auto newResLocs = updateLocAttribute(ioInfo.resTypes, oldResLocs[op]);
      newArgLocs.append(newResLocs.begin(), newResLocs.end());
      op.setAllPortLocs(newArgLocs);
      updateBlockLocations(op);
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
              instanceOp, ioInfo.argTypes,
              oldArgNames[targetModule].template getAsValueRange<StringAttr>(),
              joinChar)));
      instanceOp.setOutputNames(ArrayAttr::get(
          instanceOp.getContext(),
          updateNameAttribute(
              instanceOp, ioInfo.resTypes,
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

class FlattenIOPass : public circt::hw::FlattenIOBase<FlattenIOPass> {
public:
  FlattenIOPass(bool recursiveFlag, bool flattenExternFlag, char join) {
    recursive = recursiveFlag;
    flattenExtern = flattenExternFlag;
    joinChar = join;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!flattenExtern) {
      // Record the extern modules, donot flatten them.
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
