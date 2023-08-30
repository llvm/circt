//===- LowerClasses.cpp - Lower to OM classes and objects -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerClasses pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/OM/OMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;
using namespace circt::firrtl;
using namespace circt::om;

namespace {

/// Helper class to capture details about a property.
struct Property {
  size_t index;
  StringRef name;
  Type type;
  Location loc;
};

/// Helper class to capture state about a Class being lowered.
struct ClassLoweringState {
  FModuleLike moduleLike;
  om::ClassOp classOp;
};

struct LowerClassesPass : public LowerClassesBase<LowerClassesPass> {
  void runOnOperation() override;

private:
  // Predicate to check if a module-like needs a Class to be created.
  bool shouldCreateClass(FModuleLike moduleLike);

  // Create an OM Class op from a FIRRTL Class op.
  ClassLoweringState createClass(FModuleLike moduleLike);

  // Lower the FIRRTL Class to OM Class.
  void lowerClass(ClassLoweringState &state);

  // Update Object instantiations in a FIRRTL Module or OM Class.
  void updateInstances(Operation *op);

  // Convert to OM ops and types in Classes or Modules.
  LogicalResult dialectConversion(Operation *op);
};

} // namespace

/// Lower FIRRTL Class and Object ops to OM Class and Object ops
void LowerClassesPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  // Get the CircuitOp.
  CircuitOp circuit = getOperation();

  // Create new OM Class ops serially.
  SmallVector<ClassLoweringState> loweringState;
  for (auto moduleLike : circuit.getOps<FModuleLike>())
    if (shouldCreateClass(moduleLike))
      loweringState.push_back(createClass(moduleLike));

  // Move ops from FIRRTL Class to OM Class in parallel.
  mlir::parallelForEach(ctx, loweringState,
                        [this](auto state) { lowerClass(state); });

  // Completely erase Class module-likes
  for (auto state : loweringState) {
    if (isa<firrtl::ClassOp>(state.moduleLike))
      state.moduleLike.erase();
  }

  // Collect ops where Objects can be instantiated.
  SmallVector<Operation *> objectContainers;
  for (auto &op : circuit.getOps())
    if (isa<FModuleOp, om::ClassOp>(op))
      objectContainers.push_back(&op);

  // Update Object creation ops in Classes or Modules in parallel.
  mlir::parallelForEach(ctx, objectContainers,
                        [this](auto *op) { updateInstances(op); });

  // Convert to OM ops and types in Classes or Modules in parallel.
  if (failed(mlir::failableParallelForEach(
          ctx, objectContainers,
          [this](auto *op) { return dialectConversion(op); })))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerClassesPass() {
  return std::make_unique<LowerClassesPass>();
}

// Predicate to check if a module-like needs a Class to be created.
bool LowerClassesPass::shouldCreateClass(FModuleLike moduleLike) {
  if (isa<firrtl::ClassOp>(moduleLike))
    return true;

  return llvm::any_of(moduleLike.getPorts(), [](PortInfo port) {
    return isa<PropertyType>(port.type);
  });
}

// Create an OM Class op from a FIRRTL Class op or Module op with properties.
ClassLoweringState LowerClassesPass::createClass(FModuleLike moduleLike) {
  // Collect the parameter names from input properties.
  SmallVector<StringRef> formalParamNames;
  for (auto [index, port] : llvm::enumerate(moduleLike.getPorts()))
    if (port.isInput() && isa<PropertyType>(port.type))
      formalParamNames.push_back(port.name);

  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBodyBlock());

  // Take the name from the FIRRTL Class or Module to create the OM Class name.
  StringRef className = moduleLike.getName();

  // If the op is a Module, the OM Class would conflict with the HW Module, so
  // give it a suffix. There is no formal ABI for this yet.
  StringRef suffix = isa<FModuleOp>(moduleLike) ? "_Class" : "";

  // Construct the OM Class with the FIRRTL Class name and parameter names.
  auto loweredClassOp = builder.create<om::ClassOp>(
      moduleLike.getLoc(), className + suffix, formalParamNames);

  return {moduleLike, loweredClassOp};
}

// Lower the FIRRTL Class to OM Class.
void LowerClassesPass::lowerClass(ClassLoweringState &state) {
  // Extract values from state.
  firrtl::FModuleLike moduleLike = state.moduleLike;
  om::ClassOp classOp = state.classOp;

  // Map from Values in the FIRRTL Class to Values in the OM Class.
  IRMapping mapping;

  // Collect information about property ports.
  SmallVector<Property> inputProperties;
  BitVector portsToErase(moduleLike.getNumPorts());
  for (auto [index, port] : llvm::enumerate(moduleLike.getPorts())) {
    // For Module ports that aren't property types, move along.
    if (!isa<PropertyType>(port.type))
      continue;

    // Remember input properties to create the OM Class formal parameters.
    if (port.isInput())
      inputProperties.push_back({index, port.name, port.type, port.loc});

    // In case this is a Module, remember to erase this port.
    portsToErase.set(index);
  }

  // Construct the OM Class body with block arguments for each input property,
  // updating the mapping to map from the input property to the block argument.
  Block *classBody = &classOp.getRegion().emplaceBlock();
  for (auto inputProperty : inputProperties) {
    BlockArgument parameterValue =
        classBody->addArgument(inputProperty.type, inputProperty.loc);
    BlockArgument inputValue =
        moduleLike->getRegion(0).getArgument(inputProperty.index);
    mapping.map(inputValue, parameterValue);
  }

  // Clone the property ops from the FIRRTL Class or Module to the OM Class.
  SmallVector<Operation *> opsToErase;
  OpBuilder builder = OpBuilder::atBlockBegin(classOp.getBodyBlock());
  for (auto &op : moduleLike->getRegion(0).getOps()) {
    // Check if any operand is a property.
    auto propertyOperands = llvm::any_of(op.getOperandTypes(), [](Type type) {
      return isa<PropertyType>(type);
    });

    // Check if any result is a property.
    auto propertyResults = llvm::any_of(
        op.getResultTypes(), [](Type type) { return isa<PropertyType>(type); });

    // If there are no properties here, move along.
    if (!propertyOperands && !propertyResults)
      continue;

    // Actually clone the op over to the OM Class.
    builder.clone(op, mapping);

    // In case this is a Module, remember to erase this op.
    opsToErase.push_back(&op);
  }

  // Convert any output property assignments to Field ops.
  for (auto op : llvm::make_early_inc_range(classOp.getOps<PropAssignOp>())) {
    // Property assignments will currently be pointing back to the original
    // FIRRTL Class for output ports.
    auto outputPort = dyn_cast<BlockArgument>(op.getDest());
    if (!outputPort)
      continue;

    // Get the original port name, create a Field, and erase the propassign.
    auto name = moduleLike.getPortName(outputPort.getArgNumber());
    builder.create<ClassFieldOp>(op.getLoc(), name, op.getSrc());
    op.erase();
  }

  // If the module-like is a Class, it will be completely erased later.
  // Otherwise, erase just the property ports and ops.
  if (!isa<firrtl::ClassOp>(moduleLike)) {
    // Erase ops in use before def order, thanks to FIRRTL's SSA regions.
    for (auto *op : llvm::reverse(opsToErase))
      op->erase();

    // Erase property typed ports.
    moduleLike.erasePorts(portsToErase);
  }
}

// Update Object instantiations in a FIRRTL Module or OM Class.
void LowerClassesPass::updateInstances(Operation *op) {
  OpBuilder builder(op);
  // For each Object instance.
  for (auto firrtlObject : llvm::make_early_inc_range(
           op->getRegion(0).getOps<firrtl::ObjectOp>())) {
    // Collect its input actual parameters by finding any subfield ops that are
    // assigned to. Take the source of the assignment as the actual parameter.
    SmallVector<Value> actualParameters;
    for (auto *user : firrtlObject->getUsers())
      if (auto subfield = dyn_cast<ObjectSubfieldOp>(user))
        for (auto *subfieldUser : subfield->getUsers())
          if (auto propassign = dyn_cast<PropAssignOp>(subfieldUser))
            if (propassign.getDest() == subfield.getResult())
              actualParameters.push_back(propassign.getSrc());

    // Convert the FIRRTL Class type to an OM Class type.
    auto classType =
        om::ClassType::get(op->getContext(), firrtlObject.getClassNameAttr());

    // Create the new Object op.
    builder.setInsertionPoint(firrtlObject);
    auto object = builder.create<om::ObjectOp>(
        firrtlObject.getLoc(), classType,
        firrtlObject.getClassNameAttr().getAttr(), actualParameters);

    // Replace uses of the FIRRTL Object with the OM Object. The later dialect
    // conversion will take care of converting the types.
    firrtlObject.replaceAllUsesWith(object.getResult());

    // Erase the original Object, now that we're done with it.
    firrtlObject.erase();
  }
}

// Pattern rewriters for dialect conversion.

struct FIntegerConstantOpConversion
    : public OpConversionPattern<FIntegerConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FIntegerConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::ConstantOp>(
        op, adaptor.getValueAttr().getType(), adaptor.getValueAttr());
    return success();
  }
};

struct BoolConstantOpConversion : public OpConversionPattern<BoolConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BoolConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::ConstantOp>(
        op, rewriter.getBoolAttr(adaptor.getValue()));
    return success();
  }
};

struct StringConstantOpConversion
    : public OpConversionPattern<StringConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringType = om::StringType::get(op.getContext());
    rewriter.replaceOpWithNewOp<om::ConstantOp>(
        op, stringType, StringAttr::get(op.getValue(), stringType));
    return success();
  }
};

struct ListCreateOpConversion
    : public OpConversionPattern<firrtl::ListCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::ListCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto listType = getTypeConverter()->convertType<om::ListType>(op.getType());
    if (!listType)
      return failure();
    rewriter.replaceOpWithNewOp<om::ListCreateOp>(op, listType,
                                                  adaptor.getElements());
    return success();
  }
};

struct MapCreateOpConversion : public OpConversionPattern<firrtl::MapCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(firrtl::MapCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mapType = getTypeConverter()->convertType<om::MapType>(op.getType());
    if (!mapType)
      return failure();
    auto keys = adaptor.getKeys();
    auto values = adaptor.getValues();
    SmallVector<Value> tuples;
    for (auto [key, value] : llvm::zip(keys, values))
      tuples.push_back(rewriter.create<om::TupleCreateOp>(
          op.getLoc(), ArrayRef<Value>{key, value}));
    rewriter.replaceOpWithNewOp<om::MapCreateOp>(op, mapType, tuples);
    return success();
  }
};

struct ClassFieldOpConversion : public OpConversionPattern<ClassFieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassFieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ClassFieldOp>(op, adaptor.getSymNameAttr(),
                                              adaptor.getValue());
    return success();
  }
};

struct ClassOpSignatureConversion : public OpConversionPattern<om::ClassOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(om::ClassOp classOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *body = classOp.getBodyBlock();
    TypeConverter::SignatureConversion result(body->getNumArguments());

    // Convert block argument types.
    if (failed(typeConverter->convertSignatureArgs(body->getArgumentTypes(),
                                                   result)))
      return failure();

    // Convert the body.
    if (failed(rewriter.convertRegionTypes(body->getParent(), *typeConverter,
                                           &result)))
      return failure();

    rewriter.updateRootInPlace(classOp, []() {});

    return success();
  }
};

// Helpers for dialect conversion setup.

static void populateConversionTarget(ConversionTarget &target) {
  // FIRRTL dialect operations inside ClassOps or not using only OM types must
  // be legalized.
  target.addDynamicallyLegalDialect<FIRRTLDialect>(
      [](Operation *op) { return !op->getParentOfType<om::ClassOp>(); });

  // OM dialect operations are legal if they don't use FIRRTL types.
  target.addDynamicallyLegalDialect<OMDialect>([](Operation *op) {
    auto containsFIRRTLType = [](Type type) {
      return type
          .walk([](Type type) {
            return failure(isa<FIRRTLDialect>(type.getDialect()));
          })
          .wasInterrupted();
    };
    auto noFIRRTLOperands =
        llvm::none_of(op->getOperandTypes(), [&containsFIRRTLType](Type type) {
          return containsFIRRTLType(type);
        });
    auto noFIRRTLResults =
        llvm::none_of(op->getResultTypes(), [&containsFIRRTLType](Type type) {
          return containsFIRRTLType(type);
        });
    return noFIRRTLOperands && noFIRRTLResults;
  });

  // OM Class ops are legal if they don't use FIRRTL types for block arguments.
  target.addDynamicallyLegalOp<om::ClassOp>([](om::ClassOp op) {
    return llvm::none_of(op.getBodyBlock()->getArgumentTypes(), [](Type type) {
      return isa<FIRRTLDialect>(type.getDialect());
    });
  });
}

static void populateTypeConverter(TypeConverter &converter) {
  // Convert FIntegerType to IntegerType.
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](FIntegerType type) {
    // The actual width of the IntegerType doesn't actually get used; it will be
    // folded away by the dialect conversion infrastructure to the type of the
    // APSIntAttr used in the FIntegerConstantOp.
    return IntegerType::get(type.getContext(), 64);
  });

  // Convert FIRRTL StringType to OM StringType.
  converter.addConversion([](om::StringType type) { return type; });
  converter.addConversion([](firrtl::StringType type) {
    return om::StringType::get(type.getContext());
  });

  // Convert FIRRTL Class type to OM Class type.
  converter.addConversion([](om::ClassType type) { return type; });
  converter.addConversion([](firrtl::ClassType type) {
    return om::ClassType::get(type.getContext(), type.getNameAttr());
  });

  // Convert FIRRTL List type to OM List type.
  converter.addConversion(
      [&converter](om::ListType type) -> std::optional<mlir::Type> {
        // Convert any om.list<firrtl> -> om.list<om>
        auto elementType = converter.convertType(type.getElementType());
        if (!elementType)
          return {};
        return om::ListType::get(elementType);
      });

  converter.addConversion(
      [&converter](firrtl::ListType type) -> std::optional<mlir::Type> {
        // Convert any firrtl.list<firrtl> -> om.list<om>
        auto elementType = converter.convertType(type.getElementType());
        if (!elementType)
          return {};
        return om::ListType::get(elementType);
      });

  auto convertMapType = [&converter](auto type) -> std::optional<mlir::Type> {
    auto keyType = converter.convertType(type.getKeyType());
    // Reject key types that are not string or integer.
    if (!isa_and_nonnull<om::StringType, mlir::IntegerType>(keyType))
      return {};

    auto valueType = converter.convertType(type.getValueType());
    if (!valueType)
      return {};

    return om::MapType::get(keyType, valueType);
  };

  // Convert FIRRTL Map type to OM Map type.
  converter.addConversion(
      [convertMapType](om::MapType type) -> std::optional<mlir::Type> {
        // Convert any om.map<firrtl, firrtl> -> om.map<om, om>
        return convertMapType(type);
      });

  converter.addConversion(
      [convertMapType](firrtl::MapType type) -> std::optional<mlir::Type> {
        // Convert any firrtl.map<firrtl, firrtl> -> om.map<om, om>
        return convertMapType(type);
      });

  // Convert FIRRTL Bool type to OM
  converter.addConversion(
      [](BoolType type) { return IntegerType::get(type.getContext(), 1); });

  // Add a target materialization to fold away unrealized conversion casts.
  converter.addTargetMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        return values[0];
      });
}

static void populateRewritePatterns(RewritePatternSet &patterns,
                                    TypeConverter &converter) {
  patterns.add<FIntegerConstantOpConversion>(converter, patterns.getContext());
  patterns.add<StringConstantOpConversion>(converter, patterns.getContext());
  patterns.add<ClassFieldOpConversion>(converter, patterns.getContext());
  patterns.add<ClassOpSignatureConversion>(converter, patterns.getContext());
  patterns.add<ListCreateOpConversion>(converter, patterns.getContext());
  patterns.add<MapCreateOpConversion>(converter, patterns.getContext());
  patterns.add<BoolConstantOpConversion>(converter, patterns.getContext());
}

// Convert to OM ops and types in Classes or Modules.
LogicalResult LowerClassesPass::dialectConversion(Operation *op) {
  ConversionTarget target(getContext());
  populateConversionTarget(target);

  TypeConverter typeConverter;
  populateTypeConverter(typeConverter);

  RewritePatternSet patterns(&getContext());
  populateRewritePatterns(patterns, typeConverter);

  return applyPartialConversion(op, target, std::move(patterns));
}
