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
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/OM/OMAttributes.h"
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

/// Helper class which holds a hierarchical path op reference and a pointer to
/// to the targeted operation.
struct PathInfo {
  PathInfo() = default;
  PathInfo(Operation *op, FlatSymbolRefAttr symRef) : op(op), symRef(symRef) {
    assert(op && "op must not be null");
    assert(symRef && "symRef must not be null");
  }

  operator bool() const { return op != nullptr; }

  Operation *op = nullptr;
  FlatSymbolRefAttr symRef = nullptr;
};

/// Maps a FIRRTL path id to the lowered PathInfo.
using PathInfoTable = DenseMap<DistinctAttr, PathInfo>;

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
  om::ClassLike classLike;
};

struct LowerClassesPass : public LowerClassesBase<LowerClassesPass> {
  void runOnOperation() override;

private:
  LogicalResult lowerPaths(PathInfoTable &pathInfoTable);

  // Predicate to check if a module-like needs a Class to be created.
  bool shouldCreateClass(FModuleLike moduleLike);

  // Create an OM Class op from a FIRRTL Class op.
  ClassLoweringState createClass(FModuleLike moduleLike);

  // Lower the FIRRTL Class to OM Class.
  void lowerClassLike(ClassLoweringState state);
  void lowerClass(om::ClassOp classOp, FModuleLike moduleLike);
  void lowerClassExtern(ClassExternOp classExternOp, FModuleLike moduleLike);

  // Update Object instantiations in a FIRRTL Module or OM Class.
  void updateInstances(Operation *op);

  // Convert to OM ops and types in Classes or Modules.
  LogicalResult dialectConversion(Operation *op,
                                  const PathInfoTable &pathInfoTable);
};

} // namespace

/// This pass removes the OMIR tracker annotations from operations, and ensures
/// that each thing that was targeted has a hierarchical path targeting it. It
/// builds a table which maps the OMIR tracker annotation IDs to the
/// corresponding hierarchical paths. We use this table to convert FIRRTL path
/// ops to OM. FIRRTL paths refer to their target using a target ID, while OM
/// paths refer to their target using hierarchical paths.
LogicalResult LowerClassesPass::lowerPaths(PathInfoTable &pathInfoTable) {
  auto *context = &getContext();
  auto circuit = getOperation();
  auto &symbolTable = getAnalysis<SymbolTable>();
  hw::InnerSymbolNamespaceCollection namespaces;
  HierPathCache cache(circuit, symbolTable);

  auto processPathTrackers = [&](AnnoTarget target) -> LogicalResult {
    auto error = false;
    auto annotations = target.getAnnotations();
    auto *op = target.getOp();
    FModuleLike module;
    annotations.removeAnnotations([&](Annotation anno) {
      // If there has been an error, just skip this annotation.
      if (error)
        return false;

      // We are looking for OMIR tracker annotations.
      if (!anno.isClass("circt.tracker"))
        return false;

      // The token must have a valid ID.
      auto id = anno.getMember<DistinctAttr>("id");
      if (!id) {
        op->emitError("circt.tracker annotation missing id field");
        error = true;
        return false;
      }

      // Get the fieldID.  If there is none, it is assumed to be 0.
      uint64_t fieldID = anno.getFieldID();

      // Attach an inner sym to the operation.
      Attribute targetSym;
      if (auto portTarget = target.dyn_cast<PortAnnoTarget>()) {
        targetSym = getInnerRefTo(
            {portTarget.getPortNo(), portTarget.getOp(), fieldID},
            [&](FModuleLike module) -> hw::InnerSymbolNamespace & {
              return namespaces[module];
            });
      } else if (auto module = dyn_cast<FModuleLike>(op)) {
        assert(!fieldID && "field not valid for modules");
        targetSym = FlatSymbolRefAttr::get(module.getModuleNameAttr());
      } else {
        targetSym = getInnerRefTo(
            {target.getOp(), fieldID},
            [&](FModuleLike module) -> hw::InnerSymbolNamespace & {
              return namespaces[module];
            });
      }

      // Create the hierarchical path.
      SmallVector<Attribute> path;
      if (auto hierName = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        auto hierPathOp =
            dyn_cast<hw::HierPathOp>(symbolTable.lookup(hierName.getAttr()));
        if (!hierPathOp) {
          op->emitError("annotation does not point at a HierPathOp");
          error = true;
          return false;
        }
        // Copy the old path, dropping the module name.
        auto oldPath = hierPathOp.getNamepath().getValue();
        llvm::append_range(path, oldPath.drop_back());
      }
      path.push_back(targetSym);

      // Create the HierPathOp.
      auto pathAttr = ArrayAttr::get(context, path);
      auto &pathInfo = pathInfoTable[id];
      if (pathInfo) {
        auto diag =
            emitError(pathInfo.op->getLoc(), "duplicate identifier found");
        diag.attachNote(op->getLoc()) << "other identifier here";
        error = true;
        return false;
      }

      // Record the path operation associated with the path op.
      pathInfo = {op, cache.getRefFor(pathAttr)};

      // Remove this annotation from the operation.
      return true;
    });

    if (error)
      return failure();
    target.setAnnotations(annotations);
    return success();
  };

  for (auto module : circuit.getOps<FModuleLike>()) {
    // Process the module annotations.
    if (failed(processPathTrackers(OpAnnoTarget(module))))
      return failure();
    // Process module port annotations.
    for (unsigned i = 0, e = module.getNumPorts(); i < e; ++i)
      if (failed(processPathTrackers(PortAnnoTarget(module, i))))
        return failure();
    // Process ops in the module body.
    auto result = module.walk([&](hw::InnerSymbolOpInterface op) {
      if (failed(processPathTrackers(OpAnnoTarget(op))))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  }
  return success();
}

/// Lower FIRRTL Class and Object ops to OM Class and Object ops
void LowerClassesPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  // Get the CircuitOp.
  CircuitOp circuit = getOperation();

  // Rewrite all path annotations into inner symbol targets.
  PathInfoTable pathInfoTable;
  if (failed(lowerPaths(pathInfoTable))) {
    signalPassFailure();
    return;
  }

  // Create new OM Class ops serially.
  SmallVector<ClassLoweringState> loweringState;
  for (auto moduleLike : circuit.getOps<FModuleLike>())
    if (shouldCreateClass(moduleLike))
      loweringState.push_back(createClass(moduleLike));

  // Move ops from FIRRTL Class to OM Class in parallel.
  mlir::parallelForEach(ctx, loweringState,
                        [this](auto state) { lowerClassLike(state); });

  // Completely erase Class module-likes
  for (auto state : loweringState) {
    if (isa<firrtl::ClassLike>(state.moduleLike.getOperation()))
      state.moduleLike.erase();
  }

  // Collect ops where Objects can be instantiated.
  SmallVector<Operation *> objectContainers;
  for (auto &op : circuit.getOps())
    if (isa<FModuleOp, om::ClassLike>(op))
      objectContainers.push_back(&op);

  // Update Object creation ops in Classes or Modules in parallel.
  mlir::parallelForEach(ctx, objectContainers,
                        [this](auto *op) { updateInstances(op); });

  // Convert to OM ops and types in Classes or Modules in parallel.
  if (failed(
          mlir::failableParallelForEach(ctx, objectContainers, [&](auto *op) {
            return dialectConversion(op, pathInfoTable);
          })))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerClassesPass() {
  return std::make_unique<LowerClassesPass>();
}

// Predicate to check if a module-like needs a Class to be created.
bool LowerClassesPass::shouldCreateClass(FModuleLike moduleLike) {
  if (isa<firrtl::ClassLike>(moduleLike.getOperation()))
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
  om::ClassLike loweredClassOp;
  if (isa<firrtl::ExtClassOp>(moduleLike.getOperation()))
    loweredClassOp = builder.create<om::ClassExternOp>(
        moduleLike.getLoc(), className + suffix, formalParamNames);
  else
    loweredClassOp = builder.create<om::ClassOp>(
        moduleLike.getLoc(), className + suffix, formalParamNames);

  return {moduleLike, loweredClassOp};
}

void LowerClassesPass::lowerClassLike(ClassLoweringState state) {
  auto moduleLike = state.moduleLike;
  auto classLike = state.classLike;

  if (auto classOp = dyn_cast<om::ClassOp>(classLike.getOperation())) {
    return lowerClass(classOp, moduleLike);
  }
  if (auto classExternOp =
          dyn_cast<om::ClassExternOp>(classLike.getOperation())) {
    return lowerClassExtern(classExternOp, moduleLike);
  }
  llvm_unreachable("unhandled class-like op");
}

void LowerClassesPass::lowerClass(om::ClassOp classOp, FModuleLike moduleLike) {
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
  Block *classBody = &classOp->getRegion(0).emplaceBlock();
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
  if (!isa<firrtl::ClassLike>(moduleLike.getOperation())) {
    // Erase ops in use before def order, thanks to FIRRTL's SSA regions.
    for (auto *op : llvm::reverse(opsToErase))
      op->erase();

    // Erase property typed ports.
    moduleLike.erasePorts(portsToErase);
  }
}

void LowerClassesPass::lowerClassExtern(ClassExternOp classExternOp,
                                        FModuleLike moduleLike) {
  // Construct the OM Class body.
  // Add a block arguments for each input property.
  // Add a class.extern.field op for each output.
  BitVector portsToErase(moduleLike.getNumPorts());
  Block *classBody = &classExternOp.getRegion().emplaceBlock();
  OpBuilder builder = OpBuilder::atBlockBegin(classBody);

  for (unsigned i = 0, e = moduleLike.getNumPorts(); i < e; ++i) {
    auto type = moduleLike.getPortType(i);
    if (!isa<PropertyType>(type))
      continue;

    auto loc = moduleLike.getPortLocation(i);
    auto direction = moduleLike.getPortDirection(i);
    if (direction == Direction::In)
      classBody->addArgument(type, loc);
    else {
      auto name = moduleLike.getPortNameAttr(i);
      builder.create<om::ClassExternFieldOp>(loc, name, type);
    }

    // In case this is a Module, remember to erase this port.
    portsToErase.set(i);
  }

  // If the module-like is a Class, it will be completely erased later.
  // Otherwise, erase just the property ports and ops.
  if (!isa<firrtl::ClassLike>(moduleLike.getOperation())) {
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
    auto className = firrtlObject.getType().getNameAttr();
    auto classType = om::ClassType::get(op->getContext(), className);

    // Create the new Object op.
    builder.setInsertionPoint(firrtlObject);
    auto object =
        builder.create<om::ObjectOp>(firrtlObject.getLoc(), classType,
                                     className.getAttr(), actualParameters);

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

struct PathOpConversion : public OpConversionPattern<firrtl::PathOp> {

  PathOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   const PathInfoTable &pathInfoTable,
                   PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        pathInfoTable(pathInfoTable) {}

  LogicalResult
  matchAndRewrite(firrtl::PathOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    auto pathType = om::PathType::get(context);
    auto pathInfo = pathInfoTable.lookup(op.getTarget());

    // If the target was optimized away, then replace the path operation with
    // a deleted path.
    if (!pathInfo) {
      if (op.getTargetKind() == firrtl::TargetKind::DontTouch)
        return emitError(op.getLoc(), "DontTouch target was deleted");
      auto pathAttr = om::PathAttr::get(StringAttr::get(context, "OMDeleted"));
      rewriter.replaceOpWithNewOp<om::ConstantOp>(op, pathAttr);
      return success();
    }

    auto symbol = pathInfo.symRef;

    // Convert the target kind to an OMIR target.  Member references are updated
    // to reflect the current kind of reference.
    om::TargetKind targetKind;
    switch (op.getTargetKind()) {
    case firrtl::TargetKind::DontTouch:
      targetKind = om::TargetKind::DontTouch;
      break;
    case firrtl::TargetKind::Reference:
      targetKind = om::TargetKind::Reference;
      break;
    case firrtl::TargetKind::MemberInstance:
    case firrtl::TargetKind::MemberReference:
      if (isa<InstanceOp, FModuleLike>(pathInfo.op))
        targetKind = om::TargetKind::MemberInstance;
      else
        targetKind = om::TargetKind::MemberReference;
      break;
    }

    rewriter.replaceOpWithNewOp<om::PathOp>(
        op, pathType, om::TargetKindAttr::get(op.getContext(), targetKind),
        symbol);
    return success();
  }

  const PathInfoTable &pathInfoTable;
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

struct ClassExternFieldOpConversion
    : public OpConversionPattern<ClassExternFieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassExternFieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(adaptor.getType());
    if (!type)
      return failure();
    rewriter.replaceOpWithNewOp<ClassExternFieldOp>(
        op, adaptor.getSymNameAttr(), type);
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

struct ClassExternOpSignatureConversion
    : public OpConversionPattern<om::ClassExternOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(om::ClassExternOp classOp, OpAdaptor adaptor,
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
      [](Operation *op) { return !op->getParentOfType<om::ClassLike>(); });

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

  // the OM op class.extern.field doesn't have operands or results, so we must
  // check it's type for a firrtl dialect.
  target.addDynamicallyLegalOp<ClassExternFieldOp>(
      [](ClassExternFieldOp op) { return !isa<FIRRTLType>(op.getType()); });

  // OM Class ops are legal if they don't use FIRRTL types for block arguments.
  target.addDynamicallyLegalOp<om::ClassOp, om::ClassExternOp>(
      [](Operation *op) -> std::optional<bool> {
        auto classLike = dyn_cast<om::ClassLike>(op);
        if (!classLike)
          return std::nullopt;

        return llvm::none_of(
            classLike.getBodyBlock()->getArgumentTypes(),
            [](Type type) { return isa<FIRRTLDialect>(type.getDialect()); });
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

  // Convert FIRRTL PathType to OM PathType.
  converter.addConversion([](om::PathType type) { return type; });
  converter.addConversion([](firrtl::PathType type) {
    return om::PathType::get(type.getContext());
  });

  // Convert FIRRTL Class type to OM Class type.
  converter.addConversion([](om::ClassType type) { return type; });
  converter.addConversion([](firrtl::ClassType type) {
    return om::ClassType::get(type.getContext(), type.getNameAttr());
  });

  // Convert FIRRTL List type to OM List type.
  auto convertListType = [&converter](auto type) -> std::optional<mlir::Type> {
    auto elementType = converter.convertType(type.getElementType());
    if (!elementType)
      return {};
    return om::ListType::get(elementType);
  };

  converter.addConversion(
      [convertListType](om::ListType type) -> std::optional<mlir::Type> {
        // Convert any om.list<firrtl> -> om.list<om>
        return convertListType(type);
      });

  converter.addConversion(
      [convertListType](firrtl::ListType type) -> std::optional<mlir::Type> {
        // Convert any firrtl.list<firrtl> -> om.list<om>
        return convertListType(type);
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
                                    TypeConverter &converter,
                                    const PathInfoTable &pathInfoTable) {
  patterns.add<FIntegerConstantOpConversion>(converter, patterns.getContext());
  patterns.add<StringConstantOpConversion>(converter, patterns.getContext());
  patterns.add<PathOpConversion>(converter, patterns.getContext(),
                                 pathInfoTable);
  patterns.add<ClassFieldOpConversion>(converter, patterns.getContext());
  patterns.add<ClassExternFieldOpConversion>(converter, patterns.getContext());
  patterns.add<ClassOpSignatureConversion>(converter, patterns.getContext());
  patterns.add<ClassExternOpSignatureConversion>(converter,
                                                 patterns.getContext());
  patterns.add<ListCreateOpConversion>(converter, patterns.getContext());
  patterns.add<MapCreateOpConversion>(converter, patterns.getContext());
  patterns.add<BoolConstantOpConversion>(converter, patterns.getContext());
}

// Convert to OM ops and types in Classes or Modules.
LogicalResult
LowerClassesPass::dialectConversion(Operation *op,
                                    const PathInfoTable &pathInfoTable) {
  ConversionTarget target(getContext());
  populateConversionTarget(target);

  TypeConverter typeConverter;
  populateTypeConverter(typeConverter);

  RewritePatternSet patterns(&getContext());
  populateRewritePatterns(patterns, typeConverter, pathInfoTable);

  return applyPartialConversion(op, target, std::move(patterns));
}
