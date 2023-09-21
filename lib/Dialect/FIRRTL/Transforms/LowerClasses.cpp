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

/// The suffix to append to lowered module names.
static constexpr StringRef kClassNameSuffix = "_Class";

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
  LogicalResult updateInstances(Operation *op, InstanceGraph &instanceGraph);

  // Convert to OM ops and types in Classes or Modules.
  LogicalResult dialectConversion(
      Operation *op, const PathInfoTable &pathInfoTable,
      const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable);
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

  // Get the InstanceGraph.
  InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();

  // Rewrite all path annotations into inner symbol targets.
  PathInfoTable pathInfoTable;
  if (failed(lowerPaths(pathInfoTable))) {
    signalPassFailure();
    return;
  }

  // Create new OM Class ops serially.
  DenseMap<StringAttr, firrtl::ClassType> classTypeTable;
  SmallVector<ClassLoweringState> loweringState;
  for (auto moduleLike : circuit.getOps<FModuleLike>()) {
    if (shouldCreateClass(moduleLike)) {
      loweringState.push_back(createClass(moduleLike));
      if (auto classLike =
              dyn_cast<firrtl::ClassLike>(moduleLike.getOperation()))
        classTypeTable[classLike.getNameAttr()] = classLike.getInstanceType();
    }
  }

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
  if (failed(mlir::failableParallelForEach(
          ctx, objectContainers, [this, &instanceGraph](auto *op) {
            return updateInstances(op, instanceGraph);
          })))
    return signalPassFailure();

  // Convert to OM ops and types in Classes or Modules in parallel.
  if (failed(
          mlir::failableParallelForEach(ctx, objectContainers, [&](auto *op) {
            return dialectConversion(op, pathInfoTable, classTypeTable);
          })))
    return signalPassFailure();

  // We keep the instance graph up to date, so mark that analysis preserved.
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerClassesPass() {
  return std::make_unique<LowerClassesPass>();
}

// Predicate to check if a module-like needs a Class to be created.
bool LowerClassesPass::shouldCreateClass(FModuleLike moduleLike) {
  if (isa<firrtl::ClassLike>(moduleLike.getOperation()))
    return true;

  // Always create a class for public modules.
  if (moduleLike.isPublic())
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

  // If the op is a Module or ExtModule, the OM Class would conflict with the HW
  // Module, so give it a suffix. There is no formal ABI for this yet.
  StringRef suffix =
      isa<FModuleOp, FExtModuleOp>(moduleLike) ? kClassNameSuffix : "";

  // Construct the OM Class with the FIRRTL Class name and parameter names.
  om::ClassLike loweredClassOp;
  if (isa<firrtl::ExtClassOp, firrtl::FExtModuleOp>(moduleLike.getOperation()))
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

    // In case this is a Module, remember to erase this op, unless it is an
    // instance. Instances are handled later in updateInstances.
    if (!isa<InstanceOp>(op))
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

// Helper to update an Object instantiation. FIRRTL Object instances are
// converted to OM Object instances.
static LogicalResult
updateObjectInstance(firrtl::ObjectOp firrtlObject, OpBuilder &builder,
                     SmallVectorImpl<Operation *> &opsToErase) {
  // build a table mapping the indices of input ports to their position in the
  // om class's parameter list.
  auto firrtlClassType = firrtlObject.getType();
  auto numElements = firrtlClassType.getNumElements();
  llvm::SmallVector<unsigned> argIndexTable;
  argIndexTable.resize(numElements);

  unsigned nextArgIndex = 0;
  for (unsigned i = 0; i < numElements; ++i) {
    auto direction = firrtlClassType.getElement(i).direction;
    if (direction == Direction::In)
      argIndexTable[i] = nextArgIndex++;
  }

  // Collect its input actual parameters by finding any subfield ops that are
  // assigned to. Take the source of the assignment as the actual parameter.

  llvm::SmallVector<Value> args;
  args.resize(nextArgIndex);

  for (auto *user : llvm::make_early_inc_range(firrtlObject->getUsers())) {
    if (auto subfield = dyn_cast<ObjectSubfieldOp>(user)) {
      auto index = subfield.getIndex();
      auto direction = firrtlClassType.getElement(index).direction;

      // We only lower "writes to input ports" here. Reads from output
      // ports will be handled using the conversion framework.
      if (direction == Direction::Out)
        continue;

      for (auto *subfieldUser :
           llvm::make_early_inc_range(subfield->getUsers())) {
        if (auto propassign = dyn_cast<PropAssignOp>(subfieldUser)) {
          // the operands of the propassign may have already been converted to
          // om. Use the generic operand getters to get the operands as
          // untyped values.
          auto dst = propassign.getOperand(0);
          auto src = propassign.getOperand(1);
          if (dst == subfield.getResult()) {
            args[argIndexTable[index]] = src;
            opsToErase.push_back(propassign);
          }
        }
      }

      opsToErase.push_back(subfield);
    }
  }

  // Check that all input ports have been initialized.
  for (unsigned i = 0; i < numElements; ++i) {
    auto element = firrtlClassType.getElement(i);
    if (element.direction == Direction::Out)
      continue;

    auto argIndex = argIndexTable[i];
    if (!args[argIndex])
      return emitError(firrtlObject.getLoc())
             << "uninitialized input port " << element.name;
  }

  // Convert the FIRRTL Class type to an OM Class type.
  auto className = firrtlObject.getType().getNameAttr();
  auto classType = om::ClassType::get(firrtlObject->getContext(), className);

  // Create the new Object op.
  builder.setInsertionPoint(firrtlObject);
  auto object = builder.create<om::ObjectOp>(
      firrtlObject.getLoc(), classType, firrtlObject.getClassNameAttr(), args);

  // Replace uses of the FIRRTL Object with the OM Object. The later dialect
  // conversion will take care of converting the types.
  firrtlObject.replaceAllUsesWith(object.getResult());

  // Erase the original Object, now that we're done with it.
  opsToErase.push_back(firrtlObject);

  return success();
}

// Helper to update a Module instantiation in a Class. Module instances within a
// Class are converted to OM Object instances of the Class derived from the
// Module.
static LogicalResult
updateModuleInstanceClass(InstanceOp firrtlInstance, OpBuilder &builder,
                          SmallVectorImpl<Operation *> &opsToErase) {
  // Collect the FIRRTL instance inputs to form the Object instance actual
  // parameters. The order of the SmallVector needs to match the order the
  // formal parameters are declared on the corresponding Class.
  SmallVector<Value> actualParameters;
  for (auto result : firrtlInstance.getResults()) {
    // If the port is an output, continue.
    if (firrtlInstance.getPortDirection(result.getResultNumber()) ==
        Direction::Out)
      continue;

    // If the port is not a property type, continue.
    auto propertyResult = dyn_cast<FIRRTLPropertyValue>(result);
    if (!propertyResult)
      continue;

    // Get the property assignment to the input, and track the assigned
    // Value as an actual parameter to the Object instance.
    auto propertyAssignment = getPropertyAssignment(propertyResult);
    assert(propertyAssignment && "properties require single assignment");
    actualParameters.push_back(propertyAssignment.getSrc());

    // Erase the property assignment.
    opsToErase.push_back(propertyAssignment);
  }

  // Convert the FIRRTL Module name to an OM Class type.
  auto className = FlatSymbolRefAttr::get(
      builder.getStringAttr(firrtlInstance.getModuleName() + kClassNameSuffix));
  auto classType = om::ClassType::get(firrtlInstance->getContext(), className);

  // Create the new Object op.
  builder.setInsertionPoint(firrtlInstance);
  auto object =
      builder.create<om::ObjectOp>(firrtlInstance.getLoc(), classType,
                                   className.getAttr(), actualParameters);

  // Replace uses of the FIRRTL instance outputs with field access into
  // the OM Object. The later dialect conversion will take care of
  // converting the types.
  for (auto result : firrtlInstance.getResults()) {
    // If the port isn't an output, continue.
    if (firrtlInstance.getPortDirection(result.getResultNumber()) !=
        Direction::Out)
      continue;

    // If the port is not a property type, continue.
    if (!isa<PropertyType>(result.getType()))
      continue;

    // The path to the field is just this output's name.
    auto objectFieldPath = builder.getArrayAttr({FlatSymbolRefAttr::get(
        firrtlInstance.getPortName(result.getResultNumber()))});

    // Create the field access.
    auto objectField = builder.create<ObjectFieldOp>(
        object.getLoc(), result.getType(), object, objectFieldPath);

    result.replaceAllUsesWith(objectField);
  }

  // Erase the original instance, now that we're done with it.
  opsToErase.push_back(firrtlInstance);

  return success();
}

// Helper to update a Module instantiation in a Module. Module instances within
// a Module are updated to remove the property typed ports.
static LogicalResult
updateModuleInstanceModule(InstanceOp firrtlInstance, OpBuilder &builder,
                           SmallVectorImpl<Operation *> &opsToErase,
                           InstanceGraph &instanceGraph) {
  // Collect property typed ports to erase.
  BitVector portsToErase(firrtlInstance.getNumResults());
  for (auto result : firrtlInstance.getResults())
    if (isa<PropertyType>(result.getType()))
      portsToErase.set(result.getResultNumber());

  // If there are none, nothing to do.
  if (portsToErase.none())
    return success();

  // Create a new instance with the property ports removed.
  builder.setInsertionPoint(firrtlInstance);
  InstanceOp newInstance = firrtlInstance.erasePorts(builder, portsToErase);

  // Replace the instance in the instance graph. This is called from multiple
  // threads, but because the instance graph data structure is not mutated, and
  // only one thread ever sets the instance pointer for a given instance, this
  // should be safe.
  instanceGraph.replaceInstance(firrtlInstance, newInstance);

  // Erase the original instance, which is now replaced.
  opsToErase.push_back(firrtlInstance);

  return success();
}

// Update Object or Module instantiations in a FIRRTL Module or OM Class.
LogicalResult LowerClassesPass::updateInstances(Operation *op,
                                                InstanceGraph &instanceGraph) {
  OpBuilder builder(op);

  // Track ops to erase at the end. We can't do this eagerly, since we want to
  // loop over each op in the container's body, and we may end up removing some
  // ops later in the body when we visit instances earlier in the body.
  SmallVector<Operation *> opsToErase;

  // Dispatch on each Object or Module instance.
  for (auto &instance : llvm::make_early_inc_range(op->getRegion(0).getOps())) {
    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(&instance)
            .Case([&](firrtl::ObjectOp firrtlObject) {
              // Convert FIRRTL Object instance to OM Object instance.
              return updateObjectInstance(firrtlObject, builder, opsToErase);
            })
            .Case([&](InstanceOp firrtlInstance) {
              return TypeSwitch<Operation *, LogicalResult>(op)
                  .Case([&](om::ClassLike) {
                    // Convert FIRRTL Module instance within a Class to OM
                    // Object instance.
                    return updateModuleInstanceClass(firrtlInstance, builder,
                                                     opsToErase);
                  })
                  .Case([&](FModuleOp) {
                    // Convert FIRRTL Module instance within a Module to remove
                    // property ports if necessary.
                    return updateModuleInstanceModule(
                        firrtlInstance, builder, opsToErase, instanceGraph);
                  })
                  .Default([](auto *op) { return success(); });
            })
            .Default([](auto *op) { return success(); });

    if (failed(result))
      return result;
  }

  // Erase the ops marked to be erased.
  for (auto *op : opsToErase)
    op->erase();

  return success();
}

// Pattern rewriters for dialect conversion.

struct FIntegerConstantOpConversion
    : public OpConversionPattern<FIntegerConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FIntegerConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::ConstantOp>(
        op, om::OMIntegerType::get(op.getContext()),
        om::IntegerAttr::get(op.getContext(), adaptor.getValueAttr()));
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

struct DoubleConstantOpConversion
    : public OpConversionPattern<DoubleConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DoubleConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<om::ConstantOp>(op, adaptor.getValue());
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
      if (op.getTargetKind() == firrtl::TargetKind::Instance)
        return emitError(op.getLoc(), "Instance target was deleted");
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
    case firrtl::TargetKind::Instance:
      if (!isa<InstanceOp, FModuleLike>(pathInfo.op))
        return emitError(op.getLoc(), "invalid target for instance path")
                   .attachNote(pathInfo.op->getLoc())
               << "target not instance or module";
      targetKind = om::TargetKind::Instance;
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

struct WireOpConversion : public OpConversionPattern<WireOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WireOp wireOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto wireValue = dyn_cast<FIRRTLPropertyValue>(wireOp.getResult());

    // If the wire isn't a Property, not much we can do here.
    if (!wireValue)
      return failure();

    // If the wire isn't inside a graph region, we can't trivially remove it. In
    // practice, this pattern does run for wires in graph regions, so this check
    // should pass and we can proceed with the trivial rewrite.
    auto regionKindInterface = wireOp->getParentOfType<RegionKindInterface>();
    if (!regionKindInterface)
      return failure();
    if (regionKindInterface.getRegionKind(0) != RegionKind::Graph)
      return failure();

    // Find the assignment to the wire.
    PropAssignOp propAssign = getPropertyAssignment(wireValue);

    // Use the source of the assignment instead of the wire.
    rewriter.replaceOp(wireOp, propAssign.getSrc());

    // Erase the source of the assignment.
    rewriter.eraseOp(propAssign);

    return success();
  }
};

struct AnyCastOpConversion : public OpConversionPattern<ObjectAnyRefCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ObjectAnyRefCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AnyCastOp>(op, adaptor.getInput());
    return success();
  }
};

struct ObjectSubfieldOpConversion
    : public OpConversionPattern<firrtl::ObjectSubfieldOp> {
  using OpConversionPattern::OpConversionPattern;

  ObjectSubfieldOpConversion(
      const TypeConverter &typeConverter, MLIRContext *context,
      const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable)
      : OpConversionPattern(typeConverter, context),
        classTypeTable(classTypeTable) {}

  LogicalResult
  matchAndRewrite(firrtl::ObjectSubfieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto omClassType = dyn_cast<om::ClassType>(adaptor.getInput().getType());
    if (!omClassType)
      return failure();

    // Convert the field-index used by the firrtl implementation, to a symbol,
    // as used by the om implementation.
    auto firrtlClassType =
        classTypeTable.lookup(omClassType.getClassName().getAttr());
    if (!firrtlClassType)
      return failure();

    const auto &element = firrtlClassType.getElement(op.getIndex());
    // We cannot convert input ports to fields.
    if (element.direction == Direction::In)
      return failure();

    auto field = FlatSymbolRefAttr::get(element.name);
    auto path = rewriter.getArrayAttr({field});
    auto type = typeConverter->convertType(element.type);
    rewriter.replaceOpWithNewOp<om::ObjectFieldOp>(op, type, adaptor.getInput(),
                                                   path);
    return success();
  }

  const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable;
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

struct ObjectOpConversion : public OpConversionPattern<om::ObjectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(om::ObjectOp objectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the object with a new object using the converted actual parameter
    // types from the adaptor.
    rewriter.replaceOpWithNewOp<om::ObjectOp>(objectOp, objectOp.getType(),
                                              adaptor.getClassNameAttr(),
                                              adaptor.getActualParams());
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

struct ObjectFieldOpConversion : public OpConversionPattern<ObjectFieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ObjectFieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the object field with a new object field of the appropriate
    // result type based on the type converter.
    auto type = typeConverter->convertType(op.getType());
    if (!type)
      return failure();

    rewriter.replaceOpWithNewOp<ObjectFieldOp>(op, type, adaptor.getObject(),
                                               adaptor.getFieldPathAttr());

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
  converter.addConversion(
      [](IntegerType type) { return OMIntegerType::get(type.getContext()); });
  converter.addConversion([](FIntegerType type) {
    // The actual width of the IntegerType doesn't actually get used; it will be
    // folded away by the dialect conversion infrastructure to the type of the
    // APSIntAttr used in the FIntegerConstantOp.
    return OMIntegerType::get(type.getContext());
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

  // Convert FIRRTL AnyRef type to OM Any type.
  converter.addConversion([](om::AnyType type) { return type; });
  converter.addConversion([](firrtl::AnyRefType type) {
    return om::AnyType::get(type.getContext());
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

  // Convert FIRRTL double type to OM.
  converter.addConversion(
      [](DoubleType type) { return FloatType::getF64(type.getContext()); });

  // Add a target materialization to fold away unrealized conversion casts.
  converter.addTargetMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        return values[0];
      });

  // Add a source materialization to fold away unrealized conversion casts.
  converter.addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        return values[0];
      });
}

static void populateRewritePatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    const PathInfoTable &pathInfoTable,
    const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable) {
  patterns.add<FIntegerConstantOpConversion>(converter, patterns.getContext());
  patterns.add<StringConstantOpConversion>(converter, patterns.getContext());
  patterns.add<PathOpConversion>(converter, patterns.getContext(),
                                 pathInfoTable);
  patterns.add<WireOpConversion>(converter, patterns.getContext());
  patterns.add<AnyCastOpConversion>(converter, patterns.getContext());
  patterns.add<ObjectSubfieldOpConversion>(converter, patterns.getContext(),
                                           classTypeTable);
  patterns.add<ClassFieldOpConversion>(converter, patterns.getContext());
  patterns.add<ClassExternFieldOpConversion>(converter, patterns.getContext());
  patterns.add<ClassOpSignatureConversion>(converter, patterns.getContext());
  patterns.add<ClassExternOpSignatureConversion>(converter,
                                                 patterns.getContext());
  patterns.add<ObjectOpConversion>(converter, patterns.getContext());
  patterns.add<ObjectFieldOpConversion>(converter, patterns.getContext());
  patterns.add<ListCreateOpConversion>(converter, patterns.getContext());
  patterns.add<MapCreateOpConversion>(converter, patterns.getContext());
  patterns.add<BoolConstantOpConversion>(converter, patterns.getContext());
  patterns.add<DoubleConstantOpConversion>(converter, patterns.getContext());
}

// Convert to OM ops and types in Classes or Modules.
LogicalResult LowerClassesPass::dialectConversion(
    Operation *op, const PathInfoTable &pathInfoTable,
    const DenseMap<StringAttr, firrtl::ClassType> &classTypeTable) {
  ConversionTarget target(getContext());
  populateConversionTarget(target);

  TypeConverter typeConverter;
  populateTypeConverter(typeConverter);

  RewritePatternSet patterns(&getContext());
  populateRewritePatterns(patterns, typeConverter, pathInfoTable,
                          classTypeTable);

  return applyPartialConversion(op, target, std::move(patterns));
}
