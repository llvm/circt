//===- LowerBundleVectorTypes.cpp - Expand WhenOps into muxed operations ---*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerBundleVectorTypes pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Parallel.h"

using namespace circt;
using namespace firrtl;

// Helper to peel off the outer most flip type from an aggregate type that has
// all flips canonicalized to the outer level, or just return the bundle
// directly. For any ground type, returns null.
static FIRRTLType getCanonicalAggregateType(Type originalType) {
  FIRRTLType unflipped = originalType.dyn_cast<FIRRTLType>();
  if (auto flipType = originalType.dyn_cast<FlipType>())
    unflipped = flipType.getElementType();

  return TypeSwitch<FIRRTLType, FIRRTLType>(unflipped)
      .Case<BundleType, FVectorType>([](auto a) { return a; })
      .Default([](auto) { return nullptr; });
}
// TODO: check all argument types
namespace {
// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  // This is the underlying ground type of the field.
  FIRRTLType type;
  // This is a suffix to add to the field name to make it unique.
  std::string suffix;
  // This indicates whether the field was flipped to be an output.
  bool isOutput;

  // Helper to determine if a fully flattened type needs to be flipped.
  FIRRTLType getPortType() { return isOutput ? FlipType::get(type) : type; }
};
} // end anonymous namespace
/// Copy annotations from \p annotations to \p loweredAttrs, except annotations
/// with "target" key, that do not match the field suffix.
static void filterAnnotations(ArrayAttr annotations,
                              SmallVector<Attribute> &loweredAttrs,
                              StringRef suffix) {
  if (!annotations || annotations.empty())
    return;

  for (auto opAttr : annotations) {
    auto di = opAttr.dyn_cast<DictionaryAttr>();
    if (!di) {
      loweredAttrs.push_back(opAttr);
      continue;
    }
    auto targetAttr = di.get("target");
    if (!targetAttr) {
      loweredAttrs.push_back(opAttr);
      continue;
    }

    ArrayAttr subFieldTarget = targetAttr.cast<ArrayAttr>();
    SmallString<16> targetStr;
    for (auto fName : subFieldTarget) {
      std::string fNameStr = fName.cast<StringAttr>().getValue().str();
      // The fNameStr will begin with either '[' or '.', replace it with an
      // '_' to construct the suffix.
      fNameStr[0] = '_';
      // If it ends with ']', then just remove it.
      if (fNameStr.back() == ']')
        fNameStr.erase(fNameStr.size() - 1);

      targetStr += fNameStr;
    }
    // If no subfield attribute, then copy the annotation.
    if (targetStr.empty()) {
      loweredAttrs.push_back(opAttr);
      continue;
    }
    // If the subfield suffix doesn't match, then ignore the annotation.
    if (suffix.find(targetStr.str().str()) != 0)
      continue;

    NamedAttrList modAttr;
    for (auto attr : di.getValue()) {
      // Ignore the actual target annotation, but copy the rest of annotations.
      if (attr.first.str() == "target")
        continue;
      modAttr.push_back(attr);
    }
    loweredAttrs.push_back(
        DictionaryAttr::get(annotations.getContext(), modAttr));
  }
}

/// Copy annotations from \p annotations into a new AnnotationSet and return it.
/// This removes annotations with "target" key that does not match the field
/// suffix.
static AnnotationSet filterAnnotations(AnnotationSet annotations,
                                       StringRef suffix) {
  if (annotations.empty())
    return annotations;
  SmallVector<Attribute> loweredAttrs;
  filterAnnotations(annotations.getArrayAttr(), loweredAttrs, suffix);
  return AnnotationSet(ArrayAttr::get(annotations.getContext(), loweredAttrs));
}

//===----------------------------------------------------------------------===//
// Module Type Lowering
//===----------------------------------------------------------------------===//

namespace {
using LoweredValues = SmallVector<std::pair<Value, bool>>;
class doLowering {
public:
  doLowering(FModuleOp &m, ImplicitLocOpBuilder *b, MLIRContext *c)
      : module(m), builder(*b), context(*c) {

    auto *body = module.getBodyBlock();
    SmallVector<BlockArgument, 8> args(body->args_begin(), body->args_end());
    // Initialize the worklist with block arguments.
    for (BlockArgument arg : args)
      if (arg.getType().isa<FIRRTLType>())
        worklist.push_back(arg);

    runLowering();
    cleanupModule();
  }

private:
  // using ArgOrOperation = llvm::PointerUnion<BlockArgument, Operation>;
  SmallVector<Value> worklist;
  FModuleOp &module;
  ImplicitLocOpBuilder &builder;
  MLIRContext &context;
  DenseMap<Value, LoweredValues> loweredValueMap;
  SmallVector<Operation *> listOfUses;
  // State to track the new attributes for the module.
  SmallVector<NamedAttribute, 8> newModuleAttrs;
  SmallVector<Attribute> newArgNames;
  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute, 8> newArgAttrs;
  SmallVector<unsigned, 8> argsToRemove;

  void cleanupModule();
  void runLowering();
  void lowerArg(BlockArgument arg);
  void lowerSubfieldOp(SubfieldOp op);
  void lowerConnectOp(ConnectOp op);
  Value addArg(FModuleOp module, Type type, unsigned oldArgNumber,
               Direction direction, StringRef nameSuffix);
  // Convert an aggregate type into a flat list of fields.  This is used
  // when working with instances and mems to flatten them.
  void flattenType(FIRRTLType type, bool isFlipped,
                   SmallVectorImpl<FlatBundleFieldEntry> &results);
  void setBundleLowering(Value item, LoweredValues &l) {
    auto &s = loweredValueMap[item];
    llvm::errs() << "setting lowered for " << item;
    assert(s.empty() && "Bundle lowering already set");
    s = l;
  }
  bool isLowered(Value item) { return loweredValueMap.count(item); }

  bool getBundleLowering(Value item, LoweredValues &l) {
    if (!isLowered(item))
      return false;
    l = loweredValueMap[item];
    return true;
  }
};

void doLowering::cleanupModule() {
  if (argsToRemove.empty())
    return;

  Block *body = module.getBodyBlock();
  // Remove block args that have been lowered.
  body->eraseArguments(argsToRemove);

  // Remember the original argument attributess.
  SmallVector<NamedAttribute, 8> originalArgAttrs;
  DictionaryAttr originalAttrs = module->getAttrDictionary();

  // Copy over any attributes that weren't original argument attributes.
  auto *argAttrBegin = originalArgAttrs.begin();
  auto *argAttrEnd = originalArgAttrs.end();
  for (auto attr : originalAttrs)
    if (std::lower_bound(argAttrBegin, argAttrEnd, attr) == argAttrEnd)
      // Drop old "portNames", directions, and argument attributes.  These are
      // handled differently below.
      if (attr.first != "portNames" && attr.first != direction::attrKey &&
          attr.first != mlir::function_like_impl::getArgDictAttrName())
        newModuleAttrs.push_back(attr);

  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get("portNames", &context),
                     builder.getArrayAttr(newArgNames)));
  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get(direction::attrKey, &context),
                     direction::packAttribute(newArgDirections, &context)));

  // Attach new argument attributes.
  newModuleAttrs.push_back(NamedAttribute(
      builder.getIdentifier(mlir::function_like_impl::getArgDictAttrName()),
      builder.getArrayAttr(newArgAttrs)));

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
  newModuleAttrs.clear();

  // Keep the module's type up-to-date.
  auto moduleType = builder.getFunctionType(body->getArgumentTypes(), {});
  module->setAttr(module.getTypeAttrName(), TypeAttr::get(moduleType));
}

void doLowering::lowerSubfieldOp(SubfieldOp op) {

  Value input = op.input();
  StringRef fieldname = op.fieldname();
  FIRRTLType resultType = op.getType();

  // Flatten any nested bundle types the usual way.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(resultType, false, fieldTypes);
  llvm::errs() << "\n replace subfield : " << input;
  LoweredValues updatedInputs;
  if (!getBundleLowering(input, updatedInputs))
    assert("Not yet lowered");

  auto fieldIndex =
      input.getType().cast<FIRRTLType>().cast<BundleType>().getElementIndex(
          fieldname);
  assert(fieldIndex && "field index does not exist :");
  assert(updatedInputs.size() > fieldIndex.getValue() &&
         "flattened array small");

  auto newValue = updatedInputs[fieldIndex.getValue()].first;
  op.replaceAllUsesWith(newValue);
  op.erase();
}

void doLowering::lowerConnectOp(ConnectOp op) {
  Value dest = op.dest();
  Value src = op.src();

  // llvm::errs() << "\n connect;" << op;
  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType destType = getCanonicalAggregateType(dest.getType());
  FIRRTLType srcType = getCanonicalAggregateType(src.getType());

  // If we aren't connecting two bundles, there is nothing to do.
  if (!destType || !srcType)
    return;

  // SmallVector<std::pair<Value, bool>, 8> destValues;
  LoweredValues destValues;
  assert(getBundleLowering(dest, destValues) && "Not yet lowered");

  // allVector<std::pair<Value, bool>, 8> srcValues;
  LoweredValues srcValues;
  assert(getBundleLowering(src, srcValues) && "not yet lowered");

  for (auto tuple : llvm::zip_first(destValues, srcValues)) {

    auto newDest = std::get<0>(tuple).first;
    auto newDestFlipped = std::get<0>(tuple).second;
    auto newSrc = std::get<1>(tuple).first;

    // Flow checks guarantee that the connection is valid.  Therfore,
    // no flow checks are needed and just the type of the LHS
    // determines whether or not this is a reverse connection.
    if (newDestFlipped)
      std::swap(newSrc, newDest);

    builder.create<ConnectOp>(newDest, newSrc);
  }
  op.erase();
}

void doLowering::runLowering() {
  while (!worklist.empty()) {
    auto item = worklist.front();
    worklist.erase(worklist.begin());
    if (isLowered(item))
      continue;
    if (auto arg = item.dyn_cast<BlockArgument>()) {
      builder.setInsertionPoint(module);
      builder.setLoc(module.getLoc());
      lowerArg(arg);
    }
    for (auto user : item.getUsers()) {
      auto &u = *user;
      listOfUses.push_back(&u);
      builder.setInsertionPoint(&u);
      builder.setLoc(u.getLoc());
      TypeSwitch<Operation *>(&u)
                   .Case<ConnectOp>([&](auto op) {
                     bool ready = true;
                     for (auto operand : op.getOperands()) {
                       if (!isLowered(operand)) {
                         ready = false;
                         worklist.push_back(operand);
                       }
                     }
                     if (ready)
                       lowerConnectOp(op);
                   })
                   .Case<SubfieldOp>([&](auto op) {
                     lowerSubfieldOp(op);
                   })
                   .Default([](Operation *op) { 
                       op->emitError(" operation not handled ");
                       });

    }
  }
}

void doLowering::lowerArg(BlockArgument arg) {
  unsigned argNumber = arg.getArgNumber();
  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  flattenType(arg.getType().cast<FIRRTLType>(), false, fieldTypes);
  LoweredValues updatedValues;
  for (auto field : fieldTypes) {

    // Create new block arguments.
    auto newType = field.type;
    // Flip the direction if the field is an output.
    auto direction = (Direction)(
        (unsigned)getModulePortDirection(module, argNumber) ^ field.isOutput);
    auto newValue = addArg(module, newType, argNumber, direction, field.suffix);

    // If this field was flattened from a bundle.
    if (!field.suffix.empty()) {
      updatedValues.push_back({newValue, field.isOutput});
    } else {
      // Lower any other arguments by copying them to keep the relative order.
      arg.replaceAllUsesWith(newValue);
    }
  }
  setBundleLowering(arg, updatedValues);
  argsToRemove.push_back(argNumber);
}

// Creates and returns a new block argument of the specified type to the
// module. This also maintains the name attribute for the new argument,
// possibly with a new suffix appended.
Value doLowering::addArg(FModuleOp module, Type type, unsigned oldArgNumber,
                         Direction direction, StringRef nameSuffix) {
  Block *body = module.getBodyBlock();

  // Append the new argument.
  auto newValue = body->addArgument(type);

  // Save the name attribute for the new argument.
  StringAttr nameAttr = getModulePortName(module, oldArgNumber);
  Attribute newArg =
      builder.getStringAttr((nameAttr.getValue() + nameSuffix).str());
  newArgNames.push_back(newArg);
  newArgDirections.push_back(direction);

  // Decode the annotations and any additional attributes.
  SmallVector<NamedAttribute> attributes;
  auto annotations = AnnotationSet::forPort(module, oldArgNumber, attributes);

  AnnotationSet newAnnotations = filterAnnotations(annotations, nameSuffix);

  // Populate the new arg attributes.
  newArgAttrs.push_back(newAnnotations.getArgumentAttrDict(attributes));
  return newValue;
}

void doLowering::flattenType(FIRRTLType type, bool isFlipped,
                             SmallVectorImpl<FlatBundleFieldEntry> &results) {
  if (auto flip = type.dyn_cast<FlipType>())
    return flattenType(flip.getElementType(), !isFlipped, results);

  TypeSwitch<FIRRTLType>(type)
      .Case<BundleType>([&](BundleType bundle) {
        // Otherwise, we have a bundle type.  Break it down.
        for (auto &elt : bundle.getElements()) {
          auto elemType = elt.type;
          auto elemFlipped = isFlipped;
          if (auto f = elemType.dyn_cast<FlipType>()) {
            elemType = f.getElementType();
            elemFlipped = !isFlipped;
          }
          results.push_back({elemType, (Twine("_") + elt.name.getValue()).str(),
                             elemFlipped});
        }
        return;
      })
      .Case<FVectorType>([&](FVectorType vector) {
        auto elemType = vector.getElementType();
        for (size_t i = 0, e = vector.getNumElements(); i != e; ++i)
          results.push_back({elemType, "_" + std::to_string(i), isFlipped});
        return;
      })
      .Default([&](auto) {
        results.push_back({type, "", isFlipped});
        return;
      });

  return;
}

class IRVisitor : public FIRRTLVisitor<IRVisitor> {
public:
  IRVisitor(MLIRContext *context) : context(context) {}
  using FIRRTLVisitor<IRVisitor>::visitDecl;

  // If the referenced operation is a FModuleOp or an FExtModuleOp, perform type
  // lowering on all operations.
  void lowerModule(Operation *op);

  void visitDecl(FModuleOp op);

private:
  MLIRContext *context;

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;
};

void IRVisitor::lowerModule(Operation *op) {
  if (auto module = dyn_cast<FModuleOp>(op))
    return visitDecl(module);
  op->emitError(" operation not yet handled ");
  // TODO : if (auto extModule = dyn_cast<FExtModuleOp>(op))
}

void IRVisitor::visitDecl(FModuleOp module) {
  auto *body = module.getBodyBlock();

  ImplicitLocOpBuilder theBuilder(module.getLoc(), context);
  builder = &theBuilder;

  // Lower the module block arguments.
  SmallVector<BlockArgument, 8> args(body->args_begin(), body->args_end());
  doLowering d(module, builder, context);
}

//
//
//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

struct LowerBundleVectorPass
    : public LowerBundleVectorBase<LowerBundleVectorPass> {
  void runOnOperation() override;

private:
  void runAsync();
  void runSync();
};
} // end anonymous namespace

void LowerBundleVectorPass::runAsync() {
  // Collect the operations to iterate in a vector. We can't use parallelFor
  // with the regular op list, since it requires a RandomAccessIterator. This
  // also lets us use parallelForEachN, which means we don't have to
  // llvm::enumerate the ops with their index. TODO(mlir): There should really
  // be a way to do this without collecting the operations first.
  auto &body = getOperation().getBody()->getOperations();
  std::vector<Operation *> ops;
  llvm::for_each(body, [&](Operation &op) { ops.push_back(&op); });

  mlir::ParallelDiagnosticHandler diagHandler(&getContext());
  llvm::parallelForEachN(0, ops.size(), [&](auto index) {
    // Notify the handler the op index and then perform lowering.
    diagHandler.setOrderIDForThread(index);
    IRVisitor(&getContext()).lowerModule(ops[index]);
    diagHandler.eraseOrderIDForThread();
  });
}

void LowerBundleVectorPass::runSync() {
  auto circuit = getOperation();
  for (auto &op : circuit.getBody()->getOperations()) {
    IRVisitor(&getContext()).lowerModule(&op);
  }
}

// This is the main entrypoint for the lowering pass.
void LowerBundleVectorPass::runOnOperation() {
  if (getContext().isMultithreadingEnabled()) {
    runAsync();
  } else {
    runSync();
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerBundleVectorTypesPass() {
  return std::make_unique<LowerBundleVectorPass>();
}
