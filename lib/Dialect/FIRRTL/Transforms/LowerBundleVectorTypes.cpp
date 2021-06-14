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
  SmallString<16> suffix;
  // This indicates whether the field was flipped to be an output.
  bool isOutput;

  // Helper to determine if a fully flattened type needs to be flipped.
  FIRRTLType getPortType() { return isOutput ? FlipType::get(type) : type; }
  FlatBundleFieldEntry(const FIRRTLType& type, StringRef suffix, bool isOutput)
  :type(type), suffix(suffix), isOutput(isOutput) {}
};
} // end anonymous namespace

static SmallVector<FlatBundleFieldEntry> peelType(FIRRTLType type) {
  // Field ID is the index of each field when walking aggregate types in a
  // depth-first order. See FieldRef for more information about how this
  // works.
  bool isFlipped = false;
  SmallVector<FlatBundleFieldEntry> retval;
        // Strip off an outer flip type. Flip types don't don't have a field ID,
        // and so there is no need to increment the field ID.
        if (auto flip = type.dyn_cast<FlipType>()) {
          type = flip.getElementType();
          isFlipped = !isFlipped;
        }
        TypeSwitch<FIRRTLType>(type)
            .Case<BundleType>([&](auto bundle) {
              SmallString<16> tmpSuffix;
              // Otherwise, we have a bundle type.  Break it down.
              for (auto &elt : bundle.getElements()) {
                // Construct the suffix to pass down.
                tmpSuffix.resize(0);
                tmpSuffix.push_back('_');
                tmpSuffix.append(elt.name.getValue());
                if (auto flip = elt.type.template dyn_cast<FlipType>()) {
                  retval.emplace_back(flip.getElementType(), tmpSuffix, !isFlipped);
                } else {
                  retval.emplace_back(elt.type, tmpSuffix, isFlipped);
                }
                                          }
            })
            .Case<FVectorType>([&](auto vector) {
              // Increment the field ID to point to the first element.
              for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
                retval.emplace_back(vector.getElementType(),
                        "_" + std::to_string(i),
                        isFlipped);
              }
            })
            .Default([&](auto) {
              retval.emplace_back(type, "", isFlipped);
            });
return retval;
}


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


namespace {
class AggregateUserVisitor
    : public FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>> {
public:
  AggregateUserVisitor(MLIRContext *context, ImplicitLocOpBuilder *builder)
      : context(context), builder(builder) {}
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitDecl;
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitExpr;
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitStmt;

  void visitExpr(SubfieldOp op, ArrayRef<Value> mapping);
  void visitExpr(SubindexOp op, ArrayRef<Value> mapping);
  void visitExpr(SubaccessOp op, ArrayRef<Value> mapping);

private:
  MLIRContext *context;

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;
};
}

void AggregateUserVisitor::visitExpr(SubindexOp op, ArrayRef<Value> mapping) {
  // Get the input bundle type.
  Value input = op.input();
  auto inputType = input.getType();
  if (auto flipType = inputType.dyn_cast<FlipType>())
    inputType = flipType.getElementType();

  op.replaceAllUsesWith(mapping[op.index()]);

  // Remember to remove the original op.
  op.erase();
}

void AggregateUserVisitor::visitExpr(SubaccessOp op, ArrayRef<Value> mapping) {
  // Get the input bundle type.
  Value input = op.input();
  auto inputType = input.getType();
  if (auto flipType = inputType.dyn_cast<FlipType>())
    inputType = flipType.getElementType();

  auto selectWidth =
      op.index().getType().cast<FIRRTLType>().getBitWidthOrSentinel();

  // if we were given a mapping, this is a write and the mapping is the write
  // Value
  for (size_t index = 0, e = inputType.cast<FVectorType>().getNumElements();
       index < e; ++index) {
    auto cond = builder->create<EQPrimOp>(
        op.index(), builder->createOrFold<ConstantOp>(
                        UIntType::get(op.getContext(), selectWidth),
                        APInt(selectWidth, index)));
    builder->create<WhenOp>(cond, false, [&]() {
      builder->create<ConnectOp>(builder->create<SubindexOp>(input, index),
                                 mapping[0]);
    });
  }
}

void AggregateUserVisitor::visitExpr(SubfieldOp op, ArrayRef<Value> mapping) {
  // Get the input bundle type.
  Value input = op.input();
  auto inputType = input.getType();
  if (auto flipType = inputType.dyn_cast<FlipType>())
    inputType = flipType.getElementType();
  auto bundleType = inputType.cast<BundleType>();

  op.replaceAllUsesWith(mapping[*bundleType.getElementIndex(op.fieldname())]);

  // Remember to remove the original op.
  op.erase();
}

  //===----------------------------------------------------------------------===//
  // Module Type Lowering
  //===----------------------------------------------------------------------===//
  namespace {
  class TypeLoweringVisitor : public FIRRTLVisitor<TypeLoweringVisitor> {
  public:
    TypeLoweringVisitor(MLIRContext *context) : context(context) {}
    using FIRRTLVisitor<TypeLoweringVisitor>::visitDecl;
    using FIRRTLVisitor<TypeLoweringVisitor>::visitExpr;
    using FIRRTLVisitor<TypeLoweringVisitor>::visitStmt;

    // If the referenced operation is a FModuleOp or an FExtModuleOp, perform
    // type lowering on all operations.
    void lowerModule(Operation *op);

  // Lowering module block arguments.
  void lowerArg(FModuleOp module, BlockArgument arg, FIRRTLType type);

  // Helpers to manage state.
  Value addArg(FModuleOp module, Type type, unsigned oldArgNumber,
               Direction direction, StringRef nameSuffix = "");
    void visitDecl(FExtModuleOp op);
    void visitDecl(FModuleOp op);
    void visitDecl(InstanceOp op);
//    void visitDecl(MemOp op, ArrayRef<Value> mapping);
    void visitDecl(NodeOp op);
    void visitDecl(RegOp op);
    void visitDecl(WireOp op);
    void visitDecl(RegResetOp op);
    void visitExpr(InvalidValueOp op);
//    void visitExpr(SubfieldOp op);
//    void visitExpr(SubindexOp op);
    void visitExpr(SubaccessOp op);
    void visitExpr(MuxPrimOp op);
    void visitStmt(ConnectOp op);
    void visitStmt(WhenOp op);
//    void visitStmt(PartialConnectOp op, ArrayRef<Value> mapping);

  private:

  void processUsers(Operation* op, ArrayRef<Value> mapping);
  void processUsers(Value *v, ArrayRef<Value> mapping);

void cleanup(FModuleOp module);
  MLIRContext* context;

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<Operation *, 16> opsToRemove;

  SmallVector<Attribute> newArgNames;
  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute, 8> newArgAttrs;
  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<unsigned, 8> argsToRemove;
};
}

void TypeLoweringVisitor::processUsers(Operation *op, ArrayRef<Value> mapping) {
  AggregateUserVisitor aggV(context, builder);
  for (auto user : llvm::make_early_inc_range(op->getUsers())) {
    aggV.dispatchVisitor(user, mapping);
  }
}

void TypeLoweringVisitor::processUsers(Value *v, ArrayRef<Value> mapping) {
  AggregateUserVisitor aggV(context, builder);
  for (auto user : llvm::make_early_inc_range(v->getUsers())) {
    aggV.dispatchVisitor(user, mapping);
  }
}

void TypeLoweringVisitor::lowerModule(Operation *op) {
  if (auto module = dyn_cast<FModuleOp>(op))
    return visitDecl(module);
  if (auto extModule = dyn_cast<FExtModuleOp>(op))
    return visitDecl(extModule);
}

// Expand connects of aggregates
void TypeLoweringVisitor::visitStmt(ConnectOp op) {
  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(op.src().getType());

  // Ground Type
  if (!resultType) {
    if (SubaccessOp sao = dyn_cast_or_null<SubaccessOp>(op.dest().getDefiningOp())) {
      AggregateUserVisitor(context, builder)
          .visitExpr(sao, ArrayRef<Value>(op.src()));
      opsToRemove.push_back(op);
    }
    return;
  }

  SmallVector<FlatBundleFieldEntry> fields = peelType(resultType);

  // Loop over the leaf aggregates.
  for (auto field : llvm::enumerate(fields)) {
    Value src, dest;
    if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
    src = builder->create<SubfieldOp>(op.src(), bundle.getElement(field.index()).name);
    dest = builder->create<SubfieldOp>(op.dest(),
                                       bundle.getElement(field.index()).name);
    } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
      src = builder->create<SubindexOp>(op.src(), field.index());
      dest = builder->create<SubindexOp>(op.dest(), field.index());
    } else {
      op->emitError("Unknown aggregate type");
    }
    if (field.value().isOutput)
    std::swap(src,dest);
    builder->create<ConnectOp>(dest, src);
  }
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitStmt(WhenOp op) {
  // The WhenOp itself does not require any lowering, the only value it uses is
  // a one-bit predicate.  Recursively visit all regions so internal operations
  // are lowered.

  // Visit operations in the then block.
  for (auto &op : op.getThenBlock()) {
    builder->setInsertionPoint(&op);
    builder->setLoc(op.getLoc());
    dispatchVisitor(&op);
  }

  // If there is no else block, return.
  if (!op.hasElseRegion())
    return;

  // Visit operations in the else block.
  for (auto &op : op.getElseBlock()) {
    builder->setInsertionPoint(&op);
    builder->setLoc(op.getLoc());
    dispatchVisitor(&op);
  }
}

// Expand muxes of aggregates
void TypeLoweringVisitor::visitExpr(MuxPrimOp op) {
  Value result = op.result();

  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
  SmallVector<Value> lowered;

  // Loop over the leaf aggregates.
  for (auto field : llvm::enumerate(fieldTypes)) {
    Value high, low;
    if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
      high = builder->create<SubfieldOp>(op.high(),
                                        bundle.getElement(field.index()).name);
      low = builder->create<SubfieldOp>(op.low(),
                                         bundle.getElement(field.index()).name);
    } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
      high = builder->create<SubindexOp>(op.high(), field.index());
      low = builder->create<SubindexOp>(op.low(), field.index());
    } else {
      op->emitError("Unknown aggregate type");
    }

    auto mux = builder->create<MuxPrimOp>(op.sel(), high, low);
    lowered.push_back(mux.getResult());
  }
  processUsers(op, lowered);
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitDecl(FExtModuleOp module) {
}

void TypeLoweringVisitor::visitDecl(FModuleOp module) {
  auto *body = module.getBodyBlock();

  ImplicitLocOpBuilder theBuilder(module.getLoc(), context);
  builder = &theBuilder;

  // Lower the operations.
  for (auto iop = body->rbegin(), iep = body->rend(); iop != iep; ++iop) {
    // We erase old ops eagerly so we don't have dangling uses we've already lowered.
    for (auto *op : opsToRemove)
      op->erase();
    opsToRemove.clear();

    builder->setInsertionPoint(&*iop);
    builder->setLoc(iop->getLoc());
    dispatchVisitor(&*iop);
  }

  for (auto *op : opsToRemove)
    op->erase();
  opsToRemove.clear();

  bool argNeedsLowering = false;
  do {
    argNeedsLowering = false;
    // Lower the module block arguments.
    SmallVector<BlockArgument, 8> args(body->args_begin(),       body->args_end()); 
    for (auto arg : args)
      if (auto  resultType = getCanonicalAggregateType(arg.getType()))
        argNeedsLowering = true;
    if (!argNeedsLowering) 
      break;
    newArgNames.clear();
    newArgDirections.clear();
    newArgAttrs.clear();
    for (auto arg : args)
      if (auto type = arg.getType().dyn_cast<FIRRTLType>())
        lowerArg(module, arg, type);

    if (argsToRemove.empty())
      return;
    cleanup(module);
  }while (true);

}

void TypeLoweringVisitor::cleanup(FModuleOp module){
  auto *body = module.getBodyBlock();
  // Remove block args that have been lowered.
  body->eraseArguments(argsToRemove);

  // Remember the original argument attributess.
  SmallVector<NamedAttribute, 8> originalArgAttrs;
  DictionaryAttr originalAttrs = module->getAttrDictionary();

  // Copy over any attributes that weren't original argument attributes.
  auto *argAttrBegin = originalArgAttrs.begin();
  auto *argAttrEnd = originalArgAttrs.end();
  SmallVector<NamedAttribute, 8> newModuleAttrs;
  for (auto attr : originalAttrs)
    if (std::lower_bound(argAttrBegin, argAttrEnd, attr) == argAttrEnd)
      // Drop old "portNames", directions, and argument attributes.  These are
      // handled differently below.
      if (attr.first != "portNames" && attr.first != direction::attrKey &&
          attr.first != mlir::function_like_impl::getArgDictAttrName())
        newModuleAttrs.push_back(attr);

  newModuleAttrs.push_back(NamedAttribute(Identifier::get("portNames", context),
        builder->getArrayAttr(newArgNames)));
  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get(direction::attrKey, context),
        direction::packAttribute(newArgDirections, context)));

  // Attach new argument attributes.
  newModuleAttrs.push_back(NamedAttribute(
        builder->getIdentifier(mlir::function_like_impl::getArgDictAttrName()),
        builder->getArrayAttr(newArgAttrs)));

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
  newModuleAttrs.clear();

  // Keep the module's type up-to-date.
  auto moduleType = builder->getFunctionType(body->getArgumentTypes(), {});
  module->setAttr(module.getTypeAttrName(), TypeAttr::get(moduleType));
}

// Creates and returns a new block argument of the specified type to the
// module. This also maintains the name attribute for the new argument,
// possibly with a new suffix appended.
Value TypeLoweringVisitor::addArg(FModuleOp module, Type type,
                                  unsigned oldArgNumber, Direction direction,
                                  StringRef nameSuffix) {
  Block *body = module.getBodyBlock();

  // Append the new argument.
  auto newValue = body->addArgument(type);

  // Save the name attribute for the new argument.
  StringAttr nameAttr = getModulePortName(module, oldArgNumber);
  Attribute newArg =
      builder->getStringAttr(nameAttr.getValue().str() + nameSuffix.str());
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
// Lower arguments with bundle type by flattening them.
void TypeLoweringVisitor::lowerArg(FModuleOp module, BlockArgument arg,
                                   FIRRTLType type) {
  unsigned argNumber = arg.getArgNumber();

  // Flatten any bundle types.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(type);
    SmallVector<Value> lowered;

  for (auto field : fieldTypes) {

    // Create new block arguments.
    auto type = field.type;
    // Flip the direction if the field is an output.
    auto direction =
        (Direction)((unsigned)getModulePortDirection(module, argNumber) ^
                    field.isOutput);
    auto newValue = addArg(module, type, argNumber, direction, field.suffix);

    // If this field was flattened from a bundle.
    if (!field.suffix.empty()) {
      // Map the flattened suffix for the original bundle to the new value.
      lowered.push_back(newValue);
      //setBundleLowering(FieldRef(arg, field.fieldID), newValue);
    } else {
      // Lower any other arguments by copying them to keep the relative order.
      arg.replaceAllUsesWith(newValue);
    }
  }
  processUsers(&arg, lowered);

  // Remember to remove the original block argument.
  argsToRemove.push_back(argNumber);
}

  /// Lower a wire op with a bundle to multiple non-bundled wires.
  void TypeLoweringVisitor::visitDecl(WireOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : fieldTypes) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.suffix);
      auto wire =
          builder->create<WireOp>(field.type, loweredName, loweredAttrs);
      lowered.push_back(wire.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  /// Lower a reg op with a bundle to multiple non-bundled regs.
  void TypeLoweringVisitor::visitDecl(RegOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : fieldTypes) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.suffix);
      auto reg =
          builder->create<RegOp>(field.getPortType(), op.clockVal(), loweredName, loweredAttrs);
      lowered.push_back(reg.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  /// Lower a reg op with a bundle to multiple non-bundled regs.
void TypeLoweringVisitor::visitDecl(InstanceOp op) {

  SmallVector<Type, 8> resultTypes;
  SmallVector<size_t, 8> numFieldsPerResult;
  SmallVector<StringAttr, 8> resultNames;
  bool needsLowering = false;
  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    if ( getCanonicalAggregateType(op.getType(i).cast<FIRRTLType>()))
      needsLowering = true;

    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(op.getType(i).cast<FIRRTLType>());

    for (auto field : fieldTypes) {
      // Store the flat type for the new bundle type.
      resultTypes.push_back(field.getPortType());
    }
    numFieldsPerResult.push_back(fieldTypes.size());
  }
  if (!needsLowering)
    return ;
  auto newInstance = builder->create<InstanceOp>(
      resultTypes, op.moduleNameAttr(), op.nameAttr(), op.annotations());
  size_t nextResult = 0;
  for (size_t i =0, e = op.getNumResults() ; i != e ; ++i) {
    Value result = op.getResult(i);
    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());
    if (resultType){
      SmallVector<Value> lowered;
      for (size_t j=0, e = numFieldsPerResult[i]; j != e; ++j) {
        Value newResult = newInstance.getResult(nextResult);
        lowered.push_back(newResult);
        ++nextResult;
      }
      processUsers(&result, lowered);
    } else {
      Value newResult = newInstance.getResult(nextResult);
      result.replaceAllUsesWith(newResult);
    ++nextResult;
    }
  }
  opsToRemove.push_back(op);
}

  /// Lower a reg op with a bundle to multiple non-bundled regs.
  void TypeLoweringVisitor::visitDecl(RegResetOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : llvm::enumerate(fieldTypes)) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.value().suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.value().suffix);
      Value resetVal;
      if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
        resetVal = builder->create<SubfieldOp>(
            op.resetValue(), bundle.getElement(field.index()).name);
      } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
        resetVal = builder->create<SubindexOp>(op.resetValue(), field.index());
      } else {
        op->emitError("Unknown aggregate type");
      }

      auto reg = builder->create<RegResetOp>(field.value().getPortType(), op.clockVal(), op.resetSignal(),
                                        resetVal, loweredName, loweredAttrs);
      lowered.push_back(reg.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  /// Lower a wire op with a bundle to multiple non-bundled wires.
  void TypeLoweringVisitor::visitDecl(NodeOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    // Loop over the leaf aggregates.
    auto name = op.name().str();
    for (auto field : llvm::enumerate(fieldTypes)) {
      SmallString<16> loweredName;
      if (!name.empty())
        loweredName = (name + field.value().suffix).str();
      SmallVector<Attribute> loweredAttrs;
      // For all annotations on the parent op, filter them based on the target
      // attribute.
      filterAnnotations(op.annotations(), loweredAttrs, field.value().suffix);
      Value input;
      if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
        input = builder->create<SubfieldOp>(
            op.input(), bundle.getElement(field.index()).name);
      } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
        input = builder->create<SubindexOp>(op.input(), field.index());
      } else {
        op->emitError("Unknown aggregate type");
      }

      auto node =
          builder->create<NodeOp>(field.value().type, input, loweredName, loweredAttrs);
      lowered.push_back(node.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  /// Lower an InvalidValue op with a bundle to multiple non-bundled InvalidOps.
  void TypeLoweringVisitor::visitExpr(InvalidValueOp op) {
    Value result = op.result();

    // Attempt to get the bundle types, potentially unwrapping an outer flip
    // type that wraps the whole bundle.
    FIRRTLType resultType = getCanonicalAggregateType(result.getType());

    // If the wire is not a bundle, there is nothing to do.
    if (!resultType)
      return;

    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
    SmallVector<Value> lowered;

    for (auto field : fieldTypes) {
      auto wire =
          builder->create<InvalidValueOp>(field.type);
      lowered.push_back(wire.getResult());
    }
    processUsers(op, lowered);
    opsToRemove.push_back(op);
  }

  void TypeLoweringVisitor::visitExpr(SubaccessOp op) {
    // Get the input bundle type.
    Value input = op.input();
    auto inputType = input.getType();
    if (auto flipType = inputType.dyn_cast<FlipType>())
      inputType = flipType.getElementType();

    auto selectWidth =
        op.index().getType().cast<FIRRTLType>().getBitWidthOrSentinel();

      //Reads.  All writes have been eliminated before now
      
      auto vType = inputType.cast<FVectorType>();
       Value mux =
          builder->create<InvalidValueOp>(vType.getElementType());
       
          for (size_t index = vType.getNumElements(); index > 0; --index) {
            auto cond = builder->create<EQPrimOp>(
            op.index(), builder->createOrFold<ConstantOp>(
                            UIntType::get(op.getContext(), selectWidth),
                            APInt(selectWidth, index-1)));
                            auto access = builder->create<SubindexOp>(input, index-1);
            mux = builder->create<MuxPrimOp>(cond, access, mux);
      }
      op.replaceAllUsesWith(mux);
      opsToRemove.push_back(op);


        return;
  }

  //
  //
  //===----------------------------------------------------------------------===//
  // Pass Infrastructure
  //===----------------------------------------------------------------------===//

namespace {
  struct LowerBundleVectorPass
      : public LowerBundleVectorBase<LowerBundleVectorPass> {
    void runOnOperation() override;

  private:
    void runAsync();
    void runSync();
  };
  } // end anonymous namespace

  void LowerBundleVectorPass::runSync() {
    auto circuit = getOperation();
    for (auto &op : circuit.getBody()->getOperations()) {
      TypeLoweringVisitor(&getContext()).lowerModule(&op);
    }
  }

  // This is the main entrypoint for the lowering pass.
  void LowerBundleVectorPass::runOnOperation() {
      runSync();
  }

  /// This is the pass constructor.
  std::unique_ptr<mlir::Pass>
  circt::firrtl::createLowerBundleVectorTypesPass() {
    return std::make_unique<LowerBundleVectorPass>();
  }
