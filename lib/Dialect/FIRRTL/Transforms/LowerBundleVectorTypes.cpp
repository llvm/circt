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
#include <deque>

using namespace circt;
using namespace firrtl;

// Helper to peel off the outer most flip type from an aggregate type that has
// all flips canonicalized to the outer level, or just return the bundle
// directly. For any ground type, returns null.
static FIRRTLType getCanonicalAggregateType(Type origType) {
  FIRRTLType type = origType.dyn_cast<FIRRTLType>();
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<BundleType, FVectorType>([](auto a) { return a; })
      .Default([](auto) { return nullptr; });
}
// TODO: check all argument types
namespace {
// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  // This is the underlying ground type of the field.
  FIRRTLType type;
  // the index in the parent type
  size_t index;
  // This is a suffix to add to the field name to make it unique.
  SmallString<16> suffix;
  // This indicates whether the field was flipped to be an output.
  bool isOutput;

  FlatBundleFieldEntry(const FIRRTLType &type, size_t index, StringRef suffix,
                       bool isOutput)
      : type(type), index(index), suffix(suffix), isOutput(isOutput) {}
};
} // end anonymous namespace

// Peal one layer of an aggregate type into its components.
static SmallVector<FlatBundleFieldEntry> peelType(FIRRTLType type) {
  bool isFlipped = false;
  SmallVector<FlatBundleFieldEntry> retval;
  TypeSwitch<FIRRTLType>(type)
      .Case<BundleType>([&](auto bundle) {
        SmallString<16> tmpSuffix;
        // Otherwise, we have a bundle type.  Break it down.
        for (auto &elt : llvm::enumerate(bundle.getElements())) {
          // Construct the suffix to pass down.
          tmpSuffix.resize(0);
          tmpSuffix.push_back('_');
          tmpSuffix.append(elt.value().name.getValue());
          retval.emplace_back(elt.value().type, elt.index(), tmpSuffix,
                              isFlipped ^ elt.value().isFlip);
        }
      })
      .Case<FVectorType>([&](auto vector) {
        // Increment the field ID to point to the first element.
        for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
          retval.emplace_back(vector.getElementType(), i,
                              "_" + std::to_string(i), isFlipped);
        }
      })
      .Default([&](auto) { retval.emplace_back(type, -1, "", isFlipped); });
  return retval;
}

// Convert an aggregate type into a flat list of fields.  This is used
// when working with instances and mems to flatten them.
static void flattenType(FIRRTLType type,
                        SmallVectorImpl<FlatBundleFieldEntry> &results) {
  std::function<void(FIRRTLType, StringRef, bool)> flatten =
      [&](FIRRTLType type, StringRef suffixSoFar, bool isFlipped) {
        // Strip off an outer flip type.
        TypeSwitch<FIRRTLType>(type)
            .Case<BundleType>([&](auto bundle) {
              SmallString<16> tmpSuffix(suffixSoFar);
              // Otherwise, we have a bundle type.  Break it down.
              for (auto &elt : bundle.getElements()) {
                // Construct the suffix to pass down.
                tmpSuffix.resize(suffixSoFar.size());
                tmpSuffix.push_back('_');
                tmpSuffix.append(elt.name.getValue());
                // Recursively process subelements.
                flatten(elt.type, tmpSuffix, isFlipped ^ elt.isFlip);
              }
              return;
            })
            .Case<FVectorType>([&](auto vector) {
              for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
                flatten(vector.getElementType(),
                        (suffixSoFar + "_" + std::to_string(i)).str(),
                        isFlipped);
              }
              return;
            })
            .Default([&](auto) {
              results.emplace_back(type, -1, suffixSoFar.str(), isFlipped);
              return;
            });
      };

  return flatten(type, "", false);
}

static MemOp cloneMemWithNewType(ImplicitLocOpBuilder *b, MemOp op,
                                 FIRRTLType type, StringRef suffix) {
  SmallVector<Type, 8> ports;
  SmallVector<Attribute, 8> portNames;
  for (auto port : op.getPorts()) {
    ports.push_back(MemOp::getTypeForPort(op.depth(), type, port.second));
    portNames.push_back(port.first);
  }
  return b->create<MemOp>(ports, op.readLatency(), op.writeLatency(),
                          op.depth(), op.ruw(), b->getArrayAttr(portNames),
                          (op.name() + suffix).str(), op.annotations());
}

// Look through and collect subfields leading to a subaccess
static SmallVector<Operation *> getSAWritePath(Operation *op) {
  SmallVector<Operation *> retval;
  Value lhs = op->getOperand(0);
  while (true) {
    if (lhs.getDefiningOp() &&
        isa<SubfieldOp, SubindexOp, SubaccessOp>(lhs.getDefiningOp())) {
      retval.push_back(lhs.getDefiningOp());
      lhs = lhs.getDefiningOp()->getOperand(0);
    } else {
      break;
    }
  }
  // Trim to the subaccess
  while (!retval.empty() && !isa<SubaccessOp>(retval.back()))
    retval.pop_back();
  return retval;
}

/// Copy annotations from \p annotations to \p loweredAttrs, except annotations
/// with "target" key, that do not match the field suffix.
static SmallVector<Attribute> filterAnnotations(ArrayAttr annotations,
                                                StringRef suffix) {
  SmallVector<Attribute> retval;
  if (!annotations || annotations.empty())
    return retval;

  for (auto opAttr : annotations) {
    auto di = opAttr.dyn_cast<DictionaryAttr>();
    if (!di) {
      retval.push_back(opAttr);
      continue;
    }
    auto targetAttr = di.get("target");
    if (!targetAttr) {
      retval.push_back(opAttr);
      continue;
    }

    ArrayAttr subFieldTarget = targetAttr.cast<ArrayAttr>();
    SmallString<16> targetStr = subFieldTarget[0].cast<StringAttr>().getValue();
      // The fNameStr will begin with either '[' or '.', replace it with an
      // '_' to construct the suffix.
      targetStr[0] = '_';
      // If it ends with ']', then just remove it.
      if (targetStr.back() == ']')
        targetStr.pop_back();
    
    // If no subfield attribute, then copy the annotation.
    if (targetStr.empty()) {
      assert(0);
      retval.push_back(opAttr);
      continue;
    }
    // If the subfield suffix doesn't match, then ignore the annotation.
    if (!targetStr.equals(suffix))
      continue;
    
    NamedAttrList modAttr;
    for (auto attr : di.getValue()) {
      // Ignore the actual target annotation, but copy the rest of annotations.
      if (attr.first.str() == "target") {
        if (subFieldTarget.size() > 1)
          modAttr.append(attr.first, ArrayAttr::get(annotations.getContext(),
                                           subFieldTarget.getValue().slice(1)));

              } else {
      modAttr.push_back(attr);
        }
            }
    retval.push_back(DictionaryAttr::get(annotations.getContext(), modAttr));
  }
  return retval;
}

/// Copy annotations from \p annotations into a new AnnotationSet and return it.
/// This removes annotations with "target" key that does not match the field
/// suffix.
static AnnotationSet filterAnnotations(AnnotationSet annotations,
                                       StringRef suffix) {
  if (annotations.empty())
    return annotations;
  SmallVector<Attribute> loweredAttrs =
      filterAnnotations(annotations.getArrayAttr(), suffix);
  return AnnotationSet(ArrayAttr::get(annotations.getContext(), loweredAttrs));
}

namespace {
class AggregateUserVisitor
    : public FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>> {
public:
  AggregateUserVisitor(ImplicitLocOpBuilder *builder) : builder(builder) {}
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitDecl;
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitExpr;
  using FIRRTLVisitor<AggregateUserVisitor, void, ArrayRef<Value>>::visitStmt;

  void visitExpr(SubfieldOp op, ArrayRef<Value> mapping);
  void visitExpr(SubindexOp op, ArrayRef<Value> mapping);
  //  void visitExpr(SubaccessOp op, ArrayRef<Value> mapping);
  //  void visitExpr(AsPassivePrimOp op, ArrayRef<Value> mapping);

private:
  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;
};
} // namespace

#if 0
void AggregateUserVisitor::visitExpr(AsPassivePrimOp op, ArrayRef<Value> mapping) {
  Value repl = mapping[0];
//  if (repl.getType().isa<FlipType>())
//    repl = builder->createOrFold<AsPassivePrimOp>(repl);
  op.replaceAllUsesWith(repl);
  op.erase();  
}
#endif

void AggregateUserVisitor::visitExpr(SubindexOp op, ArrayRef<Value> mapping) {
  Value repl = mapping[op.index()];
  //  if (repl.getType().isa<FlipType>())
  //    repl = builder->createOrFold<AsPassivePrimOp>(repl);
  op.replaceAllUsesWith(repl);
  op.erase();
}

void AggregateUserVisitor::visitExpr(SubfieldOp op, ArrayRef<Value> mapping) {
  // Get the input bundle type.
  Value input = op.input();
  auto bundleType = input.getType().cast<BundleType>();
  Value repl = mapping[*bundleType.getElementIndex(op.fieldname())];
  //  if (repl.getType().isa<FlipType>())
  //    repl = builder->createOrFold<AsPassivePrimOp>(repl);
  op.replaceAllUsesWith(repl);
  op.erase();
}

#if 0
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
    //                        if (isa<ConnectOp>(mapping[0].getDefiningOp()))
    builder->create<WhenOp>(cond, false, [&]() {
      // Recreate the write Path
      Value leaf = builder->create<SubindexOp>(input, index);
      for (int i = mapping.size() - 2; i > 0; --i)
        leaf = builder->create<SubfieldOp>(
            leaf, cast<SubfieldOp>(mapping[i].getDefiningOp()).fieldname());
      if (leaf.getType() == mapping[0].getType())
                            if (foldFlow(leaf) == Flow::Source || foldFlow(mapping[0]) == Flow::Sink)
        builder->create<ConnectOp>(mapping[0], leaf);
else
        builder->create<ConnectOp>(leaf, mapping[0]);
      else
        builder->create<PartialConnectOp>(leaf, mapping[0]);
    });
    // else
    //  builder->create<WhenOp>(cond, false, [&]() {
    //    builder->create<PartialConnectOp>(builder->create<SubindexOp>(input,
    //    index),
    //                               mapping[0]);
    //  });
  }
}
#endif

//===----------------------------------------------------------------------===//
// Module Type Lowering
//===----------------------------------------------------------------------===//
namespace {
struct TypeLoweringVisitor : public FIRRTLVisitor<TypeLoweringVisitor> {

  TypeLoweringVisitor(MLIRContext *context) : context(context) {}
  using FIRRTLVisitor<TypeLoweringVisitor>::visitDecl;
  using FIRRTLVisitor<TypeLoweringVisitor>::visitExpr;
  using FIRRTLVisitor<TypeLoweringVisitor>::visitStmt;

  // If the referenced operation is a FModuleOp or an FExtModuleOp, perform
  // type lowering on all operations.
  void lowerModule(Operation *op);

  bool lowerArg(FModuleOp module, size_t argIndex,
                SmallVectorImpl<ModulePortInfo> &newArgs);
  std::pair<BlockArgument, firrtl::ModulePortInfo>
  addArg(FModuleOp module, unsigned insertPt, FIRRTLType type, bool isOutput,
         StringRef nameSuffix, ModulePortInfo &oldArg);

  // Helpers to manage state.
  void visitDecl(FExtModuleOp op);
  void visitDecl(FModuleOp op);
  void visitDecl(InstanceOp op);
  void visitDecl(MemOp op);
  void visitDecl(NodeOp op);
  void visitDecl(RegOp op);
  void visitDecl(WireOp op);
  void visitDecl(RegResetOp op);
  void visitExpr(InvalidValueOp op);
  void visitExpr(SubaccessOp op);
  void visitExpr(MuxPrimOp op);
  void visitExpr(AsPassivePrimOp op);
  void visitStmt(ConnectOp op);
  void visitStmt(PartialConnectOp op);
  void visitStmt(WhenOp op);

private:
  void processUsers(Value val, ArrayRef<Value> mapping);
  void lowerBlock(Block*);
  void lowerSAWritePath(Operation *, ArrayRef<Operation *> writePath);
  void lowerProducer(Operation *op,
                     std::function<Operation *(FlatBundleFieldEntry, StringRef,
                                               SmallVector<Attribute> &)>
                         clone);
  Value getSubWhatever(Value val, size_t index);

  MLIRContext *context;

  // The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  // State to keep track of arguments and operations to clean up at the end.
  SmallVector<Operation *, 16> opsToRemove;
};
} // namespace

Value TypeLoweringVisitor::getSubWhatever(Value val, size_t index) {
  if (BundleType bundle = val.getType().dyn_cast<BundleType>()) {
    return builder->create<SubfieldOp>(val, bundle.getElement(index).name);
  } else if (FVectorType fvector = val.getType().dyn_cast<FVectorType>()) {
    return builder->create<SubindexOp>(val, index);
  }
  llvm_unreachable("Unknown aggregate type");
  return nullptr;
}

void TypeLoweringVisitor::lowerBlock(Block* block) {
  // Lower the operations.
  for (auto iop = block->rbegin(), iep = block->rend(); iop != iep; ++iop) {
    // We erase old ops eagerly so we don't have dangling uses we've already
    // lowered.
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
}

void TypeLoweringVisitor::lowerProducer(
    Operation *op, std::function<Operation *(FlatBundleFieldEntry, StringRef,
                                             SmallVector<Attribute> &)>
                       clone) {
  Value result = op->getResult(0);

  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(result.getType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
  SmallVector<Value> lowered;

  // Loop over the leaf aggregates.
  SmallString<16> loweredName;
  if (auto nameAttr = op->getAttr("name"))
    if (auto nameStrAttr = nameAttr.dyn_cast<StringAttr>())
      loweredName = nameStrAttr.getValue();
  auto baseNameLen = loweredName.size();
  for (auto field : fieldTypes) {
    if (!loweredName.empty()) {
      loweredName.resize(baseNameLen);
      loweredName += field.suffix;
    }
    // For all annotations on the parent op, filter them based on the target
    // attribute.
    auto oldAnno = op->getAttr("annotations");
    SmallVector<Attribute> loweredAttrs;
    if (auto anno = oldAnno.dyn_cast_or_null<ArrayAttr>())
      loweredAttrs = filterAnnotations(anno, field.suffix);
    auto newOp = clone(field, loweredName, loweredAttrs);
    lowered.push_back(newOp->getResult(0));
  }

  processUsers(op->getResult(0), lowered);
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::processUsers(Value val, ArrayRef<Value> mapping) {
  AggregateUserVisitor aggV(builder);
  for (auto user : llvm::make_early_inc_range(val.getUsers())) {
    aggV.dispatchVisitor(user, mapping);
  }
}

void TypeLoweringVisitor::lowerModule(Operation *op) {
  if (auto module = dyn_cast<FModuleOp>(op))
    return visitDecl(module);
  if (auto extModule = dyn_cast<FExtModuleOp>(op))
    return visitDecl(extModule);
}

// Creates and returns a new block argument of the specified type to the
// module. This also maintains the name attribute for the new argument,
// possibly with a new suffix appended.
std::pair<BlockArgument, firrtl::ModulePortInfo>
TypeLoweringVisitor::addArg(FModuleOp module, unsigned insertPt,
                            FIRRTLType type, bool isOutput,
                            StringRef nameSuffix, ModulePortInfo &oldArg) {
  Block *body = module.getBodyBlock();

  // Append the new argument.
  auto newValue = body->insertArgument(insertPt, type);

  auto nameStr = oldArg.name.getValue().str() + nameSuffix.str();
  // Save the name attribute for the new argument.
  auto name = builder->getStringAttr(nameStr);

  // Populate the new arg attributes.
  AnnotationSet newAnnotations = oldArg.annotations;
  if (!getCanonicalAggregateType(type))
    newAnnotations = filterAnnotations(oldArg.annotations, nameStr);

  // Flip the direction if the field is an output.
  auto direction = (Direction)((unsigned)oldArg.direction ^ isOutput);

  return std::make_pair(newValue,
                        firrtl::ModulePortInfo{name, type, direction,
                                               oldArg.loc, newAnnotations});
}

// Lower arguments with bundle type by flattening them.
bool TypeLoweringVisitor::lowerArg(FModuleOp module, size_t argIndex,
                                   SmallVectorImpl<ModulePortInfo> &newArgs) {

  auto arg = module.getPortArgument(argIndex);
  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(arg.getType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return false;

  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry> fieldTypes = peelType(resultType);
  SmallVector<Value> lowering;
  for (auto field : llvm::enumerate(fieldTypes)) {
    auto newValue =
        addArg(module, 1 + argIndex + field.index(), field.value().type,
               field.value().isOutput, field.value().suffix, newArgs[argIndex]);
    newArgs.insert(newArgs.begin() + 1 + argIndex + field.index(),
                   newValue.second);
    // Lower any other arguments by copying them to keep the relative order.
    lowering.push_back(newValue.first);
  }

  processUsers(arg, lowering);

  return true;
}

static Value cloneAccess(ImplicitLocOpBuilder *builder, Operation *op,
                         Value rhs) {
  if (auto rop = dyn_cast<SubfieldOp>(op))
    return builder->create<SubfieldOp>(rhs, rop.fieldname());
  if (auto rop = dyn_cast<SubindexOp>(op))
    return builder->create<SubindexOp>(rhs, rop.index());
  if (auto rop = dyn_cast<SubaccessOp>(op))
    return builder->create<SubaccessOp>(rhs, rop.index());
  op->emitError("Unknown accessor");
  return nullptr;
}

void TypeLoweringVisitor::lowerSAWritePath(Operation *op,
                                           ArrayRef<Operation *> writePath) {
  SubaccessOp sao = cast<SubaccessOp>(writePath.back());
  auto saoType = sao.input().getType().cast<FVectorType>();
  auto selectWidth =
      sao.index().getType().cast<FIRRTLType>().getBitWidthOrSentinel();

  for (size_t index = 0, e = saoType.getNumElements(); index < e; ++index) {
    auto cond = builder->create<EQPrimOp>(
        sao.index(),
        builder->createOrFold<ConstantOp>(UIntType::get(context, selectWidth),
                                          APInt(selectWidth, index)));
    builder->create<WhenOp>(cond, false, [&]() {
      // Recreate the write Path
      Value leaf = builder->create<SubindexOp>(sao.input(), index);
      for (int i = writePath.size() - 2; i >= 0; --i)
        leaf = cloneAccess(builder, writePath[i], leaf);

      if (isa<ConnectOp>(op))
        builder->create<ConnectOp>(leaf, writePath[0]->getResult(0));
      else
        builder->create<PartialConnectOp>(leaf, writePath[0]->getResult(0));
    });
  }
}

// Expand connects of aggregates
void TypeLoweringVisitor::visitStmt(ConnectOp op) {
  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(op.src().getType());

  // Is this a write?
  SmallVector<Operation *> writePath = getSAWritePath(op);
  if (!writePath.empty()) {
    lowerSAWritePath(op, writePath);
    // unhook the writePath from the connect.  This isn't the right type, but we
    // are deleting the op anyway.
    op.setOperand(0, writePath.back()->getResult(0));
    for (size_t i = 1; i < writePath.size() - 1; ++i) {
      if (writePath[i]->use_empty())
        writePath[i]->erase();
    }
    opsToRemove.push_back(op);
    return;
  }

  // Ground Type
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry> fields = peelType(resultType);

  // Loop over the leaf aggregates.
  for (auto field : llvm::enumerate(fields)) {
    Value src, dest;
    if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
      src = builder->create<SubfieldOp>(op.src(),
                                        bundle.getElement(field.index()).name);
      dest = builder->create<SubfieldOp>(op.dest(),
                                         bundle.getElement(field.index()).name);
    } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
      src = builder->create<SubindexOp>(op.src(), field.index());
      dest = builder->create<SubindexOp>(op.dest(), field.index());
    } else {
      op->emitError("Unknown aggregate type");
    }
    if (field.value().isOutput)
      std::swap(src, dest);
    if (foldFlow(dest) == Flow::Source || foldFlow(src) == Flow::Sink)
      std::swap(src, dest);
    builder->create<ConnectOp>(dest, src);
  }
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitStmt(PartialConnectOp op) {
  auto destType = getCanonicalAggregateType(op.dest().getType());
  auto srcType = getCanonicalAggregateType(op.src().getType());

  // Should this be an assertion?
  if ((destType && !srcType) || (srcType && !destType)) {
    op.emitError("partial connect of aggregate to non-aggregate");
    return;
  }

  // Is this a write?
  SmallVector<Operation *> writePath = getSAWritePath(op);
  if (!writePath.empty()) {
    lowerSAWritePath(op, writePath);
    // unhook the writePath from the connect.  This isn't the right type, but we
    // are deleting the op anyway.
    op.setOperand(0, writePath.back()->getResult(0));
    for (size_t i = 1; i < writePath.size() - 1; ++i) {
      if (writePath[i]->use_empty())
        writePath[i]->erase();
    }
    opsToRemove.push_back(op);
    return;
  }

  // Ground Type
  if (!destType) {
    // check for truncation
    srcType = op.src().getType().cast<FIRRTLType>();
    destType = op.dest().getType().cast<FIRRTLType>();
    auto srcWidth = srcType.getBitWidthOrSentinel();
    auto destWidth = destType.getBitWidthOrSentinel();
    Value src = op.src();
    Value dest = op.dest();

    if (destType == srcType) {
      builder->create<ConnectOp>(dest, src);
      opsToRemove.push_back(op);
    } else if (destType.isa<IntType>() && srcType.isa<IntType>() && destWidth >= 0 &&
        destWidth < srcWidth) {
      // firrtl.tail always returns uint even for sint operands.
      IntType tmpType = destType.cast<IntType>();
      if (tmpType.isSigned())
        tmpType = UIntType::get(destType.getContext(), destWidth);
      //        if (srcInfo.second)
      //          src = builder->create<AsPassivePrimOp>(src);
      // assert(!src.getType().isa<FlipType>());
      src = builder->create<TailPrimOp>(tmpType, src, srcWidth - destWidth);
      // Insert the cast back to signed if needed.
      if (tmpType != destType)
        src = builder->create<AsSIntPrimOp>(destType, src);
      if (foldFlow(dest) == Flow::Source || foldFlow(src) == Flow::Sink)
        std::swap(src, dest);

      builder->create<ConnectOp>(dest, src);
      opsToRemove.push_back(op);
    }
    return;
  }

  SmallVector<FlatBundleFieldEntry> srcFields = peelType(srcType);
  SmallVector<FlatBundleFieldEntry> destFields = peelType(destType);

  if (FVectorType fvector = srcType.dyn_cast<FVectorType>()) {
    for (int index = 0, e = std::min(srcFields.size(), destFields.size());
         index != e; ++index) {
      Value src = builder->create<SubindexOp>(op.src(), index);
      Value dest = builder->create<SubindexOp>(op.dest(), index);
      if (srcFields[index].isOutput)
        std::swap(src, dest);
      if (foldFlow(dest) == Flow::Source || foldFlow(src) == Flow::Sink)
        std::swap(src, dest);

      if (src.getType() == dest.getType())
        builder->create<ConnectOp>(dest, src);
      else
        builder->create<PartialConnectOp>(dest, src);
    }
  } else if (BundleType srcBundle = srcType.dyn_cast<BundleType>()) {
    // Pairwise connect on matching field names
    BundleType destBundle = destType.cast<BundleType>();
    for (int srcIndex = 0, srcEnd = srcBundle.getNumElements();
         srcIndex < srcEnd; ++srcIndex) {
      auto srcName = srcBundle.getElement(srcIndex).name;
      for (int destIndex = 0, destEnd = destBundle.getNumElements();
           destIndex < destEnd; ++destIndex) {
        auto destName = destBundle.getElement(destIndex).name;
        if (srcName == destName) {
          Value src = builder->create<SubfieldOp>(op.src(), srcName);
          Value dest = builder->create<SubfieldOp>(op.dest(), destName);
          if (srcFields[srcIndex].isOutput)
            std::swap(src, dest);
          if (foldFlow(dest) == Flow::Source || foldFlow(src) == Flow::Sink)
            std::swap(src, dest);

          if (src.getType().isa<AnalogType>())
            builder->create<AttachOp>(ArrayRef<Value>{dest, src});
          else if (src.getType() == dest.getType())
            builder->create<ConnectOp>(dest, src);
          else
            builder->create<PartialConnectOp>(dest, src);
        }
      }
    }
  } else {
    op.emitError("Unknown aggregate type");
  }

  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitStmt(WhenOp op) {
  // The WhenOp itself does not require any lowering, the only value it uses is
  // a one-bit predicate.  Recursively visit all regions so internal operations
  // are lowered.

  // Visit operations in the then block.
  lowerBlock(&op.getThenBlock());

  // If there is no else block, return.
  if (!op.hasElseRegion())
    return;

  // Visit operations in the else block.
  lowerBlock(&op.getElseBlock());
}

#if 0
void TypeLoweringVisitor::visitDecl(MemOp op) {
  //We are splitting data, which must be the same across memory ports
  // Attempt to get the bundle types, potentially unwrapping an outer flip
  // type that wraps the whole bundle.
  FIRRTLType resultType = getCanonicalAggregateType(op.getDataType());

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);
  SmallVector<Value> lowered;
  SmallVector<MemOp> newMemories;

  // Loop over the leaf aggregates and make new memories.
  for (auto field : fieldTypes)
    newMemories.push_back(cloneMemWithNewType(builder, op, field.type, field.suffix));

  for (int portIndex = 0 , e = op.getNumResults(); portIndex < e; ++portIndex) {
    auto port = builder->create<WireOp>(op.getResult(portIndex).getType().cast<FIRRTLType>().stripFlip().first);
    FIRRTLType portType = getCanonicalAggregateType(op.getResult(portIndex).getType());
    BundleType bundle = portType.cast<BundleType>();
    SmallVector<FlatBundleFieldEntry, 8> portFields = peelType(portType);
    for (auto field : llvm::enumerate(portFields)) {
      Value origPortField = builder->create<SubfieldOp>(
          port, bundle.getElement(field.index()).name);
      if (bundle.getElement(field.index()).name.getValue() == "data") {
        auto comboData = builder->create<WireOp>(resultType);
        builder->create<ConnectOp>(comboData, origPortField);
        for (auto newMem : llvm::enumerate(newMemories)) {
          Value newPortField = builder->create<SubfieldOp>(
              newMem.value().getResult(portIndex),
              bundle.getElement(field.index()).name);

          builder->create<ConnectOp>(
              builder->create<SubfieldOp>(
                  comboData, resultType.cast<BundleType>().getElement(newMem.index()).name),
              newPortField);
        }

      } else {      
      for (auto newMem : newMemories) {
        Value newPortField = builder->create<SubfieldOp>(
            newMem.getResult(portIndex), bundle.getElement(field.index()).name);
            builder->create<ConnectOp>(origPortField, newPortField);
      }
      }
  }
    op.getResult(portIndex).replaceAllUsesWith(port);
  }
//   }
//   Value high, low;
//   if (BundleType bundle = resultType.dyn_cast<BundleType>()) {
//     high = builder->create<SubfieldOp>(op.high(),
//                                        bundle.getElement(field.index()).name);
//     low = builder->create<SubfieldOp>(op.low(),
//                                       bundle.getElement(field.index()).name);
//   } else if (FVectorType fvector = resultType.dyn_cast<FVectorType>()) {
//     high = builder->create<SubindexOp>(op.high(), field.index());
//     low = builder->create<SubindexOp>(op.low(), field.index());
//   } else {
//     op->emitError("Unknown aggregate type");
//   }

//   auto mux = builder->create<MuxPrimOp>(op.sel(), high, low);
//   lowered.push_back(mux.getResult());
// }
// processUsers(op, lowered);
  opsToRemove.push_back(op);
}
#endif

/// Lower memory operations. A new memory is created for every leaf
/// element in a memory's data type.
void TypeLoweringVisitor::visitDecl(MemOp op) {
  auto resultType = op.getDataType();
  if (!getCanonicalAggregateType(resultType))
    return;

  // If the wire is not a bundle, there is nothing to do.
  if (!resultType)
    return;

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(resultType);

  SmallVector<MemOp> newMemories;
  SmallVector<Value> wireToOldResult;
  for (auto field : fieldTypes)
    newMemories.push_back(
        cloneMemWithNewType(builder, op, field.type, field.suffix));

  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {

    SmallVector<Value> lowered;
    for (auto memResultType : op.getResult(i)
                                  .getType()
                                  .cast<FIRRTLType>()
                                  .getPassiveType()
                                  .cast<BundleType>()
                                  .getElements()) {
      auto wire = builder->create<WireOp>(memResultType.type);
      wireToOldResult.push_back(wire.getResult());
      lowered.push_back(wire.getResult());
    }
    processUsers(op.getResult(i), lowered);
  }

  SmallVector<SmallVector<Value, 4>, 8> memResultToWire;
  for (auto field : llvm::enumerate(fieldTypes)) {
    auto newMem = newMemories[field.index()];
    size_t tempWireIndex = 0;
    for (size_t i = 0, e = newMem.getNumResults(); i != e; ++i) {
      auto res = newMem.getResult(i);
      for (auto memResultType : llvm::enumerate(res.getType()
                                                    .cast<FIRRTLType>()
                                                    .cast<BundleType>()
                                                    .getElements())) {
        auto tempWire = wireToOldResult[tempWireIndex];
        ++tempWireIndex;

        auto newMemSub =
            builder->create<SubfieldOp>(res, memResultType.value().name);
        if (memResultType.value().name.getValue().contains("data") ||
            memResultType.value().name.getValue().contains("mask")) {
          Value dataAccess;
          if (tempWire.getType().cast<FIRRTLType>().dyn_cast<FVectorType>()) {
            StringRef numStr = field.value().suffix.str().drop_front(1);
            uint64_t ind;
            bool err = !numStr.getAsInteger(10, ind);
            assert(err && " Cant parse int ");
            dataAccess = builder->create<SubindexOp>(tempWire, ind);
          } else {
            dataAccess = builder->create<SubfieldOp>(
                tempWire, field.value().suffix.str().drop_front(1));
          }
          if (memResultType.value().isFlip) {
            builder->create<ConnectOp>(dataAccess, newMemSub);
          } else
            builder->create<ConnectOp>(newMemSub, dataAccess);
        } else {
          if (memResultType.value().isFlip) {
            builder->create<ConnectOp>(tempWire, newMemSub);
          } else
            builder->create<ConnectOp>(newMemSub, tempWire);
        }
      }
    }
  }

  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitDecl(FExtModuleOp extModule) {
  OpBuilder builder(context);

  // Create an array of the result types and results names.
  SmallVector<Type, 8> inputTypes;
  SmallVector<NamedAttribute, 8> attributes;
  SmallVector<Attribute, 8> argAttrDicts;

  SmallVector<Attribute> portNames;
  SmallVector<Direction> portDirections;
  unsigned oldArgNumber = 0;
  SmallString<8> attrNameBuf;
  for (auto &port : extModule.getPorts()) {
    // Flatten the port type.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    flattenType(port.type, fieldTypes);

    // Pre-populate argAttrs with the current arg attributes that are not
    // annotations.  Populate oldAnnotations with the current annotations.
    SmallVector<NamedAttribute> argAttrs;
    AnnotationSet oldAnnotations =
        AnnotationSet::forPort(extModule, oldArgNumber, argAttrs);

    // For each field, add record its name and type.
    for (auto field : fieldTypes) {
      Attribute pName;
      inputTypes.push_back(field.type);
      std::string nameStr;
      if (port.name) {
        nameStr = port.name.getValue().str();
        pName = builder.getStringAttr((port.getName() + field.suffix).str());
      } else
        pName = builder.getStringAttr("");
      nameStr += field.suffix;
      portNames.push_back(pName);
      // Flip the direction if the field is an output.
      portDirections.push_back((Direction)(
          (unsigned)getModulePortDirection(extModule, oldArgNumber) ^
          field.isOutput));

      // Populate newAnnotations with the old annotations filtered to those
      // associated with just this field.
      AnnotationSet newAnnotations = oldAnnotations;
      if (!getCanonicalAggregateType(field.type))
        newAnnotations = filterAnnotations(oldAnnotations, nameStr);

      // Populate the new arg attributes.
      argAttrDicts.push_back(newAnnotations.getArgumentAttrDict(argAttrs));
    }
    ++oldArgNumber;
  }

  // Add port names attribute.
  attributes.push_back(
      {Identifier::get(mlir::function_like_impl::getArgDictAttrName(), context),
       builder.getArrayAttr(argAttrDicts)});
  attributes.push_back(
      {Identifier::get("portNames", context), builder.getArrayAttr(portNames)});
  attributes.push_back({Identifier::get(direction::attrKey, context),
                        direction::packAttribute(portDirections, context)});

  // Copy over any lingering attributes which are not "portNames", directions,
  // or argument attributes.
  for (auto a : extModule->getAttrs()) {
    if (a.first == "portNames" || a.first == direction::attrKey ||
        a.first == mlir::function_like_impl::getArgDictAttrName())
      continue;
    attributes.push_back(a);
  }

  // Set the attributes.
  extModule->setAttrs(builder.getDictionaryAttr(attributes));

  // Set the type and then bulk set all the names.
  extModule.setType(builder.getFunctionType(inputTypes, {}));
}

static void hackBody(Block *b) {
  for (auto &bop : b->getOperations()) {
    if (ConnectOp con = dyn_cast<ConnectOp>(bop)) {
      if (foldFlow(con.dest()) == Flow::Source ||
          foldFlow(con.src()) == Flow::Sink) {
        Value lhs = con.dest();
        con.setOperand(0, con.src());
        con.setOperand(1, lhs);
      }
    } else if (WhenOp won = dyn_cast<WhenOp>(bop))
      for (auto r : won.getRegions())
        if (!r->empty())
          hackBody(&*r->begin());
  }
}

void TypeLoweringVisitor::visitDecl(FModuleOp module) {
  auto *body = module.getBodyBlock();

  ImplicitLocOpBuilder theBuilder(module.getLoc(), context);
  builder = &theBuilder;

  // Lower the operations.
  lowerBlock(body);

  // Lower the module block arguments.
  SmallVector<unsigned> argsToRemove;
  // First get all the info for existing ports
  SmallVector<ModulePortInfo> newArgs = module.getPorts();
  for (size_t argIndex = 0; argIndex < newArgs.size(); ++argIndex)
    if (lowerArg(module, argIndex, newArgs))
      argsToRemove.push_back(argIndex);
  // lowerArg might have invalidated any reference to newArgs, be careful

  // Remove block args that have been lowered.
  body->eraseArguments(argsToRemove);
  for (auto ii = argsToRemove.rbegin(), ee = argsToRemove.rend(); ii != ee;
       ++ii)
    newArgs.erase(newArgs.begin() + *ii);

  SmallVector<NamedAttribute, 8> newModuleAttrs;

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

  SmallVector<Attribute> newArgNames;
  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute, 8> newArgAttrs;
  for (auto &port : newArgs) {
    newArgNames.push_back(port.name);
    newArgDirections.push_back(port.direction);
    newArgAttrs.push_back(port.annotations.getArgumentAttrDict());
  }
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

  hackBody(body);
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
void TypeLoweringVisitor::visitDecl(WireOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   SmallVector<Attribute> &attrs) -> Operation * {
    return builder->create<WireOp>(field.type, name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
void TypeLoweringVisitor::visitDecl(RegOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   SmallVector<Attribute> &attrs) -> Operation * {
    return builder->create<RegOp>(field.type, op.clockVal(), name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
void TypeLoweringVisitor::visitDecl(RegResetOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   SmallVector<Attribute> &attrs) -> Operation * {
    auto resetVal = getSubWhatever(op.resetValue(), field.index);
    return builder->create<RegResetOp>(field.type, op.clockVal(),
                                       op.resetSignal(), resetVal, name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
void TypeLoweringVisitor::visitDecl(NodeOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   SmallVector<Attribute> &attrs) -> Operation * {
    auto input = getSubWhatever(op.input(), field.index);
    return builder->create<NodeOp>(field.type, input, name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower an InvalidValue op with a bundle to multiple non-bundled InvalidOps.
void TypeLoweringVisitor::visitExpr(InvalidValueOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   SmallVector<Attribute> &attrs) -> Operation * {
    return builder->create<InvalidValueOp>(field.type);
  };
  lowerProducer(op, clone);
}

// Expand muxes of aggregates
void TypeLoweringVisitor::visitExpr(MuxPrimOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   SmallVector<Attribute> &attrs) -> Operation * {
    auto high = getSubWhatever(op.high(), field.index);
    auto low = getSubWhatever(op.low(), field.index);
    return builder->create<MuxPrimOp>(op.sel(), high, low);
  };
  lowerProducer(op, clone);
}

// Expand AsPassivePrimOp of aggregates
void TypeLoweringVisitor::visitExpr(AsPassivePrimOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   SmallVector<Attribute> &attrs) -> Operation * {
    auto input = getSubWhatever(op.input(), field.index);
    return builder->create<AsPassivePrimOp>(field.type, op.input());
  };
  lowerProducer(op, clone);
}

void TypeLoweringVisitor::visitDecl(InstanceOp op) {
  SmallVector<Type, 8> resultTypes;
  SmallVector<int64_t, 8> numFieldsPerResult;
  SmallVector<StringAttr, 8> resultNames;
  bool skip = true;

  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    auto srcType = op.getType(i).cast<FIRRTLType>();

    if (!getCanonicalAggregateType(srcType)) {
      resultTypes.push_back(srcType);
      numFieldsPerResult.push_back(-1);
      continue;
    }
    skip = false;
    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes = peelType(srcType);

    // Store the flat type for the new bundle type.
    for (auto field : fieldTypes)
      resultTypes.push_back(field.type);
    numFieldsPerResult.push_back(fieldTypes.size());
  }
  if (skip)
    return;

  auto newInstance = builder->create<InstanceOp>(
      resultTypes, op.moduleNameAttr(), op.nameAttr(), op.annotations());
  size_t nextResult = 0;
  for (size_t aggIndex = 0, eAgg = numFieldsPerResult.size(); aggIndex != eAgg;
       ++aggIndex) {
    if (numFieldsPerResult[aggIndex] == -1) {
      op.getResult(aggIndex).replaceAllUsesWith(
          newInstance.getResult(nextResult++));
      continue;
    }
    SmallVector<Value> lowered;
    for (size_t fieldIndex = 0, eField = numFieldsPerResult[aggIndex];
         fieldIndex < eField; ++fieldIndex) {
      Value newResult = newInstance.getResult(nextResult++);
      lowered.push_back(newResult);
    }
    processUsers(op.getResult(aggIndex), lowered);
  }
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitExpr(SubaccessOp op) {
  // Get the input bundle type.
  Value input = op.input();
  auto inputType = input.getType();

  auto selectWidth =
      op.index().getType().cast<FIRRTLType>().getBitWidthOrSentinel();

  // Reads.  All writes have been eliminated before now

  auto vType = inputType.cast<FVectorType>();
  Value mux = builder->create<SubindexOp>(input, vType.getNumElements() - 1);

  for (size_t index = vType.getNumElements() - 1; index > 0; --index) {
    auto cond = builder->create<EQPrimOp>(
        op.index(), builder->createOrFold<ConstantOp>(
                        UIntType::get(op.getContext(), selectWidth),
                        APInt(selectWidth, index - 1)));
    auto access = builder->create<SubindexOp>(input, index - 1);
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
  void runParallel();
};
} // end anonymous namespace

void LowerBundleVectorPass::runParallel() {
  // Collect the operations to iterate in a vector. We can't use parallelFor
  // with the regular op list, since it requires a RandomAccessIterator. This
  // also lets us use parallelForEachN, which means we don't have to
  // llvm::enumerate the ops with their index. TODO(mlir): There should really
  // be a way to do this without collecting the operations first.
  std::deque<Operation *> ops;
  llvm::for_each(getOperation().getBody()->getOperations(),
                 [&](Operation &op) { ops.push_back(&op); });

  mlir::ParallelDiagnosticHandler diagHandler(&getContext());
  llvm::parallelForEachN(0, ops.size(), [&](auto index) {
    // Notify the handler the op index and then perform lowering.
    diagHandler.setOrderIDForThread(index);
    TypeLoweringVisitor(&getContext()).lowerModule(ops[index]);
    diagHandler.eraseOrderIDForThread();
  });
}

// This is the main entrypoint for the lowering pass.
void LowerBundleVectorPass::runOnOperation() {
  auto *context = &getContext();
  if (context->isMultithreadingEnabled())
    runParallel();
  else
    for (auto &op : getOperation().getBody()->getOperations())
      TypeLoweringVisitor(context).lowerModule(&op);
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerBundleVectorTypesPass() {
  return std::make_unique<LowerBundleVectorPass>();
}
