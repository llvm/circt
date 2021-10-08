//===- LowerTypes.cpp - Lower Aggregate Types -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerTypes pass.  This pass replaces aggregate types
// with expanded values.
//
// This pass walks the operations in reverse order. This lets it visit users
// before defs. Users can usually be expanded out to multiple operations (think
// mux of a bundle to muxes of each field) with a temporary subWhatever op
// inserted. When processing an aggregate producer, we blow out the op as
// appropriate, then walk the users, often those are subWhatever ops which can
// be bypassed and deleted. Function arguments are logically last on the
// operation visit order and walked left to right, being peeled one layer at a
// time with replacements inserted to the right of the original argument.
//
// Each processing of an op peels one layer of aggregate type off.  Because new
// ops are inserted immediately above the current up, the walk will visit them
// next, effectively recusing on the aggregate types, without recusing.  These
// potentially temporary ops(if the aggregate is complex) effectively serve as
// the worklist.  Often aggregates are shallow, so the new ops are the final
// ones.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include <deque>

using namespace circt;
using namespace firrtl;

// TODO: check all argument types
namespace {
/// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  /// This is the underlying ground type of the field.
  FIRRTLType type;
  /// The index in the parent type
  size_t index;
  /// The fieldID
  unsigned fieldID;
  /// This is a suffix to add to the field name to make it unique.
  SmallString<16> suffix;
  /// This indicates whether the field was flipped to be an output.
  bool isOutput;

  FlatBundleFieldEntry(const FIRRTLType &type, size_t index, unsigned fieldID,
                       StringRef suffix, bool isOutput)
      : type(type), index(index), fieldID(fieldID), suffix(suffix),
        isOutput(isOutput) {}

  void dump() const {
    llvm::errs() << "FBFE{" << type << " index<" << index << "> fieldID<"
                 << fieldID << "> suffix<" << suffix << "> isOutput<"
                 << isOutput << ">}\n";
  }
};
} // end anonymous namespace

/// Peel one layer of an aggregate type into its components.  Type may be
/// complex, but empty, in which case fields is empty, but the return is true.
static bool peelType(Type type, SmallVectorImpl<FlatBundleFieldEntry> &fields) {
  return TypeSwitch<Type, bool>(type)
      .Case<BundleType>([&](auto bundle) {
        SmallString<16> tmpSuffix;
        // Otherwise, we have a bundle type.  Break it down.
        for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i) {
          auto elt = bundle.getElement(i);
          // Construct the suffix to pass down.
          tmpSuffix.resize(0);
          tmpSuffix.push_back('_');
          tmpSuffix.append(elt.name.getValue());
          fields.emplace_back(elt.type, i, bundle.getFieldID(i), tmpSuffix,
                              elt.isFlip);
        }
        return true;
      })
      .Case<FVectorType>([&](auto vector) {
        // Increment the field ID to point to the first element.
        for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
          fields.emplace_back(vector.getElementType(), i, vector.getFieldID(i),
                              "_" + std::to_string(i), false);
        }
        return true;
      })
      .Default([](auto op) { return false; });
}

/// Return if something is not a normal subaccess.  Non-normal includes
/// zero-length vectors and constant indexes (which are really subindexes).
static bool isNotSubAccess(Operation *op) {
  SubaccessOp sao = dyn_cast<SubaccessOp>(op);
  if (!sao)
    return true;
  ConstantOp arg = dyn_cast_or_null<ConstantOp>(sao.index().getDefiningOp());
  if (arg && sao.input().getType().cast<FVectorType>().getNumElements() != 0)
    return true;
  return false;
}

/// Look through and collect subfields leading to a subaccess.
static SmallVector<Operation *> getSAWritePath(Operation *op) {
  SmallVector<Operation *> retval;
  auto defOp = op->getOperand(0).getDefiningOp();
  while (defOp && isa<SubfieldOp, SubindexOp, SubaccessOp>(defOp)) {
    retval.push_back(defOp);
    defOp = defOp->getOperand(0).getDefiningOp();
  }
  // Trim to the subaccess
  while (!retval.empty() && isNotSubAccess(retval.back()))
    retval.pop_back();
  return retval;
}

/// Returns whether the given annotation requires precise tracking of the field
/// ID as it gets replicated across lowered operations.
static bool isAnnotationSensitiveToFieldID(Annotation anno) {
  return anno.isClass("sifive.enterprise.grandcentral.SignalDriverAnnotation");
}

/// If an annotation on one operation is replicated across multiple IR
/// operations as a result of type lowering, the replicated annotations may want
/// to track which field ID they were applied to. This function adds a fieldID
/// to such a replicated operation, if the annotation in question requires it.
static Attribute updateAnnotationFieldID(MLIRContext *ctxt, Attribute attr,
                                         unsigned fieldID) {
  DictionaryAttr dict = attr.cast<DictionaryAttr>();

  // No need to do anything if the annotation applies to the entire field.
  if (fieldID == 0)
    return attr;

  // Only certain annotations require precise tracking of field IDs.
  Annotation anno(dict);
  if (!isAnnotationSensitiveToFieldID(anno))
    return attr;

  // Add the new ID to the existing field ID in the annotation.
  if (auto existingFieldID = anno.getMember<IntegerAttr>("fieldID"))
    fieldID += existingFieldID.getValue().getZExtValue();
  NamedAttrList fields(dict);
  fields.set("fieldID", IntegerAttr::get(IntegerType::get(ctxt, 64), fieldID));
  return DictionaryAttr::get(ctxt, fields);
}

/// Copy annotations from \p annotations to \p loweredAttrs, except annotations
/// with "target" key, that do not match the field suffix.
static ArrayAttr filterAnnotations(MLIRContext *ctxt, ArrayAttr annotations,
                                   FIRRTLType srcType,
                                   FlatBundleFieldEntry field) {
  SmallVector<Attribute> retval;
  if (!annotations || annotations.empty())
    return ArrayAttr::get(ctxt, retval);
  for (auto opAttr : annotations) {
    if (auto subAnno = opAttr.dyn_cast<SubAnnotationAttr>()) {
      // Apply annotations to all elements if fieldID is equal to zero.
      if (subAnno.getFieldID() == 0) {
        retval.push_back(subAnno.getAnnotations());
        continue;
      }

      // Check whether the annotation falls into the range of the current field.
      if (subAnno.getFieldID() >= field.fieldID &&
          subAnno.getFieldID() <= field.fieldID + field.type.getMaxFieldID()) {
        if (auto newFieldID = subAnno.getFieldID() - field.fieldID) {
          // If the target is a subfield/subindex of the current field, create a
          // new sub-annotation with a new field ID.
          retval.push_back(SubAnnotationAttr::get(ctxt, newFieldID,
                                                  subAnno.getAnnotations()));
        } else {
          // Otherwise, if the current field is exactly the target, degenerate
          // the sub-annotation to a normal annotation.
          retval.push_back(subAnno.getAnnotations());
        }
      }
    } else {
      retval.push_back(updateAnnotationFieldID(ctxt, opAttr, field.fieldID));
    }
  }
  return ArrayAttr::get(ctxt, retval);
}

static MemOp cloneMemWithNewType(ImplicitLocOpBuilder *b, MemOp op,
                                 FlatBundleFieldEntry field) {
  SmallVector<Type, 8> ports;
  SmallVector<Attribute, 8> portNames;

  auto oldPorts = op.getPorts();
  for (size_t portIdx = 0, e = oldPorts.size(); portIdx < e; ++portIdx) {
    auto port = oldPorts[portIdx];
    ports.push_back(MemOp::getTypeForPort(op.depth(), field.type, port.second));
    portNames.push_back(port.first);
  }

  // It's easier to duplicate the old annotations, then fix and filter them.
  auto newMem = b->create<MemOp>(
      ports, op.readLatency(), op.writeLatency(), op.depth(), op.ruw(),
      portNames, (op.name() + field.suffix).str(), op.annotations().getValue(),
      op.portAnnotations().getValue());

  SmallVector<Attribute> newAnnotations;
  for (size_t portIdx = 0, e = newMem.getNumResults(); portIdx < e; ++portIdx) {
    auto portType = newMem.getResult(portIdx).getType().cast<BundleType>();
    auto oldPortType = op.getResult(portIdx).getType().cast<BundleType>();
    SmallVector<Attribute> portAnno;
    for (auto attr : newMem.getPortAnnotation(portIdx)) {
      if (auto subAnno = attr.dyn_cast<SubAnnotationAttr>()) {
        auto targetIndex = oldPortType.getIndexForFieldID(subAnno.getFieldID());

        // Apply annotations to all elements if the target is the whole
        // sub-field.
        if (subAnno.getFieldID() == oldPortType.getFieldID(targetIndex)) {
          portAnno.push_back(SubAnnotationAttr::get(
              b->getContext(), portType.getFieldID(targetIndex),
              subAnno.getAnnotations()));
          continue;
        }

        // Handle aggregate sub-fields, including `(r/w)data` and `(w)mask`.
        if (oldPortType.getElement(targetIndex).type.isa<BundleType>()) {
          // Check whether the annotation falls into the range of the current
          // field. Note that the `field` here is peeled from the `data`
          // sub-field of the memory port, thus we need to add the fieldID of
          // `data` or `mask` sub-field to get the "real" fieldID.
          auto fieldID = field.fieldID + oldPortType.getFieldID(targetIndex);
          if (subAnno.getFieldID() >= fieldID &&
              subAnno.getFieldID() <= fieldID + field.type.getMaxFieldID()) {
            // Create a new sub-annotation with a new field ID. Similarly, we
            // need to add the fieldID of `data` or `mask` sub-field in the new
            // memory port type here.
            auto newFieldID = subAnno.getFieldID() - fieldID +
                              portType.getFieldID(targetIndex);
            portAnno.push_back(SubAnnotationAttr::get(
                b->getContext(), newFieldID, subAnno.getAnnotations()));
          }
        }
      } else
        portAnno.push_back(attr);
    }
    newAnnotations.push_back(b->getArrayAttr(portAnno));
  }
  newMem.setAllPortAnnotations(newAnnotations);
  return newMem;
}

//===----------------------------------------------------------------------===//
// Module Type Lowering
//===----------------------------------------------------------------------===//
namespace {
struct TypeLoweringVisitor : public FIRRTLVisitor<TypeLoweringVisitor> {

  TypeLoweringVisitor(MLIRContext *context) : context(context) {}
  using FIRRTLVisitor<TypeLoweringVisitor>::visitDecl;
  using FIRRTLVisitor<TypeLoweringVisitor>::visitExpr;
  using FIRRTLVisitor<TypeLoweringVisitor>::visitStmt;

  /// If the referenced operation is a FModuleOp or an FExtModuleOp, perform
  /// type lowering on all operations.
  void lowerModule(Operation *op);

  bool lowerArg(Operation *module, size_t argIndex,
                SmallVectorImpl<PortInfo> &newArgs,
                SmallVectorImpl<Value> &lowering);
  std::pair<Value, PortInfo> addArg(Operation *module, unsigned insertPt,
                                    FIRRTLType srcType,
                                    FlatBundleFieldEntry field,
                                    PortInfo &oldArg);

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
  void visitExpr(mlir::UnrealizedConversionCastOp op);
  void visitExpr(BitCastOp op);
  void visitStmt(ConnectOp op);
  void visitStmt(PartialConnectOp op);
  void visitStmt(WhenOp op);

private:
  void processUsers(Value val, ArrayRef<Value> mapping);
  bool processSAPath(Operation *);
  void lowerBlock(Block *);
  void lowerSAWritePath(Operation *, ArrayRef<Operation *> writePath);
  void lowerProducer(Operation *op,
                     llvm::function_ref<Operation *(FlatBundleFieldEntry,
                                                    StringRef, ArrayAttr)>
                         clone);
  Value getSubWhatever(Value val, size_t index);

  MLIRContext *context;

  /// The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  /// State to keep track of arguments and operations to clean up at the end.
  SmallVector<Operation *, 16> opsToRemove;
};
} // namespace

Value TypeLoweringVisitor::getSubWhatever(Value val, size_t index) {
  if (BundleType bundle = val.getType().dyn_cast<BundleType>()) {
    return builder->create<SubfieldOp>(val, index);
  } else if (FVectorType fvector = val.getType().dyn_cast<FVectorType>()) {
    return builder->create<SubindexOp>(val, index);
  }
  llvm_unreachable("Unknown aggregate type");
  return nullptr;
}

/// Conditionally expand a subaccessop write path
bool TypeLoweringVisitor::processSAPath(Operation *op) {
  // Does this LHS have a subaccessop?
  SmallVector<Operation *> writePath = getSAWritePath(op);
  if (!writePath.empty()) {
    lowerSAWritePath(op, writePath);
    // Unhook the writePath from the connect.  This isn't the right type, but we
    // are deleting the op anyway.
    op->eraseOperands(0, 2);
    // See how far up the tree we can delete things.
    for (size_t i = 0; i < writePath.size(); ++i) {
      if (writePath[i]->use_empty()) {
        writePath[i]->erase();
      } else {
        break;
      }
    }
    opsToRemove.push_back(op);
    return true;
  }
  return false;
}

void TypeLoweringVisitor::lowerBlock(Block *block) {
  // Lower the operations.
  for (auto &iop : llvm::reverse(*block)) {
    // We erase old ops eagerly so we don't have dangling uses we've already
    // lowered.
    for (auto *op : opsToRemove)
      op->erase();
    opsToRemove.clear();

    builder->setInsertionPoint(&iop);
    builder->setLoc(iop.getLoc());
    dispatchVisitor(&iop);
  }

  for (auto *op : opsToRemove)
    op->erase();
  opsToRemove.clear();
}

void TypeLoweringVisitor::lowerProducer(
    Operation *op,
    llvm::function_ref<Operation *(FlatBundleFieldEntry, StringRef, ArrayAttr)>
        clone) {
  // If this is not a bundle, there is nothing to do.
  auto srcType = op->getResult(0).getType().cast<FIRRTLType>();
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  if (!peelType(srcType, fieldTypes))
    return;

  SmallVector<Value> lowered;
  // Loop over the leaf aggregates.
  SmallString<16> loweredName;
  if (auto nameAttr = op->getAttr("name"))
    if (auto nameStrAttr = nameAttr.dyn_cast<StringAttr>())
      loweredName = nameStrAttr.getValue();
  auto baseNameLen = loweredName.size();
  auto oldAnno = op->getAttr("annotations").dyn_cast_or_null<ArrayAttr>();

  for (auto field : fieldTypes) {
    if (!loweredName.empty()) {
      loweredName.resize(baseNameLen);
      loweredName += field.suffix;
    }
    // For all annotations on the parent op, filter them based on the target
    // attribute.
    ArrayAttr loweredAttrs =
        filterAnnotations(context, oldAnno, srcType, field);
    auto newOp = clone(field, loweredName, loweredAttrs);
    lowered.push_back(newOp->getResult(0));
  }

  processUsers(op->getResult(0), lowered);
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::processUsers(Value val, ArrayRef<Value> mapping) {
  for (auto user : llvm::make_early_inc_range(val.getUsers())) {
    if (SubindexOp sio = dyn_cast<SubindexOp>(user)) {
      Value repl = mapping[sio.index()];
      sio.replaceAllUsesWith(repl);
      sio.erase();
    } else if (SubfieldOp sfo = dyn_cast<SubfieldOp>(user)) {
      // Get the input bundle type.
      Value repl = mapping[sfo.fieldIndex()];
      sfo.replaceAllUsesWith(repl);
      sfo.erase();
    } else {
      llvm_unreachable("Unknown aggregate user");
    }
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
std::pair<Value, PortInfo>
TypeLoweringVisitor::addArg(Operation *module, unsigned insertPt,
                            FIRRTLType srcType, FlatBundleFieldEntry field,
                            PortInfo &oldArg) {
  Value newValue;
  if (auto mod = dyn_cast<FModuleOp>(module)) {
    Block *body = mod.getBody();
    // Append the new argument.
    newValue = body->insertArgument(insertPt, field.type);
  }

  // Save the name attribute for the new argument.
  auto name =
      builder->getStringAttr(oldArg.name.getValue().str() + field.suffix.str());

  // Populate the new arg attributes.
  auto newAnnotations = filterAnnotations(
      context, oldArg.annotations.getArrayAttr(), srcType, field);

  // Flip the direction if the field is an output.
  auto direction = (Direction)((unsigned)oldArg.direction ^ field.isOutput);

  return std::make_pair(newValue,
                        PortInfo{name, field.type, direction, oldArg.loc,
                                 AnnotationSet(newAnnotations)});
}

// Lower arguments with bundle type by flattening them.
bool TypeLoweringVisitor::lowerArg(Operation *module, size_t argIndex,
                                   SmallVectorImpl<PortInfo> &newArgs,
                                   SmallVectorImpl<Value> &lowering) {

  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry> fieldTypes;
  auto srcType = newArgs[argIndex].type.cast<FIRRTLType>();
  if (!peelType(srcType, fieldTypes))
    return false;

  for (auto field : llvm::enumerate(fieldTypes)) {
    auto newValue = addArg(module, 1 + argIndex + field.index(), srcType,
                           field.value(), newArgs[argIndex]);
    newArgs.insert(newArgs.begin() + 1 + argIndex + field.index(),
                   newValue.second);
    // Lower any other arguments by copying them to keep the relative order.
    lowering.push_back(newValue.first);
  }
  return true;
}

static Value cloneAccess(ImplicitLocOpBuilder *builder, Operation *op,
                         Value rhs) {
  if (auto rop = dyn_cast<SubfieldOp>(op))
    return builder->create<SubfieldOp>(rhs, rop.fieldIndex());
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
        builder->create<ConnectOp>(leaf, op->getOperand(1));
      else
        builder->create<PartialConnectOp>(leaf, op->getOperand(1));
    });
  }
}

// Expand connects of aggregates
void TypeLoweringVisitor::visitStmt(ConnectOp op) {
  if (processSAPath(op))
    return;

  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;
  if (!peelType(op.dest().getType(), fields))
    return;

  // Loop over the leaf aggregates.
  for (auto field : llvm::enumerate(fields)) {
    Value src = getSubWhatever(op.src(), field.index());
    Value dest = getSubWhatever(op.dest(), field.index());
    if (field.value().isOutput)
      std::swap(src, dest);
    if (src.getType().isa<AnalogType>())
      builder->create<AttachOp>(ArrayRef<Value>{dest, src});
    else
      builder->create<ConnectOp>(dest, src);
  }
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitStmt(PartialConnectOp op) {
  if (processSAPath(op))
    return;

  SmallVector<FlatBundleFieldEntry> srcFields, destFields;
  peelType(op.src().getType(), srcFields);
  bool dValid = peelType(op.dest().getType(), destFields);

  // Ground Type
  if (!dValid) {
    // check for truncation
    Value src = op.src();
    Value dest = op.dest();
    auto srcType = src.getType().cast<FIRRTLType>();
    auto destType = dest.getType().cast<FIRRTLType>();
    auto srcWidth = srcType.getBitWidthOrSentinel();
    auto destWidth = destType.getBitWidthOrSentinel();

    if (destType == srcType) {
      builder->create<ConnectOp>(dest, src);
      opsToRemove.push_back(op);
    } else if (destType.isa<IntType>() && srcType.isa<IntType>() &&
               destWidth >= 0) {
      if (destWidth < srcWidth) {
        // firrtl.tail always returns uint even for sint operands.
        IntType tmpType = destType.cast<IntType>();
        if (tmpType.isSigned())
          tmpType = UIntType::get(destType.getContext(), destWidth);
        src = builder->create<TailPrimOp>(tmpType, src, srcWidth - destWidth);
        // Insert the cast back to signed if needed.
        if (tmpType != destType)
          src = builder->create<AsSIntPrimOp>(destType, src);
      } else {
        // Need to extend arg
        src = builder->create<PadPrimOp>(src, destWidth);
      }
      builder->create<ConnectOp>(dest, src);
      opsToRemove.push_back(op);
    }
    return;
  }

  // Aggregates
  if (FVectorType fvector = op.src().getType().dyn_cast<FVectorType>()) {
    for (int index = 0, e = std::min(srcFields.size(), destFields.size());
         index != e; ++index) {
      Value src = builder->create<SubindexOp>(op.src(), index);
      Value dest = builder->create<SubindexOp>(op.dest(), index);
      if (src.getType() == dest.getType())
        builder->create<ConnectOp>(dest, src);
      else
        builder->create<PartialConnectOp>(dest, src);
    }
  } else if (BundleType srcBundle = op.src().getType().dyn_cast<BundleType>()) {
    // Pairwise connect on matching field names
    BundleType destBundle = op.dest().getType().cast<BundleType>();
    for (int srcIndex = 0, srcEnd = srcBundle.getNumElements();
         srcIndex < srcEnd; ++srcIndex) {
      auto srcName = srcBundle.getElement(srcIndex).name;
      for (int destIndex = 0, destEnd = destBundle.getNumElements();
           destIndex < destEnd; ++destIndex) {
        auto destName = destBundle.getElement(destIndex).name;
        if (srcName == destName) {
          Value src = builder->create<SubfieldOp>(op.src(), srcIndex);
          Value dest = builder->create<SubfieldOp>(op.dest(), destIndex);
          if (destFields[destIndex].isOutput)
            std::swap(src, dest);
          if (src.getType().isa<AnalogType>())
            builder->create<AttachOp>(ArrayRef<Value>{dest, src});
          else if (src.getType() == dest.getType())
            builder->create<ConnectOp>(dest, src);
          else
            builder->create<PartialConnectOp>(dest, src);
          break;
        }
      }
    }
  } else {
    op.emitError("Unknown aggregate type");
  }

  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitStmt(WhenOp op) {
  // The WhenOp itself does not require any lowering, the only value it uses
  // is a one-bit predicate.  Recursively visit all regions so internal
  // operations are lowered.

  // Visit operations in the then block.
  lowerBlock(&op.getThenBlock());

  // If there is no else block, return.
  if (!op.hasElseRegion())
    return;

  // Visit operations in the else block.
  lowerBlock(&op.getElseBlock());
}

/// Lower memory operations. A new memory is created for every leaf
/// element in a memory's data type.
void TypeLoweringVisitor::visitDecl(MemOp op) {
  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;
  if (!peelType(op.getDataType(), fields))
    return;

  SmallVector<MemOp> newMemories;
  SmallVector<Value> wireToOldResult;
  SmallVector<WireOp> oldPorts;

  // Wires for old ports
  for (unsigned int index = 0, end = op.getNumResults(); index < end; ++index) {
    auto result = op.getResult(index);
    auto wire = builder->create<WireOp>(
        result.getType(),
        (op.name() + "_" + op.getPortName(index).getValue()).str());
    oldPorts.push_back(wire);
    result.replaceAllUsesWith(wire.getResult());
  }

  // Memory for each field
  for (auto field : fields)
    newMemories.push_back(cloneMemWithNewType(builder, op, field));

  // Hook up the new memories to the wires the old memory was replaced with.
  for (size_t index = 0, rend = op.getNumResults(); index < rend; ++index) {
    auto result = oldPorts[index];
    auto rType = result.getType().cast<BundleType>();
    for (size_t fieldIndex = 0, fend = rType.getNumElements();
         fieldIndex != fend; ++fieldIndex) {
      auto name = rType.getElement(fieldIndex).name.getValue();
      auto oldField = builder->create<SubfieldOp>(result, fieldIndex);
      // data and mask depend on the memory type which was split.  They can also
      // go both directions, depending on the port direction.
      if (name == "data" || name == "mask" || name == "wdata" ||
          name == "wmask" || name == "rdata") {
        for (auto field : fields) {
          auto realOldField = getSubWhatever(oldField, field.index);
          auto newField = getSubWhatever(
              newMemories[field.index].getResult(index), fieldIndex);
          if (rType.getElement(fieldIndex).isFlip)
            std::swap(realOldField, newField);
          builder->create<ConnectOp>(newField, realOldField);
        }
      } else {
        for (auto mem : newMemories) {
          auto newField =
              builder->create<SubfieldOp>(mem.getResult(index), fieldIndex);
          builder->create<ConnectOp>(newField, oldField);
        }
      }
    }
  }
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitDecl(FExtModuleOp extModule) {
  ImplicitLocOpBuilder theBuilder(extModule.getLoc(), context);
  builder = &theBuilder;

  // Top level builder
  OpBuilder builder(context);

  // Lower the module block arguments.
  SmallVector<unsigned> argsToRemove;

  auto newArgs = extModule.getPorts();
  for (size_t argIndex = 0; argIndex < newArgs.size(); ++argIndex) {
    SmallVector<Value> lowering;
    if (lowerArg(extModule, argIndex, newArgs, lowering))
      argsToRemove.push_back(argIndex);
    // lowerArg might have invalidated any reference to newArgs, be careful
  }

  // Remove block args that have been lowered
  for (auto ii = argsToRemove.rbegin(), ee = argsToRemove.rend(); ii != ee;
       ++ii)
    newArgs.erase(newArgs.begin() + *ii);

  SmallVector<NamedAttribute, 8> newModuleAttrs;

  // Copy over any attributes that weren't original argument attributes.
  for (auto attr : extModule->getAttrDictionary())
    // Drop old "portNames", directions, and argument attributes.  These are
    // handled differently below.
    if (attr.first != "portDirections" && attr.first != "portNames" &&
        attr.first != "portTypes" && attr.first != "portAnnotations")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute> newArgNames;
  SmallVector<Attribute, 8> newPortTypes;
  SmallVector<Attribute, 8> newArgAnnotations;

  for (auto &port : newArgs) {
    newArgDirections.push_back(port.direction);
    newArgNames.push_back(port.name);
    newPortTypes.push_back(TypeAttr::get(port.type));
    newArgAnnotations.push_back(port.annotations.getArrayAttr());
  }

  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get("portDirections", context),
                     direction::packAttribute(context, newArgDirections)));

  newModuleAttrs.push_back(NamedAttribute(Identifier::get("portNames", context),
                                          builder.getArrayAttr(newArgNames)));

  newModuleAttrs.push_back(NamedAttribute(Identifier::get("portTypes", context),
                                          builder.getArrayAttr(newPortTypes)));

  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get("portAnnotations", context),
                     builder.getArrayAttr(newArgAnnotations)));

  // Update the module's attributes.
  extModule->setAttrs(newModuleAttrs);
}

void TypeLoweringVisitor::visitDecl(FModuleOp module) {
  auto *body = module.getBody();

  ImplicitLocOpBuilder theBuilder(module.getLoc(), context);
  builder = &theBuilder;

  // Lower the operations.
  lowerBlock(body);

  // Lower the module block arguments.
  SmallVector<unsigned> argsToRemove;
  auto newArgs = module.getPorts();
  for (size_t argIndex = 0; argIndex < newArgs.size(); ++argIndex) {
    SmallVector<Value> lowerings;
    if (lowerArg(module, argIndex, newArgs, lowerings)) {
      auto arg = module.getArgument(argIndex);
      processUsers(arg, lowerings);
      argsToRemove.push_back(argIndex);
    }
    // lowerArg might have invalidated any reference to newArgs, be careful
  }

  // Remove block args that have been lowered.
  body->eraseArguments(argsToRemove);
  for (auto deadArg : llvm::reverse(argsToRemove))
    newArgs.erase(newArgs.begin() + deadArg);

  SmallVector<NamedAttribute, 8> newModuleAttrs;

  // Copy over any attributes that weren't original argument attributes.
  for (auto attr : module->getAttrDictionary())
    // Drop old "portNames", directions, and argument attributes.  These are
    // handled differently below.
    if (attr.first != "portNames" && attr.first != "portDirections" &&
        attr.first != "portTypes" && attr.first != "portAnnotations")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute> newArgNames;
  SmallVector<Attribute> newArgTypes;
  SmallVector<Attribute, 8> newArgAnnotations;

  for (auto &port : newArgs) {
    newArgDirections.push_back(port.direction);
    newArgNames.push_back(port.name);
    newArgTypes.push_back(TypeAttr::get(port.type));
    newArgAnnotations.push_back(port.annotations.getArrayAttr());
  }

  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get("portDirections", context),
                     direction::packAttribute(context, newArgDirections)));

  newModuleAttrs.push_back(NamedAttribute(Identifier::get("portNames", context),
                                          builder->getArrayAttr(newArgNames)));

  newModuleAttrs.push_back(NamedAttribute(Identifier::get("portTypes", context),
                                          builder->getArrayAttr(newArgTypes)));
  newModuleAttrs.push_back(
      NamedAttribute(Identifier::get("portAnnotations", context),
                     builder->getArrayAttr(newArgAnnotations)));

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
void TypeLoweringVisitor::visitDecl(WireOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   ArrayAttr attrs) -> Operation * {
    return builder->create<WireOp>(field.type, name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
void TypeLoweringVisitor::visitDecl(RegOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   ArrayAttr attrs) -> Operation * {
    return builder->create<RegOp>(field.type, op.clockVal(), name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
void TypeLoweringVisitor::visitDecl(RegResetOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   ArrayAttr attrs) -> Operation * {
    auto resetVal = getSubWhatever(op.resetValue(), field.index);
    return builder->create<RegResetOp>(field.type, op.clockVal(),
                                       op.resetSignal(), resetVal, name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
void TypeLoweringVisitor::visitDecl(NodeOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   ArrayAttr attrs) -> Operation * {
    auto input = getSubWhatever(op.input(), field.index);
    return builder->create<NodeOp>(field.type, input, name, attrs);
  };
  lowerProducer(op, clone);
}

/// Lower an InvalidValue op with a bundle to multiple non-bundled InvalidOps.
void TypeLoweringVisitor::visitExpr(InvalidValueOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   ArrayAttr attrs) -> Operation * {
    return builder->create<InvalidValueOp>(field.type);
  };
  lowerProducer(op, clone);
}

// Expand muxes of aggregates
void TypeLoweringVisitor::visitExpr(MuxPrimOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   ArrayAttr attrs) -> Operation * {
    auto high = getSubWhatever(op.high(), field.index);
    auto low = getSubWhatever(op.low(), field.index);
    return builder->create<MuxPrimOp>(op.sel(), high, low);
  };
  lowerProducer(op, clone);
}

// Expand UnrealizedConversionCastOp of aggregates
void TypeLoweringVisitor::visitExpr(mlir::UnrealizedConversionCastOp op) {
  auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                   ArrayAttr attrs) -> Operation * {
    auto input = getSubWhatever(op.getOperand(0), field.index);
    return builder->create<mlir::UnrealizedConversionCastOp>(field.type, input);
  };
  lowerProducer(op, clone);
}

// Expand BitCastOp of aggregates
void TypeLoweringVisitor::visitExpr(BitCastOp op) {
  Value srcLoweredVal = op.input();
  // If the input is of aggregate type, then cat all the leaf fields to form a
  // UInt type result. That is, first bitcast the aggregate type to a UInt.
  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;
  if (peelType(op.input().getType(), fields)) {
    size_t uptoBits = 0;
    // Loop over the leaf aggregates and concat each of them to get a UInt.
    // Bitcast the fields to handle nested aggregate types.
    for (auto field : llvm::enumerate(fields)) {
      Value src = getSubWhatever(op.input(), field.index());
      auto fieldType = src.getType().cast<FIRRTLType>();
      auto fieldBitwidth = getBitWidth(fieldType).getValue();
      // The src could be an aggregate type, bitcast it to a UInt type.
      src = builder->createOrFold<BitCastOp>(
          UIntType::get(context, fieldBitwidth), src);
      // Take the first field, or else Cat the previous fields with this field.
      if (uptoBits == 0)
        srcLoweredVal = src;
      else
        srcLoweredVal = builder->create<CatPrimOp>(src, srcLoweredVal);
      // Record the total bits already accumulated.
      uptoBits += fieldBitwidth;
    }
  } else
    srcLoweredVal = builder->createOrFold<AsUIntPrimOp>(srcLoweredVal);
  // Now the input has been cast to srcLoweredVal, which is of UInt type.
  // If the result is an aggregate type, then use lowerProducer.
  if (op.getResult().getType().isa<BundleType, FVectorType>()) {
    // uptoBits is used to keep track of the bits that have been extracted.
    size_t uptoBits = 0;
    auto clone = [&](FlatBundleFieldEntry field, StringRef name,
                     ArrayAttr attrs) -> Operation * {
      // All the fields must have valid bitwidth, a requirement for BitCastOp.
      auto fieldBits = getBitWidth(field.type).getValue();
      // Assign the field to the corresponding bits from the input.
      // Bitcast the field, incase its an aggregate type.
      auto extractBits = builder->create<BitsPrimOp>(
          srcLoweredVal, uptoBits + fieldBits - 1, uptoBits);
      uptoBits += fieldBits;
      return extractBits;
    };
    lowerProducer(op, clone);
  } else {
    // If ground type, then replace the result.
    op.getResult().replaceAllUsesWith(srcLoweredVal);
    opsToRemove.push_back(op);
  }
}

void TypeLoweringVisitor::visitDecl(InstanceOp op) {
  SmallVector<Type, 8> resultTypes;
  SmallVector<int64_t, 8> endFields; // Compressed sparse row encoding
  SmallVector<StringAttr, 8> resultNames;
  bool skip = true;
  auto oldPortAnno = op.portAnnotations();
  SmallVector<Attribute> newPortAnno;

  endFields.push_back(0);
  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    auto srcType = op.getType(i).cast<FIRRTLType>();

    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    if (!peelType(srcType, fieldTypes)) {
      resultTypes.push_back(srcType);
      newPortAnno.push_back(oldPortAnno[i]);
    } else {
      skip = false;
      // Store the flat type for the new bundle type.
      for (auto field : fieldTypes) {
        resultTypes.push_back(field.type);
        newPortAnno.push_back(filterAnnotations(
            context, oldPortAnno[i].dyn_cast_or_null<ArrayAttr>(), srcType,
            field));
      }
    }
    endFields.push_back(resultTypes.size());
  }

  if (skip)
    return;

  // FIXME: annotation update
  auto newInstance = builder->create<InstanceOp>(
      resultTypes, op.moduleNameAttr(), op.nameAttr(), op.annotations(),
      builder->getArrayAttr(newPortAnno), op.lowerToBindAttr());

  SmallVector<Value> lowered;
  for (size_t aggIndex = 0, eAgg = op.getNumResults(); aggIndex != eAgg;
       ++aggIndex) {
    lowered.clear();
    for (size_t fieldIndex = endFields[aggIndex],
                eField = endFields[aggIndex + 1];
         fieldIndex < eField; ++fieldIndex)
      lowered.push_back(newInstance.getResult(fieldIndex));
    if (lowered.size() != 1 ||
        op.getType(aggIndex) != resultTypes[endFields[aggIndex]])
      processUsers(op.getResult(aggIndex), lowered);
    else
      op.getResult(aggIndex).replaceAllUsesWith(lowered[0]);
  }
  opsToRemove.push_back(op);
}

void TypeLoweringVisitor::visitExpr(SubaccessOp op) {
  auto input = op.input();
  auto vType = input.getType().cast<FVectorType>();

  // Check for empty vectors
  if (vType.getNumElements() == 0) {
    Value inv = builder->create<InvalidValueOp>(vType.getElementType());
    op.replaceAllUsesWith(inv);
    opsToRemove.push_back(op);
    return;
  }

  // Check for constant instances
  if (ConstantOp arg =
          dyn_cast_or_null<ConstantOp>(op.index().getDefiningOp())) {
    auto sio =
        builder->create<SubindexOp>(op.input(), arg.value().getExtValue());
    op.replaceAllUsesWith(sio.getResult());
    opsToRemove.push_back(op);
    return;
  }

  // Reads.  All writes have been eliminated before now
  auto selectWidth =
      op.index().getType().cast<FIRRTLType>().getBitWidthOrSentinel();

  // We have at least one element
  Value mux = builder->create<SubindexOp>(input, 0);
  // Build up the mux
  for (size_t index = 1, e = vType.getNumElements(); index < e; ++index) {
    auto cond = builder->create<EQPrimOp>(
        op.index(), builder->createOrFold<ConstantOp>(
                        UIntType::get(op.getContext(), selectWidth),
                        APInt(selectWidth, index)));
    auto access = builder->create<SubindexOp>(input, index);
    mux = builder->create<MuxPrimOp>(cond, access, mux);
  }
  op.replaceAllUsesWith(mux);
  opsToRemove.push_back(op);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerTypesPass : public LowerFIRRTLTypesBase<LowerTypesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerTypesPass::runOnOperation() {
  std::deque<Operation *> ops;
  llvm::for_each(getOperation().getBody()->getOperations(),
                 [&](Operation &op) { ops.push_back(&op); });

  mlir::parallelForEachN(&getContext(), 0, ops.size(), [&](auto index) {
    TypeLoweringVisitor(&getContext()).lowerModule(ops[index]);
  });
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLTypesPass() {
  return std::make_unique<LowerTypesPass>();
}
