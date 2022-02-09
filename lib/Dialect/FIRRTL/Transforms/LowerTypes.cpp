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
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Parallel.h"

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

struct NlaNameNewSym {
  StringAttr nlaName;
  StringAttr newSym;
};
} // end anonymous namespace

/// Return true if the type has more than zero bitwidth.
static bool hasZeroBitWidth(FIRRTLType type) {
  return TypeSwitch<FIRRTLType, bool>(type)
      .Case<BundleType>([&](auto bundle) {
        for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i) {
          auto elt = bundle.getElement(i);
          if (hasZeroBitWidth(elt.type))
            return true;
        }
        return bundle.getNumElements() == 0;
      })
      .Case<FVectorType>([&](auto vector) {
        if (vector.getNumElements() == 0)
          return true;
        return hasZeroBitWidth(vector.getElementType());
      })
      .Default([](auto groundType) {
        return firrtl::getBitWidth(groundType).getValueOr(0) == 0;
      });
}

/// Return true if we can preserve the aggregate type. We can a preserve the
/// type iff (i) the type is not passive, (ii) the type doesn't contain analog
/// and (iii) type don't contain zero bitwidth.
static bool isPreservableAggregateType(Type type) {
  auto firrtlType = type.cast<FIRRTLType>();
  return firrtlType.isPassive() && !firrtlType.containsAnalog() &&
         !hasZeroBitWidth(firrtlType);
}

/// Peel one layer of an aggregate type into its components.  Type may be
/// complex, but empty, in which case fields is empty, but the return is true.
static bool peelType(Type type, SmallVectorImpl<FlatBundleFieldEntry> &fields,
                     bool allowedToPreserveAggregate = false) {
  // If the aggregate preservation is enabled and the type is preservable,
  // then just return.
  if (allowedToPreserveAggregate && isPreservableAggregateType(type))
    return false;

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
                                         unsigned fieldID, Type i64ty) {
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
  fields.set("fieldID", IntegerAttr::get(i64ty, fieldID));
  return DictionaryAttr::get(ctxt, fields);
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
      op.portAnnotations().getValue(), op.inner_symAttr());
  if (op.inner_sym())
    newMem.inner_symAttr(
        StringAttr::get(b->getContext(), op.inner_symAttr().getValue() +
                                             (op.name() + field.suffix)));

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

struct AttrCache {
  AttrCache(MLIRContext *context) {
    i64ty = IntegerType::get(context, 64);
    innerSymAttr = StringAttr::get(context, "inner_sym");
    nameAttr = StringAttr::get(context, "name");
    sPortDirections = StringAttr::get(context, "portDirections");
    sPortNames = StringAttr::get(context, "portNames");
    sPortTypes = StringAttr::get(context, "portTypes");
    sPortSyms = StringAttr::get(context, "portSyms");
    sPortAnnotations = StringAttr::get(context, "portAnnotations");
    sEmpty = StringAttr::get(context, "");
  }
  AttrCache(const AttrCache &) = default;

  Type i64ty;
  StringAttr innerSymAttr, nameAttr, sPortDirections, sPortNames, sPortTypes,
      sPortSyms, sPortAnnotations, sEmpty;
};

// The visitors all return true if the operation should be deleted, false if
// not.
struct TypeLoweringVisitor : public FIRRTLVisitor<TypeLoweringVisitor, bool> {

  TypeLoweringVisitor(MLIRContext *context, bool flattenAggregateMemData,
                      bool preserveAggregate, bool preservePublicTypes,
                      SymbolTable &symTbl, const AttrCache &cache)
      : context(context), flattenAggregateMemData(flattenAggregateMemData),
        preserveAggregate(preserveAggregate),
        preservePublicTypes(preservePublicTypes), symTbl(symTbl), cache(cache) {
  }
  using FIRRTLVisitor<TypeLoweringVisitor, bool>::visitDecl;
  using FIRRTLVisitor<TypeLoweringVisitor, bool>::visitExpr;
  using FIRRTLVisitor<TypeLoweringVisitor, bool>::visitStmt;

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
  bool visitDecl(FExtModuleOp op);
  bool visitDecl(FModuleOp op);
  bool visitDecl(InstanceOp op);
  bool visitDecl(MemOp op);
  bool visitDecl(NodeOp op);
  bool visitDecl(RegOp op);
  bool visitDecl(WireOp op);
  bool visitDecl(RegResetOp op);
  bool visitExpr(InvalidValueOp op);
  bool visitExpr(SubaccessOp op);
  bool visitExpr(MultibitMuxOp op);
  bool visitExpr(MuxPrimOp op);
  bool visitExpr(mlir::UnrealizedConversionCastOp op);
  bool visitExpr(BitCastOp op);
  bool visitStmt(ConnectOp op);
  bool visitStmt(PartialConnectOp op);
  bool visitStmt(WhenOp op);

  SmallVectorImpl<NlaNameNewSym> &getNLAs() { return nlaNameToNewSymList; };

private:
  void processUsers(Value val, ArrayRef<Value> mapping);
  bool processSAPath(Operation *);
  void lowerBlock(Block *);
  void lowerSAWritePath(Operation *, ArrayRef<Operation *> writePath);
  bool lowerProducer(
      Operation *op,
      llvm::function_ref<Operation *(FlatBundleFieldEntry, ArrayAttr)> clone);
  /// Copy annotations from \p annotations to \p loweredAttrs, except
  /// annotations with "target" key, that do not match the field suffix. Also if
  /// the target contains a DontTouch, remove it and set the flag.
  ArrayAttr filterAnnotations(MLIRContext *ctxt, ArrayAttr annotations,
                              FIRRTLType srcType, FlatBundleFieldEntry field,
                              bool &needsSym, StringRef sym);

  bool isModuleAllowedToPreserveAggregate(Operation *moduleLike);
  Value getSubWhatever(Value val, size_t index);

  size_t uniqueIdx = 0;
  std::string uniqueName() {
    auto myID = uniqueIdx++;
    return (Twine("__GEN_") + Twine(myID)).str();
  }

  MLIRContext *context;
  /// Create a single memory from an aggregate type (instead of one per field)
  /// if this flag is enabled.
  bool flattenAggregateMemData;

  /// Not to lower passive aggregate types as much as possible if this flag is
  /// enabled.
  bool preserveAggregate;

  /// Exteranal modules and toplevel modules should have lowered types if this
  /// flag is enabled.
  bool preservePublicTypes;

  /// The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  // After a bundle type is lowered to its field elements, they can have a
  // reference to a NonLocalAnchor. This new lowered field element will get a
  // new symbol, and the corresponding NLA needs to be updatd with this symbol.
  // This is a list of the NLA name  which needs to be updated, to the new
  // lowered field symbol name.
  SmallVector<NlaNameNewSym> nlaNameToNewSymList;

  // Keep a symbol table around for resolving symbols
  SymbolTable &symTbl;

  // Cache some attributes
  const AttrCache &cache;
};
} // namespace

/// Return true if we can preserve the arguments of the given module.
/// Exteranal modules and toplevel modules are sometimes assumed to have lowered
/// types.
bool TypeLoweringVisitor::isModuleAllowedToPreserveAggregate(
    Operation *moduleLike) {

  if (!preserveAggregate)
    return false;

  // If it is not forced to lower toplevel and external modules, it's ok to
  // preserve.
  if (!preservePublicTypes)
    return true;

  if (isa<FExtModuleOp>(moduleLike))
    return false;
  auto module = cast<FModuleOp>(moduleLike);
  return cast<CircuitOp>(module->getParentOp()).getMainModule() != module;
}

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
  if (writePath.empty())
    return false;

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
  return true;
}

void TypeLoweringVisitor::lowerBlock(Block *block) {
  // Lower the operations bottom up.
  for (auto it = block->rbegin(), e = block->rend(); it != e;) {
    auto &iop = *it;
    builder->setInsertionPoint(&iop);
    builder->setLoc(iop.getLoc());
    bool removeOp = dispatchVisitor(&iop);
    ++it;
    // Erase old ops eagerly so we don't have dangling uses we've already
    // lowered.
    if (removeOp)
      iop.erase();
  }
}

ArrayAttr TypeLoweringVisitor::filterAnnotations(
    MLIRContext *ctxt, ArrayAttr annotations, FIRRTLType srcType,
    FlatBundleFieldEntry field, bool &needsSym, StringRef sym) {
  SmallVector<Attribute> retval;
  if (!annotations || annotations.empty())
    return ArrayAttr::get(ctxt, retval);
  for (auto opAttr : annotations) {
    auto subAnno = opAttr.dyn_cast<SubAnnotationAttr>();
    if (!subAnno) {
      retval.push_back(
          updateAnnotationFieldID(ctxt, opAttr, field.fieldID, cache.i64ty));
      continue;
    }
    /* subAnno handling... */
    // Apply annotations to all elements if fieldID is equal to zero.
    if (subAnno.getFieldID() == 0) {
      retval.push_back(subAnno.getAnnotations());
      continue;
    }

    // Check whether the annotation falls into the range of the current field.
    if (!(subAnno.getFieldID() >= field.fieldID &&
          subAnno.getFieldID() <= field.fieldID + field.type.getMaxFieldID()))
      continue;

    if (auto newFieldID = subAnno.getFieldID() - field.fieldID) {
      // If the target is a subfield/subindex of the current field, create a
      // new sub-annotation with a new field ID.
      retval.push_back(
          SubAnnotationAttr::get(ctxt, newFieldID, subAnno.getAnnotations()));
      continue;
    }
    // Otherwise, if the current field is exactly the target, degenerate
    // the sub-annotation to a normal annotation.
    if (Annotation(opAttr).getClass() ==
        "firrtl.transforms.DontTouchAnnotation") {
      needsSym = true;
      continue;
    }
    // If this subfield has a nonlocal anchor, then we need to update the
    // NLA with the new symbol that would be added to the field after
    // lowering.
    if (auto nla = subAnno.getAnnotations().getAs<FlatSymbolRefAttr>(
            "circt.nonlocal")) {
      nlaNameToNewSymList.push_back(
          {nla.getAttr(), StringAttr::get(ctxt, sym)});
      needsSym = true;
    }
    retval.push_back(subAnno.getAnnotations());
  }
  return ArrayAttr::get(ctxt, retval);
}

bool TypeLoweringVisitor::lowerProducer(
    Operation *op,
    llvm::function_ref<Operation *(FlatBundleFieldEntry, ArrayAttr)> clone) {
  // If this is not a bundle, there is nothing to do.
  auto srcType = op->getResult(0).getType().cast<FIRRTLType>();
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;

  if (!peelType(srcType, fieldTypes, preserveAggregate))
    return false;

  SmallVector<Value> lowered;
  // Loop over the leaf aggregates.
  SmallString<16> loweredName;
  SmallString<16> loweredSymName;

  if (auto innerSymAttr = op->getAttrOfType<StringAttr>(cache.innerSymAttr))
    loweredSymName = innerSymAttr.getValue();
  if (auto nameAttr = op->getAttrOfType<StringAttr>(cache.nameAttr))
    loweredName = nameAttr.getValue();
  if (loweredSymName.empty())
    loweredSymName = loweredName;
  if (loweredSymName.empty())
    loweredSymName = uniqueName();
  auto baseNameLen = loweredName.size();
  auto baseSymNameLen = loweredSymName.size();
  auto oldAnno = op->getAttr("annotations").dyn_cast_or_null<ArrayAttr>();

  for (auto field : fieldTypes) {
    if (!loweredName.empty()) {
      loweredName.resize(baseNameLen);
      loweredName += field.suffix;
    }
    if (!loweredSymName.empty()) {
      loweredSymName.resize(baseSymNameLen);
      loweredSymName += field.suffix;
    }
    bool needsSym = false;

    // For all annotations on the parent op, filter them based on the target
    // attribute.
    ArrayAttr loweredAttrs = filterAnnotations(context, oldAnno, srcType, field,
                                               needsSym, loweredSymName);
    auto *newOp = clone(field, loweredAttrs);
    // Carry over the name, if present.
    if (!loweredName.empty())
      newOp->setAttr(cache.nameAttr, StringAttr::get(context, loweredName));
    // Carry over the inner_sym name, if present.
    if (needsSym || op->hasAttr(cache.innerSymAttr)) {
      newOp->setAttr(cache.innerSymAttr,
                     StringAttr::get(context, loweredSymName));
      assert(!loweredSymName.empty());
    }
    lowered.push_back(newOp->getResult(0));
  }

  processUsers(op->getResult(0), lowered);
  return true;
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
    visitDecl(module);
  else if (auto extModule = dyn_cast<FExtModuleOp>(op))
    visitDecl(extModule);
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
  auto name = builder->getStringAttr(oldArg.name.getValue() + field.suffix);

  SmallString<16> symtmp;
  StringRef sym;
  if (oldArg.sym) {
    symtmp = (oldArg.sym.getValue() + field.suffix).str();
    sym = symtmp;
  } else
    sym = name.getValue();

  bool needsSym = false;
  // Populate the new arg attributes.
  auto newAnnotations =
      filterAnnotations(context, oldArg.annotations.getArrayAttr(), srcType,
                        field, needsSym, sym);

  // Flip the direction if the field is an output.
  auto direction = (Direction)((unsigned)oldArg.direction ^ field.isOutput);

  StringAttr newSym = {};
  if ((needsSym || (oldArg.sym && !oldArg.sym.getValue().empty())))
    newSym = StringAttr::get(context, sym);
  return std::make_pair(newValue,
                        PortInfo{name, field.type, direction, newSym,
                                 oldArg.loc, AnnotationSet(newAnnotations)});
}

// Lower arguments with bundle type by flattening them.
bool TypeLoweringVisitor::lowerArg(Operation *module, size_t argIndex,
                                   SmallVectorImpl<PortInfo> &newArgs,
                                   SmallVectorImpl<Value> &lowering) {

  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry> fieldTypes;
  auto srcType = newArgs[argIndex].type.cast<FIRRTLType>();
  if (!peelType(srcType, fieldTypes,
                isModuleAllowedToPreserveAggregate(module)))
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
  auto selectWidth = llvm::Log2_64_Ceil(saoType.getNumElements());

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
bool TypeLoweringVisitor::visitStmt(ConnectOp op) {
  if (processSAPath(op))
    return true;

  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;

  // We have to expand connections even if the aggregate preservation is true.
  if (!peelType(op.dest().getType(), fields,
                /* allowedToPreserveAggregate */ false))
    return false;

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
  return true;
}

bool TypeLoweringVisitor::visitStmt(PartialConnectOp op) {
  if (processSAPath(op))
    return true;

  SmallVector<FlatBundleFieldEntry> srcFields, destFields;

  // For partial connects, we give up to preserve aggregates.
  peelType(op.src().getType(), srcFields,
           /* allowedToPreserveAggregate */ false);
  bool dValid = peelType(op.dest().getType(), destFields,
                         /* allowedToPreserveAggregate */ false);

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
      return true;
    }

    if (!destType.isa<IntType>() || !srcType.isa<IntType>() || destWidth < 0)
      return false;

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
    return true;
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

  return true;
}

bool TypeLoweringVisitor::visitStmt(WhenOp op) {
  // The WhenOp itself does not require any lowering, the only value it uses
  // is a one-bit predicate.  Recursively visit all regions so internal
  // operations are lowered.

  // Visit operations in the then block.
  lowerBlock(&op.getThenBlock());

  // Visit operations in the else block.
  if (op.hasElseRegion())
    lowerBlock(&op.getElseBlock());
  return false; // don't delete the when!
}

// Convert an aggregate type into a flat list of fields.
// This is used to flatten the aggregate memory datatype.
// Recursively populate the results with each ground type field.
static bool flattenType(FIRRTLType type, SmallVectorImpl<IntType> &results) {
  std::function<bool(FIRRTLType)> flatten = [&](FIRRTLType type) -> bool {
    return TypeSwitch<FIRRTLType, bool>(type)
        .Case<BundleType>([&](auto bundle) {
          for (auto &elt : bundle)
            if (!flatten(elt.type))
              return false;
          return true;
        })
        .Case<FVectorType>([&](auto vector) {
          for (size_t i = 0, e = vector.getNumElements(); i != e; ++i)
            if (!flatten(vector.getElementType()))
              return false;
          return true;
        })
        .Case<IntType>([&](auto iType) {
          results.push_back({iType});
          return iType.getWidth().hasValue();
        })
        .Default([&](auto) { return false; });
  };
  if (flatten(type))
    return true;
  return false;
}

/// Lower memory operations. A new memory is created for every leaf
/// element in a memory's data type.
bool TypeLoweringVisitor::visitDecl(MemOp op) {
  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;

  // MemOp should have ground types so we can't preserve aggregates.
  if (!peelType(op.getDataType(), fields, false))
    return false;

  SmallVector<MemOp> newMemories;
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
  // The vector of leaf elements type after flattening the data.
  SmallVector<IntType> flatMemType;
  // MaskGranularity : how many bits each mask bit controls.
  size_t maskGran = 1;
  // Total mask bitwidth after flattening.
  uint32_t totalmaskWidths = 0;
  // How many mask bits each field type requires.
  SmallVector<unsigned> maskWidths;

  auto hasSubAnno = [&]() -> bool {
    for (size_t portIdx = 0, e = op.getNumResults(); portIdx < e; ++portIdx)
      for (auto attr : op.getPortAnnotation(portIdx)) {
        if (auto subAnno = attr.dyn_cast<SubAnnotationAttr>())
          return true;
      }
    return false;
  };
  // If subannotations present on aggregate fields, we cannot flatten the
  // memory. It must be split into one memory per aggregate field.
  if (flattenAggregateMemData)
    if (hasSubAnno() || !flattenType(op.getDataType(), flatMemType))
      flattenAggregateMemData = false;

  if (flattenAggregateMemData) {
    SmallVector<Operation *, 8> flatData;
    SmallVector<int32_t> memWidths;
    // Get the width of individual aggregate leaf elements.
    for (auto f : flatMemType)
      memWidths.push_back(f.getWidth().getValue());

    maskGran = memWidths[0];
    size_t memFlatWidth = 0;
    // Compute the GCD of all data bitwidths.
    for (auto w : memWidths) {
      memFlatWidth += w;
      maskGran = llvm::GreatestCommonDivisor64(maskGran, w);
    }
    for (auto w : memWidths) {
      // How many mask bits required for each flattened field.
      auto mWidth = w / maskGran;
      maskWidths.push_back(mWidth);
      totalmaskWidths += mWidth;
    }
    // Now create a new memory of type flattened data.
    // ----------------------------------------------
    SmallVector<Type, 8> ports;
    SmallVector<Attribute, 8> portNames;

    // Create a new memoty data type of unsigned and computed width.
    auto flatType = UIntType::get(context, memFlatWidth);
    auto opPorts = op.getPorts();
    for (size_t portIdx = 0, e = opPorts.size(); portIdx < e; ++portIdx) {
      auto port = opPorts[portIdx];
      ports.push_back(MemOp::getTypeForPort(op.depth(), flatType, port.second,
                                            totalmaskWidths));
      portNames.push_back(port.first);
    }

    auto flatMem = builder->create<MemOp>(
        ports, op.readLatency(), op.writeLatency(), op.depth(), op.ruw(),
        portNames, op.name(), op.annotations().getValue(),
        op.portAnnotations().getValue(), op.inner_symAttr());
    // Done creating the memory.
    // ----------------------------------------------
    newMemories.push_back(flatMem);
  } else {
    // Memory for each field
    for (auto field : fields)
      newMemories.push_back(cloneMemWithNewType(builder, op, field));
  }
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
        if (flattenAggregateMemData) {
          // If memory was flattened instead of one memory per aggregate field.
          Value newField =
              getSubWhatever(newMemories[0].getResult(index), fieldIndex);
          Value realOldField = oldField;
          if (rType.getElement(fieldIndex).isFlip) {
            // Cast the memory read data from flat type to aggregate.
            newField = builder->createOrFold<BitCastOp>(
                oldField.getType().cast<FIRRTLType>(), newField);
            // Write the aggregate read data.
            builder->create<ConnectOp>(realOldField, newField);
          } else {
            // Cast the input aggregate write data to flat type.
            realOldField = builder->create<BitCastOp>(
                newField.getType().cast<FIRRTLType>(), oldField);
            // Mask bits require special handling, since some of the mask bits
            // need to be repeated, direct bitcasting wouldn't work. Depending
            // on the mask granularity, some mask bits will be repeated.
            if ((name == "mask" || name == "wmask") &&
                (maskWidths.size() < totalmaskWidths)) {
              Value catMasks;
              for (auto m : llvm::enumerate(maskWidths)) {
                // Get the mask bit.
                auto mBit = builder->createOrFold<BitsPrimOp>(
                    realOldField, m.index(), m.index());
                // Check how many times the mask bit needs to be prepend.
                for (size_t repeat = 0; repeat < m.value(); repeat++)
                  if (m.index() == 0 && repeat == 0)
                    catMasks = mBit;
                  else
                    catMasks = builder->createOrFold<CatPrimOp>(mBit, catMasks);
              }
              realOldField = catMasks;
            }
            // Now set the mask or write data.
            // Ensure that the types match.
            builder->create<ConnectOp>(
                newField,
                builder->createOrFold<BitCastOp>(
                    newField.getType().cast<FIRRTLType>(), realOldField));
          }
        } else {
          for (auto field : fields) {
            auto realOldField = getSubWhatever(oldField, field.index);
            auto newField = getSubWhatever(
                newMemories[field.index].getResult(index), fieldIndex);
            if (rType.getElement(fieldIndex).isFlip)
              std::swap(realOldField, newField);
            builder->create<ConnectOp>(newField, realOldField);
          }
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
  return true;
}

bool TypeLoweringVisitor::visitDecl(FExtModuleOp extModule) {
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
    if (attr.getName() != "portDirections" && attr.getName() != "portNames" &&
        attr.getName() != "portTypes" && attr.getName() != "portAnnotations" &&
        attr.getName() != "portSyms")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute> newArgNames;
  SmallVector<Attribute, 8> newPortTypes;
  SmallVector<Attribute, 8> newArgSyms;
  SmallVector<Attribute, 8> newArgAnnotations;

  for (auto &port : newArgs) {
    newArgDirections.push_back(port.direction);
    newArgNames.push_back(port.name);
    newPortTypes.push_back(TypeAttr::get(port.type));
    newArgSyms.push_back(port.sym ? port.sym : cache.sEmpty);
    newArgAnnotations.push_back(port.annotations.getArrayAttr());
  }

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortDirections,
                     direction::packAttribute(context, newArgDirections)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortNames, builder.getArrayAttr(newArgNames)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortTypes, builder.getArrayAttr(newPortTypes)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortSyms, builder.getArrayAttr(newArgSyms)));

  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortAnnotations, builder.getArrayAttr(newArgAnnotations)));

  // Update the module's attributes.
  extModule->setAttrs(newModuleAttrs);
  return false;
}

bool TypeLoweringVisitor::visitDecl(FModuleOp module) {
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
    if (attr.getName() != "portNames" && attr.getName() != "portDirections" &&
        attr.getName() != "portTypes" && attr.getName() != "portAnnotations" &&
        attr.getName() != "portSyms")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute> newArgNames;
  SmallVector<Attribute> newArgTypes;
  SmallVector<Attribute> newArgSyms;
  SmallVector<Attribute, 8> newArgAnnotations;
  for (auto &port : newArgs) {
    newArgDirections.push_back(port.direction);
    newArgNames.push_back(port.name);
    newArgTypes.push_back(TypeAttr::get(port.type));
    newArgSyms.push_back(port.sym ? port.sym : cache.sEmpty);
    newArgAnnotations.push_back(port.annotations.getArrayAttr());
  }

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortDirections,
                     direction::packAttribute(context, newArgDirections)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortNames, builder->getArrayAttr(newArgNames)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortTypes, builder->getArrayAttr(newArgTypes)));
  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortSyms, builder->getArrayAttr(newArgSyms)));
  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortAnnotations, builder->getArrayAttr(newArgAnnotations)));

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
  return false;
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
bool TypeLoweringVisitor::visitDecl(WireOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    return builder->create<WireOp>(field.type, "", attrs, StringAttr{});
  };
  return lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
bool TypeLoweringVisitor::visitDecl(RegOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    return builder->create<RegOp>(field.type, op.clockVal(), "", attrs,
                                  StringAttr{});
  };
  return lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
bool TypeLoweringVisitor::visitDecl(RegResetOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    auto resetVal = getSubWhatever(op.resetValue(), field.index);
    return builder->create<RegResetOp>(field.type, op.clockVal(),
                                       op.resetSignal(), resetVal, "", attrs,
                                       StringAttr{});
  };
  return lowerProducer(op, clone);
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
bool TypeLoweringVisitor::visitDecl(NodeOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    auto input = getSubWhatever(op.input(), field.index);
    return builder->create<NodeOp>(field.type, input, "", attrs, StringAttr{});
  };
  return lowerProducer(op, clone);
}

/// Lower an InvalidValue op with a bundle to multiple non-bundled InvalidOps.
bool TypeLoweringVisitor::visitExpr(InvalidValueOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    return builder->create<InvalidValueOp>(field.type);
  };
  return lowerProducer(op, clone);
}

// Expand muxes of aggregates
bool TypeLoweringVisitor::visitExpr(MuxPrimOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    auto high = getSubWhatever(op.high(), field.index);
    auto low = getSubWhatever(op.low(), field.index);
    return builder->create<MuxPrimOp>(op.sel(), high, low);
  };
  return lowerProducer(op, clone);
}

// Expand UnrealizedConversionCastOp of aggregates
bool TypeLoweringVisitor::visitExpr(mlir::UnrealizedConversionCastOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    auto input = getSubWhatever(op.getOperand(0), field.index);
    return builder->create<mlir::UnrealizedConversionCastOp>(field.type, input);
  };
  return lowerProducer(op, clone);
}

// Expand BitCastOp of aggregates
bool TypeLoweringVisitor::visitExpr(BitCastOp op) {
  Value srcLoweredVal = op.input();
  // If the input is of aggregate type, then cat all the leaf fields to form a
  // UInt type result. That is, first bitcast the aggregate type to a UInt.
  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;
  if (peelType(op.input().getType(), fields,
               /* allowedToPreserveAggregate */ false)) {
    size_t uptoBits = 0;
    // Loop over the leaf aggregates and concat each of them to get a UInt.
    // Bitcast the fields to handle nested aggregate types.
    for (auto field : llvm::enumerate(fields)) {
      auto fieldBitwidth = getBitWidth(field.value().type).getValue();
      // Ignore zero width fields, like empty bundles.
      if (fieldBitwidth == 0)
        continue;
      Value src = getSubWhatever(op.input(), field.index());
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
  } else {
    srcLoweredVal = builder->createOrFold<AsUIntPrimOp>(srcLoweredVal);
  }
  // Now the input has been cast to srcLoweredVal, which is of UInt type.
  // If the result is an aggregate type, then use lowerProducer.
  if (op.getResult().getType().isa<BundleType, FVectorType>()) {
    // uptoBits is used to keep track of the bits that have been extracted.
    size_t uptoBits = 0;
    auto clone = [&](FlatBundleFieldEntry field,
                     ArrayAttr attrs) -> Operation * {
      // All the fields must have valid bitwidth, a requirement for BitCastOp.
      auto fieldBits = getBitWidth(field.type).getValue();
      // If empty field, then it doesnot have any use, so replace it with an
      // invalid op, which should be trivially removed.
      if (fieldBits == 0)
        return builder->create<InvalidValueOp>(field.type);

      // Assign the field to the corresponding bits from the input.
      // Bitcast the field, incase its an aggregate type.
      auto extractBits = builder->create<BitsPrimOp>(
          srcLoweredVal, uptoBits + fieldBits - 1, uptoBits);
      uptoBits += fieldBits;
      return builder->create<BitCastOp>(field.type, extractBits);
    };
    return lowerProducer(op, clone);
  }

  // If ground type, then replace the result.
  if (op.getType().dyn_cast<SIntType>())
    srcLoweredVal = builder->create<AsSIntPrimOp>(srcLoweredVal);
  op.getResult().replaceAllUsesWith(srcLoweredVal);
  return true;
}

bool TypeLoweringVisitor::visitDecl(InstanceOp op) {
  bool skip = true;
  SmallVector<Type, 8> resultTypes;
  SmallVector<int64_t, 8> endFields; // Compressed sparse row encoding
  auto oldPortAnno = op.portAnnotations();
  SmallVector<Direction> newDirs;
  SmallVector<Attribute> newNames;
  SmallVector<Attribute> newPortAnno;
  bool allowedToPreserveAggregate =
      isModuleAllowedToPreserveAggregate(op.getReferencedModule(symTbl));

  endFields.push_back(0);
  bool needsSymbol = false;
  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    auto srcType = op.getType(i).cast<FIRRTLType>();

    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    if (!peelType(srcType, fieldTypes, allowedToPreserveAggregate)) {
      newDirs.push_back(op.getPortDirection(i));
      newNames.push_back(op.getPortName(i));
      resultTypes.push_back(srcType);
      newPortAnno.push_back(oldPortAnno[i]);
    } else {
      skip = false;
      auto oldName = op.getPortNameStr(i);
      auto oldDir = op.getPortDirection(i);
      // Store the flat type for the new bundle type.
      for (auto field : fieldTypes) {
        newDirs.push_back(direction::get((unsigned)oldDir ^ field.isOutput));
        newNames.push_back(builder->getStringAttr(oldName + field.suffix));
        resultTypes.push_back(field.type);
        auto annos = filterAnnotations(
            context, oldPortAnno[i].dyn_cast_or_null<ArrayAttr>(), srcType,
            field, needsSymbol, "");
        newPortAnno.push_back(annos);
      }
    }
    endFields.push_back(resultTypes.size());
  }

  if (skip)
    return false;
  auto sym = op.inner_symAttr();
  if (!sym || sym.getValue().empty())
    if (needsSymbol)
      sym = StringAttr::get(builder->getContext(),
                            "sym" + op.nameAttr().getValue());
  // FIXME: annotation update
  auto newInstance = builder->create<InstanceOp>(
      resultTypes, op.moduleNameAttr(), op.nameAttr(),
      direction::packAttribute(context, newDirs),
      builder->getArrayAttr(newNames), op.annotations(),
      builder->getArrayAttr(newPortAnno), op.lowerToBindAttr(), sym);

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
  return true;
}

bool TypeLoweringVisitor::visitExpr(SubaccessOp op) {
  auto input = op.input();
  auto vType = input.getType().cast<FVectorType>();

  // Check for empty vectors
  if (vType.getNumElements() == 0) {
    Value inv = builder->create<InvalidValueOp>(vType.getElementType());
    op.replaceAllUsesWith(inv);
    return true;
  }

  // Check for constant instances
  if (ConstantOp arg =
          dyn_cast_or_null<ConstantOp>(op.index().getDefiningOp())) {
    auto sio =
        builder->create<SubindexOp>(op.input(), arg.value().getExtValue());
    op.replaceAllUsesWith(sio.getResult());
    return true;
  }

  // Construct a multibit mux
  SmallVector<Value> inputs;
  inputs.reserve(vType.getNumElements());
  for (int index = vType.getNumElements() - 1; index >= 0; index--)
    inputs.push_back(builder->create<SubindexOp>(input, index));

  Value multibitMux = builder->create<MultibitMuxOp>(op.index(), inputs);
  op.replaceAllUsesWith(multibitMux);
  return true;
}

bool TypeLoweringVisitor::visitExpr(MultibitMuxOp op) {
  auto clone = [&](FlatBundleFieldEntry field, ArrayAttr attrs) -> Operation * {
    SmallVector<Value> newInputs;
    newInputs.reserve(op.inputs().size());
    for (auto input : op.inputs()) {
      auto inputSub = getSubWhatever(input, field.index);
      newInputs.push_back(inputSub);
    }
    return builder->create<MultibitMuxOp>(op.index(), newInputs);
  };
  return lowerProducer(op, clone);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerTypesPass : public LowerFIRRTLTypesBase<LowerTypesPass> {
  LowerTypesPass(bool flattenAggregateMemDataFlag, bool preserveAggregateFlag,
                 bool preservePublicTypesFlag) {
    flattenAggregateMemData = flattenAggregateMemDataFlag;
    preserveAggregate = preserveAggregateFlag;
    preservePublicTypes = preservePublicTypesFlag;
  }
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerTypesPass::runOnOperation() {
  std::vector<Operation *> ops;
  // Map of name of the NonLocalAnchor to the operation.
  DenseMap<StringAttr, Operation *> nlaMap;
  // Symbol Table
  SymbolTable symTbl(getOperation());
  // Cached attr
  AttrCache cache(&getContext());

  // Record all operations in the circuit.
  llvm::for_each(getOperation().getBody()->getOperations(), [&](Operation &op) {
    // Creating a map of all ops in the circt, but only modules are relevant.
    if (isa<FModuleOp, FExtModuleOp>(op))
      ops.push_back(&op);
    // Record the NonLocalAnchor and its name.
    if (auto nla = dyn_cast<NonLocalAnchor>(op))
      nlaMap[nla.sym_nameAttr()] = nla;
  });

  // Lower each module and return a list of Nlas which need to be updated with
  // the new symbol names.
  SmallVector<NlaNameNewSym> nlaToNewSymList;
  std::mutex nlaAppendLock;
  auto lowerModules = [&](auto op) {
    auto tl = TypeLoweringVisitor(&getContext(), flattenAggregateMemData,
                                  preserveAggregate, preservePublicTypes,
                                  symTbl, cache);
    tl.lowerModule(op);
    std::lock_guard<std::mutex> lg(nlaAppendLock);
    nlaToNewSymList.append(tl.getNLAs());
  };
  parallelForEach(&getContext(), ops.begin(), ops.end(), lowerModules);
  // Fixup the nla, with the updated symbol names.
  // This can only update the final element on which the nla is applied, because
  // lowering cannot update the symbols on the InstanceOp. Also, the final
  // element cannot be a module.
  for (auto nlaToSym : nlaToNewSymList) {
    // Get the nla with the corresponding name. This is guaranteed to exist.
    auto nla = cast<NonLocalAnchor>(nlaMap[nlaToSym.nlaName]);
    auto namepath = nla.namepath();
    SmallVector<Attribute> updatedPath(namepath.begin(), namepath.end());
    // Update the final element, which must be an InnerRefAttr.
    auto leaf = namepath[namepath.size() - 1].cast<hw::InnerRefAttr>();
    updatedPath[namepath.size() - 1] =
        hw::InnerRefAttr::get(&getContext(), leaf.getModule(), nlaToSym.newSym);
    nla.namepathAttr(ArrayAttr::get(&getContext(), updatedPath));
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLTypesPass(
    bool replSeqMem, bool preserveAggregate, bool preservePublicTypes) {

  return std::make_unique<LowerTypesPass>(replSeqMem, preserveAggregate,
                                          preservePublicTypes);
}
