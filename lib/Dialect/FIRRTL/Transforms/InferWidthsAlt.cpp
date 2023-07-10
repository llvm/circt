//===- InferWidthsAlt.cpp - Infer width of types ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an alternate InferWidths pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "infer-widths-alt"

using mlir::InferTypeOpInterface;
using mlir::WalkOrder;

using namespace circt;
using namespace firrtl;

/// Resize a `uint`, `sint`, or `analog` type to a specific width.
static FIRRTLBaseType resizeType(FIRRTLBaseType type, uint32_t newWidth) {
  auto *context = type.getContext();
  return TypeSwitch<FIRRTLBaseType, FIRRTLBaseType>(type)
      .Case<UIntType>([&](auto type) {
        return UIntType::get(context, newWidth, type.isConst());
      })
      .Case<SIntType>([&](auto type) {
        return SIntType::get(context, newWidth, type.isConst());
      })
      .Case<AnalogType>([&](auto type) {
        return AnalogType::get(context, newWidth, type.isConst());
      })
      .Default([&](auto type) { return type; });
}

/// Check if a type contains any FIRRTL type with uninferred widths.
static bool hasUninferredWidth(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<FIRRTLBaseType>([](auto base) { return base.hasUninferredWidth(); })
      .Case<RefType>(
          [](auto ref) { return ref.getType().hasUninferredWidth(); })
      .Default([](auto) { return false; });
}

// Implementation of getUIninferred
void findUninferred(SmallVectorImpl<unsigned> &fields, Type type,
                    unsigned offset) {
  llvm::errs() << "Type visit " << type << " at " << offset << "\n";
  TypeSwitch<Type, void>(type)
      .Case<IntType>([&](auto t) {
        if (t.hasUninferredWidth())
          fields.push_back(offset);
      })
      .Case<AnalogType>([&](auto t) {
        if (t.hasUninferredWidth())
          fields.push_back(offset);
      })
      .Case<BundleType>([&](auto t) {
        for (auto [idx, element] : llvm::enumerate(t.getElements())) {
          findUninferred(fields, element.type, offset + t.getFieldID(idx));
        }
      })
      .Case<FVectorType>([&](auto t) {
        findUninferred(fields, t.getElementType(), offset + 1);
      })
      .Case<FEnumType>([&](auto t) {
        for (auto [idx, element] : llvm::enumerate(t.getElements())) {
          findUninferred(fields, element.type, offset + t.getFieldID(idx));
        }
      });
}

static SmallVector<unsigned, 4> getUninferred(Type t) {
  SmallVector<unsigned, 4> fields;
  findUninferred(fields, t, 0);
  return fields;
}

static SmallVector<unsigned, 4> getUninferred(Type t1, Type t2) {
  SmallVector<unsigned, 4> fields;
  findUninferred(fields, t1, 0);
  findUninferred(fields, t2, 0);
  std::sort(fields.begin(), fields.end());
  fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
  return fields;
}

static SmallVector<unsigned, 4> getUninferred(Value v) {
  return getUninferred(v.getType());
}

static SmallVector<unsigned, 4> getUninferred(Value v1, Value v2) {
  return getUninferred(v1.getType(), v2.getType());
}

namespace {

// RefArg (Field of argument relative field), RefStorage (source of value) ->
// changed
struct InferWidth : public FIRRTLVisitor<InferWidth, void, FieldRef> {

  InferWidth(SymbolTable &symtbl) : symtbl(symtbl) {}

  void prepare(FModuleOp fmod);
  void solve(FModuleOp fmod);
  void apply(FModuleOp fmod);

  void addSource(FieldRef ref, unsigned width = 0);
  void addAlias(FieldRef dest, FieldRef src);

  bool updateProposal(FieldRef ref, unsigned width);
  bool updateProposalOrAlias(FieldRef ref, unsigned width);

  std::tuple<bool, unsigned> getWidth(FieldRef);
  Type makeTypeForValue(Value v);
  Type updateBaseType(Value v, Type type, unsigned fieldID);

  void queueUsers(FieldRef refArg);

  using FIRRTLVisitor<InferWidth, bool, FieldRef>::visitExpr;
  using FIRRTLVisitor<InferWidth, bool, FieldRef>::visitStmt;
  using FIRRTLVisitor<InferWidth, bool, FieldRef>::visitDecl;

  void visitExpr(ConstantOp op, FieldRef);
  //   bool visitExpr(SpecialConstantOp op);
  //   bool visitExpr(AggregateConstantOp op);

  bool visitExpr(SubfieldOp op, FieldRef);
  bool visitExpr(SubindexOp op, FieldRef);
  bool visitExpr(SubaccessOp op, FieldRef);

  void lowerAutoInfer(Operation* op, FieldRef ref);
  void visitExpr(AddPrimOp op, FieldRef ref) { lowerAutoInfer(op, ref); }

  bool lowerNoopMux(Operation *op, FieldRef);
  bool visitExpr(MuxPrimOp op, FieldRef ref) { return lowerNoopMux(op, ref); }
  bool visitExpr(Mux2CellIntrinsicOp op, FieldRef ref) {
    return lowerNoopMux(op, ref);
  }
  bool visitExpr(Mux4CellIntrinsicOp op, FieldRef ref) {
    return lowerNoopMux(op, ref);
  }

  bool lowerNoopCast(Operation *op, FieldRef);
  bool visitExpr(AsSIntPrimOp op, FieldRef ref) {
    return lowerNoopCast(op, ref);
  }
  bool visitExpr(AsUIntPrimOp op, FieldRef ref) {
    return lowerNoopCast(op, ref);
  }
  bool visitExpr(DShlwPrimOp op, FieldRef);
  bool visitExpr(DShlPrimOp op, FieldRef);
  bool visitExpr(ShlPrimOp op, FieldRef);
  bool visitExpr(DShrPrimOp op, FieldRef);
  bool visitExpr(ShrPrimOp op, FieldRef);

  bool visitStmt(ConnectOp op, FieldRef);
  bool visitStmt(AttachOp op, FieldRef);

  bool visitDecl(WireOp op, FieldRef);
  bool visitDecl(NodeOp op, FieldRef ref) { return lowerNoopCast(op, ref); }

  bool visitUnhandledOp(Operation *op, FieldRef) { return false; }

  DenseMap<FieldRef, unsigned> proposals;

  // Unification tracking
  // Result -> Source
  DenseMap<FieldRef, FieldRef> sources;

  // Value updated, Defining source
  std::deque<std::tuple<Operation *, FieldRef>> worklist;
  SymbolTable &symtbl;
};
} // namespace

Type InferWidth::updateBaseType(Value v, Type type, unsigned fieldID) {
  if (isa<RefType>(type))
    return type;
  if (isa<StringType, BigIntType, ListType, MapType>(type))
    return type;
  FIRRTLBaseType fbt = cast<FIRRTLBaseType>(type);
  auto width = fbt.getBitWidthOrSentinel();
  if (width >= 0) {
    // Known width integers return themselves.
    return fbt;
  } else if (width == -1) {
    assert(proposals.count(FieldRef(v, fieldID)));
    return resizeType(fbt, proposals[FieldRef(v, fieldID)]);
  } else if (auto bundleType = dyn_cast<BundleType>(fbt)) {
    // Bundle types recursively update all bundle elements.
    llvm::SmallVector<BundleType::BundleElement, 3> elements;
    bool changed = false;
    for (auto [idx, element] : llvm::enumerate(bundleType)) {
      auto updatedBase =
          updateBaseType(v, element.type, fieldID + bundleType.getFieldID(idx));
      elements.emplace_back(element.name, element.isFlip,
                            cast<FIRRTLBaseType>(updatedBase));
      changed |= (updatedBase != element.type);
    }
    if (changed)
      return BundleType::get(bundleType.getContext(), elements,
                             bundleType.isConst());
    return bundleType;
  } else if (auto vecType = dyn_cast<FVectorType>(fbt)) {
    auto updatedBase = updateBaseType(v, vecType.getElementType(), fieldID + 1);
    if (updatedBase != vecType.getElementType())
      return FVectorType::get(cast<FIRRTLBaseType>(updatedBase),
                              vecType.getNumElements(), vecType.isConst());
    return vecType;
  } else if (auto enumType = dyn_cast<FEnumType>(fbt)) {
    llvm::SmallVector<FEnumType::EnumElement> elements;
    bool changed = false;
    for (auto [idx, element] : llvm::enumerate(enumType.getElements())) {
      auto updatedBase =
          updateBaseType(v, element.type, enumType.getFieldID(idx));
      changed |= (updatedBase != element.type);
      elements.emplace_back(element.name, cast<FIRRTLBaseType>(updatedBase));
    }
    if (changed)
      return FEnumType::get(enumType.getContext(), elements,
                            enumType.isConst());
    return enumType;
  }
  llvm_unreachable("Unknown type inside a bundle!");
}

Type InferWidth::makeTypeForValue(Value v) {
  return updateBaseType(v, v.getType(), 0);
}

void InferWidth::prepare(FModuleOp fmod) {

  // Scan block arguments for uninferred
  size_t firrtlArg = 0;
  for (auto port : fmod.getPorts()) {
    auto arg = fmod.getBody().getArgument(firrtlArg++);
    for (auto uninferred : getUninferred(arg))
      addSource({arg, uninferred});
  }

  // Initially scan for sources of uninferred
  for (auto &op : fmod.getBodyBlock()->getOperations()) {
    bool hasUninferred = false;
    for (auto result : op.getResults())
      hasUninferred |= hasUninferredWidth(result.getType());
    if (!hasUninferred)
      continue;
    if (isa<WireOp, InvalidValueOp>(op)) {
      for (auto uninferred : getUninferred(op.getResult(0)))
        addSource({op.getResult(0), uninferred});
    } else if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto refdModule = inst.getReferencedModule(symtbl);
      auto module = dyn_cast<FModuleOp>(&*refdModule);
      if (!module) {
        mlir::emitError(inst.getLoc())
            << "extern module has ports of uninferred width";
        continue;
      }
      for (auto it : llvm::zip(inst.getResults(), module.getArguments())) {
        for (auto uninferred : getUninferred(std::get<0>(it))) {
          FieldRef ref(std::get<0>(it), uninferred);
          FieldRef src(std::get<1>(it), uninferred);
          addAlias(ref, src);
        }
      }
    } else if (auto cst = dyn_cast<ConstantOp>(op)) {
    }
  }
}

void InferWidth::solve(FModuleOp fmod) {
  // Iterate on the worklist
  while (!worklist.empty()) {
    auto [op, refArg] = worklist.front();
    worklist.pop_front();
    if (dispatchVisitor(op, refArg)) {
      llvm::errs() << "Changed\n";
    }
  }
}

void InferWidth::apply(FModuleOp fmod) {
  // Update Ports
  // Update the block argument types.
  SmallVector<Attribute> argTypes;
  argTypes.reserve(fmod.getNumPorts());
  bool argsChanged = false;
  for (auto arg : fmod.getArguments()) {
    auto result = makeTypeForValue(arg);
    if (arg.getType() != result) {
      arg.setType(result);
      argsChanged = true;
    }
    argTypes.push_back(TypeAttr::get(result));
  }

  // Update the module function type if needed.
  if (argsChanged) {
    fmod->setAttr(FModuleLike::getPortTypesAttrName(),
                  ArrayAttr::get(fmod.getContext(), argTypes));
  }

  // Update Operations
  for (auto p : proposals) {
    llvm::errs() << "Set [" << p.first.getValue()
                 << "]:" << p.first.getFieldID() << " to " << p.second << "\n";
    // If this is an operation that does not generate any free variables that
    // are determined during width inference, simply update the value type based
    // on the operation arguments.
    if (auto op = dyn_cast_or_null<InferTypeOpInterface>(
            p.first.getValue().getDefiningOp())) {
      SmallVector<Type, 2> types;
      auto res = op.inferReturnTypes(op->getContext(), op->getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getPropertiesStorage(),
                                     op->getRegions(), types);
      assert(succeeded(res));
      //    if (failed(res))
      //      return failure();

      assert(types.size() == op->getNumResults());
      for (auto it : llvm::zip(op->getResults(), types)) {
        LLVM_DEBUG(llvm::dbgs() << "Inferring " << std::get<0>(it) << " as "
                                << std::get<1>(it) << "\n");
        std::get<0>(it).setType(std::get<1>(it));
      }
      //  return true;
      continue;
    } else if(auto inv = dyn_cast_or_null<InvalidValueOp>(p.first.getValue().getDefiningOp())) {
      auto result = makeTypeForValue(inv);
      inv.getResult().setType(cast<FIRRTLBaseType>(result));
    } else if (auto wire = dyn_cast_or_null<WireOp>(p.first.getValue().getDefiningOp())) {
      auto result = makeTypeForValue(wire.getResult());
      wire.getResult().setType(result);
    }
  }
}

void InferWidth::addSource(FieldRef ref, unsigned width) {
  llvm::errs() << "Source: [" << ref.getValue() << "]:" << ref.getFieldID() << "\n";
  sources[ref] = ref;
  updateProposal(ref, width);
}

  void InferWidth::addAlias(FieldRef dest, FieldRef src) {
    assert(sources.count(src));
    auto realSrc = sources[src];
    sources[dest] = realSrc;
    queueUsers(dest);
    llvm::errs() << "Alias: [" << dest.getValue() << "]:" << dest.getFieldID() << " [" << src.getValue() << "]:" << src.getFieldID() << " [" << realSrc.getValue() << "]:" << realSrc.getFieldID() << "\n";
  }

bool InferWidth::updateProposal(FieldRef ref, unsigned width) {
  auto it = proposals.find(ref);
  if (it == proposals.end()) {
    proposals[ref] = width;
    queueUsers(ref);
    return true;
  }
  if (it->second == width)
    return false;
  assert(width > it->second);
  it->second = width;
  llvm::errs() << "Updated [" << ref.getValue() << "]:" << ref.getFieldID()
               << " to " << width << "\n";
  queueUsers(ref);
  return true;
}

bool InferWidth::updateProposalOrAlias(FieldRef ref, unsigned width) {
  assert(sources.count(ref));
  auto src = sources[ref];
  auto &val = proposals[src];
  if (val == width)
    return false;
  assert(width > val);
  val = width;
  llvm::errs() << "Updated [" << src.getValue() << "]:" << src.getFieldID()
               << " to " << width << "\n";
  queueUsers(src);
  return true;
}


std::tuple<bool, unsigned> InferWidth::getWidth(FieldRef ref) {
  auto type =
      cast<FIRRTLBaseType>(cast<FIRRTLBaseType>(ref.getValue().getType())
                               .getFinalTypeByFieldID(ref.getFieldID()));
  auto bw = type.getBitWidthOrSentinel();
  if (bw >= 0)
    return std::make_tuple(true, bw);
  auto src = sources.find(ref);
  if (src == sources.end())
    return std::make_tuple(false, proposals[ref]);
  return std::make_tuple(false, proposals[src->second]);
}

// Val is the result that is being used, ref is the originating definition.
void InferWidth::queueUsers(FieldRef refArg) {
  for (auto user : refArg.getValue().getUsers())
    worklist.emplace_back(user, refArg);
}

bool InferWidth::visitExpr(SubfieldOp op, FieldRef refArg) {
  auto bundleType = op.getInput().getType();
  auto [childField, isvalid] =
      bundleType.rootChildFieldID(refArg.getFieldID(), op.getFieldIndex());
  llvm::errs() << "Subfield source type: " << bundleType << " idx: " << op.getFieldIndex() << " fieldID: "
                << " " << refArg.getFieldID() << " childFieldID: "
               << childField << " isvalid: " << isvalid << "\n";
  if (!isvalid)
    return false;
  addAlias({op, (unsigned)childField}, refArg);
  return false;
}

void InferWidth::visitExpr(ConstantOp cst, FieldRef refArg) {
        // Completely resolve constants
      auto v = cst.getValue();
      auto w = v.getBitWidth() -
               (v.isNegative() ? v.countLeadingOnes() : v.countLeadingZeros());
      if (v.isSigned())
        w += 1;
      if (v.getBitWidth() > unsigned(w))
        v = v.trunc(w);
      // Go ahead and pre-update constants then we don't have to track them.
      cst.setValue(v);
      cst.getResult().setType(cast<IntType>(resizeType(cst.getType(), w)));
      queueUsers({cst, 0});
}

// collapse vector types
bool InferWidth::visitExpr(SubindexOp op, FieldRef refArg) {
  auto vectorType = op.getInput().getType();
  auto [childField, isvalid] =
      vectorType.rootChildFieldID(refArg.getFieldID(), 0);
  llvm::errs() << "Subindex source type: " << vectorType <<  " fieldID: "
               << refArg.getFieldID() << " childFieldID: " << childField << " isvalid: " << isvalid
               << "\n";
  assert(isvalid);
  if (!isvalid)
    return false;
  addAlias({op, (unsigned)childField}, refArg);
  return false;
}

// collapse vector types
bool InferWidth::visitExpr(SubaccessOp op, FieldRef refArg) {
  auto vectorType = op.getInput().getType();
  auto [childField, isvalid] =
      vectorType.rootChildFieldID(refArg.getFieldID(), 0);
  llvm::errs() << "Subaccess " << vectorType << " " << refArg.getValue() << " "
               << refArg.getFieldID() << " " << childField << " " << isvalid
               << "\n";
  assert(isvalid);
  if (!isvalid)
    return false;
  addAlias({op, (unsigned)childField}, refArg);
  return false;
}

void InferWidth::lowerAutoInfer(Operation* op, FieldRef ref) {
  llvm::err() << "Auto " << *op << "\n";
      auto inf = dyn_cast<InferTypeOpInterface>(op);
      SmallVector<Type, 2> types;
      auto res = op.inferReturnTypes(op->getContext(), op->getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getPropertiesStorage(),
                                     op->getRegions(), types);
      assert(succeeded(res));
      //    if (failed(res))
      //      return failure();

      assert(types.size() == op->getNumResults());
      for (auto it : llvm::zip(op->getResults(), types)) {
        LLVM_DEBUG(llvm::dbgs() << "Inferring " << std::get<0>(it) << " as "
                                << std::get<1>(it) << "\n");
        std::get<0>(it).setType(std::get<1>(it));
      }
}

bool InferWidth::visitExpr(AddPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op.getOperand(0), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op.getOperand(1), 0));
  auto w = 1 + std::max(src1Width, src2Width);
  llvm::errs() << "Add " << op << " of " << src1Width << "," << src2Width
               << "->" << w << "\n";
  return updateProposal({op.getResult(), 0}, w);
}

bool InferWidth::visitExpr(DShlwPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getLhs(), 0));
  auto w = srcWidth;
  llvm::errs() << "DShlw " << op << " of " << srcWidth << "->" << w << "\n";
  return updateProposal({op.getResult(), 0}, w);
}

bool InferWidth::visitExpr(DShrPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getLhs(), 0));
  auto w = srcWidth;
  llvm::errs() << "DShr " << op << " of " << srcWidth << "->" << w << "\n";
  return updateProposal({op.getResult(), 0}, w);
}

bool InferWidth::visitExpr(DShlPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op.getLhs(), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op.getRhs(), 0));
  auto w = src1Width + (1 << src2Width) - 1;
  llvm::errs() << "DShl " << op << " of " << src1Width << "," << src2Width
               << "->" << w << "\n";
  return updateProposal({op.getResult(), 0}, w);
}

bool InferWidth::visitExpr(ShlPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = srcWidth + op.getAmount();
  llvm::errs() << "Shr " << op << " of " << srcWidth << "," << op.getAmount()
               << "->" << w << "\n";
  return updateProposal({op.getResult(), 0}, w);
}

bool InferWidth::visitExpr(ShrPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = std::max(1U, srcWidth + op.getAmount());
  llvm::errs() << "Shr " << op << " of " << srcWidth << "," << op.getAmount()
               << "->" << w << "\n";
  return updateProposal({op.getResult(), 0}, w);
}

bool InferWidth::lowerNoopMux(Operation *op, FieldRef refArg) {
  if (!cast<FIRRTLBaseType>(op->getResult(0).getType()).hasUninferredWidth())
    return false;
  unsigned w = 0;
  for (unsigned i = 1; i < op->getNumOperands(); ++i)
    w = std::max(w, std::get<1>(getWidth(
                        FieldRef(op->getOperand(i), refArg.getFieldID()))));

  llvm::errs() << "Mux " << op << " ->" << w << "\n";
  return updateProposal({op->getResult(0), refArg.getFieldID()}, w);
}

bool InferWidth::lowerNoopCast(Operation *op, FieldRef refArg) {
  assert(cast<FIRRTLBaseType>(op->getResult(0).getType()).hasUninferredWidth());
  addAlias({op->getResult(0), refArg.getFieldID()}, refArg);
  return updateProposal({op->getResult(0), refArg.getFieldID()}, std::get<1>(getWidth(refArg)));
}

bool InferWidth::visitStmt(ConnectOp op, FieldRef refArg) {
  FieldRef destRef = FieldRef(op.getDest(), refArg.getFieldID());
  FieldRef srcRef = FieldRef(op.getSrc(), refArg.getFieldID());
  auto [dstFixed, dstWidth] =
      getWidth(destRef);
  auto [srcFixed, srcWidth] =
      getWidth(srcRef);
  llvm::errs() << op << "\nConnect field " << refArg.getFieldID() 
               << " src:" << srcFixed << "," << srcWidth << "->d:" << dstFixed
               << "," << dstWidth << "\n";
  bool changed = false;
  // No truncation!
  if (!dstFixed && dstWidth < srcWidth)
    changed |= updateProposalOrAlias(destRef, srcWidth);
  return changed;
}

bool InferWidth::visitStmt(AttachOp op, FieldRef refArg) {
  unsigned w = 0;
  for (auto operand : op.getAttached()) {
    w = std::max(w,
                 std::get<1>(getWidth(FieldRef(operand, refArg.getFieldID()))));
  }
  llvm::errs() << "Attach field " << refArg.getFieldID() << " -> " << w << "\n";

  bool changed = false;
  for (auto operand : op.getAttached()) {
    FieldRef ref(operand, refArg.getFieldID());
    auto [isfixed, oldW] = getWidth(FieldRef(operand, refArg.getFieldID()));
    if (oldW != w) {
      if (isfixed)
        abort();
      assert(sources.count(ref));
      auto src = sources[ref];
      changed |= updateProposalOrAlias(src, w);
    }
  }
  return changed;
}

// If this got in the worklist, it was from a connect, so just trigger children
bool InferWidth::visitDecl(WireOp op, FieldRef refArg) { return true; }

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InferWidthsAltPass : public InferWidthsAltBase<InferWidthsAltPass> {
  void runOnOperation() override;
};
} // namespace

void InferWidthsAltPass::runOnOperation() {
  CircuitOp circuit = getOperation();
  SymbolTable symtbl(circuit);
  llvm::errs() << "Starting " << circuit.getName() << "\n";
  InferWidth iw(symtbl);
  for (auto op : circuit.getOps<FModuleOp>()) {
    llvm::errs() << "** prepare: " << op.getModuleName() << " **\n";
    op.dump();
    iw.prepare(op);
    llvm::errs() << "\n";
  }
  for (auto op : circuit.getOps<FModuleOp>()) {
    llvm::errs() << "** solve: " << op.getModuleName() << " **\n";
    op.dump();
    iw.solve(op);
    llvm::errs() << "\n";
  }
  for (auto op : circuit.getOps<FModuleOp>()) {
    llvm::errs() << "** apply: " << op.getModuleName() << " **\n";
    op.dump();
    iw.apply(op);
    llvm::errs() << "\n";
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsAltPass() {
  return std::make_unique<InferWidthsAltPass>();
}
