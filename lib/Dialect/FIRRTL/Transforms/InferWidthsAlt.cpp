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
#include "mlir/IR/AsmState.h"
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
  TypeSwitch<Type, void>(type)
      .Case<RefType>(
          [&](auto t) { findUninferred(fields, t.getType(), offset + 1); })
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
  void solve();
  void apply(FModuleOp fmod);

  void addSource(FieldRef ref);
  void addAlias(FieldRef dest, FieldRef src);

  void updateProposal(FieldRef ref, unsigned width);
  void updateProposalOrAlias(FieldRef ref, unsigned width);

  std::tuple<bool, unsigned> getWidth(FieldRef);
  Type makeTypeForValue(Value v);
  Type updateBaseType(Value v, Type type, unsigned fieldID);

  void queueUsers(FieldRef refArg);

  using FIRRTLVisitor<InferWidth, void, FieldRef>::visitExpr;
  using FIRRTLVisitor<InferWidth, void, FieldRef>::visitStmt;
  using FIRRTLVisitor<InferWidth, void, FieldRef>::visitDecl;

  void unify(Operation *op, int destAdj, FieldRef refArg, int srcAdj) {
    addAlias({op->getResult(0), refArg.getFieldID() + destAdj},
             {refArg.getValue(), refArg.getFieldID() + srcAdj});
  }

  void visitExpr(ConstantOp op, FieldRef);
  //   bool visitExpr(SpecialConstantOp op);
  //   bool visitExpr(AggregateConstantOp op);

  void visitExpr(SubfieldOp op, FieldRef);
  void visitExpr(SubindexOp op, FieldRef);
  void visitExpr(SubaccessOp op, FieldRef);
  void visitExpr(SubtagOp op, FieldRef);

  void visitBinarySumExpr(Operation *op, FieldRef ref);
  void visitExpr(CatPrimOp op, FieldRef ref) { visitBinarySumExpr(op, ref); }

  void visitArithExpr(Operation *op, FieldRef);
  void visitExpr(AddPrimOp op, FieldRef ref) { visitArithExpr(op, ref); }
  void visitExpr(SubPrimOp op, FieldRef ref) { visitArithExpr(op, ref); }
  void visitExpr(MulPrimOp op, FieldRef ref) { visitBinarySumExpr(op, ref); }
  void visitExpr(DivPrimOp op, FieldRef);
  void visitExpr(RemPrimOp op, FieldRef);

  void visitLogicExpr(Operation *op, FieldRef ref);
  void visitExpr(AndPrimOp op, FieldRef ref) { visitLogicExpr(op, ref); }
  void visitExpr(OrPrimOp op, FieldRef ref) { visitLogicExpr(op, ref); }
  void visitExpr(XorPrimOp op, FieldRef ref) { visitLogicExpr(op, ref); }

  void visitExpr(NegPrimOp op, FieldRef);

  void visitExpr(TailPrimOp op, FieldRef);
  void visitExpr(PadPrimOp op, FieldRef);

  void lowerNoopMux(Operation *op, FieldRef);
  void visitExpr(MuxPrimOp op, FieldRef ref) { lowerNoopMux(op, ref); }
  void visitExpr(Mux2CellIntrinsicOp op, FieldRef ref) {
    lowerNoopMux(op, ref);
  }
  void visitExpr(Mux4CellIntrinsicOp op, FieldRef ref) {
    lowerNoopMux(op, ref);
  }

  void lowerNoopCast(Operation *op, FieldRef);
  void visitExpr(AsSIntPrimOp op, FieldRef ref) { lowerNoopCast(op, ref); }
  void visitExpr(AsUIntPrimOp op, FieldRef ref) { lowerNoopCast(op, ref); }
  void visitExpr(CvtPrimOp op, FieldRef);

  void visitExpr(NotPrimOp op, FieldRef ref) { lowerNoopCast(op, ref); }

  void visitExpr(DShlwPrimOp op, FieldRef);
  void visitExpr(DShlPrimOp op, FieldRef);
  void visitExpr(ShlPrimOp op, FieldRef);
  void visitExpr(DShrPrimOp op, FieldRef);
  void visitExpr(ShrPrimOp op, FieldRef);

  void visitStmt(ConnectOp op, FieldRef);
  void visitStmt(AttachOp op, FieldRef);
  void visitStmt(RefDefineOp op, FieldRef);
  void visitExpr(RefCastOp op, FieldRef);
  void visitExpr(RefSendOp op, FieldRef);
  void visitExpr(RefResolveOp op, FieldRef);
  void visitExpr(RefSubOp op, FieldRef);

  void visitDecl(WireOp op, FieldRef);
  void visitDecl(RegOp op, FieldRef);
  void visitDecl(RegResetOp op, FieldRef);
  void visitDecl(NodeOp op, FieldRef ref) { lowerNoopCast(op, ref); }

  void visitDecl(MemOp op, FieldRef);

  void visitUnhandledOp(Operation *op, FieldRef) {
    llvm::errs() << "Unhandled ";
    op->dump();
    llvm::errs() << "\n";
  }

  DenseMap<FieldRef, unsigned> proposals;

  // Unification tracking
  // Result -> Source
  DenseMap<FieldRef, FieldRef> sources;

  // map from source to aliases.  Maintained by union
  DenseMap<FieldRef, SmallVector<FieldRef>> invSources;

  // Keep track of aliased values so we can be sure to update their ops
  DenseSet<Value> aliased;

  // Blockarg -> instance results
  // Note, instance results are marked as aliased to the block arts in `sources`
  // so a connect to an instance result will update the block arg.  This is to
  // know what to iterate on when that update happens.
  DenseMap<FieldRef, SmallVector<FieldRef>> moduleParams;

  // Value updated, Defining source
  std::deque<std::tuple<Operation *, FieldRef>> worklist;
  SymbolTable &symtbl;
};
} // namespace

Type InferWidth::updateBaseType(Value v, Type type, unsigned fieldID) {
  if (auto refType = dyn_cast<RefType>(type)) {
    auto type = updateBaseType(v, refType.getType(), fieldID + 1);
    if (type == refType.getType())
      return refType;
    return RefType::get(cast<FIRRTLBaseType>(type), refType.getForceable());
  }
  if (isa<StringType, BigIntType, ListType, MapType>(type))
    return type;
  FIRRTLBaseType fbt = cast<FIRRTLBaseType>(type);
  auto width = fbt.getBitWidthOrSentinel();
  if (width >= 0) {
    // Known width integers return themselves.
    return fbt;
  } else if (width == -1) {
    FieldRef ref(v, fieldID);
    auto it = sources.find(ref);
    if (it != sources.end())
      ref = it->second;
    assert(proposals.count(ref));
    return resizeType(fbt, proposals[ref]);
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
    if (isa<WireOp, InvalidValueOp, RegOp, RegResetOp>(op)) {
      bool forceable = op.getNumResults() == 2;
      for (auto uninferred : getUninferred(op.getResult(0))) {
        addSource({op.getResult(0), uninferred});
        if (forceable)
          addAlias({op.getResult(1), uninferred + 1},
                   {op.getResult(0), uninferred});
      }
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
          moduleParams[src].push_back(ref);
        }
      }
    } else if (auto cst = dyn_cast<ConstantOp>(op)) {
      visitExpr(cst, FieldRef());
    } else if (auto mem = dyn_cast<MemOp>(op)) {
      // Assume there is at least one valid port
      unsigned nonDebugPort = 0;
      while (isa<RefType>(mem.getResult(nonDebugPort).getType()))
        ++nonDebugPort;

      // A helper function that returns the indeces of the "data", "rdata",
      // and "wdata" fields in the bundle corresponding to a memory port.
      auto dataFieldIndices = [](MemOp::PortKind kind) -> ArrayRef<unsigned> {
        static const unsigned indices[] = {3, 5};
        static const unsigned debug[] = {0};
        switch (kind) {
        case MemOp::PortKind::Read:
        case MemOp::PortKind::Write:
          return ArrayRef<unsigned>(indices, 1); // {3}
        case MemOp::PortKind::ReadWrite:
          return ArrayRef<unsigned>(indices); // {3, 5}
        case MemOp::PortKind::Debug:
          return ArrayRef<unsigned>(debug);
        }
        llvm_unreachable("Imposible PortKind");
      };

      // This creates independent variables for every data port. Yet, what we
      // actually want is for all data ports to share the same variable. To do
      // this, we find the first data port declared, and use that port's vars
      // for all the other ports.
      unsigned firstFieldIndex =
          dataFieldIndices(mem.getPortKind(nonDebugPort))[0];
      auto memtype = mem.getPortType(nonDebugPort)
                         .getPassiveType()
                         .template cast<BundleType>()
                         .getElementType(firstFieldIndex);
      auto uninferred = getUninferred(memtype);
      FieldRef firstData(mem.getResult(nonDebugPort),
                         mem.getPortType(nonDebugPort)
                             .getPassiveType()
                             .template cast<BundleType>()
                             .getFieldID(firstFieldIndex));
      for (auto un : uninferred)
        addSource(firstData.getSubField(un));

      // Reuse data port variables.
      for (unsigned i = 0, e = mem.getNumResults(); i < e; ++i) {
        auto result = mem.getResult(i);
        if (isa<RefType>(result.getType())) {
          // Debug ports are firrtl.ref<vector<data-type, depth>>
          // Use FieldRef of 2, to indicate the first vector element must be
          // of the dataType.
          for (auto un : uninferred)
            addAlias({result, 2 + un}, firstData.getSubField(un));
          continue;
        }
        auto portType = cast<BundleType>(mem.getPortType(i).getPassiveType());
        for (auto fieldIndex : dataFieldIndices(mem.getPortKind(i)))
          if (i != nonDebugPort || fieldIndex != firstFieldIndex)
            for (auto un : uninferred)
              addAlias({result, (unsigned)portType.getFieldID(fieldIndex) + un},
                       firstData.getSubField(un));
      }
    }
  }
}

void InferWidth::solve() {
  // Iterate on the worklist
  while (!worklist.empty()) {
    auto [op, refArg] = worklist.front();
    worklist.pop_front();
    dispatchVisitor(op, refArg);
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

  auto updateValue = [&](Value v) {
    if (isa<BlockArgument>(v))
      return;
    Operation *op = v.getDefiningOp();
    for (auto r : op->getResults())
      if (r == v)
        r.setType(makeTypeForValue(v));
  };

  // Update Operations
  DenseSet<Value> seen;
  for (auto p : proposals) {
    if (seen.count(p.first.getValue()))
      continue;
    updateValue(p.first.getValue());
  }
  for (auto v : aliased) {
    if (seen.count(v))
      continue;
    updateValue(v);
  }
}

void InferWidth::addSource(FieldRef ref) {
  mlir::AsmState asmstate(ref.getValue().getParentBlock()->getParentOp());
  llvm::errs() << "Source: [";
  ref.getValue().printAsOperand(llvm::errs(), asmstate);
  llvm::errs() << "]:" << ref.getFieldID() << "\n";
  sources[ref] = ref;
  updateProposal(ref, 0);
}

void InferWidth::addAlias(FieldRef dest, FieldRef src) {
  assert(sources.count(src));
  // First time seeing dest
  if (!sources.count(dest)) {
    auto realSrc = sources[src];
    sources[dest] = realSrc;
    aliased.insert(dest.getValue());
    invSources[realSrc].append(invSources[dest]);
    invSources[realSrc].push_back(dest);
    invSources.erase(dest);
    mlir::AsmState asmstate(dest.getValue().getParentBlock()->getParentOp());
    llvm::errs() << "Alias: [";
    dest.getValue().printAsOperand(llvm::errs(), asmstate);
    llvm::errs() << "]:" << dest.getFieldID() << " [";
    src.getValue().printAsOperand(llvm::errs(), asmstate);
    llvm::errs() << "]:" << src.getFieldID() << " [";
    realSrc.getValue().printAsOperand(llvm::errs(), asmstate);
    llvm::errs() << "]:" << realSrc.getFieldID() << "\n";
  }
  assert(sources[dest] == sources[src]);
  queueUsers(dest);
}

void InferWidth::updateProposal(FieldRef ref, unsigned width) {
  auto it = proposals.find(ref);
  if (it == proposals.end()) {
    proposals[ref] = width;
    queueUsers(ref);
    return;
  }
  if (it->second == width)
    return;
  assert(width > it->second);
  it->second = width;
  mlir::AsmState asmstate(ref.getValue().getParentBlock()->getParentOp());

  llvm::errs() << "Updated [";
  ref.getValue().printAsOperand(llvm::errs(), asmstate);
  llvm::errs() << "]:" << ref.getFieldID() << " to " << width << "\n";
  queueUsers(ref);
}

void InferWidth::updateProposalOrAlias(FieldRef ref, unsigned width) {
  assert(sources.count(ref));
  auto src = sources[ref];
  auto &val = proposals[src];
  if (val == width)
    return;
  assert(width > val);
  val = width;
  mlir::AsmState asmstate(ref.getValue().getParentBlock()->getParentOp());
  llvm::errs() << "Updated [";
  src.getValue().printAsOperand(llvm::errs(), asmstate);
  llvm::errs() << "]:" << src.getFieldID() << " Initial [";
  ref.getValue().printAsOperand(llvm::errs(), asmstate);
  llvm::errs() << "]:" << ref.getFieldID() << " to " << width << "\n";
  if (auto *op = src.getValue().getDefiningOp())
    worklist.emplace_back(op, src);
  if (auto blkarg = dyn_cast<BlockArgument>(src.getValue()))
    for (auto mp : moduleParams[src])
      updateProposal(mp, width);
  auto it = invSources.find(src);
  if (it != invSources.end())
    for (auto ali : it->second)
      queueUsers(ali);
  queueUsers(src);
}

std::tuple<bool, unsigned> InferWidth::getWidth(FieldRef ref) {
  FIRRTLBaseType type;
  if (auto bt = dyn_cast<FIRRTLBaseType>(ref.getValue().getType())) {
    type = cast<FIRRTLBaseType>(bt.getFinalTypeByFieldID(ref.getFieldID()));
  } else {
    type = cast<FIRRTLBaseType>(cast<RefType>(ref.getValue().getType())
                                    .getFinalTypeByFieldID(ref.getFieldID()));
  }
  auto bw = type.getBitWidthOrSentinel();
  if (bw >= 0)
    return std::make_tuple(true, bw);
  auto it = sources.find(ref);
  if (it != sources.end())
    ref = it->second;
  return std::make_tuple(false, proposals[ref]);
}

// Val is the result that is being used, ref is the originating definition.
void InferWidth::queueUsers(FieldRef refArg) {
  for (auto user : refArg.getValue().getUsers())
    worklist.emplace_back(user, refArg);
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
  return;
}

void InferWidth::visitExpr(SubfieldOp op, FieldRef refArg) {
  auto bundleType = op.getInput().getType();
  auto [childField, isvalid] =
      bundleType.rootChildFieldID(refArg.getFieldID(), op.getFieldIndex());
  if (!isvalid)
    return;
  addAlias({op, (unsigned)childField}, refArg);
}

// collapse vector types
void InferWidth::visitExpr(SubindexOp op, FieldRef refArg) {
  auto vectorType = op.getInput().getType();
  auto [childField, isvalid] =
      vectorType.rootChildFieldID(refArg.getFieldID(), 0);
  assert(isvalid);
  if (!isvalid)
    return;
  addAlias({op, (unsigned)childField}, refArg);
}

// collapse vector types
void InferWidth::visitExpr(SubaccessOp op, FieldRef refArg) {
  auto vectorType = op.getInput().getType();
  auto [childField, isvalid] =
      vectorType.rootChildFieldID(refArg.getFieldID(), 0);
  assert(isvalid);
  if (!isvalid)
    return;
  addAlias({op, (unsigned)childField}, refArg);
}

void InferWidth::visitExpr(SubtagOp op, FieldRef refArg) {
  auto enumType = op.getInput().getType();
  auto [childField, isvalid] =
      enumType.rootChildFieldID(refArg.getFieldID(), op.getFieldIndex());
  if (!isvalid)
    return;
  addAlias({op, (unsigned)childField}, refArg);
}

void InferWidth::visitArithExpr(Operation *op, FieldRef refArg) {
  assert(cast<FIRRTLBaseType>(op->getResult(0).getType()).hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op->getOperand(0), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op->getOperand(1), 0));
  auto w = 1 + std::max(src1Width, src2Width);
  updateProposal({op->getResult(0), 0}, w);
}

void InferWidth::visitLogicExpr(Operation *op, FieldRef refArg) {
  assert(cast<FIRRTLBaseType>(op->getResult(0).getType()).hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op->getOperand(0), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op->getOperand(1), 0));
  auto w = std::max(src1Width, src2Width);
  updateProposal({op->getResult(0), 0}, w);
}

void InferWidth::visitBinarySumExpr(Operation *op, FieldRef refArg) {
  assert(cast<FIRRTLBaseType>(op->getResult(0).getType()).hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op->getOperand(0), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op->getOperand(1), 0));
  auto w = src1Width + src2Width;
  updateProposal({op->getResult(0), 0}, w);
}
void InferWidth::visitExpr(DivPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op.getOperand(0), 0));
  auto w = src1Width + (op.getLhs().getType().isSigned() ? 1 : 0);
  updateProposal({op.getResult(), 0}, w);
}
void InferWidth::visitExpr(RemPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op.getOperand(0), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op.getOperand(1), 0));
  auto w = std::min(src1Width, src2Width);
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(DShlwPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getLhs(), 0));
  auto w = srcWidth;
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(DShrPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getLhs(), 0));
  auto w = srcWidth;
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(DShlPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op.getLhs(), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op.getRhs(), 0));
  auto w = src1Width + (1 << src2Width) - 1;
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(ShlPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = srcWidth + op.getAmount();
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(ShrPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = srcWidth > op.getAmount() ? srcWidth - op.getAmount() : 1;
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::lowerNoopMux(Operation *op, FieldRef refArg) {
  if (!cast<FIRRTLBaseType>(op->getResult(0).getType()).hasUninferredWidth())
    return;
  unsigned w = 0;
  for (unsigned i = 1; i < op->getNumOperands(); ++i)
    w = std::max(w, std::get<1>(getWidth(
                        FieldRef(op->getOperand(i), refArg.getFieldID()))));

  updateProposal({op->getResult(0), refArg.getFieldID()}, w);
}

void InferWidth::lowerNoopCast(Operation *op, FieldRef refArg) {
  assert(cast<FIRRTLBaseType>(op->getResult(0).getType()).hasUninferredWidth());
  // addAlias({op->getResult(0), refArg.getFieldID()}, refArg);
  updateProposal({op->getResult(0), refArg.getFieldID()},
                 std::get<1>(getWidth(refArg)));
}

void InferWidth::visitExpr(CvtPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = srcWidth + (op.getInput().getType().isSigned() ? 0 : 1);
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(NegPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = srcWidth + 1;
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(TailPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = srcWidth - op.getAmount();
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitExpr(PadPrimOp op, FieldRef refArg) {
  assert(op.getType().hasUninferredWidth());
  assert(refArg.getFieldID() == 0);
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getInput(), 0));
  auto w = std::max(srcWidth, op.getAmount());
  updateProposal({op.getResult(), 0}, w);
}

void InferWidth::visitStmt(ConnectOp op, FieldRef refArg) {
  FieldRef destRef = FieldRef(op.getDest(), refArg.getFieldID());
  FieldRef srcRef = FieldRef(op.getSrc(), refArg.getFieldID());
  auto [dstFixed, dstWidth] = getWidth(destRef);
  auto [srcFixed, srcWidth] = getWidth(srcRef);
  // No truncation!
  if (!dstFixed && dstWidth < srcWidth)
    updateProposalOrAlias(destRef, srcWidth);
}

void InferWidth::visitStmt(AttachOp op, FieldRef refArg) {
  unsigned w = 0;
  for (auto operand : op.getAttached()) {
    w = std::max(w,
                 std::get<1>(getWidth(FieldRef(operand, refArg.getFieldID()))));
  }

  for (auto operand : op.getAttached()) {
    FieldRef ref(operand, refArg.getFieldID());
    auto [isfixed, oldW] = getWidth(FieldRef(operand, refArg.getFieldID()));
    if (oldW != w) {
      if (isfixed)
        abort();
      assert(sources.count(ref));
      auto src = sources[ref];
      updateProposalOrAlias(src, w);
    }
  }
}

void InferWidth::visitStmt(RefDefineOp op, FieldRef refArg) {
  FieldRef destRef = FieldRef(op.getDest(), refArg.getFieldID());
  FieldRef srcRef = FieldRef(op.getSrc(), refArg.getFieldID());
  auto [dstFixed, dstWidth] = getWidth(destRef);
  auto [srcFixed, srcWidth] = getWidth(srcRef);
  // No truncation!
  if (!dstFixed && dstWidth < srcWidth)
    updateProposalOrAlias(destRef, srcWidth);
}

void InferWidth::visitExpr(RefSendOp op, FieldRef refArg) {
  addAlias({op, refArg.getFieldID() + 1}, refArg);
}

void InferWidth::visitExpr(RefCastOp op, FieldRef refArg) {
  addAlias({op, refArg.getFieldID()}, refArg);
}

void InferWidth::visitExpr(RefResolveOp op, FieldRef refArg) {
  addAlias({op, refArg.getFieldID() - 1}, refArg);
}

void InferWidth::visitExpr(RefSubOp op, FieldRef refArg) {
  unsigned childField;
  bool isvalid;
  if (auto bundleType =
          dyn_cast<BundleType>(op.getInput().getType().getType())) {
    std::tie(childField, isvalid) =
        bundleType.rootChildFieldID(refArg.getFieldID() - 1, op.getIndex());
  } else if (auto vectorType =
                 dyn_cast<FVectorType>(op.getInput().getType().getType())) {
    // collapse vector types
    std::tie(childField, isvalid) =
        vectorType.rootChildFieldID(refArg.getFieldID() - 1, 0);
  } else {
    llvm_unreachable("unknown type");
  }
  if (!isvalid)
    return;
  addAlias({op, (unsigned)childField + 1}, refArg);
}

// If this got in the worklist, it was from a connect, so just trigger children
void InferWidth::visitDecl(WireOp op, FieldRef refArg) {}
void InferWidth::visitDecl(RegOp op, FieldRef refArg) {}
void InferWidth::visitDecl(RegResetOp op, FieldRef refArg) {
  auto [resetFixed, resetWidth] =
      getWidth({op.getResetValue(), refArg.getFieldID()});
  auto [regFixed, regWidth] = getWidth({op.getResult(), refArg.getFieldID()});
  // NO TRUNCATION
  if (resetWidth > regWidth)
    updateProposal({op.getResult(), refArg.getFieldID()}, resetWidth);
}

void InferWidth::visitDecl(MemOp op, FieldRef refArg) {
#if 0
  unsigned w = 0;
  for (auto r : op.getResults()) {
    for (auto uninferred : getUninferred(r.getType()))
      w = std::max(w, std::get<1>(getWidth({r, uninferred})));
  }
  for (auto r : op.getResults()) {
    for (auto uninferred : getUninferred(r.getType()))
      updateProposal({r, uninferred}, w);
  }
#endif
}

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

  llvm::errs() << "** solve **\n";
  iw.solve();
  llvm::errs() << "\n";

  for (auto op : circuit.getOps<FModuleOp>()) {
    llvm::errs() << "** apply: " << op.getModuleName() << " **\n";
    iw.apply(op);
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsAltPass() {
  return std::make_unique<InferWidthsAltPass>();
}
