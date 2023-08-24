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
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <numeric>
#include <queue>

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

////////////////////////////////////////////////////////////////////////////////
// Transfer functions
////////////////////////////////////////////////////////////////////////////////

template <typename AppFn>
static unsigned accumulateIfValid(ArrayRef<unsigned> args, AppFn fn) {
  unsigned init = ~0U;
  for (auto v : args)
    if (v != ~0U) {
      if (init != ~0U)
        init = fn(init, v);
      else
        init = v;
    }
  return init;
}

static unsigned trSum(ArrayRef<unsigned> args) {
  return accumulateIfValid(args, [](unsigned a, unsigned b) { return a + b; });
}

static unsigned trSub2(ArrayRef<unsigned> args) {
  if (args[0] == ~0U || args[1] == ~0U)
    return ~0U;
  return args[0] - args[1];
}

static unsigned trShr(ArrayRef<unsigned> args) {
  if (args[0] == ~0U || args[1] == ~0U)
    return 1;
  return args[0] > args[1] ? args[0] - args[1] : 1;
}

static unsigned trMax(ArrayRef<unsigned> args) {
  return accumulateIfValid(
      args, [](unsigned a, unsigned b) { return std::max(a, b); });
}

static unsigned trMaxPlusOne(ArrayRef<unsigned> args) {
  auto sum = trMax(args);
  return 1 + (sum == ~0U ? 0 : sum);
}

static unsigned trMin(ArrayRef<unsigned> args) {
  return *std::min_element(args.begin(), args.end()); // ~0 is handled!
}

static unsigned trSumPlusOne(ArrayRef<unsigned> args) {
  auto sum = trSum(args);
  return 1 + (sum == ~0U ? 0 : sum);
}

static unsigned trDshl(ArrayRef<unsigned> args) {
  return (args[0] == ~0UL ? 0 : args[0]) +
         (1 << (args[1] == ~0U ? 0 : args[1])) - 1;
}
namespace {

// RefArg (Field of argument relative field), RefStorage (source of value) ->
// changed
struct InferWidth : public FIRRTLVisitor<InferWidth, void> {

  using RawFnTy = unsigned (*)(ArrayRef<unsigned>);
  using FnTy = std::function<unsigned(ArrayRef<unsigned>)>;

  InferWidth(SymbolTable &symtbl, CircuitOp circuit)
      : circuit(circuit), symtbl(symtbl), failed(false) {}

  void dump_graph();

  void prepare(FModuleOp fmod);
  void computeInverse();
  void solve();
  void apply(FModuleOp fmod);
  LogicalResult apply();

  // Add a free variable to the system.
  void addSource(FieldRef ref);
  // Sometimes we need a fake input to track op constants.
  FieldRef addSynthetic(unsigned val);
  // Unify dest to src
  void addAlias(FieldRef dest, FieldRef src);

  // dest <= src
  void addUpdate(FieldRef dest, FieldRef src);

  // Get the source of the ref by lookup in the alias map.
  FieldRef getSource(FieldRef ref);

  unsigned getProposed(FieldRef ref);

  // Solve one node, return things to evaluate.
  ArrayRef<FieldRef> eval(FieldRef ref);

  std::tuple<bool, unsigned> getWidth(FieldRef);
  Type makeTypeForValue(Value v);
  Type updateBaseType(Value v, Type type, unsigned fieldID);

  void queueUsers(FieldRef refArg);

  using FIRRTLVisitor<InferWidth, void>::visitExpr;
  using FIRRTLVisitor<InferWidth, void>::visitStmt;
  using FIRRTLVisitor<InferWidth, void>::visitDecl;

  FieldRef getSrc(FieldRef);

  void setEq(FieldRef dest, FnTy fn, ArrayRef<FieldRef> r);
  void setEqUnary(Operation *op, FnTy);
  void setEqBinary(Operation *op, FnTy);

  void visitExpr(ConstantOp op);
  //   bool visitExpr(SpecialConstantOp op);
  //   bool visitExpr(AggregateConstantOp op);

  void visitExpr(SubfieldOp op);
  void visitExpr(SubindexOp op);
  void visitExpr(SubaccessOp op);
  void visitExpr(SubtagOp op);

  void visitExpr(CatPrimOp op) { setEqBinary(op, trSum); }

  void visitExpr(AddPrimOp op) { setEqBinary(op, trMaxPlusOne); }
  void visitExpr(SubPrimOp op) { setEqBinary(op, trMaxPlusOne); }
  void visitExpr(MulPrimOp op) { setEqBinary(op, trSum); }
  void visitExpr(DivPrimOp op);
  void visitExpr(RemPrimOp op) { setEqBinary(op, trMin); }

  void visitExpr(AndPrimOp op) { setEqBinary(op, trMax); }
  void visitExpr(OrPrimOp op) { setEqBinary(op, trMax); }
  void visitExpr(XorPrimOp op) { setEqBinary(op, trMax); }

  void visitExpr(NegPrimOp op) { setEqUnary(op, trMaxPlusOne); }

  void visitExpr(TailPrimOp op);
  void visitExpr(PadPrimOp op);

  void visitMux(Operation *op);
  void visitExpr(MuxPrimOp op) { visitMux(op); }
  void visitExpr(Mux2CellIntrinsicOp op) { visitMux(op); }
  void visitExpr(Mux4CellIntrinsicOp op) { visitMux(op); }

  void visitNoopCast(Operation *op);
  void visitExpr(AsSIntPrimOp op) { visitNoopCast(op); }
  void visitExpr(AsUIntPrimOp op) { visitNoopCast(op); }
  void visitExpr(ConstCastOp op) { visitNoopCast(op); }
  void visitExpr(CvtPrimOp op);

  void visitExpr(NotPrimOp op) { visitNoopCast(op); }

  void visitExpr(DShlwPrimOp op) { setEqUnary(op, trMax); }
  void visitExpr(DShlPrimOp op);
  void visitExpr(ShlPrimOp op);
  void visitExpr(DShrPrimOp op) { setEqUnary(op, trMax); }
  void visitExpr(ShrPrimOp op);

  void visitStmt(ConnectOp op);
  void visitStmt(AttachOp op);
  //  void visitStmt(RefDefineOp op, FieldRef);
  //  void visitExpr(RefCastOp op, FieldRef refArg) { unify(op, 0, refArg); }
  //  void visitExpr(RefSendOp op, FieldRef refArg) { unify(op, 1, refArg); }
  //  void visitExpr(RefResolveOp op, FieldRef refArg) { unify(op, -1, refArg);
  //  } void visitExpr(RefSubOp op, FieldRef);
  void visitDecl(InstanceOp op);

  void visitExpr(InvalidValueOp op);

  void visitDecl(WireOp op);
  void visitDecl(RegOp op);
  void visitDecl(RegResetOp op);
  void visitDecl(NodeOp op) { visitNoopCast(op); }
  void visitDecl(MemOp op);

  // Nothing interesting
  void visitExpr(LEQPrimOp) {}
  void visitExpr(LTPrimOp) {}
  void visitExpr(GEQPrimOp) {}
  void visitExpr(GTPrimOp) {}
  void visitExpr(NEQPrimOp) {}
  void visitExpr(EQPrimOp) {}
  void visitExpr(HeadPrimOp) {}
  void visitExpr(BitsPrimOp) {}
  void visitExpr(AndRPrimOp) {}
  void visitExpr(OrRPrimOp) {}
  void visitExpr(XorRPrimOp) {}

  void visitUnhandledOp(Operation *op) {
    if (isa<mlir::UnrealizedConversionCastOp>(op)) {
      visitNoopCast(op);
      return;
    }
    llvm::errs() << "Unhandled ";
    op->dump();
    llvm::errs() << "\n";
  }

  //////////////////////////////////////////////////////////////////////////////
  CircuitOp circuit;

  // Unification tracking
  // Value -> Source or value
  DenseMap<FieldRef, FieldRef> sources;

  struct Eq {
    SmallVector<FieldRef, 2> fields;
    SmallVector<FieldRef, 2> children;
    SmallVector<FieldRef, 2> aliases;
    unsigned width;
    FnTy trFn;
  };

  DenseMap<FieldRef, Eq> eqs;

  // It's supper annoying that invalids back-propagate
  SmallVector<FieldRef, 4> invalids;

  // map from source to aliases.  Maintained by union
  DenseMap<FieldRef, SmallVector<FieldRef>> invSources;

  // Keep track of aliased values so we can be sure to update their ops
  DenseSet<Value> aliased;

  // Connects which might truncate.  This is stupidly error inducing in designs.
  DenseSet<ConnectOp> potentialTrunc;

  // Blockarg -> instance results
  // Note, instance results are marked as aliased to the block arts in `sources`
  // so a connect to an instance result will update the block arg.  This is to
  // know what to iterate on when that update happens.
  DenseMap<FieldRef, SmallVector<FieldRef>> moduleParams;

  // Value updated, Defining source
  std::deque<std::tuple<Operation *, FieldRef>> worklist;

  SymbolTable &symtbl;
  bool failed;
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
    ref = getSource(ref);
    assert(eqs.count(ref));
    return resizeType(fbt, eqs[ref].width);
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

void InferWidth::dump_graph() {
  mlir::AsmState asmstate(circuit);
  for (auto it : eqs) {
    llvm::errs() << "[";
    it.first.getValue().printAsOperand(llvm::errs(), asmstate);
    llvm::errs() << "]:" << it.first.getFieldID();
    if (it.second.width != ~0U)
      llvm::errs() << " width:" << it.second.width;
    llvm::errs() << " <" << it.second.trFn.target<RawFnTy>() << "> fields:{";
    for (auto &e : it.second.fields)
      if (e.getValue() || e.getFieldID()) {
        llvm::errs() << " [";
        if (e.getValue())
          e.getValue().printAsOperand(llvm::errs(), asmstate);
        else
          llvm::errs() << "cst";
        llvm::errs() << "]:" << e.getFieldID();
      }
    llvm::errs() << "}";
    if (!it.second.aliases.empty()) {
      llvm::errs() << " aliases:{";
      for (auto &a : it.second.aliases) {
        llvm::errs() << " [";
        a.getValue().printAsOperand(llvm::errs(), asmstate);
        llvm::errs() << "]:" << a.getFieldID();
      }
      llvm::errs() << "}";
    }
    llvm::errs() << "\n";
  }
}

unsigned InferWidth::getProposed(FieldRef ref) {
  if (ref.getValue() == 0) // synthetic
    return ref.getFieldID();
  assert(eqs.count(ref));
  return eqs[ref].width;
}

ArrayRef<FieldRef> InferWidth::eval(FieldRef ref) {
  assert(eqs.count(ref));
  SmallVector<unsigned, 2> args;
  auto &eq = eqs[ref];
  for (auto f : eq.fields)
    args.push_back(getProposed(f));
  assert(eq.trFn);
  unsigned newW = eq.trFn(args);
  if (newW == eq.width)
    return {};

  LLVM_DEBUG(mlir::AsmState asmstate(circuit); llvm::errs() << "Updating [";
             ref.getValue().printAsOperand(llvm::errs(), asmstate);
             llvm::errs() << "]:" << ref.getFieldID() << " " << eq.width << "->"
                          << newW << "\n";);
  eq.width = newW;
  return eq.children;
}

void InferWidth::prepare(FModuleOp fmod) {

  // Scan block arguments for uninferred
  size_t firrtlArg = 0;
  for (auto port : fmod.getPorts()) {
    auto arg = fmod.getBody().getArgument(firrtlArg++);
    for (auto uninferred : getUninferred(arg.getType()))
      addSource({arg, uninferred});
  }

  fmod.walk([&](Operation *op) { dispatchVisitor(op); });
}

void InferWidth::computeInverse() {
  for (auto it : eqs) {
    auto dest = it.first;
    auto &eq = it.second;
    for (auto r : eq.fields) {
      if (eqs.count(r)) {
        eqs[r].children.push_back(dest);
      } else {
        llvm::errs() << "No inverse\n";
      }
    }
  }
  for (auto it : eqs) {
    auto dest = it.first;
    auto &eq = it.second;
    std::sort(eq.children.begin(), eq.children.end());
    auto itemp = std::unique(eq.children.begin(), eq.children.end());
    eq.children.erase(itemp, eq.children.end());
  }
}

void InferWidth::solve() {

  computeInverse();

  std::queue<FieldRef> worklist;
  for (auto r : eqs)
    worklist.push(r.first);

  // Iterate on the worklist
  while (!worklist.empty()) {
    auto ref = worklist.front();
    worklist.pop();
    auto rt = eval(ref);
    for (auto &r : rt)
      worklist.push(r);
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
}

LogicalResult InferWidth::apply() {
  auto updateValue = [&](Value v) {
    if (isa<BlockArgument>(v))
      return;
    Operation *op = v.getDefiningOp();
    for (auto r : op->getResults())
      if (r == v) {
        llvm::errs() << "Apply " << v << " | ";
        r.setType(makeTypeForValue(v));
        llvm::errs() << r << "\n";
      }
  };

  // Update Operations
  DenseSet<Value> seen;
  for (auto &p : eqs) {
    // visit each value once, but each value may have multiple equations, one
    // per-field, so be sure to visit all the fields as they come up
    if (!seen.count(p.first.getValue())) {
      seen.insert(p.first.getValue());
      updateValue(p.first.getValue());
    }
    for (auto r : p.second.aliases) {
      // here we can filter by value, since values are updated completely.
      if (seen.count(r.getValue()))
        continue;
      seen.insert(r.getValue());
      updateValue(r.getValue());
    }
  }

  // Connects can truncate sometimes, check them.
  // This shouldn't be a thing.
  //  for (auto con : potentialTrunc) {
  //    if (con.getDest().getType() == con.getSrc().getType())
  //      continue;
  //    OpBuilder builder(con);
  //    emitConnect(builder, con.getLoc(), con.getDest(), con.getSrc());
  //    con.erase();
  //  }

  return failed ? failure() : success();
}

void InferWidth::addSource(FieldRef ref) {
  LLVM_DEBUG(
      mlir::AsmState asmstate(ref.getValue().getParentBlock()->getParentOp());
      llvm::errs() << "Source: [";
      ref.getValue().printAsOperand(llvm::errs(), asmstate);
      llvm::errs() << "]:" << ref.getFieldID() << "\n";);
  sources[ref] = ref;
  assert(!eqs.count(ref));
  setEq(ref, trMax, ArrayRef<FieldRef>());
}

FieldRef InferWidth::addSynthetic(unsigned val) { return {nullptr, val}; }

void InferWidth::addAlias(FieldRef dest, FieldRef src) {
  LLVM_DEBUG(
      mlir::AsmState asmstate(dest.getValue().getParentBlock()->getParentOp());
      llvm::errs() << "Alias: [";
      dest.getValue().printAsOperand(llvm::errs(), asmstate);
      llvm::errs() << "]:" << dest.getFieldID() << " [";
      src.getValue().printAsOperand(llvm::errs(), asmstate);
      llvm::errs() << "]:" << src.getFieldID() << "\n";);

  assert(sources.count(src));
  // First time seeing dest
  if (!sources.count(dest)) {
    auto realSrc = sources[src];
    sources[dest] = realSrc;
    assert(eqs.count(realSrc));
    eqs[realSrc].aliases.push_back(dest);
  }
  assert(sources[dest] == sources[src]);
}

void InferWidth::addUpdate(FieldRef dest, FieldRef src) {
  dest = getSrc(dest);
  src = getSrc(src);
  assert(sources.count(dest));
  LLVM_DEBUG(
      mlir::AsmState asmstate(dest.getValue().getParentBlock()->getParentOp());
      llvm::errs() << "Update: [";
      dest.getValue().printAsOperand(llvm::errs(), asmstate);
      llvm::errs() << "]:" << dest.getFieldID() << " <- [";
      src.getValue().printAsOperand(llvm::errs(), asmstate);
      llvm::errs() << "]:" << src.getFieldID() << "\n";);
  assert(eqs.count(dest));
  auto &eq = eqs[dest];
  eq.fields.push_back(src);
}

FieldRef InferWidth::getSrc(FieldRef ref) {
  if (!ref.getValue()) // constants and sentinels
    return ref;
  auto it = sources.find(ref);
  if (it != sources.end())
    ref = it->second;
  // Check for constant
  auto type1 = cast<FIRRTLBaseType>(ref.getValue().getType())
                   .getFinalTypeByFieldID(ref.getFieldID());
  auto type = cast<FIRRTLBaseType>(type1);
  if (type.hasUninferredWidth())
    return ref;
  return {nullptr, (unsigned)type.getBitWidthOrSentinel()};
}

void InferWidth::setEq(FieldRef dest, FnTy fn, ArrayRef<FieldRef> refs) {
  assert(!eqs.count(dest));
  assert(fn);
  auto &eq = eqs[dest];
  eq.width = ~0U;
  eq.trFn = fn;
  for (auto r : refs)
    eq.fields.push_back(getSrc(r));
}

void InferWidth::setEqBinary(Operation *op, FnTy fn) {
  FIRRTLBaseType type = cast<FIRRTLBaseType>(op->getResult(0).getType());
  if (!type.hasUninferredWidth())
    return;
  FieldRef lhs(op->getOperand(0), 0);
  FieldRef rhs(op->getOperand(1), 0);
  setEq({op->getResult(0), 0}, fn, {lhs, rhs});
}

void InferWidth::setEqUnary(Operation *op, FnTy fn) {
  FIRRTLBaseType type = cast<FIRRTLBaseType>(op->getResult(0).getType());
  if (!type.hasUninferredWidth())
    return;
  FieldRef lhs(op->getOperand(0), 0);
  setEq({op->getResult(0), 0}, fn, lhs);
}

FieldRef InferWidth::getSource(FieldRef ref) {
  auto it = sources.find(ref);
  if (it != sources.end())
    return it->second;
  return ref;
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
  assert(eqs.count(getSource(ref)));
  return std::make_tuple(false, eqs[getSource(ref)].width);
}

// Val is the result that is being used, ref is the originating definition.
void InferWidth::queueUsers(FieldRef refArg) {
  for (auto user : refArg.getValue().getUsers())
    worklist.emplace_back(user, refArg);
}

// Constants are exceptions to inference, since they are always locally
// inferrable.  Therefore we just set them and forget them.
void InferWidth::visitExpr(ConstantOp cst) {
  if (!cst.getType().hasUninferredWidth())
    return;
  // Completely resolve constants
  auto v = cst.getValue();
  auto w = v.getBitWidth() -
           (v.isNegative() ? v.countLeadingOnes() : v.countLeadingZeros());
  if (v.isSigned())
    w += 1;
  // I don't see why we can't infer a 0-width constant for zeros
  if (!w)
    w = 1;
  if (v.getBitWidth() > unsigned(w))
    v = v.trunc(w);
  // Go ahead and pre-update constants then we don't have to track them.
  cst.setValue(v);
  cst.getResult().setType(cast<IntType>(resizeType(cst.getType(), w)));
  FieldRef ref(cst, 0);
  addSource(ref);
  eqs[ref].fields.push_back(addSynthetic(w));
}

void InferWidth::visitExpr(SubfieldOp op) {
  auto bundleType = op.getInput().getType();
  for (auto uninferred : getUninferred(op.getResult().getType()))
    addAlias({op, uninferred},
             {op.getInput(), uninferred + (unsigned)bundleType.getFieldID(
                                              op.getFieldIndex())});
}

// collapse vector types
void InferWidth::visitExpr(SubindexOp op) {
  for (auto uninferred : getUninferred(op.getResult().getType()))
    addAlias({op, uninferred}, {op.getInput(), uninferred + 1});
}

// collapse vector types
void InferWidth::visitExpr(SubaccessOp op) {
  for (auto uninferred : getUninferred(op.getResult().getType()))
    addAlias({op, uninferred}, {op.getInput(), uninferred + 1});
}

void InferWidth::visitExpr(SubtagOp op) {
  auto enumType = op.getInput().getType();
  for (auto uninferred : getUninferred(op.getResult().getType()))
    addAlias({op, uninferred},
             {op.getInput(),
              uninferred + (unsigned)enumType.getFieldID(op.getFieldIndex())});
}

void InferWidth::visitExpr(DivPrimOp op) {
  if (op.getLhs().getType().isSigned())
    setEqUnary(op, trMaxPlusOne);
  else
    setEqUnary(op, trMax);
}

void InferWidth::visitExpr(DShlPrimOp op) { setEqBinary(op, trDshl); }

void InferWidth::visitExpr(ShlPrimOp op) {
  setEq({op, 0}, trSum, {{op.getInput(), 0}, addSynthetic(op.getAmount())});
}

void InferWidth::visitExpr(ShrPrimOp op) {
  setEq({op, 0}, trShr, {{op.getInput(), 0}, addSynthetic(op.getAmount())});
}

void InferWidth::visitMux(Operation *op) {
  bool is4 = op->getNumResults() == 5;
  for (auto uninferred : getUninferred(op->getResult(0).getType()))
    if (is4)
      setEq({op->getResult(0), uninferred}, trMax,
            {{op->getOperand(1), uninferred},
             {op->getOperand(2), uninferred},
             {op->getOperand(3), uninferred},
             {op->getOperand(4), uninferred}});
    else
      setEq({op->getResult(0), uninferred}, trMax,
            {{op->getOperand(1), uninferred}, {op->getOperand(2), uninferred}});
}

void InferWidth::visitNoopCast(Operation *op) {
  for (auto uninferred : getUninferred(op->getResult(0).getType()))
    addAlias({op->getResult(0), uninferred}, {op->getOperand(0), uninferred});
}

void InferWidth::visitExpr(CvtPrimOp op) {
  if (op.getInput().getType().isSigned())
    setEqUnary(op, trMax);
  else
    setEqUnary(op, trMaxPlusOne);
}

void InferWidth::visitExpr(TailPrimOp op) {
  setEq({op, 0}, trSub2, {{op.getInput(), 0}, addSynthetic(op.getAmount())});
}

void InferWidth::visitExpr(PadPrimOp op) {
  setEq({op, 0}, trMax, {{op.getInput(), 0}, addSynthetic(op.getAmount())});
}

void InferWidth::visitStmt(ConnectOp op) {
  for (auto uninferred : getUninferred(op.getDest().getType())) {
    FieldRef destRef = FieldRef(op.getDest(), uninferred);
    FieldRef srcRef = FieldRef(op.getSrc(), uninferred);
    addUpdate(destRef, srcRef);
    // Try to back propagate to invalids.  This isn't perfect.
    auto realSrc = getSrc(srcRef);
    if (realSrc.getValue() &&
        realSrc.getValue().getDefiningOp<InvalidValueOp>()) {
      auto &eq = eqs[realSrc];
      eq.width = ~0U;
      eq.trFn = trMax;
      eq.fields.push_back(getSrc(destRef));
    }
  }
}

void InferWidth::visitStmt(AttachOp op) {
  // This is stupid, but rare.  We don't know which operand has which fields
  // uninferred.
  llvm::SmallSet<unsigned, 8> seen;
  for (auto oper : op.getAttached()) {
    for (auto uninferred : getUninferred(oper.getType())) {
      if (seen.count(uninferred))
        continue;
      seen.insert(uninferred);

      unsigned w = ~0U;
      for (auto operand : op.getAttached()) {
        auto oldW = std::get<1>(getWidth(FieldRef(operand, uninferred)));
        if (oldW == ~0U)
          continue;
        w = (w == ~0U) ? oldW : std::max(w, oldW);
      }
      if (w == ~0U)
        llvm::errs() << "No width source for attach\n";
      llvm::errs() << "Attach: " << w << "\n";
      for (auto operand : op.getAttached()) {
        FieldRef ref(operand, uninferred);
        auto [isfixed, oldW] = getWidth(ref);
        if (oldW != w) {
          if (isfixed)
            abort();
          llvm::errs() << " up " << ref.getValue() << "\n";
          addUpdate(ref, addSynthetic(w));
        }
      }
    }
  }
}

// void InferWidth::visitStmt(RefDefineOp op, FieldRef refArg) {
//   FieldRef destRef = FieldRef(op.getDest(), refArg.getFieldID());
//   FieldRef srcRef = FieldRef(op.getSrc(), refArg.getFieldID());
//   auto [dstFixed, dstWidth] = getWidth(destRef);
//   auto [srcFixed, srcWidth] = getWidth(srcRef);
//   // No truncation!
//   if (!dstFixed && dstWidth < srcWidth)
//     updateProposalOrAlias(destRef, srcWidth);
// }

// void InferWidth::visitExpr(RefSubOp op, FieldRef refArg) {
//   unsigned childField;
//   bool isvalid;
//   if (auto bundleType =
//           dyn_cast<BundleType>(op.getInput().getType().getType())) {
//     std::tie(childField, isvalid) =
//         bundleType.rootChildFieldID(refArg.getFieldID() - 1, op.getIndex());
//   } else if (auto vectorType =
//                  dyn_cast<FVectorType>(op.getInput().getType().getType())) {
//     // collapse vector types
//     std::tie(childField, isvalid) =
//         vectorType.rootChildFieldID(refArg.getFieldID() - 1, 0);
//   } else {
//     llvm_unreachable("unknown type");
//   }
//   if (!isvalid)
//    return;
//  addAlias({op, (unsigned)childField + 1}, refArg);
//}

void InferWidth::visitDecl(InstanceOp inst) {
  auto refdModule = inst.getReferencedModule(symtbl);
  auto module = dyn_cast<FModuleOp>(&*refdModule);
  if (!module) {
    mlir::emitError(inst.getLoc())
        << "extern module has ports of uninferred width";
    return;
  }
  for (auto it : llvm::zip(inst.getResults(), module.getArguments())) {
    for (auto uninferred : getUninferred(std::get<0>(it).getType())) {
      FieldRef ref(std::get<0>(it), uninferred);
      FieldRef src(std::get<1>(it), uninferred);
      addAlias(ref, src);
    }
  }
}

void InferWidth::visitExpr(InvalidValueOp op) {
  for (auto uninferred : getUninferred(op.getResult().getType()))
    addSource({op, uninferred});
}

// If this got in the worklist, it was from a connect, so just trigger children
void InferWidth::visitDecl(WireOp op) {
  bool isForceable = op.getForceable();
  for (auto uninferred : getUninferred(op.getResult().getType())) {
    addSource({op.getResult(), uninferred});
    if (isForceable)
      addAlias({op.getRef(), uninferred + 1}, {op.getResult(), uninferred});
  }
}

void InferWidth::visitDecl(RegOp op) {
  bool isForceable = op.getForceable();
  for (auto uninferred : getUninferred(op.getResult().getType())) {
    addSource({op.getResult(), uninferred});
    if (isForceable)
      addAlias({op.getRef(), uninferred + 1}, {op.getResult(), uninferred});
  }
}

void InferWidth::visitDecl(RegResetOp op) {
  bool isForceable = op.getForceable();
  for (auto uninferred : getUninferred(op.getResult().getType())) {
    addSource({op.getResult(), uninferred});
    if (isForceable)
      addAlias({op.getRef(), uninferred + 1}, {op.getResult(), uninferred});
    addUpdate({op.getResult(), uninferred}, {op.getResetValue(), uninferred});
  }
}

void InferWidth::visitDecl(MemOp mem) {
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
  unsigned firstFieldIndex = dataFieldIndices(mem.getPortKind(nonDebugPort))[0];
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
  InferWidth iw(symtbl, circuit);
  for (auto op : circuit.getOps<FModuleOp>())
    iw.prepare(op);

  llvm::errs() << "\n** Graph **\n";
  iw.dump_graph();
  llvm::errs() << "\n** Solve **\n";
  iw.solve();
  llvm::errs() << "\n** Graph **\n";
  iw.dump_graph();
  llvm::errs() << "\n** Apply **\n";

  for (auto op : circuit.getOps<FModuleOp>()) {
    op.dump();
    iw.apply(op);
    op.dump();
  }
  iw.apply();

  // if (failed(iw.apply())) {
  // signalPassFailure();
  //}
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsAltPass() {
  return std::make_unique<InferWidthsAltPass>();
}
