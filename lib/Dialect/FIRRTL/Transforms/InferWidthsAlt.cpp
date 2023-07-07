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
      .Case<BundleType>([&](auto t) {
        for (auto [idx, element] : llvm::enumerate(t.getElements())) {
          findUninferred(fields, element.type, offset + t.getFieldID(idx));
        }
      })
      .Case<VectorType>([&](auto t) {
        findUninferred(fields, t.getElementType(), offset + 1);
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
struct InferWidth : public FIRRTLVisitor<InferWidth, bool, FieldRef> {

  void run(FModuleOp fmod);

  bool updateProposal(FieldRef ref, unsigned width);
  std::tuple<bool, unsigned> getWidth(FieldRef);

  void queueUsers(Value val, FieldRef refArg);

  using FIRRTLVisitor<InferWidth, bool, FieldRef>::visitExpr;
  using FIRRTLVisitor<InferWidth, bool, FieldRef>::visitStmt;
  using FIRRTLVisitor<InferWidth, bool, FieldRef>::visitDecl;

  //  bool visitExpr(SpecialConstantOp op);
  //  bool visitExpr(AggregateConstantOp op);

  bool visitExpr(SubfieldOp op, FieldRef);
  bool visitExpr(SubindexOp op, FieldRef);
  bool visitExpr(SubaccessOp op, FieldRef);
  bool visitExpr(AddPrimOp op, FieldRef);

  bool lowerNoopMux(Operation *op, FieldRef);
  bool visitExpr(MuxPrimOp op, FieldRef ref) { return lowerNoopMux(op, ref); }
  bool visitExpr(Mux2CellIntrinsicOp op, FieldRef ref) { return lowerNoopMux(op, ref); }
  bool visitExpr(Mux4CellIntrinsicOp op, FieldRef ref) { return lowerNoopMux(op, ref); }

  bool lowerNoopCast(Operation *op, FieldRef);
  bool visitExpr(AsSIntPrimOp op, FieldRef ref) { return lowerNoopCast(op, ref); }
  bool visitExpr(AsUIntPrimOp op, FieldRef ref) { return lowerNoopCast(op, ref); }

  bool visitStmt(ConnectOp op, FieldRef);

  bool visitDecl(WireOp op, FieldRef);
  bool visitDecl(NodeOp op, FieldRef ref) { return lowerNoopCast(op, ref); }

  bool visitUnhandledOp(Operation *op, FieldRef) { return false; }

  DenseMap<FieldRef, unsigned> proposals;

  // Result -> Source
  DenseMap<FieldRef, FieldRef> sources;

  // Value updated, Defining source
  std::deque<std::tuple<Operation *, FieldRef>> worklist;
};
} // namespace

void InferWidth::run(FModuleOp fmod) {

  // Scan block arguments for uninferred
  size_t firrtlArg = 0;
  for (auto port : fmod.getPorts()) {
    auto arg = fmod.getBody().getArgument(firrtlArg++);
    for (auto uninferred : getUninferred(arg)) {
      llvm::errs() << "Source: blockarg [" << port.getName() << "]:" << uninferred << "\n";
      FieldRef ref(arg, uninferred);
      sources[ref] = ref;
      updateProposal(ref, 0);
      queueUsers(arg, ref);
    }
  }

  // Initially scan for sources of uninferred
  for (auto &op : fmod.getBodyBlock()->getOperations()) {
    bool hasUninferred = false;
    for (auto result : op.getResults())
      hasUninferred |= hasUninferredWidth(result.getType());
      if (!hasUninferred)
        continue;
    if (auto wire = dyn_cast<WireOp>(op)) {
      for (auto uninferred : getUninferred(wire.getResult())) {
        llvm::errs() << "Source: wire " << uninferred << "\n";
        FieldRef ref(wire.getResult(), uninferred);
      sources[ref] = ref;
        updateProposal(ref, 0);
        queueUsers(wire.getResult(), ref);
      }
    } else if (auto cst = dyn_cast<ConstantOp>(op)) {
      auto v = cst.getValue();
      auto w = v.getBitWidth() -
               (v.isNegative() ? v.countLeadingOnes() : v.countLeadingZeros());
      if (v.isSigned())
        w += 1;
    if (v.getBitWidth() > unsigned(w))
      v = v.trunc(w);
      // Go ahead and pre-update constants
            cst.setValue(v);
            cst.getResult().setType(cast<IntType>(resizeType(cst.getType(), w)));
        llvm::errs() << "Source: constant " << cst << "\n";
      FieldRef ref(cst, 0);
      sources[ref] = ref;
        updateProposal(ref, w);
      queueUsers(cst.getResult(), ref);
    } else if (auto cst = dyn_cast<InvalidValueOp>(op)) {
        llvm::errs() << "Source: invalid " << cst << "\n";
      FieldRef ref(cst, 0);
      sources[ref] = ref;
      updateProposal(ref, 0);
      queueUsers(cst.getResult(), ref);
    }
  }

  // Iterate on the worklist
  while (!worklist.empty()) {
    auto [op, refArg] = worklist.front();
    worklist.pop_front();
    if (dispatchVisitor(op, refArg)) {
      llvm::errs() << "Changed\n";
    }
  }

  for (auto p : proposals) {
    llvm::errs() << "Set [" << p.first.getValue()
                 << "]:" << p.first.getFieldID() << " to " << p.second << "\n";
    if (p.first.getFieldID() == 0) {
      if (auto cst = p.first.getDefiningOp<ConstantOp>()) {
        cst.getResult().setType(
            cast<IntType>(resizeType(cst.getType(), p.second)));
            continue;
      }
    }
        // If this is an operation that does not generate any free variables that
  // are determined during width inference, simply update the value type based
  // on the operation arguments.
  if (auto op = dyn_cast_or_null<InferTypeOpInterface>(p.first.getValue().getDefiningOp())) {
    SmallVector<Type, 2> types;
    auto res =
        op.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(),
                            op->getAttrDictionary(), op->getPropertiesStorage(),
                            op->getRegions(), types);
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
  }

    }
}

bool InferWidth::updateProposal(FieldRef ref, unsigned width) {
  auto &val = proposals[ref];
  if (val == width)
    return false;
  val = width;
  llvm::errs() << "Updated [" << ref.getValue() << "]:" << ref.getFieldID()
               << " to " << width << "\n";
  return true;
}

std::tuple<bool, unsigned> InferWidth::getWidth(FieldRef ref) {
  auto type =
      cast<FIRRTLBaseType>(cast<FIRRTLBaseType>(ref.getValue().getType())
                               .getFinalTypeByFieldID(ref.getFieldID()));
  auto bw = type.getBitWidthOrSentinel();
  if (bw >= 0)
    return std::make_tuple(true, bw);
    if (!sources.count(ref)) {
      llvm::errs() << "Failed getwidth [" << ref.getValue() << "]:" << ref.getFieldID() << "\n";
      return std::make_tuple(false, 0);
    }
  auto src = sources[ref];
  return std::make_tuple(false, proposals[src]);
}

// Val is the result that is being used, ref is the originating definition.
void InferWidth::queueUsers(Value val, FieldRef refArg) {
  for (auto user : val.getUsers())
    worklist.emplace_back(user, refArg);
}

bool InferWidth::visitExpr(SubfieldOp op, FieldRef refArg) {
  auto bundleType = op.getInput().getType();
  auto [childField, isvalid] =
      bundleType.rootChildFieldID(refArg.getFieldID(), op.getFieldIndex());
  llvm::errs() << "Subfield " << bundleType << " " << op.getFieldIndex() << " "
               << refArg.getValue() << " " << refArg.getFieldID() << " "
               << childField << " " << isvalid << "\n";
  if (!isvalid)
    return false;
  FieldRef ref(op, childField);
  assert(sources.count(refArg));
  auto src = sources[refArg];
  sources[ref] = src;
  queueUsers(op, ref);
  return updateProposal(ref, get<1>(getWidth(src)));
}

// collapse vector types
bool InferWidth::visitExpr(SubindexOp op, FieldRef refArg) {
  auto vectorType = op.getInput().getType();
  auto [childField, isvalid] =
      vectorType.rootChildFieldID(refArg.getFieldID(), 0);
  llvm::errs() << "Subindex " << vectorType << " " << refArg.getValue() << " "
               << refArg.getFieldID() << " " << childField << " " << isvalid
               << "\n";
  if (!isvalid)
    return false;
  FieldRef ref(op, childField);
  assert(sources.count(refArg));
  auto src = sources[refArg];
  sources[ref] = src;
  queueUsers(op, ref);
  return updateProposal(ref, get<1>(getWidth(src)));
}

// collapse vector types
bool InferWidth::visitExpr(SubaccessOp op, FieldRef refArg) {
  auto vectorType = op.getInput().getType();
  auto [childField, isvalid] =
      vectorType.rootChildFieldID(refArg.getFieldID(), 0);
  llvm::errs() << "Subaccess " << vectorType << " " << refArg.getValue() << " "
               << refArg.getFieldID() << " " << childField << " " << isvalid
               << "\n";
  if (!isvalid)
    return false;
  FieldRef ref(op, childField);
  assert(sources.count(refArg));
  auto src = sources[refArg];
  sources[ref] = src;
  queueUsers(op, ref);
  return updateProposal(ref, get<1>(getWidth(src)));
}

bool InferWidth::visitExpr(AddPrimOp op, FieldRef refArg) {
  if (!op.getType().hasUninferredWidth())
    return false;
  assert(refArg.getFieldID() == 0);
  auto [src1Fixed, src1Width] = getWidth(FieldRef(op.getOperand(0), 0));
  auto [src2Fixed, src2Width] = getWidth(FieldRef(op.getOperand(1), 0));
  auto w = 1 + std::max(src1Width, src2Width);
  llvm::errs() << "Add " << op << " of " << src1Width << "," << src2Width
               << "->" << w << "\n";
  FieldRef ref(op.getResult(), 0);
  sources[ref] = ref;
  queueUsers(op, ref);
  return updateProposal(ref, w);
}

bool InferWidth::lowerNoopMux(Operation* op, FieldRef refArg) {
  unsigned w = 0;
  for (int i = 1; i < op->getNumOperands(); ++i)
    w = std::max(w, get<1>(getWidth(FieldRef(op->getOperand(i), refArg.getFieldID()))));

  llvm::errs() << "Mux " << op << " ->" << w << "\n";
  FieldRef ref(op->getResult(0), refArg.getFieldID());
  sources[ref] = ref;
  queueUsers(op->getResult(0), ref);
  return updateProposal(ref, w);
}

  bool InferWidth::lowerNoopCast(Operation *op, FieldRef refArg) {
  FieldRef ref(op->getResult(0), refArg.getFieldID());
  assert(sources.count(refArg));
  auto src = sources[refArg];
  sources[ref] = src;
  queueUsers(op->getResult(0), ref);
  return updateProposal(ref, get<1>(getWidth(src)));
  }

bool InferWidth::visitStmt(ConnectOp op, FieldRef refArg) {
  auto [dstFixed, dstWidth] = getWidth(FieldRef(op.getDest(), refArg.getFieldID()));
  auto [srcFixed, srcWidth] = getWidth(FieldRef(op.getSrc(), refArg.getFieldID()));
    llvm::errs() << "Connect field " << refArg.getFieldID() << " of " << op
                 << " src:" << srcFixed << "," << srcWidth << "->d:" << dstFixed
                 << "," << dstWidth << "\n";
    bool changed = false;
    // No truncation!
    if (!dstFixed && dstWidth < srcWidth)
      changed |= updateProposal(FieldRef(op.getDest(), refArg.getFieldID()), srcWidth);
  if (changed && op.getDest().getDefiningOp()) {
    llvm::errs() << "Worklist add ";
    op.getDest().dump();
    //    worklist.push_back(op.getDest().getDefiningOp());
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
  SymbolTable symtbl(getOperation());
  CircuitOp circuit = getOperation();
  llvm::errs() << "Starting " << circuit.getName() << "\n";
  for (auto op : circuit.getOps<FModuleOp>()) {
    llvm::errs() << "** " << op.getModuleName() << " **\n";
    InferWidth iw;
    iw.run(op);
    llvm::errs() << "\n";
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsAltPass() {
  return std::make_unique<InferWidthsAltPass>();
}
