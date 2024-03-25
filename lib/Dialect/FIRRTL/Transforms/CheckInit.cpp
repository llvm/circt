//===- CheckInit.cpp - Ensure all wires are initialized ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CheckInit pass.  This pass checks that all wires are
// initialized (connected to).
//
// One assumption is that dynamic vector indexes cannot contribute to
// initialization as you don't know the index.  It is possible that for an
// n-length vector, n dynamic indexes and connects could be constructed such
// that the dynamic indexes are coupled enough that they guarantee coverage of
// all elements.  We ignore this.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLFieldSource.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-check-init"

using llvm::BitVector;

using namespace mlir;
using namespace circt;
using namespace firrtl;

// SetSets track initialized fieldIDs
using SetSet = DenseMap<Value, BitVector>;

static void markWrite(BitVector &bv, Value v, size_t fieldID) {
  if (bv.size() <= fieldID)
    bv.resize(fieldID + 1);
  LLVM_DEBUG(llvm::errs() << "S " << v << " " << fieldID << "\n";);
  bv.set(fieldID);
}

static void dumpBV(const BitVector &vals) {
  for (size_t i = 0, e = vals.size(); i != e; ++i)
    llvm::errs() << vals[i];
}

static void unionSetSet(SetSet &dest, const SetSet &src) {
  // now merge from src for commmon elements
  for (auto &[v, bv] : src)
    dest[v] |= bv;
}

static void intersectSetSet(SetSet &dest, const SetSet &src) {

  // Filter dest to get rid of things not in src.
  SmallVector<Value> toDelete;
  for (auto &[v, bv] : dest)
    if (!src.count(v))
      toDelete.push_back(v);
  for (auto v : toDelete)
    dest.erase(v);

  // Intersect anything common.  Everything in dest is in src at this point, but
  // not vice versa.
  for (auto &[v, bv] : src)
    if (dest.count(v))
      dest[v] &= bv;
}

namespace {
class CheckInitPass : public CheckInitBase<CheckInitPass> {
  struct RegionState {
    // Values Inited in this state's regions.  To be intersected.
    SetSet init;
    // Children to merge into this state
    SmallVector<Operation *, 4> children;
    // Definitions from this state
    SmallVector<Value> dests;
  };

  bool failed;
  SmallVector<Operation *> worklist;
  DenseMap<Region *, RegionState> localInfo;

  void processOp(Operation *op, FieldSource &fieldSource);
  void reportRegion(Value decl, Type t, size_t fieldID, bool needed,
                    bool alwaysNeeded, BitVector &vals);
  void emitInitError(Value decl, size_t fieldID);

public:
  void runOnOperation() override;
};
} // end anonymous namespace

static bool checkBit(BitVector &vals, size_t idx) {
  if (vals.size() <= idx)
    return false;
  return vals[idx];
}

static std::string nameForField(Type t, size_t fieldID) {
  LLVM_DEBUG(llvm::errs() << "N: " << t << " @" << fieldID << "\n";);
  return FIRRTLTypeSwitch<Type, std::string>(t)
      .Case<BundleType, OpenBundleType>([fieldID](auto bundle) {
        auto idx = bundle.getIndexForFieldID(fieldID);
        auto fid = bundle.getFieldID(idx);
        if (fieldID == fid)
          return bundle.getElementName(idx).str();
        return bundle.getElementName(idx).str() + "." +
               nameForField(bundle.getElementType(idx), fieldID - fid);
      })
      .Case<FVectorType, OpenVectorType>([fieldID](auto vec) {
        auto idx = vec.getIndexForFieldID(fieldID);
        auto fid = vec.getFieldID(idx);
        if (fieldID == fid)
          return "[" + std::to_string(idx) + "]";
        return "[" + std::to_string(idx) + "]." +
               nameForField(vec.getElementType(), fieldID - fid);
      })
      .Default([](auto) { return "INVALID"; });
}

static StringRef getPortName(BlockArgument b) {
  FModuleOp mod = cast<FModuleOp>(b.getOwner()->getParentOp());
  return mod.getPort(b.getArgNumber()).getName();
}

static StringRef getMemPortName(Value v) {
  MemOp mem = cast<MemOp>(v.getDefiningOp());
  return mem.getPortName(cast<OpResult>(v).getResultNumber());
}

static StringRef getCMemPortName(Value v) {
  chirrtl::MemoryPortOp mem = cast<chirrtl::MemoryPortOp>(v.getDefiningOp());
  return mem.getName();
}

static StringRef getInstPortName(Value v) {
  InstanceOp inst = cast<InstanceOp>(v.getDefiningOp());
  return inst.getPortName(cast<OpResult>(v).getResultNumber());
}

void CheckInitPass::emitInitError(Value decl, size_t fieldID) {

  InFlightDiagnostic err = [&]() {
    if (auto ba = dyn_cast<BlockArgument>(decl))
      return getOperation()->emitError("Port `") << getPortName(ba) << "`";
    if (isa<InstanceOp>(decl.getDefiningOp()))
      return decl.getDefiningOp()->emitError("Instance input `")
             << getInstPortName(decl) << "`";
    if (isa<chirrtl::MemoryPortOp>(decl.getDefiningOp()))
      return decl.getDefiningOp()->emitError("Memory port `")
             << getCMemPortName(decl) << "`";
    if (isa<MemOp>(decl.getDefiningOp()))
      return decl.getDefiningOp()->emitError("Memory port `")
             << getMemPortName(decl) << "`";
    return decl.getDefiningOp()->emitError("Wire");
  }();
  if (fieldID == 0) {
    err << " is not initialized.";
  } else {
    err << " is not fully initialized at field `"
        << nameForField(decl.getType(), fieldID) << "`.";
  }
  failed = true;
}

// Check (recursively) that decl at fieldID, whose type at fieldID is t, has a
// bit set in vals.
// Pending improvement is to check children of an aggregate and if all of them
// are uninitialized, then emit the error at that point instead of for each
// field.
void CheckInitPass::reportRegion(Value decl, Type t, size_t fieldID,
                                 bool needed, bool alwaysNeeded,
                                 BitVector &vals) {
  LLVM_DEBUG(llvm::errs() << "Test[" << fieldID << "]: \n\t" << decl << " \n\t"
                          << t << " \n\t";
             dumpBV(vals);
             llvm::errs() << " n:" << needed << " a:" << alwaysNeeded << "\n";);
  // May recursively handle subtypes.
  if ((!needed && !alwaysNeeded) || checkBit(vals, fieldID) ||
      type_isa<RefType>(t) || type_isa<AnalogType>(t))
    return;

  // Explicitly test each subtype.
  FIRRTLTypeSwitch<Type>(t)
      .Case<BundleType, OpenBundleType>([&](auto bundle) {
        for (size_t idx = 0, e = bundle.getNumElements(); idx != e; ++idx)
          reportRegion(decl, bundle.getElementType(idx),
                       fieldID + bundle.getFieldID(idx),
                       needed ^ bundle.getElement(idx).isFlip, alwaysNeeded,
                       vals);
      })
      .Case<FVectorType, OpenVectorType>([&](auto vec) {
        for (size_t idx = 0, e = vec.getNumElements(); idx != e; ++idx)
          reportRegion(decl, vec.getElementType(),
                       fieldID + vec.getFieldID(idx), needed, alwaysNeeded,
                       vals);
      })
      .Default([&](auto) { emitInitError(decl, fieldID); });
}

// compute the values set by op's regions.  A when, for example, ands the init
// set as only fields set on both paths are unconditionally set by the when.
void CheckInitPass::processOp(Operation *op, FieldSource &fieldSource) {
  LLVM_DEBUG(llvm::errs() << "* " << op << " r(" << op->getNumRegions()
                          << ")\n";);

  for (auto &region : op->getRegions()) {
    auto &local = localInfo[&region];
    for (auto &block : region) {
      for (auto &opref : block) {
        Operation *op = &opref;
        if (isa<WireOp>(op)) {
          local.dests.push_back(op->getResult(0));
        } else if (auto mem = dyn_cast<MemOp>(op)) {
          for (auto result : mem.getResults())
            local.dests.push_back(result);
        } else if (auto memport = dyn_cast<chirrtl::MemoryPortOp>(op)) {
          // Inferred Memory Ports are weird and we don't try to check them
        } else if (auto inst = dyn_cast<InstanceOp>(op)) {
          for (auto [idx, arg] : llvm::enumerate(inst.getResults()))
            local.dests.push_back(arg);
        } else if (auto inst = dyn_cast<InstanceChoiceOp>(op)) {
          for (auto [idx, arg] : llvm::enumerate(inst.getResults()))
            local.dests.push_back(arg);
        } else if (auto con = dyn_cast<ConnectOp>(op)) {
          auto node = fieldSource.nodeForValue(con.getDest());
          markWrite(local.init[node.src], node.src, node.fieldID);
        } else if (auto con = dyn_cast<StrictConnectOp>(op)) {
          auto node = fieldSource.nodeForValue(con.getDest());
          markWrite(local.init[node.src], node.src, node.fieldID);
        } else if (auto def = dyn_cast<RefDefineOp>(op)) {
          auto node = fieldSource.nodeForValue(def.getDest());
          markWrite(local.init[node.src], node.src, node.fieldID);
        } else if (auto prop = dyn_cast<PropAssignOp>(op)) {
          auto node = fieldSource.nodeForValue(prop.getDest());
          markWrite(local.init[node.src], node.src, node.fieldID);
        } else if (isa<WhenOp, MatchOp>(op)) {
          local.children.push_back(op);
          worklist.push_back(op);
        }
      }
    }
  }
}

void CheckInitPass::runOnOperation() {
  failed = false;

  auto &fieldSource = getAnalysis<FieldSource>();
  worklist.push_back(getOperation());

  // Late size check as it is growing.
  for (size_t i = 0; i < worklist.size(); ++i) {
    Operation *op = worklist[i];
    processOp(op, fieldSource);
  }

  // Modules are the only blocks with arguments, so capture them here only.
  auto &topLevel = localInfo[&getOperation().getRegion()];
  for (auto arg : getOperation().getBodyBlock()->getArguments())
    topLevel.dests.push_back(arg);

  // Post-order traversal.
  for (auto opiter = worklist.rbegin(); opiter != worklist.rend(); ++opiter) {
    for (auto &r : (*opiter)->getRegions()) {
      auto &state = localInfo[&r];

      for (auto *op : state.children) {
        LLVM_DEBUG(llvm::errs()
                       << "+ " << op << " r(" << op->getNumRegions() << ")\n";);
        // Union the child's regions.  This is safe as all multi-region ops are
        // exclusive control flow.
        assert(localInfo.count(&op->getRegion(0)));
        SetSet result = localInfo[&op->getRegion(0)].init;
        for (size_t i = 1, e = op->getNumRegions(); i < e; ++i) {
          assert(localInfo.count(&op->getRegion(i)));
          intersectSetSet(result, localInfo[&op->getRegion(i)].init);
        }
        // Else blocks get elided, causing a hole in the loop above.
        if (isa<WhenOp>(op) && op->getNumRegions() == 1)
          result.clear();

        // Merge anything initialized in all paths into the current region's
        // set.  Not all ops will have
        unionSetSet(state.init, result);
      }

      for (auto v : state.dests) {
        bool needed = true;
        bool alwaysNeeded = false;
        if (auto b = dyn_cast<BlockArgument>(v)) {
          needed = getOperation().getPortDirection(b.getArgNumber()) ==
                   Direction::Out;
        } else if (auto inst = dyn_cast<InstanceOp>(v.getDefiningOp())) {
          needed = inst.getPortDirection(cast<OpResult>(v).getResultNumber()) ==
                   Direction::In;
        } else if (isa<WireOp>(v.getDefiningOp())) {
          // Wires need both flip directions to be connected.
          alwaysNeeded = true;
        }
        reportRegion(v, v.getType(), 0, needed, alwaysNeeded, state.init[v]);
      }
    }
  }

  if (failed)
    signalPassFailure();

  // Prepare for next run.
  worklist.clear();
  localInfo.clear();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckInitPass() {
  return std::make_unique<CheckInitPass>();
}
