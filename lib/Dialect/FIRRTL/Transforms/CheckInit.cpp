//===- CheckInit.cpp - Ensure all wires are initialized ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CheckInit pass.  This pass checks that all wires are
// initialized (connected too).
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

static void clearUnder(BitVector &bits, Type t, ArrayRef<int64_t> path,
                       uint64_t fieldBase = 0) {
  if (auto bundle = dyn_cast<BundleType>(t)) {
    if (path.empty()) {
      for (size_t idx = 0, e = bundle.getNumElements(); idx != e; ++idx)
        clearUnder(bits, bundle.getElementType(idx), path,
                   fieldBase + bundle.getFieldID(idx));
    } else {
      clearUnder(bits, bundle.getElementType(path.front()), path.drop_front(),
                 fieldBase + bundle.getFieldID(path.front()));
    }
  } else if (auto bundle = dyn_cast<OpenBundleType>(t)) {
    if (path.empty()) {
      for (size_t idx = 0, e = bundle.getNumElements(); idx != e; ++idx)
        clearUnder(bits, bundle.getElementType(idx), path,
                   fieldBase + bundle.getFieldID(idx));
    } else {
      clearUnder(bits, bundle.getElementType(path.front()), path.drop_front(),
                 fieldBase + bundle.getFieldID(path.front()));
    }
  } else if (auto vec = dyn_cast<FVectorType>(t)) {
    if (path.empty()) {
      for (size_t idx = 0, e = vec.getNumElements(); idx != e; ++idx)
        clearUnder(bits, vec.getElementType(), path,
                   fieldBase + vec.getFieldID(idx));
    } else {
      clearUnder(bits, vec.getElementType(), path.drop_front(),
                 fieldBase + vec.getFieldID(path.front()));
    }
  } else if (auto vec = dyn_cast<OpenVectorType>(t)) {
    if (path.empty()) {
      for (size_t idx = 0, e = vec.getNumElements(); idx != e; ++idx)
        clearUnder(bits, vec.getElementType(), path,
                   fieldBase + vec.getFieldID(idx));
    } else {
      clearUnder(bits, vec.getElementType(), path.drop_front(),
                 fieldBase + vec.getFieldID(path.front()));
    }
  } else {
    assert(bits.size() > fieldBase);
    LLVM_DEBUG({
      llvm::errs() << "found " << fieldBase;
      if (bits[fieldBase])
        llvm::errs() << " needed";
      llvm::errs() << "\n";
    });
    bits.reset(fieldBase);
  }
}

static void markLeaves(BitVector &bits, Type t, bool isPort = false,
                       bool isFlip = false, uint64_t fieldBase = 0) {
  LLVM_DEBUG({
    llvm::errs() << "port:" << isPort << " flip:" << isFlip
                 << " id:" << fieldBase << " ";
    t.dump();
  });
  if (auto bundle = dyn_cast<BundleType>(t)) {
    for (size_t idx = 0, e = bundle.getNumElements(); idx != e; ++idx)
      markLeaves(bits, bundle.getElementType(idx), isPort,
                 isFlip ^ bundle.getElement(idx).isFlip,
                 fieldBase + bundle.getFieldID(idx));
  } else if (auto bundle = dyn_cast<OpenBundleType>(t)) {
    for (size_t idx = 0, e = bundle.getNumElements(); idx != e; ++idx)
      markLeaves(bits, bundle.getElementType(idx), isPort,
                 isFlip ^ bundle.getElement(idx).isFlip,
                 fieldBase + bundle.getFieldID(idx));
  } else if (auto vec = dyn_cast<FVectorType>(t)) {
    for (size_t idx = 0, e = vec.getNumElements(); idx != e; ++idx)
      markLeaves(bits, vec.getElementType(), isPort, isFlip,
                 fieldBase + vec.getFieldID(idx));
  } else if (auto vec = dyn_cast<OpenVectorType>(t)) {
    for (size_t idx = 0, e = vec.getNumElements(); idx != e; ++idx)
      markLeaves(bits, vec.getElementType(), isPort, isFlip,
                 fieldBase + vec.getFieldID(idx));
  } else {
    if (isPort && !isFlip)
      return;
    LLVM_DEBUG({ llvm::errs() << "need " << fieldBase << "\n"; });
    bits.resize(std::max(fieldBase + 1, (uint64_t)bits.size()));
    bits.set(fieldBase);
  }
}

static void markWrite(BitVector& bv, Value v, size_t fieldID) {
    if (bv.size() <= fieldID)
        bv.resize(fieldID + 1);
    llvm::errs() << "S " << v << " " << fieldID << "\n";
    bv.set(fieldID);
}

namespace {
class CheckInitPass : public CheckInitBase<CheckInitPass> {
  // SetSets track initialized fieldIDs
  using SetSet = DenseMap<Value, BitVector>;
  struct RegionState {
    // Values Inited in this state's regions.  To be intersected.
    SetSet init;
    // Children to merge into this state
    SmallVector<Operation *, 4> children;
    // Definitions from this state
    SmallVector<Value> dests;
  };

  SmallVector<Operation *> worklist;
  DenseMap<Region*, RegionState> localInfo;

  void processOp(Operation *op, FieldSource &fieldSource);
  void reportRegion(Value decl, Type t, size_t fieldID, bool needed, BitVector& vals);
  void emitInitError(Value decl, size_t fieldID);

public:
  void runOnOperation() override;
};
} // end anonymous namespace

static bool checkBit(BitVector& vals, size_t idx) {
  if (vals.size() <= idx)
    return false;
  return vals[idx];
}

void CheckInitPass::emitInitError(Value decl, size_t fieldID) {
  bool isPort = isa<BlockArgument>(decl);
  auto err = isPort ? getOperation()->emitError("Port is not") :
  decl.getDefiningOp()->emitError("Wire is not");
  if (fieldID == 0) {
    err << " initialized.";
  } else {
    err << " fully initialized. (" << fieldID << ")";
  }
}

// Check (recursively) that decl at fieldID, whose type at fieldID is t, has a
// bit set in vals
void CheckInitPass::reportRegion(Value decl, Type t, size_t fieldID, bool needed, BitVector& vals ) {
  llvm::errs() << "Test[" << fieldID << "]: " << decl << "\n";
  // May recursively handle subtypes.
  if (!needed || checkBit(vals, fieldID))
    return;

  // Explicitly test each subtype.
  if (auto bundle = dyn_cast<BundleType>(t)) {
    for (size_t idx = 0, e = bundle.getNumElements(); idx != e; ++idx)
      reportRegion(decl, bundle.getElementType(idx), 
                 needed ^ bundle.getElement(idx).isFlip,
                 fieldID + bundle.getFieldID(idx), vals);
    return;
  } else if (auto bundle = dyn_cast<OpenBundleType>(t)) {
    for (size_t idx = 0, e = bundle.getNumElements(); idx != e; ++idx)
      reportRegion(decl, bundle.getElementType(idx), 
                 needed ^ bundle.getElement(idx).isFlip,
                 fieldID + bundle.getFieldID(idx), vals);
    return;
  } else if (auto vec = dyn_cast<FVectorType>(t)) {
    for (size_t idx = 0, e = vec.getNumElements(); idx != e; ++idx)
      reportRegion(decl, vec.getElementType(), needed,
                 fieldID + vec.getFieldID(idx), vals);
    return;
  } else if (auto vec = dyn_cast<OpenVectorType>(t)) {
    for (size_t idx = 0, e = vec.getNumElements(); idx != e; ++idx)
      reportRegion(decl, vec.getElementType(), needed,
                 fieldID + vec.getFieldID(idx), vals);
    return;
  }

  emitInitError(decl, fieldID);
}

// compute the values set by op's regions.  A when, for example, ands the init
// set as only fields set on both paths are unconditionally set by the when.
void CheckInitPass::processOp(Operation *op, FieldSource &fieldSource) {
  llvm::errs() << "* " << op << "\n";

  for (auto &region : op->getRegions()) {
    auto &local = localInfo[&region];
    for (auto &block : region) {
      for (auto &opref : block) {
        Operation *op = &opref;
        if (isa<WireOp, RegOp, RegResetOp>(op)) {
          local.dests.push_back(op->getResult(0));
        } else if (auto mem = dyn_cast<MemOp>(op)) {
          for (auto result : mem.getResults())
            local.dests.push_back(result);
        } else if (auto memport = dyn_cast<chirrtl::MemoryPortOp>(op)) {
          local.dests.push_back(memport.getResult(0));
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
        } else if (isa<WhenOp, MatchOp>(op)) {
          local.children.push_back(op);
          worklist.push_back(op);
        }
      }
    }
  }
}

void CheckInitPass::runOnOperation() {
  auto &fieldSource = getAnalysis<FieldSource>();
  worklist.push_back(getOperation());

  // Late size check as it is growing.
  for (size_t i = 0; i < worklist.size(); ++i)  {
    Operation *op = worklist[i];
    processOp(op, fieldSource);
  }

  // Modules are the only blocks with arguments, so capture them here only.
  auto& topLevel = localInfo[&getOperation().getRegion()];
  for (auto arg :
       getOperation().getBodyBlock()->getArguments())
    topLevel.dests.push_back(arg);

  // Post-order traversal.
  for (auto opiter = worklist.rbegin(); opiter != worklist.rend(); ++opiter) {
    for (auto& r : (*opiter)->getRegions()) {
      auto& state = localInfo[&r];
      for (auto v: state.dests) {
        bool needed = true;
        if (auto b = dyn_cast<BlockArgument>(v))
          needed = getOperation().getPortDirection(b.getArgNumber()) == Direction::Out;
        reportRegion(v, v.getType(), 0, needed, state.init[v]);
      }
      llvm::errs() << "\n";
      llvm::errs() << "children: " << state.children.size() << "\n";
      for (auto ii = state.init.begin(), ee = state.init.end(); ii != ee; ++ii) {
        llvm::errs() << "\t" << ii->first << ": ";
        for (auto bi = ii->second.set_bits_begin(), be = ii->second.set_bits_end(); bi != be; ++bi)
          llvm::errs() <<  *bi;
        llvm::errs() << "\n";
        for (size_t bi = 0, be = ii->second.size(); bi != be; ++bi)
          llvm::errs() <<  ii->second[bi];
        llvm::errs() << "\n";

      }
    }
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckInitPass() {
  return std::make_unique<CheckInitPass>();
}
