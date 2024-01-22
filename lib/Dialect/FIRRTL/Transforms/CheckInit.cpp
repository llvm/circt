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

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class CheckInitPass : public CheckInitBase<CheckInitPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void CheckInitPass::runOnOperation() {
  DenseMap<Value, BitVector> unsetFields;
  auto &fieldsource = getAnalysis<FieldSource>();

  for (auto [idx, arg] :
       llvm::enumerate(getOperation().getBodyBlock()->getArguments()))
    markLeaves(unsetFields[arg], arg.getType(), true,
               getOperation().getPortDirection(idx) == Direction::Out);

  getOperation().walk([&](Operation *op) {
    if (auto wire = dyn_cast<WireOp>(op)) {
      assert(unsetFields.count(wire.getResult()) == 0);
      auto &bits = unsetFields[wire.getResult()];
      markLeaves(bits, wire.getResult().getType());
    } else if (isa<RegOp, RegResetOp>(op)) {
      assert(unsetFields.count(op->getResult(0)) == 0);
      auto &bits = unsetFields[op->getResult(0)];
      markLeaves(bits, op->getResult(0).getType());
      bits.reset(); // Don't care about tracking writes
    } else if (auto mem = dyn_cast<MemOp>(op)) {
      for (auto result : mem.getResults()) {
        assert(unsetFields.count(result) == 0);
        auto &bits = unsetFields[result];
        markLeaves(bits, result.getType());
      }
    } else if (auto memport = dyn_cast<chirrtl::MemoryPortOp>(op)) {
      assert(unsetFields.count(memport.getResult(0)) == 0);
      auto &bits = unsetFields[memport.getResult(0)];
      markLeaves(bits, memport.getResult(0).getType());
      bits.reset();
    } else if (auto inst = dyn_cast<InstanceOp>(op)) {
      for (auto [idx, arg] : llvm::enumerate(inst.getResults())) {
        assert(unsetFields.count(arg) == 0);
        auto &bits = unsetFields[arg];
        markLeaves(bits, arg.getType(), true,
                   inst.getPortDirection(idx) == Direction::In);
      }
    } else if (auto inst = dyn_cast<InstanceChoiceOp>(op)) {
      for (auto [idx, arg] : llvm::enumerate(inst.getResults())) {
        assert(unsetFields.count(arg) == 0);
        auto &bits = unsetFields[arg];
        markLeaves(bits, arg.getType(), true,
                   inst.getPortDirection(idx) == Direction::In);
      }
    } else if (auto con = dyn_cast<ConnectOp>(op)) {
      auto *node = fieldsource.nodeForValue(con.getDest());
      clearUnder(unsetFields[node->src], node->src.getType(), node->path);
    } else if (auto con = dyn_cast<StrictConnectOp>(op)) {
      auto *node = fieldsource.nodeForValue(con.getDest());
      clearUnder(unsetFields[node->src], node->src.getType(), node->path);
    } else if (auto def = dyn_cast<RefDefineOp>(op)) {
      auto *node = fieldsource.nodeForValue(def.getDest());
      clearUnder(unsetFields[node->src], node->src.getType(), node->path);
    }
  });

  for (auto &[v, bits] : unsetFields) {
    for (auto b : bits.set_bits()) {
      getOperation()->dump();
      if (auto *op = v.getDefiningOp())
        op->emitError("unset field: ") << b;
      else if (auto bv = dyn_cast<BlockArgument>(v))
        bv.getOwner()->getParentOp()->emitError("unset field on port: ") << b;
    }
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckInitPass() {
  return std::make_unique<CheckInitPass>();
}
