//===- LowerSignatures.cpp - Lower Module Signatures ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerSignatures pass.  This pass replaces aggregate
// types with expanded values in module arguments as specified by the ABI
// information.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

#define DEBUG_TYPE "narrow-ports"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct NarrowPortsPass : public NarrowPortsBase<NarrowPortsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void NarrowPortsPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");

  auto circuit = getOperation();
    SymbolTable syms(circuit);

  // for each module, which are the operations which limit the ports.  By construction, each port will have a single operation which needs to be moved across the boundary.
  DenseMap<FModuleOp, SmallVector<Operation*> > targets;

  for (auto mod : circuit.getOps<FModuleOp>()) {
    if (mod.isPublic())
        continue;
    for (auto port : mod.getBodyBlock()->getArguments()) {
        if (!port.hasOneUse() && !port.use_empty())
            continue;
        for (auto use : port.getUsers()) {
            if (isa<TailPrimOp>(use)) {
                llvm::errs() << "Tail\n";
            } else if (isa<HeadPrimOp>(use)) {
                llvm::errs() << "Head\n";
            } else if (isa<BitsPrimOp>(use)) {
                llvm::errs() << "Bits " << mod.getName() << "::" << mod.getPortName(port.getArgNumber()) << "\n";
                use->dump();
                targets[mod].push_back(use);
            }
        }
      }
  }

    llvm::errs() << "\n";

    // Now update all uses
  for (auto mod : circuit.getOps<FModuleOp>()) {
    mod.walk([&](InstanceOp inst) {
        auto target = dyn_cast<FModuleOp>(inst.getReferencedOperation(syms));
        if (!target)
            return;        
        auto ii = targets.find(target);
        if (ii == targets.end())
            return;
        llvm::errs() << "Updating...\n";
        for (auto* op : ii->second) {
            ImplicitLocOpBuilder builder(op->getLoc(), inst);
            builder.setInsertionPointAfter(inst);
            auto argnum = cast<BlockArgument>(op->getOperand(0)).getArgNumber();
            auto instres = inst.getResult(argnum);
            auto bounceWire = builder.create<WireOp>(op->getResult(0).getType());
            instres.replaceAllUsesWith(bounceWire.getResult());
            auto newOp = builder.clone(*op);
            builder.create<StrictConnectOp>(instres, newOp->getResult(0));
        }
    });
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createNarrowPortsPass() {
  return std::make_unique<NarrowPortsPass>();
}
