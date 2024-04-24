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
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/BitVector.h"
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

  for (auto mod : circuit.getOps<FModuleOp>()) {
    for (auto port : mod.getBodyBlock()->getArguments()) {
        if (!port.hasOneUse() && !port.use_empty())
            continue;
        for (auto use : port.getUsers()) {
            if (isa<TailPrimOp>(use)) {
                llvm::errs() << "Tail\n";
            } else if (isa<HeadPrimOp>(use)) {
                llvm::errs() << "Head\n";
            } else if (isa<BitsPrimOp>(use)) {
                llvm::errs() << "Bits\n";
            }
        }
      }
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createNarrowPortsPass() {
  return std::make_unique<NarrowPortsPass>();
}
