//===- ESIVerifyConnections.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace esi {
#define GEN_PASS_DEF_VERIFYESICONNECTIONS
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace circt;
using namespace circt::esi;

namespace {
struct ESIVerifyConnectionsPass
    : public impl::VerifyESIConnectionsBase<ESIVerifyConnectionsPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESIVerifyConnectionsPass::runOnOperation() {
  // Walk the tree and look for ops which produce ESI types. Check each one.
  getOperation()->walk([this](Operation *op) {
    for (const OpResult &v : op->getResults())
      if (isa<ChannelBundleType>(v.getType())) {
        if (v.hasOneUse())
          continue;
        mlir::InFlightDiagnostic error =
            op->emitError("bundles must have exactly one use");
        for (Operation *user : v.getUsers())
          error.attachNote(user->getLoc()) << "bundle used here";
        signalPassFailure();

      } else if (auto cv = dyn_cast<mlir::TypedValue<ChannelType>>(v)) {
        if (failed(ChannelType::verifyChannel(cv)))
          signalPassFailure();
      }
  });
}

std::unique_ptr<OperationPass<>> circt::esi::createESIVerifyConnectionsPass() {
  return std::make_unique<ESIVerifyConnectionsPass>();
}
