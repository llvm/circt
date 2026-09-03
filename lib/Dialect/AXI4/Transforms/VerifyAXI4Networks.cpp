//===- VerifyAXI4Networks.cpp - Verify AXI4 networks ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifies the properties of an AXI4 network that span more than one operation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace axi4 {
#define GEN_PASS_DEF_VERIFYAXI4NETWORKS
#include "circt/Dialect/AXI4/AXI4Passes.h.inc"
} // namespace axi4
} // namespace circt

using namespace circt;
using namespace axi4;
using namespace mlir;

namespace {
/// The clock and reset an AXI4 op operates in.
struct Domain {
  Value clock, reset;
};
} // namespace

/// The domain of an AXI4 op, or failure for one this pass does not know.
static FailureOr<Domain> getDomain(Operation *op) {
  return TypeSwitch<Operation *, FailureOr<Domain>>(op)
      .Case<AbstractManagerOp, AbstractSubordinateOp, ChannelStructsToPortOp,
            PortToChannelStructsOp, XbarOp>([](auto op) {
        return Domain{op.getClock(), op.getReset()};
      })
      .Default([](Operation *op) -> FailureOr<Domain> {
        op->emitOpError("unsupported AXI4 network op; cannot verify which "
                        "clock and reset domain it is in");
        return failure();
      });
}

/// Report a port value with more than one consumer, or with none at all.
static LogicalResult verifyPortUses(Value port) {
  if (!isa<PortType>(port.getType()))
    return success();
  if (port.use_empty()) {
    mlir::emitWarning(port.getLoc())
        << "AXI4 port has no uses, so takes no part in a network";
    return success();
  }
  if (port.hasNUsesOrMore(2))
    return mlir::emitError(port.getLoc())
           << "AXI4 port must have at most one use; route through an "
              "'axi4.xbar' to fan out to multiple endpoints";
  return success();
}

/// Report two ops connected by a port but operating in different domains.
static void emitDomainCrossing(Operation *op, Operation *other,
                               StringRef domain) {
  auto diag = op->emitOpError()
              << "is in a different " << domain << " domain to the '"
              << other->getName().getStringRef() << "' connected to it";
  diag.attachNote(other->getLoc()) << "connected operation here";
}

namespace {
struct VerifyAXI4NetworksPass
    : public circt::axi4::impl::VerifyAXI4NetworksBase<VerifyAXI4NetworksPass> {
  void runOnOperation() override;
};
} // namespace

void VerifyAXI4NetworksPass::runOnOperation() {
  ModuleOp module = getOperation();
  Dialect *axi4Dialect = module->getContext()->getLoadedDialect<AXI4Dialect>();
  bool anyFailed = false;

  // Check uses of all axi4.port values
  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      if (failed(verifyPortUses(result)))
        anyFailed = true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (failed(verifyPortUses(arg)))
            anyFailed = true;
  });

  // Ensure connected ops are in the same clock and reset domains
  module.walk([&](Operation *op) {
    if (op->getDialect() != axi4Dialect)
      return;
    FailureOr<Domain> domain = getDomain(op);
    if (failed(domain)) {
      anyFailed = true;
      return;
    }

    for (Value operand : op->getOperands()) {
      if (!isa<PortType>(operand.getType()))
        continue;
      // A port arriving from outside the module carries no comparable clock.
      Operation *upstream = operand.getDefiningOp();
      if (!upstream || upstream->getDialect() != axi4Dialect)
        continue;

      FailureOr<Domain> upstreamDomain = getDomain(upstream);
      if (failed(upstreamDomain)) {
        anyFailed = true;
        continue;
      }
      if (domain->clock != upstreamDomain->clock) {
        emitDomainCrossing(op, upstream, "clock");
        anyFailed = true;
      }
      if (domain->reset != upstreamDomain->reset) {
        emitDomainCrossing(op, upstream, "reset");
        anyFailed = true;
      }
    }
  });

  if (anyFailed)
    signalPassFailure();
}
