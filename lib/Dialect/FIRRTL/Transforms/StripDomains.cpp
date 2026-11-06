//===- StripDomains.cpp - Strip FIRRTL domains ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_STRIPDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using llvm::concat;
using mlir::ReverseIterator;

/// A helper for stripping domains from a module based on a predicate. The
/// predicate takes a domain name and returns true if that domain should be
/// stripped.
static LogicalResult
stripModuleImpl(FModuleLike op,
                llvm::function_ref<bool(StringAttr)> shouldStripDomain) {
  auto shouldStripType = [&](Type type) {
    if (auto domainType = dyn_cast<DomainType>(type))
      return shouldStripDomain(domainType.getName().getAttr());
    return false;
  };
  WalkResult result = op->walk<mlir::WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation *op) -> WalkResult {
        return TypeSwitch<Operation *, WalkResult>(op)
            .Case<FModuleLike>([&](FModuleLike op) {
              BitVector erasures(op.getNumPorts());
              for (size_t i = 0, e = op.getNumPorts(); i < e; ++i)
                if (shouldStripType(op.getPortType(i)))
                  erasures.set(i);
              if (erasures.any())
                op.erasePorts(erasures);
              return WalkResult::advance();
            })
            .Case<DomainDefineOp>([&](DomainDefineOp op) {
              if (shouldStripType(op.getDest().getType()) ||
                  shouldStripType(op.getSrc().getType()))
                op.erase();
              return WalkResult::advance();
            })
            .Case<DomainCreateOp>([&](DomainCreateOp op) {
              if (shouldStripType(op.getType()))
                op.erase();
              return WalkResult::advance();
            })
            .Case<DomainCreateAnonOp>([&](DomainCreateAnonOp op) {
              if (shouldStripType(op.getType()))
                op.erase();
              return WalkResult::advance();
            })
            .Case<DomainSubfieldOp>([&](DomainSubfieldOp op) {
              // The subfield's result is a property value; decide
              // whether to strip based on the domain it reads from.
              if (shouldStripType(op.getInput().getType())) {
                if (!op->use_empty()) {
                  OpBuilder builder(op);
                  op.replaceAllUsesWith(
                      UnknownValueOp::create(builder, op.getLoc(), op.getType())
                          .getResult());
                }
                op.erase();
              }
              return WalkResult::advance();
            })
            .Case<UnsafeDomainCastOp>([&](UnsafeDomainCastOp op) {
              // Strip cast if any of the domains being cast should be
              // stripped.
              if (llvm::any_of(op.getDomains(), [&](Value domain) {
                    return shouldStripType(domain.getType());
                  })) {
                op.replaceAllUsesWith(op.getInput());
                op.erase();
              }
              return WalkResult::advance();
            })
            .Case<WireOp>([&](WireOp op) {
              // Erase wires of DomainType that should be stripped.
              if (shouldStripType(op.getType(0))) {
                op->erase();
                return WalkResult::advance();
              }
              BitVector erasures(op.getDomains().size());

              // Erase domain operands from regular wires.
              for (int i = 0, e = op.getDomains().size(); i < e; ++i)
                if (shouldStripType(op.getDomains()[i].getType()))
                  erasures.set(i);

              op->eraseOperands(erasures);
              return WalkResult::advance();
            })
            .Case<FInstanceLike>([&](auto op) {
              auto n = op.getNumPorts();
              BitVector erasures(n);
              for (size_t i = 0; i < n; ++i)
                if (shouldStripType(op->getResult(i).getType()))
                  erasures.set(i);
              if (erasures.any()) {
                op.cloneWithErasedPortsAndReplaceUses(erasures);
                op.erase();
              }
              return WalkResult::advance();
            })
            .Default([&](Operation *op) {
              // All operations that can have DomainType are handled
              // above. If we encounter one here, it's a bug in the IR
              // or this pass.
              for (auto type :
                   concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
                if (isa<DomainType>(type)) {
                  op->emitOpError("cannot be stripped");
                  return WalkResult::interrupt();
                }
              }
              return WalkResult::advance();
            });
      });
  return failure(result.wasInterrupted());
}

LogicalResult firrtl::stripDomainsFromCircuit(
    CircuitOp circuit, llvm::function_ref<bool(StringAttr)> shouldStripDomain) {
  // Collect modules and erase matching DomainOp declarations.
  llvm::SmallVector<FModuleLike> modules;
  for (Operation &op : make_early_inc_range(*circuit.getBodyBlock())) {
    TypeSwitch<Operation *, void>(&op)
        .Case<FModuleLike>([&](FModuleLike op) { modules.push_back(op); })
        .Case<DomainOp>([&](DomainOp op) {
          // Erase domain declaration if its name should be stripped.
          if (shouldStripDomain(op.getNameAttr()))
            op.erase();
        });
  }

  // Strip domains from all modules in parallel.
  return failableParallelForEach(
      circuit.getContext(), modules, [&](FModuleLike module) {
        return stripModuleImpl(module, shouldStripDomain);
      });
}

namespace {
struct StripDomainsPass : public impl::StripDomainsBase<StripDomainsPass> {
  void runOnOperation() override {
    if (failed(stripDomainsFromCircuit(getOperation(),
                                       [](StringAttr) { return true; })))
      signalPassFailure();
  }
};
} // namespace
