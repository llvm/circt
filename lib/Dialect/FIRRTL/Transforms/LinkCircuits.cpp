//===- LinkCircuits.cpp - Merge FIRRTL circuits --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass links multiple circuits into a single one.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LINKCIRCUITS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
struct LinkCircuitsPass
    : public circt::firrtl::impl::LinkCircuitsBase<LinkCircuitsPass> {
  void runOnOperation() override;
  void mergeCircuits();
  void linkExtmodules();
  LinkCircuitsPass(StringRef baseCircuitNameOption, bool noMangleOption) {
    baseCircuitName = std::string(baseCircuitNameOption);
    noMangle = noMangleOption;
  }
};
} // namespace

static LogicalResult mangleCircuitSymbols(CircuitOp circuit) {
  auto circuitName = circuit.getNameAttr();

  llvm::MapVector<StringRef, Operation *> renameTable;
  auto symbolTable = SymbolTable(circuit.getOperation());
  auto manglePrivateSymbol = [&](SymbolOpInterface symbolOp) {
    auto symbolName = cast<StringAttr>(symbolOp->getAttr("sym_name"));
    auto newSymbolName =
        StringAttr::get(symbolOp->getContext(),
                        circuitName.getValue() + "_" + symbolName.getValue());
    renameTable.insert(std::pair(symbolName.getValue(), symbolOp));
    return symbolTable.rename(symbolOp, newSymbolName);
  };

  for (auto &op : circuit.getOps()) {
    auto symbolOp = dyn_cast<SymbolOpInterface>(op);
    if (!symbolOp)
      continue;

    if (symbolOp.isPrivate())
      if (failed(manglePrivateSymbol(symbolOp)))
        return failure();
  }

  circuit.walk([&](Operation *op) {
    TypeSwitch<Operation *, void>(op)
        .Case<ObjectOp>([&](ObjectOp obj) {
          auto resultTypeName = obj.getResult().getType().getName();
          if (auto *newOp = renameTable.lookup(resultTypeName))
            obj.getResult().setType(dyn_cast<ClassOp>(newOp).getInstanceType());
        })
        .Default([](Operation *op) {});
  });
  return success();
}

void LinkCircuitsPass::mergeCircuits() {
  auto module = getOperation();

  SmallVector<CircuitOp> circuits;
  for (CircuitOp circuitOp : module.getOps<CircuitOp>()) {
    circuits.push_back(circuitOp);
  }

  auto mergedCircuit = OpBuilder(module).create<CircuitOp>(
      module.getLoc(), StringAttr::get(&getContext(), baseCircuitName));

  for (auto circuit : circuits) {
    if (!noMangle)
      if (failed(mangleCircuitSymbols(circuit)))
        circuit->emitError("failed to mangle private symbol");

    // TODO: merge annotations
    SmallVector<Operation *> opsToMove;
    for (auto &op : circuit.getOps())
      opsToMove.push_back(&op);
    for (auto *op : opsToMove)
      op->moveBefore(mergedCircuit.getBodyBlock(),
                     mergedCircuit.getBodyBlock()->end());

    circuit->erase();
  }

  module.insert(&module.getBody()->getOperations().front(), mergedCircuit);
}

void LinkCircuitsPass::linkExtmodules() {
  auto circuit =
      cast<CircuitOp>(getOperation().getBody()->getOperations().front());
  auto symbolTable = SymbolTable(circuit.getOperation());

  SmallVector<FExtModuleOp *> declsToReplace;
  for (auto extModule : circuit.getOps<FExtModuleOp>()) {
    auto *potentialModule = symbolTable.lookup(extModule.getSymNameAttr());
    if (!potentialModule)
      continue;

    auto compareAttr = [&extModule, &potentialModule](StringRef attrName) {
      return extModule->getAttr(attrName) == potentialModule->getAttr(attrName);
    };
    for (auto *attrName :
         {"convention", "portDirections	", "portAnnotations", "portSymbols",
          "portNames", "portTypes", "annotations", "layers"})
      if (!compareAttr(attrName))
        continue;

    declsToReplace.push_back(&extModule);
  };
  for (auto *decl : declsToReplace)
    decl->erase();
}

void LinkCircuitsPass::runOnOperation() {
  mergeCircuits();
  linkExtmodules();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createLinkCircuitsPass(StringRef baseCircuitName,
                                      bool noMangle) {
  return std::make_unique<LinkCircuitsPass>(baseCircuitName, noMangle);
}
