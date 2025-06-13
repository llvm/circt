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

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>

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
  LogicalResult mergeCircuits();
  LinkCircuitsPass(StringRef baseCircuitNameOption, bool noMangleOption) {
    baseCircuitName = std::string(baseCircuitNameOption);
    noMangle = noMangleOption;
  }
};
} // namespace

template <typename CallableT>
static DictionaryAttr transformAnnotationTarget(DictionaryAttr anno,
                                                CallableT transformTokensFn) {
  return DictionaryAttr::getWithSorted(
      anno.getContext(),
      to_vector(map_range(
          anno.getValue(), [&](NamedAttribute namedAttr) -> NamedAttribute {
            if (namedAttr.getName() == "target")
              if (auto target = dyn_cast<StringAttr>(namedAttr.getValue()))
                if (auto tokens = tokenizePath(target.getValue()))
                  return {
                      namedAttr.getName(),
                      StringAttr::get(target.getContext(),
                                      transformTokensFn(tokens.value()).str())};
            return namedAttr;
          })));
}

static LogicalResult mangleCircuitSymbols(CircuitOp circuit) {
  auto circuitName = circuit.getNameAttr();

  llvm::MapVector<StringRef, Operation *> renameTable;
  auto symbolTable = SymbolTable(circuit.getOperation());
  auto manglePrivateSymbol = [&](SymbolOpInterface symbolOp) {
    auto symbolName = symbolOp.getNameAttr();
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
    auto updateType = [&](Type type) -> Type {
      if (auto cls = dyn_cast<ClassType>(type))
        if (auto *newOp = renameTable.lookup(cls.getName()))
          return ClassType::get(FlatSymbolRefAttr::get(newOp),
                                cls.getElements());
      return type;
    };
    auto updateTypeAttr = [&](Attribute attr) -> Attribute {
      if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
        auto newType = updateType(typeAttr.getValue());
        if (newType != typeAttr.getValue())
          return TypeAttr::get(newType);
      }
      return attr;
    };
    auto updateResults = [&](auto &&results) {
      for (auto result : results)
        if (auto newType = updateType(result.getType());
            newType != result.getType())
          result.setType(newType);
    };

    TypeSwitch<Operation *>(op)
        .Case<CircuitOp>([&](CircuitOp circuit) {
          SmallVector<Attribute> newAnnotations;
          llvm::transform(
              circuit.getAnnotationsAttr(), std::back_inserter(newAnnotations),
              [&](Attribute attr) {
                return transformAnnotationTarget(
                    cast<DictionaryAttr>(attr), [&](TokenAnnoTarget &tokens) {
                      if (auto *newModule = renameTable.lookup(tokens.module))
                        tokens.module =
                            cast<SymbolOpInterface>(newModule).getName();
                      return tokens;
                    });
              });
          circuit.setAnnotationsAttr(
              ArrayAttr::get(circuit.getContext(), newAnnotations));
        })
        .Case<ObjectOp>([&](ObjectOp obj) {
          auto resultTypeName = obj.getResult().getType().getName();
          if (auto *newOp = renameTable.lookup(resultTypeName))
            obj.getResult().setType(dyn_cast<ClassOp>(newOp).getInstanceType());
        })
        .Case<FModuleOp>([&](FModuleOp module) {
          SmallVector<Attribute> newPortTypes;
          llvm::transform(module.getPortTypesAttr().getValue(),
                          std::back_inserter(newPortTypes), updateTypeAttr);
          module.setPortTypesAttr(
              ArrayAttr::get(module->getContext(), newPortTypes));
          updateResults(module.getBodyBlock()->getArguments());
        })
        .Case<InstanceOp>(
            [&](InstanceOp instance) { updateResults(instance->getResults()); })
        .Case<WireOp>([&](WireOp wire) { updateResults(wire->getResults()); })
        .Default([](Operation *op) {});
  });
  return success();
}

/// return if the incomingOp has been erased
static FailureOr<bool> linkExtmodule(SymbolOpInterface collidingOp,
                                     SymbolOpInterface incomingOp) {
  if (!((isa<FExtModuleOp>(collidingOp) && isa<FModuleOp>(incomingOp)) ||
        (isa<FExtModuleOp>(incomingOp) && isa<FModuleOp>(collidingOp))))
    return failure();
  auto [definition, declaration] = isa<FModuleOp>(collidingOp)
                                       ? std::pair(collidingOp, incomingOp)
                                       : std::pair(incomingOp, collidingOp);
  if (!definition.isPublic())
    return definition->emitOpError("should be a public symbol");
  if (!declaration.isPublic())
    return declaration->emitOpError("should be a public symbol");

  constexpr const StringRef attrsToCompare[] = {
      "portDirections", "portSymbols", "portNames", "portTypes", "layers"};
  auto allAttrsMatch = all_of(attrsToCompare, [&](StringRef attr) {
    return definition->getAttr(attr) == declaration->getAttr(attr);
  });

  if (!allAttrsMatch)
    return false;

  declaration->erase();
  return declaration == incomingOp;
}

LogicalResult LinkCircuitsPass::mergeCircuits() {
  auto module = getOperation();

  SmallVector<CircuitOp> circuits;
  for (CircuitOp circuitOp : module.getOps<CircuitOp>())
    circuits.push_back(circuitOp);

  auto builder = OpBuilder(module);
  builder.setInsertionPointToEnd(module.getBody());
  auto mergedCircuit = builder.create<CircuitOp>(
      module.getLoc(), StringAttr::get(&getContext(), baseCircuitName));
  SmallVector<Attribute> mergedAnnotations;

  for (auto circuit : circuits) {
    if (!noMangle)
      if (failed(mangleCircuitSymbols(circuit)))
        return circuit->emitError("failed to mangle private symbol");

    // TODO: other circuit attributes (such as enable_layers...)
    llvm::transform(circuit.getAnnotations().getValue(),
                    std::back_inserter(mergedAnnotations), [&](Attribute attr) {
                      return transformAnnotationTarget(
                          cast<DictionaryAttr>(attr),
                          [&](TokenAnnoTarget &tokens) {
                            tokens.circuit = mergedCircuit.getName();
                            return tokens;
                          });
                    });

    // reconstruct symbol table after each merge
    auto mergedSymbolTable = SymbolTable(mergedCircuit.getOperation());

    SmallVector<Operation *> opsToMove;
    for (auto &op : circuit.getOps())
      opsToMove.push_back(&op);
    for (auto *op : opsToMove) {
      if (auto symbolOp = dyn_cast<SymbolOpInterface>(op))
        if (auto collidingOp = cast_if_present<SymbolOpInterface>(
                mergedSymbolTable.lookup(symbolOp.getNameAttr()))) {
          auto opErased = linkExtmodule(collidingOp, symbolOp);
          if (failed(opErased))
            return mergedCircuit->emitError("has colliding symbol " +
                                            symbolOp.getName() +
                                            " which cannot be merged");
          if (opErased.value())
            continue;
        }

      op->moveBefore(mergedCircuit.getBodyBlock(),
                     mergedCircuit.getBodyBlock()->end());
    }

    circuit->erase();
  }

  mergedCircuit.setAnnotationsAttr(
      ArrayAttr::get(mergedCircuit.getContext(), mergedAnnotations));

  return mlir::detail::verifySymbolTable(mergedCircuit);
}

void LinkCircuitsPass::runOnOperation() {
  if (failed(mergeCircuits()))
    signalPassFailure();
}

std::unique_ptr<Pass>
circt::firrtl::createLinkCircuitsPass(StringRef baseCircuitName,
                                      bool noMangle) {
  return std::make_unique<LinkCircuitsPass>(baseCircuitName, noMangle);
}
