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
  using Base::Base;

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

/// Handles colliding symbols when merging circuits.
///
/// This function resolves symbol collisions between operations in different
/// circuits during the linking process. It handles three specific cases:
///
/// 1. Identical extmodules: When two extmodules have identical attributes, the
///    incoming one is removed as they are duplicates.
///
/// 2. Extmodule declaration + module definition: When an extmodule
///    (declaration) collides with a module (definition), the declaration is
///    removed in favor of the definition if their attributes match.
///
/// 3. Extmodule with empty parameters (Zaozi workaround): When two extmodules
///    collide and one has empty parameters, the one without parameters
///    (placeholder declaration) is removed. This handles a limitation in
///    Zaozi's module generation where placeholder extmodule declarations are
///    created from instance ops without knowing the actual parameters or
///    defname.
///
/// \param collidingOp The operation already present in the merged circuit
/// \param incomingOp The operation being added from another circuit
/// \return FailureOr<bool> Returns success with true if incomingOp was erased,
///         success with false if collidingOp was erased, or failure if the
///         collision cannot be resolved
///
/// \note This workaround for empty parameters should ultimately be
///       removed once ODS is updated to properly support placeholder
///       declarations.
static FailureOr<bool> handleCollidingOps(SymbolOpInterface collidingOp,
                                          SymbolOpInterface incomingOp) {
  if (!collidingOp.isPublic())
    return collidingOp->emitOpError("should be a public symbol");
  if (!incomingOp.isPublic())
    return incomingOp->emitOpError("should be a public symbol");

  if ((isa<FExtModuleOp>(collidingOp) && isa<FModuleOp>(incomingOp)) ||
      (isa<FExtModuleOp>(incomingOp) && isa<FModuleOp>(collidingOp))) {
    auto definition = collidingOp;
    auto declaration = incomingOp;
    if (!isa<FModuleOp>(collidingOp)) {
      definition = incomingOp;
      declaration = collidingOp;
    }

    constexpr const StringRef attrsToCompare[] = {
        "portDirections", "portSymbols", "portNames", "portTypes", "layers"};
    if (!all_of(attrsToCompare, [&](StringRef attr) {
          return definition->getAttr(attr) == declaration->getAttr(attr);
        }))
      return failure();

    declaration->erase();
    return declaration == incomingOp;
  }

  if (isa<FExtModuleOp>(collidingOp) && isa<FExtModuleOp>(incomingOp)) {
    constexpr const StringRef attrsToCompare[] = {
        "portDirections", "portSymbols", "portNames",
        "portTypes",      "knownLayers", "layers",
    };
    if (!all_of(attrsToCompare, [&](StringRef attr) {
          return collidingOp->getAttr(attr) == incomingOp->getAttr(attr);
        }))
      return failure();

    auto collidingParams = collidingOp->getAttrOfType<ArrayAttr>("parameters");
    auto incomingParams = incomingOp->getAttrOfType<ArrayAttr>("parameters");
    if (collidingParams == incomingParams) {
      if (collidingOp->getAttr("defname") != incomingOp->getAttr("defname"))
        return failure();
      incomingOp->erase();
      return true;
    }

    // FIXME: definition and declaration may have different defname and
    // decalration has no parameters
    if (collidingParams.empty() || incomingParams.empty()) {
      auto declaration = collidingParams.empty() ? collidingOp : incomingOp;
      declaration->erase();
      return declaration == incomingOp;
    }
  }

  return failure();
}

LogicalResult LinkCircuitsPass::mergeCircuits() {
  auto module = getOperation();

  SmallVector<CircuitOp> circuits;
  for (CircuitOp circuitOp : module.getOps<CircuitOp>())
    circuits.push_back(circuitOp);

  auto builder = OpBuilder(module);
  builder.setInsertionPointToEnd(module.getBody());
  auto mergedCircuit =
      CircuitOp::create(builder, module.getLoc(),
                        StringAttr::get(&getContext(), baseCircuitName));
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
          auto opErased = handleCollidingOps(collidingOp, symbolOp);
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
