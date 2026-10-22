//===- PropagateClockDomains.cpp - Propagate Seq clock domains --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_PROPAGATECLOCKDOMAINS
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;
using namespace mlir;

namespace {

static constexpr StringLiteral clockDomainsAttrName = "seq.clock_domains";

/// A conservative set of clock-domain identifiers.
class DomainSet {
public:
  bool join(const DomainSet &other) {
    bool changed = false;
    for (StringAttr domain : other.domains)
      changed |= domains.insert(domain);
    return changed;
  }

  bool insert(StringAttr domain) { return domains.insert(domain); }
  bool empty() const { return domains.empty(); }

  ArrayAttr asAttr(MLIRContext *context) const {
    SmallVector<Attribute> attributes;
    attributes.append(domains.begin(), domains.end());
    return ArrayAttr::get(context, attributes);
  }

private:
  SmallSetVector<StringAttr, 2> domains;
};

/// Forward may analysis over hardware *instances*. A module body is evaluated
/// once for each instance context, keeping separately-instantiated copies of a
/// module from contaminating each other's propagated domains.
class DomainPropagator {
public:
  explicit DomainPropagator(ModuleOp top)
      : top(top), context(top.getContext()),
        unknown(StringAttr::get(context, "<unknown>")) {}

  void run() {
    for (hw::HWModuleOp module : top.getOps<hw::HWModuleOp>()) {
      if (!module.isPublic())
        continue;
      SmallVector<DomainSet> inputDomains;
      for (auto [index, argument] :
           llvm::enumerate(module.getBodyBlock()->getArguments())) {
        auto domains = getPortDomains(module, index);
        if (domains.empty()) {
          if (isa<ClockType>(argument.getType())) {
            auto name = (Twine(module.getModuleName()) + "." +
                         module.getInputName(index))
                            .str();
            domains.insert(StringAttr::get(context, name));
          } else {
            domains.insert(unknown);
          }
        }
        inputDomains.push_back(std::move(domains));
      }
      propagateModule(module, inputDomains, module.getModuleName().str());
    }
    annotate();
  }

private:
  /// State local to one particular instance of a module body.
  struct InstanceContext {
    DenseMap<Value, DomainSet> values;
  };

  DomainSet getPortDomains(hw::HWModuleOp module, size_t portIndex) const {
    auto attr =
        dyn_cast_or_null<DictionaryAttr>(module.getPortAttrs(portIndex));
    if (!attr)
      return {};
    auto array = dyn_cast<ArrayAttr>(attr.get(clockDomainsAttrName));
    if (!array)
      return {};

    DomainSet result;
    for (Attribute domain : array)
      if (auto string = dyn_cast<StringAttr>(domain))
        result.insert(string);
    return result;
  }

  const DomainSet &get(const InstanceContext &context, Value value) const {
    auto it = context.values.find(value);
    return it == context.values.end() ? empty : it->second;
  }

  void record(Value value, const DomainSet &domains) {
    reportedDomains[value].join(domains);
  }

  SmallVector<DomainSet> propagateModule(hw::HWModuleOp module,
                                         ArrayRef<DomainSet> inputDomains,
                                         StringRef path) {
    // Recursive instantiation does not have a finite instance tree. Preserve
    // soundness by returning an unknown summary for this edge; structural
    // verification can diagnose the recursion independently.
    auto *operation = module.getOperation();
    if (!activeModules.insert(operation).second)
      return unknownOutputs(module);

    InstanceContext instance;
    for (auto [argument, domains] :
         llvm::zip(module.getBodyBlock()->getArguments(), inputDomains))
      instance.values[argument].join(domains);

    SmallVector<Operation *> operations;
    for (Operation &op : module.getBodyBlock()->without_terminator())
      operations.push_back(&op);
    if (!computeTopologicalSorting(operations)) {
      activeModules.erase(operation);
      return unknownOutputs(module);
    }

    unsigned nextDerivedClock = 0;
    for (Operation *operation : operations) {
      if (auto child = dyn_cast<hw::InstanceOp>(operation)) {
        propagateInstance(child, instance, path);
        continue;
      }

      for (Value result : operation->getResults()) {
        if (!isa<ClockType>(result.getType()) || isa<hw::WireOp>(operation))
          continue;
        auto name = (Twine(path) + ".clock" + Twine(nextDerivedClock++)).str();
        instance.values[result].insert(StringAttr::get(context, name));
      }

      if (auto clocked = dyn_cast<Clocked>(operation)) {
        for (Value result : operation->getResults()) {
          if (isa<ClockType>(result.getType()) && !isa<hw::WireOp>(operation))
            continue;
          instance.values[result].join(get(instance, clocked.getClk()));
        }
      } else {
        DomainSet operandDomains;
        for (Value operand : operation->getOperands())
          operandDomains.join(get(instance, operand));
        for (Value result : operation->getResults()) {
          if (isa<ClockType>(result.getType()) && !isa<hw::WireOp>(operation))
            continue;
          instance.values[result].join(operandDomains);
        }
      }

      for (Value result : operation->getResults())
        record(result, get(instance, result));
    }

    auto output = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    SmallVector<DomainSet> result;
    for (Value value : output.getOperands())
      result.push_back(get(instance, value));
    activeModules.erase(operation);
    return result;
  }

  void propagateInstance(hw::InstanceOp instance, InstanceContext &parent,
                         StringRef parentPath) {
    auto *target = SymbolTable::lookupNearestSymbolFrom(
        instance, instance.getReferencedModuleNameAttr());
    auto module = dyn_cast_or_null<hw::HWModuleOp>(target);
    if (!module) {
      for (Value result : instance.getResults())
        parent.values[result].insert(unknown);
    } else {
      SmallVector<DomainSet> inputDomains;
      for (Value input : instance.getInputs())
        inputDomains.push_back(get(parent, input));
      auto path = (Twine(parentPath) + "." + instance.getInstanceName()).str();
      auto outputDomains = propagateModule(module, inputDomains, path);
      for (auto [result, domains] :
           llvm::zip(instance.getResults(), outputDomains))
        parent.values[result].join(domains);
    }

    for (Value result : instance.getResults())
      record(result, get(parent, result));
  }

  SmallVector<DomainSet> unknownOutputs(hw::HWModuleOp module) const {
    auto output = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    SmallVector<DomainSet> result(output.getOperands().size());
    for (DomainSet &domains : result)
      domains.insert(unknown);
    return result;
  }

  void annotate() {
    auto name = StringAttr::get(context, clockDomainsAttrName);
    for (auto &entry : reportedDomains) {
      Value value = entry.first;
      auto *operation = value.getDefiningOp();
      if (!operation)
        continue;

      SmallVector<Attribute> resultDomains;
      bool hasDomains = false;
      for (Value result : operation->getResults()) {
        auto resultIt = reportedDomains.find(result);
        const DomainSet &resultDomainSet =
            resultIt == reportedDomains.end() ? empty : resultIt->second;
        resultDomains.push_back(resultDomainSet.asAttr(context));
        hasDomains |= !resultDomainSet.empty();
      }
      if (hasDomains)
        operation->setAttr(name, ArrayAttr::get(context, resultDomains));
    }
  }

  ModuleOp top;
  MLIRContext *context;
  DenseMap<Value, DomainSet> reportedDomains;
  SmallPtrSet<Operation *, 4> activeModules;
  DomainSet empty;
  StringAttr unknown;
};

struct PropagateClockDomainsPass
    : public seq::impl::PropagateClockDomainsBase<PropagateClockDomainsPass> {
  using PropagateClockDomainsBase::PropagateClockDomainsBase;

  void runOnOperation() override { DomainPropagator(getOperation()).run(); }
};

} // namespace
