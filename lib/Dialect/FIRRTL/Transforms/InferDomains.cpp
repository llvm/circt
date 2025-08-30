//===- InferDomains.cpp - Infer and Check FIRRTL Domains ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements FIRRTL domain inference and checking with canonical
// domain representation. Domain sequences are canonicalized by sorting and
// removing duplicates, making domain order irrelevant and allowing duplicate
// domains to be treated as equivalent. The result of this pass is either a
// correctly domain-inferred circuit or pass failure if the circuit contains
// illegal domain crossings.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-infer-domains"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

/// Union-Find data structure for domain variables using LLVM's
/// EquivalenceClasses. Handles sequences of domains with canonical
/// representation where domain order is irrelevant and duplicates are removed.
/// This allows 'domains %A, %B' to be equivalent to 'domains %B, %A' and
/// 'domains %A, %A' to be equivalent to 'domains %A'.
class DomainUnionFind {
public:
  using DomainSequence = llvm::SmallVector<Value, 4>;

private:
  /// Canonicalize a domain sequence by sorting and removing duplicates.
  /// This ensures that domain order doesn't matter and duplicate domains
  /// are treated as equivalent. For example:
  /// - 'domains %A, %B' and 'domains %B, %A' become the same canonical form
  /// - 'domains %A, %A' becomes 'domains %A'
  static DomainSequence canonicalizeDomains(ArrayRef<Value> domains) {
    DomainSequence canonical(domains.begin(), domains.end());

    // Sort domains deterministically.  Use block arg order if all are block
    // args.  Fallback to opaque pointer comparison otherwise.  Note: the opaque
    // pointer comparison is only stable within a run of MLIR and should not be
    // relied upon for determinism outside of a run.
    llvm::sort(canonical, [](Value a, Value b) {
      auto aArg = dyn_cast<BlockArgument>(a);
      auto bArg = dyn_cast<BlockArgument>(b);
      if (aArg && bArg)
        return aArg.getArgNumber() < bArg.getArgNumber();
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    });

    // Remove duplicates to handle cases like 'domains %A, %A'.
    canonical.erase(std::unique(canonical.begin(), canonical.end()),
                    canonical.end());

    return canonical;
  }

public:
  /// Get or create a domain variable for a value.
  Value getDomainVar(Value value) {
    auto it = valueToVar.find(value);
    if (it != valueToVar.end())
      return it->second;

    // Create a new representative value for this domain variable.
    Value representative = value;
    valueToVar[value] = representative;
    equivalenceClasses.insert(representative);
    return representative;
  }

  /// Union two domain variables.  Returns success if successful, failure if
  /// there's a domain conflict.  With canonical representation, conflicts only
  /// occur when connecting values with truly different domain sets (not just
  /// different ordering).
  [[nodiscard]] LogicalResult unifyDomains(Value var1, Value var2) {
    Value rep1 = equivalenceClasses.getLeaderValue(var1);
    Value rep2 = equivalenceClasses.getLeaderValue(var2);

    if (rep1 == rep2)
      return success();

    // Check for domain conflicts using canonical representation.
    const DomainSequence &domains1 = getDomains(rep1);
    const DomainSequence &domains2 = getDomains(rep2);

    // Since domains are already canonical, we can compare them directly.
    // Conflicts only occur when connecting truly different domain sets.  For
    // example: 'domains %A' vs 'domains %B' is a conflict, but 'domains %A, %B'
    // vs 'domains %B, %A' is not (same canonical form).
    if (!domains1.empty() && !domains2.empty() && domains1 != domains2) {
      // Conflict: both have different concrete domain sequences
      return failure();
    }

    // Merge the concrete domain sequences (both are already canonical)
    const DomainSequence &mergedDomains =
        !domains1.empty() ? domains1 : domains2;

    // Union the equivalence classes
    equivalenceClasses.unionSets(rep1, rep2);

    // Set the merged domain sequence on the new representative
    if (!mergedDomains.empty()) {
      Value newRep = equivalenceClasses.getLeaderValue(rep1);
      concreteDomains[newRep] = mergedDomains;
    }

    return success();
  }

  /// Set the concrete domain sequence for a variable.
  /// The domain sequence is automatically canonicalized to ensure consistent
  /// representation regardless of input order or duplicates.
  void setDomains(Value var, ArrayRef<Value> domains) {
    Value rep = equivalenceClasses.getLeaderValue(var);
    concreteDomains[rep] = canonicalizeDomains(domains);
  }

  /// Get the concrete domain sequence for a variable.
  /// Returns the canonical domain sequence (sorted, no duplicates).
  const DomainSequence &getDomains(Value var) {
    Value rep = equivalenceClasses.getLeaderValue(var);
    auto it = concreteDomains.find(rep);
    if (it != concreteDomains.end())
      return it->second;

    // Return empty sequence if no domains are set
    static const DomainSequence emptySequence;
    return emptySequence;
  }

  /// Clear all state.
  void clear() {
    equivalenceClasses = llvm::EquivalenceClasses<Value>();
    concreteDomains.clear();
    valueToVar.clear();
  }

private:
  llvm::EquivalenceClasses<Value> equivalenceClasses;
  llvm::DenseMap<Value, DomainSequence> concreteDomains;
  llvm::DenseMap<Value, Value> valueToVar;
};

/// Domain inference and checking pass implementation.
/// Uses canonical domain representation to allow domain order independence
/// and duplicate domain handling.
class InferDomainsPass
    : public circt::firrtl::impl::InferDomainsBase<InferDomainsPass> {

public:
  InferDomainsPass() = default;

  void runOnOperation() override;

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<InferDomainsPass>();
  }

private:
  /// Union-find structure for domain variables.
  DomainUnionFind domainUF;

  /// Track if any domain crossing errors were found.
  bool hasErrors = false;

  /// Process a module and infer domains.
  LogicalResult processModule(FModuleOp module);

  /// Process domain constraints from operations.
  void processOperation(Operation *op);

  /// Unify domains of two values.
  [[nodiscard]] LogicalResult unifyDomains(Value lhs, Value rhs);

  /// Set explicit domain sequence for a value.
  void setExplicitDomains(Value value, OperandRange domains);

  /// Helper to add domain sequence notes to diagnostics.
  void addDomainSequenceNotes(InFlightDiagnostic &diag,
                              const DomainUnionFind::DomainSequence &domains,
                              StringRef prefix);

  /// Helper to emit domain crossing error with detailed notes.
  /// Sets hasErrors flag.
  void emitDomainCrossingError(Operation *op, Value lhs, Value rhs,
                               StringRef errorMessage, StringRef lhsLabel,
                               StringRef rhsLabel);
};

} // namespace

LogicalResult InferDomainsPass::unifyDomains(Value lhs, Value rhs) {
  Value lhsVar = domainUF.getDomainVar(lhs);
  Value rhsVar = domainUF.getDomainVar(rhs);

  return domainUF.unifyDomains(lhsVar, rhsVar);
}

void InferDomainsPass::setExplicitDomains(Value value, OperandRange domains) {
  Value domainVar = domainUF.getDomainVar(value);
  // Convert OperandRange to SmallVector for setDomains.
  llvm::SmallVector<Value, 4> domainVec(domains.begin(), domains.end());
  domainUF.setDomains(domainVar, domainVec);
}

void InferDomainsPass::addDomainSequenceNotes(
    InFlightDiagnostic &diag, const DomainUnionFind::DomainSequence &domains,
    StringRef prefix) {
  if (domains.empty())
    return;

  for (size_t i = 0; i < domains.size(); ++i) {
    if (domains[i]) {
      std::string message = prefix.str();
      if (domains.size() > 1) {
        message += " (domain " + std::to_string(i + 1) + " of " +
                   std::to_string(domains.size()) + ")";
      }
      message += " is in domain defined here";
      diag.attachNote(domains[i].getLoc()).append(message);
    }
  }
}

void InferDomainsPass::emitDomainCrossingError(Operation *op, Value lhs,
                                               Value rhs,
                                               StringRef errorMessage,
                                               StringRef lhsLabel,
                                               StringRef rhsLabel) {
  // Get the concrete domain sequences for error reporting
  Value lhsVar = domainUF.getDomainVar(lhs);
  Value rhsVar = domainUF.getDomainVar(rhs);
  const auto &lhsDomains = domainUF.getDomains(lhsVar);
  const auto &rhsDomains = domainUF.getDomains(rhsVar);

  auto diag = op->emitError(errorMessage);
  addDomainSequenceNotes(diag, lhsDomains, lhsLabel);
  addDomainSequenceNotes(diag, rhsDomains, rhsLabel);
  hasErrors = true;
}

void InferDomainsPass::processOperation(Operation *op) {
  // Error on unhandled operations.
  if (isa<InstanceOp>(op)) {
    llvm::errs() << "InferDomains cannot yet handle " << op->getName() << "\n";
    return signalPassFailure();
  }

  // Handle unsafe domain casts
  if (auto domainCastOp = dyn_cast<UnsafeDomainCastOp>(op)) {
    if (!domainCastOp.getDomains().empty()) {
      // Explicitly cast to specified domain sequence
      setExplicitDomains(domainCastOp.getResult(), domainCastOp.getDomains());
    }
    return;
  }

  // For all operations (including connections), propagate domains from operands
  // to results This is a conservative approach - all operands and results share
  // domains
  if (!op->getOperands().empty()) {
    Value firstOperand = op->getOperand(0);
    for (auto operand : op->getOperands()) {
      if (failed(unifyDomains(firstOperand, operand))) {
        emitDomainCrossingError(op, firstOperand, operand,
                                "illegal domain crossing in operation",
                                "first operand", "operand");
      }
    }
    for (auto result : op->getResults()) {
      if (failed(unifyDomains(firstOperand, result))) {
        emitDomainCrossingError(op, firstOperand, result,
                                "illegal domain crossing in operation",
                                "operand", "result");
      }
    }
  }
}

LogicalResult InferDomainsPass::processModule(FModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Processing module: " << module.getName() << "\n");

  // Process module ports - domain ports define explicit domains
  for (auto [index, port] : llvm::enumerate(module.getPorts())) {
    Value portValue = module.getArgument(index);

    // This is a domain port.
    if (isa<DomainType>(port.type)) {
      Value domainVar = domainUF.getDomainVar(portValue);
      domainUF.setDomains(domainVar, {portValue});
      continue;
    }

    // This is a port with explicit domain information.
    auto domains = cast<ArrayAttr>(port.domains);
    if (domains.empty())
      continue;
    SmallVector<Value, 4> domainPorts;
    for (auto domain : domains) {
      auto index = cast<IntegerAttr>(domain).getUInt();
      domainPorts.push_back(module.getArgument(index));
    }
    Value domainVar = domainUF.getDomainVar(portValue);
    domainUF.setDomains(domainVar, domainPorts);
  }

  // Process all operations in the module
  module.walk([&](Operation *op) { processOperation(op); });

  // Check if any errors were found during processing
  if (hasErrors)
    return failure();

  // Update domain information for all non-domain ports
  SmallVector<Attribute> newDomainInfo;
  bool anyUpdated = false;

  for (auto [index, port] : llvm::enumerate(module.getPorts())) {
    // Skip domain ports - they don't need domain information
    if (isa<DomainType>(port.type)) {
      newDomainInfo.push_back(port.domains
                                  ? port.domains
                                  : ArrayAttr::get(module.getContext(), {}));
      continue;
    }

    // Get the inferred domains for this port
    Value portValue = module.getArgument(index);
    const auto &inferredDomains = domainUF.getDomains(portValue);

    // Convert domain values to domain indices
    SmallVector<Attribute> domainIndices;
    for (Value domain : inferredDomains) {
      if (auto blockArg = dyn_cast<BlockArgument>(domain)) {
        // This is a reference to a domain port
        domainIndices.push_back(IntegerAttr::get(
            IntegerType::get(module.getContext(), 32, IntegerType::Unsigned),
            blockArg.getArgNumber()));
      }
    }

    ArrayAttr newDomains = ArrayAttr::get(module.getContext(), domainIndices);

    // Check if this is different from the existing domain information
    if (!port.domains || port.domains != newDomains) {
      anyUpdated = true;
    }

    newDomainInfo.push_back(newDomains);
  }

  // Update the module's domain information if anything changed
  if (anyUpdated) {
    module->setAttr("domainInfo",
                    ArrayAttr::get(module.getContext(), newDomainInfo));
  }

  LLVM_DEBUG({
    for (auto [index, port] : llvm::enumerate(module.getPorts())) {
      llvm::dbgs() << "  - port: " << port.getName() << "\n"
                   << "    domains:\n";
      auto domains = domainUF.getDomains(module.getArgument(index));
      if (domains.empty()) {
        llvm::dbgs() << "      - inferred\n";
        continue;
      }
      for (auto domain : domains) {
        if (auto port = dyn_cast<BlockArgument>(domain)) {
          llvm::dbgs() << "      - " << module.getPortName(port.getArgNumber())
                       << "\n";
          continue;
        }
      }
    }
  });

  return success();
}

void InferDomainsPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");

  // Clear state from any previous runs
  domainUF.clear();
  hasErrors = false;

  auto circuit = getOperation();

  // Process each module in the circuit
  for (auto module : circuit.getOps<FModuleOp>()) {
    if (failed(processModule(module))) {
      signalPassFailure();
      return;
    }
  }

  // Signal failure if any domain crossing errors were found
  if (hasErrors) {
    signalPassFailure();
  }

  LLVM_DEBUG(debugFooter() << "\n");
}
