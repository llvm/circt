//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs structural hashing for Synth dialect operations
// (AIG/MIG). Unlike MLIR's general CSE pass, this is domain-specific to
// AIG/MIG operations, allowing it to reorder operands based on their
// structural properties and take inversion flags into account for
// canonicalization.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Naming.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "synth-structural-hash"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_STRUCTURALHASH
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

/// A struct that represents the key used for structural hashing. It contains
/// the operation name and a sorted vector of pointer-integer pairs, which
/// represent the inputs to the operation and their inversion status.
/// This key is used to identify structurally equivalent operations for CSE.
struct StructuralHashKey {
  OperationName opName;
  llvm::SmallVector<llvm::PointerIntPair<Value, 1>, 3> operandPairs;

  /// Constructor.
  StructuralHashKey(OperationName name,
                    llvm::SmallVector<llvm::PointerIntPair<Value, 1>, 3> inps)
      : opName(name), operandPairs(std::move(inps)) {}
};

// DenseMapInfo specialization for StructuralHashKey
template <>
struct llvm::DenseMapInfo<StructuralHashKey> {
  static StructuralHashKey getEmptyKey() {
    return StructuralHashKey(llvm::DenseMapInfo<OperationName>::getEmptyKey(),
                             {});
  }

  static StructuralHashKey getTombstoneKey() {
    return StructuralHashKey(
        llvm::DenseMapInfo<OperationName>::getTombstoneKey(), {});
  }

  static unsigned getHashValue(const StructuralHashKey &key) {
    auto hash = hash_value(key.opName);
    for (const auto &operand : key.operandPairs)
      hash = llvm::hash_combine(hash, operand.getOpaqueValue());
    return static_cast<unsigned>(hash);
  }

  static bool isEqual(const StructuralHashKey &lhs,
                      const StructuralHashKey &rhs) {
    return llvm::DenseMapInfo<OperationName>::isEqual(lhs.opName, rhs.opName) &&
           lhs.operandPairs == rhs.operandPairs;
  }
};

namespace {
/// Pass definition.
struct StructuralHashPass
    : public impl::StructuralHashBase<StructuralHashPass> {
  void runOnOperation() override;
};
} // namespace

namespace {
/// The main driver class that implements the structural hashing algorithm.
/// This class manages the state for value numbering, inversion tracking,
/// and the hash table for CSE. It processes operations in topological order
/// and performs operand reordering and inversion propagation for
/// canonicalization.
class StructuralHashDriver {
public:
  StructuralHashDriver() = default;
  void visitOp(Operation *op, ArrayRef<bool> inverted);
  void visitUnaryOp(Operation *op, bool inverted);
  void visitVariadicOp(Operation *op, ArrayRef<bool> inverted);
  uint64_t getNumber(Value v);

  /// Runs the structural hashing pass on the given module.
  /// Performs topological sorting, assigns value numbers to arguments,
  /// processes target operations, and cleans up unused operations.
  llvm::LogicalResult run(hw::HWModuleOp op);

private:
  /// Maps values to unique numbers for deterministic operand sorting.
  DenseMap<Value, uint64_t> valueNumber;
  uint64_t constantCounter = 0;

  /// Hash table mapping structural keys to canonical operations for CSE.
  DenseMap<StructuralHashKey, Operation *> hashTable;

  /// Maps inverted values to their non-inverted equivalents for propagation.
  /// For example, if we have:
  /// ```
  /// %b = synth.aig.and_inv not %a
  /// %c = synth.aig.and_inv not %b
  /// ```
  /// Then `inversion[%b] = %a`, and when visiting `%c`, we can query
  /// `inversion[%b]` to directly obtain `%a`.
  DenseMap<Value, Value> inversion;
};
} // namespace

void StructuralHashDriver::visitOp(Operation *op, ArrayRef<bool> inverted) {
  /// Dispatches to the appropriate visitor based on the number of operands.
  /// For unary operations, calls visitUnaryOp; for variadic operations,
  /// calls visitVariadicOp.
  if (op->getNumOperands() == 1) {
    visitUnaryOp(op, inverted[0]);
    return;
  }
  visitVariadicOp(op, inverted);
}

/// Handles unary operations (single operand).
/// If not inverted, replaces the operation with its operand.
/// If inverted, attempts to propagate inversion through the inversion map
/// or records the inversion for later propagation.
void StructuralHashDriver::visitUnaryOp(Operation *op, bool inverted) {
  if (!inverted) {
    op->replaceAllUsesWith(ArrayRef<Value>{op->getOperand(0)});
    op->erase();
    return;
  }
  // Check if we can propagate inversion through the inversion map.
  auto operand = op->getOperand(0);
  auto it = inversion.find(operand);
  if (it != inversion.end()) {
    // Found, replace the operand with the mapped value
    op->replaceAllUsesWith(ArrayRef<Value>{it->second});
    op->erase();
  } else {
    // Not found, insert into the map
    inversion[op->getResult(0)] = operand;
  }
}

/// Computes a structural hash key, sorts operands for canonicalization,
/// and performs CSE by checking the hash table for equivalent operations.
void StructuralHashDriver::visitVariadicOp(Operation *op,
                                           ArrayRef<bool> inverted) {

  // Compute the structural hash key for the operation.
  StructuralHashKey key(op->getName(), {});
  for (auto [input, inverted] : llvm::zip(op->getOperands(), inverted)) {
    bool isInverted = inverted;
    // Check if we can propagate inversion through the inversion map
    auto it = inversion.find(input);
    if (it != inversion.end()) {
      // Found, use the mapped value and flip the inversion status
      input = it->second;
      isInverted = !isInverted;
    }

    key.operandPairs.push_back(
        llvm::PointerIntPair<Value, 1>(input, isInverted));
    // Ensure the operand has a number assigned, otherwise sorting might be
    // non-deterministic.
    (void)getNumber(input);
  }

  // Sort operands based on their assigned numbers.
  llvm::sort(key.operandPairs, [&](auto a, auto b) {
    size_t aNum = getNumber(a.getPointer());
    size_t bNum = getNumber(b.getPointer());
    if (aNum != bNum)
      return aNum < bNum;
    return a.getInt() < b.getInt();
  });

  // Insert the key into the hash table.
  auto [it, inserted] = hashTable.try_emplace(key, op);
  if (inserted) {
    // New entry, keep the operation and sort its operands.
    op->setOperands(llvm::to_vector<3>(llvm::map_range(
        key.operandPairs, [](auto p) { return p.getPointer(); })));
    SmallVector<bool, 3> newInversion(
        llvm::map_range(key.operandPairs, [](auto p) { return p.getInt(); }));
    op->setAttr("inverted",
                mlir::DenseBoolArrayAttr::get(op->getContext(), newInversion));
    // Assign a number to the result for future sorting.
    (void)getNumber(op->getResult(0));
  } else {
    LDBG() << "Structural Hash: Replacing " << *op << " with " << *(it->second)
           << "\n";
    // Existing entry, replace all uses and erase the operation.
    // Propagate namehints.
    auto name = circt::chooseName(op, it->second);
    if (name && !it->second->hasAttr("sv.namehint"))
      it->second->setAttr("sv.namehint", name);
    op->replaceAllUsesWith(it->second);
    op->erase();
  }
}

/// Assigns or retrieves a unique number for a value. Used for deterministic
/// operand sorting.
uint64_t StructuralHashDriver::getNumber(Value v) {
  auto it = valueNumber.find(v);
  if (it != valueNumber.end())
    return it->second;

  // Assign a new number. Constants get high numbers to make constants are
  // pushed to the back.
  if (auto *op = v.getDefiningOp();
      op && op->hasTrait<mlir::OpTrait::ConstantLike>()) {
    auto [it, inserted] = valueNumber.try_emplace(
        v, std::numeric_limits<uint64_t>::max() - constantCounter++);
    return it->second;
  }

  return valueNumber.try_emplace(v, valueNumber.size() - constantCounter)
      .first->second;
}

llvm::LogicalResult StructuralHashDriver::run(hw::HWModuleOp moduleOp) {
  auto isOperationReady = [&](Value value, Operation *op) -> bool {
    // Otherthan target ops, all other ops are always ready.
    return !isa<circt::synth::aig::AndInverterOp,
                circt::synth::mig::MajorityInverterOp>(op);
  };

  if (!mlir::sortTopologically(moduleOp.getBodyBlock(), isOperationReady))
    return failure();

  for (auto arg : moduleOp.getBodyBlock()->getArguments())
    (void)getNumber(arg);

  // Process target ops.
  // NOTE: Don't use walk here since the pass currently doesn't handle nested
  // regions.
  for (auto &op :
       llvm::make_early_inc_range(moduleOp.getBodyBlock()->getOperations())) {
    mlir::TypeSwitch<Operation *>(&op)
        .Case<circt::synth::aig::AndInverterOp,
              circt::synth::mig::MajorityInverterOp>([&](auto invertibleOp) {
          visitOp(invertibleOp, invertibleOp.getInverted());
        })
        .Default([&](Operation *op) {});
  }

  // Run DCE to remove dangling ops.
  mlir::PatternRewriter rewriter(moduleOp.getContext());
  (void)mlir::runRegionDCE(rewriter, moduleOp->getRegions());
  return mlir::success();
}

void StructuralHashPass::runOnOperation() {
  auto topOp = getOperation();
  StructuralHashDriver driver;
  if (failed(driver.run(topOp)))
    return signalPassFailure();
}
