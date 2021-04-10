//===- IMConstProp.cpp - Intermodule ConstProp and DCE ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single lattice value. A lattive value corresponds to
/// the various different states that a value in the SCCP dataflow analysis can
/// take. See 'Kind' below for more details on the different states a value can
/// take.
class LatticeValue {
  enum Kind {
    /// A value with a yet to be determined value. This state may be changed to
    /// anything.
    Unknown,

    /// A value that is known to be a constant. This state may be changed to
    /// overdefined.
    Constant,

    /// A value that cannot statically be determined to be a constant. This
    /// state cannot be changed.
    Overdefined
  };

public:
  /// Initialize a lattice value with "Unknown".
  LatticeValue() : constantAndTag(nullptr, Kind::Unknown) {}
  /// Initialize a lattice value with a constant.
  LatticeValue(Attribute attr) : constantAndTag(attr, Kind::Constant) {}

  static LatticeValue getOverdefined() {
    LatticeValue result;
    result.markOverdefined();
    return result;
  }

  bool isUnknown() const { return constantAndTag.getInt() == Kind::Unknown; }
  bool isConstant() const { return constantAndTag.getInt() == Kind::Constant; }
  bool isOverdefined() const {
    return constantAndTag.getInt() == Kind::Overdefined;
  }

  /// Mark the lattice value as overdefined.
  void markOverdefined() {
    constantAndTag.setPointerAndInt(nullptr, Kind::Overdefined);
  }

  /// Mark the lattice value as constant.
  void markConstant(Attribute value) {
    constantAndTag.setPointerAndInt(value, Kind::Constant);
  }

  /// If this lattice is constant, return the constant. Returns nullptr
  /// otherwise.
  Attribute getConstant() const { return constantAndTag.getPointer(); }

  /// Merge in the value of the 'rhs' lattice into this one. Returns true if the
  /// lattice value changed.
  bool meet(LatticeValue rhs) {
    // If we are already overdefined, or rhs is unknown, there is nothing to do.
    if (isOverdefined() || rhs.isUnknown())
      return false;
    // If we are unknown, just take the value of rhs.
    if (isUnknown()) {
      constantAndTag = rhs.constantAndTag;
      return true;
    }

    // Otherwise, if this value doesn't match rhs go straight to overdefined.
    if (constantAndTag != rhs.constantAndTag) {
      markOverdefined();
      return true;
    }
    return false;
  }

private:
  /// The attribute value if this is a constant and the tag for the element
  /// kind.
  llvm::PointerIntPair<Attribute, 2, Kind> constantAndTag;
};
} // end anonymous namespace

namespace {
struct IMConstPropPass : public IMConstPropBase<IMConstPropPass> {
  void runOnOperation() override;
  void rewriteModuleBody(FModuleOp module);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  bool isOverdefined(Value value) const {
    auto it = latticeValues.find(value);
    return it != latticeValues.end() && it->second.isOverdefined();
  }

  /// Mark the given value as overdefined. This means that we cannot refine a
  /// specific constant for this value.
  void markOverdefined(Value value) {
    auto &entry = latticeValues[value];
    if (!entry.isOverdefined()) {
      entry.markOverdefined();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  /// Merge information from the 'from' lattice value into value.  If it
  /// changes, then users of the value are added to the worklist for
  /// revisitation.
  void mergeLatticeValue(Value value, LatticeValue from) {
    mergeLatticeValue(value, latticeValues[value], from);
  }
  void mergeLatticeValue(Value value, LatticeValue &valueEntry,
                         LatticeValue from) {
    if (valueEntry.meet(from))
      changedLatticeValueWorklist.push_back(value);
  }

  /// Mark the given block as executable.
  void markBlockExecutable(Block *block);
  void markWire(WireOp wire);
  void markConstant(ConstantOp constant);

  void visitConnect(ConnectOp connect);
  void visitPartialConnect(PartialConnectOp connect);
  void visitOperation(Operation *op);

private:
  /// This keeps track of the current state of each tracked value.
  DenseMap<Value, LatticeValue> latticeValues;

  /// The set of blocks that are known to execute, or are intrinsically live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// A worklist containing blocks that need to be processed.
  SmallVector<Block *, 64> blockWorklist;

  /// A worklist of values whose LatticeValue recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<Value, 64> changedLatticeValueWorklist;
};
} // end anonymous namespace

// TODO: handle annotations: [[OptimizableExtModuleAnnotation]],
//  [[DontTouchAnnotation]]
void IMConstPropPass::runOnOperation() {
  auto circuit = getOperation();

  // If the top level module is an external module, mark the input ports
  // overdefined.
  if (auto module = dyn_cast<FModuleOp>(circuit.getMainModule())) {
    markBlockExecutable(module.getBodyBlock());
    for (auto port : module.getBodyBlock()->getArguments())
      markOverdefined(port);
  } else {
    // Otherwise, mark all module ports as being overdefined.
    for (auto &circuitBodyOp : circuit.getBody()->getOperations()) {
      if (auto module = dyn_cast<FModuleOp>(circuitBodyOp)) {
        markBlockExecutable(module.getBodyBlock());
        for (auto port : module.getBodyBlock()->getArguments())
          markOverdefined(port);
      }
    }
  }

  // Drive the worklist to convergence.
  while (!changedLatticeValueWorklist.empty()) {
    // If a value changed lattice state then reprocess any of its users.
    while (!changedLatticeValueWorklist.empty()) {
      Value changedVal = changedLatticeValueWorklist.pop_back_val();
      for (Operation *user : changedVal.getUsers()) {
        if (isBlockExecutable(user->getBlock()))
          visitOperation(user);
      }
    }
  }

  // Rewrite any constants in the modules.
  // TODO: parallelize.
  for (auto &circuitBodyOp : *circuit.getBody())
    if (auto module = dyn_cast<FModuleOp>(circuitBodyOp))
      rewriteModuleBody(module);

  // Clean up our state for next time.
  latticeValues.clear();
  executableBlocks.clear();
}

/// Mark a block executable if it isn't already.  This does an initial scan of
/// the block, processing nullary operations like wires, instances, and
/// constants that only get processed once.
void IMConstPropPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  for (auto &op : *block) {
    // We only handle nullary firrtl nodes in the prepass.  Other nodes will get
    // handled as part of top-down worklist processing.
    if (op.getNumOperands() != 0)
      continue;

    // Handle each of the nullary operations in the firrtl dialect.
    if (auto wire = dyn_cast<WireOp>(op))
      markWire(wire);
    else if (auto constant = dyn_cast<ConstantOp>(op))
      markConstant(constant);
    else {
      // TODO: Mems, instances, etc.
      for (auto result : op.getResults())
        markOverdefined(result);
    }
  }
}

void IMConstPropPass::markWire(WireOp wire) {
  // If the wire has a non-ground type, then it is too complex for us to handle,
  // mark the wire as overdefined.
  // TODO: Eventually add a field-sensitive model.
  if (!wire.getType().getPassiveType().isGround())
    return markOverdefined(wire);

  // Otherwise, we leave this value undefined and allow connects to change its
  // state.
}

void IMConstPropPass::markConstant(ConstantOp constant) {
  mergeLatticeValue(constant, LatticeValue(constant.valueAttr()));
}

void IMConstPropPass::rewriteModuleBody(FModuleOp module) {
  // TODO: If a module is unreachable, then nuke its body.
  if (!executableBlocks.count(module.getBodyBlock()))
    return;

  OpBuilder builder(module);

  // TODO: Walk 'when's.
  for (auto &op : llvm::make_early_inc_range(*module.getBodyBlock())) {
    // Connects to values that we found to be constant can be dropped.  These
    // will already have been replaced since we're walking top-down.
    if (auto connect = dyn_cast<ConnectOp>(op)) {
      if (connect.dest().getDefiningOp<ConstantOp>()) {
        connect.erase();
        continue;
      }
    }

    // If the op had any constants folded, replace them.
    if (op.getNumResults() != 0 && !isa<ConstantOp>(op)) {
      for (auto result : op.getResults()) {
        auto it = latticeValues.find(result);
        if (it != latticeValues.end() && it->second.isConstant()) {
          builder.setInsertionPoint(&op);
          auto cstAttr = it->second.getConstant();
          auto *cst = op.getDialect()->materializeConstant(
              builder, cstAttr, result.getType(), op.getLoc());
          if (cst)
            result.replaceAllUsesWith(cst->getResult(0));
        }
      }
      if (op.use_empty() && (wouldOpBeTriviallyDead(&op) || isa<WireOp>(op))) {
        op.erase();
        continue;
      }
    }
  }
}

void IMConstPropPass::visitConnect(ConnectOp connect) {
  // We merge the value from the RHS into the value of the LHS.
  mergeLatticeValue(connect.dest(), latticeValues[connect.src()]);
}

void IMConstPropPass::visitPartialConnect(PartialConnectOp partialConnect) {
  // We don't handle partial connects yet, just be super conservative.
  markOverdefined(partialConnect.dest());
  markOverdefined(partialConnect.src());
}

/// This method is invoked when an operand of the specified op changes its
/// lattice value state and when the block containing the operation is first
/// noticed as being alive.
///
/// This should update the lattice value state for any result values.
///
void IMConstPropPass::visitOperation(Operation *op) {
  // If this is a operation with special handling, handle it specially.
  if (auto connectOp = dyn_cast<ConnectOp>(op))
    return visitConnect(connectOp);
  if (auto partialConnectOp = dyn_cast<PartialConnectOp>(op))
    return visitPartialConnect(partialConnectOp);
  // TODO: Handle instances and when operations.

  // If this op produces no results, it can't produce any constants.
  if (op->getNumResults() == 0)
    return;

  // Collect all of the constant operands feeding into this operation. If any
  // are not ready to be resolved, bail out and wait for them to resolve.
  SmallVector<Attribute, 8> operandConstants;
  operandConstants.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    // Make sure all of the operands are resolved first.
    auto &operandLattice = latticeValues[operand];
    if (operandLattice.isUnknown())
      return;
    operandConstants.push_back(operandLattice.getConstant());
  }

  // If all of the results of this operation are already overdefined, bail out
  // early.
  auto isOverdefinedFn = [&](Value value) { return isOverdefined(value); };
  if (llvm::all_of(op->getResults(), isOverdefinedFn))
    return;

  // Save the original operands and attributes just in case the operation folds
  // in-place. The constant passed in may not correspond to the real runtime
  // value, so in-place updates are not allowed.
  SmallVector<Value, 8> originalOperands(op->getOperands());
  DictionaryAttr originalAttrs = op->getAttrDictionary();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(operandConstants, foldResults))) {
    for (auto value : op->getResults())
      markOverdefined(value);
    return;
  }

  // If the folding was in-place, mark the results as overdefined and reset the
  // operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    for (auto value : op->getResults())
      markOverdefined(value);
    return;
  }

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
    // Merge in the result of the fold, either a constant or a value.
    LatticeValue resultLattice;
    OpFoldResult foldResult = foldResults[i];
    if (Attribute foldAttr = foldResult.dyn_cast<Attribute>())
      resultLattice = LatticeValue(foldAttr);
    else // Folding to an operand results in its value.
      resultLattice = latticeValues[foldResult.get<Value>()];
    mergeLatticeValue(op->getResult(i), resultLattice);
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMConstPropPass() {
  return std::make_unique<IMConstPropPass>();
}
