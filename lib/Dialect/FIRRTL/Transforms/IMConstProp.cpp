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

  /// Visit the users of the given IR that reside within executable blocks.
  template <typename T>
  void visitUsers(T &valueOrOperation) {
    for (Operation *user : valueOrOperation.getUsers())
      if (isBlockExecutable(user->getBlock()))
        visitOperation(user);
  }

  /// Mark the given value as overdefined. This means that we cannot refine a
  /// specific constant for this value.
  void markOverdefined(Value value) { latticeValues[value].markOverdefined(); }

  /// Mark the given block as executable. Returns false if the block was already
  /// marked executable.
  bool markBlockExecutable(Block *block);

  void meet(Operation *owner, LatticeValue &to, LatticeValue from);

  void visitWire(WireOp wire);
  void visitConnect(ConnectOp connect);
  void visitOperation(Operation *op);

private:
  /// This keeps track of the current state of each tracked value.
  DenseMap<Value, LatticeValue> latticeValues;

  /// The set of blocks that are known to execute, or are intrinsically live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// A worklist containing blocks that need to be processed.
  SmallVector<Block *, 64> blockWorklist;

  /// A worklist of operations that need to be processed.
  SmallVector<Operation *, 64> opWorklist;
};
} // end anonymous namespace

void IMConstPropPass::meet(Operation *owner, LatticeValue &to,
                           LatticeValue from) {
  if (to.meet(from))
    opWorklist.push_back(owner);
}

bool IMConstPropPass::markBlockExecutable(Block *block) {
  bool marked = executableBlocks.insert(block).second;
  if (marked)
    blockWorklist.push_back(block);
  return marked;
}

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
  while (!blockWorklist.empty() || !opWorklist.empty()) {
    // Process any operations in the op worklist.
    while (!opWorklist.empty())
      visitUsers(*opWorklist.pop_back_val());

    // Process any blocks in the block worklist.
    while (!blockWorklist.empty()) {
      for (Operation &op : *blockWorklist.pop_back_val())
        visitOperation(&op);
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

void IMConstPropPass::visitWire(WireOp wire) {
  // If the wire has a non-ground type, then it is too complex for us to handle,
  // mark the wire as overdefined.
  if (!wire.getType().getPassiveType().isGround())
    return meet(wire, latticeValues[wire], LatticeValue::getOverdefined());

  // Otherwise, we leave this value undefined and allow connects to change its
  // state.
}

void IMConstPropPass::visitConnect(ConnectOp connect) {
  // We merge the value from the RHS into the value of the LHS.
  auto rhs = latticeValues[connect.src()];
  Value dest = connect.dest();
  if (latticeValues[dest].meet(rhs))
    visitUsers(dest);
}

void IMConstPropPass::visitOperation(Operation *op) {
  // If this is a operation with special handling, handle it specially.
  if (auto wireOp = dyn_cast<WireOp>(op))
    return visitWire(wireOp);
  if (auto connectOp = dyn_cast<ConnectOp>(op))
    return visitConnect(connectOp);

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

  // TODO: Handle instances and when operations.

  // If this op produces no results, it can't produce any constants.
  if (op->getNumResults() == 0)
    return;

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
    LatticeValue &resultLattice = latticeValues[op->getResult(i)];

    // Merge in the result of the fold, either a constant or a value.
    OpFoldResult foldResult = foldResults[i];
    if (Attribute foldAttr = foldResult.dyn_cast<Attribute>())
      meet(op, resultLattice, LatticeValue(foldAttr));
    else
      meet(op, resultLattice, latticeValues[foldResult.get<Value>()]);
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMConstPropPass() {
  return std::make_unique<IMConstPropPass>();
}
