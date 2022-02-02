//===- PrepareForEmission.cpp - IR Prepass for Emitter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the "prepare" pass that walks the IR before the emitter
// gets involved.  This allows us to do some transformations that would be
// awkward to implement inline in the emitter.
//
// NOTE: This file covers the preparation phase of `ExportVerilog` which mainly
// legalizes the IR and makes adjustments necessary for emission. This is the
// place to mutate the IR if emission needs it. The IR cannot be modified during
// emission itself, which happens in parallel.
//
//===----------------------------------------------------------------------===//

#include "ExportVerilogInternals.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace comb;
using namespace hw;
using namespace sv;
using namespace ExportVerilog;

// Check if the value is from read of a wire or reg or is a port.
static bool isSimpleReadOrPort(Value v) {
  if (v.isa<BlockArgument>())
    return true;
  auto vOp = v.getDefiningOp();
  if (!vOp)
    return false;
  auto read = dyn_cast<ReadInOutOp>(vOp);
  if (!read)
    return false;
  auto readSrc = read.input().getDefiningOp();
  if (!readSrc)
    return false;
  return isa<WireOp, RegOp>(readSrc);
}

// Given an invisible instance, make sure all inputs are driven from
// wires or ports.
static void lowerBoundInstance(InstanceOp op) {
  Block *block = op->getParentOfType<HWModuleOp>().getBodyBlock();
  auto builder = ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), block);

  SmallString<32> nameTmp;
  nameTmp = (op.instanceName() + "_").str();
  auto namePrefixSize = nameTmp.size();

  size_t nextOpNo = 0;
  for (auto &port : getModulePortInfo(op).inputs) {
    auto src = op.getOperand(nextOpNo);
    ++nextOpNo;

    if (isSimpleReadOrPort(src))
      continue;

    nameTmp.resize(namePrefixSize);
    if (port.name)
      nameTmp += port.name.getValue().str();
    else
      nameTmp += std::to_string(nextOpNo - 1);

    auto newWire = builder.create<WireOp>(src.getType(), nameTmp);
    auto newWireRead = builder.create<ReadInOutOp>(newWire);
    auto connect = builder.create<AssignOp>(newWire, src);
    newWireRead->moveBefore(op);
    connect->moveBefore(op);
    op.setOperand(nextOpNo - 1, newWireRead);
  }
}

static bool onlyUseIsAssign(Value v) {
  if (!v.hasOneUse())
    return false;
  if (!dyn_cast_or_null<AssignOp>(v.getDefiningOp()))
    return false;
  return true;
}

// Ensure that each output of an instance are used only by a wire
static void lowerInstanceResults(InstanceOp op) {
  Block *block = op->getParentOfType<HWModuleOp>().getBodyBlock();
  auto builder = ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), block);

  SmallString<32> nameTmp;
  nameTmp = (op.instanceName() + "_").str();
  auto namePrefixSize = nameTmp.size();

  size_t nextResultNo = 0;
  for (auto &port : getModulePortInfo(op).outputs) {
    auto result = op.getResult(nextResultNo);
    ++nextResultNo;

    if (onlyUseIsAssign(result))
      continue;

    bool isOneUseOutput = false;
    if (result.hasOneUse()) {
      OpOperand &use = *result.getUses().begin();
      isOneUseOutput = dyn_cast_or_null<OutputOp>(use.getOwner()) != nullptr;
    }

    if (!isOneUseOutput) {
      nameTmp.resize(namePrefixSize);
      if (port.name)
        nameTmp += port.name.getValue().str();
      else
        nameTmp += std::to_string(nextResultNo - 1);

      auto newWire = builder.create<WireOp>(result.getType(), nameTmp);
      while (!result.use_empty()) {
        auto newWireRead = builder.create<ReadInOutOp>(newWire);
        OpOperand &use = *result.getUses().begin();
        use.set(newWireRead);
        newWireRead->moveBefore(use.getOwner());
      }

      auto connect = builder.create<AssignOp>(newWire, result);
      connect->moveAfter(op);
    }
  }
}

// Given a side effect free "always inline" operation, make sure that it
// exists in the same block as its users and that it has one use for each one.
static void lowerAlwaysInlineOperation(Operation *op) {
  assert(op->getNumResults() == 1 &&
         "only support 'always inline' ops with one result");

  // Nuke use-less operations.
  if (op->use_empty()) {
    op->erase();
    return;
  }

  // Moving/cloning an op should pull along its operand tree with it if they
  // are always inline.  This happens when an array index has a constant
  // operand for example.
  auto recursivelyHandleOperands = [](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (auto *operandOp = operand.getDefiningOp())
        if (isExpressionAlwaysInline(operandOp))
          lowerAlwaysInlineOperation(operandOp);
    }
  };

  // If this operation has multiple uses, duplicate it into N-1 of them in
  // turn.
  while (!op->hasOneUse()) {
    OpOperand &use = *op->getUses().begin();
    Operation *user = use.getOwner();

    // Clone the op before the user.
    auto *newOp = op->clone();
    user->getBlock()->getOperations().insert(Block::iterator(user), newOp);
    // Change the user to use the new op.
    use.set(newOp->getResult(0));

    // If any of the operations of the moved op are always inline, recursively
    // handle them too.
    recursivelyHandleOperands(newOp);
  }

  // Finally, ensures the op is in the same block as its user so it can be
  // inlined.
  Operation *user = *op->getUsers().begin();
  if (op->getBlock() != user->getBlock()) {
    op->moveBefore(user);

    // If any of the operations of the moved op are always inline, recursively
    // move/clone them too.
    recursivelyHandleOperands(op);
  }
  return;
}

/// Lower a variadic fully-associative operation into an expression tree.  This
/// enables long-line splitting to work with them.
static Value lowerFullyAssociativeOp(Operation &op, OperandRange operands,
                                     SmallVector<Operation *> &newOps) {
  // save the top level name
  auto name = op.getAttr("sv.namehint");
  Value lhs, rhs;
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    return operands[0];
  case 2:
    lhs = operands[0];
    rhs = operands[1];
    break;
  default:
    auto firstHalf = operands.size() / 2;
    lhs = lowerFullyAssociativeOp(op, operands.take_front(firstHalf), newOps);
    rhs = lowerFullyAssociativeOp(op, operands.drop_front(firstHalf), newOps);
    break;
  }

  OperationState state(op.getLoc(), op.getName());
  state.addOperands(ValueRange{lhs, rhs});
  state.addTypes(op.getResult(0).getType());
  auto *newOp = Operation::create(state);
  op.getBlock()->getOperations().insert(Block::iterator(&op), newOp);
  newOps.push_back(newOp);
  if (name)
    newOp->setAttr("sv.namehint", name);
  return newOp->getResult(0);
}

/// When we find that an operation is used before it is defined in a graph
/// region, we emit an explicit wire to resolve the issue.
static void lowerUsersToTemporaryWire(Operation &op) {
  Block *block = op.getBlock();
  auto builder = ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), block);

  for (auto result : op.getResults()) {
    auto newWire = builder.create<WireOp>(result.getType());

    while (!result.use_empty()) {
      auto newWireRead = builder.create<ReadInOutOp>(newWire);
      OpOperand &use = *result.getUses().begin();
      use.set(newWireRead);
      newWireRead->moveBefore(use.getOwner());
    }

    auto connect = builder.create<AssignOp>(newWire, result);
    connect->moveAfter(&op);
  }
}

/// Transform "a + -cst" ==> "a - cst" for prettier output.  This returns the
/// first operation emitted.
static Operation *rewriteAddWithNegativeConstant(comb::AddOp add,
                                                 hw::ConstantOp rhsCst) {
  ImplicitLocOpBuilder builder(add.getLoc(), add);

  // Get the positive constant.
  auto negCst = builder.create<hw::ConstantOp>(-rhsCst.getValue());
  auto sub = builder.create<comb::SubOp>(add.getOperand(0), negCst);
  add.getResult().replaceAllUsesWith(sub);
  add.erase();
  if (rhsCst.use_empty())
    rhsCst.erase();
  return negCst;
}

/// Given an operation in a procedural region, scan up the region tree to find
/// the first operation in a graph region (typically an always or initial op).
///
/// By looking for a graph region, we will stop at graph-region #ifdef's that
/// may enclose this operation.
static Operation *findParentInNonProceduralRegion(Operation *op) {
  Operation *parentOp = op->getParentOp();
  assert(parentOp->hasTrait<ProceduralRegion>() &&
         "we should only be hoisting from procedural");
  while (parentOp->getParentOp()->hasTrait<ProceduralRegion>())
    parentOp = parentOp->getParentOp();
  return parentOp;
}

/// This function is invoked on side effecting Verilog expressions when we're in
/// 'disallowLocalVariables' mode for old Verilog clients.  This ensures that
/// any side effecting expressions are only used by a single BPAssign to a
/// sv.reg operation.  This ensures that the verilog emitter doesn't have to
/// worry about spilling them.
///
/// This returns true if the op was rewritten, false otherwise.
static bool rewriteSideEffectingExpr(Operation *op) {
  assert(op->getNumResults() == 1 && "isn't a verilog expression");

  // Check to see if this is already rewritten.
  if (op->hasOneUse()) {
    if (auto assign = dyn_cast<BPAssignOp>(*op->user_begin()))
      if (assign.dest().getDefiningOp<RegOp>())
        return false;
  }

  // Otherwise, we have to transform it.  Insert a reg at the top level, make
  // everything using the side effecting expression read the reg, then assign to
  // it after the side effecting operation.
  Value opValue = op->getResult(0);

  // Scan to the top of the region tree to find out where to insert the reg.
  Operation *parentOp = findParentInNonProceduralRegion(op);
  OpBuilder builder(parentOp);
  auto reg = builder.create<RegOp>(op->getLoc(), opValue.getType());

  // Everything using the expr now uses a read_inout of the reg.
  auto value = builder.create<ReadInOutOp>(op->getLoc(), reg);
  opValue.replaceAllUsesWith(value);

  // We assign the side effect expr to the reg immediately after that expression
  // is computed.
  builder.setInsertionPointAfter(op);
  builder.create<BPAssignOp>(op->getLoc(), reg, opValue);
  return true;
}

/// This function is called for non-side-effecting Verilog expressions when
/// we're in 'disallowLocalVariables' mode for old Verilog clients.  It hoists
/// non-constant expressions out to the top level so they don't turn into local
/// variable declarations.
static bool hoistNonSideEffectExpr(Operation *op) {
  // Never hoist "always inline" expressions except for inout stuffs - they will
  // never generate a temporary and in fact must always be emitted inline.
  if (isExpressionAlwaysInline(op) &&
      !(isa<sv::ReadInOutOp>(op) ||
        op->getResult(0).getType().isa<hw::InOutType>()))
    return false;

  // Scan to the top of the region tree to find out where to move the op.
  Operation *parentOp = findParentInNonProceduralRegion(op);

  // We can typically hoist all the way out to the top level in one step, but
  // there may be intermediate operands that aren't hoistable.  If so, just
  // hoist one level.
  bool cantHoist = false;
  if (llvm::any_of(op->getOperands(), [&](Value operand) -> bool {
        // The operand value dominates the original operation, but may be
        // defined in one of the procedural regions between the operation and
        // the top level of the module.  We can tell this quite efficiently by
        // looking for ops in a procedural region - because procedural regions
        // live in graph regions but not visa-versa.
        Operation *operandOp = operand.getDefiningOp();
        if (!operandOp) // References to ports are always ok.
          return false;

        if (operandOp->getParentOp()->hasTrait<ProceduralRegion>()) {
          cantHoist |= operandOp->getBlock() == op->getBlock();
          return true;
        }
        return false;
      })) {

    // If the operand is in the same block as the expression then we can't hoist
    // this out at all.
    if (cantHoist)
      return false;

    // Otherwise, we can hoist it, but not all the way out in one step.  Just
    // hoist one level out.
    parentOp = op->getParentOp();
  }

  op->moveBefore(parentOp);
  return true;
}

/// For each module we emit, do a prepass over the structure, pre-lowering and
/// otherwise rewriting operations we don't want to emit.
void ExportVerilog::prepareHWModule(Block &block,
                                    const LoweringOptions &options) {

  // First step, check any nested blocks that exist in this region.  This walk
  // can pull things out to our level of the hierarchy.
  for (auto &op : block) {
    // If the operations has regions, prepare each of the region bodies.
    for (auto &region : op.getRegions()) {
      if (!region.empty())
        prepareHWModule(region.front(), options);
    }
  }

  // Next, walk all of the operations at this level.

  // True if these operations are in a procedural region.
  bool isProceduralRegion = block.getParentOp()->hasTrait<ProceduralRegion>();
  for (Block::iterator opIterator = block.begin(), e = block.end();
       opIterator != e;) {
    auto &op = *opIterator++;

    // Lower variadic fully-associative operations with more than two operands
    // into balanced operand trees so we can split long lines across multiple
    // statements.
    // TODO: This is checking the Commutative property, which doesn't seem
    // right in general.  MLIR doesn't have a "fully associative" property.
    if (op.getNumOperands() > 2 && op.getNumResults() == 1 &&
        op.hasTrait<mlir::OpTrait::IsCommutative>() &&
        mlir::MemoryEffectOpInterface::hasNoEffect(&op) &&
        op.getNumRegions() == 0 && op.getNumSuccessors() == 0 &&
        (op.getAttrs().empty() ||
         (op.getAttrs().size() == 1 && op.hasAttr("sv.namehint")))) {
      // Lower this operation to a balanced binary tree of the same operation.
      SmallVector<Operation *> newOps;
      auto result = lowerFullyAssociativeOp(op, op.getOperands(), newOps);
      op.getResult(0).replaceAllUsesWith(result);
      op.erase();

      // Make sure we revisit the newly inserted operations.
      opIterator = Block::iterator(newOps.front());
      continue;
    }

    // Turn a + -cst  ==> a - cst
    if (auto addOp = dyn_cast<comb::AddOp>(op)) {
      if (auto cst = addOp.getOperand(1).getDefiningOp<hw::ConstantOp>()) {
        assert(addOp.getNumOperands() == 2 && "commutative lowering is done");
        if (cst.getValue().isNegative()) {
          Operation *firstOp = rewriteAddWithNegativeConstant(addOp, cst);
          opIterator = Block::iterator(firstOp);
          continue;
        }
      }
    }

    // Name legalization should have happened in a different pass for these sv
    // elements and we don't want to change their name through re-legalization
    // (e.g. letting a temporary take the name of an unvisited wire). Adding
    // them now ensures any temporary generated will not use one of the names
    // previously declared.
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      // Anchor return values to wires early
      lowerInstanceResults(instance);
      // Anchor ports of bound instances
      if (instance->hasAttr("doNotPrint"))
        lowerBoundInstance(instance);
    }

    // Force any expression used in the event control of an always process to be
    // a trivial wire, if the corresponding option is set.
    if (!options.allowExprInEventControl) {
      auto enforceWire = [&](Value expr) {
        // Direct port uses are fine.
        if (expr.isa<BlockArgument>())
          return;
        // If this is a read from a wire, we're fine.
        if (auto read = expr.getDefiningOp<ReadInOutOp>())
          if (read.input().getDefiningOp<WireOp>())
            return;
        auto builder = ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), &block);
        auto newWire = builder.create<WireOp>(expr.getType());
        builder.setInsertionPoint(&op);
        builder.create<AssignOp>(newWire, expr);
        auto newWireRead = builder.create<ReadInOutOp>(newWire);
        op.replaceUsesOfWith(expr, newWireRead);
      };
      if (auto always = dyn_cast<AlwaysOp>(op)) {
        for (auto clock : always.clocks())
          enforceWire(clock);
        continue;
      }
      if (auto always = dyn_cast<AlwaysFFOp>(op)) {
        enforceWire(always.clock());
        if (auto reset = always.reset())
          enforceWire(reset);
        continue;
      }
    }

    // If the target doesn't support local variables, hoist all the expressions
    // out to the nearest non-procedural region.
    if (options.disallowLocalVariables && isVerilogExpression(&op) &&
        isProceduralRegion) {

      // Force any side-effecting expressions in nested regions into a sv.reg
      // if we aren't allowing local variable declarations.  The Verilog emitter
      // doesn't want to have to have to know how to synthesize a reg in the
      // case they have to be spilled for whatever reason.
      if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op)) {
        if (rewriteSideEffectingExpr(&op))
          continue;
      }

      // Hoist other expressions out to the parent region.
      //
      // NOTE: This effectively disables inlining of expressions into if
      // conditions, $fwrite statements, and instance inputs.  We could be
      // smarter in ExportVerilog itself, but we'd have to teach it to put
      // spilled expressions (due to line length, multiple-uses, and
      // non-inlinable expressions) in the outer scope.
      if (hoistNonSideEffectExpr(&op))
        continue;
    }

    // Duplicate "always inline" expression for each of their users and move
    // them to be next to their users.
    if (isExpressionAlwaysInline(&op)) {
      lowerAlwaysInlineOperation(&op);
      continue;
    }
  }

  // Now that all the basic ops are settled, check for any use-before def issues
  // in graph regions.  Lower these into explicit wires to keep the emitter
  // simple.
  if (!isProceduralRegion) {
    SmallPtrSet<Operation *, 32> seenOperations;

    for (auto &op : llvm::make_early_inc_range(block)) {
      // Check the users of any expressions to see if they are
      // lexically below the operation itself.  If so, it is being used out
      // of order.
      bool haveAnyOutOfOrderUses = false;
      for (auto *userOp : op.getUsers()) {
        // If the user is in a suboperation like an always block, then zip up
        // to the operation that uses it.
        while (&block != &userOp->getParentRegion()->front())
          userOp = userOp->getParentOp();

        if (seenOperations.count(userOp)) {
          haveAnyOutOfOrderUses = true;
          break;
        }
      }

      // Remember that we've seen this operation.
      seenOperations.insert(&op);

      // If all the uses of the operation are below this, then we're ok.
      if (!haveAnyOutOfOrderUses)
        continue;

      // If this is a reg/wire declaration, then we move it to the top of the
      // block.  We can't abstract the inout result.
      if (op.getNumResults() == 1 &&
          op.getResult(0).getType().isa<InOutType>() &&
          op.getNumOperands() == 0) {
        op.moveBefore(&block.front());
        continue;
      }

      // If this is a constant, then we move it to the top of the block.
      if (isConstantExpression(&op)) {
        op.moveBefore(&block.front());
        continue;
      }

      // Otherwise, we need to lower this to a wire to resolve this.
      lowerUsersToTemporaryWire(op);
    }
  }
}
