//===- LowerObjects.cpp - Convert objects to SSA values -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWEROBJECTS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::firrtl;

namespace {
struct ObjectInputs {
  SmallVector<PropAssignOp> assignments;
  SmallVector<ObjectSubfieldOp> fields;
};

struct LowerObjectsPass
    : public circt::firrtl::impl::LowerObjectsBase<LowerObjectsPass> {
  void runOnOperation() override;

private:
  LogicalResult collectInputs(ObjectOp object, ObjectInputs &inputs);
  LogicalResult lowerObject(ObjectOp object, DominanceInfo &dominance,
                            DenseMap<Operation *, size_t> &positions,
                            DenseSet<Operation *> &converted);
};
} // namespace

/// Collect input assignments before changing the IR. A field may have several
/// subfield operations, so detect multiple drivers by field index rather than
/// by the SSA value of an individual subfield.
LogicalResult LowerObjectsPass::collectInputs(ObjectOp object,
                                              ObjectInputs &inputs) {
  auto type = object.getType();
  inputs.assignments.resize(type.getNumElements());
  for (auto *user : object->getUsers()) {
    auto field = dyn_cast<ObjectSubfieldOp>(user);
    if (!field || type.getElement(field.getIndex()).direction == Direction::Out)
      continue;
    inputs.fields.push_back(field);
    for (auto &use : field->getUses()) {
      auto assign = dyn_cast<PropAssignOp>(use.getOwner());
      if (!assign || use.getOperandNumber() != 0)
        return field.emitOpError("input port may only be used as an assignment "
                                 "destination");
      auto &previous = inputs.assignments[field.getIndex()];
      if (previous) {
        auto diag = assign.emitError("input port ")
                    << type.getElement(field.getIndex()).name
                    << " cannot be driven by multiple operations";
        diag.attachNote(previous.getLoc()) << "other driver is here";
        return failure();
      }
      previous = assign;
    }
  }

  for (auto [index, element] : llvm::enumerate(type.getElements()))
    if (element.direction == Direction::In && !inputs.assignments[index])
      return object.emitError("uninitialized input port ") << element.name;
  return success();
}

LogicalResult
LowerObjectsPass::lowerObject(ObjectOp object, DominanceInfo &dominance,
                              DenseMap<Operation *, size_t> &positions,
                              DenseSet<Operation *> &converted) {
  auto position = positions.lookup(object);
  // Plan a local topological order before mutating the IR. Object declarations
  // have virtual dependencies on their input drivers. Other dependencies must
  // be regionless operations in the same block. FIRRTL's declarative semantics
  // allow reordering these producers without requiring them to be pure.
  // Values that already dominate the declaration require no changes.
  DenseMap<Operation *, ObjectInputs> objectInputs;
  DenseMap<Operation *, bool> visited;
  SmallVector<std::pair<Operation *, bool>> worklist{{object, false}};
  SmallVector<Operation *> order;
  while (!worklist.empty()) {
    auto [op, exiting] = worklist.pop_back_val();
    if (exiting) {
      visited[op] = true;
      order.push_back(op);
      continue;
    }
    auto [it, inserted] = visited.try_emplace(op, false);
    if (!inserted) {
      // Encountering an active node means that creating this object would
      // introduce an SSA cycle. Leave this entire plan untouched.
      if (!it->second)
        return success();
      continue;
    }

    SmallVector<Value> operands;
    if (auto dependency = dyn_cast<ObjectOp>(op)) {
      auto &inputs = objectInputs[op];
      if (failed(collectInputs(dependency, inputs)))
        return failure();
      for (auto assign : inputs.assignments)
        if (assign)
          operands.push_back(assign.getSrc());
    } else {
      if (op->getNumRegions())
        return success();
      llvm::append_range(operands, op->getOperands());
    }

    worklist.emplace_back(op, true);
    for (auto operand : llvm::reverse(operands)) {
      auto *def = operand.getDefiningOp();
      if (!def || def->getBlock() != object->getBlock()) {
        if (dominance.properlyDominates(operand, object))
          continue;
        // In particular, do not pull values out of nested regions or move
        // operations across block boundaries.
        return success();
      }
      if (positions.lookup(def) < position)
        continue;
      worklist.emplace_back(def, false);
    }
  }

  // The plan is valid. Materialize its dependencies before the declaration,
  // then replace it directly with the value-like object, without a wire.
  OpBuilder builder(object);
  for (auto *op : order) {
    auto declaration = dyn_cast<ObjectOp>(op);
    if (!declaration) {
      op->moveBefore(object);
      positions[op] = position;
      continue;
    }
    auto &info = objectInputs[op];
    SmallVector<Value> inputs;
    // Read the sources now: a preceding replacement may have updated them.
    for (auto assign : info.assignments)
      if (assign)
        inputs.push_back(assign.getSrc());
    auto create = ObjectCreateOp::create(builder, declaration.getLoc(),
                                         declaration.getType(),
                                         declaration.getNameAttr(), inputs);
    create->setDiscardableAttrs(declaration->getDiscardableAttrDictionary());
    positions[create] = position;
    for (auto assign : info.assignments)
      if (assign) {
        positions.erase(assign);
        assign.erase();
        ++numInputAssignmentsRemoved;
        ++numRemoved;
      }
    for (auto field : info.fields) {
      positions.erase(field);
      field.erase();
      ++numInputFieldsRemoved;
      ++numRemoved;
    }
    declaration.replaceAllUsesWith(create.getResult());
    converted.insert(op);
    positions.erase(op);
    declaration.erase();
    ++numObjectsConverted;
    ++numRemoved;
  }
  return success();
}

void LowerObjectsPass::runOnOperation() {
  DominanceInfo dominance;
  auto result = getOperation()->walk([&](Block *block) {
    // A plan may also convert declarations later in the block. Snapshot the
    // original declarations and check the converted set before dereferencing
    // them, since those operations may already have been erased.
    SmallVector<ObjectOp> objects(block->getOps<ObjectOp>());
    if (objects.empty())
      return WalkResult::advance();

    // Stable positions avoid recomputing MLIR's whole-block operation order
    // after every hoist. Declarations are processed in original block order;
    // anything hoisted to an earlier declaration dominates all later ones.
    DenseMap<Operation *, size_t> positions;
    for (auto [index, op] : llvm::enumerate(*block))
      positions[&op] = index;
    DenseSet<Operation *> converted;
    for (auto object : objects) {
      if (converted.contains(object.getOperation()))
        continue;
      if (failed(lowerObject(object, dominance, positions, converted)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    signalPassFailure();
}
