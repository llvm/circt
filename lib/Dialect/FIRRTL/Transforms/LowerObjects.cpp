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
  // Input assignments in port order; none are null.
  SmallVector<PropAssignOp> assignments;
  SmallVector<ObjectSubfieldOp> fields;
};

struct LowerObjectsPass
    : public circt::firrtl::impl::LowerObjectsBase<LowerObjectsPass> {
  void runOnOperation() override;

private:
  LogicalResult collectInputs(ObjectOp object, ObjectInputs &inputs);
  LogicalResult lowerObject(ObjectOp object,
                            DenseMap<Operation *, size_t> &positions,
                            DenseSet<Operation *> &converted);
};
} // namespace

/// Check drivers by port index, since a port can have multiple accesses.
LogicalResult LowerObjectsPass::collectInputs(ObjectOp object,
                                              ObjectInputs &inputs) {
  auto type = object.getType();
  SmallVector<PropAssignOp> assignments(type.getNumElements());
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
      auto &previous = assignments[field.getIndex()];
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

  for (auto [index, element] : llvm::enumerate(type.getElements())) {
    if (element.direction != Direction::In)
      continue;
    auto assign = assignments[index];
    if (!assign)
      return object.emitError("uninitialized input port ") << element.name;
    inputs.assignments.push_back(assign);
  }
  return success();
}

LogicalResult
LowerObjectsPass::lowerObject(ObjectOp object,
                              DenseMap<Operation *, size_t> &positions,
                              DenseSet<Operation *> &converted) {
  auto position = positions.lookup(object);
  // Find the operations that must move before this object.
  // Only move operations in this block that have no regions.
  // FIRRTL allows these operations to move even if they have side effects.
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
      // Skip lowering if there is a cycle.
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
        operands.push_back(assign.getSrc());
    } else {
      if (op->getNumRegions())
        return success();
      llvm::append_range(operands, op->getOperands());
    }

    worklist.emplace_back(op, true);
    for (auto operand : llvm::reverse(operands)) {
      auto *def = operand.getDefiningOp();
      if (!def)
        continue;
      auto it = positions.find(def);
      // Skip lowering if a value comes from a nested region.
      if (it == positions.end())
        return success();
      if (it->second < position)
        continue;
      worklist.emplace_back(def, false);
    }
  }

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
    // Earlier replacements may have changed the sources.
    for (auto assign : info.assignments)
      inputs.push_back(assign.getSrc());
    auto create = ObjectCreateOp::create(builder, declaration.getLoc(),
                                         declaration.getType(),
                                         declaration.getNameAttr(), inputs);
    create->setDiscardableAttrs(declaration->getDiscardableAttrDictionary());
    positions[create] = position;
    for (auto assign : info.assignments) {
      positions.erase(assign);
      assign.erase();
      ++numRemoved;
    }
    for (auto field : info.fields) {
      positions.erase(field);
      field.erase();
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
  auto *block = getOperation().getBodyBlock();
  // Keep the original object list; lowering may erase later objects.
  SmallVector<ObjectOp> objects(block->getOps<ObjectOp>());
  if (objects.empty())
    return;

  // Use the original order to check which values are already available.
  DenseMap<Operation *, size_t> positions;
  for (auto [index, op] : llvm::enumerate(*block))
    positions[&op] = index;
  DenseSet<Operation *> converted;
  for (auto object : objects) {
    if (converted.contains(object.getOperation()))
      continue;
    if (failed(lowerObject(object, positions, converted))) {
      signalPassFailure();
      return;
    }
  }
}
