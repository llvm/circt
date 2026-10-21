//===- LowerObjects.cpp - Convert objects to SSA values -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"

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
struct LowerObjectsPass
    : public circt::firrtl::impl::LowerObjectsBase<LowerObjectsPass> {
  void runOnOperation() override;
};
} // namespace

/// Collect input assignments before changing the IR. A field may have several
/// subfield operations, so detect multiple drivers by field index rather than
/// by the SSA value of an individual subfield.
static LogicalResult lowerObject(ObjectOp object) {
  auto type = object.getType();
  SmallVector<PropAssignOp> assignments(type.getNumElements());
  SmallVector<ObjectSubfieldOp> inputFields;
  for (auto *user : object->getUsers()) {
    auto field = dyn_cast<ObjectSubfieldOp>(user);
    if (!field || type.getElement(field.getIndex()).direction == Direction::Out)
      continue;
    inputFields.push_back(field);
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

  SmallVector<Value> inputs;
  for (auto [index, element] : llvm::enumerate(type.getElements())) {
    if (element.direction == Direction::Out)
      continue;
    auto assign = assignments[index];
    if (!assign)
      return object.emitError("uninitialized input port ") << element.name;
    inputs.push_back(assign.getSrc());
  }

  OpBuilder builder(object);
  auto create = ObjectCreateOp::create(builder, object.getLoc(), type,
                                       object.getNameAttr(), inputs);
  // Preserve any discardable attributes attached to the declaration.
  create->setDiscardableAttrs(object->getDiscardableAttrDictionary());
  for (auto assign : assignments)
    if (assign)
      assign.erase();
  for (auto field : inputFields)
    field.erase();
  object.replaceAllUsesWith(create.getResult());
  object.erase();
  return success();
}

void LowerObjectsPass::runOnOperation() {
  // Collect first since lowering erases the declarations and some of their
  // users. Keep block order deterministic when restoring SSA dominance.
  SmallVector<ObjectOp> objects;
  getOperation().walk([&](ObjectOp object) { objects.push_back(object); });
  llvm::SetVector<Block *> blocks;
  for (auto object : objects) {
    blocks.insert(object->getBlock());
    if (failed(lowerObject(object)))
      return signalPassFailure();
  }

  // Input assignments may follow both the object and its output accesses.
  // Turning their sources into operands introduces forward references; sort
  // the affected blocks to restore SSA dominance, including object chains.
  for (auto *block : blocks)
    if (!sortTopologically(block)) {
      block->getParentOp()->emitError("cyclic object input dependencies");
      return signalPassFailure();
    }
}
