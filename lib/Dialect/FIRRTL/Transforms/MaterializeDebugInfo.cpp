//===- MaterializeDebugInfo.cpp - DI materialization ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass must run BEFORE `firrtl-lower-intrinsics`. The skip-set built
// from `circt_debug_var`-intrinsics keeps us from duplicating debug variables
// that the intrinsic lowering will materialize. If the intrinsics have already
// been lowered, this pass detects the presence of `dbg.variable` ops and
// bails out per-module to avoid creating duplicates.
//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_MATERIALIZEDEBUGINFO
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
struct MaterializeDebugInfoPass
    : public circt::firrtl::impl::MaterializeDebugInfoBase<
          MaterializeDebugInfoPass> {
  void runOnOperation() override;
  void materializeVariable(OpBuilder &builder, StringAttr name, Value value);
  Value convertToDebugAggregates(OpBuilder &builder, Value value);
};
} // namespace

void MaterializeDebugInfoPass::runOnOperation() {
  auto module = getOperation();

  // 1-operand vars are keyed by SSA Value (no cross-scope name collisions);
  // 0-operand (memory) vars are keyed by name. `seenNames` spans both arities
  // for duplicate-name warnings.
  bool hasExistingVariables = false;
  DenseSet<Value> coveredByIntrinsic;
  DenseSet<StringAttr> coveredNameByIntrinsic;
  DenseSet<StringAttr> seenNames;
  module.walk([&](Operation *op) -> WalkResult {
    if (isa<debug::VariableOp>(op)) {
      hasExistingVariables = true;
      return WalkResult::interrupt();
    }
    auto gi = dyn_cast<firrtl::GenericIntrinsicOp>(op);
    if (!gi || gi.getIntrinsic() != "circt_debug_var")
      return WalkResult::advance();
    auto varName = GenericIntrinsic(gi).getParamValue<StringAttr>("name");
    if (!varName || varName.getValue().empty())
      return WalkResult::advance();
    if (!seenNames.insert(varName).second)
      gi.emitWarning() << "duplicate circt_debug_var '" << varName.getValue()
                       << "' ignored by MaterializeDebugInfo";
    if (gi.getNumOperands() == 1)
      coveredByIntrinsic.insert(gi.getOperand(0));
    else if (gi.getNumOperands() == 0)
      coveredNameByIntrinsic.insert(varName);
    return WalkResult::advance();
  });
  if (hasExistingVariables) {
    module.emitWarning()
        << "MaterializeDebugInfo: dbg.variable ops already present "
           "(LowerIntrinsics likely ran first); skipping to avoid duplicates";
    return;
  }

  auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());

  for (const auto &[port, value] :
       llvm::zip(module.getPorts(), module.getArguments())) {
    if (!coveredByIntrinsic.contains(value) &&
        !coveredNameByIntrinsic.contains(port.name))
      materializeVariable(builder, port.name, value);
  }

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op).Case<WireOp, NodeOp, RegOp, RegResetOp>(
        [&](auto op) {
          if (!coveredByIntrinsic.contains(op.getResult()) &&
              !coveredNameByIntrinsic.contains(op.getNameAttr())) {
            builder.setInsertionPointAfter(op);
            materializeVariable(builder, op.getNameAttr(), op.getResult());
          }
        });
  });
}

/// Materialize debug variable ops for a value.
void MaterializeDebugInfoPass::materializeVariable(OpBuilder &builder,
                                                   StringAttr name,
                                                   Value value) {
  if (!name || isUselessName(name.getValue()))
    return;
  if (name.getValue().starts_with("_"))
    return;
  if (auto dbgValue = convertToDebugAggregates(builder, value))
    debug::VariableOp::create(builder, value.getLoc(), name, dbgValue,
                              /*scope=*/Value{});
}

/// Unpack all aggregates in a FIRRTL value and repack them as debug aggregates.
/// For example, converts a FIRRTL vector `v` into `dbg.array [v[0],v[1],...]`.
Value MaterializeDebugInfoPass::convertToDebugAggregates(OpBuilder &builder,
                                                         Value value) {
  return FIRRTLTypeSwitch<Type, Value>(value.getType())
      .Case<BundleType>([&](auto type) {
        SmallVector<Value> fields;
        SmallVector<Attribute> names;
        SmallVector<Operation *> subOps;
        for (auto [index, element] : llvm::enumerate(type.getElements())) {
          auto subOp =
              SubfieldOp::create(builder, value.getLoc(), value, index);
          subOps.push_back(subOp);
          if (auto dbgValue = convertToDebugAggregates(builder, subOp)) {
            fields.push_back(dbgValue);
            names.push_back(element.name);
          }
        }
        auto result = debug::StructOp::create(builder, value.getLoc(), fields,
                                              builder.getArrayAttr(names));
        for (auto *subOp : subOps)
          if (subOp->use_empty())
            subOp->erase();
        return result;
      })
      .Case<FVectorType>([&](auto type) -> Value {
        SmallVector<Value> elements;
        SmallVector<Operation *> subOps;
        for (unsigned index = 0; index < type.getNumElements(); ++index) {
          auto subOp =
              SubindexOp::create(builder, value.getLoc(), value, index);
          subOps.push_back(subOp);
          if (auto dbgValue = convertToDebugAggregates(builder, subOp))
            elements.push_back(dbgValue);
        }
        Value result;
        if (!elements.empty() && elements.size() == type.getNumElements())
          result = debug::ArrayOp::create(builder, value.getLoc(), elements);
        for (auto *subOp : subOps)
          if (subOp->use_empty())
            subOp->erase();
        return result;
      })
      .Case<FIRRTLBaseType>(
          [&](auto type) { return type.isGround() ? value : Value{}; })
      .Default({});
}
