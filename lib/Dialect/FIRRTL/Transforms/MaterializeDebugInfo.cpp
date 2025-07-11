//===- MaterializeDebugInfo.cpp - DI materialization ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

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
  auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());

  // Create DI variables for each port.
  for (const auto &[port, value] :
       llvm::zip(module.getPorts(), module.getArguments())) {
    materializeVariable(builder, port.name, value);
  }

  // Create DI variables for each declaration in the module body.
  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op).Case<WireOp, NodeOp, RegOp, RegResetOp>(
        [&](auto op) {
          builder.setInsertionPointAfter(op);
          materializeVariable(builder, op.getNameAttr(), op.getResult());
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
    builder.create<debug::VariableOp>(value.getLoc(), name, dbgValue,
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
          auto subOp = builder.create<SubfieldOp>(value.getLoc(), value, index);
          subOps.push_back(subOp);
          if (auto dbgValue = convertToDebugAggregates(builder, subOp)) {
            fields.push_back(dbgValue);
            names.push_back(element.name);
          }
        }
        auto result = builder.create<debug::StructOp>(
            value.getLoc(), fields, builder.getArrayAttr(names));
        for (auto *subOp : subOps)
          if (subOp->use_empty())
            subOp->erase();
        return result;
      })
      .Case<FVectorType>([&](auto type) -> Value {
        SmallVector<Value> elements;
        SmallVector<Operation *> subOps;
        for (unsigned index = 0; index < type.getNumElements(); ++index) {
          auto subOp = builder.create<SubindexOp>(value.getLoc(), value, index);
          subOps.push_back(subOp);
          if (auto dbgValue = convertToDebugAggregates(builder, subOp))
            elements.push_back(dbgValue);
        }
        Value result;
        if (!elements.empty() && elements.size() == type.getNumElements())
          result = builder.create<debug::ArrayOp>(value.getLoc(), elements);
        for (auto *subOp : subOps)
          if (subOp->use_empty())
            subOp->erase();
        return result;
      })
      .Case<FIRRTLBaseType>(
          [&](auto type) { return type.isGround() ? value : Value{}; })
      .Default({});
}
