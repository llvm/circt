//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProjectionUtils.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "mlir/IR/Matchers.h"
#include <cstdint>

using namespace mlir;
using namespace circt;
using namespace llhd;

/// Check if an op is one of the supported projection types.
bool llhd::isProjection(const Operation *op) {
  return isa_and_nonnull<SigArrayGetOp, SigExtractOp, SigStructExtractOp>(op);
}

/// Calculate a projection's range.
/// Returns nullopt if the range could not be calculated for this projection.
/// Asserts if the provided Operation is not a (supported) projection.
std::optional<ProjectionRange> llhd::getProjectionRange(Operation *op) {
  APInt index;
  uint64_t offset = 0;
  int64_t width = 0;
  Value dynIndex;

  TypeSwitch<Operation *>(op)
      .Case<SigArrayGetOp>([&](auto op) {
        if (!matchPattern(op.getIndex(), m_ConstantInt(&index))) {
          dynIndex = op.getIndex();
        } else {
          width = hw::getBitWidth(getStoredType(op.getResult()));
          offset = index.getZExtValue() * width;
        }
      })
      .Case<SigExtractOp>([&](auto op) {
        if (!matchPattern(op.getLowBit(), m_ConstantInt(&index))) {
          dynIndex = op.getLowBit();
        } else {
          width = op.getResultWidth();
          offset = index.getZExtValue();
        }
      })
      .Case<SigStructExtractOp>([&](auto op) {
        auto inputType = cast<RefType>(op.getInput().getType()).getNestedType();
        if (auto unionType = dyn_cast<hw::UnionType>(inputType)) {
          width = hw::getBitWidth(unionType);
        } else {
          auto structType = cast<hw::StructType>(inputType);
          auto elements = structType.getElements();
          auto fieldIdx = *structType.getFieldIndex(op.getFieldAttr());
          bool fieldWidthsKnown = true;
          for (auto field : elements.drop_front(fieldIdx + 1)) {
            auto fieldWidth = hw::getBitWidth(field.type);
            if (fieldWidth < 0) {
              fieldWidthsKnown = false;
              break;
            }
            offset += fieldWidth;
          }
          width =
              fieldWidthsKnown ? hw::getBitWidth(elements[fieldIdx].type) : -1;
        }
      })
      .Default([](auto) { llvm_unreachable("Unsupported projection type."); });

  if (dynIndex)
    return DynProjectionRange{dynIndex};

  if (width < 0)
    return std::nullopt;

  return ConstProjectionRange{offset, static_cast<uint64_t>(width)};
}

/// Check whether two projections provably describe disjoint storage.
bool llhd::isProvablyDisjoint(const ProjectionRange &a,
                              const ProjectionRange &b) {
  auto *ca = std::get_if<ConstProjectionRange>(&a);
  auto *cb = std::get_if<ConstProjectionRange>(&b);
  // At least one side is dynamic: can't prove disjointness.
  if (!ca || !cb)
    return false;
  // The ranges don't overlap: disjoint.
  if (ca->offset + ca->width <= cb->offset ||
      cb->offset + cb->width <= ca->offset)
    return true;
  return false;
}

Operation *llhd::getRootSignal(Value sig) {
  if (!sig)
    return nullptr;
  Operation *op = sig.getDefiningOp();

  while (isProjection(op)) {
    op = op->getOperand(0).getDefiningOp();
    if (!op)
      return nullptr;
  }
  if (isa_and_nonnull<SignalOp>(op))
    return op;
  return nullptr;
}

std::pair<Operation *, std::optional<ProjectionRange>>
llhd::getRootSignalAndRange(Value sig) {
  if (!sig)
    return {};

  // Initialize the range parameters.
  bool isConst = true;
  uint64_t offset = 0;
  auto width = hw::getBitWidth(getStoredType(sig));
  if (width < 0)
    isConst = false;

  // Walk up the projection chain.
  Operation *op = sig.getDefiningOp();
  while (isProjection(op)) {
    auto projRange = getProjectionRange(op);
    // Update the range only if all the encountered projections are constant.
    if (isConst && projRange && !isDynamic(*projRange)) {
      auto constRange = std::get<ConstProjectionRange>(*projRange);
      offset += constRange.offset;
    } else {
      isConst = false;
    }
    op = op->getOperand(0).getDefiningOp();
  }

  // If the chain doesn't bottom out in a signal there's not value nor range
  // to return.
  if (!isa_and_nonnull<SignalOp>(op))
    return {};

  if (!isConst)
    return {op, std::nullopt};

  return {op, ConstProjectionRange{offset, static_cast<uint64_t>(width)}};
}
