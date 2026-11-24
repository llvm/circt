//===- CheckVarDrivers.cpp - Check SystemVerilog variable drivers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass checks if it is illegal for a variable to have
// more than one continuous driver, or to be driven by both a continous
// assignment and a procedural statement, on overlapping bits. Each write's
// "longest static prefix" (see SV Spec 11.5.3) is resolved and flattened to a
// half-open bit interval [low, high) on the root variable; two drivers
// conflict if their intervals overlap and at least one is continuous.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <optional>
#include <tuple>

namespace circt::sv {
#define GEN_PASS_DEF_CHECKVARDRIVERS
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace circt::sv

using namespace circt;
using namespace circt::sv;

namespace {

//===----------------------------------------------------------------------===//
// Driver model
//===----------------------------------------------------------------------===//

enum class DriverKind { Continuous, Procedural };

struct Driver {
  int64_t lo;
  int64_t hi;
  DriverKind kind;
  Operation *op;
};

/// The result of resolving a write's longest static prefix: which root
/// variable it writes, and which bits of that root it covers.
struct ResolvedRange {
  Value root;
  int64_t lo;
  int64_t hi;
};

//===----------------------------------------------------------------------===//
// Static prefix resolution
//===----------------------------------------------------------------------===//

/// TODO: Once we have migrated from !hw.inout declarations this function will
/// be made redundant and we will just use the element type T of !sv.var<T>
/// directly.
static Type getLvalueElementType(Type type) {
  if (auto var = dyn_cast<sv::VarType>(type))
    return var.getElementType();
  if (auto inout = dyn_cast<hw::InOutType>(type))
    return inout.getElementType();
  return {};
}

/// Returns the width of `type` in bits, or std::nullopt if it is null or has
/// no known width.
static std::optional<int64_t> getTypeBitWidth(Type type) {
  if (!type)
    return std::nullopt;
  int64_t width = hw::getBitWidth(type);
  if (width < 0)
    return std::nullopt;
  return width;
}

/// Returns the width in bits of the data held by handle
static std::optional<int64_t> getLvalueBitWidth(Value handle) {
  return getTypeBitWidth(getLvalueElementType(handle.getType()));
}

/// Returns true if `value` is the result of an op that declares a SystemVerilog
/// variable.
///
/// TODO: sv.reg and sv.logic are legacy !hw.inout declarations. Drop them once
/// variables are represented solely by sv.var. Thus we will call sv.var op
/// directly and this function will be made redundant.
static bool isVariableRoot(Value value) {
  return isa_and_nonnull<sv::VarOp, sv::RegOp, sv::LogicOp>(
      value.getDefiningOp());
}

/// Matches `value` against a constant that can be used as an array index or
/// part-select base. Returns std::nullopt if `value` is not a resolvable
/// constant (e.g. a dynamic index), in which case the caller must treat the
/// enclosing select as covering its whole parent.
static std::optional<int64_t> getConstantIndex(Value value) {
  APInt constant;
  if (!mlir::matchPattern(value, mlir::m_ConstantInt(&constant)) ||
      constant.getActiveBits() > 63)
    return std::nullopt;
  return constant.getZExtValue();
}

/// Resolves the longest static prefix of a handle, it walks back through
/// the projection ops to the root variable, accumulating a bit offset.
/// As soon as a dynamic (non-constant) index or base is found, we collapse
/// the range to whole enclosing aggregate.
class StaticPrefixResolver {
public:
  static std::optional<ResolvedRange> resolve(Value dest) {
    auto width = getLvalueBitWidth(dest);
    if (!width)
      return std::nullopt;
    return StaticPrefixResolver().walk(dest, *width);
  }

private:
  std::optional<ResolvedRange> walk(Value value, int64_t selectedWidth) {
    while (true) {
      if (isVariableRoot(value))
        return rootRange(value, selectedWidth);

      if (auto idx = value.getDefiningOp<sv::ArrayIndexInOutOp>()) {
        if (failed(stepArrayIndex(idx)))
          return wholeParentRange(idx.getInput());
        value = idx.getInput();
        continue;
      }

      if (auto select = value.getDefiningOp<sv::IndexedPartSelectInOutOp>()) {
        if (failed(stepPartSelect(select)))
          return wholeParentRange(select.getInput());
        value = select.getInput();
        continue;
      }

      if (auto field = value.getDefiningOp<sv::StructFieldInOutOp>()) {
        if (failed(stepStructField(field)))
          return std::nullopt;
        value = field.getInput();
        continue;
      }

      // An unknown or unhandled projection: bail out, so the caller skips this
      // write rather than wrongly reporting it.
      return std::nullopt;
    }
  }

  /// We end the walk once we have got to the root variable.
  std::optional<ResolvedRange> rootRange(Value root, int64_t selectedWidth) {
    auto rootWidth = getLvalueBitWidth(root);
    if (!rootWidth || offset > *rootWidth - selectedWidth)
      return std::nullopt;
    return ResolvedRange{root, offset, offset + selectedWidth};
  }

  /// Resolves a write whose longest static prefix ends at `parent` as a
  /// whole (i.e. a dynamic index/base was found one level down): the
  /// static prefix collapses to all of `parent`.
  std::optional<ResolvedRange> wholeParentRange(Value parent) {
    auto width = getLvalueBitWidth(parent);
    if (!width)
      return std::nullopt;
    // Any offsets added up below the dynamic select are not part of the
    // static prefix; hence we begin from the parent here.
    offset = 0;
    return walk(parent, *width);
  }

  /// Accumulates the offset contributed by a constant array index.
  LogicalResult stepArrayIndex(sv::ArrayIndexInOutOp idx) {
    Type parentType = getLvalueElementType(idx.getInput().getType());
    auto index = getConstantIndex(idx.getIndex());
    auto stride = getTypeBitWidth(sv::getAnyHWArrayElementType(parentType));
    auto parentWidth = getTypeBitWidth(parentType);
    if (!index || !stride || !parentWidth || *stride == 0 ||
        *index >= *parentWidth / *stride)
      return failure();

    return addOffset(*index, *stride);
  }

  /// Accumulates the offset contributed by a constant-base indexed
  /// part-select.
  LogicalResult stepPartSelect(sv::IndexedPartSelectInOutOp select) {
    Type parentType = getLvalueElementType(select.getInput().getType());
    auto base = getConstantIndex(select.getBase());
    auto parentWidth = getTypeBitWidth(parentType);
    int64_t width = select.getWidth();
    if (!base || !parentWidth || width <= 0)
      return failure();

    // The base and width count bits when selecting from an integer, but
    // elements when selecting from an array.
    int64_t unit = 1;
    int64_t extent = *parentWidth;
    if (Type elementType = sv::getAnyHWArrayElementType(parentType)) {
      auto elementWidth = getTypeBitWidth(elementType);
      if (!elementWidth || *elementWidth == 0)
        return failure();
      unit = *elementWidth;
      extent = *parentWidth / unit;
    }

    // `base -: width` selects [base - width + 1, base], so the lower
    // bound is base - (width - 1).
    int64_t low = *base;
    if (select.getDecrement())
      low -= width - 1;
    if (low < 0 || low > extent - width)
      return failure();

    return addOffset(low, unit);
  }

  /// Accumulates the offset contributed by a struct field select: the sum
  /// of the widths of all fields declared after this one.
  LogicalResult stepStructField(sv::StructFieldInOutOp field) {
    Type parentType =
        hw::getCanonicalType(getLvalueElementType(field.getInput().getType()));
    auto structType = dyn_cast<hw::StructType>(parentType);
    auto fieldIndex = structType
                          ? structType.getFieldIndex(field.getFieldAttr())
                          : std::nullopt;
    if (!fieldIndex)
      return failure();

    int64_t fieldOffset = 0;
    for (auto element : structType.getElements().drop_front(*fieldIndex + 1)) {
      auto width = getTypeBitWidth(element.type);
      auto sum = width ? llvm::checkedAdd(fieldOffset, *width) : std::nullopt;
      if (!sum)
        return failure();
      fieldOffset = *sum;
    }

    return addOffset(fieldOffset);
  }

  /// Adds `count * unit` to the running offset, failing on any overflow.
  LogicalResult addOffset(int64_t count, int64_t unit = 1) {
    auto next = llvm::checkedMulAdd<int64_t>(count, unit, offset);
    if (!next)
      return failure();
    offset = *next;
    return success();
  }

  int64_t offset = 0;
};

/// Sorts `drivers` by start offset and sweeps for overlaps, reporting an
/// error on the first conflicting driver whose range overlaps a prior
/// driver when at least one of the two is continuous.
static bool sweepForOverlaps(SmallVectorImpl<Driver> &drivers) {
  // We use llvm::stable_sort instead of llvm::sort to ensure that
  // drivers which happen to have identical ranges keep their program order
  // and so that the diagnostics emitted are deterministic too.
  llvm::stable_sort(drivers, [](const Driver &lhs, const Driver &rhs) {
    return std::tie(lhs.lo, lhs.hi) < std::tie(rhs.lo, rhs.hi);
  });

  bool emittedError = false;
  // The furthest-reaching driver seen so far in the sweep: one across all
  // drivers, one across continuous drivers only. Because `drivers` is sorted by
  // `lo`, a later driver whose `lo` falls before that `hi` overlaps it.
  const Driver *furthestAny = nullptr;
  const Driver *furthestContinuous = nullptr;

  for (const Driver &current : drivers) {
    const Driver *conflict = nullptr;
    if (current.kind == DriverKind::Continuous) {
      // A continuous driver conflicts with any driver that reaches into its
      // range.
      if (furthestAny && furthestAny->hi > current.lo)
        conflict = furthestAny;
    } else if (furthestContinuous && furthestContinuous->hi > current.lo) {
      // A procedural driver only conflicts with a continuous one: overlapping
      // procedural drivers are legal, as the last write wins.
      conflict = furthestContinuous;
    }

    if (conflict) {
      auto diag = current.op->emitError(
          "variable has multiple continuous and/or conflicting procedural "
          "drivers on overlapping bits");
      diag.attachNote(conflict->op->getLoc()) << "other driver is here";
      emittedError = true;
    }

    if (!furthestAny || current.hi > furthestAny->hi)
      furthestAny = &current;
    if (current.kind == DriverKind::Continuous &&
        (!furthestContinuous || current.hi > furthestContinuous->hi))
      furthestContinuous = &current;
  }

  return emittedError;
}

struct DriverInfo {
  Value dest;
  DriverKind kind;
};

/// Classifies `op` as a driver of a variable, or returns std::nullopt if it
/// does not write to one.
static std::optional<DriverInfo> classifyDriver(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<DriverInfo>>(op)
      // TODO: sv.assign will become the net assignment only, but it currently
      // acts as a stand-in for sv.var_assign while variables are still
      // declared with sv.reg/sv.logic. We only resolve SystemVerilog
      // variables here. We will remove this once variables are driven
      // exclusively through sv.var_assign for single driver cases.
      .Case<sv::VarAssignOp, sv::AssignOp>([](auto assign) {
        return DriverInfo{assign.getDest(), DriverKind::Continuous};
      })
      .Case<sv::BPAssignOp, sv::PAssignOp>([](auto assign) {
        return DriverInfo{assign.getDest(), DriverKind::Procedural};
      })
      .Default([](Operation *) { return std::nullopt; });
}

struct CheckVarDriversPass
    : public sv::impl::CheckVarDriversBase<CheckVarDriversPass> {
  void runOnOperation() override {
    llvm::MapVector<Value, SmallVector<Driver>> driversByRoot;

    getOperation().walk([&](Operation *op) {
      auto info = classifyDriver(op);
      if (!info)
        return;

      // Writes whose prefix cannot be resolved are skipped, and so are
      // zero-width ones: they drive no bits and so cannot conflict.
      auto range = StaticPrefixResolver::resolve(info->dest);
      if (!range || range->lo == range->hi)
        return;

      driversByRoot[range->root].push_back(
          {range->lo, range->hi, info->kind, op});
    });

    // Every variable is swept, so that all the conflicts get reported rather
    // than only those of the first variable that has one.
    bool emittedError = false;
    for (auto &entry : driversByRoot)
      emittedError |= sweepForOverlaps(entry.second);

    if (emittedError)
      signalPassFailure();
  }
};

} // namespace
