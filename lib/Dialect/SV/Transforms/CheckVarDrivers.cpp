//===- CheckVarDrivers.cpp - Check SystemVerilog variable drivers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass checks SystemVerilog Clause 6.5 / 10.3.2: it is illegal for a
// variable to have more than one continuous driver, or to be driven by both
// a continuous assignment and a procedural statement, on overlapping bits.
// Each write's "longest static prefix" (Clause 11.5.3) is resolved and
// flattened to a half-open bit interval [lo, hi) on the root variable; two
// drivers conflict if their intervals overlap and at least one is
// continuous. Procedural-vs-procedural overlap is always legal (the last
// write wins).
//
// NOTE: sv.bpassign/sv.passign and the inout projection ops currently only
// accept !hw.inout, so variables are checked through their legacy
// declarations (sv.reg/sv.logic) and continuous drivers through sv.assign,
// alongside sv.var/sv.var_assign. Nets (sv.wire) are never checked. Once the
// var/net migration lands, drop the legacy handling (see the TODOs below) and
// convert the tests to sv.var/sv.var_assign.
//
// Package variables, ref ports/aliasing, the always_comb/always_ff
// single-writer-process rule, and ifdef/generate guard filtering are out of
// scope for this pass; see the SV var/net RFC.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
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

/// A single write resolved to a half-open bit range [lo, hi) on some root
/// variable, with the operation that produced it (for diagnostics).
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

/// Returns the element type of an lvalue-producing type (!sv.var<T> or
/// !hw.inout<T>), or a null Type if `type` is neither.
static Type getLvalueElementType(Type type) {
  if (auto var = dyn_cast<sv::VarType>(type))
    return var.getElementType();
  if (auto inout = dyn_cast<hw::InOutType>(type))
    return inout.getElementType();
  return {};
}

/// Returns true if `value` is the result of an op that declares a SystemVerilog
/// variable, i.e. something that may have at most one continuous driver per
/// bit. sv.wire is deliberately excluded: it is a net, which may legally have
/// multiple drivers.
///
/// TODO: sv.reg and sv.logic are legacy !hw.inout declarations. Drop them once
/// variables are represented solely by sv.var.
static bool isVariableRoot(Value value) {
  return isa_and_nonnull<sv::VarOp, sv::RegOp, sv::LogicOp>(
      value.getDefiningOp());
}

/// Matches `value` against a constant small enough to use as
/// an array index or part-select base. Returns std::nullopt if `value` is
/// not a resolvable constant (e.g. a dynamic index), in which case the
/// caller must treat the enclosing select as covering its whole parent.
static std::optional<uint64_t> getConstantIndex(Value value) {
  APInt constant;
  // Indices are unsigned: the top bit of e.g. `1 : i1` is a value bit, not a
  // sign. Out-of-range values are rejected by the callers' bounds checks.
  if (!mlir::matchPattern(value, mlir::m_ConstantInt(&constant)) ||
      constant.getActiveBits() > 64)
    return std::nullopt;
  return constant.getZExtValue();
}

/// Resolves the longest static prefix of an lvalue, per SV Clause 11.5.3:
/// walks back through projection ops (array index, indexed part-select,
/// struct field) to the root sv.var, accumulating a bit offset. Stops and
/// returns the *whole* enclosing aggregate's range as soon as a dynamic
/// (non-constant) index/base is found, since the spec defines the static
/// prefix as ending there. Returns std::nullopt if the chain doesn't bottom
/// out at a known-width sv.var (e.g. an unhandled projection kind, or a
/// type whose width can't be determined) -- callers must treat that as "no
/// information," not as an error, so an unresolvable write is simply
/// skipped rather than reported.
class StaticPrefixResolver {
public:
  static std::optional<ResolvedRange> resolve(Value dest) {
    Type elementType = getLvalueElementType(dest.getType());
    int64_t width = elementType ? hw::getBitWidth(elementType) : -1;
    if (width < 0)
      return std::nullopt;
    return StaticPrefixResolver().walk(dest, width);
  }

private:
  /// Walks back from `value` (whose own selected width is `selectedWidth`)
  /// to the root variable, accumulating `offset` along the way.
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

      // Unknown or unhandled projection: bail out without an opinion, so
      // the caller skips this write rather than mis-reporting it.
      return std::nullopt;
    }
  }

  /// Finishes the walk once the root variable is reached.
  std::optional<ResolvedRange> rootRange(Value root, int64_t selectedWidth) {
    Type rootType = getLvalueElementType(root.getType());
    int64_t rootWidth = rootType ? hw::getBitWidth(rootType) : -1;
    if (rootWidth < 0 || offset > rootWidth - selectedWidth)
      return std::nullopt;
    return ResolvedRange{root, offset, offset + selectedWidth};
  }

  /// Resolves a write whose longest static prefix ends at `parent` as a
  /// whole (i.e. a dynamic index/base was found one level down): the
  /// static prefix collapses to all of `parent`.
  std::optional<ResolvedRange> wholeParentRange(Value parent) {
    Type parentType = getLvalueElementType(parent.getType());
    int64_t width = parentType ? hw::getBitWidth(parentType) : -1;
    if (width < 0)
      return std::nullopt;
    // Offsets accumulated below the dynamic select are no longer part of the
    // static prefix; restart from the parent.
    offset = 0;
    return walk(parent, width);
  }

  /// Accumulates the offset contributed by a constant array index. Returns
  /// failure if the index is dynamic (caller falls back to the whole
  /// array) or if the offset can't be computed/would overflow.
  LogicalResult stepArrayIndex(sv::ArrayIndexInOutOp idx) {
    auto constantIdx = getConstantIndex(idx.getIndex());
    if (!constantIdx)
      return failure();

    Type parentType = getLvalueElementType(idx.getInput().getType());
    Type elementType = sv::getAnyHWArrayElementType(parentType);
    int64_t stride = elementType ? hw::getBitWidth(elementType) : -1;
    int64_t parentWidth = parentType ? hw::getBitWidth(parentType) : -1;
    if (stride <= 0 || parentWidth < 0 ||
        *constantIdx >= static_cast<uint64_t>(parentWidth / stride))
      return failure();

    return addOffset(*constantIdx * static_cast<uint64_t>(stride));
  }

  /// Accumulates the offset contributed by a constant-base indexed
  /// part-select (`base +: width` or `base -: width`). Returns failure if
  /// the base is dynamic (caller falls back to the whole parent) or if the
  /// range is out of bounds/would overflow.
  LogicalResult stepPartSelect(sv::IndexedPartSelectInOutOp select) {
    auto base = getConstantIndex(select.getBase());
    if (!base)
      return failure();

    Type parentType = getLvalueElementType(select.getInput().getType());
    int64_t parentWidth = parentType ? hw::getBitWidth(parentType) : -1;
    int64_t width = select.getWidth();
    if (parentWidth < 0 || width <= 0)
      return failure();

    uint64_t localOffset = *base;
    if (select.getDecrement()) {
      // `base -: width` selects [base - width + 1, base], so the lower
      // bound is base - (width - 1).
      if (*base + 1 < static_cast<uint64_t>(width))
        return failure();
      localOffset -= static_cast<uint64_t>(width - 1);
    }

    if (localOffset > static_cast<uint64_t>(parentWidth - width))
      return failure();

    return addOffset(localOffset);
  }

  /// Accumulates the offset contributed by a struct field select: the sum
  /// of the widths of all fields declared after this one (fields are laid
  /// out MSB-first, matching hw::StructType's element order).
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
      int64_t width = hw::getBitWidth(element.type);
      if (width < 0 || fieldOffset > INT64_MAX - width)
        return failure();
      fieldOffset += width;
    }

    return addOffset(static_cast<uint64_t>(fieldOffset));
  }

  /// Adds `delta` to the running offset, failing on overflow.
  LogicalResult addOffset(uint64_t delta) {
    if (delta >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - offset))
      return failure();
    offset += static_cast<int64_t>(delta);
    return success();
  }

  int64_t offset = 0;
};

//===----------------------------------------------------------------------===//
// Overlap sweep
//===----------------------------------------------------------------------===//

/// Sorts `drivers` by start offset and sweeps for overlaps, reporting an
/// error on the first conflicting driver whose range overlaps a prior
/// driver when at least one of the two is continuous. Returns true if any
/// error was reported.
static bool sweepForOverlaps(SmallVectorImpl<Driver> &drivers) {
  // Stable, so drivers with identical ranges keep their program order and the
  // diagnostics are deterministic.
  llvm::stable_sort(drivers, [](const Driver &lhs, const Driver &rhs) {
    return std::tie(lhs.lo, lhs.hi) < std::tie(rhs.lo, rhs.hi);
  });

  bool emittedError = false;
  // Watermarks tracking the furthest-reaching driver seen so far in the
  // sweep: one across all drivers, one across continuous drivers only.
  // Because `drivers` is sorted by `lo`, any later driver whose `lo` falls
  // before a watermark's `hi` overlaps that watermark's driver.
  const Driver *furthestAny = nullptr;
  const Driver *furthestContinuous = nullptr;

  for (const Driver &current : drivers) {
    const Driver *conflict = nullptr;
    if (current.kind == DriverKind::Continuous) {
      // A continuous driver conflicts with anything (continuous or
      // procedural) that reaches into its range.
      if (furthestAny && furthestAny->hi > current.lo)
        conflict = furthestAny;
    } else if (furthestContinuous && furthestContinuous->hi > current.lo) {
      // A procedural driver only conflicts with a continuous one;
      // procedural-vs-procedural overlap is always legal.
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

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CheckVarDriversPass
    : public sv::impl::CheckVarDriversBase<CheckVarDriversPass> {
  void runOnOperation() override {
    llvm::MapVector<Value, SmallVector<Driver>> driversByRoot;

    auto recordDriver = [&](Operation *op, Value dest, DriverKind kind) {
      auto range = StaticPrefixResolver::resolve(dest);
      if (!range)
        return;
      driversByRoot[range->root].push_back({range->lo, range->hi, kind, op});
    };

    getOperation().walk([&](Operation *op) {
      if (auto assign = dyn_cast<sv::VarAssignOp>(op)) {
        recordDriver(assign, assign.getDest(), DriverKind::Continuous);
        return;
      }

      // TODO: sv.assign is the net assignment and is only checked here as a
      // stand-in for sv.var_assign while variables are still declared with
      // sv.reg/sv.logic. An assign whose root is not a variable (e.g. an
      // sv.wire) never resolves, so nets are skipped. Remove once variables
      // are driven exclusively through sv.var_assign.
      if (auto assign = dyn_cast<sv::AssignOp>(op)) {
        recordDriver(assign, assign.getDest(), DriverKind::Continuous);
        return;
      }

      if (auto assign = dyn_cast<sv::BPAssignOp>(op)) {
        recordDriver(assign, assign.getDest(), DriverKind::Procedural);
        return;
      }
      if (auto assign = dyn_cast<sv::PAssignOp>(op)) {
        recordDriver(assign, assign.getDest(), DriverKind::Procedural);
        return;
      }
    });

    bool emittedError = false;
    for (auto &entry : driversByRoot)
      emittedError |= sweepForOverlaps(entry.second);

    if (emittedError)
      signalPassFailure();
  }
};

} // namespace
