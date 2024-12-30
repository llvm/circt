//===- MergeExtractRef.cpp - Merge moore.extract_ref ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MergeExtractRef pass.
// It's used to merge moore.extract_ref together in the big-endian order without
// existing in if/for statements and nested moore.extract_ref.
// For example:
// ```
// bit [31:0] arr;
// assign arr[23:16] = 8'd5;
// always_comb begin
// arr[15:8] = 8'd10;
// end
// ```
// After running this pass:
// ```
// bit [31:0] arr;
// ...
// always_comb begin
// ...
// %concat = {arr[31:24], 8'd5, 8'd10, arr[7:0]} // concat
// bit [31:0] arr = %concat
// arr[15:8] = 8'd10;
// end
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_MERGEEXTRACTREF
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;
using namespace mlir;

namespace {
// A helper struct to capture the lowBit of moore.extract_ref and the value
// assigned for it.
struct IndexedSrc {
  IndexedSrc(uint64_t i, Value s) : index(i), src(s) {}

  uint64_t index; // lowBit
  Value src;      // like 8'd5
};

// A helper function to capture all moore.extract_ref, however, except the
// nested moore.extract_ref.
static void
collectExtractRef(ExtractRefOp &extrRef, Value src,
                  DenseMap<Value, SmallVector<IndexedSrc>> &extractRefs) {
  if (auto refOp = extrRef.getInput().getDefiningOp<ExtractRefOp>()) {
    emitError(extrRef->getLoc()) << "Unsupported nested extract_ref op";
    return;
  }
  auto index =
      *extrRef.getLowBit().getDefiningOp<ConstantOp>().getValue().getRawData();
  extractRefs[extrRef.getInput()].emplace_back(index, src);
}

// A helper function to call `collectExtractRef()`, if want to merge
// moore.extract_ref, must to hanlde the moore.*assign.
static void run(Operation *op,
                DenseMap<Value, SmallVector<IndexedSrc>> &extractRefs) {
  TypeSwitch<Operation *, void>(op)
      .Case<ContinuousAssignOp, BlockingAssignOp, NonBlockingAssignOp>(
          [&](auto op) {
            if (auto extrRef =
                    op.getDst().template getDefiningOp<ExtractRefOp>()) {
              collectExtractRef(extrRef, op.getSrc(), extractRefs);
            }
          });
}

struct MergeExtractRefPass
    : public circt::moore::impl::MergeExtractRefBase<MergeExtractRefPass> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createMergeExtractRefPass() {
  return std::make_unique<MergeExtractRefPass>();
}

void MergeExtractRefPass::runOnOperation() {
  mlir::OpBuilder builder(&getContext());

  // Used to collect the input and lowBit of moore.extract_ref and its value.
  // Like <arr, <{(16, 8'd5), (8, 8'd10)}>>
  DenseMap<Value, SmallVector<IndexedSrc>> extractRefs;
  // Walk "Operation * op" can handle the body of procedure.
  getOperation()->walk([&](Operation *op) { run(op, extractRefs); });

  for (auto &[extrRef, indexedSrcs] : extractRefs) {
    auto name = extrRef.getDefiningOp()->getAttrOfType<StringAttr>("name");
    auto type = extrRef.getType();

    // Sort all moore.extract_ref in the big endian order.
    bool success = true;
    llvm::sort(indexedSrcs, [&](const IndexedSrc &a, const IndexedSrc &b) {
      if (a.index == b.index)
        return success = false;
      return a.index > b.index;
    });

    if (!success) {
      emitError(extrRef.getLoc())
          << "Assign a value to the same slice unsupported";
      return;
    }

    auto lo = indexedSrcs.back().index;
    auto hi = indexedSrcs.front().index +
              cast<UnpackedType>(indexedSrcs.front().src.getType())
                  .getBitSize()
                  .value();

    auto width = cast<RefType>(type).getBitSize().value();
    auto domain = cast<RefType>(type).getDomain();
    auto loc = extrRef.getLoc();

    // Collect and then concat all values together for the same variable.
    SmallVector<Value> values;
    for (const auto &indexedSrc : indexedSrcs)
      values.push_back(indexedSrc.src);

    builder.setInsertionPointAfterValue(indexedSrcs.back().src);
    Value concat = builder.create<ConcatOp>(loc, values);

    if (cast<IntType>(concat.getType()).getWidth() == width) {
      ; // Here directly goto the bottom to create moore.variable with a value.
    } else if ((lo != uint64_t(0)) || (hi != uint64_t(width))) {
      Value loExtract, hiExtract;
      // Like above mentioned "arr[7:0]";
      if (lo != uint64_t(0)) {
        auto resultType = IntType::get(&getContext(), lo, domain);
        Value lowBit = builder.create<ConstantOp>(loc, resultType, 0);
        auto read = builder.create<ReadOp>(
            loc, cast<RefType>(type).getNestedType(), extrRef);
        loExtract = builder.create<ExtractOp>(loc, resultType, read, lowBit);
      }
      // Like above mentioned "arr[31:24]";
      if (hi != uint64_t(width)) {
        auto resultType = IntType::get(&getContext(), width - hi, domain);
        Value lowBit = builder.create<ConstantOp>(loc, resultType, hi);
        auto read = builder.create<ReadOp>(
            loc, cast<RefType>(type).getNestedType(), extrRef);
        hiExtract = builder.create<ExtractOp>(loc, resultType, read, lowBit);
      }

      if (loExtract && hiExtract) {
        concat = builder.create<ConcatOp>(
            loc, ValueRange{hiExtract, concat, loExtract});
      } else
        concat =
            loExtract
                ? builder.create<ConcatOp>(loc, ValueRange{concat, loExtract})
                : builder.create<ConcatOp>(loc, ValueRange{hiExtract, concat});

    } else {
      // TODO:
      // Like 4-bits arr, arr[3:2] = 2'b0; arr[0] = 1'b0;. "arr[1]" don't be
      // assigned a value, this situation belongs to missing some middle bits.
      emitError(loc)
          << "Unsupported the complex situations of missing some middle bits ";
      return;
    }
    builder.create<VariableOp>(loc, type, name, concat);
  }
}
