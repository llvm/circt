//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/UnusedOpPruner.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_EMBEDVALIDATIONVALUESPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "rtg-embed-validation-values"

//===----------------------------------------------------------------------===//
// Embed Validation Values Pass
//===----------------------------------------------------------------------===//

namespace {
struct EmbedValidationValuesPass
    : public rtg::impl::EmbedValidationValuesPassBase<
          EmbedValidationValuesPass> {
  using Base::Base;

  void runOnOperation() override;
  LogicalResult parseFile(DenseMap<StringAttr, rtg::ValidateOp> &opById,
                          DenseMap<StringAttr, TypedAttr> &valueMap);
};
} // namespace

LogicalResult EmbedValidationValuesPass::parseFile(
    DenseMap<StringAttr, rtg::ValidateOp> &opById,
    DenseMap<StringAttr, TypedAttr> &valueMap) {
  auto buff = llvm::MemoryBuffer::getFile(filename, /*IsText=*/true);
  if (auto ec = buff.getError())
    return emitError(UnknownLoc::get(&getContext()))
           << "cannot open file '" << filename << "': " << ec.message();

  for (llvm::line_iterator i(**buff); !i.is_at_eof(); ++i) {
    auto *ctxt = &getContext();
    auto [id, value] = i->split('=');
    Location idLoc = FileLineColLoc::get(ctxt, filename, i.line_number(), 1);
    Location valueLoc =
        FileLineColLoc::get(ctxt, filename, i.line_number(), id.size() + 1);
    if (value.empty())
      return emitError(valueLoc) << "no value for ID '" << id << "'";

    auto idAttr = StringAttr::get(ctxt, id);

    auto op = opById[idAttr];
    if (!op)
      continue;

    auto valueAttr =
        op.getRef().getType().parseContentValue(value, op.getType());
    if (!valueAttr)
      return emitError(valueLoc)
             << "cannot parse value of type " << op.getType()
             << " from string '" << value << "'";

    if (!valueMap.insert({idAttr, valueAttr}).second)
      return emitError(idLoc) << "duplicate ID in input file: " << idAttr;

    LLVM_DEBUG(llvm::dbgs() << "- Parsed value for " << idAttr << "\n");
  }

  return success();
}

void EmbedValidationValuesPass::runOnOperation() {
  DenseMap<StringAttr, TypedAttr> valueMap;
  DenseMap<StringAttr, rtg::ValidateOp> opById;
  SmallVector<rtg::ValidateOp> validateOps;

  auto result = getOperation().walk([&](rtg::ValidateOp op) -> WalkResult {
    if (op.getIdAttr()) {
      validateOps.push_back(op);
      if (!opById.insert({op.getIdAttr(), op}).second)
        return op->emitError("at least two validate ops have the same ID: ")
               << op.getIdAttr();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted() || failed(parseFile(opById, valueMap)))
    return signalPassFailure();

  UnusedOpPruner pruner;
  for (auto op : validateOps) {
    auto value = valueMap[op.getIdAttr()];
    // If the input file did not contain a value for this ID, keep the validate
    // operation as-is.
    if (!value)
      continue;

    OpBuilder builder(op);
    auto *constOp = value.getDialect().materializeConstant(
        builder, value, value.getType(), op.getLoc());
    if (!constOp) {
      op.emitOpError("materializer of dialect '")
          << value.getDialect().getNamespace()
          << "' unable to materialize value for attribute '" << value << "'";
      return signalPassFailure();
    }

    op.getValue().replaceAllUsesWith(constOp->getResult(0));
    pruner.eraseNow(op);
  }

  pruner.eraseNow();
}
