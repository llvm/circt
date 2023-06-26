//===- HWLoopRerolling.cpp - HW Memory Implementation Pass
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts generated FIRRTL memory modules to
// simulation models.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/Namespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Support/LoopRerolling.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// HWLoopRerollingPass Pass
//===----------------------------------------------------------------------===//

struct HWLoopRerollingPass
    : public sv::HWLoopRerollingBase<HWLoopRerollingPass> {
  void runOnOperation() override;
};

void HWLoopRerollingPass::runOnOperation() {
  auto topModule = getOperation();
  auto exts = topModule.getOps<comb::ExtractOp>();
  for (auto e : exts) {
    if (e.getType().isInteger(1)) {
      OpBuilder builder(e);
      auto width = e.getInput().getType().getIntOrFloatBitWidth();
      auto c = builder.create<hw::ConstantOp>(
          e.getLoc(),
          APInt(std::max(1u, llvm::Log2_64_Ceil(width)), e.getLowBit()));
      auto in = builder.create<hw::BitcastOp>(
          e.getLoc(), hw::ArrayType::get(e.getType(), width), e.getInput());
      Value arrayGet = builder.create<hw::ArrayGetOp>(e.getLoc(), in, c);
      e.replaceAllUsesWith(arrayGet);
    }
  }
  //for (auto e : exts) {
  //  if (e.getType().isInteger(1)) {
  //  OpBuilder builder(e);
  //  auto width = e.getInput().getType().getIntOrFloatBitWidth();
  //  auto c = builder.create<hw::ConstantOp>(
  //      e.getLoc(),
  //      APInt(std::max(1u, llvm::Log2_64_Ceil(width)), e.getLowBit()));
  //  auto intype = hw::ArrayType::get(builder.getIntegerType(1), width);
  //  auto outype = hw::ArrayType::get(builder.getIntegerType(1),
  //                                   e.getType().getIntOrFloatBitWidth());
  //  auto in = builder.create<hw::BitcastOp>(
  //      e.getLoc(), intype, e.getInput());
  //  Value arrayGet =
  //      builder.create<hw::ArraySliceOp>(e.getLoc(), outype, in, c);
  //  arrayGet = builder.create<hw::BitcastOp>(e.getLoc(), e.getType(), arrayGet);
  //  e.replaceAllUsesWith(arrayGet);
  //  }
  //}

  for (auto &op :
       llvm::make_early_inc_range(topModule.getBodyBlock()->getOperations())) {
    if (!isa<comb::OrOp, comb::AndOp, comb::XorOp, comb::ConcatOp>(&op))
      continue;
    auto type = op.getResult(0).getType();
    if (isa<comb::ConcatOp>(&op)) {
      if (!llvm::all_of(op.getOperands(), [&](Value v) {
            return v.getType() == op.getOperand(0).getType();
          }))
        continue;
      type = op.getOperand(0).getType();
    }
    auto loc = op.getLoc();
    unsigned size = op.getNumOperands();
    // If the loop is small enough.
    if (size <= 8)
      continue;

    ImplicitLocOpBuilder builder(loc, &op);
    // Unpacked array.
    mlir::Type wireTy = hw::UnpackedArrayType::get(type, size);
    if (isa<comb::ConcatOp>(&op))
      wireTy = hw::ArrayType::get(type, size);
    auto wire = builder.create<sv::LogicOp>(wireTy);

    unsigned start = 0;

    unsigned failure = 0;
    unsigned count = 0;
    auto fn = [&](unsigned idx) {
      // Assign.
      ++failure;
      builder.create<sv::BPAssignOp>(
          builder.create<sv::ArrayIndexInOutOp>(
              wire, builder.create<hw::ConstantOp>(
                        APInt(llvm::Log2_64_Ceil(size), idx))),
          op.getOperand(idx));
    };

    auto always_comb = builder.create<sv::AlwaysCombOp>(loc);
    // PR ONLY: Remove
    std::string t = "SV_FOR_LOOP_";
    t += op.getName().getStringRef();
    circt::sv::setSVAttributes(
        always_comb, sv::SVAttributeAttr::get(builder.getContext(), t,
                                              /*emitAsComment=*/true));

    while (start < size) {
      builder.setInsertionPointToEnd(always_comb.getBodyBlock());
      if (start == size - 1) {
        fn(start);
        break;
      }
      circt::LoopReroller reroller(builder, 1024);
      auto value = op.getOperand(start);
      auto next = op.getOperand(start + 1);
      if (failed(reroller.unifyTwoValues(value, next)) || reroller.getTermSize() < 3) {
        fn(start++);
        continue;
      }
      count++;
      // Ok, at least we can create for-loop for the range [start, start+1].
      // Let's try extending it.
      unsigned end = start + 2;
      for (; end < size; end++) {
        if (failed(reroller.unifyIntoTemplate(op.getOperand(end))))
          break;
      }
      // Create a loop [start, end).
      // FIXME: looplength must be end
      unsigned loopLength = end - start;
      auto lb = builder.create<hw::ConstantOp>(
          APInt(llvm::Log2_64_Ceil(loopLength + 1), 0));
      auto ub = builder.create<hw::ConstantOp>(
          APInt(llvm::Log2_64_Ceil(loopLength + 1), loopLength));
      auto c = builder.create<hw::ConstantOp>(
          APInt(llvm::Log2_64_Ceil(loopLength + 1), 1));
      builder.create<sv::ForOp>(lb, ub, c, "i", [&](BlockArgument arg) {
        Value iterValue = arg;
        auto *block = builder.getBlock();
        // Insert all operations in the sandbox
        block->getOperations().splice(block->begin(), reroller.getOperations());
        builder.setInsertionPointToStart(block);
        if (!iterValue.getType().isInteger(llvm::Log2_64_Ceil(loopLength))) {
          iterValue = builder.create<comb::ExtractOp>(
              iterValue, 0, llvm::Log2_64_Ceil(loopLength));
        }
        SmallVector<Value> operands;
        for (int i = end - 1; i + 1 != start; --i) {
          operands.push_back(builder.create<hw::ConstantOp>(
              APInt(llvm::Log2_64_Ceil(size), i)));
        }

        auto templateVal = reroller.getTemplateValue();
        for (auto [value, operands] :
             llvm::make_early_inc_range(reroller.getDummyValues())) {
          assert(operands.size() >= loopLength);
          // It's possible that the size is differnt;
          operands.resize(loopLength);
          // Make sure to reverse.
          std::reverse(operands.begin(), operands.end());

          builder.setInsertionPointAfterValue(value);
          auto arrayCreate = builder.create<hw::ArrayCreateOp>(operands);
          auto arrayGet =
              builder.create<hw::ArrayGetOp>(arrayCreate, iterValue);
          if (templateVal == value)
            templateVal = arrayGet;
          value.replaceAllUsesWith(arrayGet);
          value.getDefiningOp()->erase();
        }

        builder.setInsertionPointToEnd(block);
        Value idx = builder.create<hw::ArrayCreateOp>(operands);
        idx = builder.create<hw::ArrayGetOp>(idx, iterValue);
        builder.create<sv::BPAssignOp>(
            builder.create<sv::ArrayIndexInOutOp>(wire, idx), templateVal);
      });
      start = end;
    }
    if ((failure + count) * 2 > size) {
      always_comb.erase();
      wire.erase();
      continue;
    }
    builder.setInsertionPointAfterValue(wire);
    auto w = builder.create<sv::ReadInOutOp>(wire);
    Value result;
    if (isa<comb::OrOp>(op)) {
      result =
          builder.create<sv::UnpackedArrayOrOp>(op.getResult(0).getType(), w);
      ++numOrOpsRerolled;
    }
    if (isa<comb::XorOp>(op)) {
      result =
          builder.create<sv::UnpackedArrayXOrOp>(op.getResult(0).getType(), w);
      ++numXorOpsRerolled;
    }
    if (isa<comb::AndOp>(op)) {
      result =
          builder.create<sv::UnpackedArrayAndOp>(op.getResult(0).getType(), w);
      ++numAndOpsRerolled;
    }
    if (isa<comb::ConcatOp>(op)) {
      result = builder.create<hw::BitcastOp>(op.getResult(0).getType(), w);

      ++numConcatOpsRerolled;
    }

    op.getResult(0).replaceAllUsesWith(result);
    op.erase();
  }
}

std::unique_ptr<Pass> circt::sv::createHWLoopRerollingPass() {
  auto pass = std::make_unique<HWLoopRerollingPass>();
  return pass;
}
