//===- DCTestSCFToDC.cpp - SCF to DC test pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This pass tests the SCF to DC conversion patterns by defining a simple
//  func.func-based conversion pass. It should not be used for anything but
//  testing the conversion patterns, given its lack of handling anything but
//  SCF ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"

#include "mlir/Dialect/SCF/SCFOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace dc;
using namespace mlir;

namespace circt {
namespace dc {

class ControlFlowConverter;

// A ControlFlowConversionPattern represents the lowering of a control flow
// construct to DC.
class ControlFlowConversionPatternBase {
public:
  ControlFlowConversionPatternBase(ControlFlowConverter &converter)
      : converter(converter), b(converter.b) {}

  // Return true if this pattern matches the given operation.
  bool matches(Operation *op) const = 0;

protected:
  ControlFlowConverter &converter;
  OpBuilder &b;
};

template <typename TOp>
class ControlFlowConversionPattern : public ControlFlowConversionPatternBase {
public:
  using OpTy = TOp;
  // Convert the registered operation to DC.
  // The 'control' represents the incoming !dc.token-typed control value that
  // hands off control to this control operator.
  // The conversion is expected to return a !dc.value-typed value that
  // represents the outgoing control value that hands off control away from
  // this operation after it has been executed.
  virtual FailureOr<Value> convert(Value control, TOp op) = 0;
  bool matches(Operation *op) const override { return isa<TOp>(op); }
}

// The ControlFlowConverter is a container of ControlFlowConvertionPatterns and
// is what drives a conversion from control flow ops to DC.
class ControlFlowConverter {
public:
  // Run conversion on the given region.
  LogicalResult go(Region &region);

  OpBuilder &b;

  template <typename TConverter>
  void add() {
    static_assert(
        std::is_base_of<ControlFlowConversionPatternBase, TConverter>::value,
        "TConverter must be a subclass of ControlFlowConversionPatternBase");
    auto &converter =
        converters.emplace_back(std::make_unique<TConverter>(*this));
    converterLookup[TConverter::OpTy::getOperationName()] = converter.get();
  }

protected:
  // Converts a region to DC using the registered patterns.
  virtual LogicalResult convert(Region &region) = 0;

  // An analysis which determines which SSA values used within a region that
  // are defined outside of said region (i.e. a value is referenced via.
  // dominance).
  DominanceValueUsages valueUsage;

  llvm::SmallVector<std::unique_ptr<ControlFlowConversionPatternBase>>
      converters;
  llvm::DenseMap<llvm::StringLiteral, &ControlFlowConversionPatternBase>
      converterLookup;
};

LogicalResult ControlFlowConverter::go(Region &region) {
  // initialization...
  return convert(region);
}

class TestControlFlowConverter : public ControlFlowConverter {
public:
protected:
  LogicalResult convert(Region &region) override;
};

LogicalResult TestControlFlowConverter::convert(Region &region) {
  if (!region.hasOneBlock())
    assert(false && "Only single-block regions are supported");

  for (auto op : llvm::make_early_inc_range(region.front())) {
    auto it = converterLookup.find(op.getName());
    if (it == converterLookup.end())
      assert(false && "No converter found for op");

    auto &converter = it->second;
  }
}

class IfOpConversionPattern : public ControlFlowConversionPattern<scf::IfOp> {
public:
  FailureOr<Value> convert(mlir::Value control, scf::IfOp op) override {
    auto thenBranchRes = converter.convert(op.getThenRegion());
    if (!op.hasElseRegion())
      return thenBranchRes;

    // Have to merge the true and false branches to generate the output control
    // token.
    std::optional<FailureOr<Value>> elseBranchRes;
    elseBranchRes = converter.convert(op.getElseRegion());

    if (failed(thenBranchRes) || failed(*elseBranchRes))
      return failure();

    return b.create<dc::MergeOp>(op.getLoc(), *thenBranch, *elseBranchRes)
        .getResult();
  }
};

} // namespace dc
} // namespace circt

namespace {
struct DCTestSCFToDCPass : public DCTestSCFToDCBase<DCTestSCFToDCPass> {
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    for (auto func : llvm::make_early_inc_range(mod.getOps<mlir::FuncOp>())) {
      // In this test pass, we converter all arguments of a func.func to
      // !dc.value's.

      ControlFlowConversion conversion;
      conversion.add<IfOpConverter>();
      if (failed(conversion.convert()))
        return signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<mlir::Pass> circt::dc::createDCTestSCFToDCPass() {
  return std::make_unique<DCTestSCFToDCPass>();
}
