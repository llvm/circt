//===- KanagawaCallPrep.cpp - Implementation of call prep lowering --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"

#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWACALLPREP
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace circt;
using namespace kanagawa;

/// Build indexes to make lookups faster. Create the new argument types as well.
struct CallPrepPrecomputed {
  CallPrepPrecomputed(DesignOp);

  // Lookup a class from its symbol.
  DenseMap<hw::InnerRefAttr, ClassOp> classSymbols;

  // Mapping of method to argument type.
  DenseMap<hw::InnerRefAttr, std::pair<hw::StructType, Location>> argTypes;

  // Lookup the class to which a particular instance (in a particular class) is
  // referring.
  DenseMap<std::pair<ClassOp, StringAttr>, ClassOp> instanceMap;

  // Lookup an entry in instanceMap. If not found, return null.
  ClassOp lookupNext(ClassOp scope, StringAttr instSym) const {
    auto entry = instanceMap.find(std::make_pair(scope, instSym));
    if (entry == instanceMap.end())
      return {};
    return entry->second;
  }

  // Utility function to create a hw::InnerRefAttr to a method.
  static hw::InnerRefAttr getSymbol(MethodOp method) {
    auto design = method->getParentOfType<DesignOp>();
    return hw::InnerRefAttr::get(design.getSymNameAttr(),
                                 method.getInnerSym().getSymName());
  }
};

CallPrepPrecomputed::CallPrepPrecomputed(DesignOp design) {
  auto *ctxt = design.getContext();
  StringAttr modName = design.getNameAttr();

  // Populate the class-symbol lookup table.
  for (auto cls : design.getOps<ClassOp>())
    classSymbols[hw::InnerRefAttr::get(
        modName, cls.getInnerSymAttr().getSymName())] = cls;

  for (auto cls : design.getOps<ClassOp>()) {
    // Compute new argument types for each method.
    for (auto method : cls.getOps<MethodOp>()) {

      // Create the struct type.
      SmallVector<hw::StructType::FieldInfo> argFields;
      for (auto [argName, argType] :
           llvm::zip(method.getArgNamesAttr().getAsRange<StringAttr>(),
                     cast<MethodLikeOpInterface>(method.getOperation())
                         .getArgumentTypes()))
        argFields.push_back({argName, argType});
      auto argStruct = hw::StructType::get(ctxt, argFields);

      // Later we're gonna want the block locations, so compute a fused location
      // and store it.
      Location argLoc = UnknownLoc::get(ctxt);
      if (method->getNumRegions() > 0) {
        SmallVector<Location> argLocs;
        Block *body = &method.getBody().front();
        for (auto arg : body->getArguments())
          argLocs.push_back(arg.getLoc());
        argLoc = FusedLoc::get(ctxt, argLocs);
      }

      // Add both to the lookup table.
      argTypes.insert(
          std::make_pair(getSymbol(method), std::make_pair(argStruct, argLoc)));
    }

    // Populate the instances table.
    for (auto inst : cls.getOps<InstanceOp>()) {
      auto clsEntry = classSymbols.find(inst.getTargetNameAttr());
      assert(clsEntry != classSymbols.end() &&
             "class being instantiated doesn't exist");
      instanceMap[std::make_pair(cls, inst.getInnerSym().getSymName())] =
          clsEntry->second;
    }
  }
}

namespace {
/// For each CallOp, the corresponding method signature will have changed. Pack
/// all the operands into a struct.
struct MergeCallArgs : public OpConversionPattern<CallOp> {
  MergeCallArgs(MLIRContext *ctxt, const CallPrepPrecomputed &info)
      : OpConversionPattern(ctxt), info(info) {}

  LogicalResult
  matchAndRewrite(CallOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override final;

private:
  const CallPrepPrecomputed &info;
};
} // anonymous namespace

LogicalResult
MergeCallArgs::matchAndRewrite(CallOp call, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto loc = call.getLoc();
  rewriter.setInsertionPoint(call);
  auto method = call->getParentOfType<kanagawa::MethodLikeOpInterface>();

  // Use the 'info' accelerator structures to find the argument type.
  auto argStructEntry = info.argTypes.find(call.getCalleeAttr());
  assert(argStructEntry != info.argTypes.end() && "Method symref not found!");
  auto [argStruct, argLoc] = argStructEntry->second;

  // Pack all of the operands into it.
  auto newArg = hw::StructCreateOp::create(rewriter, loc, argStruct,
                                           adaptor.getOperands());
  newArg->setAttr("sv.namehint",
                  rewriter.getStringAttr(call.getCallee().getName().getValue() +
                                         "_args_called_from_" +
                                         method.getMethodName().getValue()));

  // Update the call to use just the new struct.
  rewriter.modifyOpInPlace(call, [&]() {
    call.getOperandsMutable().clear();
    call.getOperandsMutable().append(newArg.getResult());
  });

  return success();
}

namespace {
/// Change the method signatures to only have one argument: a struct capturing
/// all of the original arguments.
struct MergeMethodArgs : public OpConversionPattern<MethodOp> {
  MergeMethodArgs(MLIRContext *ctxt, const CallPrepPrecomputed &info)
      : OpConversionPattern(ctxt), info(info) {}

  LogicalResult
  matchAndRewrite(MethodOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final override;

private:
  const CallPrepPrecomputed &info;
};
} // anonymous namespace

LogicalResult
MergeMethodArgs::matchAndRewrite(MethodOp func, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = func.getLoc();
  auto *ctxt = getContext();

  // Find the pre-computed arg struct for this method.
  auto argStructEntry =
      info.argTypes.find(CallPrepPrecomputed::getSymbol(func));
  assert(argStructEntry != info.argTypes.end() && "Cannot find symref!");
  auto [argStruct, argLoc] = argStructEntry->second;

  // Create a new method with the new signature.
  FunctionType funcType = func.getFunctionType();
  FunctionType newFuncType =
      FunctionType::get(ctxt, {argStruct}, funcType.getResults());
  auto newArgNames = ArrayAttr::get(ctxt, {StringAttr::get(ctxt, "arg")});
  auto newMethod =
      MethodOp::create(rewriter, loc, func.getInnerSym(), newFuncType,
                       newArgNames, ArrayAttr(), ArrayAttr());

  if (func->getNumRegions() > 0) {
    // Create a body block with a struct explode to the arg struct into the
    // original arguments.
    Block *b =
        rewriter.createBlock(&newMethod.getRegion(), {}, {argStruct}, {argLoc});
    rewriter.setInsertionPointToStart(b);
    auto replacementArgs =
        hw::StructExplodeOp::create(rewriter, loc, b->getArgument(0));

    // Merge the original method body, rewiring the args.
    Block *funcBody = &func.getBody().front();
    rewriter.mergeBlocks(funcBody, b, replacementArgs.getResults());
  }

  rewriter.eraseOp(func);

  return success();
}

namespace {
/// Run all the physical lowerings.
struct CallPrepPass
    : public circt::kanagawa::impl::KanagawaCallPrepBase<CallPrepPass> {
  void runOnOperation() override;

private:
  // Merge the arguments into one struct.
  LogicalResult merge(const CallPrepPrecomputed &);
};
} // anonymous namespace

void CallPrepPass::runOnOperation() {
  CallPrepPrecomputed info(getOperation());

  if (failed(merge(info))) {
    signalPassFailure();
    return;
  }
}

LogicalResult CallPrepPass::merge(const CallPrepPrecomputed &info) {
  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addDynamicallyLegalOp<CallOp>([](CallOp call) {
    auto argValues = call.getArgOperands();
    return argValues.size() == 1 &&
           hw::type_isa<hw::StructType>(argValues.front().getType());
  });
  target.addDynamicallyLegalOp<MethodOp>([](MethodOp func) {
    ArrayRef<Type> argTypes = func.getFunctionType().getInputs();
    return argTypes.size() == 1 &&
           hw::type_isa<hw::StructType>(argTypes.front());
  });

  // Add patterns to merge the args on both the call and method sides.
  RewritePatternSet patterns(&getContext());
  patterns.insert<MergeCallArgs>(&getContext(), info);
  patterns.insert<MergeMethodArgs>(&getContext(), info);

  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

std::unique_ptr<Pass> circt::kanagawa::createCallPrepPass() {
  return std::make_unique<CallPrepPass>();
}
