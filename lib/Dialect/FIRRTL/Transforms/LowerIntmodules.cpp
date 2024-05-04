//===- LowerIntmodules.cpp - Lower intmodules to ops ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntmodules pass.  This pass processes
// FIRRTL intmodules and replaces all instances with generic intrinsic ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntmodulesPass : public LowerIntmodulesBase<LowerIntmodulesPass> {
  void runOnOperation() override;
  using LowerIntmodulesBase::fixupEICGWrapper;
};
} // namespace

static LogicalResult checkModForAnnotations(FModuleLike mod, StringRef name) {
  if (!AnnotationSet(mod).empty())
    return mod.emitError(name)
           << " cannot have annotations since it is an intrinsic";
  return success();
}

static LogicalResult checkInstForAnnotations(FInstanceLike inst,
                                             StringRef name) {
  if (!AnnotationSet(inst).empty())
    return inst.emitError(name)
           << " instance cannot have annotations since it is an intrinsic";
  return success();
}

// This is the main entrypoint for the conversion pass.
void LowerIntmodulesPass::runOnOperation() {
  auto &ig = getAnalysis<InstanceGraph>();

  bool changed = false;

  // Convert to int ops.
  for (auto op :
       llvm::make_early_inc_range(getOperation().getOps<FIntModuleOp>())) {
    auto *node = ig.lookup(op);
    changed = true;

    if (failed(checkModForAnnotations(op, op.getIntrinsic())))
      return signalPassFailure();

    for (auto *use : llvm::make_early_inc_range(node->uses())) {
      auto inst = use->getInstance<InstanceOp>();
      if (failed(checkInstForAnnotations(inst, op.getIntrinsic())))
        return signalPassFailure();

      // Replace the instance of this intmodule with firrtl.int.generic.
      // Inputs become operands, outputs are the result (if any).
      ImplicitLocOpBuilder builder(op.getLoc(), inst);

      SmallVector<Value> inputs;
      struct OutputInfo {
        Value result;
        BundleType::BundleElement element;
      };
      SmallVector<OutputInfo> outputs;
      for (auto [idx, result] : llvm::enumerate(inst.getResults())) {
        // Replace inputs with wires that will be used as operands.
        if (inst.getPortDirection(idx) != Direction::Out) {
          auto w = builder.create<WireOp>(result.getLoc(), result.getType())
                       .getResult();
          result.replaceAllUsesWith(w);
          inputs.push_back(w);
          continue;
        }

        // Gather outputs.  This will become a bundle if more than one, but
        // typically there are zero or one.
        auto ftype = dyn_cast<FIRRTLBaseType>(inst.getType(idx));
        if (!ftype) {
          inst.emitError("intrinsic has non-FIRRTL or non-base port type")
              << inst.getType(idx);
          signalPassFailure();
          return;
        }
        outputs.push_back(
            OutputInfo{inst.getResult(idx),
                       BundleType::BundleElement(inst.getPortName(idx),
                                                 /*isFlip=*/false, ftype)});
      }

      // Create the replacement operation.
      if (outputs.empty()) {
        // If no outputs, just create the operation.
        builder.create<GenericIntrinsicOp>(/*result=*/Type(),
                                           op.getIntrinsicAttr(), inputs,
                                           op.getParameters());

      } else if (outputs.size() == 1) {
        // For single output, the result is the output.
        auto resultType = outputs.front().element.type;
        auto intop = builder.create<GenericIntrinsicOp>(
            resultType, op.getIntrinsicAttr(), inputs, op.getParameters());
        outputs.front().result.replaceAllUsesWith(intop.getResult());
      } else {
        // For multiple outputs, create a bundle with fields for each output
        // and replace users with subfields.
        auto resultType = builder.getType<BundleType>(llvm::map_to_vector(
            outputs, [](const auto &info) { return info.element; }));
        auto intop = builder.create<GenericIntrinsicOp>(
            resultType, op.getIntrinsicAttr(), inputs, op.getParameters());
        for (auto &output : outputs)
          output.result.replaceAllUsesWith(builder.create<SubfieldOp>(
              intop.getResult(), output.element.name));
      }
      // Remove instance from IR and instance graph.
      use->erase();
      inst.erase();
      ++numInstances;
    }
    // Remove intmodule from IR and instance graph.
    ig.erase(node);
    op.erase();
    ++numIntmodules;
  }

  // Special handling for magic EICG wrapper extmodule.  Deprecate and remove.
  if (fixupEICGWrapper) {
    constexpr StringRef eicgName = "EICG_wrapper";
    for (auto op :
         llvm::make_early_inc_range(getOperation().getOps<FExtModuleOp>())) {
      if (op.getDefname() != eicgName)
        continue;

      // FIXME: Dedup group annotation could be annotated to EICG_wrapper but
      //        it causes an error with `fixupEICGWrapper`. For now drop the
      //        annotation until we fully migrate into EICG intrinsic.
      if (AnnotationSet::removeAnnotations(op, firrtl::dedupGroupAnnoClass))
        op.emitWarning() << "Annotation " << firrtl::dedupGroupAnnoClass
                         << " on EICG_wrapper is dropped";

      if (failed(checkModForAnnotations(op, eicgName)))
        return signalPassFailure();

      auto *node = ig.lookup(op);
      changed = true;
      for (auto *use : llvm::make_early_inc_range(node->uses())) {
        auto inst = use->getInstance<InstanceOp>();
        if (failed(checkInstForAnnotations(inst, eicgName)))
          return signalPassFailure();

        ImplicitLocOpBuilder builder(op.getLoc(), inst);
        auto replaceResults = [](OpBuilder &b, auto &&range) {
          return llvm::map_to_vector(range, [&b](auto v) {
            auto w = b.create<WireOp>(v.getLoc(), v.getType()).getResult();
            v.replaceAllUsesWith(w);
            return w;
          });
        };

        auto inputs = replaceResults(builder, inst.getResults().drop_back());
        auto intop = builder.create<GenericIntrinsicOp>(
            builder.getType<ClockType>(), "circt_clock_gate", inputs,
            op.getParameters());
        inst.getResults().back().replaceAllUsesWith(intop.getResult());

        // Remove instance from IR and instance graph.
        use->erase();
        inst.erase();
      }
      // Remove extmodule from IR and instance graph.
      ig.erase(node);
      op.erase();
    }
  }

  markAnalysesPreserved<InstanceGraph>();

  if (!changed)
    markAllAnalysesPreserved();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerIntmodulesPass(bool fixupEICGWrapper) {
  auto pass = std::make_unique<LowerIntmodulesPass>();
  pass->fixupEICGWrapper = fixupEICGWrapper;
  return pass;
}
