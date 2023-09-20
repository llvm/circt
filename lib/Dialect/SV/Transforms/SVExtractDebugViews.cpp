//===- SVExtractDebugViews.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Analysis/DebugAnalysis.h"
#include "circt/Analysis/DebugInfo.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "di-views"

using namespace circt;
using namespace sv;
using namespace hw;
using circt::igraph::InstanceGraphNode;

namespace {

#define GEN_PASS_DEF_SVEXTRACTDEBUGVIEWS
#include "circt/Dialect/SV/SVPasses.h.inc"

struct SVExtractDebugViewsPass
    : public impl::SVExtractDebugViewsBase<SVExtractDebugViewsPass> {
  void runOnOperation() override;
  LogicalResult runOnModule(InstanceGraphNode *node, DIModule &module,
                            DebugAnalysis &debugAnalysis, OpBuilder &builder);

private:
  unsigned refSymCount = 0;
};

struct GlobalContext {};

struct ModuleContext {
  GlobalContext &global;
};

} // namespace

std::unique_ptr<Pass> circt::sv::createSVExtractDebugViewsPass() {
  return std::make_unique<SVExtractDebugViewsPass>();
}

void SVExtractDebugViewsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Extracting debug views\n");
  auto &debugInfo = getAnalysis<DebugInfo>();
  auto &debugAnalysis = getAnalysis<DebugAnalysis>();
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  // Nothing to do if the instance graph is empty.
  if (instanceGraph.begin() == instanceGraph.end()) {
    markAllAnalysesPreserved();
    return;
  }

  // Find the top-levels modules. Each will get its own debug view hierarchy.
  auto maybeTopLevels = instanceGraph.getInferredTopLevelNodes();
  if (failed(maybeTopLevels))
    return signalPassFailure();
  ArrayRef<InstanceGraphNode *> topLevels = *maybeTopLevels;

  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
  for (auto *topLevel : topLevels) {
    auto moduleOp = topLevel->getModule();
    if (auto *diModule =
            debugInfo.moduleNodes.lookup(moduleOp.getModuleNameAttr()))
      if (isa_and_nonnull<HWModuleOp>(diModule->op))
        if (failed(runOnModule(topLevel, *diModule, debugAnalysis, builder)))
          return signalPassFailure();
  }

  // Remove debug-only operations in the original debug hierarchy.
  for (auto *op : debugAnalysis.debugOps)
    op->dropAllReferences();
  for (auto *op : debugAnalysis.debugOps)
    op->erase();
}

LogicalResult SVExtractDebugViewsPass::runOnModule(InstanceGraphNode *node,
                                                   DIModule &module,
                                                   DebugAnalysis &debugAnalysis,
                                                   OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs() << "- Generating debug view of top-level "
                          << module.name << "\n");

  SmallVector<std::tuple<DIModule *, HWModuleOp, SmallVector<Attribute>>>
      worklist, backWorklist;
  SmallDenseMap<DIModule *, unsigned> viewIndices;

  auto addToWorklist = [&](DIModule &module,
                           ArrayRef<Attribute> hierPath) -> HWModuleOp {
    auto viewName = builder.getStringAttr(module.name.getValue() + "_dbg" +
                                          Twine(viewIndices[&module]++));
    auto viewModule = builder.create<HWModuleOp>(
        node->getModule()->getLoc(), viewName, ArrayRef<PortInfo>{});
    // TODO: Insert through symbol table to legalize name.
    worklist.emplace_back(&module, viewModule, hierPath);
    return viewModule;
  };

  auto topViewModule = addToWorklist(module, {});

  // Instantiate the debug view in the top-level.
  auto debugInstSym = InnerSymAttr::get(
      StringAttr::get(&getContext(), "dbg" + Twine(refSymCount++)));
  OpBuilder::atBlockBegin(&node->getModule()->getRegion(0).front())
      .create<InstanceOp>(topViewModule.getLoc(), topViewModule, "dbgview",
                          ArrayRef<Value>{}, ArrayAttr{}, debugInstSym);

  while (!worklist.empty()) {
    std::swap(worklist, backWorklist);
    for (auto &worklistItem : backWorklist) {
      auto *module = std::get<0>(worklistItem);
      auto viewModule = std::get<1>(worklistItem);
      auto &hierPath = std::get<2>(worklistItem);
      LLVM_DEBUG(llvm::dbgs()
                 << "- Generating debug view of " << module->name << "\n");

      auto innerBuilder = OpBuilder::atBlockBegin(viewModule.getBodyBlock());

      auto alwaysOp =
          innerBuilder.create<sv::AlwaysCombOp>(viewModule.getLoc());
      auto alwaysBuilder = OpBuilder::atBlockBegin(alwaysOp.getBodyBlock());
      innerBuilder.setInsertionPoint(alwaysOp);

      IRMapping outlinedValues;

      std::function<Value(Value, bool)> outlineValue =
          [&](Value value, bool castClocks) -> Value {
        if (auto alreadyOutlined = outlinedValues.lookupOrNull(value))
          return alreadyOutlined;

        Value outlinedValue;
        auto *defOp = value.getDefiningOp();
        if (defOp && debugAnalysis.debugOps.contains(defOp)) {
          // Look through HW wires instead of outlining them.
          if (auto wireOp = dyn_cast<hw::WireOp>(defOp))
            return outlineValue(wireOp.getInput(), castClocks);

          // Destroy debug structs and reconstruct them as SV structs.
          if (auto structOp = dyn_cast<debug::StructOp>(defOp)) {
            // LLVM_DEBUG(llvm::dbgs() << "  - Destructure " << *defOp << "\n");
            SmallVector<Value> fieldValues;
            SmallVector<StructType::FieldInfo> fieldInfos;
            for (auto [value, name] :
                 llvm::zip(structOp.getFields(), structOp.getNames())) {
              // LLVM_DEBUG(llvm::dbgs() << "    - Field " << name << "\n");
              value = outlineValue(value, true);
              if (!value)
                continue;
              fieldValues.push_back(value);
              auto &info = fieldInfos.emplace_back();
              info.name = cast<StringAttr>(name);
              info.type = value.getType();
            }
            if (fieldInfos.empty())
              return {};
            auto structType = StructType::get(value.getContext(), fieldInfos);
            auto createOp = alwaysBuilder.create<StructCreateOp>(
                value.getLoc(), structType, fieldValues);
            outlinedValues.map(value, createOp);
            return createOp;
          }

          // Destroy debug arrays and reconstruct them as SV arrays.
          if (auto arrayOp = dyn_cast<debug::ArrayOp>(defOp)) {
            // LLVM_DEBUG(llvm::dbgs() << "  - Destructure " << *defOp << "\n");
            SmallVector<Value> elementValues;
            Type elementType;
            for (auto value : arrayOp.getElements()) {
              // LLVM_DEBUG(llvm::dbgs()
              //            << "    - Element " << elementValues.size() <<
              //            "\n");
              value = outlineValue(value, true);
              elementValues.push_back(value);
              if (!elementType) {
                elementType = value.getType();
              } else if (elementType != value.getType()) {
                LLVM_DEBUG(
                    llvm::dbgs()
                    << "    - Ignoring array with non-uniform element type: "
                    << arrayOp << "\n");
                return {};
              }
            }
            if (elementValues.empty() || !elementType)
              return {};
            auto arrayType = ArrayType::get(elementType, elementValues.size());
            std::reverse(elementValues.begin(), elementValues.end());
            auto createOp = alwaysBuilder.create<ArrayCreateOp>(
                value.getLoc(), arrayType, elementValues);
            outlinedValues.map(value, createOp);
            return createOp;
          }

          // Clone the operation into the debug view.
          assert(!isa<hw::InstanceOp>(defOp));
          // LLVM_DEBUG(llvm::dbgs() << "  - Outline " << *defOp << "\n");
          for (auto operand : defOp->getOperands())
            outlineValue(operand, false);
          alwaysBuilder.clone(*defOp, outlinedValues);
          outlinedValue = outlinedValues.lookup(value);
        } else {
          // LLVM_DEBUG(llvm::dbgs() << "  - Reference " << value << "\n");
          OpBuilder wireBuilder(value.getContext());
          if (auto *defOp = value.getDefiningOp())
            wireBuilder.setInsertionPointAfter(defOp);
          else
            wireBuilder.setInsertionPointToStart(value.getParentBlock());
          auto wireSym = InnerSymAttr::get(
              wireBuilder.getStringAttr("dbg" + Twine(refSymCount++)));
          auto castValue = value;
          bool hasClockCast = false;
          if (isa<seq::ClockType>(castValue.getType())) {
            castValue = wireBuilder.createOrFold<seq::FromClockOp>(
                castValue.getLoc(), castValue);
            hasClockCast = true;
          }
          wireBuilder.create<hw::WireOp>(value.getLoc(), castValue,
                                         StringRef("_dbg"), wireSym);

          hierPath.push_back(InnerRefAttr::get(
              cast<HWModuleLike>(module->op).getModuleNameAttr(),
              wireSym.getSymName()));
          auto hierPathOp = builder.create<HierPathOp>(
              value.getLoc(),
              builder.getStringAttr("dbg" + Twine(refSymCount++)),
              builder.getArrayAttr(hierPath));
          hierPath.pop_back();
          // TODO: Use global sym table to insert and uniquify hierPath symbol.

          outlinedValue = alwaysBuilder.create<sv::XMRRefOp>(
              value.getLoc(), InOutType::get(castValue.getType()),
              hierPathOp.getSymName());
          outlinedValue = alwaysBuilder.create<sv::ReadInOutOp>(value.getLoc(),
                                                                outlinedValue);
          if (hasClockCast)
            outlinedValue = alwaysBuilder.createOrFold<seq::ToClockOp>(
                outlinedValue.getLoc(), outlinedValue);
        }
        if (!outlinedValue)
          return {};
        if (castClocks && isa<seq::ClockType>(outlinedValue.getType()))
          outlinedValue = alwaysBuilder.createOrFold<seq::FromClockOp>(
              outlinedValue.getLoc(), outlinedValue);
        outlinedValues.map(value, outlinedValue);
        return outlinedValue;
      };

      for (auto *var : module->variables) {
        if (!var->value)
          continue;
        auto outlinedValue = outlineValue(var->value, true);
        if (!outlinedValue)
          continue;

        if (auto fileLoc = dyn_cast<FileLineColLoc>(var->loc)) {
          auto lineDirective = alwaysBuilder.getStringAttr(
              "`line " + Twine(fileLoc.getLine()) + " \"" +
              fileLoc.getFilename().getValue() + "\" 0");
          innerBuilder.create<sv::VerbatimOp>(var->loc, lineDirective);
        }

        auto debugDeclSym = InnerSymAttr::get(
            innerBuilder.getStringAttr("dbg" + Twine(refSymCount++)));
        auto declOp = innerBuilder.create<sv::RegOp>(
            var->loc, outlinedValue.getType(), var->name, debugDeclSym);
        alwaysBuilder.create<sv::BPAssignOp>(var->loc, declOp, outlinedValue);
      }

      for (auto *inst : module->instances) {
        auto instOp = dyn_cast_or_null<InstanceOp>(inst->op);
        if (!instOp)
          continue;

        InnerSymAttr instSym = instOp.getInnerSymAttr();
        if (!instSym) {
          instSym = InnerSymAttr::get(
              innerBuilder.getStringAttr("dbg" + Twine(refSymCount++)));
          instOp.setInnerSymAttr(instSym);
        }
        hierPath.push_back(InnerRefAttr::get(
            cast<HWModuleLike>(module->op).getModuleNameAttr(),
            instSym.getSymName()));
        auto viewModule = addToWorklist(*inst->module, hierPath);
        hierPath.pop_back();
        auto debugInstSym = InnerSymAttr::get(
            innerBuilder.getStringAttr("dbg" + Twine(refSymCount++)));
        innerBuilder.create<InstanceOp>(inst->op->getLoc(), viewModule,
                                        inst->name, ArrayRef<Value>{},
                                        ArrayAttr{}, debugInstSym);
      }
    }
    backWorklist.clear();
  }

  return success();
}
