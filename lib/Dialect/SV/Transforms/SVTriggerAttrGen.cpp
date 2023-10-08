//===- SVTriggerAttrGen.cpp -------- SV Trigger Attributes Generator-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This generation pass will generate trigger attribute and attach it to
// corresponding place if this blockArgument / operation is used as trigger
// signal in sequential block.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sv-trigger-attr-gen"
using namespace circt;
using namespace mlir;
using namespace sv;
using namespace hw;

using llvm::dbgs;

// Define a type for representing the relations between trigger signals and
// operations.
using OpsToEventsMap = DenseMap<Operation *, SmallVector<IntegerAttr>>;

//===----------------------------------------------------------------------===//
// SVTriggerAttrGen Helpers
//===----------------------------------------------------------------------===//
namespace {
/// The function to generate trigger attribute in module.
static void genTriggerAttribute(ModuleOp module, OpsToEventsMap &opsMap) {
  // Walk through the module to collect the info to generate trigger attributes.
  module.walk([&](sv::AlwaysOp op) {
    for (size_t i = 0, e = op.getNumConditions(); i != e; i++) {
      unsigned index = 0;
      auto *parent = op->getParentOp();
      auto signal = op.getCondition(i).value;
      auto *signalOp = signal.getDefiningOp();
      // Do it when this signal comes from input argument of module.
      // These signals come from sources that are not reachable in the
      // current domain.
      if (signal.getDefiningOp() == nullptr) {
        index = cast<BlockArgument>(signal).getArgNumber();
        opsMap[parent].push_back(
            IntegerAttr::get(IntegerType::get(module.getContext(), 8), index));

        // Do it when this signal comes from currently reacheable source
        // such as results from callOp or from combinational logic part.
      } else {
        index = cast<OpResult>(signal).getResultNumber();

        // When the operation waited to be added 'sv.trigger' attribute, judge
        // whether the operation is hw::InstanceOp or not. If it is instanceOp
        // in hw dialect, additional offset will be added to the index from the
        // number of ports in referenced module.
        if (auto inst = dyn_cast<hw::InstanceOp>(signalOp)) {
          auto *refedModule = inst.getReferencedModuleSlow();

          if (auto ref = dyn_cast<HWModuleOp>(refedModule))
            index += ref.getNumArguments();
          else
            index += dyn_cast<HWModuleExternOp>(refedModule).getNumArguments();
        }
        opsMap[signalOp].push_back(
            IntegerAttr::get(IntegerType::get(module.getContext(), 8), index));
      }
    }
  });

  StringAttr nameAttr = StringAttr::get(module.getContext(), "sv.trigger");

  // Remove duplicate elements and reorder the value in container, then create
  // trigger attribute and set it to corresponding operation.
  for (auto &map : opsMap) {
    auto *defineOp = map.first;

    // Remove duplicates using SetVector and then create a SmallVector.
    llvm::SetVector<IntegerAttr> setVector(map.second.begin(),
                                           map.second.end());
    llvm::SmallVector<IntegerAttr> sortedAttrs(setVector.begin(),
                                               setVector.end());

    // Sort the SmallVector of IntegerAttr elements in ascending order.
    llvm::sort(sortedAttrs, [](mlir::IntegerAttr a, mlir::IntegerAttr b) {
      return a.getValue().getSExtValue() < b.getValue().getSExtValue();
    });

    // Create an ArrayAttr from the sorted vector.
    SmallVector<Attribute> attrVector(sortedAttrs.begin(), sortedAttrs.end());
    ArrayAttr arrayAttr = ArrayAttr::get(module.getContext(), attrVector);

    // Attach the trigger information to the corresponding operation.
    defineOp->setAttr(nameAttr, arrayAttr);
  }
}

/// Reset the place to set the value of trigger attribute on arguments.
static void resetArgTriggerAttr(HWModuleOp hwModuleOp, unsigned &sequenceId,
                                StringAttr &nameAttr) {
  auto argType = hwModuleOp.getArgument(0).getType();
  auto triAttr = cast<ArrayAttr>(hwModuleOp->getAttr("sv.trigger"));
  auto argTriggerNum = triAttr.size();

  for (auto attr : triAttr) {
    auto triValue = attr.cast<IntegerAttr>().getValue().getZExtValue();

    for (auto gep : hwModuleOp.getOps<LLVM::GEPOp>())
      if (argType == gep.getOperand(0).getType()) {
        auto indices = gep.getIndices();
        auto offset = cast<IntegerAttr>(indices[1]).getValue().getZExtValue();
        auto *load = *gep->getUsers().begin();

        if (triValue == offset && isa<LLVM::LoadOp>(load)) {
          auto integerAttr = IntegerAttr::get(
              IntegerType::get(hwModuleOp.getContext(), 8), sequenceId);
          auto arrayAttr =
              ArrayAttr::get(hwModuleOp->getContext(), integerAttr);
          load->setAttr(nameAttr, arrayAttr);
          sequenceId++;
        }
      }
  }

  assert(
      argTriggerNum == sequenceId &&
      "Not all suitable operations were found to reset the trigger attribute! "
      "You need to execute the GenStateStruct pass first with "
      "'--state-struct-generate'.");

  hwModuleOp->removeAttr(nameAttr);
}

/// Reset the place to set the value of trigger attribute on instances.
static void resetInstTriggerAttr(hw::InstanceOp instanceOp,
                                 unsigned &sequenceId, StringAttr &nameAttr) {
  auto instOperandTy = instanceOp->getOperand(0).getType();
  auto triAttr = cast<ArrayAttr>(instanceOp->getAttr("sv.trigger"));
  auto instTriggerNum = triAttr.size() + sequenceId;

  // Get the struct type in pointer to check whether is opaque type or not.
  auto instOpElementTy =
      cast<LLVM::LLVMPointerType>(instOperandTy).getElementType();
  auto instOpStructTy = cast<LLVM::LLVMStructType>(instOpElementTy);

  if (instOpStructTy.isOpaque())
    for (auto bitcast :
         cast<HWModuleOp>(instanceOp->getParentOp()).getOps<LLVM::BitcastOp>())
      if (bitcast->getOperand(0).getType() == instOperandTy) {
        instOperandTy = bitcast.getType();
        break;
      }

  for (auto attr : triAttr) {
    auto triValue = attr.cast<IntegerAttr>().getValue().getZExtValue();

    for (auto gep :
         cast<HWModuleOp>(instanceOp->getParentOp()).getOps<LLVM::GEPOp>()) {
      if (instOperandTy == gep.getOperand(0).getType()) {
        auto indices = gep.getIndices();
        auto offset = cast<IntegerAttr>(indices[1]).getValue().getZExtValue();
        auto *load = *gep->getUsers().begin();

        if (triValue == offset && isa<LLVM::LoadOp>(load)) {
          auto integerAttr = IntegerAttr::get(
              IntegerType::get(instanceOp.getContext(), 8), sequenceId);
          auto arrayAttr =
              ArrayAttr::get(instanceOp->getContext(), integerAttr);
          load->setAttr(nameAttr, arrayAttr);
          sequenceId++;
        }
      }
    }
  }

  assert(
      instTriggerNum == sequenceId &&
      "Not all suitable operations were found to reset the trigger attribute! "
      "You need to execute the GenStateStruct pass first with "
      "'--state-struct-generate'.");

  instanceOp->removeAttr(nameAttr);
}

/// Reset the place to set the value of trigger attribute for ordinary
/// operations.
static void resetOpTriggerAttr(Operation *op, unsigned &sequenceId,
                               StringAttr &nameAttr) {
  auto triAttr = cast<ArrayAttr>(op->getAttr("sv.trigger"));

  // FIXME: This might be an improvement if we could support more than one
  // trigger attribute on the operations other than module or instance
  // operation. Temporarily, there is no good solution when removing the
  // original value. (equivalent to mapping relations) when the size is more
  // than one, it would be difficult to find out the original mapping
  // relations in subsequent passes.
  assert(triAttr.size() == 1 && "Warning: not support to reset the trigger "
                                "value if its size exceeds one!");

  op->removeAttr(nameAttr);

  auto integerAttr =
      IntegerAttr::get(IntegerType::get(op->getContext(), 8), sequenceId);
  auto arrayAttr = ArrayAttr::get(op->getContext(), integerAttr);
  op->setAttr(nameAttr, arrayAttr);

  sequenceId++;
}

/// The function to reset values of trigger attribute in module with the
/// predefined rule.
static void resetTriggerAttrValue(ModuleOp module) {
  StringAttr nameAttr = StringAttr::get(module.getContext(), "sv.trigger");

  module->walk([&](HWModuleOp hwModuleOp) {
    unsigned sequenceId = 0;
    LLVM_DEBUG(dbgs() << "Start to reset trigger attribute in the module: "
                      << hwModuleOp.getModuleName() << "\n");

    if (hwModuleOp->hasAttr("sv.trigger"))
      resetArgTriggerAttr(hwModuleOp, sequenceId, nameAttr);

    LLVM_DEBUG(
        dbgs() << "Print the sequenceId (Phase #1 -- Trigger on arguments): "
               << sequenceId << "\n");

    for (Operation &op : *hwModuleOp.getBodyBlock()) {
      if (op.hasAttr("sv.trigger")) {
        if (isa<hw::InstanceOp>(op)) {
          resetInstTriggerAttr(cast<hw::InstanceOp>(op), sequenceId, nameAttr);
          LLVM_DEBUG(
              dbgs()
              << "Print the sequenceId (Phase #2 -- Trigger on instances): "
              << sequenceId << "\n");
          continue;
        }

        // Must exclude the loadOp， Why？ Beacuse the loadOp that has trigger
        // attribute is formed by previously resetting behavior. For correctness
        // after resetting the value, We must avoid double counting.
        if (!isa<LLVM::LoadOp>(op)) {
          resetOpTriggerAttr(&op, sequenceId, nameAttr);
          LLVM_DEBUG(dbgs() << "Print the sequenceId (Phase #3 -- Trigger on "
                               "oridinary operations): "
                            << sequenceId << "\n");
        }
      }
    }
  });
}
} // namespace

//===----------------------------------------------------------------------===//
// SVTriggerAttrGen Pass
//===----------------------------------------------------------------------===//
namespace {
struct SVTriggerAttrGenPass
    : public SVTriggerAttrGenBase<SVTriggerAttrGenPass> {
  SVTriggerAttrGenPass(bool resetTriggerValue) {
    this->resetTriggerValue = resetTriggerValue;
  }
  void runOnOperation() override;
};
} // end anonymous namespace

void SVTriggerAttrGenPass::runOnOperation() {
  OpsToEventsMap opsToEventsMap;
  ModuleOp module = getOperation();

  // The option '--reset-trigger-value' is to reset all values of trigger
  // attributes in the current module in ascending order. Attention: please use
  // this option after executing the GenStateStruct pass with
  // '--state-struct-generate'.
  if (!resetTriggerValue) {
    genTriggerAttribute(module, opsToEventsMap);
  } else {
    resetTriggerAttrValue(module);
  }
}
//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
circt::sv::createSVTriggerAttrGenPass(bool resetTriggerValue) {
  return std::make_unique<SVTriggerAttrGenPass>(resetTriggerValue);
}
