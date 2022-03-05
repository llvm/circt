//===- HWElaborateGenerate.cpp - hw.generate elaboration ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform performs elaboration of hw.generate operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

using InstanceParameters = llvm::DenseMap<hw::HWModuleOp, ArrayAttr>;

// Evaluate 'ssaValue' to a constant value. If 'ssaValue' did not evaluate to a
// constant, returns failure.
static FailureOr<APInt> ssaToConstantValue(ArrayAttr parameters,
                                           Value ssaValue) {
  if (ssaValue.isa<BlockArgument>())
    return {emitError(ssaValue.getLoc())
            << "expected SSA value to be some constant, but got a block "
               "argument!"};

  Operation *defOp = ssaValue.getDefiningOp();
  return llvm::TypeSwitch<Operation *, FailureOr<APInt>>(defOp)
      .Case<hw::ConstantOp>([&](auto op) {
        return op.valueAttr().template cast<IntegerAttr>().getValue();
      })
      .Case<arith::ConstantOp>([&](auto op) {
        return op.getValue().template cast<IntegerAttr>().getValue();
      })
      .Case<arith::IndexCastOp>([&](auto op) {
        return ssaToConstantValue(parameters, op.getOperand());
      })
      .Case<ParamValueOp>([&](auto op) {
        return evaluateParametricAttr(defOp->getLoc(), parameters, op.value());
      })
      .Default([&](auto) -> FailureOr<APInt> {
        return {defOp->emitOpError()
                << "expected operation to resolve to a constant value."};
      });
}

// Narrows 'value' using a comb.extract operation to the width of the
// hw.array-typed 'array'.
static Value narrowValueToArrayWidth(OpBuilder &builder, Value array,
                                     Value value) {
  OpBuilder::InsertionGuard g(builder);
  if (value.isa<BlockArgument>())
    builder.setInsertionPointToStart(value.getParentBlock());
  else
    builder.setInsertionPointAfter(value.getDefiningOp());
  auto arrayType = array.getType().cast<hw::ArrayType>();
  return builder.create<comb::ExtractOp>(
      value.getLoc(), value, /*lowBit=*/0,
      llvm::Log2_64_Ceil(arrayType.getSize()));
}

// Copies an operation to the current rewriter scope, replacing its operands
// with operands from the 'valueMap', and registerring its result values in
// the 'valueMap'.
// If 'replaceResults' is set, results from the copied operation are allowed to
// override existing values in the 'valueMap'.
static LogicalResult copyOperation(OpBuilder &builder, ArrayAttr parameters,
                                   ValueMapper &valueMap, Operation *op,
                                   bool replaceResults) {
  assert(op->getNumRegions() == 0 &&
         "Trying to copy an operation with nested regions?");
  BlockAndValueMapping mapping;
  Operation *clonedOp;

  auto res =
      llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case<ParamValueOp>([&](auto paramValueOp) {
            // Substitute the param value op with an evaluated constant
            // operation.
            auto paramValue = evaluateParametricAttr(
                paramValueOp.getLoc(), parameters, paramValueOp.value());
            if (failed(paramValue))
              return failure();
            clonedOp = builder.create<hw::ConstantOp>(
                op->getLoc(), paramValueOp.getType(),
                paramValue.getValue().getSExtValue());
            return success();
          })
          .Case<hw::ArrayGetOp>([&](auto op) {
            // HW array operations require indexes to be of equal width of the
            // array itself. Since indexes may originate from loop induction
            // variables (index typed => 64 bit), emit comb.extract operations
            // to fulfil this invariant.
            llvm::SmallVector<Value, 2> narrowedOperands;
            narrowedOperands.push_back(valueMap.get(op.input()));
            narrowedOperands.push_back(narrowValueToArrayWidth(
                builder, valueMap.get(op.input()), valueMap.get(op.index())));

            mapping.map(op->getOperands(), narrowedOperands);
            clonedOp = builder.clone(*op, mapping);
            return success();
          })
          .Default([&](auto op) {
            // Else, just copy a cloned version of the operation, using mapped
            // operands.
            mapping.map(op->getOperands(), valueMap.get(op->getOperands()));
            clonedOp = builder.clone(*op, mapping);
            return success();
          });

  if (failed(res))
    return failure();

  // Record the cloned value in the value map.
  valueMap.set(op->getResults(), clonedOp->getResults(),
               /*replace=*/replaceResults);
  return success();
}

static LogicalResult elaborateOperation(OpBuilder &builder,
                                        ArrayAttr parameters,
                                        ValueMapper &mapper, Operation *op,
                                        bool replaceResults);

// Convenience function for running elaborateOperation on all operations within
// a single nested region of 'op'.
static LogicalResult elaborateRegionOp(OpBuilder &builder, ArrayAttr parameters,
                                       ValueMapper &mapper, Operation *op,
                                       bool replaceResults) {
  assert(op->getNumRegions() == 1 && "Expected op to have a single region");
  for (auto &bodyOp : op->getRegion(0).getOps())
    if (failed(elaborateOperation(builder, parameters, mapper, &bodyOp,
                                  replaceResults)))
      return failure();
  return success();
}

LogicalResult elaborateForOp(OpBuilder &builder, ValueMapper &valueMap,
                             ArrayAttr parameters, scf::ForOp forOp) {
  auto iv = ssaToConstantValue(parameters, forOp.getLowerBound());
  auto ub = ssaToConstantValue(parameters, forOp.getUpperBound());
  auto step = ssaToConstantValue(parameters, forOp.getStep());
  if (failed(iv) || failed(ub) || failed(step))
    return failure();

  // Create loop entry mapping for any iter args.
  for (auto &&[iterArg, init] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getIterOperands()))
    valueMap.set(iterArg, valueMap.get(init), /*replace=*/false);

  auto yieldOp = *forOp.getOps<scf::YieldOp>().begin();

  for (int64_t ivIt = iv->getSExtValue(); ivIt < ub->getSExtValue();
       ivIt += step->getSExtValue()) {
    // Create IV as a constant and update the value map. This will make any
    // operation within the loop body refer to the current constant value of the
    // induction variable at this loop iteration.
    auto constantOp = builder.create<hw::ConstantOp>(
        forOp.getLoc(), builder.getI64Type(), ivIt);
    valueMap.set(forOp.getInductionVar(), constantOp, /*replace=*/true);

    // Elaborate the internals of the for operation. Since we're looping across
    // the same set of operations, we'll allow SSA value replacement in the
    // ValueMapper.
    if (failed(elaborateRegionOp(builder, parameters, valueMap, forOp,
                                 /*replace=*/true)))
      return failure();

    // Associate the scf.yield'ed values with either the iter args or the
    // scf.for return value.
    bool isFinalIteration = ivIt + 1 == ub->getSExtValue();
    if (isFinalIteration) {
      for (auto &&[iterArg, yieldVal] :
           llvm::zip(forOp.getResults(), yieldOp.getOperands()))
        valueMap.set(iterArg, valueMap.get(yieldVal), /*replace=*/false);
    } else {
      for (auto &&[res, yieldVal] :
           llvm::zip(forOp.getRegionIterArgs(), yieldOp.getOperands()))
        valueMap.set(res, valueMap.get(yieldVal), /*replace=*/true);
    }
  }

  return success();
}

// Elaborates 'op' based on the set of provided parameters.
// If 'replaceResults' is set, results from the copied operation are allowed to
// override existing values in the 'valueMap'.
static LogicalResult elaborateOperation(OpBuilder &builder,
                                        ArrayAttr parameters,
                                        ValueMapper &mapper, Operation *op,
                                        bool replaceResults) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<scf::ForOp>([&](auto op) {
        return elaborateForOp(builder, mapper, parameters, op);
      })
      .Case<scf::WhileOp, scf::IfOp>([&](auto) {
        llvm_unreachable("Not yet implemented!");
        return failure();
      })
      .Case<hw::HWGenerateOp>([&](auto op) {
        return elaborateRegionOp(builder, parameters, mapper, op,
                                 /*replace=*/false);
      })
      .Case<scf::YieldOp>([&](auto) {
        // Handled during scf.for op lowering.
        return success();
      })
      .Case<arith::IndexCastOp>([&](auto op) {
        // Index casts only make sense within an scf operation. After
        // elaboration, everything that would have required an index cast has
        // been lowered to a known-width value. To support this, simply map the
        // index cast result directly to its operand.
        mapper.set(op.getResult(), mapper.get(op.getOperand()), replaceResults);
        return success();
      })
      .Default([&](auto op) {
        return copyOperation(builder, parameters, mapper, op, replaceResults);
      });
}

// Elaborates the body of 'base' into 'target' using the set of provided
// 'parameters'.
static LogicalResult elaborateModuleBody(OpBuilder builder,
                                         ArrayAttr parameters, HWModuleOp base,
                                         HWModuleOp target) {
  builder.setInsertionPointToStart(target.getBodyBlock());

  BackedgeBuilder bb(builder, target.getLoc());
  ValueMapper mapper(&bb);

  // Prime value mapper with module block arguments.
  mapper.set(base.getArguments(), target.getArguments());

  // Elaborate every operation of the base module into the target module.
  for (auto &op : *base.getBodyBlock())
    if (failed(elaborateOperation(builder, parameters, mapper, &op,
                                  /*replace=*/false)))
      return failure();

  return success();
}

struct HWElaborateGeneratePass
    : public sv::HWElaborateGenerateBase<HWElaborateGeneratePass> {
  void runOnOperation() override;
};

// Generates a module name by composing the name of 'moduleOp' and the set of
// provided 'parameters'.
static std::string generateModuleName(hw::HWModuleOp moduleOp,
                                      ArrayAttr parameters) {
  assert(parameters.size() != 0);
  std::string name = moduleOp.getName().str();
  for (auto param : parameters) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    int64_t paramValue = paramAttr.getValue().cast<IntegerAttr>().getInt();
    name += "_" + paramAttr.getName().str() + "_" + std::to_string(paramValue);
  }
  return name;
}

// Elaborates the provided 'base' module into the 'target' module. By doing so,
// we create a new module which
// 1. has no parameters
// 2. has a name composing the name of 'base' as well as the 'parameters'
// parameters.
// 3. Has a top-level interface with any parametric types resolved.
// 4. Has any `hw.generate` regions elaborated.
static LogicalResult elaborateModule(OpBuilder builder, ArrayAttr parameters,
                                     HWModuleOp base, HWModuleOp &target) {
  // Update the types of the base module ports based on evaluating any
  // parametric in/output ports.
  auto ports = base.getPorts();
  for (auto &in : llvm::enumerate(base.getFunctionType().getInputs())) {
    auto resType =
        evaluateParametricType(base.getLoc(), parameters, in.value());
    if (failed(resType))
      return failure();
    ports.inputs[in.index()].type = resType.getValue();
  }
  for (auto &out : llvm::enumerate(base.getFunctionType().getResults())) {
    auto resType =
        evaluateParametricType(base.getLoc(), parameters, out.value());
    if (failed(resType))
      return failure();
    ports.outputs[out.index()].type = resType.getValue();
  }

  // Create the elaborated module using the evaluated port info.
  target = builder.create<HWModuleOp>(
      base.getLoc(),
      StringAttr::get(builder.getContext(),
                      generateModuleName(base, parameters)),
      ports);

  // Erase the default created hw.output op - we'll copy the correct operation
  // during body elaboration.
  (*target.getOps<hw::OutputOp>().begin()).erase();

  if (failed(elaborateModuleBody(builder, parameters, base, target)))
    return failure();

  return success();
}

void HWElaborateGeneratePass::runOnOperation() {
  ModuleOp module = getOperation();

  // Record unique module parameterss and references to these.
  llvm::DenseMap<hw::HWModuleOp, llvm::SetVector<ArrayAttr>>
      uniqueModuleparameterss;
  llvm::DenseMap<hw::HWModuleOp,
                 llvm::DenseMap<ArrayAttr, llvm::SmallVector<hw::InstanceOp>>>
      parametersUsers;
  for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
    for (auto instanceOp : hwModule.getOps<hw::InstanceOp>()) {
      auto targetOp = instanceOp.getReferencedModule();
      auto targetHWModule = dyn_cast<hw::HWModuleOp>(targetOp);
      if (!targetHWModule) {
        continue; // Won't elaborate external modules.
      }
      if (targetHWModule.parameters().size() == 0)
        continue; // nothing to record.

      auto parameters = instanceOp.parameters();
      uniqueModuleparameterss[targetHWModule].insert(parameters);
      parametersUsers[targetHWModule][parameters].push_back(instanceOp);
    }
  }

  // Create elaborated modules.
  InstanceParameters parameterss;
  OpBuilder builder = OpBuilder(&getContext());
  builder.setInsertionPointToStart(module.getBody());
  for (auto it : uniqueModuleparameterss) {
    for (auto parameters : it.getSecond()) {
      HWModuleOp elaboratedModule;
      if (failed(elaborateModule(builder, parameters, it.getFirst(),
                                 elaboratedModule))) {
        signalPassFailure();
        return;
      }
      parameterss[elaboratedModule] = parameters;

      // Rewrite instances of the elaborated module to the elaborated module.
      for (auto instanceOp : parametersUsers[it.getFirst()][parameters]) {
        instanceOp->setAttr("moduleName",
                            FlatSymbolRefAttr::get(elaboratedModule));
        instanceOp->setAttr("parameters", ArrayAttr::get(&getContext(), {}));
      }
    }
  }
}

} // namespace

std::unique_ptr<Pass> circt::sv::createHWElaborateGeneratePass() {
  return std::make_unique<HWElaborateGeneratePass>();
}
