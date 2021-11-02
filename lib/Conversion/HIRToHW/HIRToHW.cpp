//===----- HIRToHW.cpp - HIR To HW Conversion Pass-------*-C++-*-===//
//
// This pass converts HIR to HW, Comb and SV dialect.
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "HIRToHWUtils.h"
#include "circt/Conversion/HIRToHW/HIRToHW.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
using namespace circt;

class HIRToHWPass : public HIRToHWBase<HIRToHWPass> {
public:
  void runOnOperation() override;

private:
  void updateHIRToHWMapForFuncInputs(hw::HWModuleOp,
                                     mlir::Block::BlockArgListType,
                                     FuncToHWModulePortMap);

  LogicalResult visitRegion(mlir::Region &);
  LogicalResult visitOperation(Operation *);

  LogicalResult visitOp(hir::BusAssignOp);
  LogicalResult visitOp(hir::BusBroadcastOp);
  LogicalResult visitOp(hir::BusOp);
  LogicalResult visitOp(hir::BusOrOp);
  LogicalResult visitOp(hir::BusSelectOp);
  LogicalResult visitOp(hir::CallOp);
  LogicalResult visitOp(hir::CommentOp);
  LogicalResult visitOp(hir::CreateTupleOp);
  LogicalResult visitOp(hir::FuncExternOp);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::IsFirstIterOp);
  LogicalResult visitOp(hir::RecvOp);
  LogicalResult visitOp(hir::ReturnOp);
  LogicalResult visitOp(hir::SendOp);
  LogicalResult visitOp(hir::TensorExtractOp);
  LogicalResult visitOp(hir::TensorInsertOp);
  LogicalResult visitOp(hir::TimeOp);
  LogicalResult visitOp(hir::WhileOp);
  LogicalResult visitOp(mlir::arith::ConstantOp);

private:
  OpBuilder *builder;
  llvm::DenseMap<Value, Value> mapHIRToHWValue;
  llvm::DenseMap<StringRef, uint64_t> mapFuncNameToInstanceCount;
  SmallVector<Operation *> opsToErase;
  Value clk;
  hw::HWModuleOp hwModuleOp;
};

LogicalResult HIRToHWPass::visitOp(mlir::arith::ConstantOp op) {
  if (op.getType().isa<mlir::IndexType>())
    return success();
  if (!op.getType().isa<mlir::IntegerType>())
    return op.emitError(
        "hir-to-hw pass only supports IntegerType/IndexType constants.");

  mapHIRToHWValue[op.getResult()] = builder->create<hw::ConstantOp>(
      op.getLoc(), op.value().dyn_cast<IntegerAttr>());
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusOp op) {
  // Add a placeholder SSA Var for the buses. CallOp visit will replace them.
  // We need to do this because HW dialect does not have SSA dominance.
  auto *constantXOp = getConstantX(builder, op.getType());
  auto placeHolderSSAVar = constantXOp->getResult(0);
  auto name = helper::getOptionalName(constantXOp, 0);
  if (name)
    helper::setNames(constantXOp, {name.getValue()});
  mapHIRToHWValue[op.res()] = placeHolderSSAVar;
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::CommentOp op) {
  builder->create<sv::VerbatimOp>(builder->getUnknownLoc(),
                                  "//COMMENT: " + op.comment());
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::CallOp op) {
  assert(op.offset().getValue() == 0);
  auto filteredOperands = filterCallOpArgs(op.getFuncType(), op.operands());

  // Get the mapped inputs and create the input types for instance op.
  SmallVector<Value> hwInputs;
  for (auto input : filteredOperands.first) {
    auto hwInput = mapHIRToHWValue[input];
    // FIXME: Once all the op visitors are written, replace this with
    // assert(hwInput).
    if (!hwInput)
      return success();
    hwInputs.push_back(hwInput);
  }
  assert(!op.offset() || op.offset().getValue() == 0);

  hwInputs.push_back(mapHIRToHWValue[op.tstart()]);
  hwInputs.push_back(this->clk);

  auto hwInputTypes = helper::getTypes(hwInputs);

  auto sendBuses = filteredOperands.second;
  auto sendBusTypes = helper::getTypes(sendBuses);

  // Create instance op result types.
  SmallVector<Type> hwResultTypes;
  for (auto ty : sendBusTypes)
    hwResultTypes.push_back(convertToHWType(ty));

  for (auto ty : op.getResultTypes())
    hwResultTypes.push_back(convertToHWType(ty));

  auto instanceName = builder->getStringAttr(
      op.callee().str() + "_inst" +
      std::to_string(mapFuncNameToInstanceCount[op.callee()]++));

  auto calleeHWModule = dyn_cast_or_null<hw::HWModuleOp>(op.getCalleeDecl());
  auto calleeHWModuleExtern =
      dyn_cast_or_null<hw::HWModuleExternOp>(op.getCalleeDecl());
  if (!(calleeHWModule || calleeHWModuleExtern)) {
    op.emitError() << "Could not find decl for the hw module.";
    return success();
  }
  hw::InstanceOp instanceOp;
  if (calleeHWModule)
    instanceOp = builder->create<hw::InstanceOp>(
        op.getLoc(), calleeHWModule, instanceName, hwInputs,
        getHWParams(op->getAttr("params")), StringAttr());
  else
    instanceOp = builder->create<hw::InstanceOp>(
        op.getLoc(), calleeHWModuleExtern, instanceName, hwInputs,
        getHWParams(op->getAttr("params")), StringAttr());

  // Map callop input send buses to the results of the instance op and replace
  // all prev uses of the placeholder hw ssa vars corresponding to these send
  // buses.
  uint64_t i;
  for (i = 0; i < sendBuses.size(); i++) {
    auto placeHolderSSAVar = mapHIRToHWValue[sendBuses[i]];
    assert(placeHolderSSAVar);
    placeHolderSSAVar.replaceAllUsesWith(instanceOp.getResult(i));
    mapHIRToHWValue[sendBuses[i]] = instanceOp.getResult(i);
  }

  // Map the CallOp return vars to instance op return vars.
  for (uint64_t j = 0; i + j < instanceOp.getNumResults(); j++)
    mapHIRToHWValue[op.getResult(j)] = instanceOp.getResult(i + j);

  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::TimeOp op) {
  auto tIn = mapHIRToHWValue[op.timevar()];
  if (!tIn) // FIXME
    return success();

  auto name = helper::getOptionalName(op, 0);
  auto nameAttr = name ? builder->getStringAttr(name.getValue()) : StringAttr();
  auto reg =
      builder->create<sv::RegOp>(op.getLoc(), builder->getI1Type(), nameAttr);
  builder->create<sv::AlwaysOp>(
      op.getLoc(), sv::EventControl::AtPosEdge, clk, [&reg, tIn, this] {
        this->builder->create<sv::PAssignOp>(builder->getUnknownLoc(),
                                             reg.result(), tIn);
      });

  auto tOut = builder->create<sv::ReadInOutOp>(op.getLoc(), reg);
  mapHIRToHWValue[op.res()] = tOut;
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::WhileOp op) {
  assert(op.offset().getValue() == 0);
  auto &bb = op.body().front();
  auto nextIterOp = dyn_cast<hir::NextIterOp>(bb.back());
  assert(nextIterOp);
  assert(!nextIterOp.offset() || nextIterOp.offset().getValue() == 0);
  assert(!op.offset() || op.offset().getValue() == 0);

  auto conditionBegin = mapHIRToHWValue[op.condition()];
  auto tstartBegin = mapHIRToHWValue[op.tstart()];

  // FIXME
  if (!conditionBegin || !tstartBegin)
    return success();

  auto conditionTrue = builder->create<mlir::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(builder->getI1Type(), 1));

  // If no condition is provided in the next_iter op then condition is always
  // true.
  Value conditionIter;
  if (nextIterOp.condition())
    conditionIter = getConstantX(builder, builder->getI1Type())->getResult(0);
  else
    conditionIter = conditionTrue;

  auto tstartIter = getConstantX(builder, builder->getI1Type())->getResult(0);

  // Placeholder mappings. Will be replaced when next_iter op is processed.
  if (nextIterOp.condition())
    mapHIRToHWValue[nextIterOp.condition()] = conditionIter;
  mapHIRToHWValue[nextIterOp.tstart()] = tstartIter;

  auto tNextBegin = builder->create<comb::AndOp>(builder->getUnknownLoc(),
                                                 tstartBegin, conditionBegin);
  auto tNextIter = builder->create<comb::AndOp>(builder->getUnknownLoc(),
                                                tstartIter, conditionIter);
  auto iterTimeVar =
      builder->create<comb::OrOp>(op.getLoc(), tNextBegin, tNextIter);
  auto notConditionBegin = builder->create<comb::XorOp>(
      builder->getUnknownLoc(), conditionBegin, conditionTrue);
  auto notConditionIter = builder->create<comb::XorOp>(
      builder->getUnknownLoc(), conditionIter, conditionTrue);
  auto tLastBegin = builder->create<comb::AndOp>(
      builder->getUnknownLoc(), tstartBegin, notConditionBegin);
  auto tLastIter = builder->create<comb::AndOp>(builder->getUnknownLoc(),
                                                tstartIter, notConditionIter);
  auto tLast = builder->create<comb::OrOp>(op.getLoc(), tLastBegin, tLastIter);
  auto tLastName = helper::getOptionalName(op, 0);
  if (tLastName)
    tLast->setAttr("name", builder->getStringAttr(tLastName.getValue()));

  mapHIRToHWValue[op.getIterTimeVar()] = iterTimeVar;
  mapHIRToHWValue[op.res()] = tLast;
  return visitRegion(op.body());
}

LogicalResult HIRToHWPass::visitOp(hir::IsFirstIterOp op) {
  auto isFirstIter =
      mapHIRToHWValue[op->getParentOfType<hir::WhileOp>().tstart()];
  assert(isFirstIter);
  mapHIRToHWValue[op.res()] = isFirstIter;
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::RecvOp op) {
  assert(op.offset().getValue() == 0);
  assert(mapHIRToHWValue[op.bus()]);
  mapHIRToHWValue[op.res()] = mapHIRToHWValue[op.bus()];
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::ReturnOp op) {
  auto funcOp = op->getParentOfType<hir::FuncOp>();

  auto portMap =
      getHWModulePortMap(*builder, funcOp.getFuncType(), funcOp.getInputNames(),
                         funcOp.getResultNames());
  auto funcArgs = funcOp.getFuncBody().front().getArguments();

  // hwOutputs are the outputs to be returned in the hw module.
  SmallVector<Value> hwOutputs;

  // Insert 'send' buses in the input args of hir.func. These buses are outputs
  // in the hw dialect.
  for (size_t i = 0; i < funcArgs.size(); i++) {
    auto modulePortInfo = portMap.getPortInfoForFuncInput(i);
    if (modulePortInfo.direction == hw::PortDirection::OUTPUT) {
      hwOutputs.push_back(mapHIRToHWValue[funcArgs[i]]);
    }
  }

  // Insert the hir.func outputs.
  for (Value funcResult : op.operands()) {
    assert(mapHIRToHWValue[funcResult]);
    hwOutputs.push_back(mapHIRToHWValue[funcResult]);
  }

  auto *oldOutputOp = hwModuleOp.getBodyBlock()->getTerminator();
  oldOutputOp->replaceAllUsesWith(
      builder->create<hw::OutputOp>(builder->getUnknownLoc(), hwOutputs));
  opsToErase.push_back(oldOutputOp);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::SendOp op) {
  assert(op.offset().getValue() == 0);
  auto value = mapHIRToHWValue[op.value()];
  auto placeHolderBus = mapHIRToHWValue[op.bus()];
  auto tstart = mapHIRToHWValue[op.tstart()];
  Value defaultValue;
  if (auto defaultAttr = op->getAttr("default")) {
    assert(defaultAttr.isa<IntegerAttr>());
    defaultValue = builder->create<hw::ConstantOp>(
        builder->getUnknownLoc(), defaultAttr.dyn_cast<IntegerAttr>());
  } else {
    defaultValue = builder->create<sv::ConstantXOp>(builder->getUnknownLoc(),
                                                    value.getType());
  }
  auto newBus = builder->create<comb::MuxOp>(
      builder->getUnknownLoc(), value.getType(), tstart, value, defaultValue);
  placeHolderBus.replaceAllUsesWith(newBus);
  mapHIRToHWValue[op.bus()] = newBus;
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusSelectOp op) {
  auto combinedArrayOfValues = builder->create<hw::ArrayCreateOp>(
      builder->getUnknownLoc(),
      SmallVector<Value>(
          {mapHIRToHWValue[op.false_bus()], mapHIRToHWValue[op.true_bus()]}));

  mapHIRToHWValue[op.res()] = builder->create<hw::ArrayGetOp>(
      builder->getUnknownLoc(), combinedArrayOfValues,
      mapHIRToHWValue[op.select_bus()]);

  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::FuncExternOp op) {
  builder = new OpBuilder(op);
  builder->setInsertionPoint(op);
  auto portMap = getHWModulePortMap(*builder, op.getFuncType(),
                                    op.getInputNames(), op.getResultNames());
  auto name = builder->getStringAttr(op.getNameAttr().getValue().str());

  builder->create<hw::HWModuleExternOp>(
      op.getLoc(), name, portMap.getPortInfoList(),
      op.getNameAttr().getValue().str(),
      getHWParams(op->getAttr("params"), true));

  delete (builder);
  opsToErase.push_back(op);
  return success();
}

void HIRToHWPass::updateHIRToHWMapForFuncInputs(
    hw::HWModuleOp hwModuleOp, mlir::Block::BlockArgListType funcArgs,
    FuncToHWModulePortMap portMap) {
  for (size_t i = 0; i < funcArgs.size(); i++) {
    auto modulePortInfo = portMap.getPortInfoForFuncInput(i);
    if (modulePortInfo.direction == hw::PortDirection::INPUT) {
      auto hwArg =
          hwModuleOp.getBodyBlock()->getArgument(modulePortInfo.argNum);
      mapHIRToHWValue[funcArgs[i]] = hwArg;
    } else {
      mapHIRToHWValue[funcArgs[i]] =
          getConstantX(builder, funcArgs[i].getType())->getResult(0);
    }
  }
}

LogicalResult HIRToHWPass::visitOp(hir::FuncOp op) {
  this->builder = new OpBuilder(op);
  this->builder->setInsertionPoint(op);
  auto portMap = getHWModulePortMap(*builder, op.getFuncType(),
                                    op.getInputNames(), op.getResultNames());
  auto name = builder->getStringAttr("hw_" + op.getNameAttr().getValue().str());

  this->hwModuleOp = builder->create<hw::HWModuleOp>(op.getLoc(), name,
                                                     portMap.getPortInfoList());
  this->builder->setInsertionPointToStart(hwModuleOp.getBodyBlock());

  updateHIRToHWMapForFuncInputs(
      hwModuleOp, op.getFuncBody().front().getArguments(), portMap);

  this->clk = hwModuleOp.getBodyBlock()->getArguments().back();
  auto visitResult = visitRegion(op.getFuncBody());

  opsToErase.push_back(op);
  delete (builder);
  return visitResult;
}

LogicalResult HIRToHWPass::visitOp(hir::TensorExtractOp op) {
  auto tensorTy = op.tensor().getType().dyn_cast<mlir::TensorType>();
  auto shape = tensorTy.getShape();
  SmallVector<Value> indices;
  assert(mapHIRToHWValue[op.tensor()]);
  for (auto idx : op.indices()) {
    indices.push_back(idx);
  }

  auto linearIdxValue = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(tensorTy.getNumElements())),
          helper::calcLinearIndex(indices, shape).getValue()));

  mapHIRToHWValue[op.res()] = builder->create<hw::ArrayGetOp>(
      builder->getUnknownLoc(), mapHIRToHWValue[op.tensor()], linearIdxValue);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::TensorInsertOp op) {
  // left = hw.slice
  // right = hw.slice
  // element = hw.array_create op.element
  // hw.array_concat left, element, right
  // calc left slice.
  auto tensorTy = op.tensor().getType().dyn_cast<mlir::TensorType>();
  SmallVector<Value> indices;
  for (auto idx : op.indices()) {
    indices.push_back(idx);
  }
  auto idx = helper::calcLinearIndex(indices, tensorTy.getShape()).getValue();
  auto leftWidth = idx;

  auto leftIdx = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(tensorTy.getNumElements())),
          0));
  auto rightWidth = tensorTy.getNumElements() - idx - 1;
  auto rightIdx = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(tensorTy.getNumElements())),
          idx + 1));

  auto leftSliceTy =
      hw::ArrayType::get(builder->getContext(),
                         convertToHWType(tensorTy.getElementType()), leftWidth);
  auto leftSlice =
      builder->create<hw::ArraySliceOp>(builder->getUnknownLoc(), leftSliceTy,
                                        mapHIRToHWValue[op.tensor()], leftIdx);
  auto element = builder->create<hw::ArrayCreateOp>(
      builder->getUnknownLoc(), mapHIRToHWValue[op.element()]);

  auto rightSliceTy = hw::ArrayType::get(
      builder->getContext(), convertToHWType(tensorTy.getElementType()),
      rightWidth);
  auto rightSlice =
      builder->create<hw::ArraySliceOp>(builder->getUnknownLoc(), rightSliceTy,
                                        mapHIRToHWValue[op.tensor()], rightIdx);
  mapHIRToHWValue[op.res()] = builder->create<hw::ArrayConcatOp>(
      builder->getUnknownLoc(),
      SmallVector<Value>({leftSlice, element, rightSlice}));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusAssignOp op) {
  assert(mapHIRToHWValue[op.src()]);
  assert(mapHIRToHWValue[op.dest()]);
  mapHIRToHWValue[op.dest()].replaceAllUsesWith(mapHIRToHWValue[op.src()]);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusBroadcastOp op) {
  auto tensorTy = op.res().getType().dyn_cast<mlir::TensorType>();
  SmallVector<Value> replicatedArray;
  for (int i = 0; i < tensorTy.getNumElements(); i++) {
    replicatedArray.push_back(mapHIRToHWValue[op.bus()]);
  }

  mapHIRToHWValue[op.res()] = builder->create<hw::ArrayCreateOp>(
      builder->getUnknownLoc(), replicatedArray);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusOrOp op) {
  mapHIRToHWValue[op.res()] = builder->create<comb::OrOp>(
      builder->getUnknownLoc(), mapHIRToHWValue[op.left()],
      mapHIRToHWValue[op.right()]);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::CreateTupleOp op) {
  SmallVector<Value> mappedArgs;
  SmallVector<Type> mappedArgTypes;
  for (auto arg : op.args()) {
    assert(mapHIRToHWValue[arg]);
    mappedArgs.push_back(mapHIRToHWValue[arg]);
    mappedArgTypes.push_back(mapHIRToHWValue[arg].getType());
  }

  mapHIRToHWValue[op.res()] = builder->create<comb::ConcatOp>(
      builder->getUnknownLoc(), convertToHWType(op.res().getType()),
      mappedArgs);
  return success();
}

LogicalResult HIRToHWPass::visitRegion(mlir::Region &region) {
  Block &bb = *region.begin();
  for (Operation &operation : bb)
    if (failed(visitOperation(&operation)))
      return failure();
  return success();
}

LogicalResult HIRToHWPass::visitOperation(Operation *operation) {
  if (auto op = dyn_cast<mlir::arith::ConstantOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusBroadcastOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CommentOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CallOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::TimeOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::WhileOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::IsFirstIterOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::SendOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::RecvOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::TensorExtractOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::TensorInsertOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusAssignOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusOrOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusSelectOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CreateTupleOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ReturnOp>(operation))
    return visitOp(op);

  return operation->emitError() << "Unsupported operation for hir-to-hw pass.";
}

void HIRToHWPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  WalkResult result = moduleOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::FuncOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    } else if (auto op = dyn_cast<hir::FuncExternOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
  helper::eraseOps(opsToErase);
}

/// hir-to-hw pass Constructor
std::unique_ptr<mlir::Pass> circt::createHIRToHWPass() {
  return std::make_unique<HIRToHWPass>();
}
