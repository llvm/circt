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

  std::string getUniquePostfix() { return "_" + std::to_string(uniqueInt++); }

  LogicalResult visitHWOp(Operation *);
  LogicalResult visitOperation(Operation *);
  LogicalResult visitOp(hir::BusAssignOp);
  LogicalResult visitOp(hir::BusBroadcastOp);
  LogicalResult visitOp(hir::BusMapOp);
  LogicalResult visitOp(hir::BusOp);
  LogicalResult visitOp(hir::BusRecvOp);
  LogicalResult visitOp(hir::BusSendOp);
  LogicalResult visitOp(hir::BusTensorAssignOp);
  LogicalResult visitOp(hir::BusTensorAssignElementOp);
  LogicalResult visitOp(hir::BusTensorGetElementOp);
  LogicalResult visitOp(hir::BusTensorInsertElementOp);
  LogicalResult visitOp(hir::BusTensorMapOp);
  LogicalResult visitOp(hir::BusTensorOp);
  LogicalResult visitOp(hir::CallOp);
  LogicalResult visitOp(hir::CastOp);
  LogicalResult visitOp(hir::CommentOp);
  LogicalResult visitOp(mlir::arith::ConstantOp);
  LogicalResult visitOp(hir::DelayOp);
  LogicalResult visitOp(hir::FuncExternOp);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::IsFirstIterOp);
  LogicalResult visitOp(hir::LatchOp);
  LogicalResult visitOp(hir::NextIterOp);
  LogicalResult visitOp(hir::ProbeOp);
  LogicalResult visitOp(hir::ReturnOp);
  LogicalResult visitOp(hir::TimeOp);
  LogicalResult visitOp(hir::WhileOp);
  LogicalResult visitRegion(mlir::Region &);

private:
  Operation *getEmittedHWModuleOp(StringRef hwModuleName);

private:
  OpBuilder *builder;
  HIRToHWMapping mapHIRToHWValue;
  llvm::DenseMap<StringRef, uint64_t> mapFuncNameToInstanceCount;
  Value clk;
  Value reset;
  hw::HWModuleOp hwModuleOp;
  mlir::ModuleOp mlirModuleOp; // Enclosing module{}
  size_t uniqueInt = 0;
  DenseMap<Value, SmallVector<Value>> mapArrayToElements;
  DenseMap<StringRef, Operation *> mapNameToHWModuleOp;
};

LogicalResult HIRToHWPass::visitOp(hir::BusMapOp op) {
  SmallVector<Value> hwOperands;
  for (auto operand : op.operands())
    hwOperands.push_back(mapHIRToHWValue.lookup(operand));
  auto results = insertBusMapLogic(*builder, op.body().front(), hwOperands);
  for (size_t i = 0; i < op.getNumResults(); i++)
    mapHIRToHWValue.map(op.getResult(i), results[i]);

  return success();
}

Operation *HIRToHWPass::getEmittedHWModuleOp(StringRef hwModuleName) {
  auto *operation = mapNameToHWModuleOp[hwModuleName];
  assert(operation);
  auto hwModuleOp = dyn_cast<hw::HWModuleOp>(operation);
  auto hwModuleExternOp = dyn_cast<hw::HWModuleExternOp>(operation);
  assert(hwModuleOp || hwModuleExternOp);
  if (hwModuleOp)
    return hwModuleOp;
  return hwModuleExternOp;
}

LogicalResult HIRToHWPass::visitOp(hir::BusOp op) {
  // Add a placeholder SSA Var for the buses. CallOp visit will replace them.
  // We need to do this because HW dialect does not have SSA dominance.
  auto *constantXOp = getConstantX(builder, op.getType());
  auto placeHolderSSAVar = constantXOp->getResult(0);
  auto name = helper::getOptionalName(constantXOp, 0);
  if (name)
    helper::setNames(constantXOp, {name.getValue()});
  mapHIRToHWValue.map(op.res(), placeHolderSSAVar);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorMapOp op) {
  auto uLoc = builder->getUnknownLoc();
  SmallVector<Value> hwOperandTensors;
  for (auto operand : op.operands())
    hwOperandTensors.push_back(mapHIRToHWValue.lookup(operand));
  auto hwArrayTy = hwOperandTensors[0].getType().dyn_cast<hw::ArrayType>();

  // If the tensor has only one element then treat this like BusMapOp.
  if (!hwArrayTy) {
    auto results =
        insertBusMapLogic(*builder, op.body().front(), hwOperandTensors);
    for (size_t i = 0; i < op.getNumResults(); i++)
      mapHIRToHWValue.map(op.getResult(i), results[i]);
    return success();
  }

  // Copy the bus map logic as many times as there are elements in the tensor.
  // Save the outputs in the results[array-index][result-number] array.
  SmallVector<SmallVector<Value>> results;
  for (size_t arrayIdx = 0; arrayIdx < hwArrayTy.getSize(); arrayIdx++) {
    SmallVector<Value> hwOperands;
    for (auto hwOperandT : hwOperandTensors)
      hwOperands.push_back(
          insertConstArrayGetLogic(*builder, hwOperandT, arrayIdx));
    results.push_back(
        insertBusMapLogic(*builder, op.body().front(), hwOperands));
  }

  // For each result (i.e. results[*][result-num]), create a hw array from the
  // elements.
  for (size_t resultNum = 0; resultNum < op.getNumResults(); resultNum++) {
    SmallVector<Value> resultElements;
    for (size_t arrayIdx = 0; arrayIdx < hwArrayTy.getSize(); arrayIdx++)
      resultElements.push_back(results[arrayIdx][resultNum]);
    mapHIRToHWValue.map(
        op.getResult(resultNum),
        builder->create<hw::ArrayCreateOp>(uLoc, resultElements));
  }
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorOp op) {
  auto *constantXOp =
      getConstantXArray(builder, op.getType(), mapArrayToElements);
  auto placeHolderSSAVar = constantXOp->getResult(0);
  auto name = helper::getOptionalName(constantXOp, 0);
  if (name)
    helper::setNames(constantXOp, {name.getValue()});
  mapHIRToHWValue.map(op.res(), placeHolderSSAVar);
  return success();
}
LogicalResult HIRToHWPass::visitOp(hir::CommentOp op) {
  builder->create<sv::VerbatimOp>(builder->getUnknownLoc(),
                                  "//COMMENT: " + op.comment());
  return success();
}

LogicalResult HIRToHWPass::visitOp(mlir::arith::ConstantOp op) {
  if (op.result().getType().isa<mlir::IntegerType>())
    return op.emitError()
           << "hir-to-hw pass supports only hw.constant op for integer types.";
  return success();
}
LogicalResult HIRToHWPass::visitOp(hir::CallOp op) {
  assert(op.offset().getValue() == 0);
  auto filteredOperands = filterCallOpArgs(op.getFuncType(), op.operands());

  // Get the mapped inputs and create the input types for instance op.
  SmallVector<Value> hwInputs;
  for (auto input : filteredOperands.first) {
    auto hwInput = mapHIRToHWValue.lookup(input);
    hwInputs.push_back(hwInput);
  }
  assert(!op.offset() || op.offset().getValue() == 0);

  hwInputs.push_back(mapHIRToHWValue.lookup(op.tstart()));
  hwInputs.push_back(this->clk);
  hwInputs.push_back(this->reset);

  auto hwInputTypes = helper::getTypes(hwInputs);

  auto sendBuses = filteredOperands.second;
  auto sendBusTypes = helper::getTypes(sendBuses);

  // Create instance op result types.
  SmallVector<Type> hwResultTypes;
  for (auto ty : sendBusTypes)
    hwResultTypes.push_back(*helper::convertToHWType(ty));

  for (auto ty : op.getResultTypes())
    hwResultTypes.push_back(*helper::convertToHWType(ty));

  auto instanceName =
      op.instance_nameAttr()
          ? op.instance_nameAttr()
          : builder->getStringAttr(
                op.callee().str() + "_inst" +
                std::to_string(mapFuncNameToInstanceCount[op.callee()]++));

  auto *calleeHWModule = getEmittedHWModuleOp(op.callee());
  hw::InstanceOp instanceOp;
  instanceOp = builder->create<hw::InstanceOp>(
      op.getLoc(), calleeHWModule, instanceName, hwInputs,
      getHWParams(op->getAttr("params")), StringAttr());

  // Map callop input send buses to the results of the instance op and replace
  // all prev uses of the placeholder hw ssa vars corresponding to these send
  // buses.
  uint64_t i;
  for (i = 0; i < sendBuses.size(); i++) {
    auto placeHolderSSAVar = mapHIRToHWValue.lookup(sendBuses[i]);
    mapHIRToHWValue.replaceAllHWUses(placeHolderSSAVar,
                                     instanceOp.getResult(i));
  }

  // Map the CallOp return vars to instance op return vars.
  for (uint64_t j = 0; i + j < instanceOp.getNumResults(); j++)
    mapHIRToHWValue.map(op.getResult(j), instanceOp.getResult(i + j));

  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::CastOp op) {
  if (op.input().getType().isa<mlir::IndexType>()) {
    assert(op.res().getType().isa<mlir::IntegerType>());
    auto constantOp =
        dyn_cast<mlir::arith::ConstantOp>(op.input().getDefiningOp());
    assert(constantOp);
    auto value = constantOp.value().dyn_cast<mlir::IntegerAttr>().getInt();
    mapHIRToHWValue.map(
        op.res(),
        builder->create<hw::ConstantOp>(
            builder->getUnknownLoc(),
            IntegerAttr::get(*helper::convertToHWType(op.res().getType()),
                             value)));
    return success();
  }

  assert(*helper::convertToHWType(op.input().getType()) ==
         *helper::convertToHWType(op.res().getType()));
  mapHIRToHWValue.map(op.res(), mapHIRToHWValue.lookup(op.input()));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::TimeOp op) {
  auto tIn = mapHIRToHWValue.lookup(op.timevar());
  if (!tIn.getType().isa<mlir::IntegerType>())
    return tIn.getDefiningOp()->emitError()
           << "Expected converted type to be i1.";
  auto name = helper::getOptionalName(op, 0);
  Value res =
      getDelayedValue(builder, tIn, op.delay(), name, op.getLoc(), clk, reset);
  assert(res.getType().isa<mlir::IntegerType>());
  mapHIRToHWValue.map(op.res(), res);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::WhileOp op) {
  assert(op.offset().getValue() == 0);
  auto &bb = op.body().front();
  assert(!op.offset() || op.offset().getValue() == 0);

  auto conditionBegin = mapHIRToHWValue.lookup(op.condition());
  auto tstartBegin = mapHIRToHWValue.lookup(op.tstart());

  assert(conditionBegin && tstartBegin);

  auto conditionTrue = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(builder->getI1Type(), 1));

  // Placeholder values.
  auto tstartNextIterOp =
      getConstantX(builder, builder->getI1Type())->getResult(0);
  auto conditionNextIterOp =
      getConstantX(builder, builder->getI1Type())->getResult(0);

  auto tNextBegin = builder->create<comb::AndOp>(builder->getUnknownLoc(),
                                                 tstartBegin, conditionBegin);
  auto tNextIter = builder->create<comb::AndOp>(
      builder->getUnknownLoc(), tstartNextIterOp, conditionNextIterOp);
  Value iterTimeVar =
      builder->create<comb::OrOp>(op.getLoc(), tNextBegin, tNextIter);
  auto notConditionBegin = builder->create<comb::XorOp>(
      builder->getUnknownLoc(), conditionBegin, conditionTrue);
  auto notConditionIter = builder->create<comb::XorOp>(
      builder->getUnknownLoc(), conditionNextIterOp, conditionTrue);
  auto tLastBegin = builder->create<comb::AndOp>(
      builder->getUnknownLoc(), tstartBegin, notConditionBegin);
  auto tLastIter = builder->create<comb::AndOp>(
      builder->getUnknownLoc(), tstartNextIterOp, notConditionIter);
  auto tLast = builder->create<comb::OrOp>(op.getLoc(), tLastBegin, tLastIter);
  auto tLastName = helper::getOptionalName(op, 0);
  if (tLastName)
    tLast->setAttr("name", builder->getStringAttr(tLastName.getValue()));
  mapHIRToHWValue.map(op.getIterTimeVar(), iterTimeVar);
  mapHIRToHWValue.map(op.res(), tLast);
  auto visitResult = visitRegion(op.body());

  auto nextIterOp = dyn_cast<hir::NextIterOp>(bb.back());
  assert(nextIterOp);

  // Replace placeholder values.
  tstartNextIterOp.replaceAllUsesWith(
      mapHIRToHWValue.lookup(nextIterOp.tstart()));
  conditionNextIterOp.replaceAllUsesWith(
      mapHIRToHWValue.lookup(nextIterOp.condition()));

  if (failed(visitResult))
    return failure();
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::IsFirstIterOp op) {
  auto isFirstIter =
      mapHIRToHWValue.lookup(op->getParentOfType<hir::WhileOp>().tstart());
  mapHIRToHWValue.map(op.res(), isFirstIter);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::LatchOp op) {
  auto tstart = mapHIRToHWValue.lookup(op.tstart());
  auto uLoc = builder->getUnknownLoc();
  auto hwInput = mapHIRToHWValue.lookup(op.input());
  auto latchReg = builder->create<sv::RegOp>(uLoc, hwInput.getType());
  auto bodyCtor = [&uLoc, &latchReg, hwInput, &tstart, this] {
    builder->create<sv::IfOp>(uLoc, tstart, [&uLoc, &latchReg, hwInput, this] {
      builder->create<sv::PAssignOp>(uLoc, latchReg.result(), hwInput);
    });
  };
  builder->create<sv::AlwaysOp>(uLoc, sv::EventControl::AtPosEdge, clk,
                                bodyCtor);
  auto latchRegOut = builder->create<sv::ReadInOutOp>(uLoc, latchReg.result());
  auto latchOut =
      builder->create<comb::MuxOp>(uLoc, tstart, hwInput, latchRegOut);
  mapHIRToHWValue.map(op.res(), latchOut);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::NextIterOp op) {
  assert(op.offset().getValue() == 0);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::ProbeOp op) {
  auto input = mapHIRToHWValue.lookup(op.input());
  assert(input);
  auto wire = builder
                  ->create<sv::WireOp>(builder->getUnknownLoc(),
                                       input.getType(), op.verilog_nameAttr())
                  .getResult();
  builder->create<sv::AssignOp>(builder->getUnknownLoc(), wire, input);
  builder->create<sv::VerbatimOp>(builder->getUnknownLoc(),
                                  builder->getStringAttr("//PROBE: {{0}}"),
                                  wire, builder->getArrayAttr({}));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusRecvOp op) {
  assert(op.offset().getValue() == 0);
  mapHIRToHWValue.map(op.res(), mapHIRToHWValue.lookup(op.bus()));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::ReturnOp op) {
  auto funcOp = op->getParentOfType<hir::FuncOp>();

  auto portMap =
      getHWModulePortMap(*builder, op.getLoc(), funcOp.getFuncType(),
                         funcOp.getInputNames(), funcOp.getResultNames());
  auto funcArgs = funcOp.getFuncBody().front().getArguments();

  // hwOutputs are the outputs to be returned in the hw module.
  SmallVector<Value> hwOutputs;

  // Insert 'send' buses in the input args of hir.func. These buses are outputs
  // in the hw dialect.
  for (size_t i = 0; i < funcArgs.size(); i++) {
    auto modulePortInfo = portMap.getPortInfoForFuncInput(i);
    if (modulePortInfo.direction == hw::PortDirection::OUTPUT) {
      hwOutputs.push_back(mapHIRToHWValue.lookup((Value)funcArgs[i]));
    }
  }

  // Insert the hir.func outputs.
  for (Value funcResult : op.operands()) {
    hwOutputs.push_back(mapHIRToHWValue.lookup(funcResult));
  }

  auto *oldOutputOp = hwModuleOp.getBodyBlock()->getTerminator();
  oldOutputOp->replaceAllUsesWith(
      builder->create<hw::OutputOp>(builder->getUnknownLoc(), hwOutputs));
  oldOutputOp->erase();
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusSendOp op) {
  assert(op.offset().getValue() == 0);
  auto value = mapHIRToHWValue.lookup(op.value());
  if (!value)
    return op.emitError() << "Could not find mapped value.";
  auto placeHolderBus = mapHIRToHWValue.lookup(op.bus());
  auto tstart = mapHIRToHWValue.lookup(op.tstart());
  Value defaultValue;
  if (auto defaultAttr = op->getAttr("default")) {
    assert(defaultAttr.isa<IntegerAttr>());
    defaultValue = builder->create<hw::ConstantOp>(
        builder->getUnknownLoc(), defaultAttr.dyn_cast<IntegerAttr>());
  } else {
    defaultValue = getConstantX(builder, value.getType())->getResult(0);
  }
  auto newBus = builder->create<comb::MuxOp>(
      builder->getUnknownLoc(), value.getType(), tstart, value, defaultValue);
  mapHIRToHWValue.replaceAllHWUses(placeHolderBus, newBus);
  return success();
}

Value instantiateBusSelectLogic(OpBuilder &builder, Value selectBus,
                                Value trueBus, Value falseBus) {
  auto combinedArrayOfValues = builder.create<hw::ArrayCreateOp>(
      builder.getUnknownLoc(), SmallVector<Value>({trueBus, falseBus}));

  return builder.create<hw::ArrayGetOp>(builder.getUnknownLoc(),
                                        combinedArrayOfValues, selectBus);
}

LogicalResult HIRToHWPass::visitOp(hir::FuncExternOp op) {
  auto portMap = getHWModulePortMap(*builder, op.getLoc(), op.getFuncType(),
                                    op.getInputNames(), op.getResultNames());
  auto name = builder->getStringAttr(op.getName());
  auto verilogNameAttr = op->getAttrOfType<StringAttr>("verilogName");
  auto hwOp = builder->create<hw::HWModuleExternOp>(
      op.getLoc(), name, portMap.getPortInfoList(),
      verilogNameAttr ? verilogNameAttr.getValue() : op.getName(),
      getHWParams(op->getAttr("params"), true));
  mapNameToHWModuleOp[op.getName()] = hwOp;
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
      mapHIRToHWValue.map(funcArgs[i], hwArg);
    } else {
      if (funcArgs[i].getType().isa<hir::BusType>())
        mapHIRToHWValue.map(
            funcArgs[i],
            getConstantX(builder, funcArgs[i].getType())->getResult(0));
      else if (funcArgs[i].getType().isa<hir::BusTensorType>())
        mapHIRToHWValue.map(funcArgs[i],
                            getConstantXArray(builder, funcArgs[i].getType(),
                                              mapArrayToElements)
                                ->getResult(0));
      else
        assert(false);
    }
  }
}

LogicalResult HIRToHWPass::visitOp(hir::FuncOp op) {
  auto portMap = getHWModulePortMap(*builder, op.getLoc(), op.getFuncType(),
                                    op.getInputNames(), op.getResultNames());
  auto name = builder->getStringAttr(op.getNameAttr().getValue().str());

  this->hwModuleOp = builder->create<hw::HWModuleOp>(op.getLoc(), name,
                                                     portMap.getPortInfoList());
  mapNameToHWModuleOp[op.getName()] = hwModuleOp;
  OpBuilder::InsertionGuard guard(*this->builder);
  this->builder->setInsertionPointToStart(hwModuleOp.getBodyBlock());

  updateHIRToHWMapForFuncInputs(
      hwModuleOp, op.getFuncBody().front().getArguments(), portMap);

  this->clk = getClkFromHWModule(hwModuleOp);
  this->reset = getResetFromHWModule(hwModuleOp);
  auto visitResult = visitRegion(op.getFuncBody());

  return visitResult;
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorGetElementOp op) {
  auto busTensorTy = op.tensor().getType().dyn_cast<hir::BusTensorType>();
  auto shape = busTensorTy.getShape();
  SmallVector<Value> indices;
  auto hwTensor = mapHIRToHWValue.lookup(op.tensor());
  assert(hwTensor);
  if (!hwTensor.getType().isa<hw::ArrayType>()) {
    mapHIRToHWValue.map(op.res(), hwTensor);
    return success();
  }

  for (auto idx : op.indices()) {
    indices.push_back(idx);
  }

  auto linearIdxValue = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(busTensorTy.getNumElements())),
          helper::calcLinearIndex(indices, shape).getValue()));
  mapHIRToHWValue.map(
      op.res(), builder->create<hw::ArrayGetOp>(builder->getUnknownLoc(),
                                                hwTensor, linearIdxValue));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorInsertElementOp op) {
  // left = hw.slice
  // right = hw.slice
  // element = hw.array_create op.element
  // hw.array_concat left, element, right
  // calc left slice.

  auto busTensorTy = op.tensor().getType().dyn_cast<hir::BusTensorType>();
  assert(op.element().getType().dyn_cast<hir::BusType>().getElementType() ==
         busTensorTy.getElementType());
  auto hwTensor = mapHIRToHWValue.lookup(op.tensor());
  if (!hwTensor.getType().isa<hw::ArrayType>()) {
    mapHIRToHWValue.map(op.res(), mapHIRToHWValue.lookup(op.element()));
    return success();
  }

  assert(busTensorTy.getNumElements() > 1);
  SmallVector<Value> indices;
  for (auto idx : op.indices()) {
    indices.push_back(idx);
  }
  // Note that hw dialect arranges arrays right to left - index zero goes to
  // rightmost end syntactically.
  // Strangely, the operand vectors passed to the hw op builders are printed
  // left to right. So operand[0] would actually be the last
  auto idx =
      helper::calcLinearIndex(indices, busTensorTy.getShape()).getValue();
  auto leftWidth = busTensorTy.getNumElements() - idx - 1;
  auto rightWidth = idx;

  auto leftIdx = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(busTensorTy.getNumElements())),
          idx + 1));
  auto rightIdx = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(busTensorTy.getNumElements())),
          0));

  auto leftSliceTy = hw::ArrayType::get(
      builder->getContext(),
      *helper::convertToHWType(busTensorTy.getElementType()), leftWidth);
  auto rightSliceTy = hw::ArrayType::get(
      builder->getContext(),
      *helper::convertToHWType(busTensorTy.getElementType()), rightWidth);

  auto leftSlice = builder->create<hw::ArraySliceOp>(
      builder->getUnknownLoc(), leftSliceTy, hwTensor, leftIdx);
  auto rightSlice = builder->create<hw::ArraySliceOp>(
      builder->getUnknownLoc(), rightSliceTy, hwTensor, rightIdx);
  auto hwElement = mapHIRToHWValue.lookup(op.element());
  if (*helper::convertToHWType(busTensorTy.getElementType()) !=
      hwElement.getType()) {
    return op.emitError() << "Incompatible hw types "
                          << *helper::convertToHWType(
                                 busTensorTy.getElementType())
                          << " and " << hwElement.getType();
  }
  auto elementAsArray =
      builder->create<hw::ArrayCreateOp>(builder->getUnknownLoc(), hwElement);

  assert(leftSlice.getType().isa<hw::ArrayType>());
  assert(rightSlice.getType().isa<hw::ArrayType>());
  assert(elementAsArray.getType().isa<hw::ArrayType>());
  assert(helper::getElementType(leftSliceTy) ==
         helper::getElementType(elementAsArray.getType()));
  assert(helper::getElementType(rightSlice.getType()) ==
         helper::getElementType(elementAsArray.getType()));
  mapHIRToHWValue.map(
      op.res(),
      builder->create<hw::ArrayConcatOp>(
          builder->getUnknownLoc(),
          SmallVector<Value>({leftSlice, elementAsArray, rightSlice})));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusAssignOp op) {
  auto hwDest = mapHIRToHWValue.lookup(op.dest());
  assert(isa<sv::ConstantXOp>(hwDest.getDefiningOp()));
  auto hwSrc = mapHIRToHWValue.lookup(op.src());
  mapHIRToHWValue.replaceAllHWUses(hwDest, hwSrc);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorAssignOp op) {
  // dest is predefined using a placeholder.
  mapHIRToHWValue.replaceAllHWUses(mapHIRToHWValue.lookup(op.dest()),
                                   mapHIRToHWValue.lookup(op.src()));
  return success();
}
LogicalResult HIRToHWPass::visitOp(hir::BusTensorAssignElementOp op) {
  auto hwArray = mapHIRToHWValue.lookup(op.tensor());
  auto hwElement = mapHIRToHWValue.lookup(op.bus());
  if (!hwArray.getType().isa<hw::ArrayType>()) {
    mapHIRToHWValue.replaceAllHWUses(hwArray, hwElement);
    mapHIRToHWValue.map(op.tensor(), hwElement);
    return success();
  }
  auto arrayElements = mapArrayToElements[hwArray];
  SmallVector<Value> indices;
  for (auto idx : op.indices()) {
    indices.push_back(idx);
  }
  auto idx =
      helper::calcLinearIndex(
          indices,
          op.tensor().getType().dyn_cast<hir::BusTensorType>().getShape())
          .getValue();
  arrayElements[arrayElements.size() - idx - 1].replaceAllUsesWith(hwElement);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusBroadcastOp op) {
  auto busTensorTy = op.res().getType().dyn_cast<hir::BusTensorType>();
  auto hwBus = mapHIRToHWValue.lookup(op.bus());
  if (busTensorTy.getNumElements() == 1) {
    mapHIRToHWValue.map(op.res(), hwBus);
    return success();
  }

  SmallVector<Value> replicatedArray;
  for (size_t i = 0; i < busTensorTy.getNumElements(); i++) {
    replicatedArray.push_back(hwBus);
  }

  mapHIRToHWValue.map(op.res(), builder->create<hw::ArrayCreateOp>(
                                    builder->getUnknownLoc(), replicatedArray));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::DelayOp op) {
  auto input = mapHIRToHWValue.lookup(op.input());
  assert(input);
  auto name = helper::getOptionalName(op, 0);
  mapHIRToHWValue.map(op.res(), getDelayedValue(builder, input, op.delay(),
                                                name, op.getLoc(), clk, reset));
  return success();
}

LogicalResult HIRToHWPass::visitHWOp(Operation *operation) {
  auto operandMap = mapHIRToHWValue.getBlockAndValueMapping();
  auto *clonedOperation = builder->clone(*operation, operandMap);
  for (size_t i = 0; i < operation->getNumResults(); i++) {
    mapHIRToHWValue.map(operation->getResult(i), clonedOperation->getResult(i));
  }
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
  if (auto op = dyn_cast<hir::BusMapOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusBroadcastOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorMapOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CommentOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::arith::ConstantOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CallOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::DelayOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::TimeOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::WhileOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::IsFirstIterOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::LatchOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusSendOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusRecvOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorGetElementOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorInsertElementOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusAssignOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorAssignOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorAssignElementOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ReturnOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::NextIterOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CastOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ProbeOp>(operation))
    return visitOp(op);
  if (auto *dialect = operation->getDialect();
      isa<comb::CombDialect, hw::HWDialect, sv::SVDialect>(dialect))
    return visitHWOp(operation);

  return operation->emitError() << "Unsupported operation for hir-to-hw pass.";
}

void HIRToHWPass::runOnOperation() {
  this->mlirModuleOp = getOperation();
  this->builder = new OpBuilder(mlirModuleOp.getLoc().getContext());
  this->builder->setInsertionPointToStart(mlirModuleOp.getBody(0));
  WalkResult result =
      mlirModuleOp.walk([this](Operation *operation) -> WalkResult {
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

  // erase unnecessary ops.
  SmallVector<Operation *> opsToErase;
  for (auto &operation : getOperation()) {
    if (!isa<hw::HWDialect>(operation.getDialect()))
      opsToErase.push_back(&operation);
  }
  helper::eraseOps(opsToErase);
}

/// hir-to-hw pass Constructor
std::unique_ptr<mlir::Pass> circt::createHIRToHWPass() {
  return std::make_unique<HIRToHWPass>();
}
