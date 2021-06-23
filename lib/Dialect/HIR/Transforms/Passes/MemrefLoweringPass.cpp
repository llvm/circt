//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;
namespace {

class MemrefLoweringPass : public hir::MemrefLoweringBase<MemrefLoweringPass> {
public:
  void runOnOperation() override;
  void dispatchOp(Operation *);
  void addFuncArgs(hir::FuncOp);
  void removeFuncArgs(hir::FuncOp);
  void updateOp(hir::LoadOp);
  void updateOp(hir::StoreOp);
  void updateOp(hir::CallOp);
  void updateOp(hir::AllocaOp);

private:
  llvm::DenseMap<Value, Value> mapMemrefRdAddrSend;
  llvm::DenseMap<Value, Value> mapMemrefRdDataRecv;
  llvm::DenseMap<Value, Value> mapMemrefWrSend;
};
} // end anonymous namespace

Type buildBusTensor(MLIRContext *context, SmallVector<Type> busTypes,
                    SmallVector<PortKind> portKinds, DictionaryAttr proto,
                    int numBanks, ArrayRef<int64_t> bankShape) {
  return mlir::RankedTensorType::get(
      bankShape, hir::BusType::get(context, busTypes, portKinds, proto));
}

bool isReadOnlyMemref(hir::MemrefType memTy) {
  Attribute rdAttr = memTy.getPortAttrs().get("rd");
  Attribute wrAttr = memTy.getPortAttrs().get("wr");
  if (rdAttr && !wrAttr)
    return true;
  return false;
}

bool isWriteOnlyMemref(hir::MemrefType memTy) {
  Attribute rdAttr = memTy.getPortAttrs().get("rd");
  Attribute wrAttr = memTy.getPortAttrs().get("wr");
  if (!rdAttr && wrAttr)
    return true;
  return false;
}

void MemrefLoweringPass::updateOp(hir::AllocaOp op) {
  std::string moduleAttr =
      op.moduleAttr().dyn_cast<StringAttr>().getValue().str();
  if (moduleAttr != "bram" && moduleAttr != "reg")
    return;

  SmallVector<Value, 4> bramCallArgs;

  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  auto results = op.res();

  Type memTyI1 = helper::getIntegerType(context, 1);

  DictionaryAttr protoValid = DictionaryAttr::get(
      context, SmallVector<NamedAttribute>({builder.getNamedAttr(
                   "join", StringAttr::get(context, "join_valid"))}));
  DictionaryAttr protoEmpmemTy =
      DictionaryAttr::get(context, SmallVector<NamedAttribute>({}));

  for (auto res : results) {
    hir::MemrefType memTy = res.getType().dyn_cast<hir::MemrefType>();
    assert(memTy);
    int dataWidth = helper::getBitWidth(memTy.getElementType());
    int numBanks = memTy.getNumBanks();
    auto port = memTy.getPort();
    SmallVector<Type, 4> addrTypes;
    for (auto dimShape : memTy.getAddrShape()) {
      Type addrTy = helper::getIntegerType(context, helper::clog2(dimShape));
      addrTypes.push_back(addrTy);
    }
    Type memTyIAddr = builder.getTupleType(addrTypes);
    Type memTyIData = helper::getIntegerType(context, dataWidth);
    Type memTyIAddrAndData =
        TupleType::get(context, SmallVector<Type>({memTyIAddr, memTyIData}));

    Type memTyRdAddrSend =
        buildBusTensor(context, {memTyI1, memTyIAddr}, {wr, wr}, protoValid,
                       numBanks, memTy.getBankShape());
    Type memTyRdAddrRecv =
        buildBusTensor(context, {memTyI1, memTyIAddr}, {rd, rd}, protoEmpmemTy,
                       numBanks, memTy.getBankShape());

    Type memTyRdDataSend =
        buildBusTensor(context, {memTyIData}, {wr}, protoEmpmemTy, numBanks,
                       memTy.getBankShape());
    Type memTyRdDataRecv =
        buildBusTensor(context, {memTyIData}, {rd}, protoEmpmemTy, numBanks,
                       memTy.getBankShape());

    Type memTyWrSend =
        buildBusTensor(context, {memTyI1, memTyIAddrAndData}, {wr, wr},
                       protoValid, numBanks, memTy.getBankShape());
    Type memTyWrRecv =
        buildBusTensor(context, {memTyI1, memTyIAddrAndData}, {rd, rd},
                       protoEmpmemTy, numBanks, memTy.getBankShape());

    if (port == rd) {
      auto rdAddrBuses =
          builder
              .create<hir::AllocaOp>(
                  op.getLoc(),
                  SmallVector<Type>({memTyRdAddrSend, memTyRdAddrRecv}),
                  builder.getStringAttr("bus"))
              .getResults();
      auto rdDataBuses =
          builder
              .create<hir::AllocaOp>(
                  op.getLoc(),
                  SmallVector<Type>({memTyRdDataSend, memTyRdDataRecv}),
                  builder.getStringAttr("bus"))
              .getResults();

      // push the addr-recv and data-send bus into call args.
      // insert addr-send and data-recv buses into map for LoadOp/StoreOp
      // lowering.

      bramCallArgs.push_back(rdAddrBuses[1]);
      mapMemrefRdAddrSend[res] = rdAddrBuses[0];
      bramCallArgs.push_back(rdDataBuses[0]);
      mapMemrefRdDataRecv[res] = rdDataBuses[1];

    } else if (port == wr) {
      auto wrBuses =
          builder
              .create<hir::AllocaOp>(
                  op.getLoc(), SmallVector<Type>({memTyWrSend, memTyWrRecv}),
                  builder.getStringAttr("bus"))
              .getResults();
      bramCallArgs.push_back(wrBuses[1]);
      mapMemrefWrSend[res] = wrBuses[0];
    } else {
      assert(false && "rw is not supported yet!");
    }
  }

  SmallVector<Type, 4> bramCallArgTypes;
  SmallVector<Attribute> inputDelays;
  for (auto bramCallArg : bramCallArgs) {
    bramCallArgTypes.push_back(bramCallArg.getType());
    inputDelays.push_back(helper::getIntegerAttr(context, 64, 0));
  }

  Value tstart = getOperation().getBody().front().getArguments().back();

  FuncType funcTy = hir::FuncType::get(
      context,
      FunctionType::get(context, bramCallArgTypes, SmallVector<Type>({})),
      ArrayAttr::get(context, inputDelays),
      ArrayAttr::get(context, SmallVector<Attribute>({})));

  builder.create<hir::CallOp>(op.getLoc(), SmallVector<Type>({}),
                              FlatSymbolRefAttr::get(context, moduleAttr),
                              funcTy, Value(), bramCallArgs, tstart, Value());
}

Value createAddrTuple(OpBuilder &builder, Location loc,
                      ArrayRef<Value> indices) {
  if (indices.size() == 0)
    return Value();

  SmallVector<Type, 4> idxTypes;
  for (auto idx : indices)
    idxTypes.push_back(idx.getType());

  return builder
      .create<hir::TupleOp>(loc, builder.getTupleType(idxTypes), indices)
      .getResult();
}

Value createAddrAndDataTuple(OpBuilder &builder, Location loc,
                             ArrayRef<Value> indices, Value data) {

  SmallVector<Value, 4> addrAndDataArray;
  SmallVector<Type, 4> addrAndDataTypes;
  for (auto idx : indices) {
    addrAndDataArray.push_back(idx);
    addrAndDataTypes.push_back(idx.getType());
  }
  addrAndDataArray.push_back(data);
  addrAndDataTypes.push_back(data.getType());
  return builder
      .create<hir::TupleOp>(loc, builder.getTupleType(addrAndDataTypes),
                            addrAndDataArray)
      .getResult();
}

SmallVector<Type, 4> unpackBusTypes(BusType packedBusTy) {
  SmallVector<Type, 4> unpackedBusTypes;
  auto *context = packedBusTy.getContext();
  for (auto typeAndDirection : llvm::zip(packedBusTy.getElementTypes(),
                                         packedBusTy.getElementDirections())) {
    Type ty = std::get<0>(typeAndDirection);
    auto direction = std::get<1>(typeAndDirection);

    unpackedBusTypes.push_back(hir::BusType::get(
        context, SmallVector<Type, 1>({ty}),
        SmallVector<PortKind, 1>({direction}),
        DictionaryAttr::get(context, SmallVector<NamedAttribute, 1>({}))));
  }
  return unpackedBusTypes;
}

void MemrefLoweringPass::updateOp(hir::LoadOp op) {
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  Value mem = op.mem();
  Value c1 =
      builder
          .create<hir::ConstantOp>(op.getLoc(), IndexType::get(context),
                                   helper::getIntegerAttr(context, 64, 1))
          .getResult();
  MemrefType memTy = mem.getType().dyn_cast<MemrefType>();
  Attribute memrefRdDelayAttr = memTy.getPortAttrs().get("rd");

  assert(memrefRdDelayAttr);

  Value cMemrefRdDelay =
      builder
          .create<hir::ConstantOp>(
              op.getLoc(), IndexType::get(context),
              helper::getIntegerAttr(
                  context, 64,
                  memrefRdDelayAttr.dyn_cast<IntegerAttr>().getInt()))
          .getResult();

  Value addrBusTensor = mapMemrefRdAddrSend[mem];
  Value dataBusTensor = mapMemrefRdDataRecv[mem];
  assert(addrBusTensor);
  assert(dataBusTensor);
  SmallVector<Value, 4> bank = op.getBankIdx();

  Value addrBus =
      builder
          .create<hir::TensorExtractOp>(op.getLoc(),
                                        addrBusTensor.getType()
                                            .dyn_cast<mlir::TensorType>()
                                            .getElementType(),
                                        addrBusTensor, bank)
          .getResult();

  SmallVector<Type, 4> unpackedAddrBusTypes =
      unpackBusTypes(addrBus.getType().dyn_cast<hir::BusType>());

  auto unpackedAddrBus =
      builder
          .create<hir::BusUnpackOp>(op.getLoc(), unpackedAddrBusTypes, addrBus)
          .getResults();
  Value dataBus =
      builder
          .create<hir::TensorExtractOp>(op.getLoc(),
                                        dataBusTensor.getType()
                                            .dyn_cast<mlir::TensorType>()
                                            .getElementType(),
                                        dataBusTensor, bank)
          .getResult();

  Value tstart = op.tstart();
  Value offset = op.offset();
  assert(!offset);

  builder.create<hir::SendOp>(op.getLoc(), c1, unpackedAddrBus[0], tstart,
                              Value());

  Value addr = createAddrTuple(builder, op.getLoc(), op.getAddrIdx());

  if (addr) { // bram.
    builder.create<hir::SendOp>(op.getLoc(), addr, unpackedAddrBus[1], tstart,
                                Value());
  }

  Value tstartPlus1 =
      builder
          .create<hir::DelayOp>(op.getLoc(), helper::getTimeType(context),
                                tstart, cMemrefRdDelay, tstart, Value())
          .getResult();
  auto recvOp = builder.create<hir::RecvOp>(op.getLoc(), op.res().getType(),
                                            dataBus, tstartPlus1, Value());
  op.replaceAllUsesWith(recvOp.getOperation());
  op.getOperation()->dropAllReferences();
  op.getOperation()->dropAllUses();
  op.getOperation()->erase();
}

void MemrefLoweringPass::updateOp(hir::StoreOp op) {
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  Value mem = op.mem();
  Value value = op.value();
  Value c1 =
      builder
          .create<hir::ConstantOp>(op.getLoc(), IndexType::get(context),
                                   helper::getIntegerAttr(context, 64, 1))
          .getResult();

  SmallVector<Value, 4> bank = op.getBankIdx();
  Value writeBusTensor = mapMemrefWrSend[mem];
  Value writeBus =
      builder
          .create<hir::TensorExtractOp>(op.getLoc(),
                                        writeBusTensor.getType()
                                            .dyn_cast<mlir::TensorType>()
                                            .getElementType(),
                                        writeBusTensor, bank)
          .getResult();

  SmallVector<Type, 4> unpackedWriteBusTypes =
      unpackBusTypes(writeBus.getType().dyn_cast<hir::BusType>());

  auto unpackedWriteBus = builder
                              .create<hir::BusUnpackOp>(
                                  op.getLoc(), unpackedWriteBusTypes, writeBus)
                              .getResults();

  Value tstart = op.tstart();
  Value offset = op.offset();
  assert(!offset);

  builder.create<hir::SendOp>(op.getLoc(), c1, unpackedWriteBus[0], tstart,
                              Value());
  Value addr =
      createAddrAndDataTuple(builder, op.getLoc(), op.getAddrIdx(), value);
  builder.create<hir::SendOp>(op.getLoc(), addr, unpackedWriteBus[1], tstart,
                              Value());

  op.getOperation()->dropAllReferences();
  op.getOperation()->dropAllUses();
  op.getOperation()->erase();
}

hir::FuncType updateFuncType(OpBuilder &builder, hir::FuncType oldFuncTy,
                             ArrayRef<Value> arguments, ArrayAttr inputDelays) {

  MLIRContext *context = builder.getContext();
  SmallVector<Type, 4> argumentTypes;
  for (Value argument : arguments)
    argumentTypes.push_back(argument.getType());

  FunctionType oldFunctionTy = oldFuncTy.getFunctionType();
  auto resultTypes = oldFunctionTy.getResults();

  FunctionType newFunctionTy =
      builder.getFunctionType(argumentTypes, resultTypes);

  hir::FuncType newFuncTy = hir::FuncType::get(
      context, newFunctionTy, inputDelays, oldFuncTy.getOutputDelays());

  return newFuncTy;
}

void MemrefLoweringPass::updateOp(hir::CallOp op) {
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  auto oldFuncTy = op.funcTy().dyn_cast<hir::FuncType>();

  auto oldInputDelays = oldFuncTy.getInputDelays();
  SmallVector<Attribute, 4> newInputDelays;

  // Calculate an array of new operands, replacing memref with busses.
  SmallVector<Value> newOperands;
  auto operands = op.getOperands();
  for (int i = 0; i < (int)operands.size(); i++) {
    Value operand = operands[i];
    auto memTy = operand.getType().dyn_cast<hir::MemrefType>();
    if (!memTy) {
      newOperands.push_back(operand);
      newInputDelays.push_back(oldInputDelays[i]);
    } else if (memTy.getPort() == rd) {
      Value addrBus = mapMemrefRdAddrSend[operand];
      Value dataBus = mapMemrefRdDataRecv[operand];

      newOperands.push_back(addrBus);
      newOperands.push_back(dataBus);
      newInputDelays.push_back(helper::getIntegerAttr(context, 64, 0));
      newInputDelays.push_back(helper::getIntegerAttr(context, 64, 0));
    } else if (memTy.getPort() == wr) {
      Value writeBus = mapMemrefWrSend[operand];
      newOperands.push_back(writeBus);
    } else {
      assert(false && "We dont yet support rw access");
    }
  }
  // remove the old operands and add the new array of operands.
  op.operandsMutable().clear();
  op.operandsMutable().append(newOperands);

  // update the FuncType.
  auto newFuncTy = updateFuncType(builder, oldFuncTy, op.getOperands(),
                                  ArrayAttr::get(context, newInputDelays));
  op->setAttr("funcTy", TypeAttr::get(newFuncTy));
}

void MemrefLoweringPass::dispatchOp(Operation *operation) {
  if (auto op = dyn_cast<hir::AllocaOp>(operation))
    updateOp(op);
  else if (auto op = dyn_cast<LoadOp>(operation))
    updateOp(op);
  else if (auto op = dyn_cast<StoreOp>(operation))
    updateOp(op);
  else if (auto op = dyn_cast<CallOp>(operation))
    updateOp(op);
}

struct ArgReplacementInfo {
  int argLoc;
  Value originalArg;
  SmallVector<Type, 4> tyReplacementArgs;
};

void MemrefLoweringPass::addFuncArgs(hir::FuncOp op) {
  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);
  Type memTyI1 = helper::getIntegerType(context, 1);
  DictionaryAttr protoValid = DictionaryAttr::get(
      context, SmallVector<NamedAttribute>({builder.getNamedAttr(
                   "join", StringAttr::get(context, "join_valid"))}));
  DictionaryAttr protoEmpmemTy =
      DictionaryAttr::get(context, SmallVector<NamedAttribute>({}));

  SmallVector<ArgReplacementInfo, 4> argReplacementArray;

  for (int i = 0; i < (int)args.size(); i++) {
    Value arg = args[i];
    MemrefType memTy = arg.getType().dyn_cast<hir::MemrefType>();
    if (!memTy)
      continue;

    int dataWidth = helper::getBitWidth(memTy.getElementType());
    int numBanks = memTy.getNumBanks();
    auto port = memTy.getPort();
    SmallVector<Type, 4> addrTypes;
    for (auto dimShape : memTy.getAddrShape()) {
      Type addrTy = helper::getIntegerType(context, helper::clog2(dimShape));
      addrTypes.push_back(addrTy);
    }
    Type memTyIAddr = builder.getTupleType(addrTypes);
    Type memTyIData = helper::getIntegerType(context, dataWidth);
    Type memTyIAddrAndData =
        TupleType::get(context, SmallVector<Type>({memTyIAddr, memTyIData}));

    Type memTyRdAddrSend =
        buildBusTensor(context, {memTyI1, memTyIAddr}, {wr, wr}, protoValid,
                       numBanks, memTy.getBankShape());
    Type memTyRdDataRecv =
        buildBusTensor(context, {memTyIData}, {rd}, protoEmpmemTy, numBanks,
                       memTy.getBankShape());

    Type memTyWrSend =
        buildBusTensor(context, {memTyI1, memTyIAddrAndData}, {wr, wr},
                       protoValid, numBanks, memTy.getBankShape());

    if (port == rd) {
      SmallVector<Type, 4> tyRdBuses;
      tyRdBuses.push_back(memTyRdAddrSend);
      tyRdBuses.push_back(memTyRdDataRecv);
      argReplacementArray.push_back(
          {.argLoc = i, .originalArg = arg, .tyReplacementArgs = tyRdBuses});

    } else if (port == wr) {
      SmallVector<Type, 4> tyWrBuses;
      tyWrBuses.push_back(memTyWrSend);
      argReplacementArray.push_back(
          {.argLoc = i, .originalArg = arg, .tyReplacementArgs = tyWrBuses});

    } else {
      assert(false && "rw is not supported yet!");
    }
  }
  hir::FuncType oldFuncTy = op.funcTy().dyn_cast<hir::FuncType>();
  auto inputDelayAttrs = oldFuncTy.getInputDelays();
  SmallVector<Attribute, 4> updatedInputDelays;

  for (Attribute delay : inputDelayAttrs) {
    updatedInputDelays.push_back(delay);
  }

  // insert the new bus args and put the buses (Value) into the maps.
  for (int i = argReplacementArray.size() - 1; i >= 0; i--) {
    int argLoc = argReplacementArray[i].argLoc;
    Value originalArg = argReplacementArray[i].originalArg;
    ArrayRef<Type> tyReplacementArgs = argReplacementArray[i].tyReplacementArgs;
    if (originalArg.getType().dyn_cast<MemrefType>().getPort() == rd) {
      assert(tyReplacementArgs.size() == 2);
      Type tyRdAddrSend = tyReplacementArgs[0];
      Type tyRdDataRecv = tyReplacementArgs[1];

      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      mapMemrefRdAddrSend[originalArg] =
          entryBlock.insertArgument(argLoc++, tyRdAddrSend);

      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      mapMemrefRdDataRecv[originalArg] =
          entryBlock.insertArgument(argLoc++, tyRdDataRecv);

    } else if (originalArg.getType().dyn_cast<MemrefType>().getPort() == wr) {
      assert(tyReplacementArgs.size() == 1);
      Type tyWrSend = tyReplacementArgs[0];
      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      mapMemrefWrSend[originalArg] =
          entryBlock.insertArgument(argLoc++, tyWrSend);
    } else {
      assert(false && "rw is not supported yet!");
    }
  }

  auto newFuncTy = updateFuncType(builder, oldFuncTy, op.getOperands(),
                                  ArrayAttr::get(context, updatedInputDelays));
  op.setType(newFuncTy.getFunctionType());
  op->setAttr("funcTy", TypeAttr::get(newFuncTy));
}

void MemrefLoweringPass::removeFuncArgs(hir::FuncOp op) {

  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  hir::FuncType oldFuncTy = op.funcTy().dyn_cast<hir::FuncType>();
  auto inputDelayAttrs = oldFuncTy.getInputDelays();
  SmallVector<Attribute, 4> updatedInputDelays;

  for (Attribute delay : inputDelayAttrs) {
    updatedInputDelays.push_back(delay);
  }

  for (int i = args.size() - 1; i >= 0; i--) {
    Value arg = args[i];
    MemrefType memTy = arg.getType().dyn_cast<hir::MemrefType>();
    if (!memTy)
      continue;
    entryBlock.eraseArgument(i);
    updatedInputDelays.erase(updatedInputDelays.begin() + i);
  }
  auto newFuncTy = updateFuncType(builder, oldFuncTy, op.getOperands(),
                                  ArrayAttr::get(context, updatedInputDelays));
  op.setType(newFuncTy.getFunctionType());
  op->setAttr("funcTy", TypeAttr::get(newFuncTy));
}

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();

  addFuncArgs(funcOp);
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    dispatchOp(operation);
    return WalkResult::advance();
  });
  removeFuncArgs(funcOp);

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createMemrefLoweringPass() {
  return std::make_unique<MemrefLoweringPass>();
}
} // namespace hir
} // namespace circt
