//=========- MemrefLoweringPass.cpp - Lower memref type---===//
//
// This file implements lowering pass for memref type.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/helper.h"

using namespace mlir;
using namespace hir;
namespace {

class MemrefLoweringPass : public hir::MemrefLoweringBase<MemrefLoweringPass> {
public:
  void runOnOperation() override;
  void dispatchOp(Operation *);
  void addFuncArgs(hir::FuncOp);
  void removeFuncArgs(hir::FuncOp);
  void lowerOp(hir::ConstantOp);
  void lowerOp(hir::ForOp);
  void lowerOp(hir::UnrollForOp) {
    assert(false && "This should already be unrolled");
  }
  void lowerOp(hir::AddOp);
  void lowerOp(hir::SubtractOp);
  void lowerOp(hir::LoadOp);
  void lowerOp(hir::StoreOp);
  void lowerOp(hir::ReturnOp);
  void lowerOp(hir::YieldOp);
  void lowerOp(hir::SendOp);
  void lowerOp(hir::RecvOp);
  void lowerOp(hir::DelayOp);
  void lowerOp(hir::CallOp);
  void lowerOp(hir::AllocaOp);

private:
  llvm::DenseMap<Value, Value> mapMemrefRdAddrSend;
  llvm::DenseMap<Value, Value> mapMemrefRdDataRecv;
  llvm::DenseMap<Value, Value> mapMemrefWrSend;
};
} // end anonymous namespace

Type buildBusTensor(MLIRContext *context, SmallVector<Type> busTypes,
                    SmallVector<PortKind> portKinds, DictionaryAttr proto,
                    int numBanks, ArrayRef<int64_t> bankShape) {
  Type tyRdAddrSend = hir::BusType::get(context, busTypes, portKinds, proto);
  if (numBanks > 1)
    tyRdAddrSend = RankedTensorType::get(bankShape, tyRdAddrSend);
  return tyRdAddrSend;
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

void MemrefLoweringPass::lowerOp(hir::AllocaOp op) {
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
    for (auto dimShape : memTy.getPackedShape()) {
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

Value createAddrTuple(OpBuilder &builder, Location loc, ArrayRef<Value> indices,
                      ArrayRef<int64_t> shape) {
  assert(indices.size() == shape.size());
  if (indices.size() == 0)
    return Value();

  Value addr;
  MLIRContext *context = builder.getContext();

  SmallVector<Type, 4> castedIdxTypes;
  SmallVector<Value, 4> castedIdx;
  for (int i = 0; i < (int)shape.size(); i++) {
    auto dimShape = shape[i];
    auto idx = builder.create<hir::CastOp>(
        loc, helper::getIntegerType(context, helper::clog2(dimShape)),
        indices[i]);
    castedIdx.push_back(idx);
    castedIdxTypes.push_back(idx.getType());
  }
  addr = builder
             .create<hir::TupleOp>(loc, builder.getTupleType(castedIdxTypes),
                                   castedIdx)
             .getResult();
  return addr;
}

Value createAddrAndDataTuple(OpBuilder &builder, Location loc,
                             ArrayRef<Value> indices, ArrayRef<int64_t> shape,
                             Value data) {
  Value addrAndData;
  MLIRContext *context = builder.getContext();

  SmallVector<Type, 4> castedIdxTypes;
  SmallVector<Value, 4> castedIdx;
  for (int i = 0; i < (int)shape.size(); i++) {
    auto dimShape = shape[i];
    auto idx = builder.create<hir::CastOp>(
        loc, helper::getIntegerType(context, helper::clog2(dimShape)),
        indices[i]);
    castedIdx.push_back(idx);
    castedIdxTypes.push_back(idx.getType());
  }
  castedIdx.push_back(data);
  castedIdxTypes.push_back(data.getType());
  addrAndData = builder
                    .create<hir::TupleOp>(
                        loc, builder.getTupleType(castedIdxTypes), castedIdx)
                    .getResult();

  return addrAndData;
}

void MemrefLoweringPass::lowerOp(hir::LoadOp op) {
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  Value mem = op.mem();
  Value c0 = builder
                 .create<hir::ConstantOp>(
                     op.getLoc(), helper::getConstIntType(context),
                     helper::getIntegerAttr(context, 64, 0))
                 .getResult();
  Value c1 = builder
                 .create<hir::ConstantOp>(
                     op.getLoc(), helper::getConstIntType(context),
                     helper::getIntegerAttr(context, 64, 1))
                 .getResult();
  MemrefType memTy = mem.getType().dyn_cast<MemrefType>();
  Attribute memrefRdDelayAttr = memTy.getPortAttrs().get("rd");

  assert(memrefRdDelayAttr);

  Value cMemrefRdDelay =
      builder
          .create<hir::ConstantOp>(
              op.getLoc(), helper::getConstIntType(context),
              helper::getIntegerAttr(
                  context, 64,
                  memrefRdDelayAttr.dyn_cast<IntegerAttr>().getInt()))
          .getResult();

  Value addrBus = mapMemrefRdAddrSend[mem];
  Value dataBus = mapMemrefRdDataRecv[mem];
  assert(addrBus);
  assert(dataBus);
  Value tstart = op.tstart();
  Value offset = op.offset();
  assert(!offset);
  SmallVector<Value, 4> bankAddr = op.getBankedIdx();
  bankAddr.push_back(c0);
  builder.create<hir::SendOp>(op.getLoc(), c1, addrBus, bankAddr, tstart,
                              Value());
  bankAddr.pop_back();

  SmallVector<Value, 4> packedIdx = op.getPackedIdx();

  Value packedAddr = createAddrTuple(builder, op.getLoc(), op.getPackedIdx(),
                                     memTy.getPackedShape());

  if (packedAddr) { // bram.
    bankAddr.push_back(c1);
    builder.create<hir::SendOp>(op.getLoc(), packedAddr, addrBus, bankAddr,
                                tstart, Value());
    bankAddr.pop_back();
  }

  bankAddr.push_back(c0);
  Value tstartPlus1 =
      builder
          .create<hir::DelayOp>(op.getLoc(), helper::getTimeType(context),
                                tstart, cMemrefRdDelay, tstart, Value())
          .getResult();
  auto recvOp = builder.create<hir::RecvOp>(
      op.getLoc(), op.res().getType(), dataBus, bankAddr, tstartPlus1, Value());
  op.replaceAllUsesWith(recvOp.getOperation());
  op.getOperation()->dropAllReferences();
  op.getOperation()->dropAllUses();
  op.getOperation()->erase();
}

void MemrefLoweringPass::lowerOp(hir::StoreOp op) {
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  Value mem = op.mem();
  Value value = op.value();
  Value c0 = builder
                 .create<hir::ConstantOp>(
                     op.getLoc(), helper::getConstIntType(context),
                     helper::getIntegerAttr(context, 64, 0))
                 .getResult();
  Value c1 = builder
                 .create<hir::ConstantOp>(
                     op.getLoc(), helper::getConstIntType(context),
                     helper::getIntegerAttr(context, 64, 1))
                 .getResult();
  MemrefType memTy = mem.getType().dyn_cast<MemrefType>();
  Value wrBus = mapMemrefWrSend[mem];
  Value tstart = op.tstart();
  Value offset = op.offset();
  assert(!offset);
  SmallVector<Value, 4> bankAddr = op.getBankedIdx();
  bankAddr.push_back(c0);
  builder.create<hir::SendOp>(op.getLoc(), c1, wrBus, bankAddr, tstart,
                              Value());
  bankAddr.pop_back();
  Value packedAddr = createAddrAndDataTuple(
      builder, op.getLoc(), op.getPackedIdx(), memTy.getPackedShape(), value);
  bankAddr.push_back(c1);
  builder.create<hir::SendOp>(op.getLoc(), packedAddr, wrBus, bankAddr, tstart,
                              Value());
  op.getOperation()->dropAllReferences();
  op.getOperation()->dropAllUses();
  op.getOperation()->erase();
}

void MemrefLoweringPass::dispatchOp(Operation *operation) {
  if (auto op = dyn_cast<hir::AllocaOp>(operation))
    lowerOp(op);
  else if (auto op = dyn_cast<LoadOp>(operation))
    lowerOp(op);
  else if (auto op = dyn_cast<StoreOp>(operation))
    lowerOp(op);
}

struct ArgReplacementInfo {
  int argLoc;
  Value originalArg;
  SmallVector<Type, 4> tyReplacementArgs;
};

void updateFuncOpTypes(hir::FuncOp op, OpBuilder &builder,
                       ArrayAttr inputDelays, ArrayAttr outputDelays) {
  MLIRContext *context = builder.getContext();
  SmallVector<Type, 4> argumentTypes;
  auto argumentAndTimeTypes = op.getArgumentTypes();
  for (int i = 0; i < (int)argumentAndTimeTypes.size() - 1; i++) {
    argumentTypes.push_back(argumentAndTimeTypes[i]);
  }

  auto resultTypes = op.type().dyn_cast<FunctionType>().getResults();

  FunctionType updatedFunctionTy =
      builder.getFunctionType(argumentTypes, resultTypes);

  hir::FuncType funcTy =
      hir::FuncType::get(context, updatedFunctionTy, inputDelays, outputDelays);

  op.setType(updatedFunctionTy);
  op->setAttr("funcTy", TypeAttr::get(funcTy));
}

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
    for (auto dimShape : memTy.getPackedShape()) {
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
      SmallVector<Type, 4> tyRdBuses;
      tyRdBuses.push_back(memTyRdAddrSend);
      tyRdBuses.push_back(memTyRdAddrRecv);
      tyRdBuses.push_back(memTyRdDataSend);
      tyRdBuses.push_back(memTyRdDataRecv);
      argReplacementArray.push_back(
          {.argLoc = i, .originalArg = arg, .tyReplacementArgs = tyRdBuses});

    } else if (port == wr) {
      SmallVector<Type, 4> tyWrBuses;
      tyWrBuses.push_back(memTyWrSend);
      tyWrBuses.push_back(memTyWrRecv);
      argReplacementArray.push_back(
          {.argLoc = i, .originalArg = arg, .tyReplacementArgs = tyWrBuses});

    } else {
      assert(false && "rw is not supported yet!");
    }
  }

  auto inputDelayAttrs = op.funcTy().dyn_cast<hir::FuncType>().getInputDelays();
  auto outputDelayAttrs =
      op.funcTy().dyn_cast<hir::FuncType>().getOutputDelays();
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
      assert(tyReplacementArgs.size() == 4);
      Type tyRdAddrSend = tyReplacementArgs[0];
      Type tyRdAddrRecv = tyReplacementArgs[1];
      Type tyRdDataSend = tyReplacementArgs[2];
      Type tyRdDataRecv = tyReplacementArgs[3];

      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      mapMemrefRdAddrSend[originalArg] =
          entryBlock.insertArgument(argLoc++, tyRdAddrSend);

      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      entryBlock.insertArgument(argLoc++, tyRdAddrRecv);

      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      entryBlock.insertArgument(argLoc++, tyRdDataSend);

      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      mapMemrefRdDataRecv[originalArg] =
          entryBlock.insertArgument(argLoc++, tyRdDataRecv);

    } else if (originalArg.getType().dyn_cast<MemrefType>().getPort() == wr) {

      Type tyWrSend = tyReplacementArgs[0];
      Type tyWrRecv = tyReplacementArgs[1];
      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      mapMemrefWrSend[originalArg] =
          entryBlock.insertArgument(argLoc++, tyWrSend);
      updatedInputDelays.insert(updatedInputDelays.begin() + argLoc,
                                helper::getIntegerAttr(context, 64, 0));
      entryBlock.insertArgument(argLoc++, tyWrRecv);
    } else {
      assert(false && "rw is not supported yet!");
    }
  }

  updateFuncOpTypes(op, builder, ArrayAttr::get(context, updatedInputDelays),
                    outputDelayAttrs);
}

void MemrefLoweringPass::removeFuncArgs(hir::FuncOp op) {

  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  auto inputDelayAttrs = op.funcTy().dyn_cast<hir::FuncType>().getInputDelays();
  auto outputDelayAttrs =
      op.funcTy().dyn_cast<hir::FuncType>().getOutputDelays();
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
  updateFuncOpTypes(op, builder, ArrayAttr::get(context, updatedInputDelays),
                    outputDelayAttrs);
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

namespace mlir {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createMemrefLoweringPass() {
  return std::make_unique<MemrefLoweringPass>();
}
} // namespace hir
} // namespace mlir
