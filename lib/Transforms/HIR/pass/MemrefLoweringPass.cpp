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
  void lowerOp(hir::FuncOp op);
  void lowerOp(hir::ConstantOp op);
  void lowerOp(hir::ForOp op);
  void lowerOp(hir::UnrollForOp op);
  void lowerOp(hir::AddOp op);
  void lowerOp(hir::SubtractOp op);
  void lowerOp(hir::LoadOp op);
  void lowerOp(hir::StoreOp op);
  void lowerOp(hir::ReturnOp op);
  void lowerOp(hir::YieldOp op);
  void lowerOp(hir::SendOp op);
  void lowerOp(hir::RecvOp op);
  void lowerOp(hir::DelayOp op);
  void lowerOp(hir::CallOp op);
  void lowerOp(hir::AllocaOp op);

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

  Type tyI1 = helper::getIntegerType(context, 1);

  DictionaryAttr protoValid = DictionaryAttr::get(
      context, SmallVector<NamedAttribute>({builder.getNamedAttr(
                   "join", StringAttr::get(context, "join_valid"))}));
  DictionaryAttr protoEmpty =
      DictionaryAttr::get(context, SmallVector<NamedAttribute>({}));

  for (auto res : results) {
    hir::MemrefType ty = res.getType().dyn_cast<hir::MemrefType>();
    assert(ty);
    int dataWidth = helper::getBitWidth(ty.getElementType());
    int numBanks = ty.getNumBanks();
    auto port = ty.getPort();
    SmallVector<Type, 4> addrTypes;
    for (auto dimShape : ty.getPackedShape()) {
      Type addrTy = helper::getIntegerType(context, helper::clog2(dimShape));
      addrTypes.push_back(addrTy);
    }
    Type tyIAddr = builder.getTupleType(addrTypes);
    Type tyIData = helper::getIntegerType(context, dataWidth);
    Type tyIAddrAndData =
        TupleType::get(context, SmallVector<Type>({tyIAddr, tyIData}));

    Type tyRdAddrSend = buildBusTensor(context, {tyI1, tyIAddr}, {wr, wr},
                                       protoValid, numBanks, ty.getBankShape());
    Type tyRdAddrRecv = buildBusTensor(context, {tyI1, tyIAddr}, {rd, rd},
                                       protoEmpty, numBanks, ty.getBankShape());

    Type tyRdDataSend = buildBusTensor(context, {tyIData}, {wr}, protoEmpty,
                                       numBanks, ty.getBankShape());
    Type tyRdDataRecv = buildBusTensor(context, {tyIData}, {rd}, protoEmpty,
                                       numBanks, ty.getBankShape());

    Type tyWrSend = buildBusTensor(context, {tyI1, tyIAddrAndData}, {wr, wr},
                                   protoValid, numBanks, ty.getBankShape());
    Type tyWrRecv = buildBusTensor(context, {tyI1, tyIAddrAndData}, {rd, rd},
                                   protoEmpty, numBanks, ty.getBankShape());

    if (port == rd) {
      auto rdAddrBuses =
          builder
              .create<hir::AllocaOp>(
                  op.getLoc(), SmallVector<Type>({tyRdAddrSend, tyRdAddrRecv}),
                  builder.getStringAttr("bus"))
              .getResults();
      auto rdDataBuses =
          builder
              .create<hir::AllocaOp>(
                  op.getLoc(), SmallVector<Type>({tyRdDataSend, tyRdDataRecv}),
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
              .create<hir::AllocaOp>(op.getLoc(),
                                     SmallVector<Type>({tyWrSend, tyWrRecv}),
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

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult result = funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::AllocaOp>(operation))
      lowerOp(op);
    else if (auto op = dyn_cast<LoadOp>(operation))
      lowerOp(op);
    else if (auto op = dyn_cast<StoreOp>(operation))
      lowerOp(op);
    return WalkResult::advance();
  });

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
