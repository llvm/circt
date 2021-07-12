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
  void updateOp(hir::FuncOp);
  void updateOp(hir::LoadOp);
  void updateOp(hir::StoreOp);
  void updateOp(hir::CallOp);
  void updateOp(hir::AllocaOp);

private:
  llvm::DenseMap<Value, Value> mapMemref2RdAddrBusTensor;
  llvm::DenseMap<Value, Value> mapMemref2RdDataBusTensor;
  llvm::DenseMap<Value, Value> mapMemref2WrBusTensor;
};
} // end anonymous namespace

namespace {
//------------------------------------------------------------------------------
// Helpers.
//------------------------------------------------------------------------------
enum BusPortKind { SEND = 0, RECV = 1 };
enum MemPortKind { RD = 0, WR = 1, RW = 2 };

struct ArgReplacementInfo {
  MemPortKind portKind;
  int argLoc;
  Value originalArg;
  SmallVector<Type, 4> tyReplacementArgs;
  SmallVector<DictionaryAttr, 4> attrReplacementArgs;
};

mlir::TensorType buildBusTensor(MLIRContext *context, ArrayRef<int64_t> shape,
                                SmallVector<Type> busTypes) {
  SmallVector<BusDirection> busDirections;
  // Only same direction is allowed inside a tensor<bus>.
  busDirections.append(busTypes.size(), BusDirection::SAME);

  return mlir::RankedTensorType::get(
      shape, hir::BusType::get(context, busTypes, busDirections));
}

bool isReadOnly(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  assert(portDict);
  if (portDict.getNamed("rd_latency") && !portDict.getNamed("wr_latency"))
    return true;
  return false;
}

bool isWriteOnly(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  assert(portDict);
  if (portDict.getNamed("wr_latency") && !portDict.getNamed("rd_latency"))
    return true;
  return false;
}

Value createAddrTuple(OpBuilder &builder, Location loc,
                      ArrayRef<Value> indices) {
  if (indices.size() == 0)
    return Value();

  SmallVector<Type, 4> idxTypes;
  for (auto idx : indices)
    idxTypes.push_back(idx.getType());

  return builder
      .create<hir::CreateTupleOp>(loc, builder.getTupleType(idxTypes), indices)
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
      .create<hir::CreateTupleOp>(loc, builder.getTupleType(addrAndDataTypes),
                                  addrAndDataArray)
      .getResult();
}

hir::FuncType getFuncType(OpBuilder &builder, hir::FuncType oldFuncTy,
                          ArrayRef<Value> inputs,
                          ArrayRef<DictionaryAttr> inputAttrs) {

  MLIRContext *context = builder.getContext();
  SmallVector<Type, 4> inputTypes;
  for (Value input : inputs)
    inputTypes.push_back(input.getType());

  hir::FuncType newFuncTy = hir::FuncType::get(context, inputTypes, inputAttrs,
                                               oldFuncTy.getResultTypes(),
                                               oldFuncTy.getResultAttrs());

  return newFuncTy;
}

} // namespace
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Functions to replace defs and uses of hir.memref.
//------------------------------------------------------------------------------
void MemrefLoweringPass::updateOp(hir::FuncOp op) {
  Block &entryBlock = op.getFuncBody().front();
  auto args = entryBlock.getArguments();
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);
  Type memTyI1 = IntegerType::get(context, 1);

  SmallVector<ArgReplacementInfo, 4> argReplacementArray;

  auto inputAttrs = op.funcTy().dyn_cast<hir::FuncType>().getInputAttrs();
  for (int i = 0; i < (int)args.size(); i++) {
    Value arg = args[i];
    MemrefType memTy = arg.getType().dyn_cast<hir::MemrefType>();
    if (!memTy)
      continue;

    int dataWidth = helper::getBitWidth(memTy.getElementType());
    ArrayAttr ports =
        inputAttrs[i].getNamed("ports").getValue().second.dyn_cast<ArrayAttr>();
    // We do not support multiple ports at function argument.
    assert(ports.size() == 1);
    auto port = ports[0];
    SmallVector<Type, 4> addrTypes;
    for (auto dimShape : memTy.filterShape(ADDR)) {
      Type addrTy = IntegerType::get(context, helper::clog2(dimShape));
      addrTypes.push_back(addrTy);
    }

    Type memTyIAddr = builder.getTupleType(addrTypes);
    Type memTyIData = IntegerType::get(context, dataWidth);
    Type memTyIAddrAndData =
        TupleType::get(context, SmallVector<Type>({memTyIAddr, memTyIData}));

    Type rdAddrBusTy =
        buildBusTensor(context, memTy.filterShape(BANK), {memTyI1, memTyIAddr});
    Type rdDataBusTy =
        buildBusTensor(context, memTy.filterShape(BANK), {memTyIData});

    Type wrBusTy = buildBusTensor(context, memTy.filterShape(BANK),
                                  {memTyI1, memTyIAddrAndData});

    if (isReadOnly(port)) {
      argReplacementArray.push_back(
          {.portKind = RD,
           .argLoc = i,
           .originalArg = arg,
           .tyReplacementArgs = {rdAddrBusTy, rdDataBusTy},
           .attrReplacementArgs = {
               helper::getDictionaryAttr(builder, "ports",
                                         StringAttr::get(context, "send")),
               helper::getDictionaryAttr(builder, "ports",
                                         StringAttr::get(context, "recv"))}});

    } else if (isWriteOnly(port)) {
      argReplacementArray.push_back(
          {.portKind = WR,
           .argLoc = i,
           .originalArg = arg,
           .tyReplacementArgs = {wrBusTy},
           .attrReplacementArgs = {helper::getDictionaryAttr(
               builder, "ports", StringAttr::get(context, "send"))}});
    } else {
      assert(false && "rw is not supported yet!");
    }
  }
  hir::FuncType oldFuncTy = op.funcTy().dyn_cast<hir::FuncType>();
  SmallVector<DictionaryAttr, 4> updatedInputAttrs;

  for (auto attr : inputAttrs) {
    updatedInputAttrs.push_back(attr);
  }

  // insert the new bus args and put the buses (Value) into the maps.
  for (int i = argReplacementArray.size() - 1; i >= 0; i--) {
    int argLoc = argReplacementArray[i].argLoc;
    Value originalArg = argReplacementArray[i].originalArg;
    ArrayRef<Type> tyReplacementArgs = argReplacementArray[i].tyReplacementArgs;
    ArrayRef<DictionaryAttr> attrReplacementArgs =
        argReplacementArray[i].attrReplacementArgs;

    // Erase the original memref argument.
    updatedInputAttrs.erase(updatedInputAttrs.begin() + argLoc);
    entryBlock.eraseArgument(argLoc);

    if (argReplacementArray[i].portKind == RD) {
      assert(tyReplacementArgs.size() == 2);
      Type rdAddrBusTy = tyReplacementArgs[0];
      Type rdDataBusTy = tyReplacementArgs[1];

      // Insert the attr and type for addr bus.
      updatedInputAttrs.insert(updatedInputAttrs.begin() + argLoc,
                               attrReplacementArgs[0]);
      mapMemref2RdAddrBusTensor[originalArg] =
          entryBlock.insertArgument(argLoc++, rdAddrBusTy);

      // Insert the attr and type for data bus.
      updatedInputAttrs.insert(updatedInputAttrs.begin() + argLoc,
                               attrReplacementArgs[1]);
      mapMemref2RdDataBusTensor[originalArg] =
          entryBlock.insertArgument(argLoc++, rdDataBusTy);

    } else if (argReplacementArray[i].portKind == WR) {
      assert(tyReplacementArgs.size() == 1);
      Type wrBusTy = tyReplacementArgs[0];

      // Insert the attr and type for write bus.
      updatedInputAttrs.insert(updatedInputAttrs.begin() + argLoc,
                               attrReplacementArgs[0]);
      mapMemref2WrBusTensor[originalArg] =
          entryBlock.insertArgument(argLoc++, wrBusTy);
    } else {
      assert(false && "rw is not supported yet!");
    }
  }

  auto newFuncTy =
      getFuncType(builder, oldFuncTy, op.getOperands(), updatedInputAttrs);
  op.setType(newFuncTy.getFunctionType());
  op->setAttr("funcTy", TypeAttr::get(newFuncTy));
}

void MemrefLoweringPass::updateOp(hir::AllocaOp op) {
  hir::FuncOp funcOp = getOperation();
  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  MLIRContext *context = builder.getContext();
  builder.setInsertionPoint(op);

  auto res = op.res();

  Type memTyI1 = IntegerType::get(context, 1);

  ArrayAttr ports = op.ports();
  SmallVector<Value, 4> bramCallArgs;
  SmallVector<BusPortKind, 4> bramCallBusPorts;
  for (auto port : ports) {
    hir::MemrefType memTy = res.getType().dyn_cast<hir::MemrefType>();
    assert(memTy);
    SmallVector<Type, 4> addrTypes;

    for (auto dimShape : memTy.filterShape(ADDR))
      addrTypes.push_back(IntegerType::get(context, helper::clog2(dimShape)));

    Type memTyIAddr = builder.getTupleType(addrTypes);
    Type memTyIData =
        IntegerType::get(context, helper::getBitWidth(memTy.getElementType()));
    Type memTyIAddrAndData =
        TupleType::get(context, SmallVector<Type>({memTyIAddr, memTyIData}));

    auto rdAddrBusTy =
        buildBusTensor(context, memTy.filterShape(BANK), {memTyI1, memTyIAddr});

    Type rdDataBusTy =
        buildBusTensor(context, memTy.filterShape(BANK), {memTyIData});

    Type wrBusTy = buildBusTensor(context, memTy.filterShape(BANK),
                                  {memTyI1, memTyIAddrAndData});

    if (isReadOnly(port)) {
      auto rdAddrBus =
          builder.create<hir::BusInstantiateOp>(op.getLoc(), rdAddrBusTy)
              .getResult();
      auto rdDataBus =
          builder.create<hir::BusInstantiateOp>(op.getLoc(), rdDataBusTy)
              .getResult();

      // push the addr and data bus into call args.
      // insert addr and data buses into map for LoadOp/StoreOp
      // lowering.

      bramCallArgs.push_back(rdAddrBus);
      bramCallArgs.push_back(rdDataBus);
      bramCallBusPorts.push_back(RECV);
      bramCallBusPorts.push_back(SEND);
      mapMemref2RdAddrBusTensor[res] = rdAddrBus;
      mapMemref2RdDataBusTensor[res] = rdDataBus;

    } else if (isWriteOnly(port)) {
      auto wrBus = builder.create<hir::BusInstantiateOp>(op.getLoc(), wrBusTy)
                       .getResult();
      bramCallArgs.push_back(wrBus);
      bramCallBusPorts.push_back(RECV);
      mapMemref2WrBusTensor[res] = wrBus;
    } else {
      assert(false && "read-write ports are not yet supported!");
    }
  }

  SmallVector<Type, 4> bramCallArgTypes;
  SmallVector<DictionaryAttr> inputAttrsbramCallArgAttrs;
  for (size_t i = 0; i < bramCallArgs.size(); i++) {
    auto bramCallArg = bramCallArgs[i];
    bramCallArgTypes.push_back(bramCallArg.getType());
    if (bramCallBusPorts[i] == SEND)
      inputAttrsbramCallArgAttrs.push_back(helper::getDictionaryAttr(
          builder, "ports", StringAttr::get(context, "send")));
    else
      inputAttrsbramCallArgAttrs.push_back(helper::getDictionaryAttr(
          builder, "ports", StringAttr::get(context, "recv")));
  }

  Value tstart = funcOp.getFuncBody().front().getArguments().back();

  FuncType funcTy = hir::FuncType::get(
      context, bramCallArgTypes, inputAttrsbramCallArgAttrs,
      SmallVector<Type>({}), SmallVector<DictionaryAttr>({}));

  std::string moduleAttr =
      op.moduleAttr().dyn_cast<StringAttr>().getValue().str();
  builder.create<hir::CallOp>(op.getLoc(), SmallVector<Type>({}),
                              FlatSymbolRefAttr::get(context, moduleAttr),
                              funcTy, Value(), bramCallArgs, tstart,
                              IntegerAttr());
}

void MemrefLoweringPass::updateOp(hir::LoadOp op) {
  //  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  //  MLIRContext *context = builder.getContext();
  //  builder.setInsertionPoint(op);
  //
  //  Value mem = op.mem();
  //  Value c1 =
  //      builder
  //          .create<hir::ConstantOp>(op.getLoc(), IndexType::get(context),
  //                                   helper::getIntegerAttr(context, 1))
  //          .getResult();
  //  MemrefType memTy = mem.getType().dyn_cast<MemrefType>();
  //  Attribute memrefRdDelayAttr = memTy.getPortAttrs().get("rd");
  //
  //  assert(memrefRdDelayAttr);
  //
  //  Value cMemrefRdDelay =
  //      builder
  //          .create<hir::ConstantOp>(
  //              op.getLoc(), IndexType::get(context),
  //              helper::getIntegerAttr(
  //                  context,
  //                  memrefRdDelayAttr.dyn_cast<IntegerAttr>().getInt()))
  //          .getResult();
  //
  //  Value addrBusTensor = mapMemref2RdAddrBusTensor[mem];
  //  Value dataBusTensor = mapMemref2RdDataBusTensor[mem];
  //  assert(addrBusTensor);
  //  assert(dataBusTensor);
  //  SmallVector<Value, 4> bank = op.getBankIndices();
  //
  //  Value addrBus =
  //      builder
  //          .create<hir::TensorExtractOp>(op.getLoc(),
  //                                        addrBusTensor.getType()
  //                                            .dyn_cast<mlir::TensorType>()
  //                                            .getElementType(),
  //                                        addrBusTensor, bank)
  //          .getResult();
  //
  //  SmallVector<Type, 4> unpackedAddrBusTypes =
  //      unpackBusTypes(addrBus.getType().dyn_cast<hir::BusType>());
  //
  //  auto unpackedAddrBus =
  //      builder
  //          .create<hir::BusUnpackOp>(op.getLoc(), unpackedAddrBusTypes,
  //          addrBus) .getResults();
  //  Value dataBus =
  //      builder
  //          .create<hir::TensorExtractOp>(op.getLoc(),
  //                                        dataBusTensor.getType()
  //                                            .dyn_cast<mlir::TensorType>()
  //                                            .getElementType(),
  //                                        dataBusTensor, bank)
  //          .getResult();
  //
  //  Value tstart = op.tstart();
  //  Value offset = op.offset();
  //  assert(!offset);
  //
  //  builder.create<hir::SendOp>(op.getLoc(), c1, unpackedAddrBus[0], tstart,
  //                              Value());
  //
  //  Value addr = createAddrTuple(builder, op.getLoc(), op.getAddrIndices());
  //
  //  if (addr) { // bram.
  //    builder.create<hir::SendOp>(op.getLoc(), addr, unpackedAddrBus[1],
  //    tstart,
  //                                Value());
  //  }
  //
  //  Value tstartPlus1 =
  //      builder
  //          .create<hir::DelayOp>(op.getLoc(), helper::getTimeType(context),
  //                                tstart, cMemrefRdDelay, tstart, Value())
  //          .getResult();
  //  auto recvOp = builder.create<hir::RecvOp>(op.getLoc(), op.res().getType(),
  //                                            dataBus, tstartPlus1, Value());
  //  op.replaceAllUsesWith(recvOp.getOperation());
  //  op.getOperation()->dropAllReferences();
  //  op.getOperation()->dropAllUses();
  //  op.getOperation()->erase();
}

void MemrefLoweringPass::updateOp(hir::StoreOp op) {
  //  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  //  MLIRContext *context = builder.getContext();
  //  builder.setInsertionPoint(op);
  //
  //  Value mem = op.mem();
  //  Value value = op.value();
  //  Value c1 =
  //      builder
  //          .create<hir::ConstantOp>(op.getLoc(), IndexType::get(context),
  //                                   helper::getIntegerAttr(context,  1))
  //          .getResult();
  //
  //  SmallVector<Value, 4> bank = op.getBankIdx();
  //  Value writeBusTensor = mapMemref2WrBusTensor[mem];
  //  Value writeBus =
  //      builder
  //          .create<hir::TensorExtractOp>(op.getLoc(),
  //                                        writeBusTensor.getType()
  //                                            .dyn_cast<mlir::TensorType>()
  //                                            .getElementType(),
  //                                        writeBusTensor, bank)
  //          .getResult();
  //
  //  SmallVector<Type, 4> unpackedWriteBusTypes =
  //      unpackBusTypes(writeBus.getType().dyn_cast<hir::BusType>());
  //
  //  auto unpackedWriteBus = builder
  //                              .create<hir::BusUnpackOp>(
  //                                  op.getLoc(), unpackedWriteBusTypes,
  //                                  writeBus)
  //                              .getResults();
  //
  //  Value tstart = op.tstart();
  //  Value offset = op.offset();
  //  assert(!offset);
  //
  //  builder.create<hir::SendOp>(op.getLoc(), c1, unpackedWriteBus[0], tstart,
  //                              Value());
  //  Value addr =
  //      createAddrAndDataTuple(builder, op.getLoc(), op.getAddrIdx(), value);
  //  builder.create<hir::SendOp>(op.getLoc(), addr, unpackedWriteBus[1],
  //  tstart,
  //                              Value());
  //
  //  op.getOperation()->dropAllReferences();
  //  op.getOperation()->dropAllUses();
  //  op.getOperation()->erase();
}

void MemrefLoweringPass::updateOp(hir::CallOp op) {
  //  mlir::OpBuilder builder(op.getOperation()->getParentOp()->getContext());
  //  MLIRContext *context = builder.getContext();
  //  builder.setInsertionPoint(op);
  //
  //  auto oldFuncTy = op.funcTy().dyn_cast<hir::FuncType>();
  //
  //  auto oldInputDelays = oldFuncTy.getInputDelays();
  //  SmallVector<Attribute, 4> newInputDelays;
  //
  //  // Calculate an array of new operands, replacing memref with busses.
  //  SmallVector<Value> newOperands;
  //  auto operands = op.getOperands();
  //  for (int i = 0; i < (int)operands.size(); i++) {
  //    Value operand = operands[i];
  //    auto memTy = operand.getType().dyn_cast<hir::MemrefType>();
  //    if (!memTy) {
  //      newOperands.push_back(operand);
  //      newInputDelays.push_back(oldInputDelays[i]);
  //    } else if (memTy.getPort() == rd) {
  //      Value addrBus = mapMemref2RdAddrBusTensor[operand];
  //      Value dataBus = mapMemref2RdDataBusTensor[operand];
  //
  //      newOperands.push_back(addrBus);
  //      newOperands.push_back(dataBus);
  //      newInputDelays.push_back(helper::getIntegerAttr(context, 0));
  //      newInputDelays.push_back(helper::getIntegerAttr(context, 0));
  //    } else if (memTy.getPort() == wr) {
  //      Value writeBus = mapMemref2WrBusTensor[operand];
  //      newOperands.push_back(writeBus);
  //    } else {
  //      assert(false && "We dont yet support rw access");
  //    }
  //  }
  //  // remove the old operands and add the new array of operands.
  //  op.operandsMutable().clear();
  //  op.operandsMutable().append(newOperands);
  //
  //  // update the FuncType.
  //  auto newFuncTy = updateFuncType(builder, oldFuncTy, op.getOperands(),
  //                                  ArrayAttr::get(context, newInputDelays));
  //  op->setAttr("funcTy", TypeAttr::get(newFuncTy));
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

void MemrefLoweringPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();

  updateOp(funcOp);
  WalkResult walk = funcOp.walk([this](Operation *operation) -> WalkResult {
    dispatchOp(operation);
    return WalkResult::advance();
  });

  if (walk.wasInterrupted()) {
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
