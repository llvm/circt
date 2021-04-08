#include "HIRGen.h"
#include "circt/Dialect/HIR/helper.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace hir;

int emitMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  mlir::OpBuilder builder(&context);
  module = mlir::ModuleOp::create(builder.getUnknownLoc());
  emitLineBuffer(builder, context, module, "line_buffer");
  return 0;
}

FuncType buildFifoPopInterfaceTy(mlir::OpBuilder &builder,
                                 mlir::MLIRContext &context, Type elementTy) {
  assert(elementTy.isa<IntegerType>() || elementTy.isa<FloatType>());

  SmallVector<Type, 4> resultTypes = {TimeType::get(&context), elementTy};
  auto functionTy = FunctionType::get(&context, llvm::None, resultTypes);
  IntegerAttr zeroAttr =
      IntegerAttr::get(helper::getIntegerType(&context, 64), 0);
  return FuncType::get(&context, functionTy, ArrayAttr(),
                       ArrayAttr::get(&context, {zeroAttr, zeroAttr}));
}

SmallVector<Type, 4> buildLineBufferArgTypes(mlir::OpBuilder &builder,
                                             mlir::MLIRContext &context) {
  Type arg1Ty =
      buildFifoPopInterfaceTy(builder, context, FloatType::getF32(&context));
  Type outputArrayTy =
      ArrayType::get(&context, {2, 2}, FloatType::getF32(&context));
  Type arg2Ty =
      GroupType::get(&context, {TimeType::get(&context), outputArrayTy},
                     {Attribute(), Attribute()});
  return {arg1Ty, arg2Ty};
}

DictionaryAttr buildMemrefPortAttr(mlir::MLIRContext &context, int rd, int wr) {
  if (rd >= 0 && wr > 0)
    return DictionaryAttr::get(
        &context, {NamedAttribute(Identifier::get("rd", &context),
                                  helper::getIntegerAttr(&context, 64, rd)),
                   NamedAttribute(Identifier::get("wr", &context),
                                  helper::getIntegerAttr(&context, 64, wr))});
  if (wr >= 0)
    return DictionaryAttr::get(
        &context, {NamedAttribute(Identifier::get("wr", &context),
                                  helper::getIntegerAttr(&context, 64, wr))});
  return DictionaryAttr::get(
      &context, {NamedAttribute(Identifier::get("rd", &context),
                                helper::getIntegerAttr(&context, 64, rd))});
}

void emitLineBufferBody(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                        Block *entryBlock) {
  mlir::Value inp = entryBlock->getArgument(0);
  mlir::Value outp = entryBlock->getArgument(1);
  mlir::Value t = entryBlock->getArgument(2);

  auto bramRdAttr = buildMemrefPortAttr(context, 1, -1);
  auto bramWrAttr = buildMemrefPortAttr(context, -1, 1);
  auto regRdAttr = buildMemrefPortAttr(context, 0, -1);
  auto regWrAttr = buildMemrefPortAttr(context, -1, 1);

  auto zeroAttr = helper::getIntegerAttr(&context, 64, 0);
  auto oneAttr = helper::getIntegerAttr(&context, 64, 1);
  // %buff_r, %buff_w = hir.alloca ...
  {
    auto memref2x16bramRd =
        hir::MemrefType::get(&context, {2, 16}, FloatType::getF32(&context),
                             ArrayAttr::get(&context, {oneAttr}), bramRdAttr);
    auto memref2x16bramWr =
        hir::MemrefType::get(&context, {2, 16}, FloatType::getF32(&context),
                             ArrayAttr::get(&context, {oneAttr}), bramWrAttr);

    SmallVector<Type, 4> resultTypes = {memref2x16bramRd, memref2x16bramWr};
    builder.create<hir::AllocaOp>(builder.getUnknownLoc(), resultTypes,
                                  StringAttr::get(&context, "bram"));
  }

  // %wndw_r, %wndw_w = hir.alloca ...
  mlir::Value wndwR, wndwW;
  {
    auto memref2x2regRd = hir::MemrefType::get(
        &context, {2, 2}, FloatType::getF32(&context),
        ArrayAttr::get(&context, {zeroAttr, oneAttr}), regRdAttr);
    auto memref2x2regWr = hir::MemrefType::get(
        &context, {2, 2}, FloatType::getF32(&context),
        ArrayAttr::get(&context, {zeroAttr, oneAttr}), regWrAttr);

    SmallVector<Type, 4> resultTypes = {memref2x2regRd, memref2x2regWr};
    hir::AllocaOp wndwAllocaOp =
        builder.create<hir::AllocaOp>(builder.getUnknownLoc(), resultTypes,
                                      StringAttr::get(&context, "bram"));
    wndwR = wndwAllocaOp.getResult(0);
    wndwW = wndwAllocaOp.getResult(1);
  }

  //%0 = hir.constant ...
  mlir::Value c0, c1, c2, c3, c4, c16;
  {
    c0 = builder
             .create<hir::ConstantOp>(builder.getUnknownLoc(),
                                      helper::getConstIntType(&context), 0)
             .getResult();
    c1 = builder
             .create<hir::ConstantOp>(builder.getUnknownLoc(),
                                      helper::getConstIntType(&context), 1)
             .getResult();
    c2 = builder
             .create<hir::ConstantOp>(builder.getUnknownLoc(),
                                      helper::getConstIntType(&context), 2)
             .getResult();
    c16 = builder
              .create<hir::ConstantOp>(builder.getUnknownLoc(),
                                       helper::getConstIntType(&context), 16)
              .getResult();
  }

  // hir.for 0 to 16 step 1
  {

    hir::ForOp forI = builder.create<hir::ForOp>(builder.getUnknownLoc(),
                                                 helper::getTimeType(&context),
                                                 c0, c16, c1, t, c1);

    auto owningBlock = std::make_unique<Block>();

    forI.addEntryBlock(&context, helper::getIntegerType(&context, 32));
    forI.beginRegion(builder);
    mlir::Value ti = forI.getIterTimeVar();
    // mlir::Value i = forI.getInductionVar();

    // hir.for 0 to 16 step 1
    {
      hir::ForOp forJ = builder.create<hir::ForOp>(
          builder.getUnknownLoc(), helper::getTimeType(&context), c0, c16, c1,
          ti, c1);
      forJ.addEntryBlock(&context, helper::getIntegerType(&context, 32));
      forJ.beginRegion(builder);
      mlir::Value tj = forJ.getInductionVar();
      // mlir::Value j = forJ.getInductionVar();

      //%tv,%vg = hir.call ...
      mlir::Value tv, vg;
      {
        ArrayRef<Type> resultTypes = inp.getType()
                                         .dyn_cast<hir::FuncType>()
                                         .getFunctionType()
                                         .getResults();
        auto op = builder.create<hir::CallOp>(
            builder.getUnknownLoc(), resultTypes,
            /*callee*/ FlatSymbolRefAttr(),
            /*funcTy*/ inp.getType().dyn_cast<hir::FuncType>(),
            /*callee_var*/ inp,
            /*operands*/ SmallVector<mlir::Value, 4>(), /*tstart*/ tj,
            /*offset*/ mlir::Value());
        tv = op.getResult(0);
        vg = op.getResult(1);
      }

      //%v = hir.recv ...
      mlir::Value v;
      {
        v = builder
                .create<hir::RecvOp>(
                    builder.getUnknownLoc(), FloatType::getF32(&context), vg,
                    SmallVector<mlir::Value, 4>({c0}), tv, mlir::Value())
                .getResult();
      }

      //%v1 = hir.delay ...
      mlir::Value v1;
      {
        v1 = builder
                 .create<hir::DelayOp>(builder.getUnknownLoc(),
                                       FloatType::getF32(&context), v, c1, tv,
                                       Value())
                 .getResult();
      }

      // hir.unroll_for %k1 =
      {
        hir::UnrollForOp unrollK1 = builder.create<hir::UnrollForOp>(
            builder.getUnknownLoc(), hir::TimeType::get(&context), 0, 1, 1, tv);
        unrollK1.addEntryBlock(&context);
        unrollK1.beginRegion(builder);
        mlir::Value tk1 = unrollK1.getIterTimeVar();
        mlir::Value k1 = unrollK1.getInductionVar();

        // hir.yield at %tk1
        {
          builder.create<hir::YieldOp>(builder.getUnknownLoc(),
                                       SmallVector<Value, 4>(), tk1, Value());
        }

        //%k1Plus1 = hir.add ...
        mlir::Value k1Plus1;
        {
          k1Plus1 =
              builder
                  .create<hir::AddOp>(builder.getUnknownLoc(),
                                      helper::getConstIntType(&context), k1, c1)
                  .getResult();
        }

        unrollK1.endRegion(builder);
      }
      // finish the forJ body.
      forJ.endRegion(builder);
    }
    forI.endRegion(builder);
  }
}

void emitLineBuffer(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                    mlir::OwningModuleRef &module, StringRef funcName) {
  IntegerAttr zeroAttr = helper::getIntegerAttr(&context, 64, 0);
  auto functionTy = builder.getFunctionType(
      buildLineBufferArgTypes(builder, context), llvm::None);
  auto funcTy = FuncType::get(&context, functionTy,
                              ArrayAttr::get(&context, {zeroAttr, zeroAttr}),
                              ArrayAttr());
  auto symName = mlir::StringAttr::get(&context, funcName);
  auto funcOp = builder.create<hir::FuncOp>(builder.getUnknownLoc(),
                                            TypeAttr::get(functionTy), symName,
                                            TypeAttr::get(funcTy));
  Block *entryBlock = funcOp.addEntryBlock();
  entryBlock->addArgument(TimeType::get(&context));
  builder.setInsertionPointToStart(entryBlock);
  emitLineBufferBody(builder, context, entryBlock);
  module->push_back(funcOp);
}
