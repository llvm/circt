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
  auto f32Ty = FloatType::getF32(&context);
  Type arg1Ty = buildFifoPopInterfaceTy(builder, context, f32Ty);
  Type outputArrayTy = ArrayType::get(&context, {2, 2}, f32Ty);
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

  auto f32Ty = FloatType::getF32(&context);
  auto timeTy = hir::TimeType::get(&context);
  auto constTy = hir::ConstType::get(&context);
  // %buff_r, %buff_w = hir.alloca ...
  mlir::Value buffR, buffW;
  {
    auto memref2x16bramRd =
        hir::MemrefType::get(&context, {2, 16}, f32Ty,
                             ArrayAttr::get(&context, {oneAttr}), bramRdAttr);
    auto memref2x16bramWr =
        hir::MemrefType::get(&context, {2, 16}, f32Ty,
                             ArrayAttr::get(&context, {oneAttr}), bramWrAttr);

    SmallVector<Type, 4> resultTypes = {memref2x16bramRd, memref2x16bramWr};
    auto op =
        builder.create<hir::AllocaOp>(builder.getUnknownLoc(), resultTypes,
                                      StringAttr::get(&context, "bram"));
    buffR = op.getResult(0);
    buffW = op.getResult(1);
  }

  // %wndw_r, %wndw_w = hir.alloca ...
  mlir::Value wndwR, wndwW;
  {
    auto memref2x2regRd = hir::MemrefType::get(
        &context, {2, 2}, f32Ty, ArrayAttr::get(&context, {zeroAttr, oneAttr}),
        regRdAttr);
    auto memref2x2regWr = hir::MemrefType::get(
        &context, {2, 2}, f32Ty, ArrayAttr::get(&context, {zeroAttr, oneAttr}),
        regWrAttr);

    SmallVector<Type, 4> resultTypes = {memref2x2regRd, memref2x2regWr};
    hir::AllocaOp op =
        builder.create<hir::AllocaOp>(builder.getUnknownLoc(), resultTypes,
                                      StringAttr::get(&context, "bram"));
    wndwR = op.getResult(0);
    wndwW = op.getResult(1);
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

    //%tf=hir.for %j :i32 = %0 :!hir.const to %16 :!hir.const step %1
    //:!hir.const
    Value tf;
    {
      hir::ForOp forJ = builder.create<hir::ForOp>(
          builder.getUnknownLoc(), helper::getTimeType(&context), c0, c16, c1,
          ti, c1);
      forJ.addEntryBlock(&context, helper::getIntegerType(&context, 32));
      forJ.beginRegion(builder);
      mlir::Value tj = forJ.getInductionVar();
      mlir::Value j = forJ.getInductionVar();
      tf = forJ.getResult();

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
                .create<hir::RecvOp>(builder.getUnknownLoc(), f32Ty, vg,
                                     SmallVector<mlir::Value, 4>({c0}), tv,
                                     mlir::Value())
                .getResult();
      }

      //%v1 = hir.delay ...
      mlir::Value v1;
      {
        v1 = builder
                 .create<hir::DelayOp>(builder.getUnknownLoc(), f32Ty, v, c1,
                                       tv, Value())
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

        //%val = hir.load %buff_r[%k1Plus1,%j] at %tk1
        mlir::Value val;
        {

          val = builder
                    .create<hir::LoadOp>(
                        builder.getUnknownLoc(), f32Ty, buffR,
                        SmallVector<mlir::Value, 4>({k1Plus1, j}), tk1, Value())
                    .getResult();
        }

        // hir.store %val to %buff_w[%k1,%j] at %tk1  +  %1
        {
          builder.create<hir::StoreOp>(builder.getUnknownLoc(), val, buffW,
                                       SmallVector<mlir::Value>({k1, j}), tk1,
                                       c1);
        }

        // hir.store %val to %wndw_w[%k1,%0] at %tk1  +  %1
        {

          builder.create<hir::StoreOp>(builder.getUnknownLoc(), val, wndwW,
                                       SmallVector<mlir::Value>({k1, c0}), tk1,
                                       c1);
        }

        // hir.send %val to %outp[%1,%k1,%0] at %tv + %1
        {

          builder.create<hir::SendOp>(builder.getUnknownLoc(), val, outp,
                                      SmallVector<Value>({c1, k1, c0}), tk1,
                                      c1);
        }
        unrollK1.endRegion(builder);
      }

      // hir.store %v1 to %buff_w[%1,%j] at %tv + %1
      {
        builder.create<hir::StoreOp>(builder.getUnknownLoc(), v1, buffW,
                                     SmallVector<mlir::Value>({c1, j}), tv, c1);
      }
      // hir.store %v1 to %wndw_w[%1,%0] at %tv + %1
      {
        builder.create<hir::StoreOp>(builder.getUnknownLoc(), v1, wndwW,
                                     SmallVector<mlir::Value>({c1, c0}), tv,
                                     c1);
      }
      // hir.send %v1 to %outp[%1,%1,%0] at %tv + %1
      {

        builder.create<hir::SendOp>(builder.getUnknownLoc(), v1, outp,
                                    SmallVector<Value>({c1, c1, c0}), tv, c1);
      }
      // %t_send = hir.delay %tv by %1 at %tv :!hir.time -> !hir.time
      mlir::Value tSend;
      {

        tSend = builder
                    .create<hir::DelayOp>(builder.getUnknownLoc(), timeTy, tv,
                                          c1, tv, Value())
                    .getResult();
      }

      // hir.send %t_send to %outp[%0] at %t_send
      {

        builder.create<hir::SendOp>(builder.getUnknownLoc(), tSend, outp,
                                    SmallVector<Value>({c0}), tSend, c0);
      }

      // hir.unroll_for %k1 = 0 to 2 step 1 iter_time(%tk1 = %tv){
      {
        hir::UnrollForOp unrollK1 = builder.create<hir::UnrollForOp>(
            builder.getUnknownLoc(), timeTy, 0, 2, 1, tv);
        unrollK1.addEntryBlock(&context);
        unrollK1.beginRegion(builder);
        mlir::Value tk1 = unrollK1.getIterTimeVar();
        mlir::Value k1 = unrollK1.getInductionVar();

        // hir.yield at %tk1
        {

          builder.create<hir::YieldOp>(builder.getUnknownLoc(),
                                       SmallVector<Value, 4>(), tk1, Value());
        }

        // hir.unroll_for %k2 = 0 to 1 step 1 iter_time(%tk2 = %tk1){
        {
          hir::UnrollForOp unrollK2 = builder.create<hir::UnrollForOp>(
              builder.getUnknownLoc(), timeTy, 0, 1, 1, tk1);
          unrollK2.addEntryBlock(&context);
          unrollK2.beginRegion(builder);
          mlir::Value tk2 = unrollK2.getIterTimeVar();
          mlir::Value k2 = unrollK2.getInductionVar();
          // hir.yield at %tk2
          {

            builder.create<hir::YieldOp>(builder.getUnknownLoc(),
                                         SmallVector<Value, 4>(), tk2, Value());
          }

          //%val = hir.load %wndw_r[%k1,%k2] at %tk2
          mlir::Value val;
          {

            val = builder
                      .create<hir::LoadOp>(
                          builder.getUnknownLoc(), f32Ty, wndwR,
                          SmallVector<mlir::Value, 4>({k1, k2}), tk2, Value())
                      .getResult();
          }

          //%k2Plus1 = hir.add(%k2,%1) :(!hir.const, !hir.const) -> (!hir.const)
          mlir::Value k2Plus1;
          {
            k2Plus1 = builder
                          .create<hir::AddOp>(builder.getUnknownLoc(), constTy,
                                              k2, c1)
                          .getResult();
          }
          // hir.store %val to %wndw_w[%k1,%k2Plus1] at %tk2 + %1
          {
            builder.create<hir::StoreOp>(
                builder.getUnknownLoc(), val, wndwW,
                SmallVector<mlir::Value>({k1, k2Plus1}), tk2, c1);
          }
          // hir.send %val to %outp[%1,%k1,%k2Plus1] at %tk2 + %1
          {

            builder.create<hir::SendOp>(builder.getUnknownLoc(), val, outp,
                                        SmallVector<Value>({c1, k1, k2Plus1}),
                                        tk2, c1);
          }

          unrollK2.endRegion(builder);
        }

        unrollK1.endRegion(builder);
      }
      // hir.yield at %tv + %1
      {

        builder.create<hir::YieldOp>(builder.getUnknownLoc(),
                                     SmallVector<Value, 4>(), tv, c1);
      }
      // finish the forJ body.
      forJ.endRegion(builder);
    }
    // hir.yield at %tf + %1
    {

      builder.create<hir::YieldOp>(builder.getUnknownLoc(),
                                   SmallVector<Value, 4>(), tf, c1);
    }
    forI.endRegion(builder);
  }
  // hir.return
  {
    builder.create<hir::ReturnOp>(builder.getUnknownLoc(),
                                  SmallVector<Value>());
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
