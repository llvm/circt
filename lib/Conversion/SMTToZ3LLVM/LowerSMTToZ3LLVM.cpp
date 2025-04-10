//===- LowerSMTToZ3LLVM.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-smt-to-z3-llvm"

namespace circt {
#define GEN_PASS_DEF_LOWERSMTTOZ3LLVM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace smt;

//===----------------------------------------------------------------------===//
// SMTGlobalHandler implementation
//===----------------------------------------------------------------------===//

SMTGlobalsHandler SMTGlobalsHandler::create(OpBuilder &builder,
                                            ModuleOp module) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  SymbolCache symCache;
  symCache.addDefinitions(module);
  Namespace names;
  names.add(symCache);

  Location loc = module.getLoc();
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());

  auto createGlobal = [&](StringRef namePrefix) {
    auto global = builder.create<LLVM::GlobalOp>(
        loc, ptrTy, false, LLVM::Linkage::Internal, names.newName(namePrefix),
        Attribute{}, /*alignment=*/8);
    OpBuilder::InsertionGuard g(builder);
    builder.createBlock(&global.getInitializer());
    Value res = builder.create<LLVM::ZeroOp>(loc, ptrTy);
    builder.create<LLVM::ReturnOp>(loc, res);
    return global;
  };

  auto ctxGlobal = createGlobal("ctx");
  auto solverGlobal = createGlobal("solver");

  return SMTGlobalsHandler(std::move(names), solverGlobal, ctxGlobal);
}

SMTGlobalsHandler::SMTGlobalsHandler(Namespace &&names,
                                     mlir::LLVM::GlobalOp solver,
                                     mlir::LLVM::GlobalOp ctx)
    : solver(solver), ctx(ctx), names(names) {}

SMTGlobalsHandler::SMTGlobalsHandler(ModuleOp module,
                                     mlir::LLVM::GlobalOp solver,
                                     mlir::LLVM::GlobalOp ctx)
    : solver(solver), ctx(ctx) {
  SymbolCache symCache;
  symCache.addDefinitions(module);
  names.add(symCache);
}

//===----------------------------------------------------------------------===//
// Lowering Pattern Base
//===----------------------------------------------------------------------===//

namespace {

template <typename OpTy>
class SMTLoweringPattern : public OpConversionPattern<OpTy> {
public:
  SMTLoweringPattern(const TypeConverter &typeConverter, MLIRContext *context,
                     SMTGlobalsHandler &globals,
                     const LowerSMTToZ3LLVMOptions &options)
      : OpConversionPattern<OpTy>(typeConverter, context), globals(globals),
        options(options) {}

private:
  Value buildGlobalPtrToGlobal(OpBuilder &builder, Location loc,
                               LLVM::GlobalOp global,
                               DenseMap<Block *, Value> &cache) const {
    Block *block = builder.getBlock();
    if (auto iter = cache.find(block); iter != cache.end())
      return iter->getSecond();

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(block);
    Value globalAddr = builder.create<LLVM::AddressOfOp>(loc, global);
    return cache[block] = builder.create<LLVM::LoadOp>(
               loc, LLVM::LLVMPointerType::get(builder.getContext()),
               globalAddr);
  }

protected:
  /// A convenience function to get the pointer to the context from the 'global'
  /// operation. The result is cached for each basic block, i.e., it is assumed
  /// that this function is never called in the same basic block again at a
  /// location (insertion point of the 'builder') not dominating all previous
  /// locations this function was called at.
  Value buildContextPtr(OpBuilder &builder, Location loc) const {
    return buildGlobalPtrToGlobal(builder, loc, globals.ctx, globals.ctxCache);
  }

  /// A convenience function to get the pointer to the solver from the 'global'
  /// operation. The result is cached for each basic block, i.e., it is assumed
  /// that this function is never called in the same basic block again at a
  /// location (insertion point of the 'builder') not dominating all previous
  /// locations this function was called at.
  Value buildSolverPtr(OpBuilder &builder, Location loc) const {
    return buildGlobalPtrToGlobal(builder, loc, globals.solver,
                                  globals.solverCache);
  }

  /// Create a `llvm.call` operation to a function with the given 'name' and
  /// 'type'. If there does not already exist a (external) function with that
  /// name create a matching external function declaration.
  LLVM::CallOp buildCall(OpBuilder &builder, Location loc, StringRef name,
                         LLVM::LLVMFunctionType funcType,
                         ValueRange args) const {
    auto &funcOp = globals.funcMap[builder.getStringAttr(name)];
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(builder);
      auto module =
          builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
      builder.setInsertionPointToEnd(module.getBody());
      auto funcOpResult = LLVM::lookupOrCreateFn(
          module, name, funcType.getParams(), funcType.getReturnType(),
          funcType.getVarArg());
      assert(succeeded(funcOpResult) && "expected to lookup or create printf");
      funcOp = funcOpResult.value();
    }
    return builder.create<LLVM::CallOp>(loc, funcOp, args);
  }

  /// Build a global constant for the given string and construct an 'addressof'
  /// operation at the current 'builder' insertion point to get a pointer to it.
  /// Multiple calls with the same string will reuse the same global. It is
  /// guaranteed that the symbol of the global will be unique.
  Value buildString(OpBuilder &builder, Location loc, StringRef str) const {
    auto &global = globals.stringCache[builder.getStringAttr(str)];
    if (!global) {
      OpBuilder::InsertionGuard guard(builder);
      auto module =
          builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
      builder.setInsertionPointToEnd(module.getBody());
      auto arrayTy =
          LLVM::LLVMArrayType::get(builder.getI8Type(), str.size() + 1);
      auto strAttr = builder.getStringAttr(str.str() + '\00');
      global = builder.create<LLVM::GlobalOp>(
          loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
          globals.names.newName("str"), strAttr);
    }
    return builder.create<LLVM::AddressOfOp>(loc, global);
  }
  /// Most API functions require a pointer to the the Z3 context object as the
  /// first argument. This helper function prepends this pointer value to the
  /// call for convenience.
  LLVM::CallOp buildAPICallWithContext(OpBuilder &builder, Location loc,
                                       StringRef name, Type returnType,
                                       ValueRange args = {}) const {
    auto ctx = buildContextPtr(builder, loc);
    SmallVector<Value> arguments;
    arguments.emplace_back(ctx);
    arguments.append(SmallVector<Value>(args));
    return buildCall(
        builder, loc, name,
        LLVM::LLVMFunctionType::get(
            returnType, SmallVector<Type>(ValueRange(arguments).getTypes())),
        arguments);
  }

  /// Most API functions we need to call return a 'Z3_AST' object which is a
  /// pointer in LLVM. This helper function simplifies calling those API
  /// functions.
  Value buildPtrAPICall(OpBuilder &builder, Location loc, StringRef name,
                        ValueRange args = {}) const {
    return buildAPICallWithContext(
               builder, loc, name,
               LLVM::LLVMPointerType::get(builder.getContext()), args)
        ->getResult(0);
  }

  /// Build a value representing the SMT sort given with 'type'.
  Value buildSort(OpBuilder &builder, Location loc, Type type) const {
    // NOTE: if a type not handled by this switch is passed, an assertion will
    // be triggered.
    return TypeSwitch<Type, Value>(type)
        .Case([&](smt::IntType ty) {
          return buildPtrAPICall(builder, loc, "Z3_mk_int_sort");
        })
        .Case([&](smt::BitVectorType ty) {
          Value bitwidth = builder.create<LLVM::ConstantOp>(
              loc, builder.getI32Type(), ty.getWidth());
          return buildPtrAPICall(builder, loc, "Z3_mk_bv_sort", {bitwidth});
        })
        .Case([&](smt::BoolType ty) {
          return buildPtrAPICall(builder, loc, "Z3_mk_bool_sort");
        })
        .Case([&](smt::SortType ty) {
          Value str = buildString(builder, loc, ty.getIdentifier());
          Value sym =
              buildPtrAPICall(builder, loc, "Z3_mk_string_symbol", {str});
          return buildPtrAPICall(builder, loc, "Z3_mk_uninterpreted_sort",
                                 {sym});
        })
        .Case([&](smt::ArrayType ty) {
          return buildPtrAPICall(builder, loc, "Z3_mk_array_sort",
                                 {buildSort(builder, loc, ty.getDomainType()),
                                  buildSort(builder, loc, ty.getRangeType())});
        });
  }

  SMTGlobalsHandler &globals;
  const LowerSMTToZ3LLVMOptions &options;
};

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

/// The 'smt.declare_fun' operation is used to declare both constants and
/// functions. The Z3 API, however, uses two different functions. Therefore,
/// depending on the result type of this operation, one of the following two
/// API functions is used to create the symbolic value:
/// ```
/// Z3_ast Z3_API Z3_mk_fresh_const(Z3_context c, Z3_string prefix, Z3_sort ty);
/// Z3_func_decl Z3_API Z3_mk_fresh_func_decl(
///     Z3_context c, Z3_string prefix, unsigned domain_size,
///     Z3_sort const domain[], Z3_sort range);
/// ```
struct DeclareFunOpLowering : public SMTLoweringPattern<DeclareFunOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(DeclareFunOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    // Create the name prefix.
    Value prefix;
    if (adaptor.getNamePrefix())
      prefix = buildString(rewriter, loc, *adaptor.getNamePrefix());
    else
      prefix = rewriter.create<LLVM::ZeroOp>(
          loc, LLVM::LLVMPointerType::get(getContext()));

    // Handle the constant value case.
    if (!isa<SMTFuncType>(op.getType())) {
      Value sort = buildSort(rewriter, loc, op.getType());
      Value constDecl =
          buildPtrAPICall(rewriter, loc, "Z3_mk_fresh_const", {prefix, sort});
      rewriter.replaceOp(op, constDecl);
      return success();
    }

    // Otherwise, we declare a function.
    Type llvmPtrTy = LLVM::LLVMPointerType::get(getContext());
    auto funcType = cast<SMTFuncType>(op.getResult().getType());
    Value rangeSort = buildSort(rewriter, loc, funcType.getRangeType());

    Type arrTy =
        LLVM::LLVMArrayType::get(llvmPtrTy, funcType.getDomainTypes().size());

    Value domain = rewriter.create<LLVM::UndefOp>(loc, arrTy);
    for (auto [i, ty] : llvm::enumerate(funcType.getDomainTypes())) {
      Value sort = buildSort(rewriter, loc, ty);
      domain = rewriter.create<LLVM::InsertValueOp>(loc, domain, sort, i);
    }

    Value one =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
    Value domainStorage =
        rewriter.create<LLVM::AllocaOp>(loc, llvmPtrTy, arrTy, one);
    rewriter.create<LLVM::StoreOp>(loc, domain, domainStorage);

    Value domainSize = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), funcType.getDomainTypes().size());
    Value decl =
        buildPtrAPICall(rewriter, loc, "Z3_mk_fresh_func_decl",
                        {prefix, domainSize, domainStorage, rangeSort});

    rewriter.replaceOp(op, decl);
    return success();
  }
};

/// Lower the 'smt.apply_func' operation to Z3 API calls of the form:
/// ```
/// Z3_ast Z3_API Z3_mk_app(Z3_context c, Z3_func_decl d,
///                         unsigned num_args, Z3_ast const args[]);
/// ```
struct ApplyFuncOpLowering : public SMTLoweringPattern<ApplyFuncOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(ApplyFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type llvmPtrTy = LLVM::LLVMPointerType::get(getContext());
    Type arrTy = LLVM::LLVMArrayType::get(llvmPtrTy, adaptor.getArgs().size());

    // Create an array of the function arguments.
    Value domain = rewriter.create<LLVM::UndefOp>(loc, arrTy);
    for (auto [i, arg] : llvm::enumerate(adaptor.getArgs()))
      domain = rewriter.create<LLVM::InsertValueOp>(loc, domain, arg, i);

    // Store the array on the stack.
    Value one =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
    Value domainStorage =
        rewriter.create<LLVM::AllocaOp>(loc, llvmPtrTy, arrTy, one);
    rewriter.create<LLVM::StoreOp>(loc, domain, domainStorage);

    // Call the API function with a pointer to the function, the number of
    // arguments, and the pointer to the arguments stored on the stack.
    Value domainSize = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), adaptor.getArgs().size());
    Value returnVal =
        buildPtrAPICall(rewriter, loc, "Z3_mk_app",
                        {adaptor.getFunc(), domainSize, domainStorage});
    rewriter.replaceOp(op, returnVal);

    return success();
  }
};

/// Lower the `smt.bv.constant` operation to either
/// ```
/// Z3_ast Z3_API Z3_mk_unsigned_int64(Z3_context c, uint64_t v, Z3_sort ty);
/// ```
/// if the bit-vector fits into a 64-bit integer or convert it to a string and
/// use the sligtly slower but arbitrary precision API function:
/// ```
/// Z3_ast Z3_API Z3_mk_numeral(Z3_context c, Z3_string numeral, Z3_sort ty);
/// ```
/// Note that there is also an API function taking an array of booleans, and
/// while those are typically compiled to 'i8' in LLVM they don't necessarily
/// have to (I think).
struct BVConstantOpLowering : public SMTLoweringPattern<smt::BVConstantOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(smt::BVConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    unsigned width = op.getType().getWidth();
    auto bvSort = buildSort(rewriter, loc, op.getResult().getType());
    APInt val = adaptor.getValue().getValue();

    if (width <= 64) {
      Value bvConst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), val.getZExtValue());
      Value res = buildPtrAPICall(rewriter, loc, "Z3_mk_unsigned_int64",
                                  {bvConst, bvSort});
      rewriter.replaceOp(op, res);
      return success();
    }

    std::string str;
    llvm::raw_string_ostream stream(str);
    stream << val;
    Value bvString = buildString(rewriter, loc, str);
    Value bvNumeral =
        buildPtrAPICall(rewriter, loc, "Z3_mk_numeral", {bvString, bvSort});

    rewriter.replaceOp(op, bvNumeral);
    return success();
  }
};

/// Some of the Z3 API supports a variadic number of operands for some
/// operations (in particular if the expansion would lead to a super-linear
/// increase in operations such as with the ':pairwise' attribute). Those API
/// calls take an 'unsigned' argument indicating the size of an array of
/// pointers to the operands.
template <typename SourceTy>
struct VariadicSMTPattern : public SMTLoweringPattern<SourceTy> {
  using OpAdaptor = typename SMTLoweringPattern<SourceTy>::OpAdaptor;

  VariadicSMTPattern(const TypeConverter &typeConverter, MLIRContext *context,
                     SMTGlobalsHandler &globals,
                     const LowerSMTToZ3LLVMOptions &options,
                     StringRef apiFuncName, unsigned minNumArgs)
      : SMTLoweringPattern<SourceTy>(typeConverter, context, globals, options),
        apiFuncName(apiFuncName), minNumArgs(minNumArgs) {}

  LogicalResult
  matchAndRewrite(SourceTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().size() < minNumArgs)
      return failure();

    Location loc = op.getLoc();
    Value numOperands = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), op->getNumOperands());
    Value constOne =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type arrTy = LLVM::LLVMArrayType::get(ptrTy, op->getNumOperands());
    Value storage =
        rewriter.create<LLVM::AllocaOp>(loc, ptrTy, arrTy, constOne);
    Value array = rewriter.create<LLVM::UndefOp>(loc, arrTy);

    for (auto [i, operand] : llvm::enumerate(adaptor.getOperands()))
      array = rewriter.create<LLVM::InsertValueOp>(
          loc, array, operand, ArrayRef<int64_t>{(int64_t)i});

    rewriter.create<LLVM::StoreOp>(loc, array, storage);

    rewriter.replaceOp(op,
                       SMTLoweringPattern<SourceTy>::buildPtrAPICall(
                           rewriter, loc, apiFuncName, {numOperands, storage}));
    return success();
  }

private:
  StringRef apiFuncName;
  unsigned minNumArgs;
};

/// Lower an SMT operation to a function call with the name 'apiFuncName' with
/// arguments matching the operands one-to-one.
template <typename SourceTy>
struct OneToOneSMTPattern : public SMTLoweringPattern<SourceTy> {
  using OpAdaptor = typename SMTLoweringPattern<SourceTy>::OpAdaptor;

  OneToOneSMTPattern(const TypeConverter &typeConverter, MLIRContext *context,
                     SMTGlobalsHandler &globals,
                     const LowerSMTToZ3LLVMOptions &options,
                     StringRef apiFuncName, unsigned numOperands)
      : SMTLoweringPattern<SourceTy>(typeConverter, context, globals, options),
        apiFuncName(apiFuncName), numOperands(numOperands) {}

  LogicalResult
  matchAndRewrite(SourceTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().size() != numOperands)
      return failure();

    rewriter.replaceOp(
        op, SMTLoweringPattern<SourceTy>::buildPtrAPICall(
                rewriter, op.getLoc(), apiFuncName, adaptor.getOperands()));
    return success();
  }

private:
  StringRef apiFuncName;
  unsigned numOperands;
};

/// A pattern to lower SMT operations with a variadic number of operands
/// modelling the ':chainable' attribute in SMT to binary operations.
template <typename SourceTy>
class LowerChainableSMTPattern : public SMTLoweringPattern<SourceTy> {
  using SMTLoweringPattern<SourceTy>::SMTLoweringPattern;
  using OpAdaptor = typename SMTLoweringPattern<SourceTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SourceTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().size() <= 2)
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value> elements;
    for (int i = 1, e = adaptor.getOperands().size(); i < e; ++i) {
      Value val = rewriter.create<SourceTy>(
          loc, op->getResultTypes(),
          ValueRange{adaptor.getOperands()[i - 1], adaptor.getOperands()[i]});
      elements.push_back(val);
    }
    rewriter.replaceOpWithNewOp<smt::AndOp>(op, elements);
    return success();
  }
};

/// A pattern to lower SMT operations with a variadic number of operands
/// modelling the `:left-assoc` attribute to a sequence of binary operators.
template <typename SourceTy>
class LowerLeftAssocSMTPattern : public SMTLoweringPattern<SourceTy> {
  using SMTLoweringPattern<SourceTy>::SMTLoweringPattern;
  using OpAdaptor = typename SMTLoweringPattern<SourceTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SourceTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().size() <= 2)
      return rewriter.notifyMatchFailure(op, "must have at least two operands");

    Value runner = adaptor.getOperands()[0];
    for (Value val : adaptor.getOperands().drop_front())
      runner = rewriter.create<SourceTy>(op.getLoc(), op->getResultTypes(),
                                         ValueRange{runner, val});

    rewriter.replaceOp(op, runner);
    return success();
  }
};

/// The 'smt.solver' operation has a region that corresponds to the lifetime of
/// the Z3 context and one solver instance created within this context.
/// To create a context, a Z3 configuration has to be built first and various
/// configuration parameters can be set before creating a context from it. Once
/// we have a context, we can create a solver and store a pointer to the context
/// and the solver in an LLVM global such that operations in the child region
/// have access to them. While the context created with `Z3_mk_context` takes
/// care of the reference counting of `Z3_AST` objects, it still requires manual
/// reference counting of `Z3_solver` objects, therefore, we need to increase
/// the ref. counter of the solver we get from `Z3_mk_solver` and must decrease
/// it again once we don't need it anymore. Finally, the configuration object
/// can be deleted.
/// ```
/// Z3_config Z3_API Z3_mk_config(void);
/// void Z3_API Z3_set_param_value(Z3_config c, Z3_string param_id,
///                                Z3_string param_value);
/// Z3_context Z3_API Z3_mk_context(Z3_config c);
/// Z3_solver Z3_API Z3_mk_solver(Z3_context c);
/// void Z3_API Z3_solver_inc_ref(Z3_context c, Z3_solver s);
/// void Z3_API Z3_del_config(Z3_config c);
/// ```
/// At the end of the solver lifetime, we have to tell the context that we
/// don't need the solver anymore and delete the context itself.
/// ```
/// void Z3_API Z3_solver_dec_ref(Z3_context c, Z3_solver s);
/// void Z3_API Z3_del_context(Z3_context c);
/// ```
/// Note that the solver created here is a combined solver. There might be some
/// potential for optimization by creating more specialized solvers supported by
/// the Z3 API according the the kind of operations present in the body region.
struct SolverOpLowering : public SMTLoweringPattern<SolverOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(SolverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto voidTy = LLVM::LLVMVoidType::get(getContext());
    auto ptrToPtrFunc = LLVM::LLVMFunctionType::get(ptrTy, ptrTy);
    auto ptrPtrToPtrFunc = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, ptrTy});
    auto ptrToVoidFunc = LLVM::LLVMFunctionType::get(voidTy, ptrTy);
    auto ptrPtrToVoidFunc = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy});

    // Create the configuration.
    Value config = buildCall(rewriter, loc, "Z3_mk_config",
                             LLVM::LLVMFunctionType::get(ptrTy, {}), {})
                       .getResult();

    // In debug-mode, we enable proofs such that we can fetch one in the 'unsat'
    // region of each 'smt.check' operation.
    if (options.debug) {
      Value paramKey = buildString(rewriter, loc, "proof");
      Value paramValue = buildString(rewriter, loc, "true");
      buildCall(rewriter, loc, "Z3_set_param_value",
                LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy}),
                {config, paramKey, paramValue});
    }

    // Check if the logic is set anywhere within the solver
    std::optional<StringRef> logic = std::nullopt;
    auto setLogicOps = op.getBodyRegion().getOps<smt::SetLogicOp>();
    if (!setLogicOps.empty()) {
      // We know from before patterns were applied that there is only one
      // set_logic op
      auto setLogicOp = *setLogicOps.begin();
      logic = setLogicOp.getLogic();
      rewriter.eraseOp(setLogicOp);
    }

    // Create the context and store a pointer to it in the global variable.
    Value ctx = buildCall(rewriter, loc, "Z3_mk_context", ptrToPtrFunc, config)
                    .getResult();
    Value ctxAddr =
        rewriter.create<LLVM::AddressOfOp>(loc, globals.ctx).getResult();
    rewriter.create<LLVM::StoreOp>(loc, ctx, ctxAddr);

    // Delete the configuration again.
    buildCall(rewriter, loc, "Z3_del_config", ptrToVoidFunc, {config});

    // Create a solver instance, increase its reference counter, and store a
    // pointer to it in the global variable.
    Value solver;
    if (logic) {
      auto logicStr = buildString(rewriter, loc, logic.value());
      solver = buildCall(rewriter, loc, "Z3_mk_solver_for_logic",
                         ptrPtrToPtrFunc, {ctx, logicStr})
                   ->getResult(0);
    } else {
      solver = buildCall(rewriter, loc, "Z3_mk_solver", ptrToPtrFunc, ctx)
                   ->getResult(0);
    }
    buildCall(rewriter, loc, "Z3_solver_inc_ref", ptrPtrToVoidFunc,
              {ctx, solver});
    Value solverAddr =
        rewriter.create<LLVM::AddressOfOp>(loc, globals.solver).getResult();
    rewriter.create<LLVM::StoreOp>(loc, solver, solverAddr);

    // This assumes that no constant hoisting of the like happens inbetween
    // the patterns defined in this pass because once the solver initialization
    // and deallocation calls are inserted and the body region is inlined,
    // canonicalizations and folders applied inbetween lowering patterns might
    // hoist the SMT constants which means they would access uninitialized
    // global variables once they are lowered.
    SmallVector<Type> convertedTypes;
    if (failed(
            typeConverter->convertTypes(op->getResultTypes(), convertedTypes)))
      return failure();

    func::FuncOp funcOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto module = op->getParentOfType<ModuleOp>();
      rewriter.setInsertionPointToEnd(module.getBody());

      funcOp = rewriter.create<func::FuncOp>(
          loc, globals.names.newName("solver"),
          rewriter.getFunctionType(adaptor.getInputs().getTypes(),
                                   convertedTypes));
      rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getBody(),
                                  funcOp.end());
    }

    ValueRange results =
        rewriter.create<func::CallOp>(loc, funcOp, adaptor.getInputs())
            ->getResults();

    // At the end of the region, decrease the solver's reference counter and
    // delete the context.
    // NOTE: we cannot use the convenience helper here because we don't want to
    // load the context from the global but use the result from the 'mk_context'
    // call directly for two reasons:
    // * avoid an unnecessary load
    // * the caching mechanism of the context does not work here because it
    // would reuse the loaded context from a earlier solver
    buildCall(rewriter, loc, "Z3_solver_dec_ref", ptrPtrToVoidFunc,
              {ctx, solver});
    buildCall(rewriter, loc, "Z3_del_context", ptrToVoidFunc, ctx);

    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Lower `smt.assert` operations to Z3 API calls of the form:
/// ```
/// void Z3_API Z3_solver_assert(Z3_context c, Z3_solver s, Z3_ast a);
/// ```
struct AssertOpLowering : public SMTLoweringPattern<AssertOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    buildAPICallWithContext(
        rewriter, loc, "Z3_solver_assert",
        LLVM::LLVMVoidType::get(getContext()),
        {buildSolverPtr(rewriter, loc), adaptor.getInput()});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower `smt.reset` operations to Z3 API calls of the form:
/// ```
/// void Z3_API Z3_solver_reset(Z3_context c, Z3_solver s);
/// ```
struct ResetOpLowering : public SMTLoweringPattern<ResetOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(ResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    buildAPICallWithContext(rewriter, loc, "Z3_solver_reset",
                            LLVM::LLVMVoidType::get(getContext()),
                            {buildSolverPtr(rewriter, loc)});

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower `smt.push` operations to (repeated) Z3 API calls of the form:
/// ```
/// void Z3_API Z3_solver_push(Z3_context c, Z3_solver s);
/// ```
struct PushOpLowering : public SMTLoweringPattern<PushOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(PushOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    // SMTLIB allows multiple levels to be pushed with one push command, but the
    // Z3 C API doesn't let you provide a number of levels for push calls so
    // multiple calls have to be created.
    for (uint32_t i = 0; i < op.getCount(); i++)
      buildAPICallWithContext(rewriter, loc, "Z3_solver_push",
                              LLVM::LLVMVoidType::get(getContext()),
                              {buildSolverPtr(rewriter, loc)});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower `smt.pop` operations to Z3 API calls of the form:
/// ```
/// void Z3_API Z3_solver_pop(Z3_context c, Z3_solver s, unsigned n);
/// ```
struct PopOpLowering : public SMTLoweringPattern<PopOp> {
  using SMTLoweringPattern::SMTLoweringPattern;
  LogicalResult
  matchAndRewrite(PopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value constVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), op.getCount());
    buildAPICallWithContext(rewriter, loc, "Z3_solver_pop",
                            LLVM::LLVMVoidType::get(getContext()),
                            {buildSolverPtr(rewriter, loc), constVal});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower `smt.yield` operations to `scf.yield` operations. This not necessary
/// for the yield in `smt.solver` or in quantifiers since they are deleted
/// directly by the parent operation, but makes the lowering of the `smt.check`
/// operation simpler and more convenient since the regions get translated
/// directly to regions of `scf.if` operations.
struct YieldOpLowering : public SMTLoweringPattern<YieldOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op->getParentOfType<func::FuncOp>()) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getValues());
      return success();
    }
    if (op->getParentOfType<LLVM::LLVMFuncOp>()) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getValues());
      return success();
    }
    if (isa<scf::SCFDialect>(op->getParentOp()->getDialect())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getValues());
      return success();
    }
    return failure();
  }
};

/// Lower `smt.check` operations to Z3 API calls and control-flow operations.
/// ```
/// Z3_lbool Z3_API Z3_solver_check(Z3_context c, Z3_solver s);
///
/// typedef enum
/// {
///     Z3_L_FALSE = -1, // means unsatisfiable here
///     Z3_L_UNDEF,      // means unknown here
///     Z3_L_TRUE        // means satisfiable here
/// } Z3_lbool;
/// ```
struct CheckOpLowering : public SMTLoweringPattern<CheckOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(CheckOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto printfType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrTy}, true);

    auto getHeaderString = [](const std::string &title) {
      unsigned titleSize = title.size() + 2; // Add a space left and right
      return std::string((80 - titleSize) / 2, '-') + " " + title + " " +
             std::string((80 - titleSize + 1) / 2, '-') + "\n%s\n" +
             std::string(80, '-') + "\n";
    };

    // Get the pointer to the solver instance.
    Value solver = buildSolverPtr(rewriter, loc);

    // In debug-mode, print the state of the solver before calling 'check-sat'
    // on it. This prints the asserted SMT expressions.
    if (options.debug) {
      auto solverStringPtr =
          buildPtrAPICall(rewriter, loc, "Z3_solver_to_string", {solver});
      auto solverFormatString =
          buildString(rewriter, loc, getHeaderString("Solver"));
      buildCall(rewriter, op.getLoc(), "printf", printfType,
                {solverFormatString, solverStringPtr});
    }

    // Convert the result types of the `smt.check` operation.
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    // Call 'check-sat' and check if the assertions are satisfiable.
    Value checkResult =
        buildAPICallWithContext(rewriter, loc, "Z3_solver_check",
                                rewriter.getI32Type(), {solver})
            ->getResult(0);
    Value constOne =
        rewriter.create<LLVM::ConstantOp>(loc, checkResult.getType(), 1);
    Value isSat = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                checkResult, constOne);

    // Simply inline the 'sat' region into the 'then' region of the 'scf.if'
    auto satIfOp = rewriter.create<scf::IfOp>(loc, resultTypes, isSat);
    rewriter.inlineRegionBefore(op.getSatRegion(), satIfOp.getThenRegion(),
                                satIfOp.getThenRegion().end());

    // Otherwise, the 'else' block checks if the assertions are unsatisfiable or
    // unknown. The corresponding regions can also be simply inlined into the
    // two branches of this nested if-statement as well.
    rewriter.createBlock(&satIfOp.getElseRegion());
    Value constNegOne =
        rewriter.create<LLVM::ConstantOp>(loc, checkResult.getType(), -1);
    Value isUnsat = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                  checkResult, constNegOne);
    auto unsatIfOp = rewriter.create<scf::IfOp>(loc, resultTypes, isUnsat);
    rewriter.create<scf::YieldOp>(loc, unsatIfOp->getResults());

    rewriter.inlineRegionBefore(op.getUnsatRegion(), unsatIfOp.getThenRegion(),
                                unsatIfOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getUnknownRegion(),
                                unsatIfOp.getElseRegion(),
                                unsatIfOp.getElseRegion().end());

    rewriter.replaceOp(op, satIfOp->getResults());

    if (options.debug) {
      // In debug-mode, if the assertions are unsatisfiable we can print the
      // proof.
      rewriter.setInsertionPointToStart(unsatIfOp.thenBlock());
      auto proof = buildPtrAPICall(rewriter, op.getLoc(), "Z3_solver_get_proof",
                                   {solver});
      auto stringPtr =
          buildPtrAPICall(rewriter, op.getLoc(), "Z3_ast_to_string", {proof});
      auto formatString =
          buildString(rewriter, op.getLoc(), getHeaderString("Proof"));
      buildCall(rewriter, op.getLoc(), "printf", printfType,
                {formatString, stringPtr});

      // In debug mode, if the assertions are satisfiable we can print the model
      // (effectively a counter-example).
      rewriter.setInsertionPointToStart(satIfOp.thenBlock());
      auto model = buildPtrAPICall(rewriter, op.getLoc(), "Z3_solver_get_model",
                                   {solver});
      auto modelStringPtr =
          buildPtrAPICall(rewriter, op.getLoc(), "Z3_model_to_string", {model});
      auto modelFormatString =
          buildString(rewriter, op.getLoc(), getHeaderString("Model"));
      buildCall(rewriter, op.getLoc(), "printf", printfType,
                {modelFormatString, modelStringPtr});
    }

    return success();
  }
};

/// Lower `smt.forall` and `smt.exists` operations to the following Z3 API call.
/// ```
/// Z3_ast Z3_API Z3_mk_{forall|exists}_const(
///     Z3_context c,
///     unsigned weight,
///     unsigned num_bound,
///     Z3_app const bound[],
///     unsigned num_patterns,
///     Z3_pattern const patterns[],
///     Z3_ast body
///     );
/// ```
/// All nested regions are inlined into the parent region and the block
/// arguments are replaced with new `smt.declare_fun` constants that are also
/// passed to the `bound` argument of above API function. Patterns are created
/// with the following API function.
/// ```
/// Z3_pattern Z3_API Z3_mk_pattern(Z3_context c, unsigned num_patterns,
///                                 Z3_ast const terms[]);
/// ```
/// Where each operand of the `smt.yield` in a pattern region is a 'term'.
template <typename QuantifierOp>
struct QuantifierLowering : public SMTLoweringPattern<QuantifierOp> {
  using SMTLoweringPattern<QuantifierOp>::SMTLoweringPattern;
  using SMTLoweringPattern<QuantifierOp>::typeConverter;
  using SMTLoweringPattern<QuantifierOp>::buildPtrAPICall;
  using OpAdaptor = typename QuantifierOp::Adaptor;

  Value createStorageForValueList(ValueRange values, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type arrTy = LLVM::LLVMArrayType::get(ptrTy, values.size());
    Value constOne =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
    Value storage =
        rewriter.create<LLVM::AllocaOp>(loc, ptrTy, arrTy, constOne);
    Value array = rewriter.create<LLVM::UndefOp>(loc, arrTy);

    for (auto [i, val] : llvm::enumerate(values))
      array = rewriter.create<LLVM::InsertValueOp>(loc, array, val,
                                                   ArrayRef<int64_t>(i));

    rewriter.create<LLVM::StoreOp>(loc, array, storage);

    return storage;
  }

  LogicalResult
  matchAndRewrite(QuantifierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // no-pattern attribute not supported yet because the Z3 CAPI allows more
    // fine-grained control where a list of patterns to be banned can be given.
    // This means, the no-pattern attribute is equivalent to providing a list of
    // all possible sub-expressions in the quantifier body to the CAPI.
    if (adaptor.getNoPattern())
      return rewriter.notifyMatchFailure(
          op, "no-pattern attribute not yet supported!");

    rewriter.setInsertionPoint(op);

    // Weight attribute
    Value weight = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                     adaptor.getWeight());

    // Bound variables
    unsigned numDecls = op.getBody().getNumArguments();
    Value numDeclsVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), numDecls);

    // We replace the block arguments with constant symbolic values and inform
    // the quantifier API call which constants it should treat as bound
    // variables. We also need to make sure that we use the exact same SSA
    // values in the pattern regions since we lower constant declaration
    // operation to always produce fresh constants.
    SmallVector<Value> repl;
    for (auto [i, arg] : llvm::enumerate(op.getBody().getArguments())) {
      Value newArg;
      if (adaptor.getBoundVarNames().has_value())
        newArg = rewriter.create<smt::DeclareFunOp>(
            loc, arg.getType(),
            cast<StringAttr>((*adaptor.getBoundVarNames())[i]));
      else
        newArg = rewriter.create<smt::DeclareFunOp>(loc, arg.getType());
      repl.push_back(typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(arg.getType()), newArg));
    }

    Value boundStorage = createStorageForValueList(repl, loc, rewriter);

    // Body Expression
    auto yieldOp = cast<smt::YieldOp>(op.getBody().front().getTerminator());
    Value bodyExp = yieldOp.getValues()[0];
    rewriter.setInsertionPointAfterValue(bodyExp);
    bodyExp = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(bodyExp.getType()), bodyExp);
    rewriter.eraseOp(yieldOp);

    rewriter.inlineBlockBefore(&op.getBody().front(), op, repl);
    rewriter.setInsertionPoint(op);

    // Patterns
    unsigned numPatterns = adaptor.getPatterns().size();
    Value numPatternsVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), numPatterns);

    Value patternStorage;
    if (numPatterns > 0) {
      SmallVector<Value> patterns;
      for (Region *patternRegion : adaptor.getPatterns()) {
        auto yieldOp =
            cast<smt::YieldOp>(patternRegion->front().getTerminator());
        auto patternTerms = yieldOp.getOperands();

        rewriter.setInsertionPoint(yieldOp);
        SmallVector<Value> patternList;
        for (auto val : patternTerms)
          patternList.push_back(typeConverter->materializeTargetConversion(
              rewriter, loc, typeConverter->convertType(val.getType()), val));

        rewriter.eraseOp(yieldOp);
        rewriter.inlineBlockBefore(&patternRegion->front(), op, repl);

        rewriter.setInsertionPoint(op);
        Value numTerms = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), patternTerms.size());
        Value patternTermStorage =
            createStorageForValueList(patternList, loc, rewriter);
        Value pattern = buildPtrAPICall(rewriter, loc, "Z3_mk_pattern",
                                        {numTerms, patternTermStorage});

        patterns.emplace_back(pattern);
      }
      patternStorage = createStorageForValueList(patterns, loc, rewriter);
    } else {
      // If we set the num_patterns parameter to 0, we can just pass a nullptr
      // as storage.
      patternStorage = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);
    }

    StringRef apiCallName = "Z3_mk_forall_const";
    if (std::is_same_v<QuantifierOp, ExistsOp>)
      apiCallName = "Z3_mk_exists_const";
    Value quantifierExp =
        buildPtrAPICall(rewriter, loc, apiCallName,
                        {weight, numDeclsVal, boundStorage, numPatternsVal,
                         patternStorage, bodyExp});

    rewriter.replaceOp(op, quantifierExp);
    return success();
  }
};

/// Lower `smt.bv.repeat` operations to Z3 API function calls of the form
/// ```
/// Z3_ast Z3_API Z3_mk_repeat(Z3_context c, unsigned i, Z3_ast t1);
/// ```
struct RepeatOpLowering : public SMTLoweringPattern<RepeatOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(RepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value count = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), op.getCount());
    rewriter.replaceOp(op,
                       buildPtrAPICall(rewriter, op.getLoc(), "Z3_mk_repeat",
                                       {count, adaptor.getInput()}));
    return success();
  }
};

/// Lower `smt.bv.extract` operations to Z3 API function calls of the following
/// form, where the output bit-vector has size `n = high - low + 1`. This means,
/// both the 'high' and 'low' indices are inclusive.
/// ```
/// Z3_ast Z3_API Z3_mk_extract(Z3_context c, unsigned high, unsigned low,
/// Z3_ast t1);
/// ```
struct ExtractOpLowering : public SMTLoweringPattern<ExtractOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value low = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                  adaptor.getLowBit());
    Value high = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        adaptor.getLowBit() + op.getType().getWidth() - 1);
    rewriter.replaceOp(op, buildPtrAPICall(rewriter, loc, "Z3_mk_extract",
                                           {high, low, adaptor.getInput()}));
    return success();
  }
};

/// Lower `smt.array.broadcast` operations to Z3 API function calls of the form
/// ```
/// Z3_ast Z3_API Z3_mk_const_array(Z3_context c, Z3_sort domain, Z3_ast v);
/// ```
struct ArrayBroadcastOpLowering
    : public SMTLoweringPattern<smt::ArrayBroadcastOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(smt::ArrayBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto domainSort = buildSort(
        rewriter, op.getLoc(),
        cast<smt::ArrayType>(op.getResult().getType()).getDomainType());

    rewriter.replaceOp(op, buildPtrAPICall(rewriter, op.getLoc(),
                                           "Z3_mk_const_array",
                                           {domainSort, adaptor.getValue()}));
    return success();
  }
};

/// Lower the `smt.constant` operation to one of the following Z3 API function
/// calls depending on the value of the boolean attribute.
/// ```
/// Z3_ast Z3_API Z3_mk_true(Z3_context c);
/// Z3_ast Z3_API Z3_mk_false(Z3_context c);
/// ```
struct BoolConstantOpLowering : public SMTLoweringPattern<smt::BoolConstantOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(smt::BoolConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, buildPtrAPICall(rewriter, op.getLoc(),
                            adaptor.getValue() ? "Z3_mk_true" : "Z3_mk_false"));
    return success();
  }
};

/// Lower `smt.int.constant` operations to one of the following two Z3 API
/// function calls depending on whether the storage APInt has a bit-width that
/// fits in a `uint64_t`.
/// ```
/// Z3_sort Z3_API Z3_mk_int_sort(Z3_context c);
///
/// Z3_ast Z3_API Z3_mk_int64(Z3_context c, int64_t v, Z3_sort ty);
///
/// Z3_ast Z3_API Z3_mk_numeral(Z3_context c, Z3_string numeral, Z3_sort ty);
/// Z3_ast Z3_API Z3_mk_unary_minus(Z3_context c, Z3_ast arg);
/// ```
struct IntConstantOpLowering : public SMTLoweringPattern<smt::IntConstantOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(smt::IntConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value type = buildPtrAPICall(rewriter, loc, "Z3_mk_int_sort");
    if (adaptor.getValue().getBitWidth() <= 64) {
      Value val = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), adaptor.getValue().getSExtValue());
      rewriter.replaceOp(
          op, buildPtrAPICall(rewriter, loc, "Z3_mk_int64", {val, type}));
      return success();
    }

    std::string numeralStr;
    llvm::raw_string_ostream stream(numeralStr);
    stream << adaptor.getValue().abs();

    Value numeral = buildString(rewriter, loc, numeralStr);
    Value intNumeral =
        buildPtrAPICall(rewriter, loc, "Z3_mk_numeral", {numeral, type});

    if (adaptor.getValue().isNegative())
      intNumeral =
          buildPtrAPICall(rewriter, loc, "Z3_mk_unary_minus", intNumeral);

    rewriter.replaceOp(op, intNumeral);
    return success();
  }
};

/// Lower `smt.int.cmp` operations to one of the following Z3 API function calls
/// depending on the predicate.
/// ```
/// Z3_ast Z3_API Z3_mk_{{pred}}(Z3_context c, Z3_ast t1, Z3_ast t2);
/// ```
struct IntCmpOpLowering : public SMTLoweringPattern<IntCmpOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(IntCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op,
        buildPtrAPICall(rewriter, op.getLoc(),
                        "Z3_mk_" + stringifyIntPredicate(op.getPred()).str(),
                        {adaptor.getLhs(), adaptor.getRhs()}));
    return success();
  }
};

/// Lower `smt.int2bv` operations to the following Z3 API function calls.
/// ```
/// Z3_ast Z3_API Z3_mk_int2bv(Z3_context c, unsigned n, Z3_ast t1);
/// ```
struct Int2BVOpLowering : public SMTLoweringPattern<Int2BVOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(Int2BVOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value widthConst =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(),
                                          op.getResult().getType().getWidth());
    rewriter.replaceOp(op,
                       buildPtrAPICall(rewriter, op.getLoc(), "Z3_mk_int2bv",
                                       {widthConst, adaptor.getInput()}));
    return success();
  }
};

/// Lower `smt.bv2int` operations to the following Z3 API function call.
/// ```
/// Z3_ast Z3_API Z3_mk_bv2int(Z3_context c, Z3_ast t1, bool is_signed)
/// ```
struct BV2IntOpLowering : public SMTLoweringPattern<BV2IntOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(BV2IntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // FIXME: ideally we don't want to use i1 here, since bools can sometimes be
    // compiled to wider widths in LLVM
    Value isSignedConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI1Type(), op.getIsSigned());
    rewriter.replaceOp(op,
                       buildPtrAPICall(rewriter, op.getLoc(), "Z3_mk_bv2int",
                                       {adaptor.getInput(), isSignedConst}));
    return success();
  }
};

/// Lower `smt.bv.cmp` operations to one of the following Z3 API function calls,
/// performing two's complement comparison, depending on the predicate
/// attribute.
/// ```
/// Z3_ast Z3_API Z3_mk_bv{{pred}}(Z3_context c, Z3_ast t1, Z3_ast t2);
/// ```
struct BVCmpOpLowering : public SMTLoweringPattern<BVCmpOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(BVCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, buildPtrAPICall(rewriter, op.getLoc(),
                            "Z3_mk_bv" +
                                stringifyBVCmpPredicate(op.getPred()).str(),
                            {adaptor.getLhs(), adaptor.getRhs()}));
    return success();
  }
};

/// Expand the `smt.int.abs` operation to a `smt.ite` operation.
struct IntAbsOpLowering : public SMTLoweringPattern<IntAbsOp> {
  using SMTLoweringPattern::SMTLoweringPattern;

  LogicalResult
  matchAndRewrite(IntAbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value zero = rewriter.create<IntConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
    Value cmp = rewriter.create<IntCmpOp>(loc, IntPredicate::lt,
                                          adaptor.getInput(), zero);
    Value neg = rewriter.create<IntSubOp>(loc, zero, adaptor.getInput());
    rewriter.replaceOpWithNewOp<IteOp>(op, cmp, neg, adaptor.getInput());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerSMTToZ3LLVMPass
    : public circt::impl::LowerSMTToZ3LLVMBase<LowerSMTToZ3LLVMPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void circt::populateSMTToZ3LLVMTypeConverter(TypeConverter &converter) {
  converter.addConversion([](smt::BoolType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](smt::BitVectorType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](smt::ArrayType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](smt::IntType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](smt::SMTFuncType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](smt::SortType type) {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
}

void circt::populateSMTToZ3LLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    SMTGlobalsHandler &globals, const LowerSMTToZ3LLVMOptions &options) {
#define ADD_VARIADIC_PATTERN(OP, APINAME, MIN_NUM_ARGS)                        \
  patterns.add<VariadicSMTPattern<OP>>(/*NOLINT(bugprone-macro-parentheses)*/  \
                                       converter, patterns.getContext(),       \
                                       globals, options, APINAME,              \
                                       MIN_NUM_ARGS);

#define ADD_ONE_TO_ONE_PATTERN(OP, APINAME, NUM_ARGS)                          \
  patterns.add<OneToOneSMTPattern<OP>>(/*NOLINT(bugprone-macro-parentheses)*/  \
                                       converter, patterns.getContext(),       \
                                       globals, options, APINAME, NUM_ARGS);

  // Lower `smt.distinct` operations which allows a variadic number of operands
  // according to the `:pairwise` attribute. The Z3 API function supports a
  // variadic number of operands as well, i.e., a direct lowering is possible:
  // ```
  // Z3_ast Z3_API Z3_mk_distinct(Z3_context c, unsigned num_args, Z3_ast const
  // args[])
  // ```
  // The API function requires num_args > 1 which is guaranteed to be satisfied
  // because `smt.distinct` is verified to have > 1 operands.
  ADD_VARIADIC_PATTERN(DistinctOp, "Z3_mk_distinct", 2);

  // Lower `smt.and` operations which allows a variadic number of operands
  // according to the `:left-assoc` attribute. The Z3 API function supports a
  // variadic number of operands as well, i.e., a direct lowering is possible:
  // ```
  // Z3_ast Z3_API Z3_mk_and(Z3_context c, unsigned num_args, Z3_ast const
  // args[])
  // ```
  // The API function requires num_args > 1. This is not guaranteed by the
  // `smt.and` operation and thus the pattern will not apply when no operand is
  // present. The constant folder of the operation is assumed to fold this to
  // a constant 'true' (neutral element of AND).
  ADD_VARIADIC_PATTERN(AndOp, "Z3_mk_and", 2);

  // Lower `smt.or` operations which allows a variadic number of operands
  // according to the `:left-assoc` attribute. The Z3 API function supports a
  // variadic number of operands as well, i.e., a direct lowering is possible:
  // ```
  // Z3_ast Z3_API Z3_mk_or(Z3_context c, unsigned num_args, Z3_ast const
  // args[])
  // ```
  // The API function requires num_args > 1. This is not guaranteed by the
  // `smt.or` operation and thus the pattern will not apply when no operand is
  // present. The constant folder of the operation is assumed to fold this to
  // a constant 'false' (neutral element of OR).
  ADD_VARIADIC_PATTERN(OrOp, "Z3_mk_or", 2);

  // Lower `smt.not` operations to the following Z3 API function:
  // ```
  // Z3_ast Z3_API Z3_mk_not(Z3_context c, Z3_ast a);
  // ```
  ADD_ONE_TO_ONE_PATTERN(NotOp, "Z3_mk_not", 1);

  // Lower `smt.xor` operations which allows a variadic number of operands
  // according to the `:left-assoc` attribute. The Z3 API function, however,
  // only takes two operands.
  // ```
  // Z3_ast Z3_API Z3_mk_xor(Z3_context c, Z3_ast t1, Z3_ast t2);
  // ```
  // Therefore, we need to decompose the operation first to a sequence of XOR
  // operations matching the left associative behavior.
  patterns.add<LowerLeftAssocSMTPattern<XOrOp>>(
      converter, patterns.getContext(), globals, options);
  ADD_ONE_TO_ONE_PATTERN(XOrOp, "Z3_mk_xor", 2);

  // Lower `smt.implies` operations to the following Z3 API function:
  // ```
  // Z3_ast Z3_API Z3_mk_implies(Z3_context c, Z3_ast t1, Z3_ast t2);
  // ```
  ADD_ONE_TO_ONE_PATTERN(ImpliesOp, "Z3_mk_implies", 2);

  // All the bit-vector arithmetic and bitwise operations conveniently lower to
  // Z3 API function calls with essentially matching names and a one-to-one
  // correspondence of operands to call arguments.
  ADD_ONE_TO_ONE_PATTERN(BVNegOp, "Z3_mk_bvneg", 1);
  ADD_ONE_TO_ONE_PATTERN(BVAddOp, "Z3_mk_bvadd", 2);
  ADD_ONE_TO_ONE_PATTERN(BVMulOp, "Z3_mk_bvmul", 2);
  ADD_ONE_TO_ONE_PATTERN(BVURemOp, "Z3_mk_bvurem", 2);
  ADD_ONE_TO_ONE_PATTERN(BVSRemOp, "Z3_mk_bvsrem", 2);
  ADD_ONE_TO_ONE_PATTERN(BVSModOp, "Z3_mk_bvsmod", 2);
  ADD_ONE_TO_ONE_PATTERN(BVUDivOp, "Z3_mk_bvudiv", 2);
  ADD_ONE_TO_ONE_PATTERN(BVSDivOp, "Z3_mk_bvsdiv", 2);
  ADD_ONE_TO_ONE_PATTERN(BVShlOp, "Z3_mk_bvshl", 2);
  ADD_ONE_TO_ONE_PATTERN(BVLShrOp, "Z3_mk_bvlshr", 2);
  ADD_ONE_TO_ONE_PATTERN(BVAShrOp, "Z3_mk_bvashr", 2);
  ADD_ONE_TO_ONE_PATTERN(BVNotOp, "Z3_mk_bvnot", 1);
  ADD_ONE_TO_ONE_PATTERN(BVAndOp, "Z3_mk_bvand", 2);
  ADD_ONE_TO_ONE_PATTERN(BVOrOp, "Z3_mk_bvor", 2);
  ADD_ONE_TO_ONE_PATTERN(BVXOrOp, "Z3_mk_bvxor", 2);

  // The `smt.bv.concat` operation only supports two operands, just like the
  // Z3 API function.
  // ```
  // Z3_ast Z3_API Z3_mk_concat(Z3_context c, Z3_ast t1, Z3_ast t2);
  // ```
  ADD_ONE_TO_ONE_PATTERN(ConcatOp, "Z3_mk_concat", 2);

  // Lower the `smt.ite` operation to the following Z3 API function call, where
  // `t1` must have boolean sort.
  // ```
  // Z3_ast Z3_API Z3_mk_ite(Z3_context c, Z3_ast t1, Z3_ast t2, Z3_ast t3);
  // ```
  ADD_ONE_TO_ONE_PATTERN(IteOp, "Z3_mk_ite", 3);

  // Lower the `smt.array.select` operation to the following Z3 function call.
  // The operand declaration of the operation matches the order of arguments of
  // the API function.
  // ```
  // Z3_ast Z3_API Z3_mk_select(Z3_context c, Z3_ast a, Z3_ast i);
  // ```
  // Where `a` is the array expression and `i` is the index expression.
  ADD_ONE_TO_ONE_PATTERN(ArraySelectOp, "Z3_mk_select", 2);

  // Lower the `smt.array.store` operation to the following Z3 function call.
  // The operand declaration of the operation matches the order of arguments of
  // the API function.
  // ```
  // Z3_ast Z3_API Z3_mk_store(Z3_context c, Z3_ast a, Z3_ast i, Z3_ast v);
  // ```
  // Where `a` is the array expression, `i` is the index expression, and `v` is
  // the value expression to be stored.
  ADD_ONE_TO_ONE_PATTERN(ArrayStoreOp, "Z3_mk_store", 3);

  // Lower the `smt.int.add` operation to the following Z3 API function call.
  // ```
  // Z3_ast Z3_API Z3_mk_add(Z3_context c, unsigned num_args, Z3_ast const
  // args[]);
  // ```
  // The number of arguments must be greater than zero. Therefore, the pattern
  // will fail if applied to an operation with less than two operands.
  ADD_VARIADIC_PATTERN(IntAddOp, "Z3_mk_add", 2);

  // Lower the `smt.int.mul` operation to the following Z3 API function call.
  // ```
  // Z3_ast Z3_API Z3_mk_mul(Z3_context c, unsigned num_args, Z3_ast const
  // args[]);
  // ```
  // The number of arguments must be greater than zero. Therefore, the pattern
  // will fail if applied to an operation with less than two operands.
  ADD_VARIADIC_PATTERN(IntMulOp, "Z3_mk_mul", 2);

  // Lower the `smt.int.sub` operation to the following Z3 API function call.
  // ```
  // Z3_ast Z3_API Z3_mk_sub(Z3_context c, unsigned num_args, Z3_ast const
  // args[]);
  // ```
  // The number of arguments must be greater than zero. Since the `smt.int.sub`
  // operation always has exactly two operands, this trivially holds.
  ADD_VARIADIC_PATTERN(IntSubOp, "Z3_mk_sub", 2);

  // Lower the `smt.int.div` operation to the following Z3 API function call.
  // ```
  // Z3_ast Z3_API Z3_mk_div(Z3_context c, Z3_ast arg1, Z3_ast arg2);
  // ```
  ADD_ONE_TO_ONE_PATTERN(IntDivOp, "Z3_mk_div", 2);

  // Lower the `smt.int.mod` operation to the following Z3 API function call.
  // ```
  // Z3_ast Z3_API Z3_mk_mod(Z3_context c, Z3_ast arg1, Z3_ast arg2);
  // ```
  ADD_ONE_TO_ONE_PATTERN(IntModOp, "Z3_mk_mod", 2);

#undef ADD_VARIADIC_PATTERN
#undef ADD_ONE_TO_ONE_PATTERN

  // Lower `smt.eq` operations which allows a variadic number of operands
  // according to the `:chainable` attribute. The Z3 API function does not
  // support a variadic number of operands, but exactly two:
  // ```
  // Z3_ast Z3_API Z3_mk_eq(Z3_context c, Z3_ast l, Z3_ast r)
  // ```
  // As a result, we first apply a rewrite pattern that unfolds chainable
  // operators and then lower it one-to-one to the API function. In this case,
  // this means:
  // ```
  // eq(a,b,c,d) ->
  // and(eq(a,b), eq(b,c), eq(c,d)) ->
  // and(Z3_mk_eq(ctx, a, b), Z3_mk_eq(ctx, b, c), Z3_mk_eq(ctx, c, d))
  // ```
  // The patterns for `smt.and` will then do the remaining work.
  patterns.add<LowerChainableSMTPattern<EqOp>>(converter, patterns.getContext(),
                                               globals, options);
  patterns.add<OneToOneSMTPattern<EqOp>>(converter, patterns.getContext(),
                                         globals, options, "Z3_mk_eq", 2);

  // Other lowering patterns. Refer to their implementation directly for more
  // information.
  patterns.add<BVConstantOpLowering, DeclareFunOpLowering, AssertOpLowering,
               ResetOpLowering, PushOpLowering, PopOpLowering, CheckOpLowering,
               SolverOpLowering, ApplyFuncOpLowering, YieldOpLowering,
               RepeatOpLowering, ExtractOpLowering, BoolConstantOpLowering,
               IntConstantOpLowering, ArrayBroadcastOpLowering, BVCmpOpLowering,
               IntCmpOpLowering, IntAbsOpLowering, Int2BVOpLowering,
               BV2IntOpLowering, QuantifierLowering<ForallOp>,
               QuantifierLowering<ExistsOp>>(converter, patterns.getContext(),
                                             globals, options);
}

void LowerSMTToZ3LLVMPass::runOnOperation() {
  LowerSMTToZ3LLVMOptions options;
  options.debug = debug;

  // Check that the lowering is possible
  // Specifically, check that the use of set-logic ops is valid for z3
  auto setLogicCheck = getOperation().walk([&](SolverOp solverOp)
                                               -> WalkResult {
    // Check that solver ops only contain one set-logic op and that they're at
    // the start of the body
    auto setLogicOps = solverOp.getBodyRegion().getOps<smt::SetLogicOp>();
    auto numSetLogicOps = std::distance(setLogicOps.begin(), setLogicOps.end());
    if (numSetLogicOps > 1) {
      return solverOp.emitError(
          "multiple set-logic operations found in one solver operation - Z3 "
          "only supports setting the logic once");
    }
    if (numSetLogicOps == 1)
      // Check the only ops before the set-logic op are ConstantLike
      for (auto &blockOp : solverOp.getBodyRegion().getOps()) {
        if (isa<smt::SetLogicOp>(blockOp))
          break;
        if (!blockOp.hasTrait<OpTrait::ConstantLike>()) {
          return solverOp.emitError("set-logic operation must be the first "
                                    "non-constant operation in a solver "
                                    "operation");
        }
      }
    return WalkResult::advance();
  });
  if (setLogicCheck.wasInterrupted())
    return signalPassFailure();

  // Set up the type converter
  LLVMTypeConverter converter(&getContext());
  populateSMTToZ3LLVMTypeConverter(converter);

  RewritePatternSet patterns(&getContext());

  // Populate the func to LLVM conversion patterns for two reasons:
  // * Typically functions are represented using `func.func` and including the
  //   patterns to lower them here is more convenient for most lowering
  //   pipelines (avoids running another pass).
  // * Already having `llvm.func` in the input or lowering `func.func` before
  //   the SMT in the body leads to issues because the SCF conversion patterns
  //   don't take the type converter into consideration and thus create blocks
  //   with the old types for block arguments. However, the conversion happens
  //   top-down and thus are assumed to be converted by the parent function op
  //   which at that point would have already been lowered (and the blocks are
  //   also not there when doing everything in one pass, i.e.,
  //   `populateAnyFunctionOpInterfaceTypeConversionPattern` does not have any
  //   effect as well). Are the SCF lowering patterns actually broken and should
  //   take a type-converter?
  populateFuncToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);

  // Populate SCF to CF and CF to LLVM lowering patterns because we create
  // `scf.if` operations in the lowering patterns for convenience (given the
  // above issue we might want to lower to LLVM directly; or fix upstream?)
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

  // Create the globals to store the context and solver and populate the SMT
  // lowering patterns.
  OpBuilder builder(&getContext());
  auto globals = SMTGlobalsHandler::create(builder, getOperation());
  populateSMTToZ3LLVMConversionPatterns(patterns, converter, globals, options);

  // Do a full conversion. This assumes that all other dialects have been
  // lowered before this pass already.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<scf::YieldOp>();

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
