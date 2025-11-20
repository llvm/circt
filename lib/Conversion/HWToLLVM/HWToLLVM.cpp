//===- HWToLLVM.cpp - HW to LLVM Conversion Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTHWTOLLVM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Endianess Converter
//===----------------------------------------------------------------------===//

uint32_t
circt::HWToLLVMEndianessConverter::convertToLLVMEndianess(Type type,
                                                          uint32_t index) {
  // This is hardcoded for little endian machines for now.
  return TypeSwitch<Type, uint32_t>(type)
      .Case<hw::ArrayType>(
          [&](hw::ArrayType ty) { return ty.getNumElements() - index - 1; })
      .Case<hw::StructType>([&](hw::StructType ty) {
        return ty.getElements().size() - index - 1;
      });
}

uint32_t
circt::HWToLLVMEndianessConverter::llvmIndexOfStructField(hw::StructType type,
                                                          StringRef fieldName) {
  auto fieldIter = type.getElements();
  size_t index = 0;

  for (const auto *iter = fieldIter.begin(); iter != fieldIter.end(); ++iter) {
    if (iter->name == fieldName) {
      return HWToLLVMEndianessConverter::convertToLLVMEndianess(type, index);
    }
    ++index;
  }

  // Verifier of StructExtractOp has to ensure that the field name is indeed
  // present.
  llvm_unreachable("Field name attribute of hw::StructExtractOp invalid");
  return 0;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a zext operation by one bit on the given value. This is useful when
/// passing unsigned indexes to a GEP instruction, which treats indexes as
/// signed values, to avoid unexpected "sign overflows".
static Value zextByOne(Location loc, ConversionPatternRewriter &rewriter,
                       Value value) {
  auto valueTy = value.getType();
  auto zextTy = IntegerType::get(valueTy.getContext(),
                                 valueTy.getIntOrFloatBitWidth() + 1);
  return LLVM::ZExtOp::create(rewriter, loc, zextTy, value);
}

//===----------------------------------------------------------------------===//
// HWToLLVMArraySpillCache
//===----------------------------------------------------------------------===//

static Value spillValueOnStack(OpBuilder &builder, Location loc,
                               Value spillVal) {
  auto oneC = LLVM::ConstantOp::create(
      builder, loc, IntegerType::get(builder.getContext(), 32),
      builder.getI32IntegerAttr(1));
  Value ptr = LLVM::AllocaOp::create(
      builder, loc, LLVM::LLVMPointerType::get(builder.getContext()),
      spillVal.getType(), oneC,
      /*alignment=*/4);
  LLVM::StoreOp::create(builder, loc, spillVal, ptr);
  return ptr;
}

void HWToLLVMArraySpillCache::spillNonHWOps(OpBuilder &builder,
                                            LLVMTypeConverter &converter,
                                            Operation *containerOp) {
  OpBuilder::InsertionGuard g(builder);
  containerOp->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](Operation *op) {
        if (isa_and_nonnull<hw::HWDialect>(op->getDialect()))
          return;
        auto hasSpillingUser = [](Value arrVal) -> bool {
          for (auto user : arrVal.getUsers())
            if (isa<hw::ArrayGetOp, hw::ArraySliceOp>(user))
              return true;
          return false;
        };
        // Spill Block arguments
        for (auto &region : op->getRegions()) {
          for (auto &block : region.getBlocks()) {
            builder.setInsertionPointToStart(&block);
            for (auto &arg : block.getArguments()) {
              if (isa<hw::ArrayType>(arg.getType()) && hasSpillingUser(arg))
                spillHWArrayValue(builder, arg.getLoc(), converter, arg);
            }
          }
        }
        // Spill Op Results
        for (auto result : op->getResults()) {
          if (isa<hw::ArrayType>(result.getType()) && hasSpillingUser(result)) {
            builder.setInsertionPointAfter(op);
            spillHWArrayValue(builder, op->getLoc(), converter, result);
          }
        }
      });
}

void HWToLLVMArraySpillCache::map(Value arrayValue, Value bufferPtr) {
  assert(llvm::isa<LLVM::LLVMArrayType>(arrayValue.getType()) &&
         "Key is not an LLVM array.");
  assert(llvm::isa<LLVM::LLVMPointerType>(bufferPtr.getType()) &&
         "Value is not a pointer.");
  spillMap.insert({arrayValue, bufferPtr});
}

Value HWToLLVMArraySpillCache::lookup(Value arrayValue) {
  assert(isa<LLVM::LLVMArrayType>(arrayValue.getType()) ||
         isa<hw::ArrayType>(arrayValue.getType()) && "Not an array value");
  while (isa<LLVM::LLVMArrayType>(arrayValue.getType()) ||
         isa<hw::ArrayType>(arrayValue.getType())) {
    if (isa<LLVM::LLVMArrayType>(arrayValue.getType())) {
      auto mapVal = spillMap.lookup(arrayValue);
      if (mapVal)
        return mapVal;
    }
    if (auto castOp = arrayValue.getDefiningOp<UnrealizedConversionCastOp>())
      arrayValue = castOp.getOperand(0);
    else
      break;
  }
  return {};
}

// Materialize a LLVM Array value in a stack allocated buffer.
Value HWToLLVMArraySpillCache::spillLLVMArrayValue(OpBuilder &builder,
                                                   Location loc,
                                                   Value llvmArray) {
  assert(isa<LLVM::LLVMArrayType>(llvmArray.getType()) &&
         "Expected an LLVM array.");
  auto spillBuffer = spillValueOnStack(builder, loc, llvmArray);
  auto loadOp =
      LLVM::LoadOp::create(builder, loc, llvmArray.getType(), spillBuffer);
  map(loadOp.getResult(), spillBuffer);
  return loadOp.getResult();
}

// Materialize a HW Array value in a stack allocated buffer. Replaces
// all current uses of the SSA value with a new SSA representing the same
// array value.
Value HWToLLVMArraySpillCache::spillHWArrayValue(OpBuilder &builder,
                                                 Location loc,
                                                 LLVMTypeConverter &converter,
                                                 Value hwArray) {
  assert(isa<hw::ArrayType>(hwArray.getType()) && "Expected an HW array");
  auto targetType = converter.convertType(hwArray.getType());
  auto hwToLLVMCast =
      UnrealizedConversionCastOp::create(builder, loc, targetType, hwArray);
  auto spilled = spillLLVMArrayValue(builder, loc, hwToLLVMCast.getResult(0));
  auto llvmToHWCast = UnrealizedConversionCastOp::create(
      builder, loc, hwArray.getType(), spilled);
  hwArray.replaceAllUsesExcept(llvmToHWCast.getResult(0), hwToLLVMCast);
  return llvmToHWCast.getResult(0);
}

namespace {
// Helper for patterns using or creating buffers containing
// HW array values.
template <typename SourceOp>
struct HWArrayOpToLLVMPattern : public ConvertOpToLLVMPattern<SourceOp> {

  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  HWArrayOpToLLVMPattern(LLVMTypeConverter &converter,
                         std::optional<HWToLLVMArraySpillCache> &spillCacheOpt)
      : ConvertOpToLLVMPattern<SourceOp>(converter),
        spillCacheOpt(spillCacheOpt) {}

protected:
  std::optional<HWToLLVMArraySpillCache> &spillCacheOpt;
};

} // namespace

//===----------------------------------------------------------------------===//
// Extraction operation conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert a StructExplodeOp to the LLVM dialect.
/// Pattern: struct_explode(input) =>
///          struct_extract(input, structElements_index(index)) ...
struct StructExplodeOpConversion
    : public ConvertOpToLLVMPattern<hw::StructExplodeOp> {
  using ConvertOpToLLVMPattern<hw::StructExplodeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructExplodeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value> replacements;

    for (size_t i = 0,
                e = cast<LLVM::LLVMStructType>(adaptor.getInput().getType())
                        .getBody()
                        .size();
         i < e; ++i)

      replacements.push_back(LLVM::ExtractValueOp::create(
          rewriter, op->getLoc(), adaptor.getInput(),
          HWToLLVMEndianessConverter::convertToLLVMEndianess(
              op.getInput().getType(), i)));

    rewriter.replaceOp(op, replacements);
    return success();
  }
};
} // namespace

namespace {

/// Convert a StructExtractOp to LLVM dialect.
/// Pattern: struct_extract(input, fieldname) =>
///   extractvalue(input, fieldname_to_index(fieldname))
struct StructExtractOpConversion
    : public ConvertOpToLLVMPattern<hw::StructExtractOp> {
  using ConvertOpToLLVMPattern<hw::StructExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    uint32_t fieldIndex = HWToLLVMEndianessConverter::convertToLLVMEndianess(
        op.getInput().getType(), op.getFieldIndex());
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                      fieldIndex);
    return success();
  }
};
} // namespace

namespace {
/// Convert an ArrayInjectOp to the LLVM dialect.
/// Pattern: array_inject(input, element, index) =>
///   store(gep(store(input, alloca), zext(index)), element)
///   load(alloca)
struct ArrayInjectOpConversion
    : public HWArrayOpToLLVMPattern<hw::ArrayInjectOp> {
  using HWArrayOpToLLVMPattern<hw::ArrayInjectOp>::HWArrayOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = cast<hw::ArrayType>(op.getInput().getType());
    auto oldArrTy = adaptor.getInput().getType();
    auto newArrTy = oldArrTy;
    const size_t arrElems = inputType.getNumElements();

    if (arrElems == 0) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    auto oneC =
        LLVM::ConstantOp::create(rewriter, op->getLoc(), rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(1));
    auto zextIndex = zextByOne(op->getLoc(), rewriter, op.getIndex());

    Value arrPtr;
    if (arrElems == 1 || !llvm::isPowerOf2_64(arrElems)) {
      // Clamp index to prevent OOB access. We add an extra element to the
      // array so that OOB access modifies this element, leaving the original
      // array intact.
      auto maxIndex =
          LLVM::ConstantOp::create(rewriter, op->getLoc(), zextIndex.getType(),
                                   rewriter.getI32IntegerAttr(arrElems));
      zextIndex =
          LLVM::UMinOp::create(rewriter, op->getLoc(), zextIndex, maxIndex);

      newArrTy = typeConverter->convertType(
          hw::ArrayType::get(inputType.getElementType(), arrElems + 1));
      arrPtr = LLVM::AllocaOp::create(
          rewriter, op->getLoc(),
          LLVM::LLVMPointerType::get(rewriter.getContext()), newArrTy, oneC,
          /*alignment=*/4);
    } else {
      arrPtr = LLVM::AllocaOp::create(
          rewriter, op->getLoc(),
          LLVM::LLVMPointerType::get(rewriter.getContext()), newArrTy, oneC,
          /*alignment=*/4);
    }

    LLVM::StoreOp::create(rewriter, op->getLoc(), adaptor.getInput(), arrPtr);

    auto gep = LLVM::GEPOp::create(
        rewriter, op->getLoc(),
        LLVM::LLVMPointerType::get(rewriter.getContext()), newArrTy, arrPtr,
        ArrayRef<LLVM::GEPArg>{0, zextIndex});

    LLVM::StoreOp::create(rewriter, op->getLoc(), adaptor.getElement(), gep);
    auto loadOp =
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, oldArrTy, arrPtr);
    if (spillCacheOpt)
      spillCacheOpt->map(loadOp, arrPtr);
    return success();
  }
};
} // namespace

namespace {
/// Convert an ArrayGetOp to the LLVM dialect.
/// Pattern: array_get(input, index) =>
///   load(gep(store(input, alloca), zext(index)))
struct ArrayGetOpConversion : public HWArrayOpToLLVMPattern<hw::ArrayGetOp> {
  using HWArrayOpToLLVMPattern<hw::ArrayGetOp>::HWArrayOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value arrPtr;
    if (spillCacheOpt)
      arrPtr = spillCacheOpt->lookup(adaptor.getInput());
    if (!arrPtr)
      arrPtr = spillValueOnStack(rewriter, op.getLoc(), adaptor.getInput());

    auto arrTy = typeConverter->convertType(op.getInput().getType());
    auto elemTy = typeConverter->convertType(op.getResult().getType());
    auto zextIndex = zextByOne(op->getLoc(), rewriter, op.getIndex());

    // During the ongoing migration to opaque types, use the constructor that
    // accepts an element type when the array pointer type is opaque, and
    // otherwise use the typed pointer constructor.
    auto gep = LLVM::GEPOp::create(
        rewriter, op->getLoc(),
        LLVM::LLVMPointerType::get(rewriter.getContext()), arrTy, arrPtr,
        ArrayRef<LLVM::GEPArg>{0, zextIndex});
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemTy, gep);

    return success();
  }
};
} // namespace

namespace {
/// Convert an ArraySliceOp to the LLVM dialect.
/// Pattern: array_slice(input, lowIndex) =>
///   load(bitcast(gep(store(input, alloca), zext(lowIndex))))
struct ArraySliceOpConversion
    : public HWArrayOpToLLVMPattern<hw::ArraySliceOp> {
  using HWArrayOpToLLVMPattern<hw::ArraySliceOp>::HWArrayOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dstTy = typeConverter->convertType(op.getDst().getType());

    Value arrPtr;
    if (spillCacheOpt)
      arrPtr = spillCacheOpt->lookup(adaptor.getInput());
    if (!arrPtr)
      arrPtr = spillValueOnStack(rewriter, op.getLoc(), adaptor.getInput());

    auto zextIndex = zextByOne(op->getLoc(), rewriter, op.getLowIndex());

    // During the ongoing migration to opaque types, use the constructor that
    // accepts an element type when the array pointer type is opaque, and
    // otherwise use the typed pointer constructor.
    auto gep = LLVM::GEPOp::create(
        rewriter, op->getLoc(),
        LLVM::LLVMPointerType::get(rewriter.getContext()), dstTy, arrPtr,
        ArrayRef<LLVM::GEPArg>{0, zextIndex});

    auto loadOp = rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dstTy, gep);

    if (spillCacheOpt)
      spillCacheOpt->map(loadOp, gep);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Insertion operations conversion
//===----------------------------------------------------------------------===//

namespace {
/// Convert a StructInjectOp to LLVM dialect.
/// Pattern: struct_inject(input, index, value) =>
///   insertvalue(input, value, index)
struct StructInjectOpConversion
    : public ConvertOpToLLVMPattern<hw::StructInjectOp> {
  using ConvertOpToLLVMPattern<hw::StructInjectOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    uint32_t fieldIndex = HWToLLVMEndianessConverter::convertToLLVMEndianess(
        op.getInput().getType(), op.getFieldIndex());

    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getInput(), adaptor.getNewValue(), fieldIndex);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Concat operations conversion
//===----------------------------------------------------------------------===//

namespace {
/// Lower an ArrayConcatOp operation to the LLVM dialect.
struct ArrayConcatOpConversion
    : public HWArrayOpToLLVMPattern<hw::ArrayConcatOp> {
  using HWArrayOpToLLVMPattern<hw::ArrayConcatOp>::HWArrayOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    hw::ArrayType arrTy = cast<hw::ArrayType>(op.getResult().getType());
    Type resultTy = typeConverter->convertType(arrTy);
    auto loc = op.getLoc();

    Value arr = LLVM::UndefOp::create(rewriter, loc, resultTy);

    // Attention: j is hardcoded for little endian machines.
    size_t j = op.getInputs().size() - 1, k = 0;

    for (size_t i = 0, e = arrTy.getNumElements(); i < e; ++i) {
      Value element = LLVM::ExtractValueOp::create(rewriter, loc,
                                                   adaptor.getInputs()[j], k);
      arr = LLVM::InsertValueOp::create(rewriter, loc, arr, element, i);

      ++k;
      if (k >=
          cast<hw::ArrayType>(op.getInputs()[j].getType()).getNumElements()) {
        k = 0;
        --j;
      }
    }

    rewriter.replaceOp(op, arr);

    // If we've got a cache, spill the array right away.
    if (spillCacheOpt) {
      rewriter.setInsertionPointAfter(arr.getDefiningOp());
      auto ptr = spillValueOnStack(rewriter, loc, arr);
      spillCacheOpt->map(arr, ptr);
    }
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Bitwise conversions
//===----------------------------------------------------------------------===//

namespace {
/// Lower an ArrayConcatOp operation to the LLVM dialect.
/// Pattern: hw.bitcast(input) ==> load(bitcast_ptr(store(input, alloca)))
/// This is necessary because we cannot bitcast aggregate types directly in
/// LLVMIR.
struct BitcastOpConversion : public ConvertOpToLLVMPattern<hw::BitcastOp> {
  using ConvertOpToLLVMPattern<hw::BitcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultTy = typeConverter->convertType(op.getResult().getType());
    auto ptr = spillValueOnStack(rewriter, op.getLoc(), adaptor.getInput());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultTy, ptr);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Value creation conversions
//===----------------------------------------------------------------------===//

namespace {
struct HWConstantOpConversion : public ConvertToLLVMPattern {
  explicit HWConstantOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(hw::ConstantOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the ConstOp.
    auto constOp = cast<hw::ConstantOp>(op);
    // Get the converted llvm type.
    auto intType = typeConverter->convertType(constOp.getValueAttr().getType());
    // Replace the operation with an llvm constant op.
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, intType,
                                                  constOp.getValueAttr());

    return success();
  }
};
} // namespace

namespace {
/// Convert an ArrayCreateOp with dynamic elements to the LLVM dialect. An
/// equivalent and initialized llvm dialect array type is generated.
struct HWDynamicArrayCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::ArrayCreateOp> {
  using ConvertOpToLLVMPattern<hw::ArrayCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayTy = typeConverter->convertType(op->getResult(0).getType());
    assert(arrayTy);

    Value arr = LLVM::UndefOp::create(rewriter, op->getLoc(), arrayTy);
    for (size_t i = 0, e = op.getInputs().size(); i < e; ++i) {
      Value input =
          adaptor
              .getInputs()[HWToLLVMEndianessConverter::convertToLLVMEndianess(
                  op.getResult().getType(), i)];
      arr = LLVM::InsertValueOp::create(rewriter, op->getLoc(), arr, input, i);
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};
} // namespace

namespace {

/// Convert an ArrayCreateOp with constant elements to the LLVM dialect. An
/// equivalent and initialized llvm dialect array type is generated.
class AggregateConstantOpConversion
    : public HWArrayOpToLLVMPattern<hw::AggregateConstantOp> {
  using HWArrayOpToLLVMPattern<hw::AggregateConstantOp>::HWArrayOpToLLVMPattern;

  bool containsArrayAndStructAggregatesOnly(Type type) const;

  bool isMultiDimArrayOfIntegers(Type type,
                                 SmallVectorImpl<int64_t> &dims) const;

  void flatten(Type type, Attribute attr,
               SmallVectorImpl<Attribute> &output) const;

  Value constructAggregate(OpBuilder &builder,
                           const TypeConverter &typeConverter, Location loc,
                           Type type, Attribute data) const;

public:
  explicit AggregateConstantOpConversion(
      LLVMTypeConverter &typeConverter,
      DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp>
          &constAggregateGlobalsMap,
      Namespace &globals, std::optional<HWToLLVMArraySpillCache> &spillCacheOpt)
      : HWArrayOpToLLVMPattern(typeConverter, spillCacheOpt),
        constAggregateGlobalsMap(constAggregateGlobalsMap), globals(globals) {}

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp>
      &constAggregateGlobalsMap;
  Namespace &globals;
};
} // namespace

namespace {
/// Convert a StructCreateOp operation to the LLVM dialect. An equivalent and
/// initialized llvm dialect struct type is generated.
struct HWStructCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::StructCreateOp> {
  using ConvertOpToLLVMPattern<hw::StructCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resTy = typeConverter->convertType(op.getResult().getType());

    Value tup = LLVM::UndefOp::create(rewriter, op->getLoc(), resTy);
    for (size_t i = 0, e = cast<LLVM::LLVMStructType>(resTy).getBody().size();
         i < e; ++i) {
      Value input =
          adaptor.getInput()[HWToLLVMEndianessConverter::convertToLLVMEndianess(
              op.getResult().getType(), i)];
      tup = LLVM::InsertValueOp::create(rewriter, op->getLoc(), tup, input, i);
    }

    rewriter.replaceOp(op, tup);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern implementations
//===----------------------------------------------------------------------===//

bool AggregateConstantOpConversion::containsArrayAndStructAggregatesOnly(
    Type type) const {
  if (auto intType = dyn_cast<IntegerType>(type))
    return true;

  if (auto arrTy = dyn_cast<hw::ArrayType>(type))
    return containsArrayAndStructAggregatesOnly(arrTy.getElementType());

  if (auto structTy = dyn_cast<hw::StructType>(type)) {
    SmallVector<Type> innerTypes;
    structTy.getInnerTypes(innerTypes);
    return llvm::all_of(innerTypes, [&](auto ty) {
      return containsArrayAndStructAggregatesOnly(ty);
    });
  }

  return false;
}

bool AggregateConstantOpConversion::isMultiDimArrayOfIntegers(
    Type type, SmallVectorImpl<int64_t> &dims) const {
  if (auto intType = dyn_cast<IntegerType>(type))
    return true;

  if (auto arrTy = dyn_cast<hw::ArrayType>(type)) {
    dims.push_back(arrTy.getNumElements());
    return isMultiDimArrayOfIntegers(arrTy.getElementType(), dims);
  }

  return false;
}

void AggregateConstantOpConversion::flatten(
    Type type, Attribute attr, SmallVectorImpl<Attribute> &output) const {
  if (isa<IntegerType>(type)) {
    assert(isa<IntegerAttr>(attr));
    output.push_back(attr);
    return;
  }

  auto arrAttr = cast<ArrayAttr>(attr);
  for (size_t i = 0, e = arrAttr.size(); i < e; ++i) {
    auto element =
        arrAttr[HWToLLVMEndianessConverter::convertToLLVMEndianess(type, i)];

    flatten(cast<hw::ArrayType>(type).getElementType(), element, output);
  }
}

Value AggregateConstantOpConversion::constructAggregate(
    OpBuilder &builder, const TypeConverter &typeConverter, Location loc,
    Type type, Attribute data) const {
  Type llvmType = typeConverter.convertType(type);

  auto getElementType = [](Type type, size_t index) {
    if (auto arrTy = dyn_cast<hw::ArrayType>(type)) {
      return arrTy.getElementType();
    }

    assert(isa<hw::StructType>(type));
    auto structTy = cast<hw::StructType>(type);
    SmallVector<Type> innerTypes;
    structTy.getInnerTypes(innerTypes);
    return innerTypes[index];
  };

  return TypeSwitch<Type, Value>(type)
      .Case<IntegerType>([&](auto ty) {
        return LLVM::ConstantOp::create(builder, loc, cast<TypedAttr>(data));
      })
      .Case<hw::ArrayType, hw::StructType>([&](auto ty) {
        Value aggVal = LLVM::UndefOp::create(builder, loc, llvmType);
        auto arrayAttr = cast<ArrayAttr>(data);
        for (size_t i = 0, e = arrayAttr.size(); i < e; ++i) {
          size_t currIdx =
              HWToLLVMEndianessConverter::convertToLLVMEndianess(type, i);
          Attribute input = arrayAttr[currIdx];
          Type elementType = getElementType(ty, currIdx);

          Value element = constructAggregate(builder, typeConverter, loc,
                                             elementType, input);
          aggVal =
              LLVM::InsertValueOp::create(builder, loc, aggVal, element, i);
        }

        return aggVal;
      });
}

LogicalResult AggregateConstantOpConversion::matchAndRewrite(
    hw::AggregateConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type aggregateType = op.getResult().getType();

  // TODO: Only arrays and structs supported at the moment.
  if (!containsArrayAndStructAggregatesOnly(aggregateType))
    return failure();

  auto llvmTy = typeConverter->convertType(op.getResult().getType());
  auto typeAttrPair = std::make_pair(aggregateType, adaptor.getFields());

  if (!constAggregateGlobalsMap.count(typeAttrPair) ||
      !constAggregateGlobalsMap[typeAttrPair]) {
    auto ipSave = rewriter.saveInsertionPoint();

    Operation *parent = op->getParentOp();
    while (!isa<mlir::ModuleOp>(parent->getParentOp())) {
      parent = parent->getParentOp();
    }

    rewriter.setInsertionPoint(parent);

    // Create a global region for this static array.
    auto name = globals.newName("_aggregate_const_global");

    SmallVector<int64_t> dims;
    if (isMultiDimArrayOfIntegers(aggregateType, dims)) {
      SmallVector<Attribute> ints;
      flatten(aggregateType, adaptor.getFields(), ints);
      assert(!ints.empty());
      auto shapedType = RankedTensorType::get(
          dims, cast<IntegerAttr>(ints.front()).getType());
      auto denseAttr = DenseElementsAttr::get(shapedType, ints);

      constAggregateGlobalsMap[typeAttrPair] =
          LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmTy, true,
                                 LLVM::Linkage::Internal, name, denseAttr);
    } else {
      auto global =
          LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmTy, false,
                                 LLVM::Linkage::Internal, name, Attribute());
      Block *blk = new Block();
      global.getInitializerRegion().push_back(blk);
      rewriter.setInsertionPointToStart(blk);

      Value aggregate =
          constructAggregate(rewriter, *typeConverter, op.getLoc(),
                             aggregateType, adaptor.getFields());
      LLVM::ReturnOp::create(rewriter, op.getLoc(), aggregate);
      constAggregateGlobalsMap[typeAttrPair] = global;
    }

    rewriter.restoreInsertionPoint(ipSave);
  }

  // Get the global array address and load it to return an array value.
  auto addr = LLVM::AddressOfOp::create(rewriter, op->getLoc(),
                                        constAggregateGlobalsMap[typeAttrPair]);
  auto newOp = rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, llvmTy, addr);

  if (spillCacheOpt && llvm::isa<hw::ArrayType>(aggregateType))
    spillCacheOpt->map(newOp.getResult(), addr);

  return success();
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

static Type convertArrayType(hw::ArrayType type, LLVMTypeConverter &converter) {
  auto elementTy = converter.convertType(type.getElementType());
  return LLVM::LLVMArrayType::get(elementTy, type.getNumElements());
}

static Type convertStructType(hw::StructType type,
                              LLVMTypeConverter &converter) {
  llvm::SmallVector<Type, 8> elements;
  mlir::SmallVector<mlir::Type> types;
  type.getInnerTypes(types);

  for (int i = 0, e = types.size(); i < e; ++i)
    elements.push_back(converter.convertType(
        types[HWToLLVMEndianessConverter::convertToLLVMEndianess(type, i)]));

  return LLVM::LLVMStructType::getLiteral(&converter.getContext(), elements);
}

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct HWToLLVMLoweringPass
    : public circt::impl::ConvertHWToLLVMBase<HWToLLVMLoweringPass> {

  using circt::impl::ConvertHWToLLVMBase<
      HWToLLVMLoweringPass>::ConvertHWToLLVMBase;

  void runOnOperation() override;
};
} // namespace

void circt::populateHWToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    Namespace &globals,
    DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp>
        &constAggregateGlobalsMap,
    std::optional<HWToLLVMArraySpillCache> &spillCacheOpt) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // Value creation conversion patterns.
  patterns.add<HWConstantOpConversion>(ctx, converter);
  patterns.add<HWDynamicArrayCreateOpConversion, HWStructCreateOpConversion>(
      converter);
  patterns.add<AggregateConstantOpConversion>(
      converter, constAggregateGlobalsMap, globals, spillCacheOpt);

  // Bitwise conversion patterns.
  patterns.add<BitcastOpConversion>(converter);

  // Extraction operation conversion patterns.
  patterns.add<StructExplodeOpConversion, StructExtractOpConversion,
               StructInjectOpConversion>(converter);

  patterns.add<ArrayGetOpConversion, ArrayInjectOpConversion,
               ArraySliceOpConversion, ArrayConcatOpConversion>(converter,
                                                                spillCacheOpt);
}

void circt::populateHWToLLVMTypeConversions(LLVMTypeConverter &converter) {
  converter.addConversion(
      [&](hw::ArrayType arr) { return convertArrayType(arr, converter); });
  converter.addConversion(
      [&](hw::StructType tup) { return convertStructType(tup, converter); });
}

void HWToLLVMLoweringPass::runOnOperation() {
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  std::optional<HWToLLVMArraySpillCache> spillCacheOpt = {};
  Namespace globals;
  SymbolCache cache;
  cache.addDefinitions(getOperation());
  globals.add(cache);

  RewritePatternSet patterns(&getContext());
  auto converter = mlir::LLVMTypeConverter(&getContext());
  populateHWToLLVMTypeConversions(converter);

  if (spillArraysEarly) {
    spillCacheOpt = HWToLLVMArraySpillCache();
    OpBuilder spillBuilder(getOperation());
    spillCacheOpt->spillNonHWOps(spillBuilder, converter, getOperation());
  }

  LLVMConversionTarget target(getContext());
  target.addIllegalDialect<hw::HWDialect>();

  // Setup the conversion.
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap, spillCacheOpt);

  // Apply the partial conversion.
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns),
                                    config)))
    signalPassFailure();
}
