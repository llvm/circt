//===- HWToLLVM.cpp - HW to LLVM Conversion Patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the HW to LLVM Operation and Type Conversion Patterns.
//
//===----------------------------------------------------------------------===//

#include "HWToLLVM.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

static Type convertArrayType(hw::ArrayType type, LLVMTypeConverter &converter) {
  auto elementTy = converter.convertType(type.getElementType());
  return LLVM::LLVMArrayType::get(elementTy, type.getSize());
}

static Type convertStructType(hw::StructType type,
                              LLVMTypeConverter &converter) {
  llvm::SmallVector<Type, 8> elements;
  mlir::SmallVector<mlir::Type> types;
  type.getInnerTypes(types);
  for (auto elemTy : types)
    elements.push_back(converter.convertType(elemTy));
  return LLVM::LLVMStructType::getLiteral(&converter.getContext(), elements);
}

//===----------------------------------------------------------------------===//
// MiscOps patterns
//===----------------------------------------------------------------------===//

namespace {
struct ConstantOpConversion : public ConvertOpToLLVMPattern<hw::ConstantOp> {
  using ConvertOpToLLVMPattern<hw::ConstantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ConstantOp op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the converted llvm type.
    auto intType = typeConverter->convertType(op.valueAttr().getType());
    // Replace the operation with an llvm constant op.
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, intType, op.valueAttr());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Aggregate operation patterns
//===----------------------------------------------------------------------===//

namespace {
/// Convert an ArrayOp operation to the LLVM dialect. An equivalent and
/// initialized llvm dialect array type is generated.
struct ArrayCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::ArrayCreateOp> {
  using ConvertOpToLLVMPattern<hw::ArrayCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayCreateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto arrayTy = typeConverter->convertType(op->getResult(0).getType());
    Value arr = rewriter.create<LLVM::UndefOp>(op->getLoc(), arrayTy);

    for (size_t i = 0, e = op.inputs().size(); i < e; ++i) {
      arr = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), arrayTy, arr,
                                                 op.inputs()[i],
                                                 rewriter.getI32ArrayAttr(i));
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};
} // namespace

namespace {
/// Convert a StructCreateOp operation to the LLVM dialect. An equivalent and
/// initialized llvm dialect struct type is generated.
struct StructCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::StructCreateOp> {
  using ConvertOpToLLVMPattern<hw::StructCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructCreateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resTy = typeConverter->convertType(op->getResult(0).getType());
    Value tup = rewriter.create<LLVM::UndefOp>(op->getLoc(), resTy);

    for (size_t i = 0, e = resTy.cast<LLVM::LLVMStructType>().getBody().size();
         i < e; ++i) {
      tup = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), resTy, tup, op.input()[i], rewriter.getI32ArrayAttr(i));
    }

    rewriter.replaceOp(op, tup);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern registration
//===----------------------------------------------------------------------===//

void circt::populateHWToLLVMTypeConversions(
    mlir::LLVMTypeConverter &converter) {

  converter.addConversion(
      [&](hw::ArrayType arr) { return convertArrayType(arr, converter); });
  converter.addConversion(
      [&](hw::StructType tup) { return convertStructType(tup, converter); });
}

void circt::populateHWToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {

  // MiscOps conversion patterns.
  patterns.add<ConstantOpConversion>(converter);

  // Aggregation operation patterns.
  patterns.add<ArrayCreateOpConversion, StructCreateOpConversion>(converter);
}
