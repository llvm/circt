//===- HWToLLVM.h - HW to LLVM pass entry point -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the HWToLLVM pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWTOLLVM_HWTOLLVM_H
#define CIRCT_CONVERSION_HWTOLLVM_HWTOLLVM_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
namespace LLVM {
class GlobalOp;
} // namespace LLVM
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_CONVERTHWTOLLVM
#include "circt/Conversion/Passes.h.inc"

class Namespace;

struct HWToLLVMEndianessConverter {
  /// Convert an index into a HW ArrayType or StructType to LLVM Endianess.
  static uint32_t convertToLLVMEndianess(Type type, uint32_t index);

  /// Get the index of a specific StructType field in the LLVM lowering of the
  /// StructType
  static uint32_t llvmIndexOfStructField(hw::StructType type,
                                         StringRef fieldName);
};

/// Helper class mapping array values (HW or LLVM Dialect) to pointers to
/// buffers containing the array value.
struct HWToLLVMArraySpillCache {
  /// Spill HW array values produced by 'foreign' dialects on the stack.
  /// The converter is used to map HW array types to the corresponding
  /// LLVM array types. Should be called before dialect conversion.
  void spillNonHWOps(mlir::OpBuilder &builder,
                     mlir::LLVMTypeConverter &converter,
                     Operation *containerOp);

  /// Map an array value (HW or LLVM Dialect) to a LLVM pointer.
  /// For the entire lifetime of the array value the pointer must
  /// refer to a valid buffer containing the respective array value.
  void map(mlir::Value arrayValue, mlir::Value bufferPtr);

  /// Retrieve a pointer to a buffer containing the given array
  /// value (HW or LLVM Dialect). The buffer must not be modified or
  /// deallocated. Returns a null value if no buffer has been mapped.
  Value lookup(Value arrayValue);

private:
  Value spillLLVMArrayValue(OpBuilder &builder, Location loc, Value llvmArray);
  Value spillHWArrayValue(OpBuilder &builder, Location loc,
                          mlir::LLVMTypeConverter &converter, Value hwArray);

  llvm::DenseMap<Value, Value> spillMap;
};

/// Get the HW to LLVM type conversions.
void populateHWToLLVMTypeConversions(mlir::LLVMTypeConverter &converter);

/// Get the HW to LLVM conversion patterns.
/// Note: The spill cache may only be used when conversion
///       pattern rollback is disabled.
void populateHWToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, RewritePatternSet &patterns,
    Namespace &globals,
    DenseMap<std::pair<Type, ArrayAttr>, mlir::LLVM::GlobalOp>
        &constAggregateGlobalsMap,
    std::optional<HWToLLVMArraySpillCache> &spillCacheOpt);

} // namespace circt

#endif // CIRCT_CONVERSION_HWTOLLVM_HWTOLLVM_H
