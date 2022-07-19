//===- HWToLLVM.h - HW to LLVM pass entry point -------------*- C++ -*-===//
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

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace circt {

class HWToLLVMEndianessConverter {
public:
  static uint32_t convertToLLVMEndianess(Type type, uint32_t index) {
    // This is hardcoded for little endian machines for now.
    return TypeSwitch<Type, uint32_t>(type)
        .Case<hw::ArrayType>(
            [&](hw::ArrayType ty) { return ty.getSize() - index - 1; })
        .Case<hw::StructType>([&](hw::StructType ty) {
          return ty.getElements().size() - index - 1;
        });
  }

  static uint32_t llvmIndexOfStructField(hw::StructType type,
                                         StringRef fieldName) {
    auto fieldIter = type.getElements();
    size_t index = 0;

    for (const auto *iter = fieldIter.begin(); iter != fieldIter.end();
         ++iter) {
      if (iter->name == fieldName) {
        return convertToLLVMEndianess(type, index);
      }
      ++index;
    }

    // Verifier of StructExtractOp has to ensure that the field name is indeed
    // present.
    llvm_unreachable("Field name attribute of hw::StructExtractOp invalid");
    return 0;
  }
};

/// Get the HW to LLVM conversion patterns.
void populateHWToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns,
                                        size_t &sigCounter, size_t &regCounter);

/// Create an HW to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertHWToLLVMPass();

} // namespace circt

#endif // CIRCT_CONVERSION_HWTOLLVM_HWTOLLVM_H