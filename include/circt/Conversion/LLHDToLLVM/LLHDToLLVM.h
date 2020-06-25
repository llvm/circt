#ifndef CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H
#define CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class ModuleOp;
class LLVMTypeConverter;
template <typename T>
class OperationPass;

namespace llhd {

/// Get the LLHD to LLVM conversion patterns.
void populateLLHDToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns);

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertLLHDToLLVMPass();

/// Register the LLHD to LLVM convesion pass.
inline void initLLHDToLLVMPass() {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/LLHDToLLVM/Passes.h.inc"
}
} // namespace llhd
} // namespace mlir

#endif // CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H
