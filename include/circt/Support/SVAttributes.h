#ifndef CIRCT_SUPPORT_SV_ATTRIBUTES_H
#define CIRCT_SUPPORT_SV_ATTRIBUTES_H
#include "circt/Support/LLVM.h"

namespace circt {
bool hasSVAttributes(mlir::Operation *op);

mlir::Attribute getSVAttributes(mlir::Operation *op);

void setSVAttributes(mlir::Operation *op, mlir::Attribute);
} // namespace circt

#endif // CIRCT_SUPPORT_SV_ATTRIBUTES_H