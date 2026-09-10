//===- HLSOps.h - HLS op declarations ---------------------------*- C++ -*-===//

#ifndef CIRCT_DIALECT_HLS_HLSOPS_H
#define CIRCT_DIALECT_HLS_HLSOPS_H

#include "circt/Dialect/Resource/HLS/HLSDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"                         // OpAsmParser/Printer (already needed)
#include "mlir/Bytecode/BytecodeOpInterface.h"                // <-- BytecodeOpInterface, DialectBytecodeReader/Writer
#include "mlir/Interfaces/SideEffectInterfaces.h"             // MemoryEffectOpInterface
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"  // <-- affine::AffineWriteOpInterface

#define GET_OP_CLASSES
#include "circt/Dialect/Resource/HLS/HLS.h.inc"

#endif // CIRCT_DIALECT_HLS_HLSOPS_H
