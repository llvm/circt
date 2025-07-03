//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for exporting AIGER files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_EXPORTAIGER_H
#define CIRCT_CONVERSION_EXPORTAIGER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class MLIRContext;
class TimingScope;
} // namespace mlir

namespace circt {
namespace hw {
class HWModuleOp;
} // namespace hw

namespace aiger {

/// Handler for AIGER export. If it's passed to the exportAIGER function, the
/// handler will be called for each operand and result of the operation. Clients
/// are expected to record this information in their use case.
struct ExportAIGERHandler {
  ExportAIGERHandler() = default;
  virtual ~ExportAIGERHandler() = default;

  // Return true if the operand should be added to the output, false
  // otherwise. If returned false, outputIndex will be invalid for the given
  // operand. This callback is called for hw::OutputOp operands and unknown
  // operations.
  virtual bool operandCallback(mlir::OpOperand &operand, size_t bitPos,
                               size_t outputIndex) {
    return true;
  };

  // Return true if the result should be added to the input, false otherwise.
  // If returned false, inputIndex will be invalid for the given result. This
  // callback is called for BlockArguments and unknown operation results.
  virtual bool valueCallback(Value result, size_t bitPos, size_t inputIndex) {
    return true;
  };

  // Callback for notifying that an operation has been emitted into AIGER.
  virtual void notifyEmitted(Operation *op){};

  // Callback for notifying that a clock has been emitted.
  virtual void notifyClock(Value value){};
};

/// Options for AIGER export.
struct ExportAIGEROptions {
  /// Whether to export in binary format (aig) or ASCII format (aag).
  /// Default is ASCII format.
  bool binaryFormat = false;

  /// Whether to include symbol table in the output.
  /// Default is true.
  bool includeSymbolTable = true;
};

/// Export an MLIR module containing AIG dialect operations to AIGER format.
mlir::LogicalResult exportAIGER(hw::HWModuleOp module, llvm::raw_ostream &os,
                                const ExportAIGEROptions *options = nullptr,
                                ExportAIGERHandler *handler = nullptr);

/// Register the `export-aiger` MLIR translation.
void registerExportAIGERTranslation();

} // namespace aiger
} // namespace circt

#endif // CIRCT_CONVERSION_EXPORTAIGER_H
