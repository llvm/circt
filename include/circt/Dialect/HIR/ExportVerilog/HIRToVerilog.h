//=====- HIRToVerilog.h - Verilog Printer -----------------------*- C++ -*-===//
//
// Defines the interface to the HIR to Verilog Printer.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_VERILOG_HIRTOVERILOG_H
#define CIRCT_TARGET_VERILOG_HIRTOVERILOG_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace circt {

struct LogicalResult;
class ModuleOp;

namespace hir {

/// Translate HIR "module" into verilog and output it to "os".
circt::LogicalResult printVerilog(circt::ModuleOp module,
                                  llvm::raw_ostream &os);

/// Register the HIR to Verilog Translation pass.
void registerHIRToVerilogTranslation();

} // namespace hir.
} // namespace circt

#endif // CIRCT_TARGET_VERILOG_HIRTOVERILOG_H.
