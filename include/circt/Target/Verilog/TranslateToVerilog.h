#ifndef CIRCT_TARGET_VERILOG_TRANSLATETOVERILOG_H
#define CIRCT_TARGET_VERILOG_TRANSLATETOVERILOG_H

#include "circt/Target/Verilog/VerilogPrinter.h"
#include "mlir/Translation.h"

using namespace mlir;

namespace mlir {
namespace llhd {

void registerToVerilogTranslation() {
  TranslateFromMLIRRegistration registration(
      "llhd-to-verilog", [](ModuleOp module, raw_ostream &output) {
        formatted_raw_ostream out(output);
        llhd::VerilogPrinter printer(out);
        printer.printModule(module);
        return success();
      });
}

} // namespace llhd
} // namespace mlir

#endif // CIRCT_TARGET_VERILOG_TRANSLATETOVERILOG_H
