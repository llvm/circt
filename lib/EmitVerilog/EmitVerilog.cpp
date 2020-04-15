//===- EmitVerilog.cpp - Verilog Emitter ----------------------------------===//
//
// This is the main Verilog emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "spt/EmitVerilog.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "spt/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace spt;
using namespace firrtl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// VerilogEmitter
//===----------------------------------------------------------------------===//

namespace {

struct VerilogEmitter {
  VerilogEmitter(raw_ostream &os) : os(os) {}

  raw_ostream &os;

  LogicalResult emitModule(ModuleOp module);

private:
  VerilogEmitter(const VerilogEmitter &) = delete;
  void operator=(const VerilogEmitter &) = delete;
};

} // end anonymous namespace

LogicalResult VerilogEmitter::emitModule(ModuleOp module) {
  os << "Not implemented yet.\n";

  return success();
}

void spt::registerVerilogEmitterTranslation() {
  static TranslateFromMLIRRegistration toVerilog(
      "emit-verilog",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        return VerilogEmitter(os).emitModule(module);
      });
}
