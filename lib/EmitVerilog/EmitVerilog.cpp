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

class VerilogEmitter {
public:
  VerilogEmitter(raw_ostream &os) : os(os) {}

  LogicalResult emitMLIRModule(ModuleOp module);

  void emitError(Operation *op, const Twine &message) {
    op->emitError(message);
    encounteredError = true;
  }

  /// The stream to emit to.
  raw_ostream &os;

private:
  void emitCircuit(CircuitOp circuit);
  void emitFModule(FModuleOp module);

  bool encounteredError = false;

  VerilogEmitter(const VerilogEmitter &) = delete;
  void operator=(const VerilogEmitter &) = delete;
};

} // end anonymous namespace

void VerilogEmitter::emitFModule(FModuleOp module) {
  os << "module " << module.getName() << "(\n";

  // Emit the arguments.

  os << ");\n";

  // emit the body.

  os << "endmodule\n";
}

void VerilogEmitter::emitCircuit(CircuitOp circuit) {
  for (auto &op : *circuit.getBody()) {
    if (auto module = dyn_cast<FModuleOp>(op))
      emitFModule(module);
    else if (!isa<DoneOp>(op))
      op.emitError("unknown operation");
  }
}

LogicalResult VerilogEmitter::emitMLIRModule(ModuleOp module) {
  for (auto &op : *module.getBody()) {
    if (auto circuit = dyn_cast<CircuitOp>(op))
      emitCircuit(circuit);
    else if (!isa<ModuleTerminatorOp>(op))
      op.emitError("unknown operation");
  }

  return encounteredError ? failure() : success();
}

void spt::registerVerilogEmitterTranslation() {
  static TranslateFromMLIRRegistration toVerilog(
      "emit-verilog",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        return VerilogEmitter(os).emitMLIRModule(module);
      });
}
