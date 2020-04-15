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

/// Return the width of the specified FIRRTL type in bits or -1 if it isn't
/// supported.
static int getBitWidthOrSentinel(FIRRTLType type) {
  switch (type.getKind()) {
  case FIRRTLType::Clock:
  case FIRRTLType::Reset:
  case FIRRTLType::AsyncReset:
    return 1;

  case FIRRTLType::SInt:
  case FIRRTLType::UInt:
    return type.cast<IntType>().getWidthOrSentinel();

  case FIRRTLType::Flip:
    return getBitWidthOrSentinel(type.cast<FlipType>().getElementType());

  default:
    return -1;
  };
}

/// Given an integer value, return the number of characters it will take to
/// print its base-10 value.
static unsigned getPrintedIntWidth(unsigned value) {
  if (value < 10)
    return 1;
  if (value <= 100)
    return 2;
  if (value <= 1000)
    return 3;

  SmallVector<char, 8> spelling;
  llvm::raw_svector_ostream stream(spelling);
  stream << value;
  return stream.str().size();
}

void VerilogEmitter::emitFModule(FModuleOp module) {
  os << "module " << module.getName() << "(\n";

  // Determine the width of the widest type we have to print so everything
  // lines up nicely.
  SmallVector<ModulePortInfo, 8> portInfo;
  module.getPortInfo(portInfo);

  unsigned maxTypeWidth = 0;
  for (auto &port : portInfo) {
    int bitWidth = getBitWidthOrSentinel(port.second);
    if (bitWidth == -1 || bitWidth == 1)
      continue; // The error case is handled below.

    // Add 4 to count the width of the "[:0] ".
    unsigned thisWidth = getPrintedIntWidth(bitWidth - 1) + 5;
    maxTypeWidth = std::max(thisWidth, maxTypeWidth);
  }

  // TODO(QoI): Should emit more than one port on a line.
  //  e.g. output [2:0] auto_out_c_bits_opcode, auto_out_c_bits_param,
  //
  for (auto &port : portInfo) {
    os.indent(2);
    // Emit the arguments.
    auto portType = port.second;
    if (auto flip = portType.dyn_cast<FlipType>()) {
      portType = flip.getElementType();
      os << "output ";
    } else {
      os << "input  ";
    }

    unsigned emittedWidth = 0;

    int bitWidth = getBitWidthOrSentinel(portType);
    if (bitWidth == -1) {
      module.emitError("parameter '" + port.first.getValue() +
                       "' has an unsupported verilog type ")
          << portType;
    } else if (bitWidth != 1) {
      // Width 1 is implicit.
      os << '[' << (bitWidth - 1) << ":0] ";
      emittedWidth = getPrintedIntWidth(bitWidth - 1) + 5;
    }

    if (maxTypeWidth - emittedWidth)
      os.indent(maxTypeWidth - emittedWidth);

    os << port.first.getValue();
    if (&port != &portInfo.back())
      os << ',';
    os << "\n";
  }

  // TODO(QoI): Don't print this on its own line.
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
