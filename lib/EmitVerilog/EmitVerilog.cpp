//===- EmitVerilog.cpp - Verilog Emitter ----------------------------------===//
//
// This is the main Verilog emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "spt/EmitVerilog.h"
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "spt/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace spt;
using namespace firrtl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper routines
//===----------------------------------------------------------------------===//

/// ExprVisitor is a visitor for FIRRTL expression nodes.
template <typename ConcreteType, typename ResultType = void>
class ExprVisitor {
public:
  ResultType visitExpr(Operation *op) {
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<AndRPrimOp, XorRPrimOp>([&](auto expr) -> ResultType {
          return static_cast<ConcreteType *>(this)->visitExpr(expr);
        })
        .Default([&](auto expr) -> ResultType {
          return static_cast<ConcreteType *>(this)->visitInvalidExpr(op);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidExpr(Operation *op) {
    op->emitOpError("unknown firrtl expression");
    abort();
  }

  /// This callback is invoked on any expression operations that are not handled
  /// by the concrete visitor.
  ResultType visitUnhandledExpr(Operation *op) { return ResultType(); }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitExpr(OPTYPE op) {                                            \
    return static_cast<ConcreteType *>(this)->visitUnhandledExpr(op);          \
  }

  // Basic expressions.
  HANDLE(ConstantOp)
  HANDLE(SubfieldOp);
  HANDLE(SubindexOp);
  HANDLE(SubaccessOp);

  // Arithmetic and Logical Binary Primitives.
  HANDLE(AddPrimOp);
  HANDLE(SubPrimOp);
  HANDLE(MulPrimOp);
  HANDLE(DivPrimOp);
  HANDLE(RemPrimOp);
  HANDLE(AndPrimOp);
  HANDLE(OrPrimOp);
  HANDLE(XorPrimOp);

  // Comparisons.
  HANDLE(LEQPrimOp);
  HANDLE(LTPrimOp);
  HANDLE(GEQPrimOp);
  HANDLE(GTPrimOp);
  HANDLE(EQPrimOp);
  HANDLE(NEQPrimOp);

  // Misc Binary Primitives.
  HANDLE(CatPrimOp);
  HANDLE(DShlPrimOp);
  HANDLE(DShrPrimOp);
  HANDLE(ValidIfPrimOp);

  // Unary operators.
  HANDLE(AsSIntPrimOp);
  HANDLE(AsUIntPrimOp);
  HANDLE(AsAsyncResetPrimOp);
  HANDLE(AsClockPrimOp);
  HANDLE(CvtPrimOp);
  HANDLE(NegPrimOp);
  HANDLE(NotPrimOp);
  HANDLE(AndRPrimOp);
  HANDLE(OrRPrimOp);
  HANDLE(XorRPrimOp);

  // Miscellaneous.
  HANDLE(BitsPrimOp);
  HANDLE(HeadPrimOp);
  HANDLE(MuxPrimOp);
  HANDLE(PadPrimOp);
  HANDLE(ShlPrimOp);
  HANDLE(ShrPrimOp);
  HANDLE(TailPrimOp);
};

/// Return true if the specified operation is a firrtl expression.
static bool isExpression(Operation *op) {
  struct IsExprClassifier : public ExprVisitor<IsExprClassifier, bool> {
    bool visitInvalidExpr(Operation *op) { return false; }
    bool visitUnhandledExpr(Operation *op) { return true; }
  };

  return IsExprClassifier().visitExpr(op);
}

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

/// Return true if this expression should be emitted inline into any statement
/// that uses it.
static bool isExpressionEmittedInline(Operation *op) {
  // ConstantOp is always emitted inline.
  if (isa<ConstantOp>(op))
    return true;

  // Otherwise, if it has multiple uses, emit it out of line.
  return op->getResult(0).hasOneUse();
}

//===----------------------------------------------------------------------===//
// VerilogEmitter
//===----------------------------------------------------------------------===//

namespace {

class VerilogEmitter {
public:
  VerilogEmitter(raw_ostream &os) : os(os) {}

  LogicalResult emitMLIRModule(ModuleOp module);

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(currentIndent); }

  void addIndent() { currentIndent += 2; }
  void reduceIndent() { currentIndent -= 2; }

  /// The stream to emit to.
  raw_ostream &os;

private:
  void emitStatementExpression(Operation *op);
  void emitStatement(ConnectOp op);
  void emitOperation(Operation *op);
  void emitFModule(FModuleOp module);
  void emitCircuit(CircuitOp circuit);

  bool encounteredError = false;
  unsigned currentIndent = 0;

  VerilogEmitter(const VerilogEmitter &) = delete;
  void operator=(const VerilogEmitter &) = delete;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void VerilogEmitter::emitStatementExpression(Operation *op) {
  // Need to emit a wire ahead of time,
  //    then connect to that wire.
  // Need a naming pass of some sort.
  indent() << "assign WIRENAME = expr";
}

void VerilogEmitter::emitStatement(ConnectOp op) {
  indent() << "assign x = y;\n";
  // TODO: location information too.
}

//===----------------------------------------------------------------------===//
// Module and Circuit
//===----------------------------------------------------------------------===//

void VerilogEmitter::emitOperation(Operation *op) {
  // Handle expression statements.
  if (isExpression(op)) {
    if (!isExpressionEmittedInline(op))
      emitStatementExpression(op);
    return;
  }

  // Handle statements first.
  // TODO: Refactor out to visitors.
  bool isStatement = false;
  TypeSwitch<Operation *>(op).Case<ConnectOp>([&](auto stmt) {
    isStatement = true;
    this->emitStatement(stmt);
  });
  if (isStatement)
    return;

  // Ignore the region terminator.
  if (isa<DoneOp>(op))
    return;

  op->emitOpError("cannot emit this operation to Verilog");
  encounteredError = true;
}

void VerilogEmitter::emitFModule(FModuleOp module) {
  os << "module " << module.getName() << '(';

  // Determine the width of the widest type we have to print so everything
  // lines up nicely.
  SmallVector<ModulePortInfo, 8> portInfo;
  module.getPortInfo(portInfo);

  if (!portInfo.empty())
    os << '\n';

  unsigned maxTypeWidth = 0;
  for (auto &port : portInfo) {
    int bitWidth = getBitWidthOrSentinel(port.second);
    if (bitWidth == -1 || bitWidth == 1)
      continue; // The error case is handled below.

    // Add 4 to count the width of the "[:0] ".
    unsigned thisWidth = getPrintedIntWidth(bitWidth - 1) + 5;
    maxTypeWidth = std::max(thisWidth, maxTypeWidth);
  }

  addIndent();

  // TODO(QoI): Should emit more than one port on a line.
  //  e.g. output [2:0] auto_out_c_bits_opcode, auto_out_c_bits_param,
  //
  for (auto &port : portInfo) {
    indent();
    // Emit the arguments.
    auto portType = port.second;
    if (auto flip = portType.dyn_cast<FlipType>()) {
      portType = flip.getElementType();
      os << "output";
    } else {
      os << "input ";
    }

    unsigned emittedWidth = 0;

    int bitWidth = getBitWidthOrSentinel(portType);
    if (bitWidth == -1) {
      emitError(module, "parameter '" + port.first.getValue() +
                            "' has an unsupported verilog type ")
          << portType;
    } else if (bitWidth != 1) {
      // Width 1 is implicit.
      os << " [" << (bitWidth - 1) << ":0]";
      emittedWidth = getPrintedIntWidth(bitWidth - 1) + 5;
    }

    if (maxTypeWidth - emittedWidth)
      os.indent(maxTypeWidth - emittedWidth);

    os << ' ' << port.first.getValue();
    if (&port != &portInfo.back())
      os << ',';
    else
      os << ");";
    os << "\n";
  }

  if (portInfo.empty())
    os << ");\n";

  // Emit the body.
  for (auto &op : *module.getBodyBlock()) {
    emitOperation(&op);
  }

  reduceIndent();

  os << "endmodule\n\n";
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
