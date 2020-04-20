//===- EmitVerilog.cpp - Verilog Emitter ----------------------------------===//
//
// This is the main Verilog emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "spt/EmitVerilog.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"
#include "spt/Dialect/FIRRTL/Visitors.h"
#include "spt/Support/LLVM.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace spt;
using namespace firrtl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper routines
//===----------------------------------------------------------------------===//

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
  if (value < 100)
    return 2;
  if (value < 1000)
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
  // TODO: same for subobjects and other things like that.
  if (isa<ConstantOp>(op))
    return true;

  // Otherwise, if it has multiple uses, emit it out of line.
  return op->getResult(0).hasOneUse();
}

//===----------------------------------------------------------------------===//
// VerilogEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class VerilogEmitterState {
public:
  explicit VerilogEmitterState(raw_ostream &os) : os(os) {}

  /// The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

private:
  VerilogEmitterState(const VerilogEmitterState &) = delete;
  void operator=(const VerilogEmitterState &) = delete;
};
} // namespace

namespace {

/// This is the base class for all of the Verilog Emitter components.
class VerilogEmitterBase {
public:
  explicit VerilogEmitterBase(VerilogEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitOpError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // All of the mutable state we are maintaining.
  VerilogEmitterState &state;

  /// The stream to emit to.
  raw_ostream &os;

private:
  VerilogEmitterBase(const VerilogEmitterBase &) = delete;
  void operator=(const VerilogEmitterBase &) = delete;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ModuleEmitter
//===----------------------------------------------------------------------===//

namespace {

class ModuleEmitter : public VerilogEmitterBase {
public:
  explicit ModuleEmitter(VerilogEmitterState &state)
      : VerilogEmitterBase(state) {}

  void emitFModule(FModuleOp module);

  // Expressions

  void emitExpression(Value exp, bool forceRootExpr = false);

  // Statements.
  void emitStatementExpression(Operation *op);
  void emitStatement(ConnectOp op);
  void emitStatement(SkipOp op) {}
  void emitOperation(Operation *op);

  void collectNames(FModuleOp module);
  void collectOpNames(Operation *op);
  void addName(Value value, StringRef name);
  void addName(Value value, StringAttr nameAttr) {
    addName(value, nameAttr ? nameAttr.getValue() : "");
  }

  StringRef getName(Value value) {
    auto *entry = nameTable[value];
    assert(entry && "value expected a name but doesn't have one");
    return entry->getKey();
  }

  llvm::StringSet<> usedNames;
  llvm::DenseMap<Value, llvm::StringMapEntry<llvm::NoneType> *> nameTable;
  size_t nextGeneratedNameID = 0;
};

} // end anonymous namespace

/// Add the specified name to the name table, auto-uniquing the name if
/// required.  If the name is empty, then this creates a unique temp name.
void ModuleEmitter::addName(Value value, StringRef name) {
  if (name.empty())
    name = "_T";

  // Check to see if this name is valid.  The first character cannot be a
  // number of other weird thing.  If it is, start with an underscore.
  if (!isalpha(name.front()) && name.front() != '_') {
    SmallString<16> tmpName("_");
    tmpName += name;
    return addName(value, tmpName);
  }

  auto isValidVerilogCharacter = [](char ch) -> bool {
    return isalpha(ch) || isdigit(ch) || ch == '_';
  };

  // Check to see if the name consists of all-valid identifiers.  If not, we
  // need to escape them.
  for (char ch : name) {
    if (isValidVerilogCharacter(ch))
      continue;

    // Otherwise, we need to escape it.
    SmallString<16> tmpName;
    for (char ch : name) {
      if (isValidVerilogCharacter(ch))
        tmpName += ch;
      else if (ch == ' ')
        tmpName += '_';
      else {
        tmpName += llvm::utohexstr((unsigned char)ch);
      }
    }
    return addName(value, tmpName);
  }

  // Check to see if this name is available - if so, use it.
  auto insertResult = usedNames.insert(name);
  if (insertResult.second) {
    nameTable[value] = &*insertResult.first;
    return;
  }

  // If not, we need to auto-unique it.
  SmallVector<char, 16> nameBuffer(name.begin(), name.end());
  nameBuffer.push_back('_');
  auto baseSize = nameBuffer.size();

  // Try until we find something that works.
  while (1) {
    auto suffix = llvm::utostr(nextGeneratedNameID++);
    nameBuffer.append(suffix.begin(), suffix.end());

    insertResult =
        usedNames.insert(StringRef(nameBuffer.data(), nameBuffer.size()));
    if (insertResult.second) {
      nameTable[value] = &*insertResult.first;
      return;
    }

    // Chop off the suffix and try again.
    nameBuffer.resize(baseSize);
  }
}

//===----------------------------------------------------------------------===//
// Expression Emission
//===----------------------------------------------------------------------===//

namespace {
/// This enum keeps track of the precedence level of various binary operators,
/// where a lower number binds tighter.
enum VerilogPrecedence {
  NonExpression, // Not an inline expression.
  Unary,         // Symbols and all the unary operators.
  Multiply,      // * , / , %
  Addition,      // + , -
  Shift,         // << , >>
  Comparison,    // > , >= , < , <=
  Equality,      // == , !=
  And,           // &
  Xor,           // ^ , ^~
  Or,            // |
};

/// This is information precomputed about each subexpression in the tree we
/// are emitting as a unit.
struct SubExprInfo {
  /// The precedence of this expression.
  VerilogPrecedence precedence;

  /// The syntax corresponding to this node if known.
  const char *syntax = nullptr;
};
} // namespace

namespace {
class ExprEmitter : public ExprVisitor<ExprEmitter, void, SubExprInfo> {
public:
  ExprEmitter(raw_ostream &os, ModuleEmitter &emitter)
      : os(os), emitter(emitter) {}

  void emitExpression(Value exp, bool forceRootExpr);
  friend class ExprVisitor;

private:
  /// Compute the precedence levels and other information we need for the
  ///  subexpressions in this tree, filling in subExprInfos.
  void computeSubExprInfos(Value exp, bool forceRootExpr = false);

  SubExprInfo getInfo(Value exp) const;

  /// Emit the specified value as a subexpression to the stream.
  void emitSubExpr(Value exp, SubExprInfo exprInfo);

  void emitUnaryPrefixExpr(Operation *op, SubExprInfo opInfo);
  void emitBinaryExpr(Operation *op, SubExprInfo opInfo);
  void visitUnhandledExpr(Operation *op, SubExprInfo exprInfo);

  using ExprVisitor::visitExpr;
  void visitExpr(ConstantOp op, SubExprInfo info);
  void visitExpr(AddPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(SubPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(MulPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(DivPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(RemPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }

  void visitExpr(AndPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(OrPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(XorPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }

  // Comparison Operations
  void visitExpr(LEQPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(LTPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(GEQPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(GTPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(EQPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }
  void visitExpr(NEQPrimOp op, SubExprInfo info) { emitBinaryExpr(op, info); }

  // Cat, DShl, DShr, ValidIf

  // Unary Prefix operators.
  void visitExpr(AndRPrimOp op, SubExprInfo info) {
    emitUnaryPrefixExpr(op, info);
  }
  void visitExpr(XorRPrimOp op, SubExprInfo info) {
    emitUnaryPrefixExpr(op, info);
  }
  void visitExpr(OrRPrimOp op, SubExprInfo info) {
    emitUnaryPrefixExpr(op, info);
  }
  void visitExpr(NotPrimOp op, SubExprInfo info) {
    emitUnaryPrefixExpr(op, info);
  }

private:
  llvm::SmallDenseMap<Value, SubExprInfo, 8> subExprInfos;
  raw_ostream &os;
  ModuleEmitter &emitter;
};
} // end anonymous namespace

SubExprInfo ExprEmitter::getInfo(Value exp) const {
  auto it = subExprInfos.find(exp);
  assert(it != subExprInfos.end() && "No info computed for this expr");
  return it->second;
}

/// Emit the specified value as an expression.  If this is an inline-emitted
/// expression, we emit that expression, otherwise we emit a reference to the
/// already computed name.  If 'forceRootExpr' is true, then this emits an
/// expression even if we typically don't do it inline.
///
void ExprEmitter::emitExpression(Value exp, bool forceRootExpr) {
  // Compute information about subexpressions.
  computeSubExprInfos(exp, forceRootExpr);

  // Okay, this is an expression we should emit inline.  Do this through our
  // visitor.
  emitSubExpr(exp, getInfo(exp));
}

/// Compute the precedence levels and other information we need for the
///  subexpressions in this tree, filling in subExprInfos.
void ExprEmitter::computeSubExprInfos(Value exp, bool forceRootExpr) {
  // This is all of the things we want to look at.
  SmallVector<Value, 8> worklist;
  worklist.push_back(exp);

  while (!worklist.empty()) {
    auto exp = worklist.pop_back_val();
    auto *op = exp.getDefiningOp();
    bool shouldEmitInlineExpr =
        op && isExpression(op) &&
        (forceRootExpr || isExpressionEmittedInline(op));

    // If this is is an reference to an out-of-line expression or declaration,
    // just use it.
    if (!shouldEmitInlineExpr) {
      subExprInfos[exp] = {NonExpression, nullptr};
      continue;
    }

    // Otherwise, we have an expression.  Compute the properties we care about.
    struct ExprPropertyComputer
        : public ExprVisitor<ExprPropertyComputer, SubExprInfo> {

      using ExprVisitor::visitExpr;
      SubExprInfo visitUnhandledExpr(Operation *op) { return {Unary, ""}; }

      SubExprInfo visitExpr(AddPrimOp op) { return {Addition, "+"}; }
      SubExprInfo visitExpr(SubPrimOp op) { return {Addition, "-"}; }
      SubExprInfo visitExpr(MulPrimOp op) { return {Multiply, "*"}; }
      SubExprInfo visitExpr(DivPrimOp op) { return {Multiply, "/"}; }
      SubExprInfo visitExpr(RemPrimOp op) { return {Multiply, "%"}; }

      SubExprInfo visitExpr(AndPrimOp op) { return {And, "&"}; }
      SubExprInfo visitExpr(OrPrimOp op) { return {Or, "|"}; }
      SubExprInfo visitExpr(XorPrimOp op) { return {Xor, "^"}; }

      // Comparison Operations
      SubExprInfo visitExpr(LEQPrimOp op) { return {Comparison, "<="}; }
      SubExprInfo visitExpr(LTPrimOp op) { return {Comparison, "<"}; }
      SubExprInfo visitExpr(GEQPrimOp op) { return {Comparison, ">="}; }
      SubExprInfo visitExpr(GTPrimOp op) { return {Comparison, ">"}; }
      SubExprInfo visitExpr(EQPrimOp op) { return {Equality, "=="}; }
      SubExprInfo visitExpr(NEQPrimOp op) { return {Equality, "!="}; }

      // Cat, DShl, DShr, ValidIf

      // Unary Prefix operators.
      SubExprInfo visitExpr(AndRPrimOp op) { return {Unary, "&"}; }
      SubExprInfo visitExpr(XorRPrimOp op) { return {Unary, "^"}; }
      SubExprInfo visitExpr(OrRPrimOp op) { return {Unary, "|"}; }
      SubExprInfo visitExpr(NotPrimOp op) { return {Unary, "~"}; }
    };

    subExprInfos[exp] = ExprPropertyComputer().dispatchExprVisitor(op);

    // Visit all subexpressions.
    worklist.append(op->operand_begin(), op->operand_end());
  }
}

/// Emit the specified value as a subexpression to the stream.
void ExprEmitter::emitSubExpr(Value exp, SubExprInfo exprInfo) {
  // If this is a non-expr or shouldn't be done inline, just refer to its
  // name.
  if (exprInfo.precedence == NonExpression) {
    os << emitter.getName(exp);
    return;
  }

  // Okay, this is an expression we should emit inline.  Do this through our
  // visitor.
  dispatchExprVisitor(exp.getDefiningOp(), exprInfo);
}

void ExprEmitter::emitUnaryPrefixExpr(Operation *op, SubExprInfo opInfo) {
  auto inputInfo = getInfo(op->getOperand(0));

  os << opInfo.syntax;
  if (inputInfo.precedence > Unary)
    os << '(';
  emitSubExpr(op->getOperand(0), inputInfo);
  if (inputInfo.precedence > Unary)
    os << ')';
}

void ExprEmitter::emitBinaryExpr(Operation *op, SubExprInfo opInfo) {
  auto lhsInfo = getInfo(op->getOperand(0));
  auto rhsInfo = getInfo(op->getOperand(1));

  if (opInfo.precedence < lhsInfo.precedence)
    os << '(';
  emitSubExpr(op->getOperand(0), lhsInfo);
  if (opInfo.precedence < lhsInfo.precedence)
    os << ')';
  os << ' ' << opInfo.syntax << ' ';

  if (opInfo.precedence < rhsInfo.precedence)
    os << '(';
  emitSubExpr(op->getOperand(1), rhsInfo);
  if (opInfo.precedence < rhsInfo.precedence)
    os << ')';
}

void ExprEmitter::visitExpr(ConstantOp op, SubExprInfo exprInfo) {
  auto resType = op.getType().cast<IntType>();
  if (resType.getWidthOrSentinel() == -1)
    return visitUnhandledExpr(op, exprInfo);

  os << resType.getWidth() << "'";
  if (resType.isSigned())
    os << 's';
  os << 'h';

  SmallString<32> valueStr;
  op.value().toStringUnsigned(valueStr, 16);
  os << valueStr;
}

void ExprEmitter::visitUnhandledExpr(Operation *op, SubExprInfo exprInfo) {
  emitter.emitOpError(op, "cannot emit this expression to Verilog");
  os << "<<unsupported expr: " << op->getName().getStringRef() << ">>";
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// Emit the specified value as an expression.  If this is an inline-emitted
/// expression, we emit that expression, otherwise we emit a reference to the
/// already computed name.  If 'forceRootExpr' is true, then this emits an
/// expression even if we typically don't do it inline.
///
void ModuleEmitter::emitExpression(Value exp, bool forceRootExpr) {
  ExprEmitter(os, *this).emitExpression(exp, forceRootExpr);
}

void ModuleEmitter::emitStatementExpression(Operation *op) {
  indent() << "assign " << getName(op->getResult(0)) << " = ";
  emitExpression(op->getResult(0), /*forceRootExpr=*/true);
  os << ";\n";
  // TODO: location information too.
}

void ModuleEmitter::emitStatement(ConnectOp op) {
  indent() << "assign ";
  emitExpression(op.lhs());
  os << " = ";
  emitExpression(op.rhs());
  os << ";\n";
  // TODO: location information too.
}

//===----------------------------------------------------------------------===//
// Module Driver
//===----------------------------------------------------------------------===//

/// Build up the symbol table for all of the values that need names in the
/// module.
void ModuleEmitter::collectNames(FModuleOp module) {
  SmallVector<ModulePortInfo, 8> portInfo;
  module.getPortInfo(portInfo);

  size_t nextPort = 0;
  for (auto &port : portInfo)
    addName(module.getArgument(nextPort++), port.first);

  for (auto &op : *module.getBodyBlock())
    collectOpNames(&op);
}

void ModuleEmitter::collectOpNames(Operation *op) {
  // If the op has no results, then it doesn't produce a name.
  if (op->getNumResults() == 0) {
    // When has no results, but it does have a body!
    if (auto when = dyn_cast<WhenOp>(op)) {
      for (auto &op : when.thenRegion().front())
        collectOpNames(&op);
      if (when.hasElseRegion())
        for (auto &op : when.elseRegion().front())
          collectOpNames(&op);
    }
    return;
  }

  assert(op->getNumResults() == 1 && "firrtl only has single-op results");

  // If the expression is inline, it doesn't need a name.
  if (isExpression(op) && isExpressionEmittedInline(op))
    return;

  // Otherwise, it must be an expression or a declaration like a wire.
  addName(op->getResult(0), op->getAttrOfType<StringAttr>("name"));
}

void ModuleEmitter::emitOperation(Operation *op) {
  // Handle expression statements.
  if (isExpression(op)) {
    if (!isExpressionEmittedInline(op))
      emitStatementExpression(op);
    return;
  }

  // Handle statements.
  // TODO: Refactor out to visitors.
  bool isStatement = false;
  TypeSwitch<Operation *>(op).Case<ConnectOp, SkipOp>([&](auto stmt) {
    isStatement = true;
    this->emitStatement(stmt);
  });
  if (isStatement)
    return;

  // Ignore the region terminator.
  if (isa<DoneOp>(op))
    return;

  emitOpError(op, "cannot emit this operation to Verilog");
}

void ModuleEmitter::emitFModule(FModuleOp module) {
  // Build our name table.
  collectNames(module);

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
  size_t portNumber = 0;
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

    // Emit the name.
    os << ' ' << getName(module.getArgument(portNumber++));
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

//===----------------------------------------------------------------------===//
// CircuitEmitter
//===----------------------------------------------------------------------===//

namespace {

class CircuitEmitter : public VerilogEmitterBase {
public:
  explicit CircuitEmitter(VerilogEmitterState &state)
      : VerilogEmitterBase(state) {}

  void emitMLIRModule(ModuleOp module);

private:
  void emitCircuit(CircuitOp circuit);
};

} // end anonymous namespace

void CircuitEmitter::emitCircuit(CircuitOp circuit) {
  for (auto &op : *circuit.getBody()) {
    if (auto module = dyn_cast<FModuleOp>(op)) {
      ModuleEmitter(state).emitFModule(module);
    } else if (!isa<DoneOp>(op))
      op.emitError("unknown operation");
  }
}

void CircuitEmitter::emitMLIRModule(ModuleOp module) {
  for (auto &op : *module.getBody()) {
    if (auto circuit = dyn_cast<CircuitOp>(op))
      emitCircuit(circuit);
    else if (!isa<ModuleTerminatorOp>(op))
      op.emitError("unknown operation");
  }
}

static LogicalResult emitVerilog(ModuleOp module, llvm::raw_ostream &os) {
  VerilogEmitterState state(os);
  CircuitEmitter(state).emitMLIRModule(module);
  return failure(state.encounteredError);
}

void spt::registerVerilogEmitterTranslation() {
  static TranslateFromMLIRRegistration toVerilog("emit-verilog", emitVerilog);
}
