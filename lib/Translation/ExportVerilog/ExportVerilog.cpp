//===- ExportVerilog.cpp - Verilog Emitter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Verilog emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Translation/ExportVerilog.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/RTL/RTLVisitors.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace mlir;

/// Should we emit wire decls in a block at the top of a module, or inline?
static constexpr bool emitInlineWireDecls = true;

/// This is the preferred source width for the generated Verilog.
static constexpr size_t preferredSourceWidth = 120;

/// This is a set accessed through getReservedWords() that contains all of the
/// Verilog names and other identifiers we need to avoid because of name
/// conflicts.
static llvm::ManagedStatic<StringSet<>> reservedWordCache;

//===----------------------------------------------------------------------===//
// Helper routines
//===----------------------------------------------------------------------===//

static bool isVerilogExpression(Operation *op) {
  // Merge is an expression according to the RTL dialect, but we need it emitted
  // as a statement with its own wire declaration.
  if (isa<rtl::MergeOp>(op))
    return false;

  // All RTL combinatorial logic ops are Verilog expressions.
  return rtl::isCombinatorial(op) || sv::isExpression(op);
}

/// Return the width of the specified type in bits or -1 if it isn't
/// supported.
static int getBitWidthOrSentinel(Type type) {
  return TypeSwitch<Type, int>(type)
      .Case<IntegerType>([](IntegerType integerType) {
        // Turn zero-bit values into single bit ones for simplicity.  This
        // generates correct logic, even though it isn't efficient.
        // FIXME: This actually isn't correct at all!
        auto result = integerType.getWidth();
        return result ? result : 1;
      })
      .Case<rtl::InOutType>([](rtl::InOutType inoutType) {
        return getBitWidthOrSentinel(inoutType.getElementType());
      })
      .Default([](Type) { return -1; });
}

/// Given a type that may be an array or inout, dig through recursive arrays to
/// find the underlying element type.
static Type getUnderlyingArrayElementType(Type type) {
  if (auto arrayElt = rtl::getAnyRTLArrayElementType(type))
    return getUnderlyingArrayElementType(arrayElt);
  if (auto inoutElt = rtl::getInOutElementType(type))
    return getUnderlyingArrayElementType(inoutElt);
  return type;
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

/// Return true if this is a noop cast that will emit with no syntax.
static bool isNoopCast(Operation *op) {
  // These are always noop casts.
  if (isa<rtl::ReadInOutOp>(op))
    return true;

  return false;
}

namespace {
/// This enum keeps track of the precedence level of various binary operators,
/// where a lower number binds tighter.
enum VerilogPrecedence {
  // Normal precedence levels.
  Symbol,          // Atomic symbol like "foo"
  Unary,           // Unary operators like ~foo
  Multiply,        // * , / , %
  Addition,        // + , -
  Shift,           // << , >>
  Comparison,      // > , >= , < , <=
  Equality,        // == , !=
  And,             // &
  Xor,             // ^ , ^~
  Or,              // |
  AndShortCircuit, // &&
  Conditional,     // ? :

  LowestPrecedence,  // Sentinel which is always the lowest precedence.
  ForceEmitMultiUse, // Sentinel saying to recursively emit a multi-used expr.
};
} // end anonymous namespace

/// Return a StringSet that contains all of the reserved names (e.g. Verilog
/// keywords) that we need to avoid for fear of name conflicts.
static const StringSet<> &getReservedWords() {
  auto &set = *reservedWordCache;
  if (set.empty()) {
    static const char *const reservedWords[] = {
#include "ReservedWords.def"
    };
    for (auto *word : reservedWords)
      set.insert(word);
  }
  return set;
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

  /// Emit the verilog type for the specified ground type, padding the text to
  /// the specified number of characters.
  void emitTypePaddedToWidth(Type type, size_t padToWidth, Operation *op);

  // All of the mutable state we are maintaining.
  VerilogEmitterState &state;

  /// The stream to emit to.
  raw_ostream &os;

private:
  VerilogEmitterBase(const VerilogEmitterBase &) = delete;
  void operator=(const VerilogEmitterBase &) = delete;
};

} // end anonymous namespace

void VerilogEmitterBase::emitTypePaddedToWidth(Type type, size_t padToWidth,
                                               Operation *op) {
  int bitWidth = getBitWidthOrSentinel(type);
  assert(bitWidth != 0 && "Shouldn't emit zero bit declarations");

  if (bitWidth == -1) {
    emitError(op, "value has an unsupported verilog type ") << type;
    os << "<<invalid type>>";
    return;
  }

  size_t emittedWidth = 0;
  if (bitWidth != 1) {
    os << '[' << (bitWidth - 1) << ":0]";
    emittedWidth = getPrintedIntWidth(bitWidth - 1) + 4;
  }

  if (emittedWidth < padToWidth)
    os.indent(padToWidth - emittedWidth);
}

//===----------------------------------------------------------------------===//
// ModuleEmitter
//===----------------------------------------------------------------------===//

namespace {

class ModuleEmitter : public VerilogEmitterBase {

public:
  explicit ModuleEmitter(VerilogEmitterState &state)
      : VerilogEmitterBase(state) {}

  void emitRTLModule(rtl::RTLModuleOp module);
  void emitRTLExternModule(rtl::RTLExternModuleOp module);
  void emitExpression(Value exp, SmallPtrSet<Operation *, 8> &emittedExprs,
                      bool forceRootExpr = false);

  /// Emit the specified expression and return it as a string.
  std::string
  emitExpressionToString(Value exp, SmallPtrSet<Operation *, 8> &emittedExprs,
                         VerilogPrecedence precedence = LowestPrecedence);

  // Statements.
  void emitStatementExpression(Operation *op);
  void emitStatement(rtl::MergeOp op);
  void emitStatement(rtl::ConnectOp op);
  void emitStatement(sv::BPAssignOp op);
  void emitStatement(sv::PAssignOp op);
  void emitStatement(sv::AliasOp op);
  void emitStatement(rtl::OutputOp op);
  void emitStatement(rtl::InstanceOp op);
  void emitStatement(sv::IfDefOp op);
  void emitStatement(sv::IfOp op);
  void emitStatement(sv::AlwaysOp op);
  void emitStatement(sv::InitialOp op);
  void emitStatement(sv::FWriteOp op);
  void emitStatement(sv::FatalOp op);
  void emitStatement(sv::FinishOp op);
  void emitStatement(sv::VerbatimOp op);
  void emitStatement(sv::AssertOp op);
  void emitStatement(sv::AssumeOp op);
  void emitStatement(sv::CoverOp op);
  void emitDecl(sv::InterfaceOp op);
  void emitDecl(sv::InterfaceSignalOp op);
  void emitDecl(sv::InterfaceModportOp op);
  void emitOperation(Operation *op);

  void collectNamesEmitDecls(Block &block);
  bool collectNamesEmitWires(rtl::InstanceOp instance);
  StringRef addName(Value value, StringRef name);
  StringRef addName(Value value, StringAttr nameAttr) {
    return addName(value, nameAttr ? nameAttr.getValue() : "");
  }

  StringRef getName(Value value) {
    auto *entry = nameTable[value];
    assert(entry && "value expected a name but doesn't have one");
    return entry->getKey();
  }

  /// Return the location information as a (potentially empty) string.
  std::string getLocationInfoAsString(const SmallPtrSet<Operation *, 8> &ops);

  /// If we have location information for any of the specified operations,
  /// aggregate it together and print a pretty comment specifying where the
  /// operations came from.  In any case, print a newline.
  void emitLocationInfoAndNewLine(const SmallPtrSet<Operation *, 8> &ops);

  // Conditional statement handling.

  enum class ConditionalStmtKind {
    /// Stuff that goes at the top of the conditional statement block.
    Declaration,
    /// Stuff that goes in an 'initial' block for simulation.
    Initial,
    /// Stuff that is gated by the positive edge of a clock.
    AlwaysAtPosEdge,
  };

  // Note: Preprocessor conditions are defined in the ConditionalStatement
  // struct below.

  void addDeclaration(std::string action, std::string locInfo,
                      std::string ppCond, unsigned partialOrder = 0) {
    conditionalStmts.push_back({ConditionalStmtKind::Declaration, "",
                                std::move(ppCond), "", partialOrder,
                                std::move(action), std::move(locInfo)});
  }

  void addInitial(std::string action, std::string locInfo,
                  std::string ppCond = {}, std::string condition = {},
                  unsigned partialOrder = 0) {
    conditionalStmts.push_back({ConditionalStmtKind::Initial, "",
                                std::move(ppCond), std::move(condition),
                                partialOrder, std::move(action),
                                std::move(locInfo)});
  }

  void addAtPosEdge(std::string action, std::string locInfo, std::string clock,
                    std::string ppCond = {}, std::string condition = {},
                    unsigned partialOrder = 0) {
    conditionalStmts.push_back({ConditionalStmtKind::AlwaysAtPosEdge,
                                std::move(clock), std::move(ppCond),
                                std::move(condition), partialOrder,
                                std::move(action), std::move(locInfo)});
  }

  // Set up the per-module state for random seeding of registers/memory.
  void emitRandomizeProlog() {
    // Only emit this prolog once.
    if (emittedRandomProlog)
      return;
    emittedRandomProlog = true;
    addInitial("`INIT_RANDOM_PROLOG_", /*locInfo=*/"");
  }

  /// ConditionalStatement are emitted lexically at the end of the module
  /// into blocks.
  struct ConditionalStatement {
    /// This specifies the kind of the statement, e.g. whether it goes into an
    /// initial block of something else.
    ConditionalStmtKind kind;

    /// For "always @" blocks, this is the clock expression they are gated on.
    /// This is empty for 'initial' blocks.
    std::string clock;

    /// If non-empty, this is a preprocessor condition that gates the statement.
    /// If the string starts with a ! character, then this is an `ifndef
    /// condition, otherwise it is an `ifdef condition.
    std::string ppCond;

    /// If non-empty, this is an 'if' condition that gates the statement.  If
    /// empty, the statement is unconditional.
    std::string condition;

    /// This is used to sort entries that have some partial ordering w.r.t. each
    /// other.  This can only be used to order actions conditionalized by the
    /// same ppCond and condition.
    unsigned partialOrder;

    /// This is the statement to emit.  If there is a condition, this should
    /// always be a single verilog statement that can be emitted without a
    /// begin/end clause.
    std::string action;

    /// This is location information (if any) to print.
    std::string locInfo;

    bool operator<(const ConditionalStatement &rhs) const {
      return std::make_tuple(kind, clock, ppCond, condition, partialOrder,
                             action, locInfo) <
             std::make_tuple(rhs.kind, rhs.clock, rhs.ppCond, rhs.condition,
                             rhs.partialOrder, rhs.action, locInfo);
    }
  };

  // Per module states.
  std::vector<ConditionalStatement> conditionalStmts;

  llvm::StringSet<> usedNames;
  llvm::DenseMap<Value, llvm::StringMapEntry<llvm::NoneType> *> nameTable;
  size_t nextGeneratedNameID = 0;

  /// This set keeps track of all of the expression nodes that need to be
  /// emitted as standalone wire declarations.  This can happen because they are
  /// multiply-used or because the user requires a name to reference.
  SmallPtrSet<Operation *, 16> outOfLineExpressions;

  // This is set to true after the first RANDOMIZE prolog has been emitted in
  // this module to the initial block.
  bool emittedRandomProlog = false;
  bool emittedInitVar = false;
};

} // end anonymous namespace

/// Add the specified name to the name table, auto-uniquing the name if
/// required.  If the name is empty, then this creates a unique temp name.
StringRef ModuleEmitter::addName(Value value, StringRef name) {
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

  // Get the list of reserved words we need to avoid.  We could prepopulate this
  // into the used words cache, but it is large and immutable, so we just query
  // it when needed.
  auto &reservedWords = getReservedWords();

  // Check to see if this name is available - if so, use it.
  if (!reservedWords.count(name)) {
    auto insertResult = usedNames.insert(name);
    if (insertResult.second) {
      nameTable[value] = &*insertResult.first;
      return insertResult.first->getKey();
    }
  }

  // If not, we need to auto-unique it.
  SmallVector<char, 16> nameBuffer(name.begin(), name.end());
  nameBuffer.push_back('_');
  auto baseSize = nameBuffer.size();

  // Try until we find something that works.
  while (1) {
    auto suffix = llvm::utostr(nextGeneratedNameID++);
    nameBuffer.append(suffix.begin(), suffix.end());
    name = StringRef(nameBuffer.data(), nameBuffer.size());

    if (!reservedWords.count(name)) {
      auto insertResult = usedNames.insert(name);
      if (insertResult.second) {
        nameTable[value] = &*insertResult.first;
        return insertResult.first->getKey();
      }
    }

    // Chop off the suffix and try again.
    nameBuffer.resize(baseSize);
  }
}

/// Return the location information as a (potentially empty) string.
std::string
ModuleEmitter::getLocationInfoAsString(const SmallPtrSet<Operation *, 8> &ops) {
  std::string resultStr;
  llvm::raw_string_ostream sstr(resultStr);

  // Multiple operations may come from the same location or may not have useful
  // location info.  Unique it now.
  SmallPtrSet<Attribute, 8> locations;
  for (auto *op : ops) {
    if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
      locations.insert(loc);
  }

  auto printLoc = [&](FileLineColLoc loc) {
    sstr << loc.getFilename();
    if (auto line = loc.getLine()) {
      sstr << ':' << line;
      if (auto col = loc.getColumn())
        sstr << ':' << col;
    }
  };

  switch (locations.size()) {
  case 1:
    printLoc((*locations.begin()).cast<FileLineColLoc>());
    LLVM_FALLTHROUGH;
  case 0:
    return sstr.str();
  default:
    break;
  }

  // Sort the entries.
  SmallVector<FileLineColLoc, 8> locVector;
  locVector.reserve(locations.size());
  for (auto loc : locations)
    locVector.push_back(loc.cast<FileLineColLoc>());

  llvm::array_pod_sort(
      locVector.begin(), locVector.end(),
      [](const FileLineColLoc *lhs, const FileLineColLoc *rhs) -> int {
        if (auto fn = lhs->getFilename().compare(rhs->getFilename()))
          return fn;
        if (lhs->getLine() != rhs->getLine())
          return lhs->getLine() < rhs->getLine() ? -1 : 1;
        return lhs->getColumn() < rhs->getColumn() ? -1 : 1;
      });

  // The entries are sorted by filename, line, col.  Try to merge together
  // entries to reduce verbosity on the column info.
  StringRef lastFileName;
  for (size_t i = 0, e = locVector.size(); i != e;) {
    if (i != 0)
      sstr << ", ";

    // Print the filename if it changed.
    auto first = locVector[i];
    if (first.getFilename() != lastFileName) {
      lastFileName = first.getFilename();
      sstr << lastFileName;
    }

    // Scan for entires with the same file/line.
    size_t end = i + 1;
    while (end != e && first.getFilename() == locVector[end].getFilename() &&
           first.getLine() == locVector[end].getLine())
      ++end;

    // If we have one entry, print it normally.
    if (end == i + 1) {
      if (auto line = first.getLine()) {
        sstr << ':' << line;
        if (auto col = first.getColumn())
          sstr << ':' << col;
      }
      ++i;
      continue;
    }

    // Otherwise print a brace enclosed list.
    sstr << ':' << first.getLine() << ":{";
    while (i != end) {
      sstr << locVector[i++].getColumn();

      if (i != end)
        sstr << ',';
    }
    sstr << '}';
  }

  return sstr.str();
}

/// If we have location information for any of the specified operations,
/// aggregate it together and print a pretty comment specifying where the
/// operations came from.  In any case, print a newline.
void ModuleEmitter::emitLocationInfoAndNewLine(
    const SmallPtrSet<Operation *, 8> &ops) {
  auto locInfo = getLocationInfoAsString(ops);
  if (!locInfo.empty())
    os << "\t// " << locInfo;
  os << '\n';
}

//===----------------------------------------------------------------------===//
// Expression Emission
//===----------------------------------------------------------------------===//

namespace {

/// This enum keeps track of whether the emitted subexpression is signed or
/// unsigned as seen from the Verilog language perspective.
enum SubExprSignedness { IsSigned, IsUnsigned };

/// This is information precomputed about each subexpression in the tree we
/// are emitting as a unit.
struct SubExprInfo {
  /// The precedence of this expression.
  VerilogPrecedence precedence;

  /// The signedness of the expression.
  SubExprSignedness signedness;

  SubExprInfo(VerilogPrecedence precedence, SubExprSignedness signedness)
      : precedence(precedence), signedness(signedness) {}
};
} // namespace

namespace {
/// This builds a recursively nested expression from an SSA use-def graph.  This
/// uses a post-order walk, but it needs to obey precedence and signedness
/// constraints that depend on the behavior of the child nodes.  To handle this,
/// we emit the characters to a SmallVector which allows us to emit a bunch of
/// stuff, then pre-insert parentheses and other things if we find out that it
/// was needed later.
class ExprEmitter : public rtl::CombinatorialVisitor<ExprEmitter, SubExprInfo>,
                    public sv::Visitor<ExprEmitter, SubExprInfo> {
public:
  /// Create an ExprEmitter for the specified module emitter, and keeping track
  /// of any emitted expressions in the specified set.
  ExprEmitter(ModuleEmitter &emitter, SmallPtrSet<Operation *, 8> &emittedExprs)
      : emitter(emitter), emittedExprs(emittedExprs), os(resultBuffer) {}

  void emitExpression(Value exp, bool forceRootExpr, raw_ostream &os);

  /// Emit the specified expression and return it as a string.
  std::string emitExpressionToString(Value exp, VerilogPrecedence precedence);

  /// Do a best-effort job of looking through noop cast operations.
  Value lookThroughNoopCasts(Value value) {
    if (auto *op = value.getDefiningOp())
      if (isNoopCast(op) && !emitter.outOfLineExpressions.count(op))
        return lookThroughNoopCasts(op->getOperand(0));
    return value;
  }

  ModuleEmitter &emitter;

private:
  friend class rtl::CombinatorialVisitor<ExprEmitter, SubExprInfo>;
  friend class sv::Visitor<ExprEmitter, SubExprInfo>;

  /// Emit the specified value as a subexpression to the stream.
  SubExprInfo emitSubExpr(Value exp,
                          VerilogPrecedence parenthesizeIfLooserThan);

  SubExprInfo visitUnhandledExpr(Operation *op);
  SubExprInfo visitInvalidComb(Operation *op) { return dispatchSVVisitor(op); }
  SubExprInfo visitUnhandledComb(Operation *op) {
    return visitUnhandledExpr(op);
  }
  SubExprInfo visitUnhandledSV(Operation *op) { return visitUnhandledExpr(op); }

  using Visitor::visitSV;

  /// Emit a verilog bit selection operation like x[4:0], the bit numbers are
  /// inclusive like verilog.
  SubExprInfo emitBitSelect(Value operand, unsigned hiBit, unsigned loBit);

  SubExprInfo emitBinary(Operation *op, VerilogPrecedence prec,
                         const char *syntax, bool opSigned = false);

  SubExprInfo emitVariadic(Operation *op, VerilogPrecedence prec,
                           const char *syntax);

  SubExprInfo emitUnary(Operation *op, const char *syntax,
                        bool forceUnsigned = false);

  SubExprInfo emitNoopCast(Operation *op) {
    return emitSubExpr(op->getOperand(0), LowestPrecedence);
  }

  SubExprInfo visitSV(sv::GetModportOp op);
  SubExprInfo visitSV(sv::TextualValueOp op);

  // Noop cast operators.
  SubExprInfo visitComb(rtl::ReadInOutOp op) { return emitNoopCast(op); }

  // Other
  SubExprInfo visitComb(rtl::ArrayIndexOp op);
  SubExprInfo visitComb(rtl::MuxOp op);

  // RTL Dialect Operations
  using CombinatorialVisitor::visitComb;
  SubExprInfo visitComb(rtl::ConstantOp op);
  SubExprInfo visitComb(rtl::AddOp op) {
    return emitVariadic(op, Addition, "+");
  }
  SubExprInfo visitComb(rtl::SubOp op) { return emitBinary(op, Addition, "-"); }
  SubExprInfo visitComb(rtl::MulOp op) {
    return emitVariadic(op, Multiply, "*");
  }
  SubExprInfo visitComb(rtl::DivUOp op) {
    return emitBinary(op, Multiply, "/");
  }
  SubExprInfo visitComb(rtl::DivSOp op) {
    return emitBinary(op, Multiply, "/", true);
  }
  SubExprInfo visitComb(rtl::ModUOp op) {
    return emitBinary(op, Multiply, "%");
  }
  SubExprInfo visitComb(rtl::ModSOp op) {
    return emitBinary(op, Multiply, "%", true);
  }
  SubExprInfo visitComb(rtl::ShlOp op) { return emitBinary(op, Shift, "<<"); }
  SubExprInfo visitComb(rtl::ShrUOp op) { return emitBinary(op, Shift, ">>"); }
  SubExprInfo visitComb(rtl::ShrSOp op) {
    return emitBinary(op, Shift, ">>>", true);
  }
  SubExprInfo visitComb(rtl::AndOp op) { return emitVariadic(op, And, "&"); }
  SubExprInfo visitComb(rtl::OrOp op) { return emitVariadic(op, Or, "|"); }
  SubExprInfo visitComb(rtl::XorOp op) {
    if (op.getNumOperands() == 2)
      if (auto cst = dyn_cast_or_null<rtl::ConstantOp>(
              op.getOperand(1).getDefiningOp()))
        if (cst.getValue().isAllOnesValue())
          return emitUnary(op, "~", true);

    return emitVariadic(op, Xor, "^");
  }

  SubExprInfo visitComb(rtl::AndROp op) { return emitUnary(op, "&", true); }
  SubExprInfo visitComb(rtl::OrROp op) { return emitUnary(op, "|", true); }
  SubExprInfo visitComb(rtl::XorROp op) { return emitUnary(op, "^", true); }

  SubExprInfo visitComb(rtl::SExtOp op);
  SubExprInfo visitComb(rtl::ZExtOp op);
  SubExprInfo visitComb(rtl::ConcatOp op);
  SubExprInfo visitComb(rtl::ExtractOp op);

  // RTL Comparison Operations
  SubExprInfo visitComb(rtl::ICmpOp op) {
    std::array<const char *, 10> symop{"==", "!=", "<",  "<=", ">",
                                       ">=", "<",  "<=", ">",  ">="};
    std::array<bool, 10> signop{false, false, true,  true,  true,
                                true,  false, false, false, false};
    auto pred = static_cast<uint64_t>(op.predicate());
    auto symopstr = symop[pred];
    return emitBinary(op, Comparison, symopstr, signop[pred]);
  }

private:
  SmallPtrSet<Operation *, 8> &emittedExprs;
  SmallString<128> resultBuffer;
  llvm::raw_svector_ostream os;
};
} // end anonymous namespace

/// Emit the specified value as an expression.  If this is an inline-emitted
/// expression, we emit that expression, otherwise we emit a reference to the
/// already computed name.  If 'forceRootExpr' is true, then this emits an
/// expression even if we typically don't do it inline.
///
void ExprEmitter::emitExpression(Value exp, bool forceRootExpr,
                                 raw_ostream &os) {
  // Emit the expression.
  emitSubExpr(exp, forceRootExpr ? ForceEmitMultiUse : LowestPrecedence);

  // Once the expression is done, we can emit the result to the stream.
  os << resultBuffer;
}

/// Emit the specified expression and return it as a string.
std::string ExprEmitter::emitExpressionToString(Value exp,
                                                VerilogPrecedence precedence) {
  emitSubExpr(exp, precedence);
  return std::string(resultBuffer.begin(), resultBuffer.end());
}

SubExprInfo ExprEmitter::emitBinary(Operation *op, VerilogPrecedence prec,
                                    const char *syntax, bool opSigned) {
  if (opSigned)
    os << "$signed(";
  auto lhsInfo = emitSubExpr(op->getOperand(0), prec);
  if (opSigned)
    os << ")";
  os << ' ' << syntax << ' ';

  // The precedence of the RHS operand must be tighter than this operator if
  // they have a different opcode in order to handle things like "x-(a+b)".
  // This isn't needed on the LHS, because the relevant Verilog operators are
  // left-associative.
  //
  auto *rhsOperandOp = lookThroughNoopCasts(op->getOperand(1)).getDefiningOp();
  auto rhsPrec = VerilogPrecedence(prec - 1);
  if (rhsOperandOp && op->getName() == rhsOperandOp->getName())
    rhsPrec = prec;

  if (opSigned)
    os << "$signed(";
  auto rhsInfo = emitSubExpr(op->getOperand(1), rhsPrec);
  if (opSigned)
    os << ")";

  // If we have a strict sign, then match the firrtl operation sign.
  // Otherwise, the result is signed if both operands are signed.
  SubExprSignedness signedness;
  if (opSigned ||
      (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned))
    signedness = IsSigned;
  else
    signedness = IsUnsigned;

  return {prec, signedness};
}

SubExprInfo ExprEmitter::emitVariadic(Operation *op, VerilogPrecedence prec,
                                      const char *syntax) {
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](Value v1) { emitSubExpr(v1, prec); },
      [&] { os << ' ' << syntax << ' '; });

  return {prec, IsUnsigned};
}

SubExprInfo ExprEmitter::emitUnary(Operation *op, const char *syntax,
                                   bool forceUnsigned) {
  os << syntax;
  auto signedness = emitSubExpr(op->getOperand(0), Unary).signedness;
  return {Unary, forceUnsigned ? IsUnsigned : signedness};
}

/// Emit the specified value as a subexpression to the stream.
SubExprInfo
ExprEmitter::emitSubExpr(Value exp,
                         VerilogPrecedence parenthesizeIfLooserThan) {
  auto *op = exp.getDefiningOp();
  bool shouldEmitInlineExpr = op && isVerilogExpression(op);

  // Don't emit this expression inline if it has multiple uses.
  if (shouldEmitInlineExpr && parenthesizeIfLooserThan != ForceEmitMultiUse &&
      emitter.outOfLineExpressions.count(op))
    shouldEmitInlineExpr = false;

  // If this is a non-expr or shouldn't be done inline, just refer to its
  // name.
  if (!shouldEmitInlineExpr) {
    os << emitter.getName(exp);
    return {Symbol, IsUnsigned};
  }

  unsigned subExprStartIndex = resultBuffer.size();

  // Okay, this is an expression we should emit inline.  Do this through our
  // visitor.
  auto expInfo = dispatchCombinatorialVisitor(exp.getDefiningOp());

  // Check cases where we have to insert things before the expression now that
  // we know things about it.
  if (expInfo.precedence > parenthesizeIfLooserThan) {
    // If this subexpression would bind looser than the expression it is bound
    // into, then we need to parenthesize it.  Insert the parentheses
    // retroactively.
    resultBuffer.insert(resultBuffer.begin() + subExprStartIndex, '(');
    os << ')';
  }

  // Remember that we emitted this.
  emittedExprs.insert(exp.getDefiningOp());
  return expInfo;
}

SubExprInfo ExprEmitter::visitComb(rtl::SExtOp op) {
  auto inType = op.getOperand().getType().cast<IntegerType>();
  auto inWidth = inType.getWidth();
  auto destWidth = op.getType().cast<IntegerType>().getWidth();

  // Handle sign extend from a single bit in a pretty way.
  if (inWidth == 1) {
    os << '{' << destWidth << '{';
    emitSubExpr(op.getOperand(), LowestPrecedence);
    os << "}}";
    return {Unary, IsUnsigned};
  }

  // Otherwise, this is a sign extension of a general expression.
  // TODO(QoI): Instead of emitting the expression multiple times, we should
  // emit it to a known name.
  os << "{{" << (destWidth - inWidth) << '{';
  emitSubExpr(op.getOperand(), Unary);
  os << '[' << (inWidth - 1) << "]}}, ";
  emitSubExpr(op.getOperand(), LowestPrecedence);
  os << "}";
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(rtl::ZExtOp op) {
  auto inType = op.getOperand().getType().cast<IntegerType>();
  auto inWidth = inType.getWidth();
  auto destWidth = op.getType().cast<IntegerType>().getWidth();

  // A zero extension just fills the top with numbers.
  os << "{{" << (destWidth - inWidth) << "'d0}, ";
  emitSubExpr(op.getOperand(), LowestPrecedence);
  os << "}";
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(rtl::ConcatOp op) {
  os << '{';
  llvm::interleaveComma(op.getOperands(), os,
                        [&](Value v) { emitSubExpr(v, LowestPrecedence); });

  os << '}';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(rtl::ExtractOp op) {
  unsigned dstWidth = op.getType().cast<IntegerType>().getWidth();
  return emitBitSelect(op.input(), op.lowBit() + dstWidth - 1, op.lowBit());
}

/// Emit a verilog bit selection operation like x[4:0], the bit numbers are
/// inclusive like verilog.
///
/// Note that anything that emits a BitSelect must be handled in the
/// isExpressionUnableToInline.
SubExprInfo ExprEmitter::emitBitSelect(Value operand, unsigned hiBit,
                                       unsigned loBit) {
  auto x = emitSubExpr(operand, LowestPrecedence);
  assert(x.precedence == Symbol &&
         "should be handled by isExpressionUnableToInline");

  // If we're extracting the whole input, just return it.  This is valid but
  // non-canonical IR, and we don't want to generate invalid Verilog.
  if (loBit == 0 &&
      unsigned(getBitWidthOrSentinel(operand.getType())) == hiBit + 1)
    return x;

  os << '[' << hiBit;
  if (hiBit != loBit) // Emit x[4] instead of x[4:4].
    os << ':' << loBit;
  os << ']';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(sv::GetModportOp op) {
  os << emitter.getName(op.iface()) + "." + op.field();
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(sv::TextualValueOp op) {
  os << op.string();
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(rtl::ConstantOp op) {
  auto resType = op.getType().cast<IntegerType>();
  os << resType.getWidth() << '\'';
  if (resType.isSigned())
    os << 's';
  os << 'h';

  SmallString<32> valueStr;
  op.getValue().toStringUnsigned(valueStr, 16);
  os << valueStr;
  return {Unary, resType.isSigned() ? IsSigned : IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(rtl::ArrayIndexOp op) {
  auto arrayPrec = emitSubExpr(op.input(), Symbol);
  os << '[';
  emitSubExpr(op.index(), LowestPrecedence);
  os << ']';
  return {Symbol, arrayPrec.signedness};
}

SubExprInfo ExprEmitter::visitComb(rtl::MuxOp op) {
  // The ?: operator is right associative.
  emitSubExpr(op.cond(), VerilogPrecedence(Conditional - 1));
  os << " ? ";
  auto lhsInfo =
      emitSubExpr(op.trueValue(), VerilogPrecedence(Conditional - 1));
  os << " : ";
  auto rhsInfo = emitSubExpr(op.falseValue(), Conditional);

  SubExprSignedness signedness = IsUnsigned;
  if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
    signedness = IsSigned;

  return {Conditional, signedness};
}

SubExprInfo ExprEmitter::visitUnhandledExpr(Operation *op) {
  emitter.emitOpError(op, "cannot emit this expression to Verilog");
  os << "<<unsupported expr: " << op->getName().getStringRef() << ">>";
  return {Symbol, IsUnsigned};
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// Emit the specified value as an expression.  If this is an inline-emitted
/// expression, we emit that expression, otherwise we emit a reference to the
/// already computed name.  If 'forceRootExpr' is true, then this emits an
/// expression even if we typically don't do it inline.
///
void ModuleEmitter::emitExpression(Value exp,
                                   SmallPtrSet<Operation *, 8> &emittedExprs,
                                   bool forceRootExpr) {
  ExprEmitter(*this, emittedExprs).emitExpression(exp, forceRootExpr, os);
}

/// Emit the specified expression and return it as a string.
std::string
ModuleEmitter::emitExpressionToString(Value exp,
                                      SmallPtrSet<Operation *, 8> &emittedExprs,
                                      VerilogPrecedence precedence) {
  return ExprEmitter(*this, emittedExprs)
      .emitExpressionToString(exp, precedence);
}

void ModuleEmitter::emitStatementExpression(Operation *op) {
  // This is invoked for expressions that have a non-single use.  This could
  // either be because they are dead or because they have multiple uses.
  if (op->getResult(0).use_empty()) {
    indent() << "// Unused: ";
  } else if (emitInlineWireDecls) {
    auto type = op->getResult(0).getType();
    indent() << "wire ";

    if (getBitWidthOrSentinel(type) != 1) {
      emitTypePaddedToWidth(type, 0, op);
      os << ' ';
    }
    os << getName(op->getResult(0)) << " = ";
  } else {
    indent() << "assign " << getName(op->getResult(0)) << " = ";
  }
  SmallPtrSet<Operation *, 8> emittedExprs;
  emitExpression(op->getResult(0), emittedExprs, /*forceRootExpr=*/true);
  os << ';';
  emitLocationInfoAndNewLine(emittedExprs);
}

void ModuleEmitter::emitStatement(rtl::MergeOp op) {
  SmallPtrSet<Operation *, 8> ops;

  // Emit "a = rtl.merge x, y, z" as:
  //   assign a = x;
  //   assign a = y;
  //   assign a = z;
  for (auto operand : op.getOperands()) {
    ops.insert(op);
    indent() << "assign " << getName(op) << " = ";
    emitExpression(operand, ops);
    os << ';';
    emitLocationInfoAndNewLine(ops);
    ops.clear();
  }
}

void ModuleEmitter::emitStatement(rtl::ConnectOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "assign ";
  emitExpression(op.dest(), ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::BPAssignOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent();
  emitExpression(op.dest(), ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::PAssignOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent();
  emitExpression(op.dest(), ops);
  os << " <= ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::AliasOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "alias ";
  llvm::interleave(
      op.getOperands(), os, [&](Value v) { emitExpression(v, ops); }, " = ");
  os << ';';
  emitLocationInfoAndNewLine(ops);
}

/// For OutputOp we put "assign" statements at the end of the Verilog module to
/// assign the module outputs to intermediate wires.
void ModuleEmitter::emitStatement(rtl::OutputOp op) {
  SmallPtrSet<Operation *, 8> ops;

  SmallVector<rtl::ModulePortInfo, 8> ports;
  rtl::RTLModuleOp parent = op.getParentOfType<rtl::RTLModuleOp>();
  parent.getPortInfo(ports);
  size_t operandIndex = 0;
  for (rtl::ModulePortInfo port : ports) {
    if (!port.isOutput())
      continue;
    ops.clear();
    ops.insert(op);
    indent() << "assign " << port.getName() << " = ";
    emitExpression(op.getOperand(operandIndex), ops);
    os << ';';
    emitLocationInfoAndNewLine(ops);
    ++operandIndex;
  }
}

void ModuleEmitter::emitStatement(sv::FWriteOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "$fwrite(32'h80000002, \"";
  os.write_escaped(op.string());
  os << '"';

  for (auto operand : op.operands()) {
    os << ", " << emitExpressionToString(operand, ops);
  }
  os << ");";
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::FatalOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "$fatal;";
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::VerbatimOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Drop an extraneous \n off the end of the string if present.
  StringRef string = op.string();
  if (string.endswith("\n"))
    string = string.drop_back();

  // Emit each \n separated piece of the string with each piece properly
  // indented.  The convention is to not emit the \n so
  // emitLocationInfoAndNewLine can do that for the last line.
  bool isFirst = true;
  indent();

  // Emit each line of the string at a time.
  while (!string.empty()) {
    auto lhsRhs = string.split('\n');
    if (isFirst)
      isFirst = false;
    else {
      os << '\n';
      indent();
    }

    StringRef line = lhsRhs.first;

    // Perform operand substitions as we emit the line string.  We turn {{42}}
    // into the value of operand 42.

    // Scan 'line' for a substitution, emitting any non-substitution prefix,
    // then the mentioned operand, chopping the relevant text off 'line' and
    // returning true.  This returns false if no substitution is found.
    auto emitUntilSubstitution = [&](size_t next = 0) -> bool {
      size_t start = 0;
      while (1) {
        next = line.find("{{", next);
        if (next == StringRef::npos)
          return false;

        // Check to make sure we have a number followed by }}.  If not, we
        // ignore the {{ sequence as something that could happen in Verilog.
        next += 2;
        start = next;
        while (next < line.size() && isdigit(line[next]))
          ++next;
        // We need at least one digit.
        if (start == next)
          continue;

        // We must have a }} right after the digits.
        if (!line.substr(next).startswith("}}"))
          continue;

        // We must be able to decode the integer into an unsigned.
        unsigned operandNo = 0;
        if (line.drop_front(start)
                .take_front(next - start)
                .getAsInteger(10, operandNo)) {
          op.emitError("operand substitution too large");
          continue;
        }
        next += 2;

        if (operandNo >= op.operands().size()) {
          op.emitError("operand " + llvm::utostr(operandNo) + " isn't valid");
          continue;
        }

        // Emit any text before the substitution.
        os << line.take_front(start - 2);

        // Emit the operand.
        os << emitExpressionToString(op.operands()[operandNo], ops);

        // Forget about the part we emitted.
        line = line.drop_front(next);
        return true;
      }
    };

    // Emit all the substitutions.
    while (emitUntilSubstitution())
      ;

    // Emit any text after the last substitution.
    os << line;
    string = lhsRhs.second;
  }

  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::FinishOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "$finish;";
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::AssertOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "assert(" << emitExpressionToString(op.predicate(), ops) << ");";
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::AssumeOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "assume(" << emitExpressionToString(op.property(), ops) << ");";
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::CoverOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "cover(" << emitExpressionToString(op.property(), ops) << ");";
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(sv::IfDefOp op) {
  auto cond = op.cond();

  if (cond.startswith("!"))
    indent() << "`ifndef " << cond.drop_front(1);
  else
    indent() << "`ifdef " << cond;

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  emitLocationInfoAndNewLine(ops);

  addIndent();
  for (auto &op : op.getThenBlock()->without_terminator())
    emitOperation(&op);
  reduceIndent();

  if (op.hasElse()) {
    indent() << "`else\n";
    addIndent();
    for (auto &op : op.getElseBlock()->without_terminator())
      emitOperation(&op);
    reduceIndent();
  }

  indent() << "`endif\n";
}

/// Emit the body of a control flow statement that is surrounded by begin/end
/// markers if non-singular.  If the control flow construct is multi-line and
/// if multiLineComment is non-null, the string is included in a comment after
/// the 'end' to make it easier to associate.
static void emitBeginEndRegion(Block *block,
                               SmallPtrSet<Operation *, 8> &locationOps,
                               ModuleEmitter &emitter,
                               StringRef multiLineComment = StringRef()) {
  auto isSingleVerilogStatement = [&](Operation &op) {
    // Not all expressions and statements are guaranteed to emit a single
    // Verilog statement (for the purposes of if statements).  Just do a simple
    // check here for now.  This can be improved over time.
    return isa<sv::FWriteOp>(op) || isa<sv::FinishOp>(op) ||
           isa<sv::FatalOp>(op) || isa<sv::AssertOp>(op) ||
           isa<sv::AssumeOp>(op) || isa<sv::CoverOp>(op) ||
           isa<sv::BPAssignOp>(op) || isa<sv::PAssignOp>(op) ||
           isa<rtl::ConnectOp>(op);
  };

  // Determine if we can omit the begin/end keywords.
  bool hasOneStmt = llvm::hasSingleElement(block->without_terminator()) &&
                    isSingleVerilogStatement(block->front());
  if (!hasOneStmt)
    emitter.os << " begin";
  emitter.emitLocationInfoAndNewLine(locationOps);

  emitter.addIndent();
  for (auto &op : block->without_terminator())
    emitter.emitOperation(&op);
  emitter.reduceIndent();

  if (!hasOneStmt) {
    emitter.indent() << "end";
    if (!multiLineComment.empty())
      emitter.os << " // " << multiLineComment;
    emitter.os << '\n';
  }
}

void ModuleEmitter::emitStatement(sv::IfOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "if (" << emitExpressionToString(op.cond(), ops) << ')';
  emitBeginEndRegion(op.getBodyBlock(), ops, *this);
}

void ModuleEmitter::emitStatement(sv::AlwaysOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto printEvent = [&](sv::AlwaysOp::Condition cond) {
    os << stringifyEventControl(cond.event) << ' '
       << emitExpressionToString(cond.value, ops);
  };

  switch (op.getNumConditions()) {
  case 0:
    indent() << "always @*";
    break;
  case 1:
    indent() << "always @(";
    printEvent(op.getCondition(0));
    os << ')';
    break;
  default:
    indent() << "always @(";
    printEvent(op.getCondition(0));
    for (size_t i = 1, e = op.getNumConditions(); i != e; ++i) {
      os << " or ";
      printEvent(op.getCondition(i));
    }
    os << ')';
    break;
  }

  // Build the comment string, leave out the signal expressions (since they
  // can be large).
  std::string comment;
  if (op.getNumConditions() == 0) {
    comment = "always @*";
  } else {
    comment = "always @(";
    llvm::interleave(
        op.events(),
        [&](Attribute eventAttr) {
          auto event = EventControl(eventAttr.cast<IntegerAttr>().getInt());
          comment += stringifyEventControl(event);
        },
        [&]() { comment += ", "; });
    comment += ')';
  }

  emitBeginEndRegion(op.getBodyBlock(), ops, *this, comment);
}

void ModuleEmitter::emitStatement(sv::InitialOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "initial";
  emitBeginEndRegion(op.getBodyBlock(), ops, *this, "initial");
}

void ModuleEmitter::emitStatement(rtl::InstanceOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto *moduleOp = op.getReferencedModule();
  assert(moduleOp && "Invalid IR");

  // If this is a reference to an external module with a hard coded Verilog
  // name, then use it here.  This is a hack because we lack proper support for
  // parameterized modules in the RTL dialect.
  if (auto extMod = dyn_cast<rtl::RTLExternModuleOp>(moduleOp)) {
    indent() << extMod.getVerilogModuleName();
  } else {
    indent() << op.moduleName();
  }

  // Helper that prints a parameter constant value in a Verilog compatible way.
  auto printParmValue = [&](Attribute value) {
    if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
      os << intAttr.getValue();
    } else if (auto strAttr = value.dyn_cast<StringAttr>()) {
      os << '"';
      os.write_escaped(strAttr.getValue());
      os << '"';
    } else if (auto fpAttr = value.dyn_cast<FloatAttr>()) {
      // TODO: relying on float printing to be precise is not a good idea.
      os << fpAttr.getValueAsDouble();
    } else {
      os << "<<UNKNOWN MLIRATTR: " << value << ">>";
      emitOpError(op, "unknown extmodule parameter value");
    }
  };

  // If this is a parameterized module, then emit the parameters.
  if (auto paramDictOpt = op.parameters()) {
    DictionaryAttr paramDict = paramDictOpt.getValue();
    if (!paramDict.empty()) {
      os << " #(";
      llvm::interleaveComma(paramDict, os, [&](NamedAttribute elt) {
        os << '.' << elt.first << '(';
        printParmValue(elt.second);
        os << ')';
      });
      os << ')';
    }
  }

  os << ' ' << op.instanceName() << " (";
  emitLocationInfoAndNewLine(ops);

  SmallVector<rtl::ModulePortInfo, 8> portInfo;
  getModulePortInfo(moduleOp, portInfo);

  // Emit the argument and result ports.
  auto opArgs = op.inputs();
  auto opResults = op.getResults();
  for (auto &elt : portInfo) {
    // Figure out which value we are emitting.
    bool isLast = &elt == &portInfo.back();
    Value portVal = elt.isOutput() ? opResults[elt.argNum] : opArgs[elt.argNum];

    // Emit the port's name.
    indent() << "  ." << StringRef(elt.getName()) << " (";

    // Emit the value as an expression.
    ops.clear();
    emitExpression(portVal, ops);
    os << (isLast ? ")" : "),");
    emitLocationInfoAndNewLine(ops);
  }
  indent() << ");\n";
}

void ModuleEmitter::emitDecl(sv::InterfaceOp op) {
  os << "interface " << op.sym_name() << ";\n";

  addIndent();
  for (auto &op : op.getBodyBlock()->without_terminator())
    emitOperation(&op);
  reduceIndent();

  os << "endinterface\n\n";
}

void ModuleEmitter::emitDecl(sv::InterfaceSignalOp op) {
  indent() << "logic ";

  auto type = op.type();
  if (getBitWidthOrSentinel(type) != 1) {
    emitTypePaddedToWidth(type, 0, op);
    os << ' ';
  }

  os << op.sym_name() << ";\n";
}

void ModuleEmitter::emitDecl(sv::InterfaceModportOp op) {
  indent() << "modport " << op.sym_name() << '(';

  llvm::interleaveComma(op.ports(), os, [&](const Attribute &portAttr) {
    auto port = portAttr.cast<sv::ModportStructAttr>();
    os << port.direction().getValue() << ' ' << port.signal().getValue();
  });

  os << ");\n";
}

//===----------------------------------------------------------------------===//
// Module Driver
//===----------------------------------------------------------------------===//

/// Most expressions are invalid to bit-select from in Verilog, but some things
/// are ok.  Return true if it is ok to inline bitselect from the result of this
/// expression.  It is conservatively correct to return false.
static bool isOkToBitSelectFrom(Value v) {
  // Module ports are always ok to bit select from.
  if (v.getDefiningOp())
    // TODO: We could handle concat and other operators here.
    return false;

  return true;
}

/// Return true if we are unable to ever inline the specified operation.  This
/// happens because not all Verilog expressions are composable, notably you can
/// only use bit selects like x[4:6] on simple expressions.
static bool isExpressionUnableToInline(Operation *op) {
  // Scan the users of the operation to see if any of them need this to be
  // emitted out-of-line.
  for (auto user : op->getUsers()) {
    // Verilog bit selection is required by the standard to be:
    // "a vector, packed array, packed structure, parameter or concatenation".
    // It cannot be an arbitrary expression.
    if (isa<rtl::ExtractOp>(user))
      if (!isOkToBitSelectFrom(op->getResult(0)))
        return true;

    // Sign extend (when the operand isn't a single bit) requires a bitselect
    // syntactically.
    if (auto sext = dyn_cast<rtl::SExtOp>(user)) {
      auto sextOperandType = sext.getOperand().getType().cast<IntegerType>();
      if (sextOperandType.getWidth() != 1 &&
          !isOkToBitSelectFrom(op->getResult(0)))
        return true;
    }
  }
  return false;
}

/// Return true for operations that are always inlined.
static bool isExpressionAlwaysInline(Operation *op) {
  if (isa<rtl::ConstantOp>(op) || isa<rtl::ArrayIndexOp>(op))
    return true;

  // An SV interface modport is a symbolic name that is always inlined.
  if (isa<sv::GetModportOp>(op))
    return true;

  // If this is a noop cast and the operand is always inlined, then the noop
  // cast is always inlined.
  if (isNoopCast(op))
    if (auto *operandOp = op->getOperand(0).getDefiningOp())
      return isExpressionAlwaysInline(operandOp);

  return false;
}

/// Return true if this expression should be emitted inline into any statement
/// that uses it.
static bool isExpressionEmittedInline(Operation *op) {
  // If it isn't structurally possible to inline this expression, emit it out of
  // line.
  if (isExpressionUnableToInline(op))
    return false;

  // These are always emitted inline even if multiply referenced.
  if (isExpressionAlwaysInline(op))
    return true;

  // Otherwise, if it has multiple uses, emit it out of line.
  return op->getResult(0).hasOneUse();
}

// Print out the array subscripts after a wire/port declaration.
static void printArraySubscripts(Type type, raw_ostream &os) {
  if (auto inout = type.dyn_cast<rtl::InOutType>())
    return printArraySubscripts(inout.getElementType(), os);

  if (auto array = type.dyn_cast<rtl::ArrayType>()) {
    printArraySubscripts(array.getElementType(), os);
    os << '[' << (array.getSize() - 1) << ":0]";
  } else if (auto array = type.dyn_cast<rtl::UnpackedArrayType>()) {
    printArraySubscripts(array.getElementType(), os);
    os << '[' << (array.getSize() - 1) << ":0]";
  }
}

void ModuleEmitter::collectNamesEmitDecls(Block &block) {
  // In the first pass, we fill in the symbol table, calculate the max width
  // of the declaration words and the max type width.
  size_t maxDeclNameWidth = 0, maxTypeWidth = 0;

  // Return the word (e.g. "wire") in Verilog to declare the specified thing.
  auto getVerilogDeclWord = [](Operation *op) -> StringRef {
    if (isa<sv::RegOp>(op))
      return "reg";

    // Interfaces instances use the name of the declared interface.
    if (auto interface = dyn_cast<sv::InterfaceInstanceOp>(op))
      return interface.getInterfaceType().getInterface().getValue();

    // Note: MemOp is handled as "wire" here because each of its subcomponents
    // are wires.  The corresponding 'reg' decl is handled specially below.
    return "wire";
  };

  SmallVector<Operation *, 16> declsToEmit;
  bool rtlInstanceDeclaredWires = false;
  for (auto &op : block) {
    if (auto rtlInstance = dyn_cast<rtl::InstanceOp>(op)) {
      rtlInstanceDeclaredWires |= collectNamesEmitWires(rtlInstance);
      continue;
    }
    if (op.getNumResults() == 0)
      continue;

    assert(op.getNumResults() == 1 && "rtl/sv only has single-op results");
    Value result = op.getResult(0);

    // If this is an expression emitted inline or unused, it doesn't need a
    // name.
    bool isExpr = isVerilogExpression(&op);
    if (isExpr) {
      // If this expression is dead, or can be emitted inline, ignore it.
      if (result.use_empty() || isExpressionEmittedInline(&op))
        continue;

      // Remember that this expression should be emitted out of line.
      outOfLineExpressions.insert(&op);
    }

    // Otherwise, it must be an expression or a declaration like a RegOp/WireOp.
    addName(result, op.getAttrOfType<StringAttr>("name"));

    // If we are emitting inline wire decls, don't measure or emit this wire.
    if (isExpr && emitInlineWireDecls)
      continue;

    // FIXME(verilog dialect): This can cause name collisions, because the
    // base name may be unique but the suffixed names may not be.  The right
    // way to solve this is to change the instances and mems in a new Verilog
    // dialect to use multiple return values, exposing the independent
    // Value's.

    // Determine what kind of thing this is, and how much space it needs.
    maxDeclNameWidth =
        std::max(getVerilogDeclWord(&op).size(), maxDeclNameWidth);

    auto type = result.getType();

    // Skip over SV interface types, which don't have any emitted width.
    if (!type.isa<sv::InterfaceType>()) {

      int bitWidth = getBitWidthOrSentinel(getUnderlyingArrayElementType(type));
      if (bitWidth == -1) {
        emitError(&op, getName(result))
            << " has an unsupported verilog type " << type;
        result = Value();
        break;
      }

      if (bitWidth != 1) { // Width 1 is implicit.
        // Add 5 to count the width of the "[:0] ".
        size_t thisWidth = getPrintedIntWidth(bitWidth - 1) + 5;
        maxTypeWidth = std::max(thisWidth, maxTypeWidth);
      }
    }

    // Emit this declaration.
    if (result)
      declsToEmit.push_back(&op);
  }

  SmallPtrSet<Operation *, 8> ops;

  // Okay, now that we have measured the things to emit, emit the things.
  for (auto *decl : declsToEmit) {
    ops.clear();
    ops.insert(decl);

    auto word = getVerilogDeclWord(decl);

    // Flatten the type for processing of each individual element.
    auto type = decl->getResult(0).getType();

    indent() << word;
    os.indent(maxDeclNameWidth - word.size() + 1);

    auto elementType = getUnderlyingArrayElementType(type);

    // Skip over SV interface types, which don't have any emitted width.
    bool isInterface = type.isa<sv::InterfaceType>();
    if (!isInterface)
      emitTypePaddedToWidth(elementType, maxTypeWidth, decl);

    os << getName(decl->getResult(0));

    // Interface instantiations have parentheses like a module with no ports.
    if (isInterface) {
      os << "()";
    } else {
      // Print out any array subscripts.
      printArraySubscripts(type, os);
    }

    os << ';';
    emitLocationInfoAndNewLine(ops);
  }

  if (!declsToEmit.empty() || rtlInstanceDeclaredWires)
    os << '\n';
}

bool ModuleEmitter::collectNamesEmitWires(rtl::InstanceOp instance) {
  SmallString<32> nameTmp;

  for (size_t i = 0, e = instance.getNumResults(); i < e; ++i) {
    nameTmp = instance.instanceName().str();
    nameTmp += '_';

    auto resultName = instance.getResultName(i);
    if (resultName)
      nameTmp += resultName.getValue().str();
    else
      nameTmp += std::to_string(i);

    auto result = instance.getResult(i);
    StringRef wireName = addName(result, nameTmp);

    Type resultType = result.getType();
    if (auto intType = resultType.dyn_cast<IntegerType>()) {
      if (intType.getWidth() == 1) {
        indent() << "wire " << wireName << ";\n";
      } else {
        indent() << "wire [" << intType.getWidth() - 1 << ":0] " << wireName
                 << ";\n";
      }
    } else {
      indent() << "// Type '" << resultType
               << "' not supported in verilog output yet.\n";
      instance.emitOpError(
          "Type of result not supported for verilog output. Type: ")
          << resultType;
    }
  }
  return instance.getNumResults() != 0;
}

void ModuleEmitter::emitOperation(Operation *op) {
  // Expressions may either be ignored or emitted as an expression statements.
  if (isVerilogExpression(op)) {
    if (outOfLineExpressions.count(op))
      emitStatementExpression(op);
    return;
  }

  // This visitor dispatches based on the operation kind and returns true if
  // successfully handled.
  class RTLStmtEmitter : public rtl::StmtVisitor<RTLStmtEmitter, bool> {
  public:
    RTLStmtEmitter(ModuleEmitter &emitter) : emitter(emitter) {}

    using StmtVisitor::visitStmt;
    bool visitStmt(rtl::ConnectOp op) {
      return emitter.emitStatement(op), true;
    }
    bool visitStmt(rtl::OutputOp op) { return emitter.emitStatement(op), true; }
    bool visitStmt(rtl::InstanceOp op) {
      return emitter.emitStatement(op), true;
    }
    bool visitStmt(rtl::WireOp op) { return true; }

    bool visitUnhandledStmt(Operation *op) { return false; }
    bool visitInvalidStmt(Operation *op) { return false; }

  private:
    ModuleEmitter &emitter;
  };

  if (RTLStmtEmitter(*this).dispatchStmtVisitor(op))
    return;

  if (auto merge = dyn_cast<rtl::MergeOp>(op))
    return emitStatement(merge);

  class SVEmitter : public sv::Visitor<SVEmitter, bool> {
  public:
    SVEmitter(ModuleEmitter &emitter) : emitter(emitter) {}

    using Visitor::visitSV;
    bool visitSV(sv::IfDefOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::IfOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::AlwaysOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::InitialOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::BPAssignOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::PAssignOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::AliasOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::FWriteOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::FatalOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::FinishOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::VerbatimOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::AssertOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::AssumeOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::CoverOp op) { return emitter.emitStatement(op), true; }
    bool visitSV(sv::RegOp op) { return true; }

    bool visitSV(sv::InterfaceInstanceOp op) { return true; }
    bool visitSV(sv::InterfaceSignalOp op) {
      return emitter.emitDecl(op), true;
    }
    bool visitSV(sv::InterfaceModportOp op) {
      return emitter.emitDecl(op), true;
    }

    bool visitUnhandledSV(Operation *op) { return false; }
    bool visitInvalidSV(Operation *op) { return false; }

  private:
    ModuleEmitter &emitter;
  };

  if (SVEmitter(*this).dispatchSVVisitor(op))
    return;

  emitOpError(op, "cannot emit this operation to Verilog");
  indent() << "unknown MLIR operation " << op->getName().getStringRef() << "\n";
}

/// Split the ArrayRef (which is known to be sorted) into chunks where the
/// specified elementFn matches.  The visitFn is invoked for every element.
template <typename Compare>
static void splitByPredicate(
    ModuleEmitter &emitter,
    ArrayRef<ModuleEmitter::ConditionalStatement> fullList,
    std::function<void(ArrayRef<ModuleEmitter::ConditionalStatement> elements,
                       ModuleEmitter &emitter)>
        visitFn,
    const Compare &elementFn) {
  while (!fullList.empty()) {
    auto eltValue = elementFn(fullList[0]);

    // Scan all of the elements that match this one.
    size_t endElement = 1;
    while (endElement < fullList.size() &&
           eltValue == elementFn(fullList[endElement])) {
      ++endElement;
    }

    // Visit this slice and then drop it.
    visitFn(fullList.take_front(endElement), emitter);
    fullList = fullList.drop_front(endElement);
  };
}

/// Emit a group of actions, with a fixed condition.
static void
emitActionByCond(ArrayRef<ModuleEmitter::ConditionalStatement> elements,
                 ModuleEmitter &emitter) {
  bool indentChildren = true;
  if (!elements[0].condition.empty()) {
    emitter.indent() << "if (" << elements[0].condition << ") ";
    if (elements.size() != 1) {
      emitter.os << "begin\n";
      emitter.addIndent();
    } else {
      indentChildren = false;
    }
  }

  // Emit all the actions.
  for (auto &elt : elements) {
    if (indentChildren)
      emitter.indent();

    // Emit the action one line at a time, indenting after any embedded \n's.
    StringRef actionToEmit = elt.action;
    do {
      auto lineEnd = std::min(actionToEmit.find('\n'), actionToEmit.size());
      emitter.os << actionToEmit.take_front(lineEnd);
      actionToEmit = actionToEmit.drop_front(lineEnd);

      // If we found a \n, then emit it and indent right afterward.
      if (!actionToEmit.empty()) {
        assert(actionToEmit.front() == '\n');
        emitter.os << '\n';
        emitter.indent();
        actionToEmit = actionToEmit.drop_front();
      }
    } while (!actionToEmit.empty());

    if (!elt.locInfo.empty())
      emitter.os << "\t// " << elt.locInfo;
    emitter.os << '\n';
  }

  if (!elements[0].condition.empty() && elements.size() != 1) {
    emitter.reduceIndent();
    emitter.indent() << "end\n";
  }
}

/// Emit a group of actions, guarded by conditions, with a fixed mode.
static void
emitCondActionByPPCond(ArrayRef<ModuleEmitter::ConditionalStatement> elements,
                       ModuleEmitter &emitter) {
  if (!elements[0].ppCond.empty()) {
    if (elements[0].ppCond.front() == '!')
      emitter.indent() << "`ifndef "
                       << StringRef(elements[0].ppCond).drop_front() << '\n';
    else
      emitter.indent() << "`ifdef " << elements[0].ppCond << '\n';
    emitter.addIndent();
  }

  splitByPredicate(
      emitter, elements, emitActionByCond,
      [](const ModuleEmitter::ConditionalStatement &condStmt) -> StringRef {
        return condStmt.condition;
      });

  if (!elements[0].ppCond.empty()) {
    emitter.reduceIndent();
    // Only print the macro again if there was a reasonable amount of stuff
    // being guarded.
    if (elements.size() > 1)
      emitter.indent() << "`endif // " << elements[0].ppCond << '\n';
    else
      emitter.indent() << "`endif\n";
  }
}

/// Emit a group of "always @(posedge)" conditional statements with a matching
/// clock declaration.
static void
emitPosEdgeByClock(ArrayRef<ModuleEmitter::ConditionalStatement> elements,
                   ModuleEmitter &emitter) {
  emitter.indent() << "always @(posedge " << elements[0].clock << ") begin\n";
  emitter.addIndent();
  splitByPredicate(emitter, elements, emitCondActionByPPCond,
                   [](const ModuleEmitter::ConditionalStatement &condStmt) {
                     return condStmt.ppCond;
                   });
  emitter.reduceIndent();
  emitter.indent() << "end // always @(posedge)\n";
}

/// Emit a block of conditional statements that have the same ConditionStmtKind.
static void
emitConditionStmtKind(ArrayRef<ModuleEmitter::ConditionalStatement> elements,
                      ModuleEmitter &emitter) {
  // Separate the top level blocks with a newline.
  emitter.os << '\n';

  switch (elements[0].kind) {
  case ModuleEmitter::ConditionalStmtKind::Declaration:
    splitByPredicate(emitter, elements, emitCondActionByPPCond,
                     [](const ModuleEmitter::ConditionalStatement &condStmt) {
                       return condStmt.ppCond;
                     });

    break;
  case ModuleEmitter::ConditionalStmtKind::Initial:
    emitter.indent() << "`ifndef SYNTHESIS\n";
    emitter.indent() << "initial begin\n";
    emitter.addIndent();
    assert(elements[0].clock.empty() && "initial members can't have a clock");

    splitByPredicate(emitter, elements, emitCondActionByPPCond,
                     [](const ModuleEmitter::ConditionalStatement &condStmt) {
                       return condStmt.ppCond;
                     });

    emitter.reduceIndent();
    emitter.indent() << "end // initial\n";
    emitter.indent() << "`endif // SYNTHESIS\n";
    break;
  case ModuleEmitter::ConditionalStmtKind::AlwaysAtPosEdge:
    // Split by the clock condition.
    splitByPredicate(emitter, elements, emitPosEdgeByClock,
                     [](const ModuleEmitter::ConditionalStatement &condStmt) {
                       return condStmt.clock;
                     });
    break;
  }
}

void ModuleEmitter::emitRTLExternModule(rtl::RTLExternModuleOp module) {
  os << "// external module " << module.getName() << "\n\n";
}

void ModuleEmitter::emitRTLModule(rtl::RTLModuleOp module) {
  // Add all the ports to the name table.
  SmallVector<rtl::ModulePortInfo, 8> portInfo;
  module.getPortInfo(portInfo);

  for (auto &port : portInfo) {
    StringRef name = port.getName();
    if (name.empty()) {
      module.emitOpError(
          "Found port without a name. Port names are required for "
          "Verilog synthesis.\n");
      name = "<<NO-NAME-FOUND>>";
    }
    if (port.isOutput())
      usedNames.insert(name);
    else
      addName(module.getArgument(port.argNum), name);
  }

  os << "module " << module.getName() << '(';
  if (!portInfo.empty())
    os << '\n';

  // Determine the width of the widest type we have to print so everything
  // lines up nicely.
  bool hasOutputs = false;
  unsigned maxTypeWidth = 0;
  for (auto &port : portInfo) {
    auto portType = port.type;
    hasOutputs |= port.isOutput();

    int bitWidth = getBitWidthOrSentinel(portType);
    if (bitWidth == -1 || bitWidth == 1)
      continue; // The error case is handled below.

    // Add 4 to count the width of the "[:0] ".
    unsigned thisWidth = getPrintedIntWidth(bitWidth - 1) + 5;
    maxTypeWidth = std::max(thisWidth, maxTypeWidth);
  }

  addIndent();

  for (size_t portIdx = 0, e = portInfo.size(); portIdx != e;) {
    size_t startOfLinePos = os.tell();

    indent();
    // Emit the arguments.
    auto portType = portInfo[portIdx].type;
    rtl::PortDirection thisPortDirection = portInfo[portIdx].direction;
    switch (thisPortDirection) {
    case rtl::PortDirection::OUTPUT:
      os << "output ";
      break;
    case rtl::PortDirection::INPUT:
      os << (hasOutputs ? "input  " : "input ");
      break;
    case rtl::PortDirection::INOUT:
      os << (hasOutputs ? "inout  " : "inout ");
      break;
    }

    int bitWidth = getBitWidthOrSentinel(portType);
    emitTypePaddedToWidth(portType, maxTypeWidth, module);

    // Emit the name.
    os << portInfo[portIdx].getName();
    ++portIdx;

    // If we have any more ports with the same types and the same direction,
    // emit them in a list on the same line.
    while (portIdx != e && portInfo[portIdx].direction == thisPortDirection &&
           bitWidth == getBitWidthOrSentinel(portInfo[portIdx].type)) {
      // Don't exceed our preferred line length.
      StringRef name = portInfo[portIdx].getName();
      if (os.tell() + 2 + name.size() - startOfLinePos >
          // We use "-2" here because we need a trailing comma or ); for the
          // decl.
          preferredSourceWidth - 2)
        break;

      // Append this to the running port decl.
      os << ", " << name;
      ++portIdx;
    }

    if (portIdx != e)
      os << ',';
    else
      os << ");\n";
    os << '\n';
  }

  if (portInfo.empty())
    os << ");\n";

  // Build up the symbol table for all of the values that need names in the
  // module.
  collectNamesEmitDecls(*module.getBodyBlock());

  // Emit the body.
  for (auto &op : *module.getBodyBlock()) {
    emitOperation(&op);
  }

  // Emit the conditional statements at the bottom.  Start by sorting the list
  // to group by kind.
  std::stable_sort(conditionalStmts.begin(), conditionalStmts.end());

  // Emit conditional statements by groups.
  splitByPredicate(
      *this, conditionalStmts, emitConditionStmtKind,
      [](const ModuleEmitter::ConditionalStatement &condStmt)
          -> ModuleEmitter::ConditionalStmtKind { return condStmt.kind; });
  reduceIndent();

  os << "endmodule\n\n";
}

//===----------------------------------------------------------------------===//
// MLIRModuleEmitter
//===----------------------------------------------------------------------===//

namespace {
class MLIRModuleEmitter : public VerilogEmitterBase {
public:
  explicit MLIRModuleEmitter(VerilogEmitterState &state)
      : VerilogEmitterBase(state) {}

  void emit(ModuleOp module);
};

} // end anonymous namespace

void MLIRModuleEmitter::emit(ModuleOp module) {
  for (auto &op : *module.getBody()) {
    if (auto module = dyn_cast<rtl::RTLModuleOp>(op))
      ModuleEmitter(state).emitRTLModule(module);
    else if (auto module = dyn_cast<rtl::RTLExternModuleOp>(op))
      ModuleEmitter(state).emitRTLExternModule(module);
    else if (auto interface = dyn_cast<sv::InterfaceOp>(op))
      ModuleEmitter(state).emitDecl(interface);
    else if (auto verbatim = dyn_cast<sv::VerbatimOp>(op))
      ModuleEmitter(state).emitStatement(verbatim);
    else if (auto verbatim = dyn_cast<sv::IfDefOp>(op))
      ModuleEmitter(state).emitStatement(verbatim);
    else if (!isa<ModuleTerminatorOp>(op))
      op.emitError("unknown operation");
  }
}

LogicalResult circt::exportVerilog(ModuleOp module, llvm::raw_ostream &os) {
  VerilogEmitterState state(os);
  MLIRModuleEmitter(state).emit(module);
  return failure(state.encounteredError);
}

void circt::registerToVerilogTranslation() {
  TranslateFromMLIRRegistration toVerilog(
      "emit-verilog", exportVerilog, [](DialectRegistry &registry) {
        registry.insert<rtl::RTLDialect, sv::SVDialect>();
      });
}
