//===- ExportVerilogFIRRTL.cpp - Verilog Emitter for FIRRTL ---------------===//
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
// clang-format don't reorder #includes!
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace firrtl;
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
  // All FIRRTL expressions and RTL combinatorial logic ops are Verilog
  // expressions.
  return isExpression(op) || isa<AsPassivePrimOp>(op) ||
         isa<AsNonPassivePrimOp>(op);
}

/// Return the width of the specified FIRRTL type in bits or -1 if it isn't
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
      .Case<ClockType, ResetType, AsyncResetType>([](Type) { return 1; })
      .Case<SIntType, UIntType>([](IntType intType) {
        // Turn zero-bit values into single bit ones for simplicity.  This
        // occurs in the addr lines of mems with depth=1.
        // FIXME: This actually isn't correct at all!
        auto result = intType.getWidthOrSentinel();
        return result ? result : 1;
      })
      .Case<AnalogType>([](AnalogType analogType) {
        auto result = analogType.getWidthOrSentinel();
        return result ? result : 1;
      })
      .Case<FlipType>([](FlipType flipType) {
        return getBitWidthOrSentinel(flipType.getElementType());
      })
      .Default([](Type) { return -1; });
}

/// Return the type of the specified value, force casting to the subtype.
template <typename T = FIRRTLType>
static T getTypeOf(Value v) {
  return v.getType().cast<T>();
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
  if (isa<AsAsyncResetPrimOp>(op) || isa<AsClockPrimOp>(op) ||
      isa<AsUIntPrimOp>(op) || isa<AsSIntPrimOp>(op) ||
      isa<AsPassivePrimOp>(op) || isa<AsNonPassivePrimOp>(op))
    return true;

  // cvt from signed is noop.
  if (isa<CvtPrimOp>(op) &&
      op->getOperand(0).getType().cast<IntType>().isSigned())
    return true;

  // Shift left by zero is a noop.
  if (auto shl = dyn_cast<ShlPrimOp>(op))
    if (shl.amount() == 0)
      return true;

  return false;
}

namespace {
/// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  /// This is the underlying ground type of the field.
  Type type;
  /// This is a suffix to add to the field name to make it unique.
  std::string suffix;
  /// This indicates whether the field was flipped to be an output.
  bool isOutput;
};
} // end anonymous namespace

/// Convert a nested bundle of fields into a flat list of fields.  This is used
/// when working with instances and mems to flatten them.
static void flattenBundleTypes(Type type, StringRef suffixSoFar, bool isFlipped,
                               SmallVectorImpl<FlatBundleFieldEntry> &results) {
  if (auto flip = type.dyn_cast<FlipType>())
    return flattenBundleTypes(flip.getElementType(), suffixSoFar, !isFlipped,
                              results);

  // In the base case we record this field.
  auto bundle = type.dyn_cast<BundleType>();
  if (!bundle) {
    results.push_back({type, suffixSoFar.str(), isFlipped});
    return;
  }

  SmallString<16> tmpSuffix(suffixSoFar);

  // Otherwise, we have a bundle type.  Break it down.
  for (auto &elt : bundle.getElements()) {
    // Construct the suffix to pass down.
    tmpSuffix.resize(suffixSoFar.size());
    tmpSuffix.push_back('_');
    tmpSuffix.append(elt.name.strref());
    // Recursively process subelements.
    flattenBundleTypes(elt.type, tmpSuffix, isFlipped, results);
  }
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

  void emitFModule(FModuleOp module);
  void emitExpression(Value exp, SmallPtrSet<Operation *, 8> &emittedExprs,
                      bool forceRootExpr = false);

  /// Emit the specified expression and return it as a string.
  std::string
  emitExpressionToString(Value exp, SmallPtrSet<Operation *, 8> &emittedExprs,
                         VerilogPrecedence precedence = LowestPrecedence);

  // Statements.
  void emitStatementExpression(Operation *op);
  void emitStatement(AttachOp op);
  void emitStatement(ConnectOp op);
  void emitStatement(PartialConnectOp op);
  void emitStatement(PrintFOp op);
  void emitStatement(StopOp op);
  void emitDecl(NodeOp op);
  void emitDecl(InstanceOp op);
  void emitDecl(RegOp op);
  void emitDecl(MemOp op);
  void emitDecl(RegResetOp op);
  void emitOperation(Operation *op);

  void collectNamesEmitDecls(Block &block);
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

/// Return the verilog signedness of the specified type.
static SubExprSignedness getSignednessOf(Type type) {
  return TypeSwitch<Type, SubExprSignedness>(type)
      .Case<IntegerType>([](IntegerType integerType) {
        return integerType.isSigned() ? IsSigned : IsUnsigned;
      })
      .Case<FlipType>([](FlipType flipType) {
        return getSignednessOf(flipType.getElementType());
      })
      .Case<ClockType, ResetType, AsyncResetType>(
          [](Type) { return IsUnsigned; })
      .Case<SIntType>([](Type) { return IsSigned; })
      .Case<UIntType>([](Type) { return IsUnsigned; })
      .Default([](Type) {
        assert(0 && "unsupported type");
        return IsUnsigned;
      });
}

namespace {
/// This builds a recursively nested expression from an SSA use-def graph.  This
/// uses a post-order walk, but it needs to obey precedence and signedness
/// constraints that depend on the behavior of the child nodes.  To handle this,
/// we emit the characters to a SmallVector which allows us to emit a bunch of
/// stuff, then pre-insert parentheses and other things if we find out that it
/// was needed later.
class ExprEmitter : public ExprVisitor<ExprEmitter, SubExprInfo> {
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
  friend class ExprVisitor<ExprEmitter, SubExprInfo>;

  /// Emit the specified value as a subexpression to the stream.
  SubExprInfo emitSubExpr(Value exp, VerilogPrecedence parenthesizeIfLooserThan,
                          bool forceExpectedSign = false,
                          bool opForceSign = false);

  SubExprInfo visitUnhandledExpr(Operation *op);
  SubExprInfo visitInvalidExpr(Operation *op) { return visitUnhandledExpr(op); }

  using ExprVisitor::visitExpr;
  SubExprInfo visitExpr(firrtl::ConstantOp op);

  /// Emit a verilog concatenation of the specified values.  If the before or
  /// after strings are specified, they are included as prefix/postfix elements
  /// of the concatenation, respectively.
  SubExprInfo emitCat(ArrayRef<Value> values, StringRef before = StringRef(),
                      StringRef after = StringRef());

  /// Emit a verilog bit selection operation like x[4:0], the bit numbers are
  /// inclusive like verilog.
  SubExprInfo emitBitSelect(Value operand, unsigned hiBit, unsigned loBit);

  SubExprInfo emitBinary(Operation *op, VerilogPrecedence prec,
                         const char *syntax, bool opSigned = false);

  SubExprInfo emitFIRRTLBinary(Operation *op, VerilogPrecedence prec,
                               const char *syntax, bool skipCast = false);

  SubExprInfo emitVariadic(Operation *op, VerilogPrecedence prec,
                           const char *syntax);

  SubExprInfo emitFIRRTLVariadic(Operation *op, VerilogPrecedence prec,
                                 const char *syntax, bool skipCast = false);

  SubExprInfo emitUnary(Operation *op, const char *syntax,
                        bool forceUnsigned = false);

  SubExprInfo emitFIRRTLUnary(Operation *op, const char *syntax,
                              bool skipCast = false);

  SubExprInfo emitNoopCast(Operation *op) {
    return emitSubExpr(op->getOperand(0), LowestPrecedence);
  }

  SubExprInfo visitExpr(AddPrimOp op) {
    return emitFIRRTLVariadic(op, Addition, "+");
  }
  SubExprInfo visitExpr(SubPrimOp op) {
    return emitFIRRTLBinary(op, Addition, "-");
  }
  SubExprInfo visitExpr(MulPrimOp op) {
    return emitFIRRTLVariadic(op, Multiply, "*");
  }
  SubExprInfo visitExpr(DivPrimOp op) {
    return emitFIRRTLBinary(op, Multiply, "/");
  }
  SubExprInfo visitExpr(RemPrimOp op) {
    return emitFIRRTLBinary(op, Multiply, "%");
  }

  SubExprInfo visitExpr(AndPrimOp op) {
    return emitFIRRTLVariadic(op, And, "&");
  }
  SubExprInfo visitExpr(OrPrimOp op) {
    return emitFIRRTLVariadic(op, Or, "|", true);
  }
  SubExprInfo visitExpr(XorPrimOp op) {
    return emitFIRRTLVariadic(op, Xor, "^");
  }

  // FIRRTL Comparison Operations
  SubExprInfo visitExpr(LEQPrimOp op) {
    return emitFIRRTLBinary(op, Comparison, "<=");
  }
  SubExprInfo visitExpr(LTPrimOp op) {
    return emitFIRRTLBinary(op, Comparison, "<");
  }
  SubExprInfo visitExpr(GEQPrimOp op) {
    return emitFIRRTLBinary(op, Comparison, ">=");
  }
  SubExprInfo visitExpr(GTPrimOp op) {
    return emitFIRRTLBinary(op, Comparison, ">");
  }
  SubExprInfo visitExpr(EQPrimOp op) {
    return emitFIRRTLBinary(op, Equality, "==", true);
  }
  SubExprInfo visitExpr(NEQPrimOp op) {
    return emitFIRRTLBinary(op, Equality, "!=", true);
  }
  SubExprInfo visitExpr(DShlPrimOp op) {
    return emitFIRRTLBinary(op, Shift, "<<");
  }
  SubExprInfo visitExpr(DShlwPrimOp op) {
    return emitFIRRTLBinary(op, Shift, "<<");
  }
  SubExprInfo visitExpr(DShrPrimOp op) {
    return emitFIRRTLBinary(op, Shift, ">>>");
  }

  // Unary Prefix operators.
  SubExprInfo visitExpr(AndRPrimOp op) {
    return emitFIRRTLUnary(op, "&", true);
  }
  SubExprInfo visitExpr(XorRPrimOp op) {
    return emitFIRRTLUnary(op, "^", true);
  }
  SubExprInfo visitExpr(OrRPrimOp op) { return emitFIRRTLUnary(op, "|", true); }
  SubExprInfo visitExpr(NotPrimOp op) { return emitFIRRTLUnary(op, "~", true); }
  SubExprInfo visitExpr(NegPrimOp op) { return emitFIRRTLUnary(op, "-"); }

  // Noop cast operators.
  SubExprInfo visitExpr(AsAsyncResetPrimOp op) { return emitNoopCast(op); }
  SubExprInfo visitExpr(AsClockPrimOp op) { return emitNoopCast(op); }

  // Signedness tracks the verilog sign, not the FIRRTL sign, so we don't need
  // to emit anything for AsSInt/AsUInt.  Their results will get casted by the
  // client as necessary.
  SubExprInfo visitExpr(AsUIntPrimOp op) { return emitNoopCast(op); }
  SubExprInfo visitExpr(AsSIntPrimOp op) { return emitNoopCast(op); }
  SubExprInfo visitExpr(AsPassivePrimOp op) { return emitNoopCast(op); }
  SubExprInfo visitExpr(AsNonPassivePrimOp op) { return emitNoopCast(op); }

  // Other
  SubExprInfo visitExpr(SubfieldOp op);
  SubExprInfo visitExpr(ValidIfPrimOp op);
  SubExprInfo visitExpr(MuxPrimOp op);
  SubExprInfo visitExpr(CatPrimOp op) { return emitCat({op.lhs(), op.rhs()}); }
  SubExprInfo visitExpr(CvtPrimOp op);
  SubExprInfo visitExpr(BitsPrimOp op) {
    return emitBitSelect(op.getOperand(), op.hi(), op.lo());
  }
  SubExprInfo visitExpr(InvalidValuePrimOp op);
  SubExprInfo visitExpr(HeadPrimOp op);
  SubExprInfo visitExpr(TailPrimOp op);
  SubExprInfo visitExpr(PadPrimOp op);
  SubExprInfo visitExpr(ShlPrimOp op) { // shl(x, 4) ==> {x, 4'h0}
    auto shAmt = op.amount();
    if (shAmt)
      return emitCat(op.getOperand(), "", llvm::utostr(shAmt) + "'h0");
    return emitNoopCast(op);
  }
  SubExprInfo visitExpr(ShrPrimOp op);

  // Conversion to/from standard integer types is a noop.
  SubExprInfo visitExpr(StdIntCastOp op) { return emitNoopCast(op); }

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

SubExprInfo ExprEmitter::emitFIRRTLBinary(Operation *op, VerilogPrecedence prec,
                                          const char *syntax, bool skipCast) {
  auto lhsInfo = emitSubExpr(op->getOperand(0), prec, !skipCast);
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

  auto rhsInfo = emitSubExpr(op->getOperand(1), rhsPrec, !skipCast);

  // If we have a strict sign, then match the firrtl operation sign.
  // Otherwise, the result is signed if both operands are signed.
  SubExprSignedness signedness;
  if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
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

SubExprInfo ExprEmitter::emitFIRRTLVariadic(Operation *op,
                                            VerilogPrecedence prec,
                                            const char *syntax, bool skipCast) {
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](Value v1) { emitSubExpr(v1, prec, !skipCast); },
      [&] { os << ' ' << syntax << ' '; });

  return {prec, IsUnsigned};
}

SubExprInfo ExprEmitter::emitUnary(Operation *op, const char *syntax,
                                   bool forceUnsigned) {
  os << syntax;
  auto signedness = emitSubExpr(op->getOperand(0), Unary).signedness;
  return {Unary, forceUnsigned ? IsUnsigned : signedness};
}

SubExprInfo ExprEmitter::emitFIRRTLUnary(Operation *op, const char *syntax,
                                         bool skipCast) {
  os << syntax;
  emitSubExpr(op->getOperand(0), Unary, !skipCast);
  return {Unary, getSignednessOf(op->getResult(0).getType())};
}

/// Emit the specified value as a subexpression to the stream.
SubExprInfo ExprEmitter::emitSubExpr(Value exp,
                                     VerilogPrecedence parenthesizeIfLooserThan,
                                     bool forceExpectedSign, bool opForceSign) {
  auto *op = exp.getDefiningOp();
  bool shouldEmitInlineExpr = op && isVerilogExpression(op);

  // Don't emit this expression inline if it has multiple uses.
  if (shouldEmitInlineExpr && parenthesizeIfLooserThan != ForceEmitMultiUse &&
      emitter.outOfLineExpressions.count(op))
    shouldEmitInlineExpr = false;

  // If this is a non-expr or shouldn't be done inline, just refer to its
  // name.
  if (!shouldEmitInlineExpr) {
    if ((forceExpectedSign && getSignednessOf(exp.getType()) != IsUnsigned) ||
        opForceSign) {
      os << "$signed(" << emitter.getName(exp) << ')';
      return {Unary, IsSigned};
    }
    os << emitter.getName(exp);
    return {Symbol, IsUnsigned};
  }

  unsigned subExprStartIndex = resultBuffer.size();

  // Okay, this is an expression we should emit inline.  Do this through our
  // visitor.
  auto expInfo = dispatchExprVisitor(exp.getDefiningOp());

  // Check cases where we have to insert things before the expression now that
  // we know things about it.
  if (forceExpectedSign &&
      getSignednessOf(exp.getType()) != expInfo.signedness) {
    // If the sign of the result matters and we emitted something with the
    // wrong sign, correct it.
    StringRef cast = expInfo.signedness == IsSigned ? "$unsigned(" : "$signed(";
    resultBuffer.insert(resultBuffer.begin() + subExprStartIndex, cast.begin(),
                        cast.end());
    os << ')';

  } else if (expInfo.precedence > parenthesizeIfLooserThan) {
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

static void collectCatValues(Value operand, SmallVectorImpl<Value> &catValues,
                             ExprEmitter &emitter) {
  operand = emitter.lookThroughNoopCasts(operand);

  // If the operand is a single-use cat, flatten it into the vector we're
  // building.
  if (auto *op = operand.getDefiningOp())
    if (auto cat = dyn_cast<CatPrimOp>(op))
      if (!emitter.emitter.outOfLineExpressions.count(cat)) {
        collectCatValues(cat.lhs(), catValues, emitter);
        collectCatValues(cat.rhs(), catValues, emitter);
        return;
      }
  catValues.push_back(operand);
}

/// Emit a verilog concatenation of the specified values.  If the before or
/// after strings are specified, they are included as prefix/postfix elements
/// of the concatenation, respectively.
SubExprInfo ExprEmitter::emitCat(ArrayRef<Value> values, StringRef before,
                                 StringRef after) {
  os << '{';
  if (!before.empty()) {
    os << before;
    if (!values.empty() || !after.empty())
      os << ", ";
  }

  // Recursively flatten any nested cat-like expressions.
  SmallVector<Value, 8> elements;
  for (auto v : values)
    collectCatValues(v, elements, *this);

  llvm::interleaveComma(elements, os,
                        [&](Value v) { emitSubExpr(v, LowestPrecedence); });

  if (!after.empty()) {
    if (!values.empty())
      os << ", ";
    os << after;
  }
  os << '}';
  return {Unary, IsUnsigned};
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

SubExprInfo ExprEmitter::visitExpr(firrtl::ConstantOp op) {
  auto resType = op.getType().cast<IntType>();
  if (resType.getWidthOrSentinel() == -1)
    return visitUnhandledExpr(op);

  os << resType.getWidth() << '\'';
  if (resType.isSigned())
    os << 's';
  os << 'h';

  SmallString<32> valueStr;
  op.value().toStringUnsigned(valueStr, 16);
  os << valueStr;
  return {Unary, resType.isSigned() ? IsSigned : IsUnsigned};
}

SubExprInfo ExprEmitter::visitExpr(SubfieldOp op) {
  auto prec = emitSubExpr(op.getOperand(), Unary);
  // TODO(verilog dialect): This is a hack, relying on the fact that only
  // textual expressions that produce a name can appear here.
  assert(prec.precedence == Symbol);
  os << '_' << op.fieldname();
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitExpr(ValidIfPrimOp op) {
  // It isn't clear to me why it it is ok to ignore the binding condition,
  // but this is what the existing FIRRTL verilog emitter does.
  return emitSubExpr(op.rhs(), LowestPrecedence);
}

SubExprInfo ExprEmitter::visitExpr(MuxPrimOp op) {
  // The ?: operator is right associative.
  emitSubExpr(op.sel(), VerilogPrecedence(Conditional - 1));
  os << " ? ";
  auto lhsInfo = emitSubExpr(op.high(), VerilogPrecedence(Conditional - 1));
  os << " : ";
  auto rhsInfo = emitSubExpr(op.low(), Conditional);

  SubExprSignedness signedness = IsUnsigned;
  if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
    signedness = IsSigned;

  return {Conditional, signedness};
}

SubExprInfo ExprEmitter::visitExpr(CvtPrimOp op) {
  if (getTypeOf<IntType>(op.getOperand()).isSigned())
    return emitNoopCast(op);

  return emitCat(op.getOperand(), "1'b0");
}

SubExprInfo ExprEmitter::visitExpr(InvalidValuePrimOp op) {
  os << "'0";
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitExpr(HeadPrimOp op) {
  auto width = getTypeOf<IntType>(op.getOperand()).getWidthOrSentinel();
  if (width == -1)
    return visitUnhandledExpr(op);
  auto numBits = op.amount();
  return emitBitSelect(op.getOperand(), width - 1, width - numBits);
}

SubExprInfo ExprEmitter::visitExpr(TailPrimOp op) {
  auto width = getTypeOf<IntType>(op.getOperand()).getWidthOrSentinel();
  if (width == -1)
    return visitUnhandledExpr(op);
  auto numBits = op.amount();
  return emitBitSelect(op.getOperand(), width - 1 - numBits, 0);
}

SubExprInfo ExprEmitter::visitExpr(PadPrimOp op) {
  auto inType = getTypeOf<IntType>(op.getOperand());
  auto inWidth = inType.getWidthOrSentinel();
  if (inWidth == -1)
    return visitUnhandledExpr(op);
  auto destWidth = op.amount();

  // If the destination width is smaller than the input width, then this is a
  // truncation.
  if (destWidth == unsigned(inWidth))
    return emitNoopCast(op);
  if (destWidth < unsigned(inWidth))
    return emitBitSelect(op.getOperand(), destWidth - 1, 0);

  // If this is unsigned, it is a zero extension.
  if (!inType.isSigned()) {
    os << "{{" << (destWidth - inWidth) << "'d0}, ";
    emitSubExpr(op.getOperand(), LowestPrecedence);
    os << "}";
    return {Unary, IsUnsigned};
  }

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

// TODO(verilog dialect): There is no need to persist shifts. They are
// apparently only needed for width inference.
SubExprInfo ExprEmitter::visitExpr(ShrPrimOp op) {
  auto width = getTypeOf<IntType>(op.getOperand()).getWidthOrSentinel();
  unsigned shiftAmount = op.amount();
  if (width == -1 || shiftAmount >= unsigned(width))
    return visitUnhandledExpr(op);

  return emitBitSelect(op.getOperand(), width - 1, shiftAmount);
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

void ModuleEmitter::emitStatement(AttachOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Don't emit anything for a zero or one operand attach.
  if (op.operands().size() < 2)
    return;

  auto locStr = getLocationInfoAsString(ops);

  // If we are simulating, use an 'alias':
  std::string action = "alias ";
  llvm::interleave(
      op.operands(),
      [&](Value operand) { action += emitExpressionToString(operand, ops); },
      [&] { action += " = "; });
  addDeclaration(action + ";", locStr, "!SYNTHESIS");

  // Otherwise, emit an N^2 number of attaches both directions.
  for (auto v1 : op.operands()) {
    for (auto v2 : op.operands())
      if (v1 != v2) {
        action = "assign " + emitExpressionToString(v1, ops) + " = " +
                 emitExpressionToString(v2, ops) + ";";
        addDeclaration(action, locStr, "SYNTHESIS");
      }
  }
}

void ModuleEmitter::emitStatement(ConnectOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Connect to a register has "special" behavior.
  auto dest = op.dest();
  auto addRegAssign = [&](const std::string &clockExpr, Value value) {
    std::string action =
        getName(dest).str() + " <= " + emitExpressionToString(value, ops) + ";";
    auto locStr = getLocationInfoAsString(ops);
    addAtPosEdge(action, locStr, clockExpr);
    return;
  };

  if (auto regOp = dyn_cast_or_null<RegOp>(dest.getDefiningOp())) {
    auto clockExpr = emitExpressionToString(regOp.clockVal(), ops);
    addRegAssign(clockExpr, op.src());
    return;
  }

  if (auto regInitOp = dyn_cast_or_null<RegResetOp>(dest.getDefiningOp())) {
    auto clockExpr = emitExpressionToString(regInitOp.clockVal(), ops);
    clockExpr +=
        " or posedge " + emitExpressionToString(regInitOp.resetSignal(), ops);

    addRegAssign(clockExpr, op.src());
    return;
  }

  indent() << "assign ";
  emitExpression(dest, ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(PartialConnectOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Connect to a register has "special" behavior.
  auto dest = op.dest();
  auto addRegAssign = [&](const std::string &clockExpr, Value value) {
    std::string action =
        getName(dest).str() + " <= " + emitExpressionToString(value, ops) + ";";
    auto locStr = getLocationInfoAsString(ops);
    addAtPosEdge(action, locStr, clockExpr);
    return;
  };

  if (auto regOp = dyn_cast_or_null<RegOp>(dest.getDefiningOp())) {
    auto clockExpr = emitExpressionToString(regOp.clockVal(), ops);
    addRegAssign(clockExpr, op.src());
    return;
  }

  if (auto regInitOp = dyn_cast_or_null<RegResetOp>(dest.getDefiningOp())) {
    auto clockExpr = emitExpressionToString(regInitOp.clockVal(), ops);
    clockExpr +=
        " or posedge " + emitExpressionToString(regInitOp.resetSignal(), ops);

    addRegAssign(clockExpr, op.src());
    return;
  }

  indent() << "assign ";
  emitExpression(dest, ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitStatement(PrintFOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // TODO(verilog dialect): this is a simulation only construct, we should have
  // synthesis specific nodes, e.g. an 'if' statement, always @(posedge) blocks,
  // etc.
  auto clockExpr = emitExpressionToString(op.clock(), ops);
  auto condExpr = "`PRINTF_COND_ && " +
                  emitExpressionToString(op.cond(), ops, AndShortCircuit);

  std::string actionStr;
  llvm::raw_string_ostream action(actionStr);

  action << "$fwrite(32'h80000002, \"";
  action.write_escaped(op.formatString()) << '"';
  for (auto operand : op.operands()) {
    action << ", " << emitExpressionToString(operand, ops);
  }
  action << ");";

  auto locInfo = getLocationInfoAsString(ops);
  addAtPosEdge(action.str(), locInfo, clockExpr, "!SYNTHESIS", condExpr);
}

void ModuleEmitter::emitStatement(StopOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // TODO(verilog dialect): this is a simulation only construct, we should have
  // synthesis specific nodes, e.g. an 'if' statement, always @(posedge) blocks,
  // etc.
  auto clockExpr = emitExpressionToString(op.clock(), ops);
  auto condExpr = "`STOP_COND_ && " +
                  emitExpressionToString(op.cond(), ops, AndShortCircuit);

  const char *action = op.exitCode() ? "$fatal;" : "$finish;";
  auto locInfo = getLocationInfoAsString(ops);

  addAtPosEdge(action, locInfo, clockExpr, "!SYNTHESIS", condExpr);
}

void ModuleEmitter::emitDecl(NodeOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "assign " << getName(op.getResult());
  os << " = ";
  emitExpression(op.input(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
}

void ModuleEmitter::emitDecl(InstanceOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto *referencedModule = op.getReferencedModule();
  FExtModuleOp referencedExtModule;

  // If this is referencing an extmodule with a specified defname, then use
  // the defName from it as the actual module name we reference.  This exists
  // because FIRRTL is not parameterized like verilog is - it introduces
  // redundant extmodule instances to encode different parameter
  // configurations.
  StringRef defName = op.moduleName();
  if (!referencedModule)
    emitOpError(op, "could not find mlir node named @" + defName);
  else if ((referencedExtModule = dyn_cast<FExtModuleOp>(referencedModule)))
    if (auto defNameAttr = referencedExtModule.defname())
      defName = defNameAttr.getValue();
  indent() << defName;

  // Print a parameter constant value in a Verilog compatible way.
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
  if (referencedExtModule)
    if (auto paramDictOpt = referencedExtModule.parameters()) {
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

  auto instanceName = op.name();
  os << ' ' << instanceName << " (";
  emitLocationInfoAndNewLine(ops);

  for (size_t resultIdx = 0, e = op.getNumResults(); resultIdx != e;
       ++resultIdx) {
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    flattenBundleTypes(op.getResult(resultIdx).getType().cast<FIRRTLType>(),
                       op.getPortNameStr(resultIdx), false, fieldTypes);
    for (auto &elt : fieldTypes) {
      bool isLast = resultIdx == e - 1 && &elt == &fieldTypes.back();
      indent() << "  ." << StringRef(elt.suffix) << '(' << instanceName << '_'
               << elt.suffix << (isLast ? ")\n" : "),\n");
    }
  }
  indent() << ");\n";
}

void ModuleEmitter::emitDecl(RegOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // There is nothing to do for the register decl itself: the prepass at the
  // module level will already handle emission of the corresponding reg
  // declaration, and any assign to it will be handled at that time.  Here we
  // are concerned with the emission of the initializer expression for
  // simulation.
  emitRandomizeProlog();

  // At simulator load time, the register is set to random nothing.
  std::string action = getName(op.getResult()).str() + " = `RANDOM;";
  auto locInfo = getLocationInfoAsString(ops);
  addInitial(action, locInfo, /*ppCond*/ "RANDOMIZE_REG_INIT");
}

void ModuleEmitter::emitDecl(RegResetOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Registers with an init expression get a random init for the case where they
  // are not initialized.
  auto resetSignal = emitExpressionToString(op.resetSignal(), ops);
  auto resetValue = emitExpressionToString(op.resetValue(), ops);
  auto locInfo = getLocationInfoAsString(ops);

  emitRandomizeProlog();
  std::string action = getName(op.getResult()).str() + " = `RANDOM;";
  addInitial(action, locInfo, /*ppCond*/ "RANDOMIZE_REG_INIT",
             /*cond*/ "~" + resetSignal);

  action = getName(op.getResult()).str() + " = " + resetValue + ";";
  addInitial(action, locInfo, /*ppCond*/ "", /*cond*/ resetSignal);
}

void ModuleEmitter::emitDecl(MemOp op) {
  // Check that the MemOp has been properly lowered before this.
  if (op.readLatency() != 0 || op.writeLatency() != 1) {
    // FIXME: This should be an error.
    op.emitWarning("FIXME: need to support mem read/write latency correctly");
  }
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto memName = op.name().getValue().str();
  uint64_t depth = op.depth();
  auto locInfo = getLocationInfoAsString(ops);

  // If we haven't already emitted a declaration of initvar, do so.
  if (!emittedInitVar && depth > 1) {
    assert(depth < (1ULL << 31) &&
           "FIXME: This doesn't support mems greater than 2^32");
    addInitial("integer initvar;", "", /*ppCond*/ "RANDOMIZE_MEM_INIT",
               /*cond*/ "", /*partialOrder: initVar decl*/ 10);
    emittedInitVar = true;
  }

  // Aggregate mems may declare multiple reg's.  We need to random initialize
  // them all.
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  if (auto dataType = op.getDataTypeOrNull())
    flattenBundleTypes(dataType, "", false, fieldTypes);

  emitRandomizeProlog();

  // On initialization, fill the mem with random bits if RANDOMIZE_MEM_INIT
  // is set.
  if (depth == 1) { // Don't emit a for loop for one element.
    for (const auto &elt : fieldTypes) {
      std::string action = memName + elt.suffix + "[0] = `RANDOM;";
      addInitial(action, locInfo, /*ppCond*/ "RANDOMIZE_MEM_INIT",
                 /*cond*/ "", /*partialOrder: After initVar decl*/ 11);
    }

  } else if (!fieldTypes.empty()) {
    auto initvar = memName + "_initvar";
    std::string action = "for (" + initvar + " = 0; " + initvar + " < " +
                         llvm::utostr(depth) + "; " + initvar + " = " +
                         initvar + "+1)";

    if (fieldTypes.size() > 1)
      action += " begin";

    for (const auto &elt : fieldTypes)
      action += "\n  " + memName + elt.suffix + "[" + initvar + "] = `RANDOM;";

    if (fieldTypes.size() > 1)
      action += "\nend";
    addInitial(action, locInfo, /*ppCond*/ "RANDOMIZE_MEM_INIT",
               /*cond*/ "", /*partialOrder: After initVar decl*/ 11);
  }

  // Keep track of whether this mem is an even power of two or not.
  bool isPowerOfTwo = llvm::isPowerOf2_64(depth);

  // Iterate over all of the read/write ports, emitting logic necessary to use
  // them.
  SmallVector<std::pair<Identifier, MemOp::PortKind>, 2> ports;
  op.getPorts(ports);

  for (auto &port : ports) {
    auto portName = port.first;

    // Return the identifier emitted to Verilog that refers to the specified
    // field of the specified read/write port.  This corresponds to the naming
    // of subfield's.
    // TODO(verilog dialect): eliminate subfields.
    auto getPortName = [&](StringRef fieldName) -> std::string {
      return memName + "_" + portName.str() + "_" + fieldName.str();
    };

    switch (port.second) {
    case MemOp::PortKind::ReadWrite:
      op.emitOpError("readwrite ports should be lowered into separate read and "
                     "write ports by previous passes");
      continue;

    case MemOp::PortKind::Read:
      // Emit an assign to the read port, using the address.
      // TODO(firrtl-spec): It appears that the clock signal on the read port is
      // ignored, why does it exist?
      if (!isPowerOfTwo) {
        indent() << "`ifndef RANDOMIZE_GARBAGE_ASSIGN\n";
        addIndent();
      }

      for (const auto &elt : fieldTypes) {
        indent() << "assign " << getPortName("data") << elt.suffix << " = "
                 << memName << elt.suffix << '[' << getPortName("addr") << "];";
        emitLocationInfoAndNewLine(ops);
      }

      if (!isPowerOfTwo) {
        reduceIndent();
        indent() << "`else\n";
        addIndent();

        for (const auto &elt : fieldTypes) {
          indent() << "assign " << getPortName("data") << elt.suffix << " = ";
          os << getPortName("addr") << " < " << depth << " ? ";
          os << memName << elt.suffix << '[' << getPortName("addr") << "]";
          os << " : `RANDOM;";
          emitLocationInfoAndNewLine(ops);
        }

        reduceIndent();
        indent() << "`endif // RANDOMIZE_GARBAGE_ASSIGN\n";
      }

      break;

    case MemOp::PortKind::Write:
      auto clockExpr = getPortName("clk");
      auto locStr = getLocationInfoAsString(ops);

      // Compute the condition, which is an 'and' of the enable signal and
      // the mask predicate.
      // TODO(QoI, verilog dialect): The mask is frequently (always?) one --
      // we don't need to print it if so.
      auto condition = getPortName("en") + " & " + getPortName("mask");

      for (const auto &elt : fieldTypes) {
        std::string action = memName + elt.suffix + '[' + getPortName("addr") +
                             "] <= " + getPortName("data") + elt.suffix + ";";

        addAtPosEdge(action, locStr, clockExpr,
                     /*ppCond:*/ {}, /*condition*/ condition);
      }
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Module Driver
//===----------------------------------------------------------------------===//

/// Most expressions are invalid to bit-select from in Verilog, but some things
/// are ok.  Return true if it is ok to inline bitselect from the result of this
/// expression.  It is conservatively correct to return false.
static bool isOkToBitSelectFrom(Value v) {
  // Module ports are always ok to bit select from.
  auto *op = v.getDefiningOp();
  if (!op)
    return true;

  if (isa<SubfieldOp>(op))
    return true;

  // As{Non}PassivePrimOp is transparent.
  if (auto cast = dyn_cast<AsPassivePrimOp>(op))
    return isOkToBitSelectFrom(cast.getOperand());
  if (auto cast = dyn_cast<AsNonPassivePrimOp>(op))
    return isOkToBitSelectFrom(cast.getOperand());

  // TODO: We could handle concat and other operators here.
  return false;
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
    if (isa<HeadPrimOp>(user) || isa<TailPrimOp>(user) ||
        isa<ShrPrimOp>(user) || isa<BitsPrimOp>(user))
      if (!isOkToBitSelectFrom(op->getResult(0)))
        return true;

    // Signed pad is emits a bit extract since it is a sign extend.
    if (auto pad = dyn_cast<PadPrimOp>(user)) {
      auto inType = getTypeOf<IntType>(pad.getOperand());
      auto inWidth = inType.getWidthOrSentinel();
      if (unsigned(inWidth) > pad.amount() &&
          !isOkToBitSelectFrom(op->getResult(0)))
        return true;
    }
  }
  return false;
}

/// Return true for operations that are always inlined.
static bool isExpressionAlwaysInline(Operation *op) {
  if (isa<firrtl::ConstantOp>(op) || isa<SubfieldOp>(op))
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

void ModuleEmitter::collectNamesEmitDecls(Block &block) {
  // In the first pass, we fill in the symbol table, calculate the max width
  // of the declaration words and the max type width.
  size_t maxDeclNameWidth = 0, maxTypeWidth = 0;

  // Return the word (e.g. "wire") in Verilog to declare the specified thing.
  auto getVerilogDeclWord = [](Operation *op) -> StringRef {
    if (isa<RegOp>(op) || isa<RegResetOp>(op))
      return "reg";

    // Note: MemOp is handled as "wire" here because each of its subcomponents
    // are wires.  The corresponding 'reg' decl is handled specially below.
    return "wire";
  };

  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
  SmallVector<Value, 16> valuesToEmit;
  for (auto &op : block) {
    for (auto result : op.getResults()) {
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

      // Otherwise, it must be an expression or a declaration like a
      // RegOp/WireOp.
      auto nameAttr = op.getAttrOfType<StringAttr>("name");
      if (auto instance = dyn_cast<InstanceOp>(&op)) {
        if (nameAttr) {
          auto name = (nameAttr.getValue() + "_" +
                       instance.getPortNameStr(result.getResultNumber()))
                          .str();
          addName(result, name);
        }
      } else if (auto memory = dyn_cast<MemOp>(&op)) {
        if (nameAttr) {
          auto name = (nameAttr.getValue() + "_" +
                       memory.getPortNameStr(result.getResultNumber()))
                          .str();
          addName(result, name);
        }
      } else {
        addName(result, nameAttr);
      }

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

      // Flatten the type for processing of each individual element.
      fieldTypes.clear();
      flattenBundleTypes(result.getType(), "", false, fieldTypes);

      // Handle the reg declaration for a memory specially.
      if (auto memOp = dyn_cast<MemOp>(&op))
        if (auto dataType = memOp.getDataTypeOrNull())
          flattenBundleTypes(dataType, "", false, fieldTypes);

      bool hadError = false;
      for (const auto &elt : fieldTypes) {
        int bitWidth = getBitWidthOrSentinel(elt.type);
        if (bitWidth == -1) {
          emitError(&op, getName(result))
              << elt.suffix << " has an unsupported verilog type " << elt.type;
          hadError = true;
          break;
        }

        if (bitWidth != 1) { // Width 1 is implicit.
          // Add 5 to count the width of the "[:0] ".
          size_t thisWidth = getPrintedIntWidth(bitWidth - 1) + 5;
          maxTypeWidth = std::max(thisWidth, maxTypeWidth);
        }
      }

      if (!hadError)
        valuesToEmit.push_back(result);
    }
  }

  SmallPtrSet<Operation *, 8> ops;

  // Okay, now that we have measured the things to emit, emit the things.
  for (auto value : valuesToEmit) {

    auto *decl = value.getDefiningOp();

    // True if this is an op we haven't seen before
    auto unvisitedOp = std::get<1>(ops.insert(decl));

    auto word = getVerilogDeclWord(decl);

    // Flatten the type for processing of each individual element.
    fieldTypes.clear();
    flattenBundleTypes(value.getType(), "", false, fieldTypes);

    bool isFirst = true;
    for (const auto &elt : fieldTypes) {
      indent() << word;
      assert(maxDeclNameWidth);
      os.indent(maxDeclNameWidth - word.size() + 1);

      auto elementType = elt.type;
      emitTypePaddedToWidth(elementType, maxTypeWidth, decl);

      os << getName(value) << elt.suffix << ';';
      if (isFirst)
        emitLocationInfoAndNewLine(ops);
      else
        os << '\n';
    }

    // Handle the reg declaration for a memory specially because we need to
    // handle aggregate types and depths.
    if (auto memOp = dyn_cast<MemOp>(decl)) {
      auto memName = memOp.name().getValue().str();
      fieldTypes.clear();
      if (auto dataType = memOp.getDataTypeOrNull())
        flattenBundleTypes(dataType, "", false, fieldTypes);

      if (unvisitedOp) {
        for (auto fieldType : fieldTypes) {
          uint64_t depth = memOp.depth();
          indent() << "reg";
          os.indent(maxDeclNameWidth - 3 + 1);
          emitTypePaddedToWidth(fieldType.type, maxTypeWidth, decl);
          os << memName + fieldType.suffix;
          os << '[' << (depth - 1) << ":0];\n";
        }
      }
    }
  }

  if (!valuesToEmit.empty())
    os << '\n';
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
  class StmtDeclEmitter : public StmtVisitor<StmtDeclEmitter, bool>,
                          public DeclVisitor<StmtDeclEmitter, bool> {
  public:
    StmtDeclEmitter(ModuleEmitter &emitter) : emitter(emitter) {}

    using DeclVisitor::visitDecl;
    using StmtVisitor::visitStmt;
    bool visitStmt(AttachOp op) { return emitter.emitStatement(op), true; }
    bool visitStmt(ConnectOp op) { return emitter.emitStatement(op), true; }
    bool visitStmt(PartialConnectOp op) {
      return emitter.emitStatement(op), true;
    }
    bool visitStmt(DoneOp op) { return true; }
    bool visitStmt(PrintFOp op) { return emitter.emitStatement(op), true; }
    bool visitStmt(SkipOp op) { return true; }
    bool visitStmt(StopOp op) { return emitter.emitStatement(op), true; }
    bool visitUnhandledStmt(Operation *op) { return false; }
    bool visitInvalidStmt(Operation *op) { return dispatchDeclVisitor(op); }

    bool visitDecl(InstanceOp op) { return emitter.emitDecl(op), true; }
    bool visitDecl(NodeOp op) { return emitter.emitDecl(op), true; }
    bool visitDecl(RegOp op) { return emitter.emitDecl(op), true; }
    bool visitDecl(RegResetOp op) { return emitter.emitDecl(op), true; }
    bool visitDecl(MemOp op) { return emitter.emitDecl(op), true; }
    bool visitDecl(WireOp op) { return true; }

    bool visitUnhandledDecl(Operation *op) { return false; }
    bool visitInvalidDecl(Operation *op) { return false; }

  private:
    ModuleEmitter &emitter;
  };

  if (StmtDeclEmitter(*this).dispatchStmtVisitor(op))
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

void ModuleEmitter::emitFModule(FModuleOp module) {
  // Add all the ports to the name table.
  SmallVector<ModulePortInfo, 8> portInfo;
  module.getPortInfo(portInfo);

  size_t nextPort = 0;
  for (auto &port : portInfo)
    addName(module.getArgument(nextPort++), port.name);

  os << "module " << module.getName() << '(';
  if (!portInfo.empty())
    os << '\n';

  // Determine the width of the widest type we have to print so everything
  // lines up nicely.
  bool hasOutputs = false;
  unsigned maxTypeWidth = 0;
  for (auto &port : portInfo) {
    hasOutputs |= port.isOutput();

    int bitWidth = getBitWidthOrSentinel(port.type);
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
    bool isThisPortOutput = portInfo[portIdx].isOutput();
    if (isThisPortOutput)
      os << "output ";
    else
      os << (hasOutputs ? "input  " : "input ");

    int bitWidth = getBitWidthOrSentinel(portType);
    emitTypePaddedToWidth(portType, maxTypeWidth, module);

    // Emit the name.
    os << getName(module.getArgument(portIdx));
    ++portIdx;

    // If we have any more ports with the same types and the same direction,
    // emit them in a list on the same line.
    while (portIdx != e && portInfo[portIdx].isOutput() == isThisPortOutput &&
           bitWidth == getBitWidthOrSentinel(portInfo[portIdx].type)) {
      // Don't exceed our preferred line length.
      StringRef name = getName(module.getArgument(portIdx));
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
  // TODO(QoI): Emit file header, indicating the name of the circuit,
  // location info and any other interesting metadata (e.g. comment
  // block) attached to it.
  os << R"XXX(
// Standard header to adapt well known macros to our needs.
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
// Users can define 'PRINTF_COND' to add an extra gate to prints.
`ifdef PRINTF_COND
`define PRINTF_COND_ (`PRINTF_COND)
`else
`define PRINTF_COND_ 1
`endif
// Users can define 'STOP_COND' to add an extra gate to stop conditions.
`ifdef STOP_COND
`define STOP_COND_ (`STOP_COND)
`else
`define STOP_COND_ 1
`endif

// Users can define INIT_RANDOM as general code that gets injected into the
// initializer block for modules with registers.
`ifndef INIT_RANDOM
`define INIT_RANDOM
`endif

// If using random initialization, you can also define RANDOMIZE_DELAY to
// customize the delay used, otherwise 0.002 is used.
`ifndef RANDOMIZE_DELAY
`define RANDOMIZE_DELAY 0.002
`endif

// Define INIT_RANDOM_PROLOG_ for use in our modules below.
`ifdef RANDOMIZE
  `ifndef VERILATOR
    `define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end
  `else
    `define INIT_RANDOM_PROLOG_ `INIT_RANDOM
  `endif
`else
  `define INIT_RANDOM_PROLOG_
`endif
)XXX";

  for (auto &op : *circuit.getBody()) {
    if (auto module = dyn_cast<FModuleOp>(op)) {
      ModuleEmitter(state).emitFModule(module);
      continue;
    }

    // Ignore the done terminator at the end of the circuit.
    // Ignore 'ext modules'.
    if (isa<firrtl::DoneOp>(op) || isa<FExtModuleOp>(op))
      continue;

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

LogicalResult circt::exportFIRRTLToVerilog(ModuleOp module,
                                           llvm::raw_ostream &os) {
  VerilogEmitterState state(os);
  CircuitEmitter(state).emitMLIRModule(module);
  return failure(state.encounteredError);
}

void circt::registerFIRRTLToVerilogTranslation() {
  TranslateFromMLIRRegistration toVerilog(
      "export-firrtl-verilog", exportFIRRTLToVerilog,
      [](DialectRegistry &registry) {
        registry.insert<firrtl::FIRRTLDialect>();
      });
}
