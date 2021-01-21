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
using namespace rtl;
using namespace sv;

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
  if (isa<MergeOp>(op))
    return false;

  // All RTL combinatorial logic ops and SV expression ops are Verilog
  // expressions.
  return isCombinatorial(op) || isExpression(op);
}

/// Return the width of the specified type in bits or -1 if it isn't
/// supported.
static int getBitWidthOrSentinel(Type type) {
  return TypeSwitch<Type, int>(type)
      .Case<IntegerType>([](IntegerType integerType) {
        // Verilog doesn't support zero bit integers.  We only support them in
        // limited cases.
        return integerType.getWidth();
      })
      .Case<InOutType>([](InOutType inoutType) {
        return getBitWidthOrSentinel(inoutType.getElementType());
      })
      .Default([](Type) { return -1; });
}

/// Push this type's dimension into a vector.
static void getTypeDims(SmallVectorImpl<int64_t> &dims, Type type,
                        Location loc) {
  if (auto inout = type.dyn_cast<rtl::InOutType>())
    return getTypeDims(dims, inout.getElementType(), loc);
  if (auto uarray = type.dyn_cast<rtl::UnpackedArrayType>())
    return getTypeDims(dims, uarray.getElementType(), loc);
  if (type.isa<InterfaceType>())
    return;

  int width;
  if (auto arrayType = type.dyn_cast<rtl::ArrayType>()) {
    width = arrayType.getSize();
  } else {
    width = getBitWidthOrSentinel(type);
  }
  if (width == -1)
    mlir::emitError(loc, "value has an unsupported verilog type ") << type;

  if (width != 1) // Width 1 is implicit.
    dims.push_back(width);

  if (auto arrayType = type.dyn_cast<rtl::ArrayType>()) {
    getTypeDims(dims, arrayType.getElementType(), loc);
  }
}

/// Emit a type's packed dimensions, returning whether or not text was emitted.
static bool emitTypeDims(Type type, Location loc, raw_ostream &os) {
  SmallVector<int64_t, 4> dims;
  getTypeDims(dims, type, loc);

  bool emitted = false;
  for (int64_t width : dims)
    switch (width) {
    case -1: // -1 is an invalid type.
      os << "<<invalid type>>";
      emitted = true;
      return true;
    case 1: // Width 1 is implicit.
      assert(false && "Width 1 shouldn't be in the dim vector");
      break;
    case 0:
      os << "/*Zero Width*/";
      emitted = true;
      break;
    default:
      os << '[' << (width - 1) << ":0]";
      emitted = true;
      break;
    }
  return emitted;
}

/// True iff 'a' and 'b' have the same wire dims.
static bool haveMatchingDims(Type a, Type b, Location loc) {
  SmallVector<int64_t, 4> aDims;
  getTypeDims(aDims, a, loc);

  SmallVector<int64_t, 4> bDims;
  getTypeDims(bDims, b, loc);

  return aDims == bDims;
}

/// Emit the specified type dimensions and print out a trailing space if
/// anything is printed.
static void emitTypeDimWithSpaceIfNeeded(Type type, Location loc,
                                         raw_ostream &os) {
  if (emitTypeDims(type, loc, os))
    os << ' ';
}

/// Return true if this is a zero bit type, e.g. a zero bit integer or array
/// thereof.
static bool isZeroBitType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.getWidth() == 0;
  if (auto inout = type.dyn_cast<rtl::InOutType>())
    return isZeroBitType(inout.getElementType());
  if (auto uarray = type.dyn_cast<rtl::UnpackedArrayType>())
    return isZeroBitType(uarray.getElementType());
  if (auto array = type.dyn_cast<rtl::ArrayType>())
    return isZeroBitType(array.getElementType());

  // We have an open type system, so assume it is ok.
  return false;
}

/// Return true if this is a noop cast that will emit with no syntax.
static bool isNoopCast(Operation *op) {
  // These are always noop casts.
  if (isa<ReadInOutOp>(op))
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

class ModuleEmitter : public VerilogEmitterBase,
                      public rtl::StmtVisitor<ModuleEmitter, LogicalResult>,
                      public sv::Visitor<ModuleEmitter, LogicalResult> {

public:
  explicit ModuleEmitter(VerilogEmitterState &state)
      : VerilogEmitterBase(state) {}

  void emitRTLModule(RTLModuleOp module);
  void emitRTLExternModule(RTLExternModuleOp module);
  void emitExpression(Value exp, SmallPtrSet<Operation *, 8> &emittedExprs,
                      bool forceRootExpr = false);

  /// Emit the specified expression and return it as a string.
  std::string
  emitExpressionToString(Value exp, SmallPtrSet<Operation *, 8> &emittedExprs,
                         VerilogPrecedence precedence = LowestPrecedence);

  // Statements.
  void emitStatementExpression(Operation *op);

  // Visitor methods.
  LogicalResult visitUnhandledStmt(Operation *op) { return failure(); }
  LogicalResult visitInvalidStmt(Operation *op) { return failure(); }
  LogicalResult visitUnhandledSV(Operation *op) { return failure(); }
  LogicalResult visitInvalidSV(Operation *op) { return failure(); }
  using StmtVisitor::visitStmt;
  using Visitor::visitSV;

  void visitMerge(MergeOp op);
  LogicalResult visitStmt(WireOp op) { return success(); }
  LogicalResult visitSV(RegOp op) { return success(); }
  LogicalResult visitSV(InterfaceInstanceOp op) { return success(); }
  LogicalResult visitStmt(ConnectOp op);
  LogicalResult visitSV(BPAssignOp op);
  LogicalResult visitSV(PAssignOp op);
  LogicalResult visitSV(AliasOp op);
  LogicalResult visitStmt(OutputOp op);
  LogicalResult visitStmt(InstanceOp op);
  LogicalResult visitSV(IfDefOp op);
  LogicalResult visitSV(IfOp op);
  LogicalResult visitSV(AlwaysOp op);
  LogicalResult visitSV(InitialOp op);
  LogicalResult visitSV(FWriteOp op);
  LogicalResult visitSV(FatalOp op);
  LogicalResult visitSV(FinishOp op);
  LogicalResult visitSV(VerbatimOp op);
  LogicalResult visitSV(AssertOp op);
  LogicalResult visitSV(AssumeOp op);
  LogicalResult visitSV(CoverOp op);
  LogicalResult visitSV(InterfaceOp op);
  LogicalResult visitSV(InterfaceSignalOp op);
  LogicalResult visitSV(InterfaceModportOp op);
  LogicalResult visitSV(AssignInterfaceSignalOp op);
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

  llvm::StringSet<> usedNames;
  llvm::DenseMap<Value, llvm::StringMapEntry<llvm::NoneType> *> nameTable;
  size_t nextGeneratedNameID = 0;

  /// This set keeps track of all of the expression nodes that need to be
  /// emitted as standalone wire declarations.  This can happen because they are
  /// multiply-used or because the user requires a name to reference.
  SmallPtrSet<Operation *, 16> outOfLineExpressions;
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
enum SubExprSignResult { IsSigned, IsUnsigned };

/// This is information precomputed about each subexpression in the tree we
/// are emitting as a unit.
struct SubExprInfo {
  /// The precedence of this expression.
  VerilogPrecedence precedence;

  /// The signedness of the expression.
  SubExprSignResult signedness;

  SubExprInfo(VerilogPrecedence precedence, SubExprSignResult signedness)
      : precedence(precedence), signedness(signedness) {}
};

enum SubExprSignRequirement { NoRequirement, RequireSigned, RequireUnsigned };

} // namespace

namespace {
/// This builds a recursively nested expression from an SSA use-def graph.  This
/// uses a post-order walk, but it needs to obey precedence and signedness
/// constraints that depend on the behavior of the child nodes.  To handle this,
/// we emit the characters to a SmallVector which allows us to emit a bunch of
/// stuff, then pre-insert parentheses and other things if we find out that it
/// was needed later.
class ExprEmitter : public CombinatorialVisitor<ExprEmitter, SubExprInfo>,
                    public Visitor<ExprEmitter, SubExprInfo> {
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
  friend class CombinatorialVisitor<ExprEmitter, SubExprInfo>;
  friend class Visitor<ExprEmitter, SubExprInfo>;

  /// Emit the specified value as a subexpression to the stream.
  SubExprInfo emitSubExpr(Value exp, VerilogPrecedence parenthesizeIfLooserThan,
                          SubExprSignRequirement signReq = NoRequirement);

  SubExprInfo visitUnhandledExpr(Operation *op);
  SubExprInfo visitInvalidComb(Operation *op) { return dispatchSVVisitor(op); }
  SubExprInfo visitUnhandledComb(Operation *op) {
    return visitUnhandledExpr(op);
  }
  SubExprInfo visitUnhandledSV(Operation *op) { return visitUnhandledExpr(op); }

  using Visitor::visitSV;

  SubExprInfo emitBinary(Operation *op, VerilogPrecedence prec,
                         const char *syntax,
                         SubExprSignRequirement operandSignReq = NoRequirement);

  SubExprInfo emitVariadic(Operation *op, VerilogPrecedence prec,
                           const char *syntax);

  SubExprInfo emitUnary(Operation *op, const char *syntax,
                        bool resultAlwaysUnsigned = false);

  SubExprInfo emitNoopCast(Operation *op) {
    return emitSubExpr(op->getOperand(0), LowestPrecedence);
  }

  SubExprInfo visitSV(GetModportOp op);
  SubExprInfo visitSV(ReadInterfaceSignalOp op);
  SubExprInfo visitSV(TextualValueOp op);

  // Noop cast operators.
  SubExprInfo visitComb(ReadInOutOp op) { return emitNoopCast(op); }

  // Other
  SubExprInfo visitComb(ArraySliceOp op);
  SubExprInfo visitComb(ArrayIndexOp op);
  SubExprInfo visitComb(MuxOp op);

  // RTL Dialect Operations
  using CombinatorialVisitor::visitComb;
  SubExprInfo visitComb(ConstantOp op);
  SubExprInfo visitComb(AddOp op) { return emitVariadic(op, Addition, "+"); }
  SubExprInfo visitComb(SubOp op) { return emitBinary(op, Addition, "-"); }
  SubExprInfo visitComb(MulOp op) { return emitVariadic(op, Multiply, "*"); }
  SubExprInfo visitComb(DivUOp op) {
    return emitBinary(op, Multiply, "/", RequireUnsigned);
  }
  SubExprInfo visitComb(DivSOp op) {
    return emitBinary(op, Multiply, "/", RequireSigned);
  }
  SubExprInfo visitComb(ModUOp op) {
    return emitBinary(op, Multiply, "%", RequireUnsigned);
  }
  SubExprInfo visitComb(ModSOp op) {
    return emitBinary(op, Multiply, "%", RequireSigned);
  }
  SubExprInfo visitComb(ShlOp op) { return emitBinary(op, Shift, "<<"); }
  SubExprInfo visitComb(ShrUOp op) {
    // >> in Verilog is always an unsigned right shift.
    return emitBinary(op, Shift, ">>");
  }
  SubExprInfo visitComb(ShrSOp op) {
    // >>> is only an arithmetic shift right when both operands are signed.
    // Otherwise it does a logical shift.
    return emitBinary(op, Shift, ">>>", RequireSigned);
  }
  SubExprInfo visitComb(AndOp op) { return emitVariadic(op, And, "&"); }
  SubExprInfo visitComb(OrOp op) { return emitVariadic(op, Or, "|"); }
  SubExprInfo visitComb(XorOp op) {
    if (op.getNumOperands() == 2)
      if (auto cst =
              dyn_cast_or_null<ConstantOp>(op.getOperand(1).getDefiningOp()))
        if (cst.getValue().isAllOnesValue())
          return emitUnary(op, "~");

    return emitVariadic(op, Xor, "^");
  }

  // SystemVerilog spec 11.8.1: "Reduction operator results are unsigned,
  // regardless of the operands."
  SubExprInfo visitComb(AndROp op) { return emitUnary(op, "&", true); }
  SubExprInfo visitComb(OrROp op) { return emitUnary(op, "|", true); }
  SubExprInfo visitComb(XorROp op) { return emitUnary(op, "^", true); }

  SubExprInfo visitComb(SExtOp op);
  SubExprInfo visitComb(ConcatOp op);
  SubExprInfo visitComb(ExtractOp op);
  SubExprInfo visitComb(ICmpOp op);

  SubExprInfo visitComb(BitcastOp op);

private:
  /// This is set (before a visit method is called) if emitSubExpr would
  /// prefer to get an output of a specific sign.  This is a hint to cause the
  /// visitor to change its emission strategy, but the visit method can ignore
  /// it without a correctness problem.
  SubExprSignRequirement signPreference = NoRequirement;

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
                                    const char *syntax,
                                    SubExprSignRequirement operandSignReq) {
  auto lhsInfo = emitSubExpr(op->getOperand(0), prec, operandSignReq);
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

  auto rhsInfo = emitSubExpr(op->getOperand(1), rhsPrec, operandSignReq);

  // SystemVerilog 11.8.1 says that the result of a binary expression is signed
  // only if both operands are signed.
  SubExprSignResult signedness = IsUnsigned;
  if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
    signedness = IsSigned;

  return {prec, signedness};
}

SubExprInfo ExprEmitter::emitVariadic(Operation *op, VerilogPrecedence prec,
                                      const char *syntax) {
  // The result is signed if all the subexpressions are signed.
  SubExprSignResult sign = IsSigned;
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](Value v1) {
        if (emitSubExpr(v1, prec).signedness != IsSigned)
          sign = IsUnsigned;
      },
      [&] { os << ' ' << syntax << ' '; });

  return {prec, sign};
}

SubExprInfo ExprEmitter::emitUnary(Operation *op, const char *syntax,
                                   bool resultAlwaysUnsigned) {
  os << syntax;
  auto signedness = emitSubExpr(op->getOperand(0), Unary).signedness;
  return {Unary, resultAlwaysUnsigned ? IsUnsigned : signedness};
}

/// Emit the specified value as a subexpression to the stream.
SubExprInfo ExprEmitter::emitSubExpr(Value exp,
                                     VerilogPrecedence parenthesizeIfLooserThan,
                                     SubExprSignRequirement signRequirement) {
  auto *op = exp.getDefiningOp();
  bool shouldEmitInlineExpr = op && isVerilogExpression(op);

  // Don't emit this expression inline if it has multiple uses.
  if (shouldEmitInlineExpr && parenthesizeIfLooserThan != ForceEmitMultiUse &&
      emitter.outOfLineExpressions.count(op))
    shouldEmitInlineExpr = false;

  // If this is a non-expr or shouldn't be done inline, just refer to its name.
  if (!shouldEmitInlineExpr) {
    // All wires are declared as unsigned, so if the client needed it signed,
    // emit a conversion.
    if (signRequirement == RequireSigned) {
      os << "$signed(" << emitter.getName(exp) << ')';
      return {Symbol, IsSigned};
    }

    os << emitter.getName(exp);
    return {Symbol, IsUnsigned};
  }

  unsigned subExprStartIndex = resultBuffer.size();

  // Inform the visit method about the preferred sign we want from the result.
  // It may choose to ignore this, but some emitters can change behavior based
  // on contextual desired sign.
  signPreference = signRequirement;

  // Okay, this is an expression we should emit inline.  Do this through our
  // visitor.
  auto expInfo = dispatchCombinatorialVisitor(exp.getDefiningOp());

  // Check cases where we have to insert things before the expression now that
  // we know things about it.
  auto addPrefix = [&](StringRef prefix) {
    resultBuffer.insert(resultBuffer.begin() + subExprStartIndex,
                        prefix.begin(), prefix.end());
  };
  if (signRequirement == RequireSigned && expInfo.signedness == IsUnsigned) {
    addPrefix("$signed(");
    os << ')';
    expInfo.signedness = IsSigned;
  } else if (signRequirement == RequireUnsigned &&
             expInfo.signedness == IsSigned) {
    addPrefix("$unsigned(");
    os << ')';
    expInfo.signedness = IsUnsigned;
  } else if (expInfo.precedence > parenthesizeIfLooserThan) {
    // If this subexpression would bind looser than the expression it is bound
    // into, then we need to parenthesize it.  Insert the parentheses
    // retroactively.
    addPrefix("(");
    os << ')';
  }

  // Remember that we emitted this.
  emittedExprs.insert(exp.getDefiningOp());
  return expInfo;
}

SubExprInfo ExprEmitter::visitComb(SExtOp op) {
  auto inWidth = op.getOperand().getType().getIntOrFloatBitWidth();
  auto destWidth = op.getType().getIntOrFloatBitWidth();

  // Handle sign extend from a single bit in a pretty way.
  if (inWidth == 1) {
    os << '{' << destWidth << '{';
    emitSubExpr(op.getOperand(), LowestPrecedence);
    os << "}}";
    return {Unary, IsUnsigned};
  }

  // Otherwise, this is a sign extension of a general expression.
  os << "{{" << (destWidth - inWidth) << '{';
  emitSubExpr(op.getOperand(), Unary);
  os << '[' << (inWidth - 1) << "]}}, ";
  emitSubExpr(op.getOperand(), LowestPrecedence);
  os << '}';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(ConcatOp op) {
  // If all of the operands are the same, we emit this as a SystemVerilog
  // replicate operation, ala SV Spec 11.4.12.1.
  auto firstOperand = op.getOperand(0);
  bool allSame = llvm::all_of(op.getOperands(), [&firstOperand](auto operand) {
    return operand == firstOperand;
  });

  if (allSame) {
    os << '{' << op.getNumOperands() << '{';
    emitSubExpr(firstOperand, LowestPrecedence);
    os << "}}";
    return {Unary, IsUnsigned};
  }

  os << '{';
  llvm::interleaveComma(op.getOperands(), os,
                        [&](Value v) { emitSubExpr(v, LowestPrecedence); });

  os << '}';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(BitcastOp op) {
  // NOTE: Bitcasts are always emitted out-of-line with their own wire
  // declaration. SystemVerilog uses the wire declaration to know what type this
  // value is being casted to.
  Type toType = op.getType();
  if (!haveMatchingDims(toType, op.input().getType(), op.getLoc())) {
    os << "/*cast(bit";
    emitTypeDims(toType, op.getLoc(), os);
    os << ")*/";
  }
  return emitSubExpr(op.input(), LowestPrecedence);
}

SubExprInfo ExprEmitter::visitComb(ICmpOp op) {
  const char *symop[] = {"==", "!=", "<",  "<=", ">",
                         ">=", "<",  "<=", ">",  ">="};
  SubExprSignRequirement signop[] = {
      // Equality
      NoRequirement, NoRequirement,
      // Signed Comparisons
      RequireSigned, RequireSigned, RequireSigned, RequireSigned,
      // Unsigned Comparisons
      RequireUnsigned, RequireUnsigned, RequireUnsigned, RequireUnsigned};

  auto pred = static_cast<uint64_t>(op.predicate());
  assert(pred < sizeof(symop) / sizeof(symop[0]));
  auto result = emitBinary(op, Comparison, symop[pred], signop[pred]);

  // SystemVerilog 11.8.1: "Comparison... operator results are unsigned,
  // regardless of the operands".
  result.signedness = IsUnsigned;
  return result;
}

SubExprInfo ExprEmitter::visitComb(ExtractOp op) {
  unsigned loBit = op.lowBit();
  unsigned hiBit = loBit + op.getType().getWidth() - 1;

  auto x = emitSubExpr(op.input(), LowestPrecedence);
  assert(x.precedence == Symbol &&
         "should be handled by isExpressionUnableToInline");

  // If we're extracting the whole input, just return it.  This is valid but
  // non-canonical IR, and we don't want to generate invalid Verilog.
  if (loBit == 0 && op.input().getType().getIntOrFloatBitWidth() == hiBit + 1)
    return x;

  os << '[' << hiBit;
  if (hiBit != loBit) // Emit x[4] instead of x[4:4].
    os << ':' << loBit;
  os << ']';
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(GetModportOp op) {
  os << emitter.getName(op.iface()) + "." + op.field();
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ReadInterfaceSignalOp op) {
  os << emitter.getName(op.iface()) + "." + op.signalName();
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(TextualValueOp op) {
  os << op.string();
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(ConstantOp op) {
  bool isNegated = false;
  const APInt &value = op.getValue();
  // If this is a negative signed number and not MININT (e.g. -128), then print
  // it as a negated positive number.
  if (signPreference == RequireSigned && value.isNegative() &&
      !value.isMinSignedValue()) {
    os << '-';
    isNegated = true;
  }

  os << op.getType().getWidth() << '\'';

  // Emit this as a signed constant if the caller would prefer that.
  if (signPreference == RequireSigned)
    os << 's';
  os << 'h';

  // Print negated if required.
  SmallString<32> valueStr;
  if (isNegated) {
    (-value).toStringUnsigned(valueStr, 16);
  } else {
    value.toStringUnsigned(valueStr, 16);
  }
  os << valueStr;
  return {Unary, signPreference == RequireSigned ? IsSigned : IsUnsigned};
}

// 11.5.1 "Vector bit-select and part-select addressing" allows a '+:' syntax
// for slicing operations.
SubExprInfo ExprEmitter::visitComb(ArraySliceOp op) {
  auto arrayPrec = emitSubExpr(op.input(), Symbol);

  unsigned dstWidth = op.getType().getSize();
  os << '[';
  emitSubExpr(op.lowIndex(), LowestPrecedence);
  os << "+:" << dstWidth << ']';
  return {Unary, arrayPrec.signedness};
}

SubExprInfo ExprEmitter::visitComb(ArrayIndexOp op) {
  auto arrayPrec = emitSubExpr(op.input(), Symbol);
  os << '[';
  emitSubExpr(op.index(), LowestPrecedence);
  os << ']';
  return {Symbol, arrayPrec.signedness};
}

SubExprInfo ExprEmitter::visitComb(MuxOp op) {
  // The ?: operator is right associative.
  emitSubExpr(op.cond(), VerilogPrecedence(Conditional - 1));
  os << " ? ";
  auto lhsInfo =
      emitSubExpr(op.trueValue(), VerilogPrecedence(Conditional - 1));
  os << " : ";
  auto rhsInfo = emitSubExpr(op.falseValue(), Conditional);

  SubExprSignResult signedness = IsUnsigned;
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
  } else if (isZeroBitType(op->getResult(0).getType())) {
    indent() << "// Zero width: ";
  } else if (emitInlineWireDecls) {
    indent() << "wire ";
    emitTypeDimWithSpaceIfNeeded(op->getResult(0).getType(), op->getLoc(), os);
    os << getName(op->getResult(0)) << " = ";
  } else {
    indent() << "assign " << getName(op->getResult(0)) << " = ";
  }
  SmallPtrSet<Operation *, 8> emittedExprs;
  emitExpression(op->getResult(0), emittedExprs, /*forceRootExpr=*/true);
  os << ';';
  emitLocationInfoAndNewLine(emittedExprs);
}

void ModuleEmitter::visitMerge(MergeOp op) {
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

LogicalResult ModuleEmitter::visitStmt(ConnectOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "assign ";
  emitExpression(op.dest(), ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(BPAssignOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent();
  emitExpression(op.dest(), ops);
  os << " = ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(PAssignOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent();
  emitExpression(op.dest(), ops);
  os << " <= ";
  emitExpression(op.src(), ops);
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(AliasOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "alias ";
  llvm::interleave(
      op.getOperands(), os, [&](Value v) { emitExpression(v, ops); }, " = ");
  os << ';';
  emitLocationInfoAndNewLine(ops);
  return success();
}

/// For OutputOp we put "assign" statements at the end of the Verilog module to
/// assign the module outputs to intermediate wires.
LogicalResult ModuleEmitter::visitStmt(OutputOp op) {
  SmallPtrSet<Operation *, 8> ops;

  SmallVector<ModulePortInfo, 8> ports;
  RTLModuleOp parent = op->getParentOfType<RTLModuleOp>();
  parent.getPortInfo(ports);
  size_t operandIndex = 0;
  for (ModulePortInfo port : ports) {
    if (!port.isOutput())
      continue;
    ops.clear();
    ops.insert(op);
    indent();
    if (isZeroBitType(port.type))
      os << "// Zero width: ";
    os << "assign " << port.getName() << " = ";
    emitExpression(op.getOperand(operandIndex), ops);
    os << ';';
    emitLocationInfoAndNewLine(ops);
    ++operandIndex;
  }
  return success();
}

LogicalResult ModuleEmitter::visitSV(FWriteOp op) {
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
  return success();
}

LogicalResult ModuleEmitter::visitSV(FatalOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "$fatal;";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(VerbatimOp op) {
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
  return success();
}

LogicalResult ModuleEmitter::visitSV(FinishOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "$finish;";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(AssertOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "assert(" << emitExpressionToString(op.predicate(), ops) << ");";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(AssumeOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "assume(" << emitExpressionToString(op.property(), ops) << ");";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(CoverOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  indent() << "cover(" << emitExpressionToString(op.property(), ops) << ");";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(IfDefOp op) {
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
  return success();
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
    return isa<FWriteOp>(op) || isa<FinishOp>(op) || isa<FatalOp>(op) ||
           isa<AssertOp>(op) || isa<AssumeOp>(op) || isa<CoverOp>(op) ||
           isa<BPAssignOp>(op) || isa<PAssignOp>(op) || isa<ConnectOp>(op);
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

LogicalResult ModuleEmitter::visitSV(IfOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "if (" << emitExpressionToString(op.cond(), ops) << ')';
  emitBeginEndRegion(op.getThenBlock(), ops, *this);
  if (op.hasElse()) {
    indent() << "else";
    emitBeginEndRegion(op.getElseBlock(), ops, *this);
  }
  return success();
}

LogicalResult ModuleEmitter::visitSV(AlwaysOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto printEvent = [&](AlwaysOp::Condition cond) {
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
  return success();
}

LogicalResult ModuleEmitter::visitSV(InitialOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  indent() << "initial";
  emitBeginEndRegion(op.getBodyBlock(), ops, *this, "initial");
  return success();
}

LogicalResult ModuleEmitter::visitStmt(InstanceOp op) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto *moduleOp = op.getReferencedModule();
  assert(moduleOp && "Invalid IR");

  // If this is a reference to an external module with a hard coded Verilog
  // name, then use it here.  This is a hack because we lack proper support for
  // parameterized modules in the RTL dialect.
  if (auto extMod = dyn_cast<RTLExternModuleOp>(moduleOp)) {
    indent() << extMod.getVerilogModuleName();
  } else {
    indent() << op.moduleName();
  }

  // Helper that prints a parameter constant value in a Verilog compatible way.
  auto printParmValue = [&](Attribute value) {
    if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
      IntegerType intTy = intAttr.getType().cast<IntegerType>();
      SmallString<20> numToPrint;
      intAttr.getValue().toString(numToPrint, 10, intTy.isSigned());
      os << intTy.getWidth() << "'d" << numToPrint;
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
      os << " #(\n";
      llvm::interleave(
          paramDict, os,
          [&](NamedAttribute elt) {
            os.indent(state.currentIndent + 2) << '.' << elt.first << '(';
            printParmValue(elt.second);
            os << ')';
          },
          ",\n");
      os << '\n';
      indent() << ')';
    }
  }

  os << ' ' << op.instanceName() << " (";

  SmallVector<ModulePortInfo, 8> portInfo;
  getModulePortInfo(moduleOp, portInfo);

  // Get the max port name length so we can align the '('.
  size_t maxNameLength = 0;
  for (auto &elt : portInfo) {
    maxNameLength = std::max(maxNameLength, elt.getName().size());
  }

  // Emit the argument and result ports.
  auto opArgs = op.inputs();
  auto opResults = op.getResults();
  bool isFirst = true; // True until we print a port.
  for (auto &elt : portInfo) {
    // Figure out which value we are emitting.
    Value portVal = elt.isOutput() ? opResults[elt.argNum] : opArgs[elt.argNum];
    bool isZeroWidth = isZeroBitType(elt.type);

    // Decide if we should print a comma.  We can't do this if we're the first
    // port or if all the subsequent ports are zero width.
    if (!isFirst) {
      bool shouldPrintComma = true;
      if (isZeroWidth) {
        shouldPrintComma = false;
        for (size_t i = (&elt - portInfo.data()) + 1, e = portInfo.size();
             i != e; ++i)
          if (!isZeroBitType(portInfo[i].type)) {
            shouldPrintComma = true;
            break;
          }
      }

      if (shouldPrintComma)
        os << ',';
    }
    emitLocationInfoAndNewLine(ops);

    // Emit the port's name.
    indent();
    if (!isZeroWidth) {
      // If this is a real port we're printing, then it isn't the first one. Any
      // subsequent ones will need a comma.
      isFirst = false;
      os << "  ";
    } else {
      // We comment out zero width ports, so their presence and initializer
      // expressions are still emitted textually.
      os << "//";
    }

    os << '.' << elt.getName();
    os.indent(maxNameLength - elt.getName().size()) << " (";

    // Emit the value as an expression.
    ops.clear();
    emitExpression(portVal, ops);
    os << ')';
  }
  if (!isFirst) {
    emitLocationInfoAndNewLine(ops);
    ops.clear();
    indent();
  }
  os << ");";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult ModuleEmitter::visitSV(InterfaceOp op) {
  os << "interface " << op.sym_name() << ";\n";

  addIndent();
  for (auto &op : op.getBodyBlock()->without_terminator())
    emitOperation(&op);
  reduceIndent();

  os << "endinterface\n\n";
  return success();
}

LogicalResult ModuleEmitter::visitSV(InterfaceSignalOp op) {
  if (!isZeroBitType(op.type()))
    indent() << "logic ";
  else
    indent() << "// Zero width: logic ";

  emitTypeDimWithSpaceIfNeeded(op.type(), op.getLoc(), os);
  os << op.sym_name() << ";\n";
  return success();
}

LogicalResult ModuleEmitter::visitSV(InterfaceModportOp op) {
  indent() << "modport " << op.sym_name() << '(';

  llvm::interleaveComma(op.ports(), os, [&](const Attribute &portAttr) {
    auto port = portAttr.cast<ModportStructAttr>();
    os << port.direction().getValue() << ' ' << port.signal().getValue();
  });

  os << ");\n";
  return success();
}

LogicalResult ModuleEmitter::visitSV(AssignInterfaceSignalOp op) {
  SmallPtrSet<Operation *, 8> emitted;
  indent() << "assign ";
  emitExpression(op.iface(), emitted, /*forceRootExpr=*/true);
  os << "." << op.signalName() << " = ";
  emitExpression(op.rhs(), emitted, /*forceRootExpr=*/true);
  os << ";\n";
  return success();
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
  if (auto cast = dyn_cast<BitcastOp>(op))
    if (!haveMatchingDims(cast.input().getType(), cast.result().getType(),
                          op->getLoc()))
      // Bitcasts rely on the type being assigned to, so we cannot inline.
      return true;

  // Scan the users of the operation to see if any of them need this to be
  // emitted out-of-line.
  for (auto user : op->getUsers()) {
    // Verilog bit selection is required by the standard to be:
    // "a vector, packed array, packed structure, parameter or concatenation".
    // It cannot be an arbitrary expression.
    if (isa<ExtractOp>(user))
      if (!isOkToBitSelectFrom(op->getResult(0)))
        return true;

    // Sign extend (when the operand isn't a single bit) requires a bitselect
    // syntactically.
    if (auto sext = dyn_cast<SExtOp>(user)) {
      auto sextOperandType = sext.getOperand().getType().cast<IntegerType>();
      if (sextOperandType.getWidth() != 1 &&
          !isOkToBitSelectFrom(op->getResult(0)))
        return true;
    }
    // ArraySliceOp uses its operand twice, so we want to assign it first then
    // use that variable in the ArraySliceOp expression.
    if (isa<ArraySliceOp>(user) && !isa<ConstantOp>(op))
      return true;
  }
  return false;
}

/// Return true for operations that are always inlined.
static bool isExpressionAlwaysInline(Operation *op) {
  if (isa<ConstantOp>(op) || isa<ArrayIndexOp>(op))
    return true;

  // An SV interface modport is a symbolic name that is always inlined.
  if (isa<GetModportOp>(op) || isa<ReadInterfaceSignalOp>(op))
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
  if (auto inout = type.dyn_cast<InOutType>())
    return printArraySubscripts(inout.getElementType(), os);

  if (auto array = type.dyn_cast<UnpackedArrayType>()) {
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
    if (isa<RegOp>(op))
      return "reg";

    // Interfaces instances use the name of the declared interface.
    if (auto interface = dyn_cast<InterfaceInstanceOp>(op))
      return interface.getInterfaceType().getInterface().getValue();

    return "wire";
  };

  // This is information we keep track of for each wire/reg/interface
  // declaration we're going to emit.
  struct ValuesToEmitRecord {
    Value value;
    SmallString<8> typeString;
  };
  SmallVector<ValuesToEmitRecord, 16> valuesToEmit;

  SmallString<32> nameTmp;

  // Loop over all of the results of all of the ops.  Anything that defines a
  // value needs to be noticed.
  for (auto &op : block) {
    bool isExpr = isVerilogExpression(&op);

    for (auto result : op.getResults()) {
      // If this is an expression emitted inline or unused, it doesn't need a
      // name.
      if (isExpr) {
        // If this expression is dead, or can be emitted inline, ignore it.
        if (result.use_empty() || isExpressionEmittedInline(&op))
          continue;

        // Remember that this expression should be emitted out of line.
        outOfLineExpressions.insert(&op);
      }

      // Otherwise, it must be an expression or a declaration like a
      // RegOp/WireOp.  Remember and unique the name for this operation.
      if (auto instance = dyn_cast<InstanceOp>(&op)) {
        // The name for an instance result is custom.
        nameTmp = instance.instanceName().str() + "_";
        unsigned resultNumber = result.getResultNumber();
        auto resultName = instance.getResultName(resultNumber);
        if (resultName)
          nameTmp += resultName.getValue().str();
        else
          nameTmp += std::to_string(resultNumber);
        addName(result, nameTmp);
      } else {
        addName(result, op.getAttrOfType<StringAttr>("name"));
      }

      // If we are emitting out-of-line expressions using inline wire decls,
      // don't measure or emit this wire, it will be emitted inline.
      if (isExpr && emitInlineWireDecls)
        continue;

      // Emit this value.
      valuesToEmit.push_back(ValuesToEmitRecord{result, {}});
      auto &typeString = valuesToEmit.back().typeString;

      maxDeclNameWidth =
          std::max(getVerilogDeclWord(&op).size(), maxDeclNameWidth);

      // Convert the port's type to a string and measure it.
      {
        llvm::raw_svector_ostream stringStream(typeString);
        emitTypeDimWithSpaceIfNeeded(result.getType(), op.getLoc(),
                                     stringStream);
      }
      maxTypeWidth = std::max(typeString.size(), maxTypeWidth);
    }
  }

  SmallPtrSet<Operation *, 8> ops;

  // Okay, now that we have measured the things to emit, emit the things.
  for (const auto &record : valuesToEmit) {
    auto *decl = record.value.getDefiningOp();
    ops.clear();
    ops.insert(decl);

    // Emit the leading word, like 'wire' or 'reg'.
    auto type = record.value.getType();
    auto word = getVerilogDeclWord(decl);
    if (!isZeroBitType(type)) {
      indent() << word;
      os.indent(maxDeclNameWidth - word.size() + 1);
    } else {
      indent() << "// Zero width: " << word << ' ';
    }

    // Emit the type.
    os << record.typeString;
    if (record.typeString.size() < maxTypeWidth)
      os.indent(maxTypeWidth - record.typeString.size());

    // Emit the name.
    os << getName(record.value);

    // Interface instantiations have parentheses like a module with no ports.
    if (type.isa<InterfaceType>()) {
      os << "()";
    } else {
      // Print out any array subscripts.
      printArraySubscripts(type, os);
    }

    os << ';';
    emitLocationInfoAndNewLine(ops);
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

  // Handle RTL statements.
  if (succeeded(dispatchStmtVisitor(op)))
    return;

  // Handle SV Statements.
  if (succeeded(dispatchSVVisitor(op)))
    return;

  if (auto merge = dyn_cast<MergeOp>(op))
    return visitMerge(merge);

  emitOpError(op, "cannot emit this operation to Verilog");
  indent() << "unknown MLIR operation " << op->getName().getStringRef() << "\n";
}

void ModuleEmitter::emitRTLExternModule(RTLExternModuleOp module) {
  os << "// external module " << module.getName() << "\n\n";
}

void ModuleEmitter::emitRTLModule(RTLModuleOp module) {
  // Add all the ports to the name table.
  SmallVector<ModulePortInfo, 8> portInfo;
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
  bool hasOutputs = false, hasZeroWidth = false;
  size_t maxTypeWidth = 0;
  SmallVector<SmallString<8>, 16> portTypeStrings;

  for (auto &port : portInfo) {
    hasOutputs |= port.isOutput();
    hasZeroWidth |= isZeroBitType(port.type);

    // Convert the port's type to a string and measure it.
    portTypeStrings.push_back({});
    {
      llvm::raw_svector_ostream stringStream(portTypeStrings.back());
      emitTypeDimWithSpaceIfNeeded(port.type, module.getLoc(), stringStream);
    }

    maxTypeWidth = std::max(portTypeStrings.back().size(), maxTypeWidth);
  }

  addIndent();

  for (size_t portIdx = 0, e = portInfo.size(); portIdx != e;) {
    size_t startOfLinePos = os.tell();

    indent();
    // Emit the arguments.
    auto portType = portInfo[portIdx].type;
    bool isZeroWidth = false;
    if (hasZeroWidth) {
      isZeroWidth = isZeroBitType(portType);
      os << (isZeroWidth ? "// " : "   ");
    }

    PortDirection thisPortDirection = portInfo[portIdx].direction;
    switch (thisPortDirection) {
    case PortDirection::OUTPUT:
      os << "output ";
      break;
    case PortDirection::INPUT:
      os << (hasOutputs ? "input  " : "input ");
      break;
    case PortDirection::INOUT:
      os << (hasOutputs ? "inout  " : "inout ");
      break;
    }

    // Emit the type.
    os << portTypeStrings[portIdx];
    if (portTypeStrings[portIdx].size() < maxTypeWidth)
      os.indent(maxTypeWidth - portTypeStrings[portIdx].size());

    // Emit the name.
    os << portInfo[portIdx].getName();
    printArraySubscripts(portType, os);
    ++portIdx;

    // If we have any more ports with the same types and the same direction,
    // emit them in a list on the same line.
    while (portIdx != e && portInfo[portIdx].direction == thisPortDirection &&
           portType == portInfo[portIdx].type) {
      // Don't exceed our preferred line length.
      StringRef name = portInfo[portIdx].getName();
      if (os.tell() + 2 + name.size() - startOfLinePos >
          // We use "-2" here because we need a trailing comma or ); for the
          // decl.
          preferredSourceWidth - 2)
        break;

      // Append this to the running port decl.
      os << ", " << name;
      printArraySubscripts(portType, os);
      ++portIdx;
    }

    if (portIdx != e)
      os << ',';
    else if (isZeroWidth)
      os << "\n   );\n";
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
    if (auto module = dyn_cast<RTLModuleOp>(op))
      ModuleEmitter(state).emitRTLModule(module);
    else if (auto module = dyn_cast<RTLExternModuleOp>(op))
      ModuleEmitter(state).emitRTLExternModule(module);
    else if (isa<InterfaceOp>(op) || isa<VerbatimOp>(op) || isa<IfDefOp>(op))
      ModuleEmitter(state).emitOperation(&op);
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
        registry.insert<RTLDialect, SVDialect>();
      });
}
