//===- FIRParser.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "cirt/FIRParser.h"
#include "FIRLexer.h"
#include "cirt/Dialect/FIRRTL/Ops.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Translation.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace cirt;
using namespace firrtl;
using namespace mlir;
using llvm::SMLoc;
using llvm::SourceMgr;

/// Return true if this is a useless temporary name produced by FIRRTL.  We
/// drop these as they don't convey semantic meaning.
static bool isUselessName(StringRef name) {
  // Ignore _T and _T_123
  if (name.startswith("_T"))
    return name.size() == 2 || name[2] == '_';

  // Ignore _GEN and _GEN_123, these are produced by Namespace.scala.
  if (name.startswith("_GEN"))
    return name.size() == 4 || name[4] == '_';
  return false;
}

/// If the specified name is a useless temporary name produced by FIRRTL, return
/// an empty attribute to ignore it.  Otherwise, return the argument unmodified.
static StringAttr filterUselessName(StringAttr name) {
  return isUselessName(name.getValue()) ? StringAttr() : name;
}

//===----------------------------------------------------------------------===//
// GlobalFIRParserState
//===----------------------------------------------------------------------===//

namespace {
/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position.  This is separated out from the parser
/// so that individual subparsers can refer to the same state.
struct GlobalFIRParserState {
  GlobalFIRParserState(const llvm::SourceMgr &sourceMgr, MLIRContext *context,
                       FIRParserOptions options)
      : context(context), options(options), lex(sourceMgr, context),
        curToken(lex.lexToken()) {}

  /// The context we're parsing into.
  MLIRContext *const context;

  // Options that control the behavior of the parser.
  const FIRParserOptions options;

  /// The lexer for the source file we're parsing.
  FIRLexer lex;

  /// This is the next token that hasn't been consumed yet.
  FIRToken curToken;

private:
  GlobalFIRParserState(const GlobalFIRParserState &) = delete;
  void operator=(const GlobalFIRParserState &) = delete;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// FIRParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic common to all levels of the parser, including
/// things like types and helper logic.
struct FIRParser {
  FIRParser(GlobalFIRParserState &state) : state(state) {}

  // Helper methods to get stuff from the parser-global state.
  GlobalFIRParserState &getState() const { return state; }
  MLIRContext *getContext() const { return state.context; }
  const llvm::SourceMgr &getSourceMgr() { return state.lex.getSourceMgr(); }

  /// Return the indentation level of the specified token.
  Optional<unsigned> getIndentation() const {
    return state.lex.getIndentation(state.curToken);
  }

  /// Return the current token the parser is inspecting.
  const FIRToken &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(state.curToken.getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  class LocWithInfo;

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return state.lex.translateLocation(loc);
  }

  /// Parse an @info marker if present.  If so, apply the symbolic location
  /// specified it to all of the operations listed in subOps.
  ParseResult parseOptionalInfo(LocWithInfo &result,
                                ArrayRef<Operation *> subOps = {});

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(FIRToken::Kind kind) {
    if (state.curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(state.curToken.isNot(FIRToken::eof, FIRToken::error) &&
           "shouldn't advance past EOF or errors");
    state.curToken = state.lex.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(FIRToken::Kind kind) {
    assert(state.curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// Capture the current token's location into the specified value.  This
  /// always succeeds.
  ParseResult parseGetLocation(SMLoc &loc);
  ParseResult parseGetLocation(Location &loc);

  /// Capture the current token's spelling into the specified value.  This
  /// always succeeds.
  ParseResult parseGetSpelling(StringRef &spelling) {
    spelling = getTokenSpelling();
    return success();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(FIRToken::Kind expectedToken, const Twine &message);

  /// Parse a list of elements, terminated with an arbitrary token.
  ParseResult parseListUntil(FIRToken::Kind rightToken,
                             const std::function<ParseResult()> &parseElement);

  //===--------------------------------------------------------------------===//
  // Common Parser Rules
  //===--------------------------------------------------------------------===//

  /// Parse 'intLit' into the specified value.
  ParseResult parseIntLit(APInt &result, const Twine &message);
  ParseResult parseIntLit(int32_t &result, const Twine &message);

  // Parse ('<' intLit '>')? setting result to -1 if not present.
  ParseResult parseOptionalWidth(int32_t &result);

  // Parse the 'id' grammar, which is an identifier or an allowed keyword.
  ParseResult parseId(StringRef &result, const Twine &message);
  ParseResult parseId(StringAttr &result, const Twine &message);
  ParseResult parseFieldId(StringRef &result, const Twine &message);
  ParseResult parseType(FIRRTLType &result, const Twine &message);

  ParseResult parseOptionalRUW(RUWAttr &result);

private:
  FIRParser(const FIRParser &) = delete;
  void operator=(const FIRParser &) = delete;

  /// FIRParser is subclassed and reinstantiated.  Do not add additional
  /// non-trivial state here, add it to the GlobalFIRParserState class.
  GlobalFIRParserState &state;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

InFlightDiagnostic FIRParser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(FIRToken::error))
    diag.abandon();
  return diag;
}

//===----------------------------------------------------------------------===//
// Token Parsing
//===----------------------------------------------------------------------===//

/// Capture the current token's location into the specified value.  This
/// always succeeds.
ParseResult FIRParser::parseGetLocation(SMLoc &loc) {
  loc = getToken().getLoc();
  return success();
}

ParseResult FIRParser::parseGetLocation(Location &loc) {
  loc = translateLocation(getToken().getLoc());
  return success();
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult FIRParser::parseToken(FIRToken::Kind expectedToken,
                                  const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Parse a list of elements, terminated with an arbitrary token.
ParseResult
FIRParser::parseListUntil(FIRToken::Kind rightToken,
                          const std::function<ParseResult()> &parseElement) {

  while (!consumeIf(rightToken)) {
    if (parseElement())
      return failure();
  }
  return success();
}

//===--------------------------------------------------------------------===//
// Location Processing
//===--------------------------------------------------------------------===//

/// This helper class is used to handle Info records, which specify higher level
/// symbolic source location, that may be missing from the file.  If the higher
/// level source information is missing, we fall back to the location in the
/// .fir file.
class FIRParser::LocWithInfo {
public:
  explicit LocWithInfo(SMLoc firLoc, FIRParser *parser)
      : parser(parser), firLoc(firLoc) {}

  SMLoc getFIRLoc() const { return firLoc; }

  Location getLoc() const {
    if (infoLoc.hasValue())
      return infoLoc.getValue();
    return parser->translateLocation(firLoc);
  }

  void setInfoLocation(Location loc) { infoLoc = loc; }

private:
  FIRParser *const parser;

  /// This is the designated location in the .fir file for use when there is no
  /// @ info marker.
  SMLoc firLoc;

  /// This is the location specified by the @ marker if present.
  Optional<Location> infoLoc;
};

/// Parse an @info marker if present.  If so, apply the symbolic location
/// specified it to all of the operations listed in subOps.
///
/// info ::= FileInfo
///
ParseResult FIRParser::parseOptionalInfo(LocWithInfo &result,
                                         ArrayRef<Operation *> subOps) {
  if (getToken().isNot(FIRToken::fileinfo))
    return success();

  auto loc = getToken().getLoc();

  // See if we can parse this token into a File/Line/Column record.  If not,
  // just ignore it with a warning.
  auto unknownFormat = [&]() -> ParseResult {
    mlir::emitWarning(translateLocation(loc),
                      "ignoring unknown @ info record format");
    return success();
  };

  auto spelling = getTokenSpelling();
  consumeToken(FIRToken::fileinfo);

  // The spelling of the token looks something like "@[Decoupled.scala 221:8]".
  if (!spelling.startswith("@[") || !spelling.endswith("]"))
    return unknownFormat();

  spelling = spelling.drop_front(2).drop_back(1);

  // Split at the last space.
  auto spaceLoc = spelling.find_last_of(' ');
  if (spaceLoc == StringRef::npos)
    return unknownFormat();

  auto filename = spelling.take_front(spaceLoc);
  auto lineAndColumn = spelling.drop_front(spaceLoc + 1);

  // Decode the line/column.  If the colon is missing, then it will be empty
  // here.
  StringRef lineStr, colStr;
  std::tie(lineStr, colStr) = lineAndColumn.split(':');

  // Zero represents an unknown line/column number.
  unsigned lineNo = 0, columnNo = 0;

  // Decode the line number and the column number if present.
  if (lineStr.getAsInteger(10, lineNo))
    return unknownFormat();
  if (!colStr.empty() && colStr.getAsInteger(10, columnNo))
    return unknownFormat();

  // If info locators are ignored, don't actually apply them.  We still do all
  // the verification above though.
  if (state.options.ignoreInfoLocators)
    return success();

  auto resultLoc =
      FileLineColLoc::get(filename, lineNo, columnNo, getContext());
  result.setInfoLocation(resultLoc);

  // Now that we have a symbolic location, apply it to any subOps specified.
  for (auto *op : subOps) {
    op->setLoc(resultLoc);
  }

  return success();
}

//===--------------------------------------------------------------------===//
// Common Parser Rules
//===--------------------------------------------------------------------===//

/// intLit    ::= UnsignedInt
///           ::= SignedInt
///           ::= HexLit
///           ::= OctalLit
///           ::= BinaryLit
/// HexLit    ::= '"' 'h' ( '+' | '-' )? ( HexDigit )+ '"'
/// OctalLit  ::= '"' 'o' ( '+' | '-' )? ( OctalDigit )+ '"'
/// BinaryLit ::= '"' 'b' ( '+' | '-' )? ( BinaryDigit )+ '"'
///
ParseResult FIRParser::parseIntLit(APInt &result, const Twine &message) {
  auto spelling = getTokenSpelling();
  switch (getToken().getKind()) {
  case FIRToken::integer:
    if (spelling.getAsInteger(10, result))
      return emitError(message), failure();
    consumeToken(FIRToken::integer);
    return success();

  case FIRToken::string: {
    // Drop the quotes.
    assert(spelling.front() == '"' && spelling.back() == '"');
    spelling = spelling.drop_back().drop_front();

    // Decode the base.
    unsigned base;
    switch (spelling.empty() ? ' ' : spelling.front()) {
    case 'h':
      base = 16;
      break;
    case 'o':
      base = 8;
      break;
    case 'b':
      base = 2;
      break;
    default:
      return emitError("expected base specifier (h/o/b) in integer literal"),
             failure();
    }
    spelling = spelling.drop_front();

    // Handle the optional sign.
    bool isNegative = false;
    if (!spelling.empty() && spelling.front() == '+')
      spelling = spelling.drop_front();
    else if (!spelling.empty() && spelling.front() == '-') {
      isNegative = true;
      spelling = spelling.drop_front();
    }

    // Parse the digits.
    if (spelling.empty())
      return emitError("expected digits in integer literal"), failure();

    if (spelling.getAsInteger(base, result))
      return emitError("invalid character in integer literal"), failure();
    if (isNegative)
      result = -result;
    consumeToken(FIRToken::string);
    return success();
  }

  default:
    return emitError("expected integer literal"), failure();
  }
}

ParseResult FIRParser::parseIntLit(int32_t &result, const Twine &message) {
  APInt value;
  auto loc = getToken().getLoc();
  if (parseIntLit(value, message))
    return failure();

  result = (int32_t)value.getLimitedValue();
  if (result != value)
    return emitError(loc, "value is too big to handle"), failure();
  return success();
}

// optional-width ::= ('<' intLit '>')?
//
// This returns with result equal to -1 if not present.
ParseResult FIRParser::parseOptionalWidth(int32_t &result) {
  if (!consumeIf(FIRToken::less))
    return result = -1, success();

  // Parse a width specifier if present.
  auto widthLoc = getToken().getLoc();
  if (parseIntLit(result, "expected width") ||
      parseToken(FIRToken::greater, "expected >"))
    return failure();

  if (result < 0)
    return emitError(widthLoc, "invalid width specifier"), failure();

  return success();
}

/// id  ::= Id | keywordAsId
///
/// Parse the 'id' grammar, which is an identifier or an allowed keyword.  On
/// success, this returns the identifier in the result attribute.
ParseResult FIRParser::parseId(StringRef &result, const Twine &message) {
  switch (getToken().getKind()) {
  // The most common case is an identifier.
  case FIRToken::identifier:
// Otherwise it may be a keyword that we're allowing in an id position.
#define TOK_KEYWORD(spelling) case FIRToken::kw_##spelling:
#include "FIRTokenKinds.def"

    // Yep, this is a valid 'id'.  Turn it into an attribute.
    result = getTokenSpelling();
    consumeToken();
    return success();

  default:
    emitError(message);
    return failure();
  }
}

ParseResult FIRParser::parseId(StringAttr &result, const Twine &message) {
  StringRef name;
  if (parseId(name, message))
    return failure();

  result = StringAttr::get(name, getContext());
  return success();
}

/// fieldId ::= Id
///         ::= RelaxedId
///         ::= UnsignedInt
///         ::= keywordAsId
///
ParseResult FIRParser::parseFieldId(StringRef &result, const Twine &message) {
  // Handle the UnsignedInt case.
  result = getTokenSpelling();
  if (consumeIf(FIRToken::integer))
    return success();

  // FIXME: Handle RelaxedId

  // Otherwise, it must be Id or keywordAsId.
  if (parseId(result, message))
    return failure();

  return success();
}

/// type ::= 'Clock'
///      ::= 'Reset'
///      ::= 'UInt' optional-width
///      ::= 'SInt' optional-width
///      ::= 'Analog' optional-width
///      ::= {' field* '}'
///      ::= type '[' intLit ']'
///
/// field: 'flip'? fieldId ':' type
///
// FIXME: 'AsyncReset' is also handled by the parser but is not in the spec.
ParseResult FIRParser::parseType(FIRRTLType &result, const Twine &message) {
  switch (getToken().getKind()) {
  default:
    return emitError(message), failure();

  case FIRToken::kw_Clock:
    consumeToken(FIRToken::kw_Clock);
    result = ClockType::get(getContext());
    break;

  case FIRToken::kw_Reset:
    consumeToken(FIRToken::kw_Reset);
    result = ResetType::get(getContext());
    break;

  case FIRToken::kw_UInt:
  case FIRToken::kw_SInt:
  case FIRToken::kw_Analog: {
    auto kind = getToken().getKind();
    consumeToken();

    // Parse a width specifier if present.
    int32_t width;
    if (parseOptionalWidth(width))
      return failure();

    if (kind == FIRToken::kw_SInt)
      result = SIntType::get(getContext(), width);
    else if (kind == FIRToken::kw_UInt)
      result = UIntType::get(getContext(), width);
    else {
      assert(kind == FIRToken::kw_Analog);
      result = AnalogType::get(getContext(), width);
    }
    break;
  }

  case FIRToken::l_brace: {
    consumeToken(FIRToken::l_brace);

    SmallVector<BundleType::BundleElement, 4> elements;
    if (parseListUntil(FIRToken::r_brace, [&]() -> ParseResult {
          bool isFlipped = consumeIf(FIRToken::kw_flip);

          StringRef fieldName;
          FIRRTLType type;
          if (parseFieldId(fieldName, "expected bundle field name") ||
              parseToken(FIRToken::colon, "expected ':' in bundle") ||
              parseType(type, "expected bundle field type"))
            return failure();

          if (isFlipped)
            type = FlipType::get(type);

          elements.push_back({Identifier::get(fieldName, getContext()), type});
          return success();
        }))
      return failure();
    result = BundleType::get(elements, getContext());
    break;
  }
  }

  // Handle postfix vector sizes.
  while (consumeIf(FIRToken::l_square)) {
    auto sizeLoc = getToken().getLoc();
    int32_t size;
    if (parseIntLit(size, "expected width") ||
        parseToken(FIRToken::r_square, "expected ]"))
      return failure();

    if (size < 0)
      return emitError(sizeLoc, "invalid size specifier"), failure();

    result = FVectorType::get(result, size);
  }

  return success();
}

/// ruw ::= 'old' | 'new' | 'undefined'
ParseResult FIRParser::parseOptionalRUW(RUWAttr &result) {
  switch (getToken().getKind()) {
  default:
    break;

  case FIRToken::kw_old:
    result = RUWAttr::Old;
    consumeToken(FIRToken::kw_old);
    break;
  case FIRToken::kw_new:
    result = RUWAttr::New;
    consumeToken(FIRToken::kw_new);
    break;
  case FIRToken::kw_undefined:
    result = RUWAttr::Undefined;
    consumeToken(FIRToken::kw_undefined);
    break;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FIRScopedParser
//===----------------------------------------------------------------------===//

namespace {
/// This class is a common base class for stuff that needs local scope
/// manipulation helpers.
class FIRScopedParser : public FIRParser {
public:
  using SymbolTable =
      llvm::ScopedHashTable<Identifier, std::pair<SMLoc, Value>,
                            DenseMapInfo<Identifier>, llvm::BumpPtrAllocator>;
  using MemoryScopeTable =
      llvm::ScopedHashTable<Identifier,
                            std::pair<SymbolTable::ScopeTy *, Operation *>,
                            DenseMapInfo<Identifier>, llvm::BumpPtrAllocator>;

  FIRScopedParser(GlobalFIRParserState &state, SymbolTable &symbolTable,
                  MemoryScopeTable &memoryScopeTable)
      : FIRParser(state), symbolTable(symbolTable),
        memoryScopeTable(memoryScopeTable) {}

  /// Add a symbol entry with the specified name, returning failure if the name
  /// is already defined.
  ParseResult addSymbolEntry(StringRef name, Value value, SMLoc loc);

  /// Look up the specified name, emitting an error and returning failure if the
  /// name is unknown.
  ParseResult lookupSymbolEntry(Value &result, StringRef name, SMLoc loc);

  /// This symbol table holds the names of ports, wires, and other local decls.
  /// This is scoped because conditional statements introduce subscopes.
  SymbolTable &symbolTable;

  /// Chisel is producing mports with invalid scopes.  To work around the bug,
  /// we need to keep track of the scope (in the `symbolTable`) of each memory.
  /// This keeps track of this.
  MemoryScopeTable &memoryScopeTable;
};
} // end anonymous namespace

/// Add a symbol entry with the specified name, returning failure if the name
/// is already defined.
ParseResult FIRScopedParser::addSymbolEntry(StringRef name, Value value,
                                            SMLoc loc) {
  // TODO(firrtl spec): Should we support name shadowing?  This will reject
  // cases where we try to define a new wire in a conditional where an outer
  // name defined the same name.
  auto nameId = Identifier::get(name, getContext());
  auto prev = symbolTable.lookup(nameId);
  if (prev.first.isValid()) {
    emitError(loc, "redefinition of name '" + name.str() + "'")
            .attachNote(translateLocation(prev.first))
        << "previous definition here";
    return failure();
  }

  symbolTable.insert(nameId, {loc, value});
  return success();
}

/// Look up the specified name, emitting an error and returning null if the
/// name is unknown.
ParseResult FIRScopedParser::lookupSymbolEntry(Value &result, StringRef name,
                                               SMLoc loc) {
  auto prev = symbolTable.lookup(Identifier::get(name, getContext()));
  if (!prev.first.isValid())
    return emitError(loc, "use of unknown declaration '" + name.str() + "'"),
           failure();

  assert(prev.second && "name in symbol table without definition");
  result = prev.second;
  return success();
}

//===----------------------------------------------------------------------===//
// FIRModuleContext
//===----------------------------------------------------------------------===//

namespace {
/// This struct provides context information that is global to the module we're
/// currently parsing into.
struct FIRModuleContext {
  // The expression-oriented nature of firrtl syntax produces tons of constant
  // nodes which are obviously redundant.  Instead of literally producing them
  // in the parser, do an implicit CSE to reduce parse time and silliness in the
  // resulting IR.
  llvm::DenseMap<std::pair<Attribute, Type>, Value> constantCache;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// FIRStmtParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic and state for parsing statements, suites, and
/// similar module body constructs.
struct FIRStmtParser : public FIRScopedParser {
  explicit FIRStmtParser(OpBuilder builder, FIRScopedParser &parentParser,
                         FIRModuleContext &moduleContext)
      : FIRScopedParser(parentParser.getState(), parentParser.symbolTable,
                        parentParser.memoryScopeTable),
        builder(builder), moduleContext(moduleContext) {}

  ParseResult parseSimpleStmt(unsigned stmtIndent);
  ParseResult parseSimpleStmtBlock(unsigned indent);

private:
  using SubOpVector = SmallVectorImpl<Operation *>;

  /// Return the input operand if it has passive type, otherwise convert to
  /// a passive-typed value and return that.
  Value convertToPassive(Value input, Location loc);

  // Exp Parsing
  ParseResult parseExp(Value &result, SubOpVector &subOps,
                       const Twine &message);

  ParseResult parseOptionalExpPostscript(Value &result, SubOpVector &subOps);
  ParseResult parsePostFixFieldId(Value &result, SubOpVector &subOps);
  ParseResult parsePostFixIntSubscript(Value &result, SubOpVector &subOps);
  ParseResult parsePostFixDynamicSubscript(Value &result, SubOpVector &subOps);
  ParseResult parsePrimExp(Value &result, SubOpVector &subOps);
  ParseResult parseIntegerLiteralExp(Value &result, SubOpVector &subOps);

  Optional<ParseResult> parseExpWithLeadingKeyword(StringRef keyword,
                                                   const LocWithInfo &info);

  // Stmt Parsing
  ParseResult parseAttach();
  ParseResult parseMemPort(MemDirAttr direction);
  ParseResult parsePrintf();
  ParseResult parseSkip();
  ParseResult parseStop();
  ParseResult parseWhen(unsigned whenIndent);
  ParseResult parseLeadingExpStmt(Value lhs, SubOpVector &subOps);

  // Declarations
  ParseResult parseInstance();
  ParseResult parseCMem();
  ParseResult parseSMem();
  ParseResult parseMem(unsigned memIndent);
  ParseResult parseNode();
  ParseResult parseWire();
  ParseResult parseRegister(unsigned regIndent);

  // The builder to build into.
  OpBuilder builder;

  // Extra information maintained across a module.
  FIRModuleContext &moduleContext;
};

} // end anonymous namespace

/// Return the input operand if it has passive type, otherwise convert to
/// a passive-typed value and return that.
Value FIRStmtParser::convertToPassive(Value input, Location loc) {
  auto inType = input.getType().cast<FIRRTLType>();
  if (inType.isPassiveType())
    return input;

  return builder.create<AsPassivePrimOp>(loc, inType.getPassiveType(), input);
}

//===-------------------------------
// FIRStmtParser Expression Parsing.

/// Parse the 'exp' grammar, returning all of the suboperations in the
/// specified vector, and the ultimate SSA value in value.
///
///  exp ::= id    // Ref
///      ::= prim
///      ::= integer-literal-exp
///      ::= exp '.' fieldId
///      ::= exp '[' intLit ']'
/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
///      ::= exp '[' exp ']'
///
ParseResult FIRStmtParser::parseExp(Value &result, SubOpVector &subOps,
                                    const Twine &message) {
  switch (getToken().getKind()) {

    // Handle all the primitive ops: primop exp* intLit*  ')'
#define TOK_LPKEYWORD(SPELLING) case FIRToken::lp_##SPELLING:
#include "FIRTokenKinds.def"
    if (parsePrimExp(result, subOps))
      return failure();
    break;

  case FIRToken::kw_UInt:
  case FIRToken::kw_SInt:
    if (parseIntegerLiteralExp(result, subOps))
      return failure();
    break;

    // Otherwise there are a bunch of keywords that are treated as identifiers
    // try them.
  case FIRToken::identifier: // exp ::= id
  default: {
    StringRef name;
    auto loc = getToken().getLoc();
    if (parseId(name, message) || lookupSymbolEntry(result, name, loc))
      return failure();
    break;
  }
  }

  return parseOptionalExpPostscript(result, subOps);
}

/// Parse the postfix productions of expression after the leading expression
/// has been parsed.
///
///  exp ::= exp '.' fieldId
///      ::= exp '[' intLit ']'
/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
///      ::= exp '[' exp ']'
ParseResult FIRStmtParser::parseOptionalExpPostscript(Value &result,
                                                      SubOpVector &subOps) {

  // Handle postfix expressions.
  while (1) {
    // Subfield: exp ::= exp '.' fieldId
    if (consumeIf(FIRToken::period)) {
      if (parsePostFixFieldId(result, subOps))
        return failure();

      continue;
    }

    // Subindex: exp ::= exp '[' intLit ']' | exp '[' exp ']'
    if (consumeIf(FIRToken::l_square)) {
      if (getToken().isAny(FIRToken::integer, FIRToken::string)) {
        if (parsePostFixIntSubscript(result, subOps))
          return failure();
        continue;
      }
      if (parsePostFixDynamicSubscript(result, subOps))
        return failure();

      continue;
    }

    return success();
  }
}

/// exp ::= exp '.' fieldId
///
/// The "exp '.'" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixFieldId(Value &result,
                                               SubOpVector &subOps) {
  auto loc = getToken().getLoc();
  StringRef fieldName;
  if (parseFieldId(fieldName, "expected field name"))
    return failure();

  // Make sure the field name matches up with the input value's type and
  // compute the result type for the expression.
  auto resultType = result.getType().cast<FIRRTLType>();
  resultType = SubfieldOp::getResultType(resultType, fieldName);
  if (!resultType) {
    // TODO(QoI): This error would be nicer with a .fir pretty print of the
    // type.
    emitError(loc, "invalid subfield '" + fieldName + "' for type ")
        << result.getType();
    return failure();
  }

  // Create the result operation.
  auto op =
      builder.create<SubfieldOp>(translateLocation(loc), resultType, result,
                                 builder.getStringAttr(fieldName));
  subOps.push_back(op);
  result = op.getResult();
  return success();
}

/// exp ::= exp '[' intLit ']'
///
/// The "exp '['" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixIntSubscript(Value &result,
                                                    SubOpVector &subOps) {
  auto indexLoc = getToken().getLoc();
  int32_t indexNo;
  if (parseIntLit(indexNo, "expected index") ||
      parseToken(FIRToken::r_square, "expected ']'"))
    return failure();

  if (indexNo < 0)
    return emitError(indexLoc, "invalid index specifier"), failure();

  // Make sure the index expression is valid and compute the result type for the
  // expression.
  auto resultType = result.getType().cast<FIRRTLType>();
  resultType = SubindexOp::getResultType(resultType, indexNo);
  if (!resultType) {
    // TODO(QoI): This error would be nicer with a .fir pretty print of the
    // type.
    emitError(indexLoc, "invalid subindex '" + Twine(indexNo))
        << "' for type " << result.getType();
    return failure();
  }

  // Create the result operation.
  auto op =
      builder.create<SubindexOp>(translateLocation(indexLoc), resultType,
                                 result, builder.getI32IntegerAttr(indexNo));
  subOps.push_back(op);
  result = op.getResult();
  return success();
}

/// exp ::= exp '[' exp ']'
///
/// The "exp '['" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixDynamicSubscript(Value &result,
                                                        SubOpVector &subOps) {
  auto indexLoc = getToken().getLoc();
  Value index;
  if (parseExp(index, subOps, "expected subscript index expression") ||
      parseToken(FIRToken::r_square, "expected ']' in subscript"))
    return failure();

  // Make sure the index expression is valid and compute the result type for the
  // expression.
  auto resultType = result.getType().cast<FIRRTLType>();
  resultType = SubaccessOp::getResultType(resultType,
                                          index.getType().cast<FIRRTLType>());
  if (!resultType) {
    // TODO(QoI): This error would be nicer with a .fir pretty print of the
    // type.
    emitError(indexLoc, "invalid dynamic subaccess into ")
        << result.getType() << " with index of type " << index.getType();
    return failure();
  }

  // Create the result operation.
  auto op = builder.create<SubaccessOp>(translateLocation(indexLoc), resultType,
                                        result, index);
  subOps.push_back(op);
  result = op.getResult();
  return success();
}

/// prim ::= primop exp* intLit*  ')'
ParseResult FIRStmtParser::parsePrimExp(Value &result, SubOpVector &subOps) {
  auto kind = getToken().getKind();
  auto loc = getToken().getLoc();
  consumeToken();

  // Parse the operands and constant integer arguments.
  SmallVector<Value, 4> operands;
  SmallVector<int32_t, 4> integers;
  if (parseListUntil(FIRToken::r_paren, [&]() -> ParseResult {
        // Handle the integer constant case if present.
        if (getToken().isAny(FIRToken::integer, FIRToken::string)) {
          integers.push_back(0);
          return parseIntLit(integers.back(), "expected integer");
        }

        // Otherwise it must be a value operand.  These must all come before the
        // integers.
        if (!integers.empty())
          return emitError("expected more integer constants"), failure();

        Value operand;
        if (parseExp(operand, subOps,
                     "expected expression in primitive operand"))
          return failure();

        operands.push_back(operand);
        return success();
      }))
    return failure();

  SmallVector<FIRRTLType, 4> opTypes;
  for (auto v : operands)
    opTypes.push_back(v.getType().cast<FIRRTLType>().getPassiveType());

  auto typeError = [&](StringRef opName) -> ParseResult {
    auto diag = emitError(loc, "invalid input types for '") << opName << "': ";
    bool isFirst = true;
    for (auto t : opTypes) {
      if (!isFirst)
        diag << ", ";
      else
        isFirst = false;
      diag << t;
    }
    return failure();
  };

  SmallVector<StringRef, 2> attrNames;
  switch (kind) {
  default:
    break;
  case FIRToken::lp_bits:
    attrNames.push_back("hi");
    attrNames.push_back("lo");
    break;
  case FIRToken::lp_head:
  case FIRToken::lp_pad:
  case FIRToken::lp_shl:
  case FIRToken::lp_shr:
  case FIRToken::lp_tail:
    attrNames.push_back("amount");
    break;
  }

  if (integers.size() != attrNames.size()) {
    emitError(loc,
              "expected " + Twine(attrNames.size()) + " constant arguments");
    return failure();
  }

  SmallVector<NamedAttribute, 2> attrs;
  for (size_t i = 0, e = attrNames.size(); i != e; ++i)
    attrs.push_back(builder.getNamedAttr(
        attrNames[i], builder.getI32IntegerAttr(integers[i])));

  switch (kind) {
  default:
    emitError(loc, "primitive not supported yet");
    return failure();

#define TOK_LPKEYWORD_PRIM(SPELLING, CLASS)                                    \
  case FIRToken::lp_##SPELLING: {                                              \
    auto resultTy = CLASS::getResultType(opTypes, integers);                   \
    if (!resultTy)                                                             \
      return typeError(#SPELLING);                                             \
    result = builder.create<CLASS>(translateLocation(loc), resultTy,           \
                                   ValueRange(operands), attrs);               \
    break;                                                                     \
  }
#include "FIRTokenKinds.def"
  }

  subOps.push_back(result.getDefiningOp());
  return success();
}

/// integer-literal-exp ::= 'UInt' optional-width '(' intLit ')'
///                     ::= 'SInt' optional-width '(' intLit ')'
ParseResult FIRStmtParser::parseIntegerLiteralExp(Value &result,
                                                  SubOpVector &subOps) {
  bool isSigned = getToken().is(FIRToken::kw_SInt);
  auto loc = getToken().getLoc();
  consumeToken();

  // Parse a width specifier if present.
  int32_t width;
  APInt value;
  if (parseOptionalWidth(width) ||
      parseToken(FIRToken::l_paren, "expected '(' in integer expression") ||
      parseIntLit(value, "expected integer value") ||
      parseToken(FIRToken::r_paren, "expected ')' in integer expression"))
    return failure();

  // Construct an integer attribute of the right width.
  auto type = IntType::get(builder.getContext(), isSigned, width);

  // TODO: check to see if this extension stuff is necessary.  We should at
  // least warn for things like UInt<4>(255).
  IntegerType::SignednessSemantics signedness;
  if (type.isSigned()) {
    signedness = IntegerType::Signed;
    if (width != -1)
      value = value.sextOrTrunc(width);
  } else {
    signedness = IntegerType::Unsigned;
    if (width != -1)
      value = value.zextOrTrunc(width);
  }

  Type attrType =
      IntegerType::get(value.getBitWidth(), signedness, type.getContext());
  auto attr = builder.getIntegerAttr(attrType, value);

  // Check to see if we've already created this constant.  If so, reuse it.
  auto &entry = moduleContext.constantCache[{attr, type}];
  if (entry) {
    // If we already had an entry, reuse it.
    result = entry;
    return success();
  }

  // Make sure to insert constants at the top level of the module to maintain
  // dominance.
  OpBuilder::InsertPoint savedIP;

  auto parentOp = builder.getInsertionBlock()->getParentOp();
  if (!isa<FModuleOp>(parentOp)) {
    savedIP = builder.saveInsertionPoint();
    while (!isa<FModuleOp>(parentOp)) {
      builder.setInsertionPoint(parentOp);
      parentOp = builder.getInsertionBlock()->getParentOp();
    }
  }

  auto op = builder.create<ConstantOp>(translateLocation(loc), type, value);
  entry = op;
  subOps.push_back(op);
  result = op;

  if (savedIP.isSet())
    builder.setInsertionPoint(savedIP.getBlock(), savedIP.getPoint());
  return success();
}

/// The .fir grammar has the annoying property where:
/// 1) some statements start with keywords
/// 2) some start with an expression
/// 3) it allows the 'reference' expression to either be an identifier or a
///    keyword.
///
/// One example of this is something like, where this is not a register decl:
///   reg <- thing
///
/// Solving this requires lookahead to the second token.  We handle it by
///  factoring the lookahead inline into the code to keep the parser fast.
///
/// As such, statements that start with a leading keyword call this method to
/// check to see if the keyword they consumed was actually the start of an
/// expression.  If so, they parse the expression-based statement and return the
/// parser result.  If not, they return None and the statement is parsed like
/// normal.
Optional<ParseResult>
FIRStmtParser::parseExpWithLeadingKeyword(StringRef keyword,
                                          const LocWithInfo &info) {

  switch (getToken().getKind()) {
  default:
    // This isn't part of an expression, and isn't part of a statement.
    return None;

  case FIRToken::period:     // exp `.` identifier
  case FIRToken::l_square:   // exp `[` index `]`
  case FIRToken::kw_is:      // exp is invalid
  case FIRToken::less_equal: // exp <= thing
  case FIRToken::less_minus: // exp <- thing
    break;
  }

  Value lhs;
  SmallVector<Operation *, 8> subOps;
  if (lookupSymbolEntry(lhs, keyword, info.getFIRLoc()) ||
      parseOptionalExpPostscript(lhs, subOps))
    return ParseResult(failure());

  return parseLeadingExpStmt(lhs, subOps);
}
//===-----------------------------
// FIRStmtParser Statement Parsing

/// simple_stmt_block ::= simple_stmt*
ParseResult FIRStmtParser::parseSimpleStmtBlock(unsigned indent) {
  while (true) {
    // The outer level parser can handle these tokens.
    if (getToken().isAny(FIRToken::eof, FIRToken::error))
      return success();

    auto subIndent = getIndentation();
    if (!subIndent.hasValue())
      return emitError("expected statement to be on its own line"), failure();

    if (subIndent.getValue() <= indent)
      return success();

    // Let the statement parser handle this.
    if (parseSimpleStmt(subIndent.getValue()))
      return failure();
  }
}

/// simple_stmt ::= stmt
///
/// stmt ::= attach
///      ::= memport
///      ::= printf
///      ::= skip
///      ::= stop
///      ::= when
///      ::= leading-exp-stmt
///
/// stmt ::= instance
///      ::= cmem | smem | mem
///      ::= node | wire
///      ::= register
///
ParseResult FIRStmtParser::parseSimpleStmt(unsigned stmtIndent) {
  switch (getToken().getKind()) {
  // Statements.
  case FIRToken::kw_attach:
    return parseAttach();
  case FIRToken::kw_infer:
    return parseMemPort(MemDirAttr::Infer);
  case FIRToken::kw_read:
    return parseMemPort(MemDirAttr::Read);
  case FIRToken::kw_write:
    return parseMemPort(MemDirAttr::Write);
  case FIRToken::kw_rdwr:
    return parseMemPort(MemDirAttr::ReadWrite);
  case FIRToken::lp_printf:
    return parsePrintf();
  case FIRToken::kw_skip:
    return parseSkip();
  case FIRToken::lp_stop:
    return parseStop();
  case FIRToken::kw_when:
    return parseWhen(stmtIndent);
  default: {
    // Statement productions that start with an expression.
    Value lhs;
    SmallVector<Operation *, 8> subOps;
    if (parseExp(lhs, subOps, "unexpected token in module"))
      return failure();
    return parseLeadingExpStmt(lhs, subOps);
  }

    // Declarations
  case FIRToken::kw_inst:
    return parseInstance();
  case FIRToken::kw_cmem:
    return parseCMem();
  case FIRToken::kw_smem:
    return parseSMem();
  case FIRToken::kw_mem:
    return parseMem(stmtIndent);
  case FIRToken::kw_node:
    return parseNode();
  case FIRToken::kw_wire:
    return parseWire();
  case FIRToken::kw_reg:
    return parseRegister(stmtIndent);
  }
}

/// attach ::= 'attach' '(' exp+ ')' info?
ParseResult FIRStmtParser::parseAttach() {
  auto spelling = getTokenSpelling();
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_attach);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(spelling, info))
    return isExpr.getValue();

  if (parseToken(FIRToken::l_paren, "expected '(' after attach"))
    return failure();

  SmallVector<Operation *, 8> subOps;
  SmallVector<Value, 4> operands;
  do {
    operands.push_back({});
    if (parseExp(operands.back(), subOps, "expected operand in attach"))
      return failure();
  } while (!consumeIf(FIRToken::r_paren));

  if (parseOptionalInfo(info, subOps))
    return failure();

  builder.create<AttachOp>(info.getLoc(), operands);
  return success();
}

/// stmt ::= mdir 'mport' id '=' id '[' exp ']' exp info?
/// mdir ::= 'infer' | 'read' | 'write' | 'rdwr'
///
ParseResult FIRStmtParser::parseMemPort(MemDirAttr direction) {
  auto spelling = getTokenSpelling();
  auto mdirIndent = getIndentation();
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken();

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(spelling, info))
    return isExpr.getValue();

  StringAttr resultValue;
  StringRef memName;
  Value memory, indexExp, clock;
  SmallVector<Operation *, 8> subOps;
  if (parseToken(FIRToken::kw_mport, "expected 'mport' in memory port") ||
      parseId(resultValue, "expected result name") ||
      parseToken(FIRToken::equal, "expected '=' in memory port") ||
      parseId(memName, "expected memory name") ||
      lookupSymbolEntry(memory, memName, info.getFIRLoc()) ||
      parseToken(FIRToken::l_square, "expected '[' in memory port") ||
      parseExp(indexExp, subOps, "expected index expression") ||
      parseToken(FIRToken::r_square, "expected ']' in memory port") ||
      parseExp(clock, subOps, "expected clock expression") ||
      parseOptionalInfo(info, subOps))
    return failure();

  auto memVType = memory.getType().dyn_cast<FVectorType>();
  if (!memVType)
    return emitError(info.getFIRLoc(), "memory should have vector type");
  auto resultType = memVType.getElementType();

  auto result = builder.create<MemoryPortOp>(info.getLoc(), resultType, memory,
                                             indexExp, clock, direction,
                                             filterUselessName(resultValue));

  // TODO(firrtl scala bug): If the next operation is a skip, just eat it if it
  // is at the same indent level as us.  This is a horrible hack on top of the
  // following hack to work around a Scala bug.
  auto nextIndent = getIndentation();
  if (getToken().is(FIRToken::kw_skip) && mdirIndent.hasValue() &&
      nextIndent.hasValue() && mdirIndent.getValue() == nextIndent.getValue()) {
    if (parseSkip())
      return failure();

    nextIndent = getIndentation();
  }

  // TODO(firrtl scala bug): Chisel is creating invalid IR where the mports
  // are logically defining their name at the scope of the memory itself.  This
  // is a problem for us, because the index expression and clock may be defined
  // in a nested expression.  We can't really tell when this is happening
  //  without unbounded lookahead either.
  //
  // Fortunately, this seems to happen in very idiomatic cases, where the
  // mport happens at the end of a when block.  We detect this situation by
  // seeing if the next statement is indented less than our memport - if so,
  // this is the last statement at the end of the 'when' block.   Trigger a
  // hacky workaround just in this case.
  if (mdirIndent.hasValue() && nextIndent.hasValue() &&
      mdirIndent.getValue() > nextIndent.getValue()) {
    auto memNameId = Identifier::get(memName, getContext());

    // To make this even more gross, we have no efficient way to figure out
    // what scope a value lives in our scoped hash table.  We keep a shadow
    // table to track this.
    auto scopeAndOperation = memoryScopeTable.lookup(memNameId);
    if (!scopeAndOperation.first) {
      emitError(info.getFIRLoc(), "unknown memory '") << memNameId << "'";
      return failure();
    }

    // If we need to inject this name into a parent scope, then we have to do
    // some IR hackery.  Create a wire for the resultValue name right before
    // the mem in question, inject its name into that scope, then connect
    // the output of the mport to it.
    if (scopeAndOperation.first != symbolTable.getCurScope()) {
      OpBuilder memOpBuilder(scopeAndOperation.second);

      auto wireHack = memOpBuilder.create<WireOp>(
          info.getLoc(), result.getType(), StringAttr());
      builder.create<ConnectOp>(info.getLoc(), wireHack, result);

      // Inject this the wire's name into the same scope as the memory.
      symbolTable.insertIntoScope(
          scopeAndOperation.first,
          Identifier::get(resultValue.getValue(), getContext()),
          {info.getFIRLoc(), wireHack});
      return success();
    }
  }

  return addSymbolEntry(resultValue.getValue(), result, info.getFIRLoc());
}

/// printf ::= 'printf(' exp exp StringLit exp* ')' info?
ParseResult FIRStmtParser::parsePrintf() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::lp_printf);

  SmallVector<Operation *, 8> subOps;

  Value clock, condition;
  StringRef formatString;
  if (parseExp(clock, subOps, "expected clock expression in printf") ||
      parseExp(condition, subOps, "expected condition in printf") ||
      parseGetSpelling(formatString) ||
      parseToken(FIRToken::string, "expected format string in printf"))
    return failure();

  SmallVector<Value, 4> operands;
  while (!consumeIf(FIRToken::r_paren)) {
    operands.push_back({});
    if (parseExp(operands.back(), subOps, "expected operand in printf"))
      return failure();
  }

  if (parseOptionalInfo(info, subOps))
    return failure();

  auto formatStrUnescaped = FIRToken::getStringValue(formatString);
  builder.create<PrintFOp>(info.getLoc(), clock, condition,
                           builder.getStringAttr(formatStrUnescaped), operands);
  return success();
}

/// skip ::= 'skip' info?
ParseResult FIRStmtParser::parseSkip() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_skip);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("skip", info))
    return isExpr.getValue();

  if (parseOptionalInfo(info))
    return failure();

  builder.create<SkipOp>(info.getLoc());
  return success();
}

/// stop ::= 'stop(' exp exp intLit ')' info?
ParseResult FIRStmtParser::parseStop() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::lp_stop);

  SmallVector<Operation *, 8> subOps;

  Value clock, condition;
  int32_t exitCode;
  if (parseExp(clock, subOps, "expected clock expression in 'stop'") ||
      parseExp(condition, subOps, "expected condition in 'stop'") ||
      parseIntLit(exitCode, "expected exit code in 'stop'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'stop'") ||
      parseOptionalInfo(info, subOps))
    return failure();

  builder.create<StopOp>(info.getLoc(), clock, condition,
                         builder.getI32IntegerAttr(exitCode));
  return success();
}

/// when  ::= 'when' exp ':' info? suite? ('else' ( when | ':' info? suite?)
/// )? suite ::= simple_stmt | INDENT simple_stmt+ DEDENT
ParseResult FIRStmtParser::parseWhen(unsigned whenIndent) {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_when);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("when", info))
    return isExpr.getValue();

  Value condition;
  SmallVector<Operation *, 8> subOps;
  if (parseExp(condition, subOps, "expected condition in 'when'") ||
      parseToken(FIRToken::colon, "expected ':' in when") ||
      parseOptionalInfo(info, subOps))
    return failure();

  condition = convertToPassive(condition, info.getLoc());

  // Create the IR representation for the when.
  auto whenStmt = builder.create<WhenOp>(info.getLoc(), condition,
                                         /*createElse*/ false);

  // This is a function to parse a suite body.
  auto parseSuite = [&](OpBuilder builder) -> ParseResult {
    // Declarations within the suite are scoped to within the suite.
    SymbolTable::ScopeTy suiteScope(symbolTable);
    MemoryScopeTable::ScopeTy suiteMemoryScope(memoryScopeTable);

    // We parse the substatements into their own parser, so they get insert
    // into the specified 'when' region.
    FIRStmtParser subParser(builder, *this, moduleContext);

    // Figure out whether the body is a single statement or a nested one.
    auto stmtIndent = getIndentation();

    // Parsing a single statment is straightforward.
    if (!stmtIndent.hasValue())
      return subParser.parseSimpleStmt(whenIndent);

    if (stmtIndent.getValue() <= whenIndent)
      return emitError("statement must be indented more than 'when'"),
             failure();

    // Parse a block of statements that are indented more than the when.
    return subParser.parseSimpleStmtBlock(whenIndent);
  };

  // Parse the 'then' body into the 'then' region.
  if (parseSuite(whenStmt.getThenBodyBuilder()))
    return failure();

  // If the else is present, handle it otherwise we're done.
  if (getToken().isNot(FIRToken::kw_else))
    return success();

  // If the 'else' is less indented than the when, then it must belong to some
  // containing 'when'.
  auto elseIndent = getIndentation();
  if (elseIndent.hasValue() && elseIndent.getValue() < whenIndent)
    return success();

  LocWithInfo elseInfo(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_else);

  // Create an else block to parse into.
  whenStmt.createElseRegion();

  // If we have the ':' form, then handle it.
  if (getToken().is(FIRToken::kw_when)) {
    // TODO(completeness): Handle the 'else when' syntactic sugar when we
    // care.
    return emitError("'else when' syntax not supported yet"), failure();
  }

  // Parse the 'else' body into the 'else' region.
  if (parseToken(FIRToken::colon, "expected ':' after 'else'") ||
      parseOptionalInfo(elseInfo) || parseSuite(whenStmt.getElseBodyBuilder()))
    return failure();

  // TODO(firrtl spec): There is no reason for the 'else :' grammar to take an
  // info.  It doesn't appear to be generated either.
  return success();
}

/// leading-exp-stmt ::= exp '<=' exp info?
///                  ::= exp '<-' exp info?
///                  ::= exp 'is' 'invalid' info?
ParseResult FIRStmtParser::parseLeadingExpStmt(Value lhs, SubOpVector &subOps) {
  // Figure out which kind of statement it is.
  LocWithInfo info(getToken().getLoc(), this);

  // If 'is' grammar is special.
  if (consumeIf(FIRToken::kw_is)) {
    if (parseToken(FIRToken::kw_invalid, "expected 'invalid'") ||
        parseOptionalInfo(info, subOps))
      return failure();

    builder.create<InvalidOp>(info.getLoc(), lhs);
    return success();
  }

  auto kind = getToken().getKind();
  if (getToken().isNot(FIRToken::less_equal, FIRToken::less_minus))
    return emitError("expected '<=', '<-', or 'is' in statement"), failure();
  consumeToken();

  Value rhs;
  if (parseExp(rhs, subOps, "unexpected token in statement") ||
      parseOptionalInfo(info, subOps))
    return failure();

  if (kind == FIRToken::less_equal)
    builder.create<ConnectOp>(info.getLoc(), lhs, rhs);
  else {
    assert(kind == FIRToken::less_minus && "unexpected kind");
    builder.create<PartialConnectOp>(info.getLoc(), lhs, rhs);
  }
  return success();
}

//===-------------------------------
// FIRStmtParser Declaration Parsing

/// instance ::= 'inst' id 'of' id info?
ParseResult FIRStmtParser::parseInstance() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_inst);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("inst", info))
    return isExpr.getValue();

  StringAttr id;
  StringRef moduleName;
  if (parseId(id, "expected instance name") ||
      parseToken(FIRToken::kw_of, "expected 'of' in instance") ||
      parseId(moduleName, "expected module name") || parseOptionalInfo(info))
    return failure();

  // Look up the module that is being referenced.
  auto circuit =
      builder.getBlock()->getParentOp()->getParentOfType<CircuitOp>();
  auto referencedModule = circuit.lookupSymbol(moduleName);
  if (!referencedModule) {
    emitError(info.getFIRLoc(),
              "use of undefined module name '" + moduleName + "' in instance");
    return failure();
  }

  SmallVector<ModulePortInfo, 4> modulePorts;
  getModulePortInfo(referencedModule, modulePorts);

  // Make a bundle of the inputs and outputs of the specified module.
  SmallVector<BundleType::BundleElement, 4> results;
  results.reserve(modulePorts.size());

  for (auto port : modulePorts) {
    results.push_back({builder.getIdentifier(port.first.getValue()),
                       FlipType::get(port.second)});
  }
  auto resultType = BundleType::get(results, getContext());
  auto result = builder.create<InstanceOp>(info.getLoc(), resultType,
                                           builder.getSymbolRefAttr(moduleName),
                                           filterUselessName(id));

  return addSymbolEntry(id.getValue(), result, info.getFIRLoc());
}

/// cmem ::= 'cmem' id ':' type info?
ParseResult FIRStmtParser::parseCMem() {
  // TODO(firrtl spec) cmem is completely undocumented.
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_cmem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("cmem", info))
    return isExpr.getValue();

  StringAttr id;
  FIRRTLType type;
  if (parseId(id, "expected cmem name") ||
      parseToken(FIRToken::colon, "expected ':' in cmem") ||
      parseType(type, "expected cmem type") || parseOptionalInfo(info))
    return failure();

  auto result =
      builder.create<CMemOp>(info.getLoc(), type, filterUselessName(id));

  // Remember that this memory is in this symbol table scope.
  // TODO(chisel bug): This should be removed along with memoryScopeTable.
  memoryScopeTable.insert(Identifier::get(id.getValue(), getContext()),
                          {symbolTable.getCurScope(), result.getOperation()});

  return addSymbolEntry(id.getValue(), result, info.getFIRLoc());
}

/// smem ::= 'smem' id ':' type ruw? info?
ParseResult FIRStmtParser::parseSMem() {
  // TODO(firrtl spec) smem is completely undocumented.
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_smem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("smem", info))
    return isExpr.getValue();

  StringAttr id;
  FIRRTLType type;
  RUWAttr ruw = RUWAttr::Undefined;

  if (parseId(id, "expected cmem name") ||
      parseToken(FIRToken::colon, "expected ':' in cmem") ||
      parseType(type, "expected cmem type") || parseOptionalRUW(ruw) ||
      parseOptionalInfo(info))
    return failure();

  auto result =
      builder.create<SMemOp>(info.getLoc(), type, ruw, filterUselessName(id));

  // Remember that this memory is in this symbol table scope.
  // TODO(chisel bug): This should be removed along with memoryScopeTable.
  memoryScopeTable.insert(Identifier::get(id.getValue(), getContext()),
                          {symbolTable.getCurScope(), result.getOperation()});

  return addSymbolEntry(id.getValue(), result, info.getFIRLoc());
}

/// mem ::= 'mem' id ':' info? INDENT memField* DEDENT
/// memField ::= 'data-type' '=>' type NEWLINE
/// 	       ::= 'depth' '=>' intLit NEWLINE
/// 	       ::= 'read-latency' '=>' intLit NEWLINE
/// 	       ::= 'write-latency' '=>' intLit NEWLINE
/// 	       ::= 'read-under-write' '=>' ruw NEWLINE
/// 	       ::= 'reader' '=>' id+ NEWLINE
/// 	       ::= 'writer' '=>' id+ NEWLINE
/// 	       ::= 'readwriter' '=>' id+ NEWLINE
ParseResult FIRStmtParser::parseMem(unsigned memIndent) {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_mem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("mem", info))
    return isExpr.getValue();

  StringAttr id;
  if (parseId(id, "expected mem name") ||
      parseToken(FIRToken::colon, "expected ':' in mem") ||
      parseOptionalInfo(info))
    return failure();

  FIRRTLType type;
  int32_t depth = -1, readLatency = -1, writeLatency = -1;
  RUWAttr ruw = RUWAttr::Undefined;

  enum PortKind { Read, Write, ReadWrite };
  SmallVector<std::pair<StringRef, PortKind>, 4> ports;

  // Parse all the memfield records, which are indented more than the mem.
  while (1) {
    auto nextIndent = getIndentation();
    if (!nextIndent.hasValue() || nextIndent.getValue() <= memIndent)
      break;

    auto spelling = getTokenSpelling();
    if (parseToken(FIRToken::identifier, "unexpected token in 'mem'") ||
        parseToken(FIRToken::equal_greater, "expected '=>' in 'mem'"))
      return failure();

    if (spelling == "data-type") {
      if (type)
        return emitError("'mem' type specified multiple times"), failure();

      if (parseType(type, "expected type in data-type declaration"))
        return failure();
      continue;
    }
    if (spelling == "depth") {
      if (parseIntLit(depth, "expected integer in depth specification"))
        return failure();
      continue;
    }
    if (spelling == "read-latency") {
      if (parseIntLit(readLatency, "expected integer latency"))
        return failure();
      continue;
    }
    if (spelling == "write-latency") {
      if (parseIntLit(writeLatency, "expected integer latency"))
        return failure();
      continue;
    }
    if (spelling == "read-under-write") {
      if (getToken().isNot(FIRToken::kw_old, FIRToken::kw_new,
                           FIRToken::kw_undefined))
        return emitError("expected specifier"), failure();

      if (parseOptionalRUW(ruw))
        return failure();
      continue;
    }

    PortKind portKind;
    if (spelling == "reader")
      portKind = Read;
    else if (spelling == "writer")
      portKind = Write;
    else if (spelling == "readwriter")
      portKind = ReadWrite;
    else
      return emitError("unexpected field in 'mem' declaration"), failure();

    StringRef portName;
    if (parseId(portName, "expected port name"))
      return failure();
    ports.push_back({portName, portKind});

    while (!getIndentation().hasValue()) {
      if (parseId(portName, "expected port name"))
        return failure();
      ports.push_back({portName, portKind});
    }
  }

  // Okay, now that we parsed all the things, do some validation to make sure
  // that things are more or less sensible.
  llvm::array_pod_sort(ports.begin(), ports.end(),
                       [](const std::pair<StringRef, PortKind> *lhs,
                          const std::pair<StringRef, PortKind> *rhs) -> int {
                         return lhs->first.compare(rhs->first);
                       });

  // Reject duplicate ports.
  for (size_t i = 1, e = ports.size(); i < e; ++i)
    if (ports[i - 1].first == ports[i].first) {
      emitError(info.getFIRLoc(), "duplicate port name ") << ports[i].first;
      return failure();
    }

  if (!type.isPassiveType()) {
    emitError(info.getFIRLoc(), "'mem' data-type must be a passive type");
    return failure();
  }

  if (depth < 1)
    return emitError(info.getFIRLoc(), "invalid depth");

  if (readLatency < 0 || writeLatency < 0)
    return emitError(info.getFIRLoc(), "invalid latency");

  // Figure out the number of bits needed for the address, and thus the address
  // type to use.
  auto addressType = UIntType::get(getContext(), llvm::Log2_32_Ceil(depth));

  // Okay, we've validated the data, construct the result type.
  SmallVector<BundleType::BundleElement, 4> memFields;
  SmallVector<BundleType::BundleElement, 5> portFields;
  // Common fields for all port types.
  portFields.push_back({builder.getIdentifier("addr"), addressType});
  portFields.push_back(
      {builder.getIdentifier("en"), UIntType::get(getContext(), 1)});
  portFields.push_back(
      {builder.getIdentifier("clk"), ClockType::get(getContext())});

  for (auto port : ports) {
    // Reuse the first three fields, but drop the rest.
    portFields.erase(portFields.begin() + 3, portFields.end());
    switch (port.second) {
    case Read:
      portFields.push_back(
          {builder.getIdentifier("data"), FlipType::get(type)});
      break;

    case Write:
      portFields.push_back({builder.getIdentifier("data"), type});
      portFields.push_back({builder.getIdentifier("mask"), type.getMaskType()});
      break;
    case ReadWrite:
      portFields.push_back(
          {builder.getIdentifier("wmode"), UIntType::get(getContext(), 1)});
      portFields.push_back(
          {builder.getIdentifier("rdata"), FlipType::get(type)});
      portFields.push_back({builder.getIdentifier("wdata"), type});
      portFields.push_back(
          {builder.getIdentifier("wmask"), type.getMaskType()});
      break;
    }

    memFields.push_back({builder.getIdentifier(port.first),
                         BundleType::get(portFields, getContext())});
  }

  auto memType = BundleType::get(memFields, getContext());
  auto result = builder.create<MemOp>(
      info.getLoc(), memType, llvm::APInt(32, readLatency),
      llvm::APInt(32, writeLatency), llvm::APInt(32, depth), ruw,
      filterUselessName(id));

  // Remember that this memory is in this symbol table scope.
  // TODO(chisel bug): This should be removed along with memoryScopeTable.
  memoryScopeTable.insert(Identifier::get(id.getValue(), getContext()),
                          {symbolTable.getCurScope(), result.getOperation()});

  return addSymbolEntry(id.getValue(), result, info.getFIRLoc());
}

/// node ::= 'node' id '=' exp info?
ParseResult FIRStmtParser::parseNode() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_node);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("node", info))
    return isExpr.getValue();

  StringAttr id;
  Value initializer;
  SmallVector<Operation *, 8> subOps;
  if (parseId(id, "expected node name") ||
      parseToken(FIRToken::equal, "expected '=' in node") ||
      parseExp(initializer, subOps, "expected expression for node") ||
      parseOptionalInfo(info, subOps))
    return failure();

  // Ignore useless names like _T.
  auto actualName = filterUselessName(id);

  // The entire point of a node declaration is to carry a name.  If it got
  // dropped, then we don't even need to create a result.
  Value result;
  if (actualName)
    result = builder.create<NodeOp>(info.getLoc(), initializer, actualName);
  else
    result = initializer;
  return addSymbolEntry(id.getValue(), result, info.getFIRLoc());
}

/// wire ::= 'wire' id ':' type info?
ParseResult FIRStmtParser::parseWire() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_wire);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("wire", info))
    return isExpr.getValue();

  StringAttr id;
  FIRRTLType type;
  if (parseId(id, "expected wire name") ||
      parseToken(FIRToken::colon, "expected ':' in wire") ||
      parseType(type, "expected wire type") || parseOptionalInfo(info))
    return failure();

  auto result =
      builder.create<WireOp>(info.getLoc(), type, filterUselessName(id));
  return addSymbolEntry(id.getValue(), result, info.getFIRLoc());
}

/// register    ::= 'reg' id ':' type exp ('with' ':' reset_block)? info?
///
/// reset_block ::= INDENT simple_reset info? NEWLINE DEDENT
///             ::= '(' simple_reset ')'
///
/// simple_reset ::= simple_reset0
///              ::= '(' simple_reset0 ')'
///
/// simple_reset0:  'reset' '=>' '(' exp exp ')'
///
ParseResult FIRStmtParser::parseRegister(unsigned regIndent) {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_reg);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword("reg", info))
    return isExpr.getValue();

  StringAttr id;
  FIRRTLType type;
  Value clock;
  SmallVector<Operation *, 8> subOps;

  // TODO(firrtl spec): info? should come after the clock expression before
  // the 'with'.
  if (parseId(id, "expected reg name") ||
      parseToken(FIRToken::colon, "expected ':' in reg") ||
      parseType(type, "expected reg type") ||
      parseExp(clock, subOps, "expected expression for register clock"))
    return failure();

  // Parse the 'with' specifier if present.
  Value resetSignal, resetValue;
  if (consumeIf(FIRToken::kw_with)) {
    if (parseToken(FIRToken::colon, "expected ':' in reg"))
      return failure();

    // TODO(firrtl spec): Simplify the grammar for register reset logic.
    // Why allow multiple ambiguous parentheses?  Why rely on indentation at
    // all?

    // This implements what the examples have in practice.
    bool hasExtraLParen = consumeIf(FIRToken::l_paren);

    auto indent = getIndentation();
    if (!indent.hasValue() || indent.getValue() <= regIndent)
      if (!hasExtraLParen)
        return emitError("expected indented reset specifier in reg"), failure();

    SmallVector<Operation *, 8> subOps;
    if (parseToken(FIRToken::kw_reset, "expected 'reset' in reg") ||
        parseToken(FIRToken::equal_greater, "expected => in reset specifier") ||
        parseToken(FIRToken::l_paren, "expected '(' in reset specifier") ||
        parseExp(resetSignal, subOps, "expected expression for reset signal"))
      return failure();

    // The Scala implementation of FIRRTL represents registers without resets
    // as a self referential register... and the pretty printer doesn't print
    // the right form. Recognize that this is happening and treat it as a
    // register without a reset for compatibility.
    // TODO(firrtl scala impl): pretty print registers without resets right.
    if (getTokenSpelling() == id.getValue()) {
      consumeToken();
      if (parseToken(FIRToken::r_paren, "expected ')' in reset specifier") ||
          parseOptionalInfo(info, subOps))
        return failure();
      resetSignal = Value();
    } else {
      if (parseExp(resetValue, subOps, "expected expression for reset value") ||
          parseToken(FIRToken::r_paren, "expected ')' in reset specifier") ||
          parseOptionalInfo(info, subOps))
        return failure();
    }

    if (hasExtraLParen &&
        parseToken(FIRToken::r_paren, "expected ')' in reset specifier"))
      return failure();
  }

  // Finally, handle the last info if present, providing location info for the
  // clock expression.
  if (parseOptionalInfo(info, subOps))
    return failure();

  Value result;
  if (resetSignal)
    result = builder.create<RegInitOp>(info.getLoc(), type, clock, resetSignal,
                                       resetValue, filterUselessName(id));
  else
    result = builder.create<RegOp>(info.getLoc(), type, clock,
                                   filterUselessName(id));

  return addSymbolEntry(id.getValue(), result, info.getFIRLoc());
}

//===----------------------------------------------------------------------===//
// FIRModuleParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic and state for parsing module bodies.
struct FIRModuleParser : public FIRScopedParser {
  explicit FIRModuleParser(GlobalFIRParserState &state, CircuitOp circuit)
      : FIRScopedParser(state, symbolTable, memoryScopeTable), circuit(circuit),
        firstScope(symbolTable), firstMemoryScope(memoryScopeTable) {}

  ParseResult parseExtModule(unsigned indent);
  ParseResult parseModule(unsigned indent);

private:
  using PortInfoAndLoc = std::pair<ModulePortInfo, SMLoc>;
  ParseResult parsePortList(SmallVectorImpl<PortInfoAndLoc> &result,
                            unsigned indent);

  CircuitOp circuit;
  SymbolTable symbolTable;
  SymbolTable::ScopeTy firstScope;

  MemoryScopeTable memoryScopeTable;
  MemoryScopeTable::ScopeTy firstMemoryScope;
};

} // end anonymous namespace

/// portlist ::= port*
/// port     ::= dir id ':' type info? NEWLINE
/// dir      ::= 'input' | 'output'
///
ParseResult
FIRModuleParser::parsePortList(SmallVectorImpl<PortInfoAndLoc> &result,
                               unsigned indent) {
  // Parse any ports.
  while (getToken().isAny(FIRToken::kw_input, FIRToken::kw_output) &&
         // Must be nested under the module.
         getIndentation() > indent) {
    bool isOutput = getToken().is(FIRToken::kw_output);

    consumeToken();
    StringAttr name;
    FIRRTLType type;
    LocWithInfo info(getToken().getLoc(), this);
    if (parseId(name, "expected port name") ||
        parseToken(FIRToken::colon, "expected ':' in port definition") ||
        parseType(type, "expected a type in port declaration") ||
        parseOptionalInfo(info))
      return failure();

    // If this is an output port, flip the type.
    if (isOutput)
      type = FlipType::get(type);

    // FIXME: We should persist the info loc into the IR, not just the name
    // and type.
    result.push_back({{name, type}, info.getFIRLoc()});
  }

  return success();
}

/// module    ::=
///        'extmodule' id ':' info? INDENT portlist defname? parameter* DEDENT
/// defname   ::= 'defname' '=' id NEWLINE
///
/// parameter ::= 'parameter' id '=' intLit NEWLINE
/// parameter ::= 'parameter' id '=' StringLit NEWLINE
/// parameter ::= 'parameter' id '=' floatingpoint NEWLINE
/// parameter ::= 'parameter' id '=' RawString NEWLINE
ParseResult FIRModuleParser::parseExtModule(unsigned indent) {
  consumeToken(FIRToken::kw_extmodule);
  StringAttr name;
  SmallVector<PortInfoAndLoc, 4> portListAndLoc;

  LocWithInfo info(getToken().getLoc(), this);
  if (parseId(name, "expected module name") ||
      parseToken(FIRToken::colon, "expected ':' in extmodule definition") ||
      parseOptionalInfo(info) || parsePortList(portListAndLoc, indent))
    return failure();

  auto builder = circuit.getBodyBuilder();

  // Create the module.
  SmallVector<ModulePortInfo, 4> portList;
  portList.reserve(portListAndLoc.size());
  for (auto &elt : portListAndLoc)
    portList.push_back(elt.first);

  // Add all the ports to the symbol table even though there are no SSA values
  // for arguments in an external module.  This detects multiple definitions
  // of the same name.
  for (auto &entry : portListAndLoc) {
    if (addSymbolEntry(entry.first.first.getValue(), Value(), entry.second))
      return failure();
  }

  // Parse a defname if present.
  // TODO(firrtl spec): defname isn't documented at all, what is it?
  StringRef defName;
  if (consumeIf(FIRToken::kw_defname)) {
    if (parseToken(FIRToken::equal, "expected '=' in defname") ||
        parseId(defName, "expected defname name"))
      return failure();
  }

  SmallVector<NamedAttribute, 4> parameters;
  SmallPtrSet<Identifier, 8> seenNames;

  // Parse the parameter list.
  while (consumeIf(FIRToken::kw_parameter)) {
    auto loc = getToken().getLoc();
    StringRef paramName;
    if (parseId(paramName, "expected parameter name") ||
        parseToken(FIRToken::equal, "expected '=' in parameter"))
      return failure();

    Attribute value;
    switch (getToken().getKind()) {
    default:
      return emitError("expected parameter value"), failure();

    case FIRToken::integer: {
      APInt result;
      if (getTokenSpelling().getAsInteger(10, result))
        return emitError("invalid integer parameter"), failure();
      value = builder.getIntegerAttr(
          builder.getIntegerType(result.getBitWidth()), result);
      consumeToken(FIRToken::integer);
      break;
    }
    case FIRToken::string: {
      // Drop the quotes and unescape.
      value = builder.getStringAttr(getToken().getStringValue());
      consumeToken(FIRToken::string);
      break;
    }

    case FIRToken::floatingpoint:
      double v;
      if (!llvm::to_float(getTokenSpelling(), v))
        return emitError("invalid float parameter syntax"), failure();

      value = builder.getF64FloatAttr(v);
      consumeToken(FIRToken::floatingpoint);
      break;
    }

    auto nameId = builder.getIdentifier(paramName);
    if (!seenNames.insert(nameId).second)
      return emitError(loc, "redefinition of parameter '" + paramName + "'");
    parameters.push_back({nameId, value});
  }

  auto fmodule =
      builder.create<FExtModuleOp>(info.getLoc(), name, portList, defName);

  if (!parameters.empty())
    fmodule.setAttr("parameters",
                    DictionaryAttr::get(parameters, getContext()));

  return success();
}

/// module ::= 'module' id ':' info? INDENT portlist simple_stmt_block
/// DEDENT
///
ParseResult FIRModuleParser::parseModule(unsigned indent) {
  LocWithInfo info(getToken().getLoc(), this);
  StringAttr name;
  SmallVector<PortInfoAndLoc, 4> portListAndLoc;

  consumeToken(FIRToken::kw_module);
  if (parseId(name, "expected module name") ||
      parseToken(FIRToken::colon, "expected ':' in module definition") ||
      parseOptionalInfo(info) || parsePortList(portListAndLoc, indent))
    return failure();

  auto builder = circuit.getBodyBuilder();

  // Create the module.
  SmallVector<ModulePortInfo, 4> portList;
  portList.reserve(portListAndLoc.size());
  for (auto &elt : portListAndLoc)
    portList.push_back(elt.first);
  auto fmodule = builder.create<FModuleOp>(info.getLoc(), name, portList);

  // Install all of the ports into the symbol table, associated with their
  // block arguments.
  auto argIt = fmodule.args_begin();
  for (auto &entry : portListAndLoc) {
    if (addSymbolEntry(entry.first.first.getValue(), *argIt, entry.second))
      return failure();
    ++argIt;
  }

  FIRModuleContext moduleContext;
  FIRStmtParser stmtParser(fmodule.getBodyBuilder(), *this, moduleContext);

  // Parse the moduleBlock.
  return stmtParser.parseSimpleStmtBlock(indent);
}

//===----------------------------------------------------------------------===//
// FIRCircuitParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the outer level of the parser, including things
/// like circuit and module.
struct FIRCircuitParser : public FIRParser {
  explicit FIRCircuitParser(GlobalFIRParserState &state, ModuleOp mlirModule)
      : FIRParser(state), mlirModule(mlirModule) {}

  ParseResult parseCircuit();

private:
  ModuleOp mlirModule;
};

} // end anonymous namespace

/// file ::= circuit
/// circuit ::= 'circuit' id ':' info? INDENT module* DEDENT EOF
///
ParseResult FIRCircuitParser::parseCircuit() {
  auto indent = getIndentation();
  if (!indent.hasValue())
    return emitError("'circuit' must be first token on its line"), failure();
  unsigned circuitIndent = indent.getValue();

  LocWithInfo info(getToken().getLoc(), this);
  StringAttr name;

  // A file must contain a top level `circuit` definition.
  if (parseToken(FIRToken::kw_circuit,
                 "expected a top-level 'circuit' definition") ||
      parseId(name, "expected circuit name") ||
      parseToken(FIRToken::colon, "expected ':' in circuit definition") ||
      parseOptionalInfo(info))
    return failure();

  // Create the top-level circuit op in the MLIR module.
  OpBuilder b(mlirModule.getBodyRegion());
  auto circuit = b.create<CircuitOp>(info.getLoc(), name);

  // Parse any contained modules.
  while (true) {
    switch (getToken().getKind()) {
    // If we got to the end of the file, then we're done.
    case FIRToken::eof:
      return success();

    // If we got an error token, then the lexer already emitted an error,
    // just stop.  We could introduce error recovery if there was demand for
    // it.
    case FIRToken::error:
      return failure();

    default:
      emitError("unexpected token in circuit");
      return failure();

    case FIRToken::kw_module:
    case FIRToken::kw_extmodule: {
      auto indent = getIndentation();
      if (!indent.hasValue())
        return emitError("'module' must be first token on its line"), failure();
      unsigned moduleIndent = indent.getValue();

      if (moduleIndent <= circuitIndent)
        return emitError("module should be indented more"), failure();

      FIRModuleParser mp(getState(), circuit);
      if (getToken().is(FIRToken::kw_module) ? mp.parseModule(moduleIndent)
                                             : mp.parseExtModule(moduleIndent))
        return failure();
      break;
    }
    }
  }
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified .fir file into the specified MLIR context.
OwningModuleRef cirt::parseFIRFile(SourceMgr &sourceMgr, MLIRContext *context,
                                   FIRParserOptions options) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // This is the result module we are parsing into.
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0, context)));

  GlobalFIRParserState state(sourceMgr, context, options);
  if (FIRCircuitParser(state, *module).parseCircuit())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  if (failed(verify(*module)))
    return {};

  return module;
}

void cirt::registerFIRParserTranslation() {
  static TranslateToMLIRRegistration fromFIR(
      "parse-fir", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return parseFIRFile(sourceMgr, context);
      });
}
