//===- FIRToMLIR.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "FIRLexer.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Translation.h"
#include "spt/Dialect/FIRRTL/FIRToMLIR.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "spt/Dialect/FIRRTL/IR/Types.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace spt;
using namespace firrtl;
using namespace mlir;
using llvm::SMLoc;
using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// GlobalFIRParserState
//===----------------------------------------------------------------------===//

namespace {
/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position.  This is separated out from the parser
/// so that individual subparsers can refer to the same state.
struct GlobalFIRParserState {
  GlobalFIRParserState(const llvm::SourceMgr &sourceMgr, FIRRTLDialect *dialect)
      : context(dialect->getContext()), dialect(dialect),
        lex(sourceMgr, context), curToken(lex.lexToken()) {}

  /// The context we're parsing into.
  MLIRContext *const context;

  /// The FIRRTL dialect.
  FIRRTLDialect *const dialect;

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

  /// Parse a comma separated list of elements that must have at least one entry
  /// in it.
  ParseResult
  parseCommaSeparatedList(const std::function<ParseResult()> &parseElement);

  /// Parse a comma-separated list of elements, terminated with an arbitrary
  /// token.  This allows empty lists if allowEmptyList is true.
  ParseResult
  parseCommaSeparatedListUntil(FIRToken::Kind rightToken,
                               const std::function<ParseResult()> &parseElement,
                               bool allowEmptyList = true);

  //===--------------------------------------------------------------------===//
  // Common Parser Rules
  //===--------------------------------------------------------------------===//

  /// Parse 'intLit' into the specified value.
  ParseResult parseIntLit(int32_t &result, const Twine &message);

  // Parse ('<' intLit '>')? setting result to -1 if not present.
  ParseResult parseOptionalWidth(int32_t &result);

  // Parse the 'id' grammar, which is an identifier or an allowed keyword.
  ParseResult parseId(StringRef &result, const Twine &message);
  ParseResult parseId(StringAttr &result, const Twine &message);
  ParseResult parseFieldId(StringRef &result, const Twine &message);
  ParseResult parseType(FIRRTLType &result, const Twine &message);

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

/// Parse a comma separated list of elements that must have at least one entry
/// in it.
ParseResult FIRParser::parseCommaSeparatedList(
    const std::function<ParseResult()> &parseElement) {
  // Non-empty case starts with an element.
  if (parseElement())
    return failure();

  // Otherwise we have a list of comma separated elements.
  while (consumeIf(FIRToken::comma)) {
    if (parseElement())
      return failure();
  }
  return success();
}

/// Parse a comma-separated list of elements, terminated with an arbitrary
/// token.  This allows empty lists if allowEmptyList is true.
ParseResult FIRParser::parseCommaSeparatedListUntil(
    FIRToken::Kind rightToken, const std::function<ParseResult()> &parseElement,
    bool allowEmptyList) {

  // Handle the empty case.
  if (getToken().is(rightToken)) {
    if (!allowEmptyList)
      return emitError("expected list element");
    consumeToken(rightToken);
    return success();
  }

  if (parseCommaSeparatedList(parseElement) ||
      parseToken(rightToken, "expected ',' or end of list"))
    return failure();

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
ParseResult FIRParser::parseIntLit(int32_t &result, const Twine &message) {
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
///      ::= '{' '}' | '{' field (',' field)* '}'
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
    if (parseCommaSeparatedListUntil(FIRToken::r_brace, [&]() -> ParseResult {
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

//===----------------------------------------------------------------------===//
// FIRScopedParser
//===----------------------------------------------------------------------===//

namespace {
/// This class is a common base class for stuff that needs local scope
/// manipulation helpers.
class FIRScopedParser : public FIRParser {
public:
  using SymbolTable =
      llvm::ScopedHashTable<Identifier, std::pair<SMLoc, Value>>;
  FIRScopedParser(GlobalFIRParserState &state, SymbolTable &symbolTable)
      : FIRParser(state), symbolTable(symbolTable) {}

  /// Add a symbol entry with the specified name, returning failure if the name
  /// is already defined.
  ParseResult addSymbolEntry(StringRef name, Value value, SMLoc loc);

  /// Look up the specified name, emitting an error and returning failure if the
  /// name is unknown.
  ParseResult lookupSymbolEntry(Value &result, StringRef name, SMLoc loc);

protected:
  // This symbol table holds the names of ports, wires, and other local decls.
  // This is scoped because conditional statements introduce subscopes.
  SymbolTable &symbolTable;
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
    return emitError(loc, "use of invalid name '" + name.str() + "'"),
           failure();

  assert(prev.second && "name in symbol table without definition");
  result = prev.second;
  return success();
}

//===----------------------------------------------------------------------===//
// FIRStmtParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic and state for parsing statements, suites, and
/// similar module body constructs.
struct FIRStmtParser : public FIRScopedParser {
  explicit FIRStmtParser(GlobalFIRParserState &state, SymbolTable &symbolTable,
                         OpBuilder builder)
      : FIRScopedParser(state, symbolTable), builder(builder) {}

  ParseResult parseSimpleStmt(unsigned stmtIndent);
  ParseResult parseSimpleStmtBlock(unsigned indent);

private:
  // Exp Parsing
  ParseResult parseExp(Value &result, SmallVectorImpl<Operation *> &subOps,
                       const Twine &message);
  ParseResult parsePrimExp(Value &result, SmallVectorImpl<Operation *> &subOps);
  ParseResult parseIntegerLiteralExp(Value &result,
                                     SmallVectorImpl<Operation *> &subOps);

  // Stmt Parsing
  ParseResult parsePrintf();
  ParseResult parseSkip();
  ParseResult parseStop();
  ParseResult parseNode();
  ParseResult parseWire();
  ParseResult parseWhen(unsigned whenIndent);
  ParseResult parseLeadingExpStmt();

  // The builder to build into.
  OpBuilder builder;
};

} // end anonymous namespace

/// Parse the 'exp' grammar, returning all of the suboperations in the
/// specified vector, and the ultimate SSA value in value.
///
///  exp ::= id    // Ref
///      ::= prim
///      ::= integer-literal-exp
///      ::= exp '.' fieldId
///      ::= exp '[' intLit ']'
/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
/// XX   ::= exp '[' exp ']'
/// XX   ::= 'mux(' exp exp exp ')'
/// XX   ::= 'validif(' exp exp ')'
///
ParseResult FIRStmtParser::parseExp(Value &result,
                                    SmallVectorImpl<Operation *> &subOps,
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

  // Handle postfix expressions.
  while (1) {
    auto loc = getToken().getLoc();

    // Subfield: exp ::= exp '.' fieldId
    if (consumeIf(FIRToken::period)) {
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
                                     builder.getStringAttr(fieldName),
                                     /*optionalName*/ StringAttr());
      subOps.push_back(op);
      result = op.getResult();
      continue;
    }

    // Subindex: exp ::= exp '[' intLit ']'
    if (consumeIf(FIRToken::l_square)) {
      auto indexLoc = getToken().getLoc();
      int32_t indexNo;
      if (parseIntLit(indexNo, "expected index") ||
          parseToken(FIRToken::r_square, "expected ']'"))
        return failure();

      if (indexNo < 0)
        return emitError(indexLoc, "invalid index specifier"), failure();

      // Make sure the field name matches up with the input value's type and
      // compute the result type for the expression.
      auto resultType = result.getType().cast<FIRRTLType>();
      resultType = SubindexOp::getResultType(resultType, indexNo);
      if (!resultType) {
        // TODO(QoI): This error would be nicer with a .fir pretty print of the
        // type.
        emitError(loc, "invalid subindex '" + Twine(indexNo))
            << "' for type " << result.getType();
        return failure();
      }

      // Create the result operation.
      auto op =
          builder.create<SubindexOp>(translateLocation(loc), resultType, result,
                                     builder.getI32IntegerAttr(indexNo),
                                     /*optionalName*/ StringAttr());
      subOps.push_back(op);
      result = op.getResult();
      continue;
    }

    return success();
  }
}

/// prim ::= primop exp* intLit*  ')'
ParseResult FIRStmtParser::parsePrimExp(Value &result,
                                        SmallVectorImpl<Operation *> &subOps) {
  auto kind = getToken().getKind();
  auto loc = getToken().getLoc();
  consumeToken();

  // Parse the operands and constant integer arguments.
  SmallVector<Value, 4> operands;
  if (parseCommaSeparatedListUntil(FIRToken::r_paren, [&]() -> ParseResult {
        // FIXME: This isn't handling the intLit case.
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
    opTypes.push_back(v.getType().cast<FIRRTLType>());

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

  ArrayRef<NamedAttribute> attrs;
  switch (kind) {
  default:
    emitError(loc, "primitive not supported yet");
    return failure();

#define TOK_LPKEYWORD_PRIM(SPELLING, CLASS)                                    \
  case FIRToken::lp_##SPELLING: {                                              \
    auto resultTy = CLASS::getResultType(opTypes);                             \
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
ParseResult
FIRStmtParser::parseIntegerLiteralExp(Value &result,
                                      SmallVectorImpl<Operation *> &subOps) {
  bool isSigned = getToken().is(FIRToken::kw_SInt);
  auto loc = getToken().getLoc();
  consumeToken();

  // Parse a width specifier if present.
  int32_t width, value;
  if (parseOptionalWidth(width) ||
      parseToken(FIRToken::l_paren, "expected '(' in integer expression") ||
      parseIntLit(value, "expected integer value") ||
      parseToken(FIRToken::r_paren, "expected ')' in integer expression"))
    return failure();

  FIRRTLType resultType = getIntegerType(builder.getContext(), isSigned, width);
  result = builder.create<ConstantOp>(translateLocation(loc), resultType,
                                      builder.getI32IntegerAttr(value),
                                      /*optionalName*/ StringAttr());
  return success();
}

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
/// stmt ::= printf
///      ::= skip
///      ::= stop
///      ::= node
///      ::= wire
///      ::= when
///      ::= leading-exp-stmt
///
ParseResult FIRStmtParser::parseSimpleStmt(unsigned stmtIndent) {
  switch (getToken().getKind()) {
  case FIRToken::lp_printf:
    return parsePrintf();
  case FIRToken::kw_skip:
    return parseSkip();
  case FIRToken::lp_stop:
    return parseStop();
  case FIRToken::kw_node:
    return parseNode();
  case FIRToken::kw_wire:
    return parseWire();
  case FIRToken::kw_when:
    return parseWhen(stmtIndent);

  default:
    // Otherwise, this must be one of the productions that starts with an
    // expression.
    return parseLeadingExpStmt();
  }
}

/// printf ::= 'printf(' exp ',' exp ',' StringLit (',' exp)* ')' info?
ParseResult FIRStmtParser::parsePrintf() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::lp_printf);

  SmallVector<Operation *, 8> subOps;

  Value clock, condition;
  StringRef formatString;
  if (parseExp(clock, subOps, "expected clock expression in printf") ||
      parseToken(FIRToken::comma, "expected ',' in printf") ||
      parseExp(condition, subOps, "expected condition in printf") ||
      parseToken(FIRToken::comma, "expected ',' in printf") ||
      parseGetSpelling(formatString) ||
      parseToken(FIRToken::string, "expected format string in printf"))
    return failure();

  SmallVector<Value, 4> operands;
  while (!consumeIf(FIRToken::r_paren)) {
    operands.push_back({});
    if (parseToken(FIRToken::comma, "expected ',' in printf") ||
        parseExp(operands.back(), subOps, "expected operand in printf"))
      return failure();
  }

  if (parseOptionalInfo(info, subOps))
    return failure();

  builder.create<PrintFOp>(info.getLoc(), clock, condition,
                           builder.getStringAttr(formatString), operands);
  return success();
}

/// skip ::= 'skip' info?
ParseResult FIRStmtParser::parseSkip() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_skip);
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
      parseToken(FIRToken::comma, "expected ',' in 'stop'") ||
      parseExp(condition, subOps, "expected condition in 'stop'") ||
      parseToken(FIRToken::comma, "expected ',' in 'stop'") ||
      parseIntLit(exitCode, "expected exit code in 'stop'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'stop'") ||
      parseOptionalInfo(info, subOps))
    return failure();

  builder.create<StopOp>(info.getLoc(), clock, condition,
                         builder.getI32IntegerAttr(exitCode));
  return success();
}

/// node ::= 'node' id '=' exp info?
ParseResult FIRStmtParser::parseNode() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_node);

  StringAttr id;
  Value initializer;
  SmallVector<Operation *, 8> subOps;
  if (parseId(id, "expected node name") ||
      parseToken(FIRToken::equal, "expected '=' in node") ||
      parseExp(initializer, subOps, "unexpected expression for node") ||
      parseOptionalInfo(info, subOps))
    return failure();

  auto result = builder.create<NodeOp>(info.getLoc(), initializer, id);
  if (addSymbolEntry(id.getValue(), result, info.getFIRLoc()))
    return failure();

  return success();
}

/// wire ::= 'wire' id ':' type info?
ParseResult FIRStmtParser::parseWire() {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_wire);

  StringAttr id;
  FIRRTLType type;
  if (parseId(id, "expected wire name") ||
      parseToken(FIRToken::colon, "expected ':' in wire") ||
      parseType(type, "expected wire type") || parseOptionalInfo(info))
    return failure();

  auto result = builder.create<WireOp>(info.getLoc(), type, id);
  if (addSymbolEntry(id.getValue(), result, info.getFIRLoc()))
    return failure();

  return success();
}

/// when  ::= 'when' exp ':' info? suite? ('else' ( when | ':' info? suite?)
/// )? suite ::= simple_stmt | INDENT simple_stmt+ DEDENT
ParseResult FIRStmtParser::parseWhen(unsigned whenIndent) {
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_when);

  Value condition;
  SmallVector<Operation *, 8> subOps;
  if (parseExp(condition, subOps, "unexpected expression in when") ||
      parseToken(FIRToken::colon, "expected ':' in when") ||
      parseOptionalInfo(info, subOps))
    return failure();

  // Create the IR representation for the when.
  auto whenStmt = builder.create<WhenOp>(info.getLoc(), condition,
                                         /*createElse*/ false);

  // This is a function to parse a suite body.
  auto parseSuite = [&](OpBuilder builder) -> ParseResult {
    // Declarations within the suite are scoped to within the suite.
    SymbolTable::ScopeTy suiteScope(symbolTable);

    // We parse the substatements into their own parser, so they get insert
    // into the specified 'when' region.
    FIRStmtParser subParser(getState(), symbolTable, builder);

    // Figure out whether the body is a single statement or a nested one.
    auto stmtIndent = getIndentation();

    // Parsing a single statment is straightforward.
    if (!stmtIndent.hasValue())
      return subParser.parseSimpleStmt(whenIndent);

    if (stmtIndent <= whenIndent)
      return emitError("when body must be indented more than when"), failure();

    // Parse a block of statements that are indented more than the when.
    return subParser.parseSimpleStmtBlock(whenIndent);
  };

  // Parse the 'then' body into the 'then' region.
  if (parseSuite(whenStmt.getThenBodyBuilder()))
    return failure();

  // If the else is present, handle it otherwise we're done.
  LocWithInfo elseInfo(getToken().getLoc(), this);
  if (!consumeIf(FIRToken::kw_else))
    return success();

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

  // TODO(firtl spec): There is no reason for the 'else :' grammar to take an
  // info.  It doesn't appear to be generated either.
  return success();
}

/// leading-exp-stmt ::= exp '<=' exp info?
///                  ::= exp '<-' exp info?
///                  ::= exp 'is' 'invalid' info?
ParseResult FIRStmtParser::parseLeadingExpStmt() {
  Value lhs;
  SmallVector<Operation *, 8> subOps;
  if (parseExp(lhs, subOps, "unexpected token in module"))
    return failure();

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

//===----------------------------------------------------------------------===//
// FIRModuleParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic and state for parsing module bodies.
struct FIRModuleParser : public FIRScopedParser {
  explicit FIRModuleParser(GlobalFIRParserState &state, CircuitOp circuit)
      : FIRScopedParser(state, symbolTable), circuit(circuit),
        firstScope(symbolTable) {}

  ParseResult parseExtModule(unsigned indent);
  ParseResult parseModule(unsigned indent);

private:
  using PortInfoAndLoc = std::pair<FModuleOp::PortInfo, SMLoc>;
  ParseResult parsePortList(SmallVectorImpl<PortInfoAndLoc> &result,
                            unsigned indent);

  CircuitOp circuit;
  SymbolTable symbolTable;
  SymbolTable::ScopeTy firstScope;
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
/// parameter ::= 'parameter' id '=' DoubleLit NEWLINE
/// parameter ::= 'parameter' id '=' RawString NEWLINE
ParseResult FIRModuleParser::parseExtModule(unsigned indent) {
  consumeToken(FIRToken::kw_extmodule);
  StringRef name;
  SmallVector<PortInfoAndLoc, 4> portListAndLoc;

  LocWithInfo info(getToken().getLoc(), this);
  if (parseId(name, "expected module name") ||
      parseToken(FIRToken::colon, "expected ':' in extmodule definition") ||
      parseOptionalInfo(info) || parsePortList(portListAndLoc, indent))
    return failure();

  // Add all the ports to the symbol table even though there are no SSA values
  // for arguments in an external module.  This detects multiple definitions
  // of the same name.
  for (auto &entry : portListAndLoc) {
    if (addSymbolEntry(entry.first.first.getValue(), Value(), entry.second))
      return failure();
  }

  // Parse a defname if present.
  if (consumeIf(FIRToken::kw_defname)) {
    StringRef defName;
    if (parseToken(FIRToken::equal, "expected '=' in defname") ||
        parseId(defName, "expected defname name"))
      return failure();
  }

  // FIXME: Build a representation of this defname.

  // Parse the parameter list.
  while (consumeIf(FIRToken::kw_parameter)) {
    StringRef paramName;
    if (parseId(paramName, "expected parameter name") ||
        parseToken(FIRToken::equal, "expected '=' in parameter"))
      return failure();

    // FIXME: This isn't right.
    if (getToken().isAny(FIRToken::integer, FIRToken::string))
      consumeToken();
    else
      return emitError("expected parameter value"), failure();
  }

  // FIXME: Build a record of this parameter.

  return success();
}

/// module ::= 'module' id ':' info? INDENT portlist simple_stmt_block DEDENT
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
  SmallVector<FModuleOp::PortInfo, 4> portList;
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

  FIRStmtParser stmtParser(getState(), symbolTable, fmodule.getBodyBuilder());

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

    // If we got an error token, then the lexer already emitted an error, just
    // stop.  We could introduce error recovery if there was demand for it.
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
static OwningModuleRef parseFIRFile(SourceMgr &sourceMgr,
                                    MLIRContext *context) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // This is the result module we are parsing into.
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0, context)));

  auto *dialect = context->getRegisteredDialect<FIRRTLDialect>();
  assert(dialect && "Could not find FIRRTL dialect?");

  GlobalFIRParserState state(sourceMgr, dialect);
  if (FIRCircuitParser(state, *module).parseCircuit())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  if (failed(verify(*module)))
    return {};

  return module;
}

void spt::firrtl::registerFIRRTLToMLIRTranslation() {
  static TranslateToMLIRRegistration fromLLVM(
      "import-firrtl", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return parseFIRFile(sourceMgr, context);
      });
}
