//===- FIRToMLIR.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "FIRLexer.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"
#include "spt/Dialect/FIRRTL/FIRToMLIR.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
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
  unsigned getIndentation() const {
    return state.lex.getIndentation(state.curToken);
  }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(state.curToken.getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location getEncodedSourceLocation(llvm::SMLoc loc) {
    return state.lex.getEncodedSourceLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// Return the current token the parser is inspecting.
  const FIRToken &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

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

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(FIRToken::Kind expectedToken, const Twine &message);

  //===--------------------------------------------------------------------===//
  // Common Parser Rules
  //===--------------------------------------------------------------------===//

  // Parse the 'id' grammar, which is an identifier or an allowed keyword.
  ParseResult parseId(const Twine &message);
  ParseResult parseType(const Twine &message);

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
  auto diag = mlir::emitError(getEncodedSourceLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(FIRToken::error))
    diag.abandon();
  return diag;
}

//===----------------------------------------------------------------------===//
// Token Parsing
//===----------------------------------------------------------------------===//

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult FIRParser::parseToken(FIRToken::Kind expectedToken,
                                  const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

//===--------------------------------------------------------------------===//
// Common Parser Rules.
//===--------------------------------------------------------------------===//

// Parse the 'id' grammar, which is an identifier or an allowed keyword.
ParseResult FIRParser::parseId(const Twine &message) {
  switch (getToken().getKind()) {
  // The most common case is an identifier.
  case FIRToken::identifier:
// Otherwise it may be a keyword that we're allowing in an id position.
#define TOK_KEYWORD(spelling) case FIRToken::kw_##spelling:
#include "FIRTokenKinds.def"

    // Yep, this is a valid 'id'.
    consumeToken();
    return success();

  default:
    emitError(message);
    return failure();
  }
}

ParseResult FIRParser::parseType(const Twine &message) {
  // FIXME: This is not the correct type grammar :-)
  if (parseToken(FIRToken::kw_UInt, "expected type") ||
      parseToken(FIRToken::less, "expected <") ||
      parseToken(FIRToken::integer, "expected width") ||
      parseToken(FIRToken::greater, "expected >"))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// FIRFileParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the outer level of the parser, including things
/// like circuit and module.
struct FIRFileParser : public FIRParser {
  FIRFileParser(GlobalFIRParserState &state, ModuleOp module)
      : FIRParser(state), module(module) {}

  ParseResult parseFile();

private:
  ParseResult parseModule(unsigned indent);

  ModuleOp module;
};

} // end anonymous namespace

/// file ::= circuit
/// circuit ::= 'circuit' id ':' info? INDENT module* DEDENT EOF
///
ParseResult FIRFileParser::parseFile() {
  unsigned circuitIndent = getIndentation();

  // A file must contain a top level `circuit` definition.
  if (parseToken(FIRToken::kw_circuit,
                 "expected a top-level 'circuit' definition") ||
      parseId("expected circuit name") ||
      parseToken(FIRToken::colon, "expected ':' in circuit definition"))
    // FIXME: Parse an 'info' if present.
    return failure();

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

    case FIRToken::kw_module: {
      unsigned moduleIndent = getIndentation();
      if (moduleIndent <= circuitIndent)
        return emitError("module should be indented more"), failure();

      if (parseModule(moduleIndent))
        return failure();
      break;
    }

      // FIXME: Support extmodule.
      // 'extmodule' id ':' info? INDENT port* defname? parameter* DEDENT
    }
  }
}

/// module ::= 'module' id ':' info? INDENT port* moduleBlock DEDENT
/// moduleBlock ::= simple_stmt*
///
/// TODO: split port processing out to share with ext_module.
/// port   ::= dir id ':' type info? NEWLINE
/// dir    ::= 'input' | 'output'
///
ParseResult FIRFileParser::parseModule(unsigned indent) {
  consumeToken(FIRToken::kw_module);

  if (parseId("expected module name") ||
      parseToken(FIRToken::colon, "expected ':' in module definition"))
    // FIXME: Parse an 'info' if present.
    return failure();

  // Parse any ports.
  while (getToken().isAny(FIRToken::kw_input, FIRToken::kw_output) &&
         // Must be nested under the module.
         getIndentation() > indent) {

    consumeToken();
    if (parseId("expected port name") ||
        parseToken(FIRToken::colon, "expected ':' in port definition") ||
        parseType("expected a type in port declaration"))
      // FIXME: Parse an 'info' if present.
      return failure();
  }

  // Parse the moduleBlock.

  while (true) {
    unsigned subIndent = getIndentation();
    if (subIndent <= indent)
      return success();

    switch (getToken().getKind()) {
    // The outer level parser can handle these tokens.
    case FIRToken::eof:
    case FIRToken::error:
      return success();

    case FIRToken::identifier:
      // Hard coding identifier <= identifier
      // FIXME: This is wrong.
      consumeToken(FIRToken::identifier);
      if (parseToken(FIRToken::less_equal, "expected <=") ||
          parseToken(FIRToken::identifier, "expected identifier"))
        return failure();
      break;

    default:
      emitError("unexpected token in module");
      return failure();
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
  if (FIRFileParser(state, *module).parseFile())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  if (failed(verify(*module)))
    return {};

  return {};
}

void spt::firrtl::registerFIRRTLToMLIRTranslation() {
  static TranslateToMLIRRegistration fromLLVM(
      "import-firrtl", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return parseFIRFile(sourceMgr, context);
      });
}
