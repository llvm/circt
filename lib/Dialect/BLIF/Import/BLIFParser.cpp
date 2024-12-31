//===- FIRParser.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/BLIF/BLIFParser.h"
#include "BLIFLexer.h"
#include "circt/Dialect/BLIF/BLIFOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <utility>

using namespace circt;
using namespace blif;

using llvm::SMLoc;
using llvm::SourceMgr;
using mlir::LocationAttr;

//===----------------------------------------------------------------------===//
// BLIFParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic common to all levels of the parser, including
/// things like types and helper logic.
struct BLIFParser {
  BLIFParser(MLIRContext *ctx, BLIFLexer &lexer) : context(ctx), lexer(lexer) {}

  MLIRContext *getContext() const { return context; }

  BLIFLexer &getLexer() { return lexer; }

  /// Return the current token the parser is inspecting.
  const BLIFToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }

  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Emit a warning.
  InFlightDiagnostic emitWarning(const Twine &message = {}) {
    return emitWarning(getToken().getLoc(), message);
  }

  InFlightDiagnostic emitWarning(SMLoc loc, const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  class LocWithInfo;

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(BLIFToken::Kind kind) {
    if (getToken().isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  ///
  /// This returns the consumed token.
  BLIFToken consumeToken() {
    BLIFToken consumedToken = getToken();
    assert(consumedToken.isNot(BLIFToken::eof, BLIFToken::error) &&
           "shouldn't advance past EOF or errors");
    lexer.lexToken();
    return consumedToken;
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  ///
  /// This returns the consumed token.
  BLIFToken consumeToken(BLIFToken::Kind kind) {
    BLIFToken consumedToken = getToken();
    assert(consumedToken.is(kind) && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  /// Capture the current token's spelling into the specified value.  This
  /// always succeeds.
  ParseResult parseGetSpelling(StringRef &spelling) {
    spelling = getTokenSpelling();
    return success();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(BLIFToken::Kind expectedToken, const Twine &message);

  /// Parse a comma-separated list of elements, terminated with an arbitrary
  /// token.
  ParseResult parseListUntil(BLIFToken::Kind rightToken,
                             const std::function<ParseResult()> &parseElement);

  //===--------------------------------------------------------------------===//
  // Common Parser Rules
  //===--------------------------------------------------------------------===//

  /// Parse 'intLit' into the specified value.
  ParseResult parseIntLit(APInt &result, const Twine &message);
  ParseResult parseIntLit(int64_t &result, const Twine &message);
  ParseResult parseIntLit(int32_t &result, const Twine &message);

  // Parse the 'id' grammar, which is an identifier or an allowed keyword.
  ParseResult parseId(StringRef &result, const Twine &message);
  ParseResult parseId(StringAttr &result, const Twine &message);
  // Parse the 'id' grammar or the string literal "NIL"
  ParseResult parseIdOrNil(StringRef &result, const Twine &message);
  ParseResult parseIdOrNil(StringAttr &result, const Twine &message);

  ParseResult parseIdList(SmallVectorImpl<StringRef> &result,
                          const Twine &message, unsigned minCount = 0);

private:
  BLIFParser(const BLIFParser &) = delete;
  void operator=(const BLIFParser &) = delete;

  /// The context in which we are parsing.
  MLIRContext *context;

  /// BLIFParser is subclassed and reinstantiated.  Do not add additional
  /// non-trivial state here, add it to SharedParserConstants.
  BLIFLexer &lexer;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

InFlightDiagnostic BLIFParser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(BLIFToken::error))
    diag.abandon();
  return diag;
}

InFlightDiagnostic BLIFParser::emitWarning(SMLoc loc, const Twine &message) {
  return mlir::emitWarning(translateLocation(loc), message);
}

//===----------------------------------------------------------------------===//
// Token Parsing
//===----------------------------------------------------------------------===//

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult BLIFParser::parseToken(BLIFToken::Kind expectedToken,
                                   const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

//===--------------------------------------------------------------------===//
// Common Parser Rules
//===--------------------------------------------------------------------===//

/// id  ::= Id
///
/// Parse the 'id' grammar, which is a trivial string.  On
/// success, this returns the identifier in the result attribute.
ParseResult BLIFParser::parseId(StringRef &result, const Twine &message) {
  switch (getToken().getKind()) {
  // The most common case is an identifier.
  case BLIFToken::identifier:
    result = getTokenSpelling();
    consumeToken();
    return success();

  default:
    emitError(message);
    return failure();
  }
}

ParseResult BLIFParser::parseId(StringAttr &result, const Twine &message) {
  StringRef name;
  if (parseId(name, message))
    return failure();

  result = StringAttr::get(getContext(), name);
  return success();
}

ParseResult BLIFParser::parseIdList(SmallVectorImpl<StringRef> &result,
                                    const Twine &message, unsigned minCount) {
  while (true) {
    switch (getToken().getKind()) {
    // The most common case is an identifier.
    case BLIFToken::identifier:
      result.push_back(getTokenSpelling());
      consumeToken();
      if (minCount)
        --minCount;
      break;
    default:
      if (minCount) {
        emitError(message);
        return failure();
      }
      return success();
    }
  }
}

//===----------------------------------------------------------------------===//
// BLIFModelParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic and state for parsing statements, suites, and
/// similar module body constructs.
struct BLIFModelParser : public BLIFParser {
  explicit BLIFModelParser(Block &blockToInsertInto, BLIFLexer &lexer)
      : BLIFParser(blockToInsertInto.getParentOp()->getContext(), lexer),
        builder(UnknownLoc::get(getContext()), getContext()) {
    builder.setInsertionPointToEnd(&blockToInsertInto);
  }

  ParseResult parseSimpleStmt();
  ParseResult parseSimpleStmtBlock();

private:
  ParseResult parseSimpleStmtImpl();

  // Exp Parsing
  ParseResult parseExpImpl(Value &result, const Twine &message,
                           bool isLeadingStmt);
  ParseResult parseExp(Value &result, const Twine &message) {
    return parseExpImpl(result, message, /*isLeadingStmt:*/ false);
  }

  ParseResult parseLogicGate();
  ParseResult parseLatch();
  ParseResult parseLibraryLogicGate();
  ParseResult parseLibraryLatch();
  ParseResult parseModelReference();
  ParseResult parseModelCmdImpl();
  ParseResult parseModelCmd();
  ParseResult parseModelBody();

  // Helper to fetch a module referenced by an instance-like statement.
  Value getReferencedModel(SMLoc loc, StringRef modelName);

  // The builder to build into.
  ImplicitLocOpBuilder builder;
};

} // end anonymous namespace

Value BLIFModelParser::getReferencedModel(SMLoc loc, StringRef modelName) {
  return {};
}

/// logicgate ::= '.names' in* out NEWLINE single_output_cover*
ParseResult BLIFModelParser::parseLogicGate() {
  auto startTok = consumeToken(BLIFToken::kw_names);
  auto loc = startTok.getLoc();
  SmallVector<StringRef> inputs;
  std::string output;
  if (parseIdList(inputs, "expected input list", 1))
    return failure();
  output = inputs.back();
  inputs.pop_back();
  // TODO: READ THE TRUTH TABLE
  // builder.create<LogicGateOp>(loc, inputs, output);
  return success();
}

/// latch ::= '.latch' id_in id_out [latch_type [id | "NIL"]]? [ '0' | '1' | '2'
/// | '3']?
ParseResult BLIFModelParser::parseLatch() {
  auto startTok = consumeToken(BLIFToken::kw_latch);
  auto loc = startTok.getLoc();
  StringRef input;
  StringRef output;
  // latchType lType;
  std::string clock;
  int init_val = 3; // default
  if (parseId(input, "Expected input signal") ||
      parseId(output, "Expected output signal"))
    return failure();
  // TODO: Latch type
  // if (parseOptionalInt(init_val))
  //  return failure();
  if (init_val < 0 || init_val > 3)
    return emitError("invalid initial latch value '") << init_val << "'",
           failure();
  //  builder.create<LatchGateOp>(loc, input, output/*, lType*/, clock,
  //  init_val);
  return success();
}

// liblogic ::= '.gate' id formal-actual-list
ParseResult BLIFModelParser::parseLibraryLogicGate() { return failure(); }

// liblatch ::= '.mlatch' id formal-actual-list [id | "NIL"] [ '0' | '1' | '2' |
// '3']?
ParseResult BLIFModelParser::parseLibraryLatch() { return failure(); }

// mref ::= '.subskt' id formal-actual-list
ParseResult BLIFModelParser::parseModelReference() { return failure(); }

/// model_cmd ::= stmt
///
/// stmt ::= .names
///      ::= .latch
///      ::= .gate
///      ::= .mlatch
///      ::= .subskt
///
ParseResult BLIFModelParser::parseModelCmdImpl() {
  auto kind = getToken().getKind();
  switch (kind) {
  // Statements.
  case BLIFToken::kw_names:
    return parseLogicGate();
  case BLIFToken::kw_latch:
    return parseLatch();
  case BLIFToken::kw_gate:
    return parseLibraryLogicGate();
  case BLIFToken::kw_mlatch:
    return parseLibraryLatch();
  case BLIFToken::kw_subskt:
    return parseModelReference();
  default:
    return emitError("unexpected token in model command"), failure();
  }
}

ParseResult BLIFModelParser::parseModelCmd() {
  // locationProcessor.startStatement();
  auto result = parseModelCmdImpl();
  // locationProcessor.endStatement(*this);
  return result;
}

// Parse the body of this module.
ParseResult BLIFModelParser::parseModelBody() {
  //  auto &body = moduleOp->getRegion(0).front();

  while (true) {
    // The outer level parser can handle these tokens.
    if (getToken().isAny(BLIFToken::eof, BLIFToken::error, BLIFToken::kw_end))
      return success();

    // Let the statement parser handle this.
    if (parseModelCmd())
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BLIFFileParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the parser, namely models
struct BLIFFileParser : public BLIFParser {
  explicit BLIFFileParser(BLIFLexer &lexer, ModuleOp mlirModule)
      : BLIFParser(mlirModule.getContext(), lexer), mlirModule(mlirModule),
        builder(UnknownLoc::get(getContext()), getContext()) {
    builder.setInsertionPointToStart(mlirModule.getBody());
  }

  ParseResult parseFile();

private:
  ParseResult parseModel();

  ModuleOp mlirModule;
  ImplicitLocOpBuilder builder;
};

} // end anonymous namespace

/// model ::= '.model' id '\n' [input_line, output_line, clock_line]*
/// model_cmds* `.end`
ParseResult BLIFFileParser::parseModel() {
  StringAttr name;
  auto modLoc = getToken().getLoc();
  SmallVector<StringRef> inputs, outputs, clocks;
  consumeToken(BLIFToken::kw_model);
  if (parseId(name, "expected model name"))
    return failure();
  while (getToken().isModelHeaderKeyword()) {
    switch (getToken().getKind()) {
    case BLIFToken::kw_inputs:
      consumeToken(BLIFToken::kw_inputs);
      if (parseIdList(inputs, "expected input list", 1))
        return failure();
      break;
    case BLIFToken::kw_outputs:
      consumeToken(BLIFToken::kw_outputs);
      if (parseIdList(outputs, "expected output list", 1))
        return failure();
      break;
    case BLIFToken::kw_clock:
      consumeToken(BLIFToken::kw_clock);
      if (parseIdList(clocks, "expected clock list", 1))
        return failure();
      break;
    default:
      break;
    }
  }
  // Create the model
  auto m = builder.create<ModelOp>(name, inputs, outputs, clocks);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&m.getBody().back());
  builder.create<OutputOp>(ValueRange());

  if (/*parseModelBody() ||*/ parseToken(BLIFToken::kw_end, "expected .end"))
    return failure();

  return success();
}

/// file ::= model+
ParseResult BLIFFileParser::parseFile() {

  while (!getToken().is(BLIFToken::eof)) {
    // Parse the next model in the file
    if (getToken().is(BLIFToken::kw_model)) {
      if (parseModel())
        return failure();
    } else {
      emitError("unexpected token in file");
      return failure();
    }
  }

  // Parse any contained modules.
  while (true) {
    switch (getToken().getKind()) {
    // If we got to the end of the file, then we're done.
    case BLIFToken::eof:
      goto DoneParsing;

    // If we got an error token, then the lexer already emitted an error,
    // just stop.  We could introduce error recovery if there was demand for
    // it.
    case BLIFToken::error:
      return failure();

    default:
      emitError("unexpected token in circuit");
      return failure();

    case BLIFToken::kw_model:
      if (parseModel())
        return failure();
      break;
    }
  }

DoneParsing:
  // TODO: Fixup references
  return success();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified .fir file into the specified MLIR context.
mlir::OwningOpRef<mlir::ModuleOp>
circt::blif::importBLIFFile(SourceMgr &sourceMgr, MLIRContext *context,
                            mlir::TimingScope &ts, BLIFParserOptions options) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  context->loadDialect<BLIFDialect, hw::HWDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<mlir::ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(),
                          /*line=*/0,
                          /*column=*/0)));
  // SharedParserConstants state(context, options);
  BLIFLexer lexer(sourceMgr, context);
  if (BLIFFileParser(lexer, *module).parseFile())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto circuitVerificationTimer = ts.nest("Verify circuit");
  if (failed(verify(*module)))
    return {};

  return module;
}

void circt::blif::registerFromBLIFFileTranslation() {
  static mlir::TranslateToMLIRRegistration fromBLIF(
      "import-blif", "import .blif",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        return importBLIFFile(sourceMgr, context, ts);
      });
}
