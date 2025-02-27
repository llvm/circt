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

  // Parse the 'id' grammar, which is an identifier or an allowed keyword.
  ParseResult parseId(StringRef &result, const Twine &message);
  ParseResult parseId(StringAttr &result, const Twine &message);

  // Parse the 'id' grammar or the string literal "NIL"
  ParseResult parseIdOrNil(StringRef &result, const Twine &message);
  ParseResult parseIdOrNil(StringAttr &result, const Twine &message);

  ParseResult parseIdList(SmallVectorImpl<StringRef> &result,
                          const Twine &message, unsigned minCount = 0);

  ParseResult parseOptionalInt(int &result);

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

/// id  ::= Id
///
/// Parse the 'id' grammar, which is a trivial string.  On
/// success, this returns the identifier in the result attribute.
ParseResult BLIFParser::parseOptionalInt(int &result) {
  if (getToken().getKind() == BLIFToken::integer) {
    if (getTokenSpelling().getAsInteger(10, result)) {
      emitError("invalid integer value");
      return failure();
    }
    consumeToken();
    return success();
  }
  return success();
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
    builder.setInsertionPoint(&blockToInsertInto, blockToInsertInto.begin());
    unresolved = builder
                     .create<mlir::UnrealizedConversionCastOp>(
                         builder.getIntegerType(1), ValueRange())
                     .getResult(0);
    ModelOp model = cast<ModelOp>(blockToInsertInto.getParentOp());
    int idx = 0;
    for (auto port : model.getModuleType().getPorts()) {
      if (port.dir == hw::ModulePort::Direction::Input)
        setNamedValue(port.name.getValue(),
                      blockToInsertInto.getArgument(idx++));
    }
  }

  ParseResult parseModelBody();
  ParseResult fixupUnresolvedValues();
  ParseResult fixupOutput();

private:
  ParseResult parseSimpleStmtImpl();

  ParseResult parseOptionalTruthLine(StringRef &input, bool &output);
  ParseResult parseTruthTable(SmallVectorImpl<StringRef> &inputs,
                              SmallVectorImpl<bool> &outputs);
  ParseResult parseLogicGate();
  ParseResult parseOptionalLatchType(LatchModeEnum &lType, StringRef &name);
  ParseResult parseLatch();
  ParseResult parseLibraryLogicGate();
  ParseResult parseLibraryLatch();
  ParseResult parseModelReference();
  ParseResult parseModelCmdImpl();
  ParseResult parseModelCmd();

  // Helper to fetch a module referenced by an instance-like statement.
  Value getReferencedModel(SMLoc loc, StringRef modelName);

  /// Find the value corrosponding to the named identifier.
  Value getNamedValue(StringRef name);

  /// Set the value corrosponding to the named identifier.
  void setNamedValue(StringRef name, Value value);

  /// Remember to fix an op whose value could not be resolved
  void addFixup(Operation *op, unsigned idx, StringRef name);

  // The builder to build into.
  ImplicitLocOpBuilder builder;

  /// Keep track of names;
  DenseMap<StringRef, Value> namedValues;

  struct fixup {
    Operation *op;
    unsigned idx;
  };

  DenseMap<StringRef, SmallVector<fixup>> fixups;

  Value unresolved;
};

} // end anonymous namespace

Value BLIFModelParser::getNamedValue(StringRef name) {
  auto val = namedValues.lookup(name);
  return val ? val : unresolved;
}

void BLIFModelParser::setNamedValue(StringRef name, Value value) {
  assert(!namedValues[name] && "name already exists");
  namedValues[name] = value;
}

void BLIFModelParser::addFixup(Operation *op, unsigned idx, StringRef name) {
  fixups[name].push_back({op, idx});
}

Value BLIFModelParser::getReferencedModel(SMLoc loc, StringRef modelName) {
  return {};
}

ParseResult BLIFModelParser::parseOptionalTruthLine(StringRef &input,
                                                    bool &output) {
  switch (getToken().getKind()) {
  case BLIFToken::cover:
  case BLIFToken::integer:
    input = getTokenSpelling();
    consumeToken();
    break;
  default:
    return failure();
  }

  switch (getToken().getKind()) {
  case BLIFToken::integer:
    if (getTokenSpelling() == "0") {
      consumeToken();
      output = false;
      return success();
    }
    if (getTokenSpelling() == "1") {
      consumeToken();
      output = true;
      return success();
    }
    break;
  default:
    break;
  }

  return failure();
}

ParseResult BLIFModelParser::parseTruthTable(SmallVectorImpl<StringRef> &inputs,
                                             SmallVectorImpl<bool> &outputs) {
  StringRef input;
  bool output;
  while (succeeded(parseOptionalTruthLine(input, output))) {
    inputs.push_back(input);
    outputs.push_back(output);
  }

  return success();
}

/// logicgate ::= '.names' in* out NEWLINE single_output_cover*
ParseResult BLIFModelParser::parseLogicGate() {
  consumeToken(BLIFToken::kw_names);
  SmallVector<StringRef> inputs;
  StringRef output;
  if (parseIdList(inputs, "expected input list", 1))
    return failure();
  output = inputs.back();
  inputs.pop_back();
  // Deal with truth table
  SmallVector<StringRef> covers;
  SmallVector<bool> sets;
  if (parseTruthTable(covers, sets))
    return failure();

  SmallVector<SmallVector<int8_t>> truthTable;
  for (auto [i, o] : llvm::zip_equal(covers, sets)) {
    SmallVector<int8_t> row;
    for (auto c : i)
      if (c == '0')
        row.push_back(0);
      else if (c == '1')
        row.push_back(1);
      else if (c == '-')
        row.push_back(2);
      else
        return emitError("invalid truth table value '") << c << "'";
    row.push_back(o);
    truthTable.push_back(row);
  }

  // Resolve Inputs
  SmallVector<Value> inputValues;
  for (auto input : inputs)
    inputValues.push_back(getNamedValue(input));
  // BuildOp
  auto op = builder.create<LogicGateOp>(builder.getIntegerType(1), truthTable,
                                        inputValues);
  // Record output name
  setNamedValue(output, op.getResult());
  // Record Fixups
  for (auto [idx, name, val] : llvm::enumerate(inputs, inputValues)) {
    if (val == unresolved)
      addFixup(op, idx, name);
  }
  return success();
}

ParseResult BLIFModelParser::parseOptionalLatchType(LatchModeEnum &lType,
                                                    StringRef &clkName) {
  if (getToken().getKind() == BLIFToken::identifier) {
    lType = llvm::StringSwitch<LatchModeEnum>(getToken().getSpelling())
                .Case("fe", LatchModeEnum::FallingEdge)
                .Case("re", LatchModeEnum::RisingEdge)
                .Case("ah", LatchModeEnum::ActiveHigh)
                .Case("al", LatchModeEnum::ActiveLow)
                .Case("as", LatchModeEnum::Asynchronous)
                .Default(LatchModeEnum::Unspecified);
    if (lType == LatchModeEnum::Unspecified) {
      return success();
    }
    consumeToken();
    return parseId(clkName, "Expected clock signal");
  }
  return success();
}

/// latch ::= '.latch' id_in id_out [latch_type [id | "NIL"]]? [ '0' | '1' | '2'
/// | '3']?
ParseResult BLIFModelParser::parseLatch() {
  consumeToken(BLIFToken::kw_latch);
  StringRef input;
  StringRef output;
  // latchType lType;
  std::string clock;
  int init_val = 3; // default
  if (parseId(input, "Expected input signal") ||
      parseId(output, "Expected output signal"))
    return failure();

  LatchModeEnum lType = LatchModeEnum::Unspecified;
  StringRef clkName;
  if (parseOptionalLatchType(lType, clkName) || parseOptionalInt(init_val))
    return emitError("invalid latch type"), failure();

  if (init_val < 0 || init_val > 3)
    return emitError("invalid initial latch value '") << init_val << "'",
           failure();

  Value inVal = getNamedValue(input);
  Value clkVal = ((clkName == "NIL") | (lType == LatchModeEnum::Unspecified))
                     ? Value()
                     : getNamedValue(clkName);

  auto op = builder.create<LatchGateOp>(builder.getIntegerType(1), inVal, lType,
                                        clkVal, init_val);
  // Record output name
  setNamedValue(output, op.getResult());
  // Record Fixups
  if (inVal == unresolved)
    addFixup(op, 0, input);

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

ParseResult BLIFModelParser::fixupUnresolvedValues() {
  for (auto &fixup : fixups) {
    Value val = getNamedValue(fixup.first);
    if (val == unresolved)
      return emitError("unresolved value '") << fixup.first << "'";
    for (auto [op, idx] : fixup.second) {
      op->setOperand(idx, val);
    }
  }
  return success();
}

ParseResult BLIFModelParser::fixupOutput() {
  auto model = cast<ModelOp>(builder.getBlock()->getParentOp());
  SmallVector<Value> outputs;
  for (auto port : model.getModuleType().getPorts()) {
    if (port.dir == hw::ModulePort::Direction::Output) {
      auto val = getNamedValue(port.name.getValue());
      if (val == unresolved)
        emitWarning("unresolved output '") << port.name.getValue() << "'";
      outputs.push_back(val);
    }
  }
  builder.create<OutputOp>(outputs);
  if (unresolved && unresolved.use_empty()) {
    unresolved.getDefiningOp()->erase();
    unresolved = nullptr;
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
    case BLIFToken::kw_input: {
      consumeToken(BLIFToken::kw_input);
      StringRef tmp;
      if (parseId(tmp, "expected input"))
        return failure();
      inputs.push_back(tmp);
      break;
    }
    case BLIFToken::kw_outputs:
      consumeToken(BLIFToken::kw_outputs);
      if (parseIdList(outputs, "expected output list", 1))
        return failure();
      break;
    case BLIFToken::kw_output: {
      consumeToken(BLIFToken::kw_output);
      StringRef tmp;
      if (parseId(tmp, "expected output"))
        return failure();
      outputs.push_back(tmp);
      break;
    }
    case BLIFToken::kw_clocks:
      consumeToken(BLIFToken::kw_clocks);
      if (parseIdList(clocks, "expected clock list", 1))
        return failure();
      break;
    case BLIFToken::kw_clock: {
      consumeToken(BLIFToken::kw_clock);
      StringRef tmp;
      if (parseId(tmp, "expected clock"))
        return failure();
      clocks.push_back(tmp);
      break;
    }
    default:
      break;
    }
  }
  // Create the model
  auto m = builder.create<ModelOp>(name, inputs, outputs, clocks);
  OpBuilder::InsertionGuard guard(builder);
  BLIFModelParser bodyParser(m.getBody().back(), getLexer());
  if (bodyParser.parseModelBody() || bodyParser.fixupUnresolvedValues() ||
      bodyParser.fixupOutput() ||
      parseToken(BLIFToken::kw_end, "expected .end"))
    return failure();

  // Fix up outputs

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
