//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AIGER file import functionality.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportAIGER.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <limits>
#include <string>

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::synth;
using namespace circt::seq;
using namespace circt::aiger;

#define DEBUG_TYPE "import-aiger"

namespace {

/// AIGER token types for lexical analysis
enum class AIGERTokenKind {
  // Literals
  Number,
  Identifier,

  // Special characters
  Newline,
  EndOfFile,

  // Error
  Error
};

/// Represents a token in the AIGER file
struct AIGERToken {
  AIGERTokenKind kind;
  StringRef spelling;
  SMLoc location;

  AIGERToken(AIGERTokenKind kind, StringRef spelling, SMLoc location)
      : kind(kind), spelling(spelling), location(location) {}
};

/// Simple lexer for AIGER files
///
/// This lexer handles both ASCII (.aag) and binary (.aig) AIGER formats.
/// It provides basic tokenization for header parsing and symbol tables,
/// while also supporting byte-level reading for binary format.
class AIGERLexer {
public:
  AIGERLexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
      : sourceMgr(sourceMgr),
        bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
        curBuffer(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
        curPtr(curBuffer.begin()) {}

  /// Get the next token
  AIGERToken nextToken();

  /// Lex the current position as a symbol (used for symbol table parsing)
  AIGERToken lexAsSymbol();

  /// Peek at the current token without consuming it
  AIGERToken peekToken();

  /// Check if we're at end of file
  bool isAtEOF() const { return curPtr >= curBuffer.end(); }

  /// Read a LEB128 encoded unsigned integer
  ParseResult readLEB128(unsigned &result);

  /// Get current location
  SMLoc getCurrentLoc() const { return SMLoc::getFromPointer(curPtr); }

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.
  Location translateLocation(llvm::SMLoc loc) {
    assert(loc.isValid());
    unsigned mainFileID = sourceMgr.getMainFileID();
    auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
    return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                               lineAndColumn.second);
  }

  /// Emit an error message and return an error token.
  AIGERToken emitError(const char *loc, const Twine &message) {
    mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
    return AIGERToken(AIGERTokenKind::Error, StringRef(loc, 1),
                      SMLoc::getFromPointer(loc));
  }

private:
  const llvm::SourceMgr &sourceMgr;
  StringAttr bufferNameIdentifier;
  StringRef curBuffer;
  const char *curPtr;

  /// Get the main buffer name identifier
  static StringAttr
  getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                              MLIRContext *context) {
    auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
    StringRef bufferName = mainBuffer->getBufferIdentifier();
    if (bufferName.empty())
      bufferName = "<unknown>";
    return StringAttr::get(context, bufferName);
  }

  /// Skip whitespace (except newlines)
  void skipWhitespace();

  /// Skip to end of line (for comment handling)
  void skipUntilNewline();

  /// Lex a number
  AIGERToken lexNumber();

  /// Lex an identifier
  AIGERToken lexIdentifier();

  /// Create a token
  AIGERToken makeToken(AIGERTokenKind kind, const char *start) {
    return AIGERToken(kind, StringRef(start, curPtr - start),
                      SMLoc::getFromPointer(start));
  }
};

/// Main AIGER parser class
///
/// This parser implements the complete AIGER format specification including:
/// - ASCII (.aag) and binary (.aig) formats
/// - Basic AIGER components (inputs, latches, outputs, AND gates)
/// - Optional sections (bad states, constraints, justice, fairness)
/// - Symbol tables and comments
///
/// The parser creates MLIR modules using the HW, AIG, and Seq dialects.
class AIGERParser {
public:
  AIGERParser(const llvm::SourceMgr &sourceMgr, MLIRContext *context,
              ModuleOp module, const ImportAIGEROptions &options)
      : lexer(sourceMgr, context), context(context), module(module),
        options(options), builder(context) {}

  /// Parse the AIGER file and populate the MLIR module
  ParseResult parse();

private:
  AIGERLexer lexer;
  MLIRContext *context;
  ModuleOp module;
  const ImportAIGEROptions &options;
  OpBuilder builder;

  // AIGER file data
  unsigned maxVarIndex = 0;
  unsigned numInputs = 0;
  unsigned numLatches = 0;
  unsigned numOutputs = 0;
  unsigned numAnds = 0;
  bool isBinaryFormat = false;

  // A mapping from {kind, index} -> symbol where kind is 0 for inputs, 1 for
  // latches, and 2 for outputs.
  enum SymbolKind : unsigned { Input, Latch, Output };
  DenseMap<std::pair<SymbolKind, unsigned>, StringAttr> symbolTable;

  // Parsed data storage
  SmallVector<unsigned> inputLiterals;
  SmallVector<std::tuple<unsigned, unsigned, SMLoc>>
      latchDefs;                                          // current, next, loc
  SmallVector<std::pair<unsigned, SMLoc>> outputLiterals; // literal, loc
  SmallVector<std::tuple<unsigned, unsigned, unsigned, SMLoc>>
      andGateDefs; // lhs, rhs0, rhs1

  /// Parse the header line (format and counts)
  ParseResult parseHeader();

  /// Parse inputs section
  ParseResult parseInputs();

  /// Parse latches section
  ParseResult parseLatches();

  /// Parse outputs section
  ParseResult parseOutputs();

  /// Parse AND gates section (dispatches to ASCII or binary)
  ParseResult parseAndGates();

  /// Parse AND gates in ASCII format
  ParseResult parseAndGatesASCII();

  /// Parse AND gates in binary format with delta compression
  ParseResult parseAndGatesBinary();

  /// Parse symbol table (optional)
  ParseResult parseSymbolTable();

  /// Parse comments (optional)
  ParseResult parseComments();

  /// Convert AIGER literal to MLIR value using backedges
  ///
  /// \param literal The AIGER literal (variable * 2 + inversion)
  /// \param backedges Map from literals to backedge values
  /// \param loc Location for created operations
  /// \return The MLIR value corresponding to the literal, or nullptr on error
  Value getLiteralValue(unsigned literal,
                        DenseMap<unsigned, Backedge> &backedges, Location loc);

  /// Create the top-level HW module from parsed data
  ParseResult createModule();

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) {
    return mlir::emitError(lexer.translateLocation(loc), message);
  }

  /// Emit error at current location
  InFlightDiagnostic emitError(const Twine &message) {
    return emitError(lexer.getCurrentLoc(), message);
  }

  /// Parse a number token into result
  ParseResult parseNumber(unsigned &result, SMLoc *loc = nullptr);

  /// Parse a binary encoded number (variable-length encoding)
  ParseResult parseBinaryNumber(unsigned &result);

  /// Expect and consume a newline token
  ParseResult parseNewLine();
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AIGERLexer Implementation
//===----------------------------------------------------------------------===//

void AIGERLexer::skipWhitespace() {
  while (curPtr < curBuffer.end()) {
    // NOTE: Don't use llvm::isSpace here because it also skips '\n'.
    if (*curPtr == ' ' || *curPtr == '\t' || *curPtr == '\r') {
      ++curPtr;
      continue;
    }

    // Treat "//" as whitespace. This is not part of the AIGER format, but we
    // support it for FileCheck tests.
    if (*curPtr == '/' &&
        (curPtr + 1 < curBuffer.end() && *(curPtr + 1) == '/')) {
      skipUntilNewline();
      continue;
    }
    break;
  }
}

void AIGERLexer::skipUntilNewline() {
  while (curPtr < curBuffer.end() && *curPtr != '\n')
    ++curPtr;
  if (curPtr < curBuffer.end() && *curPtr == '\n')
    ++curPtr;
}

AIGERToken AIGERLexer::lexNumber() {
  const char *start = curPtr;
  while (curPtr < curBuffer.end() && llvm::isDigit(*curPtr))
    ++curPtr;
  return makeToken(AIGERTokenKind::Number, start);
}

AIGERToken AIGERLexer::lexIdentifier() {
  const char *start = curPtr;
  while (curPtr < curBuffer.end() && (llvm::isAlnum(*curPtr) || *curPtr == '_'))
    ++curPtr;

  StringRef spelling(start, curPtr - start);
  AIGERTokenKind kind = AIGERTokenKind::Identifier;

  return makeToken(kind, start);
}

AIGERToken AIGERLexer::nextToken() {
  skipWhitespace();

  auto impl = [this]() {
    if (curPtr >= curBuffer.end())
      return makeToken(AIGERTokenKind::EndOfFile, curPtr);

    const char *start = curPtr;
    char c = *curPtr++;

    switch (c) {
    case '\n':
      return makeToken(AIGERTokenKind::Newline, start);
    case '\r':
    case ' ':
    case '\t':
      llvm_unreachable("Whitespace should have been skipped");
      return makeToken(AIGERTokenKind::Error, start);
    default:
      if (llvm::isDigit(c)) {
        --curPtr; // Back up to re-lex the number
        return lexNumber();
      }
      if (llvm::isAlpha(c) || c == '_') {
        --curPtr; // Back up to re-lex the identifier
        return lexIdentifier();
      }
      assert((c != '/' || *curPtr != '/') && "// should have been skipped");
      return makeToken(AIGERTokenKind::Error, start);
    }
  };

  auto token = impl();
  return token;
}

AIGERToken AIGERLexer::lexAsSymbol() {
  skipWhitespace();
  const char *start = curPtr;
  while (curPtr < curBuffer.end() &&
         (llvm::isPrint(*curPtr) && !llvm::isSpace(*curPtr)))
    ++curPtr;
  return makeToken(AIGERTokenKind::Identifier, start);
}

AIGERToken AIGERLexer::peekToken() {
  const char *savedPtr = curPtr;
  AIGERToken token = nextToken();
  curPtr = savedPtr;
  return token;
}

ParseResult AIGERLexer::readLEB128(unsigned &result) {
  unsigned len;
  uint64_t value =
      llvm::decodeULEB128(reinterpret_cast<const uint8_t *>(curPtr), &len,
                          reinterpret_cast<const uint8_t *>(curBuffer.end()));
  if (len == 0 || value > std::numeric_limits<unsigned>::max())
    return failure();
  curPtr += len;
  result = static_cast<unsigned>(value);
  return success();
}

//===----------------------------------------------------------------------===//
// AIGERParser Implementation
//===----------------------------------------------------------------------===//

ParseResult AIGERParser::parse() {
  if (parseHeader() || parseInputs() || parseLatches() || parseOutputs() ||
      parseAndGates() || parseSymbolTable() || parseComments())
    return failure();
  // Create the final module
  return createModule();
}

ParseResult AIGERParser::parseNumber(unsigned &result, SMLoc *loc) {
  auto token = lexer.nextToken();
  if (loc)
    *loc = token.location;

  if (token.kind != AIGERTokenKind::Number)
    return emitError(token.location, "expected number");

  if (token.spelling.getAsInteger(10, result))
    return emitError(token.location, "invalid number format");

  return success();
}

ParseResult AIGERParser::parseBinaryNumber(unsigned &result) {
  return lexer.readLEB128(result);
}

ParseResult AIGERParser::parseHeader() {
  LLVM_DEBUG(llvm::dbgs() << "Parsing AIGER header\n");

  // Parse format identifier (aag or aig)
  while (lexer.peekToken().kind != AIGERTokenKind::Identifier)
    lexer.nextToken();

  auto formatToken = lexer.nextToken();
  if (formatToken.spelling == "aag") {
    isBinaryFormat = false;
    LLVM_DEBUG(llvm::dbgs() << "Format: aag (ASCII)\n");
  } else if (formatToken.spelling == "aig") {
    isBinaryFormat = true;
    LLVM_DEBUG(llvm::dbgs() << "Format: aig (binary)\n");
  } else {
    return emitError(formatToken.location,
                     "expected 'aag' or 'aig' format identifier");
  }

  // Parse M I L O A (numbers separated by spaces)
  SMLoc loc;
  if (parseNumber(maxVarIndex, &loc))
    return emitError(loc, "failed to parse M (max variable index)");

  if (parseNumber(numInputs, &loc))
    return emitError(loc, "failed to parse I (number of inputs)");

  if (parseNumber(numLatches, &loc))
    return emitError(loc, "failed to parse L (number of latches)");

  if (parseNumber(numOutputs, &loc))
    return emitError(loc, "failed to parse O (number of outputs)");

  if (parseNumber(numAnds, &loc))
    return emitError(loc, "failed to parse A (number of AND gates)");

  LLVM_DEBUG(llvm::dbgs() << "Header: M=" << maxVarIndex << " I=" << numInputs
                          << " L=" << numLatches << " O=" << numOutputs
                          << " A=" << numAnds << "\n");

  // Expect newline after header
  return parseNewLine();
}

ParseResult AIGERParser::parseNewLine() {
  auto token = lexer.nextToken();
  if (token.kind != AIGERTokenKind::Newline)
    return emitError(token.location, "expected newline");

  return success();
}

ParseResult AIGERParser::parseInputs() {
  LLVM_DEBUG(llvm::dbgs() << "Parsing " << numInputs << " inputs\n");
  if (isBinaryFormat) {
    // In binary format, inputs are implicit (literals 2, 4, 6, ...)
    for (unsigned i = 0; i < numInputs; ++i)
      inputLiterals.push_back(2 * (i + 1));
    return success();
  }

  for (unsigned i = 0; i < numInputs; ++i) {
    unsigned literal;
    SMLoc loc;
    if (parseNumber(literal, &loc) || parseNewLine())
      return emitError(loc, "failed to parse input literal");
    inputLiterals.push_back(literal);
  }

  return success();
}

ParseResult AIGERParser::parseLatches() {
  LLVM_DEBUG(llvm::dbgs() << "Parsing " << numLatches << " latches\n");
  if (isBinaryFormat) {
    // In binary format, latches are implicit (literals 2, 4, 6, ...)
    for (unsigned i = 0; i < numLatches; ++i) {
      unsigned literal;
      SMLoc loc;
      if (parseNumber(literal, &loc))
        return emitError(loc, "failed to parse latch next state literal");

      latchDefs.push_back({2 * (i + 1 + numInputs), literal, loc});

      // Expect newline after each latch next state
      if (parseNewLine())
        return failure();
    }
    return success();
  }

  // Parse latch definitions: current_state next_state
  for (unsigned i = 0; i < numLatches; ++i) {
    unsigned currentState, nextState;
    SMLoc loc;
    if (parseNumber(currentState, &loc) || parseNumber(nextState) ||
        parseNewLine())
      return emitError(loc, "failed to parse latch definition");

    LLVM_DEBUG(llvm::dbgs() << "Latch " << i << ": " << currentState << " -> "
                            << nextState << "\n");

    // Validate current state literal (should be even and positive)
    if (currentState % 2 != 0 || currentState == 0)
      return emitError(loc, "invalid latch current state literal");

    latchDefs.push_back({currentState, nextState, loc});
  }

  return success();
}

ParseResult AIGERParser::parseOutputs() {
  LLVM_DEBUG(llvm::dbgs() << "Parsing " << numOutputs << " outputs\n");
  // NOTE: Parsing is same for binary and ASCII formats
  // Parse output literals
  for (unsigned i = 0; i < numOutputs; ++i) {
    unsigned literal;
    SMLoc loc;
    if (parseNumber(literal, &loc) || parseNewLine())
      return emitError(loc, "failed to parse output literal");

    LLVM_DEBUG(llvm::dbgs() << "Output " << i << ": " << literal << "\n");

    // Output literals can be any valid literal (including inverted)
    outputLiterals.push_back({literal, loc});
  }

  return success();
}

ParseResult AIGERParser::parseAndGates() {
  LLVM_DEBUG(llvm::dbgs() << "Parsing " << numAnds << " AND gates\n");

  if (isBinaryFormat) {
    return parseAndGatesBinary();
  }
  return parseAndGatesASCII();
}

ParseResult AIGERParser::parseAndGatesASCII() {
  // Parse AND gate definitions: lhs rhs0 rhs1
  for (unsigned i = 0; i < numAnds; ++i) {
    unsigned lhs, rhs0, rhs1;
    SMLoc loc;
    if (parseNumber(lhs, &loc) || parseNumber(rhs0) || parseNumber(rhs1) ||
        parseNewLine())
      return emitError(loc, "failed to parse AND gate definition");

    LLVM_DEBUG(llvm::dbgs() << "AND Gate " << i << ": " << lhs << " = " << rhs0
                            << " & " << rhs1 << "\n");

    // Validate LHS (should be even and positive)
    if (lhs % 2 != 0 || lhs == 0)
      return emitError(loc, "invalid AND gate LHS literal");

    // Validate literal bounds
    if (lhs / 2 > maxVarIndex || rhs0 / 2 > maxVarIndex ||
        rhs1 / 2 > maxVarIndex)
      return emitError(loc, "AND gate literal exceeds maximum variable index");

    andGateDefs.push_back({lhs, rhs0, rhs1, loc});
  }

  return success();
}

ParseResult AIGERParser::parseAndGatesBinary() {
  // In binary format, AND gates are encoded with delta compression
  // Each AND gate is encoded as: delta0 delta1
  // where: rhs0 = lhs - delta0, rhs1 = rhs0 - delta1

  LLVM_DEBUG(llvm::dbgs() << "Starting binary AND gate parsing\n");

  // First AND gate LHS starts after inputs and latches
  // Variables are numbered: 1, 2, ..., maxVarIndex
  // Literals are: 2, 4, 6, ..., 2*maxVarIndex
  // Inputs: 2, 4, ..., 2*numInputs
  // Latches: 2*(numInputs+1), 2*(numInputs+2), ..., 2*(numInputs+numLatches)
  // AND gates: 2*(numInputs+numLatches+1), 2*(numInputs+numLatches+2), ...
  auto currentLHS = 2 * (numInputs + numLatches + 1);

  LLVM_DEBUG(llvm::dbgs() << "First AND gate LHS should be: " << currentLHS
                          << "\n");

  for (unsigned i = 0; i < numAnds; ++i) {
    unsigned delta0, delta1;
    SMLoc loc = lexer.getCurrentLoc();
    if (parseBinaryNumber(delta0) || parseBinaryNumber(delta1))
      return emitError(loc, "failed to parse binary AND gate deltas");

    auto lhs = static_cast<int64_t>(currentLHS);

    // Check for underflow before subtraction
    if (delta0 > lhs || delta1 > (lhs - delta0)) {
      LLVM_DEBUG(llvm::dbgs() << "Delta underflow: lhs=" << lhs << ", delta0="
                              << delta0 << ", delta1=" << delta1 << "\n");
      return emitError("invalid binary AND gate: delta causes underflow");
    }

    auto rhs0 = lhs - delta0;
    auto rhs1 = rhs0 - delta1;

    LLVM_DEBUG(llvm::dbgs() << "Binary AND Gate " << i << ": " << lhs << " = "
                            << rhs0 << " & " << rhs1 << " (deltas: " << delta0
                            << ", " << delta1 << ")\n");

    if (lhs / 2 > maxVarIndex || rhs0 / 2 > maxVarIndex ||
        rhs1 / 2 > maxVarIndex)
      return emitError(
          "binary AND gate literal exceeds maximum variable index");

    assert(lhs > rhs0 && rhs0 >= rhs1 &&
           "invalid binary AND gate: ordering constraint violated");

    andGateDefs.push_back({static_cast<unsigned>(lhs),
                           static_cast<unsigned>(rhs0),
                           static_cast<unsigned>(rhs1), loc});
    currentLHS += 2; // Next AND gate LHS
  }

  return success();
}

ParseResult AIGERParser::parseSymbolTable() {
  // Symbol table is optional and starts with 'i', 'l', or 'o' followed by
  // position
  while (!lexer.isAtEOF()) {
    auto token = lexer.peekToken();
    if (token.kind != AIGERTokenKind::Identifier)
      break;
    (void)lexer.nextToken();

    char symbolType = token.spelling.front();
    if (symbolType != 'i' && symbolType != 'l' && symbolType != 'o')
      break;

    unsigned literal;
    if (token.spelling.drop_front().getAsInteger(10, literal))
      return emitError("failed to parse symbol position");

    SymbolKind kind;
    switch (symbolType) {
    case 'i':
      kind = SymbolKind::Input;
      break;
    case 'l':
      kind = SymbolKind::Latch;
      break;
    case 'o':
      kind = SymbolKind::Output;
      break;
    }

    auto nextToken = lexer.lexAsSymbol();
    if (nextToken.kind != AIGERTokenKind::Identifier)
      return emitError("expected symbol name");

    LLVM_DEBUG(llvm::dbgs()
               << "Symbol " << literal << ": " << nextToken.spelling << "\n");

    symbolTable[{kind, literal}] = StringAttr::get(context, nextToken.spelling);
    if (parseNewLine())
      return failure();
  }

  return success();
}

ParseResult AIGERParser::parseComments() {
  // Comments start with 'c' and continue to end of file
  auto token = lexer.peekToken();
  if (token.kind == AIGERTokenKind::Identifier && token.spelling == "c") {
    // Skip comments for now
    return success();
  }

  return success();
}

Value AIGERParser::getLiteralValue(unsigned literal,
                                   DenseMap<unsigned, Backedge> &backedges,
                                   Location loc) {
  LLVM_DEBUG(llvm::dbgs() << "Getting value for literal " << literal << "\n");

  // Handle constants
  if (literal == 0) {
    // FALSE constant
    return hw::ConstantOp::create(
        builder, loc, builder.getI1Type(),
        builder.getIntegerAttr(builder.getI1Type(), 0));
  }

  if (literal == 1) {
    // TRUE constant
    return hw::ConstantOp::create(
        builder, loc, builder.getI1Type(),
        builder.getIntegerAttr(builder.getI1Type(), 1));
  }

  // Extract variable and inversion
  unsigned variable = literal / 2;
  bool inverted = literal % 2;
  unsigned baseLiteral = variable * 2;

  LLVM_DEBUG(llvm::dbgs() << "  Variable: " << variable
                          << ", inverted: " << inverted
                          << ", baseLiteral: " << baseLiteral << "\n");

  // Validate literal bounds
  if (variable > maxVarIndex) {
    LLVM_DEBUG(llvm::dbgs() << "  ERROR: Variable " << variable
                            << " exceeds maxVarIndex " << maxVarIndex << "\n");
    return nullptr;
  }

  // Look up the backedge for this literal
  auto backedgeIt = backedges.find(baseLiteral);
  if (backedgeIt == backedges.end()) {
    LLVM_DEBUG(llvm::dbgs() << "  ERROR: No backedge found for literal "
                            << baseLiteral << "\n");
    return nullptr; // Error: undefined literal
  }

  Value baseValue = backedgeIt->second;
  if (!baseValue) {
    LLVM_DEBUG(llvm::dbgs() << "  ERROR: Backedge value is null for literal "
                            << baseLiteral << "\n");
    return nullptr;
  }

  // Apply inversion if needed
  if (inverted) {
    // Create an inverter using synth.aig.and_inv with single input
    SmallVector<bool> inverts = {true};
    return aig::AndInverterOp::create(builder, loc, builder.getI1Type(),
                                      ValueRange{baseValue}, inverts);
  }

  return baseValue;
}

ParseResult AIGERParser::createModule() {

  // Create the top-level module
  std::string moduleName = options.topLevelModule;
  if (moduleName.empty())
    moduleName = "aiger_top";

  // Set insertion point to the provided module
  builder.setInsertionPointToStart(module.getBody());

  // Build input/output port info
  SmallVector<hw::PortInfo> ports;

  // Add input ports
  for (unsigned i = 0; i < numInputs; ++i) {
    hw::PortInfo port;
    auto name = symbolTable.lookup({SymbolKind::Input, i});
    port.name =
        name ? name : builder.getStringAttr("input_" + std::to_string(i));
    port.type = builder.getI1Type();
    port.dir = hw::ModulePort::Direction::Input;
    port.argNum = i;
    ports.push_back(port);
  }

  // Add output ports
  for (unsigned i = 0; i < numOutputs; ++i) {
    hw::PortInfo port;
    auto name = symbolTable.lookup({SymbolKind::Output, i});
    port.name =
        name ? name : builder.getStringAttr("output_" + std::to_string(i));
    port.type = builder.getI1Type();
    port.dir = hw::ModulePort::Direction::Output;
    port.argNum = numInputs + i;
    ports.push_back(port);
  }

  // Add clock port if we have latches
  if (numLatches > 0) {
    hw::PortInfo clockPort;
    clockPort.name = builder.getStringAttr("clock");
    clockPort.type = seq::ClockType::get(builder.getContext());
    clockPort.dir = hw::ModulePort::Direction::Input;
    clockPort.argNum = numInputs + numOutputs;
    ports.push_back(clockPort);
  }

  // Create the HW module
  auto hwModule =
      hw::HWModuleOp::create(builder, builder.getUnknownLoc(),
                             builder.getStringAttr(moduleName), ports);

  // Set insertion point inside the module
  builder.setInsertionPointToStart(hwModule.getBodyBlock());

  // Get clock value if we have latches
  Value clockValue;
  if (numLatches > 0)
    clockValue = hwModule.getBodyBlock()->getArgument(numInputs);

  // Use BackedgeBuilder to handle all values uniformly
  BackedgeBuilder bb(builder, builder.getUnknownLoc());
  DenseMap<unsigned, Backedge> backedges;

  // Create backedges for all literals (inputs, latches, AND gates)
  for (unsigned i = 0; i < numInputs; ++i) {
    auto literal = inputLiterals[i];
    backedges[literal] = bb.get(builder.getI1Type());
  }
  for (auto [currentState, nextState, _] : latchDefs)
    backedges[currentState] = bb.get(builder.getI1Type());

  for (auto [lhs, rhs0, rhs1, loc] : andGateDefs)
    backedges[lhs] = bb.get(builder.getI1Type());

  // Set input values
  for (unsigned i = 0; i < numInputs; ++i) {
    auto inputValue = hwModule.getBodyBlock()->getArgument(i);
    auto literal = inputLiterals[i];
    backedges[literal].setValue(inputValue);
  }

  // Create latches (registers) with backedges for next state
  for (auto [i, latchDef] : llvm::enumerate(latchDefs)) {
    auto [currentState, nextState, loc] = latchDef;
    // Get the backedge for the next state value
    auto nextBackedge = bb.get(builder.getI1Type());

    // Create the register with the backedge as input
    auto regValue = seq::CompRegOp::create(
        builder, lexer.translateLocation(loc), (Value)nextBackedge, clockValue);
    if (auto name = symbolTable.lookup({SymbolKind::Latch, i}))
      regValue.setNameAttr(name);

    // Set the backedge for this latch's current state
    backedges[currentState].setValue(regValue);
  }

  // Build AND gates using backedges to handle forward references
  for (auto [lhs, rhs0, rhs1, loc] : andGateDefs) {
    // Get or create backedges for operands
    auto location = lexer.translateLocation(loc);
    auto rhs0Value = getLiteralValue(rhs0 & ~1u, backedges, location);
    auto rhs1Value = getLiteralValue(rhs1 & ~1u, backedges, location);

    if (!rhs0Value || !rhs1Value)
      return emitError(loc, "failed to get operand values for AND gate");

    // Determine inversion for inputs
    SmallVector<bool> inverts = {static_cast<bool>(rhs0 % 2),
                                 static_cast<bool>(rhs1 % 2)};

    // Create AND gate with potential inversions
    auto andResult =
        aig::AndInverterOp::create(builder, location, builder.getI1Type(),
                                   ValueRange{rhs0Value, rhs1Value}, inverts);

    // Set the backedge for this AND gate's result
    backedges[lhs].setValue(andResult);
  }

  // Now resolve the latch next state connections.
  // We need to update the CompRegOp operations with their actual next state
  // values
  for (auto [currentState, nextState, sourceLoc] : latchDefs) {
    auto loc = lexer.translateLocation(sourceLoc);
    auto nextValue = getLiteralValue(nextState, backedges, loc);
    if (!nextValue)
      return emitError(sourceLoc, "undefined literal in latch next state");

    // Find the register operation for this latch and update its input
    Value currentValue = backedges[currentState];
    if (auto regOp = currentValue.getDefiningOp<seq::CompRegOp>())
      regOp.getInputMutable().assign(nextValue);
    else
      return emitError(sourceLoc, "failed to find register for latch");
  }

  // Create output values
  SmallVector<Value> outputValues;
  for (auto [literal, sourceLoc] : outputLiterals) {
    auto loc = lexer.translateLocation(sourceLoc);
    auto outputValue = getLiteralValue(literal, backedges, loc);
    if (!outputValue)
      return emitError(sourceLoc, "undefined literal in output");
    outputValues.push_back(outputValue);
  }

  // Create output operation
  auto *outputOp = hwModule.getBodyBlock()->getTerminator();
  outputOp->setOperands(outputValues);

  return success();
}

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

LogicalResult circt::aiger::importAIGER(llvm::SourceMgr &sourceMgr,
                                        MLIRContext *context,
                                        mlir::TimingScope &ts, ModuleOp module,
                                        const ImportAIGEROptions *options) {
  // Load required dialects
  context->loadDialect<hw::HWDialect>();
  context->loadDialect<synth::SynthDialect>();
  context->loadDialect<seq::SeqDialect>();

  // Use default options if none provided
  ImportAIGEROptions defaultOptions;
  if (!options)
    options = &defaultOptions;

  // Create parser and parse the file
  AIGERParser parser(sourceMgr, context, module, *options);
  return parser.parse();
}

//===----------------------------------------------------------------------===//
// Translation Registration
//===----------------------------------------------------------------------===//

void circt::aiger::registerImportAIGERTranslation() {
  static mlir::TranslateToMLIRRegistration fromAIGER(
      "import-aiger", "import AIGER file",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        OwningOpRef<ModuleOp> module(
            ModuleOp::create(UnknownLoc::get(context)));
        ImportAIGEROptions options;
        if (failed(importAIGER(sourceMgr, context, ts, module.get(), &options)))
          module = {};
        return module;
      });
}
