//===- VCD.cpp - VCD parser/printer implementation
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/VCD.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"

#include <functional>
#include <variant>

#define DEBUG_TYPE "vcd"

using namespace circt;
using namespace vcd;
using namespace llvm;

//===----------------------------------------------------------------------===//
// VCDFile Data Structures
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VCDFile::Metadata
//===----------------------------------------------------------------------===//

VCDFile::Metadata::Metadata(StringAttr command, ArrayAttr values)
    : Node(Node::Kind::metadata), command(command), values(values) {}

bool VCDFile::Metadata::classof(const Node *e) {
  return e->getKind() == Kind::metadata;
}

void VCDFile::Metadata::dump(mlir::raw_indented_ostream &os) const {
  os << "Metadata: " << command << "\n";
  os << "Values: " << values << "\n";
}

static void printArray(mlir::raw_indented_ostream &os, ArrayAttr array) {
  llvm::interleave(
      array.getAsValueRange<StringAttr>(), os,
      [&](StringRef attr) { os << attr; }, " ");
}

void VCDFile::Metadata::printVCD(mlir::raw_indented_ostream &os) const {
  os << command << " ";
  printArray(os, values);
  os << "$end\n";
}

//===----------------------------------------------------------------------===//
// VCDFile::Scope
//===----------------------------------------------------------------------===//

static StringRef getScopeKindName(VCDFile::Scope::ScopeType kind) {
  switch (kind) {
  case VCDFile::Scope::module:
    return "module";
  case VCDFile::Scope::begin:
    return "begin";
  }
  llvm::report_fatal_error("unknown scope kind");
}

bool VCDFile::Scope::classof(const Node *e) {
  return e->getKind() == Kind::scope;
}

void VCDFile::Scope::printVCD(mlir::raw_indented_ostream &os) const {
  os << "$scope " << getScopeKindName(kind) << " " << name.getValue() << "\n";
  {
    auto scope = os.scope();
    for (auto &[_, child] : children)
      child->printVCD(os);
  }
  os << "$upscope $end\n";
}

void VCDFile::Scope::dump(mlir::raw_indented_ostream &os) const {
  os << "Scope: " << name << "\n";
  for (auto &[_, child] : children) {
    child->dump(os);
  }
}

//===----------------------------------------------------------------------===//
// VCDFile::Variable
//===----------------------------------------------------------------------===//

static StringRef getKindName(VCDFile::Variable::VariableType kind) {
  switch (kind) {
  case VCDFile::Variable::wire:
    return "wire";
  case VCDFile::Variable::reg:
    return "reg";
  case VCDFile::Variable::integer:
    return "integer";
  case VCDFile::Variable::real:
    return "real";
  case VCDFile::Variable::time:
    return "time";
  }
}

bool VCDFile::Variable::classof(const Node *e) {
  return e->getKind() == Kind::variable;
}

void VCDFile::Variable::printVCD(mlir::raw_indented_ostream &os) const {
  os << "$var " << getKindName(kind) << " " << bitWidth << " " << id.getValue()
     << " " << name.getValue() << " $end\n";
}

void VCDFile::Variable::dump(mlir::raw_indented_ostream &os) const {
  os << "Variable: " << name << "\n";
  os << "Kind: " << getKindName(kind) << "\n";
  os << "BitWidth: " << bitWidth << "\n";
  os << "ID: " << id.getValue() << "\n";
  llvm::errs() << "Type: " << type << "\n";
}

//===----------------------------------------------------------------------===//
// VCDFile::Header
//===----------------------------------------------------------------------===//

void VCDFile::Header::printVCD(mlir::raw_indented_ostream &os) const {
  for (auto &data : metadata)
    data->printVCD(os);
}

//===----------------------------------------------------------------------===//
// VCDFile::ValueChange
//===----------------------------------------------------------------------===//

void VCDFile::ValueChange::printVCD(mlir::raw_indented_ostream &os) const {
  os << remainingBuffer << "\n";
}

// ===----------------------------------------------------------------------===//
// VCDFile::VCDFile
//===----------------------------------------------------------------------===//

void VCDFile::VCDFile::printVCD(mlir::raw_indented_ostream &os) const {
  // Print the header
  header.printVCD(os);
  // Print the scope and variable definitions
  rootScope->printVCD(os);
  // Print the value change section
  valueChange.printVCD(os);
}

VCDFile::VCDFile(VCDFile::Header header, std::unique_ptr<Scope> rootScope,
                 ValueChange valueChange)
    : header(std::move(header)), valueChange(valueChange),
      rootScope(std::move(rootScope)) {}

//===----------------------------------------------------------------------===//
// VCDLexer
//===----------------------------------------------------------------------===//

namespace {
class VCDToken {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_COMMAND(SPELLING) command_##SPELLING,
#include "VCDTokenKinds.def"
  };

  VCDToken() = default;
  VCDToken(Kind kind, StringRef spelling);

  StringRef getSpelling() const;
  Kind getKind() const;
  bool is(Kind K) const;
  bool isCommand() const;

  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  Kind kind;
  StringRef spelling;
};
class VCDLexer {
public:
  VCDLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  void lexToken();
  const VCDToken &getToken() const;
  Location translateLocation(llvm::SMLoc loc);
  StringRef remainingBuffer() const;

private:
  VCDToken lexTokenImpl();
  VCDToken formToken(VCDToken::Kind kind, const char *tokStart);
  VCDToken emitError(const char *loc, const Twine &message);

  VCDToken lexIdentifier(const char *tokStart);
  VCDToken lexCommand(const char *tokStart);
  VCDToken lexTimeValue(const char *tokStart);
  VCDToken lexInteger(const char *tokStart);
  VCDToken lexString(const char *tokStart);
  void skipWhitespace();

  const llvm::SourceMgr &sourceMgr;
  mlir::MLIRContext *context;
  const mlir::StringAttr bufferNameIdentifier;
  StringRef curBuffer;
  const char *curPtr;
  VCDToken curToken;
};
} // namespace

// VCDToken implementation
VCDToken::VCDToken(Kind kind, StringRef spelling)
    : kind(kind), spelling(spelling) {}

StringRef VCDToken::getSpelling() const { return spelling; }
VCDToken::Kind VCDToken::getKind() const { return kind; }
bool VCDToken::is(Kind K) const { return kind == K; }
bool VCDToken::isCommand() const { return spelling.starts_with("$"); }

llvm::SMLoc VCDToken::getLoc() const {
  return llvm::SMLoc::getFromPointer(spelling.data());
}

llvm::SMLoc VCDToken::getEndLoc() const {
  return llvm::SMLoc::getFromPointer(spelling.data() + spelling.size());
}

llvm::SMRange VCDToken::getLocRange() const {
  return llvm::SMRange(getLoc(), getEndLoc());
}

static StringAttr getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

VCDLexer::VCDLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      curBuffer(
          sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()), curToken(lexTokenImpl()) {}

static StringRef getTokenKindName(VCDToken::Kind kind) {
  switch (kind) {
#define TOK_MARKER(NAME)                                                       \
  case VCDToken::NAME:                                                         \
    return #NAME;
#define TOK_IDENTIFIER(NAME)                                                   \
  case VCDToken::NAME:                                                         \
    return #NAME;
#define TOK_LITERAL(NAME)                                                      \
  case VCDToken::NAME:                                                         \
    return #NAME;
#define TOK_COMMAND(SPELLING)                                                  \
  case VCDToken::command_##SPELLING:                                           \
    return #SPELLING;
#include "VCDTokenKinds.def"
  }
  llvm::report_fatal_error("unknown token kind");
}

void VCDLexer::lexToken() {
  LLVM_DEBUG(llvm::dbgs() << "lexToken\n");
  curToken = lexTokenImpl();
  LLVM_DEBUG(llvm::dbgs() << "curToken: " << curToken.getSpelling() << "\n");
}

const VCDToken &VCDLexer::getToken() const { return curToken; }

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location VCDLexer::translateLocation(llvm::SMLoc loc) {
  assert(loc.isValid());
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

StringRef VCDLexer::remainingBuffer() const {
  return StringRef(curPtr, curBuffer.end() - curPtr);
}

VCDToken VCDLexer::formToken(VCDToken::Kind kind, const char *tokStart) {
  return VCDToken(kind, StringRef(tokStart, curPtr - tokStart));
}

VCDToken VCDLexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(translateLocation(llvm::SMLoc::getFromPointer(loc)), message);
  return formToken(VCDToken::error, loc);
}

VCDToken VCDLexer::lexTokenImpl() {
  skipWhitespace();

  const char *tokStart = curPtr;
  switch (*curPtr++) {
  default:
    if (llvm::isDigit(curPtr[-1]))
      return lexInteger(tokStart);

    // Treat it as identifiers
    return lexIdentifier(tokStart);

  case 0:
    // Handle EOF
    if (curPtr - 1 == curBuffer.end())
      return formToken(VCDToken::eof, tokStart);
    [[fallthrough]];

  case ' ':
  case '\t':
  case '\n':
  case '\r':
    skipWhitespace();
    return lexTokenImpl();

  case '$':
    return lexCommand(tokStart);

  case '#':
    return lexTimeValue(tokStart);
  }
}

VCDToken VCDLexer::lexCommand(const char *tokStart) {
  LLVM_DEBUG(llvm::dbgs() << "lexCommand\n");
  // Read until whitespace
  while (!llvm::isSpace(*curPtr) && *curPtr != 0)
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // Match known commands
  VCDToken::Kind kind =
      llvm::StringSwitch<VCDToken::Kind>(spelling)
          .Case("$date", VCDToken::command_date)
          .Case("$version", VCDToken::command_version)
          .Case("$timescale", VCDToken::command_timescale)
          .Case("$scope", VCDToken::command_scope)
          .Case("$var", VCDToken::command_var)
          .Case("$upscope", VCDToken::command_upscope)
          .Case("$enddefinitions", VCDToken::command_enddefinitions)
          .Case("$dumpvars", VCDToken::command_dumpvars)
          .Case("$end", VCDToken::command_end)
          .Default(VCDToken::error);

  return VCDToken(kind, spelling);
}

VCDToken VCDLexer::lexInteger(const char *tokStart) {
  while (llvm::isDigit(*curPtr))
    ++curPtr;
  return formToken(VCDToken::scalar_value, tokStart);
}

VCDToken VCDLexer::lexTimeValue(const char *tokStart) {
  while (llvm::isDigit(*curPtr))
    ++curPtr;
  return formToken(VCDToken::time_value, tokStart);
}

VCDToken VCDLexer::lexIdentifier(const char *tokStart) {
  while (llvm::isAlnum(*curPtr) || llvm::isPunct(*curPtr))
    ++curPtr;
  return formToken(VCDToken::identifier, tokStart);
}

void VCDLexer::skipWhitespace() {
  while (llvm::isSpace(*curPtr))
    ++curPtr;
}

//===----------------------------------------------------------------------===//
// VCDParser
//===----------------------------------------------------------------------===//

namespace {
struct VCDParser {
  bool lazyLoadValueChange = false;

  StringAttr getStringAttr(StringRef str);
  ParseResult parseAsId(StringAttr &id);
  ParseResult parseVariableKind(VCDFile::Variable::VariableType &kind);
  ParseResult parseInt(APInt &result);
  ParseResult parseVariable(std::unique_ptr<VCDFile::Variable> &variable);
  ParseResult parseScopeKind(VCDFile::Scope::ScopeType &kind);
  bool isDone();

  ParseResult parseEnd();
  ParseResult parseScope(std::unique_ptr<VCDFile::Scope> &result);
  ParseResult parseStringUntilEnd(ArrayAttr &result);
  ParseResult parseAsArrayAttr(VCDToken::Kind kind, ArrayAttr &result);
  ParseResult parseAsMetadata(VCDToken::Kind kind, ArrayAttr &result);
  ParseResult parseHeader(VCDFile::Header &header);
  ParseResult parseVCDFile(std::unique_ptr<VCDFile> &file);
  LogicalResult parseVariableDefinition();
  LogicalResult parseValueChange();
  mlir::MLIRContext *context;
  mlir::MLIRContext *getContext() { return context; }
  Location translateLocation(llvm::SMLoc loc);
  InFlightDiagnostic emitError(const Twine &message = {});
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});
  void consumeToken();
  mlir::ParseResult parseExpectedKeyword(VCDToken::Kind expected);
  mlir::ParseResult parseId(StringRef &id);
  const VCDToken &getToken() const;
  LogicalResult convertValueChange();
  LogicalResult convertScope();
  const VCDToken &getNextToken();

  VCDLexer &lexer;

  VCDParser(mlir::MLIRContext *context, VCDLexer &lexer)
      : context(context), lexer(lexer) {}

  friend raw_ostream &operator<<(raw_ostream &os, const VCDToken &token);

  ParseResult parse(VCDFile &file);
};

StringAttr VCDParser::getStringAttr(StringRef str) {
  return StringAttr::get(getContext(), str);
}

ParseResult VCDParser::parseAsId(StringAttr &id) {
  auto token = lexer.getToken();
  id = getStringAttr(token.getSpelling());
  consumeToken();
  return success();
}

ParseResult
VCDParser::parseVariableKind(VCDFile::Variable::VariableType &kind) {
  auto variableKind = lexer.getToken().getSpelling();
  auto kindOpt =
      llvm::StringSwitch<std::optional<VCDFile::Variable::VariableType>>(
          variableKind)
          .Case("wire", VCDFile::Variable::VariableType::wire)
          .Case("reg", VCDFile::Variable::VariableType::reg)
          .Case("integer", VCDFile::Variable::VariableType::integer)
          .Case("real", VCDFile::Variable::VariableType::real)
          .Case("time", VCDFile::Variable::VariableType::time)
          .Default(std::nullopt);
  if (!kindOpt)
    return emitError() << "unexpected variable kind " << variableKind;
  kind = kindOpt.value();
  consumeToken();
  return success();
}

ParseResult VCDParser::parseInt(APInt &result) {
  auto token = lexer.getToken();
  if (token.getSpelling().getAsInteger(10, result))
    return emitError() << "invalid integer literal";
  consumeToken();
  return success();
}

ParseResult
VCDParser::parseVariable(std::unique_ptr<VCDFile::Variable> &variable) {
  if (parseExpectedKeyword(VCDToken::command_var))
    return failure();
  VCDFile::Variable::VariableType kind;
  APInt bitWidth;
  StringAttr id, name;
  ArrayAttr type;
  if (parseVariableKind(kind) || parseInt(bitWidth) || parseAsId(id) ||
      parseAsId(name) || parseStringUntilEnd(type))
    return failure();
  variable = std::make_unique<VCDFile::Variable>(kind, bitWidth.getZExtValue(),
                                                 id, name, type);

  return success();
}

ParseResult VCDParser::parseScopeKind(VCDFile::Scope::ScopeType &kind) {
  auto scopeKind = lexer.getToken().getSpelling();
  auto kindOpt =
      llvm::StringSwitch<std::optional<VCDFile::Scope::ScopeType>>(scopeKind)
          .Case("module", VCDFile::Scope::module)
          .Case("begin", VCDFile::Scope::begin)
          .Default(std::nullopt);

  if (!kind)
    return emitError() << "expected scope kind";
  kind = kindOpt.value();
  consumeToken();
  return success();
}

bool VCDParser::isDone() { return lexer.getToken().is(VCDToken::eof); }

ParseResult VCDParser::parseEnd() {
  return parseExpectedKeyword(VCDToken::command_end);
}

ParseResult VCDParser::parseScope(std::unique_ptr<VCDFile::Scope> &result) {
  VCDFile::Scope::ScopeType kind;
  StringAttr nameAttr;

  if (parseExpectedKeyword(VCDToken::command_scope) || parseScopeKind(kind) ||
      parseAsId(nameAttr) || parseEnd())
    return failure();

  auto scope = std::make_unique<VCDFile::Scope>(nameAttr, kind);

  while (!isDone() && getToken().getKind() != VCDToken::command_upscope) {
    auto token = getToken();
    if (token.getKind() == VCDToken::command_scope) {
      std::unique_ptr<VCDFile::Scope> child;
      if (parseScope(child))
        return failure();
      scope->getChildren().insert(
          std::make_pair(child->getName(), std::move(child)));
    } else if (token.getKind() == VCDToken::command_var) {
      std::unique_ptr<VCDFile::Variable> variable;
      if (parseVariable(variable))
        return failure();
      scope->getChildren().insert(
          std::make_pair(variable->getName(), std::move(variable)));
    } else {
      return emitError() << "expected scope or variable";
    }
  }

  if (parseExpectedKeyword(VCDToken::command_upscope) || parseEnd())
    return failure();

  result = std::move(scope);
  return success();
}

ParseResult VCDParser::parseStringUntilEnd(ArrayAttr &result) {
  SmallVector<Attribute> args;
  while (!isDone() && !lexer.getToken().is(VCDToken::command_end)) {
    args.push_back(getStringAttr(lexer.getToken().getSpelling()));
    consumeToken();
  }
  if (parseExpectedKeyword(VCDToken::command_end))
    return failure();
  result = ArrayAttr::get(getContext(), args);
  return success();
}

ParseResult VCDParser::parseAsArrayAttr(VCDToken::Kind kind,
                                        ArrayAttr &result) {
  if (parseExpectedKeyword(kind) || parseStringUntilEnd(result))
    return failure();
  return success();
}

ParseResult VCDParser::parseAsMetadata(VCDToken::Kind kind, ArrayAttr &result) {
  if (parseExpectedKeyword(kind) || parseStringUntilEnd(result))
    return failure();

  return success();
}

ParseResult VCDParser::parseHeader(VCDFile::Header &header) {
  LLVM_DEBUG(llvm::dbgs() << "parseHeader\n");
  SmallVector<VCDFile::Metadata> metadata;
  while (true) {
    auto token = getToken();
    switch (token.getKind()) {
    default:
      return success();
    case VCDToken::command_date:
    case VCDToken::command_version:
    case VCDToken::command_timescale:
      consumeToken();
      ArrayAttr attr;
      if (parseStringUntilEnd(attr))
        return failure();
      metadata.push_back(
          VCDFile::Metadata(getStringAttr(token.getSpelling()), attr));
      break;
    }
  }
  return success();
}

ParseResult VCDParser::parseVCDFile(std::unique_ptr<VCDFile> &file) {
  // 1. Parse header section
  VCDFile::Header header;
  if (parseHeader(header))
    return failure();

  // There is no scope command, weird but we are done.
  if (isDone())
    return success();

  // 2. Parse variable definition section
  std::unique_ptr<VCDFile::Scope> rootScope;
  if (parseScope(rootScope) ||
      parseExpectedKeyword(VCDToken::command_enddefinitions))
    return failure();

  // 3. Parse value change section
  file =
      std::make_unique<VCDFile>(std::move(header), std::move(rootScope),
                                VCDFile::ValueChange(lexer.remainingBuffer()));
  return success();
}

Location VCDParser::translateLocation(llvm::SMLoc loc) {
  return lexer.translateLocation(loc);
}

InFlightDiagnostic VCDParser::emitError(const Twine &message) {
  return emitError(getToken().getLoc(), message);
}

InFlightDiagnostic VCDParser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(VCDToken::error))
    diag.abandon();
  return diag;
}

void VCDParser::consumeToken() { lexer.lexToken(); }

mlir::ParseResult VCDParser::parseExpectedKeyword(VCDToken::Kind expected) {
  const auto &token = lexer.getToken();
  if (token.is(expected)) {
    consumeToken();
    return success();
  }

  emitError() << "expected keyword " << getTokenKindName(expected);
  return failure();
}

mlir::ParseResult VCDParser::parseId(StringRef &id) {
  const auto &token = lexer.getToken();
  if (!token.is(VCDToken::identifier))
    return emitError() << "expected id";

  id = token.getSpelling();
  consumeToken();
  return success();
}

const VCDToken &VCDParser::getToken() const { return lexer.getToken(); }

const VCDToken &VCDParser::getNextToken() {
  consumeToken();
  return getToken();
}

raw_ostream &operator<<(raw_ostream &os, const VCDToken &token) {
  if (token.is(VCDToken::eof))
    return os << 0;
  return os << token.getSpelling();
}

} // namespace

//===----------------------------------------------------------------------===//
// SignalMapping
//===----------------------------------------------------------------------===//

SignalMapping::SignalMapping(mlir::ModuleOp moduleOp, const VCDFile &file,
                             igraph::InstanceGraph &instanceGraph,
                             igraph::InstancePathCache &instancePathCache,
                             ArrayRef<StringAttr> pathToTop,
                             VCDFile::Scope *topScope,
                             FlatSymbolRefAttr topModuleName)
    : moduleOp(moduleOp), topScope(topScope), topModuleName(topModuleName),
      file(file), instanceGraph(instanceGraph),
      instancePathCache(instancePathCache) {
  for (auto op : moduleOp.getOps<hw::HWModuleOp>()) {
    verilogNameToOperation[op.getModuleNameAttr()].insert(
        std::make_pair(op.getModuleNameAttr(), op));
  }

  mlir::parallelForEach(
      moduleOp.getContext(), moduleOp.getOps<hw::HWModuleOp>(),
      [&](hw::HWModuleOp op) {
        auto &moduleMap = verilogNameToOperation[op.getModuleNameAttr()];
        op.walk([&](Operation *op) {
          TypeSwitch<Operation *>(op)
              .Case<sv::RegOp>(
                  [&](sv::RegOp reg) { moduleMap[reg.getNameAttr()] = reg; })
              .Case<hw::InstanceOp>([&](hw::InstanceOp instance) {
                moduleMap[instance.getInstanceNameAttr()] = instance;
              });
        });
      });
}

LogicalResult SignalMapping::run() {
  auto topModule = instanceGraph.lookup(topModuleName.getAttr());
  if (!topModule ||
      isa_and_nonnull<circt::hw::HWModuleOp>(topModule->getModule()))
    return failure();

  auto module = dyn_cast<hw::HWModuleOp>(topModule->getModule());

  SmallVector<
      std::tuple<VCDFile::Node *, circt::igraph::InstancePath, StringAttr>>
      worklist;
  worklist.push_back(std::make_tuple(topScope, circt::igraph::InstancePath(),
                                     module.getModuleNameAttr()));
  while (!worklist.empty()) {
    auto [node, path, module] = worklist.pop_back_val();
    if (auto *variable = dyn_cast<VCDFile::Variable>(node)) {
      signalMap[variable] = path;
    } else if (auto *scope = dyn_cast<VCDFile::Scope>(node)) {
      auto instanceOp = dyn_cast_or_null<hw::InstanceOp>(
          verilogNameToOperation[module][scope->getName()]);
      if (!instanceOp)
        continue;
      auto newPath = instancePathCache.appendInstance(path, instanceOp);
      auto newModule = instanceOp.getReferencedModuleNameAttr();
      for (auto &[_, child] : scope->getChildren()) {
        worklist.push_back(std::make_tuple(child.get(), newPath, newModule));
      }
    }
  }
}

// ===----------------------------------------------------------------------===//
// VCD Import
// ===----------------------------------------------------------------------===//

std::unique_ptr<VCDFile> circt::vcd::importVCDFile(llvm::SourceMgr &sourceMgr,
                                                   MLIRContext *context) {
  VCDLexer lexer(sourceMgr, context);
  VCDParser parser(context, lexer);
  std::unique_ptr<VCDFile> file;
  if (parser.parseVCDFile(file))
    return nullptr;
  return file;
}
