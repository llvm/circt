//===- VCD.h - Support for VCD parser/printer -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides support for VCD parser/printer.
//           (VCDLexer)       (VCDParser)
// VCD input -----> [VCDToken] ----> VCDFile
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_VCD_H
#define CIRCT_SUPPORT_VCD_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"

#include <functional>
#include <variant>

#define DEBUG_TYPE "vcd"

using namespace ::mlir;
using namespace ::llvm;
namespace circt {
namespace vcd {

class VCDToken {
public:
  enum Kind {
    // Markers
    eof, // End of file
    error,

    // Basic tokens
    identifier,   // Signal identifiers
    command,      // $... commands
    time_value,   // #... time values
    scalar_value, // 0,1,x,z etc
    vector_value, // b... or B...
    real_value,   // r... or R...

    // Commands
    kw_date,
    kw_version,
    kw_timescale,
    kw_scope,
    kw_var,
    kw_upscope,
    kw_enddefinitions,
    kw_dumpvars,
    kw_comment,
    kw_end,
  };

  VCDToken() = default;
  VCDToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  StringRef getSpelling() const { return spelling; }
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }
  bool isCommand() const { return spelling.starts_with("$"); }

  //===----------------------------------------------------------------------===//

  SMLoc getLoc() const { return SMLoc::getFromPointer(spelling.data()); }
  SMLoc getEndLoc() const {
    return SMLoc::getFromPointer(spelling.data() + spelling.size());
  }
  SMRange getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

private:
  Kind kind;
  StringRef spelling;
};

struct VCDFile {

  struct Node {
    enum Kind {
      metadata,
      scope,
      variable,
    };
    Kind kind;
    Kind getKind() const { return kind; }

    virtual ~Node() = default;
    virtual void dump(mlir::raw_indented_ostream &os) const = 0;
    virtual void printVCD(mlir::raw_indented_ostream &os) const = 0;
  };

  struct Metadata : Node {
    VCDToken command;
    ArrayAttr values;
    // Implement LLVM RTTI.
    static bool classof(const Node *e) {
      return e->getKind() == Kind::metadata;
    }

    Metadata(VCDToken command, ArrayAttr values)
        : command(command), values(values) {
      assert((command.is(VCDToken::kw_timescale) ||
              command.is(VCDToken::kw_date) ||
              command.is(VCDToken::kw_version) ||
              command.is(VCDToken::kw_comment)) &&
             "Invalid metadata command");
    }

    void dump(mlir::raw_indented_ostream &os) const override {
      llvm::errs() << "Metadata: " << command.getSpelling() << "\n";
      llvm::errs() << "Values: " << values << "\n";
    }

    static void printArray(mlir::raw_indented_ostream &os, ArrayAttr array) {
      llvm::interleave(
          array.getAsValueRange<StringAttr>(), os,
          [&](StringRef attr) { os << attr; }, " ");
    }
    void printVCD(mlir::raw_indented_ostream &os) const override {
      switch (command.getKind()) {
      case VCDToken::kw_timescale:
        os << "$timescale ";
        break;
      case VCDToken::kw_date:
        os << "$date ";
        break;
      case VCDToken::kw_version:
        os << "$version ";
        break;
      case VCDToken::kw_comment:
        os << "$comment ";
        break;
      }
      printArray(os, values);
      os << "$end\n";
    }
  };

  struct Scope : Node {
    StringAttr name;

    // Implement LLVM RTTI.
    static bool classof(const Node *e) { return e->getKind() == Kind::scope; }
    // SV 21.7.2.1
    enum ScopeType {
      module,
      begin,
    } kind;
    MapVector<StringAttr, std::unique_ptr<Node>> children;
    Scope(StringAttr name, ScopeType kind) : name(name), kind(kind) {}
    void dump(mlir::raw_indented_ostream &os) const override {
      llvm::errs() << "Scope: " << name << "\n";
      auto scope = os.scope();
      for (auto &[_, child] : children)
        child->dump(os);
    }
    void printVCD(mlir::raw_indented_ostream &os) const override {
      os << "$scope " << name.getValue() << "\n";
      {
        auto scope = os.scope();
        for (auto &[_, child] : children)
          child->printVCD(os);
      }
      os << "$upscope $end\n";
    }
  };

  struct Variable : Node {
    enum Kind {
      wire,
      reg,
      integer,
      real,
      time,
    } kind;

    // Implement LLVM RTTI.
    static bool classof(const Node *e) {
      return e->getKind() == Node::Kind::variable;
    }

    static StringRef getKindName(Kind kind) {
      switch (kind) {
      case wire:
        return "wire";
      case reg:
        return "reg";
      case integer:
        return "integer";
      case real:
        return "real";
      case time:
        return "time";
      }
    }

    int64_t bitWidth;
    StringAttr id;
    StringAttr name;
    ArrayAttr type; // TODO: Parse type properly
    Variable(Kind kind, int64_t bitWidth, StringAttr id, StringAttr name,
             ArrayAttr type)
        : kind(kind), bitWidth(bitWidth), id(id), name(name), type(type) {}
    void dump(mlir::raw_indented_ostream &os) const override {
      llvm::errs() << "Variable: " << name << "\n";
      llvm::errs() << "Kind: " << getKindName(kind) << "\n";
      llvm::errs() << "BitWidth: " << bitWidth << "\n";
      llvm::errs() << "ID: " << id.getValue() << "\n";
      llvm::errs() << "Type: " << type << "\n";
    }
    void printVCD(mlir::raw_indented_ostream &os) const override {
      os << "$var " << getKindName(kind) << " " << bitWidth << " "
         << id.getValue() << " " << name.getValue() << " $end\n";
    }
  };

  std::unique_ptr<Scope> rootScope;
  struct Header {
    // TODO: Parse header peroperly
    SmallVector<Metadata> metadata;

    void printVCD(mlir::raw_indented_ostream &os) const {
      for (auto &data : metadata)
        data.printVCD(os);
    }
  } header;

  struct ValueChange {
    // For lazy loading.
    StringRef remainingBuffer;
    ValueChange(StringRef remainingBuffer) : remainingBuffer(remainingBuffer) {}
    void printVCD(mlir::raw_indented_ostream &os) const {
      os << remainingBuffer;
    }
    void dump(mlir::raw_indented_ostream &os) const {
      os << "Remaining: " << remainingBuffer << "\n";
    }

  } valueChange;

  VCDFile(VCDFile::Header header, std::unique_ptr<Scope> rootScope,
          ValueChange valueChange)
      : rootScope(std::move(rootScope)), header(header),
        valueChange(valueChange) {}

  void dump(mlir::raw_indented_ostream &os) const {
    for (auto &data : header.metadata)
      data.dump(os);
    rootScope->dump(os);
    valueChange.dump(os);
  }
  void printVCD(mlir::raw_indented_ostream &os) const {
    header.printVCD(os);
    rootScope->printVCD(os);
    os << "$enddefinitions $end\n";
    valueChange.printVCD(os);
  }
};

static StringAttr getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

static StringRef getTokenKindName(VCDToken::Kind kind) {
  switch (kind) {
  case VCDToken::eof:
    return "eof";
  case VCDToken::error:
    return "error";
  case VCDToken::kw_date:
    return "kw_date";
  case VCDToken::kw_version:
    return "kw_version";
  case VCDToken::kw_timescale:
    return "kw_timescale";
  case VCDToken::kw_scope:
    return "kw_scope";
  case VCDToken::kw_var:
    return "kw_var";
  case VCDToken::kw_upscope:
    return "kw_upscope";
  case VCDToken::kw_enddefinitions:
    return "kw_enddefinitions";
  case VCDToken::kw_dumpvars:
    return "kw_dumpvars";
  case VCDToken::kw_end:
    return "kw_end";
  }
  llvm::report_fatal_error("unknown token kind");
}

class VCDLexer {
public:
  VCDLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context)
      : sourceMgr(sourceMgr),
        bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
        curBuffer(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
        curPtr(curBuffer.begin()),
        // Prime the first token.
        curToken(lexTokenImpl()) {}

  void lexToken() {
    LLVM_DEBUG(llvm::dbgs() << "lexToken\n");
    curToken = lexTokenImpl();
    LLVM_DEBUG(llvm::dbgs() << "curToken: " << curToken.getSpelling() << "\n");
  }
  const VCDToken &getToken() const { return curToken; }

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.
  Location translateLocation(llvm::SMLoc loc) {
    assert(loc.isValid());
    unsigned mainFileID = sourceMgr.getMainFileID();
    auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
    return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                               lineAndColumn.second);
  }

  StringRef remainingBuffer() const {
    return StringRef(curPtr, curBuffer.end() - curPtr);
  }

private:
  VCDToken lexTokenImpl();
  VCDToken formToken(VCDToken::Kind kind, const char *tokStart) {
    return VCDToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  VCDToken emitError(const char *loc, const Twine &message) {
    mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
    return formToken(VCDToken::error, loc);
  }

  // Lexer implementation methods
  VCDToken lexIdentifier(const char *tokStart);
  VCDToken lexCommand(const char *tokStart);
  VCDToken lexTimeValue(const char *tokStart);
  VCDToken lexInteger(const char *tokStart);
  VCDToken lexString(const char *tokStart);
  VCDToken lexValue(const char *tokStart);
  void skipWhitespace();
  const llvm::SourceMgr &sourceMgr;
  mlir::MLIRContext *context;
  const mlir::StringAttr bufferNameIdentifier;
  StringRef curBuffer;
  const char *curPtr;
  VCDToken curToken;
};

VCDToken VCDLexer::lexTokenImpl() {
  skipWhitespace();

  const char *tokStart = curPtr;
  switch (*curPtr++) {
  default:
    // Handle identifiers
    if (llvm::isAlpha(curPtr[-1]))
      return lexIdentifier(tokStart);
    if (llvm::isDigit(curPtr[-1]))
      return lexInteger(tokStart);

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
          .Case("$date", VCDToken::kw_date)
          .Case("$version", VCDToken::kw_version)
          .Case("$timescale", VCDToken::kw_timescale)
          .Case("$scope", VCDToken::kw_scope)
          .Case("$var", VCDToken::kw_var)
          .Case("$upscope", VCDToken::kw_upscope)
          .Case("$enddefinitions", VCDToken::kw_enddefinitions)
          .Case("$dumpvars", VCDToken::kw_dumpvars)
          .Case("$end", VCDToken::kw_end)
          .Default(VCDToken::command);

  if (kind == VCDToken::command)
    return VCDToken(VCDToken::identifier, spelling);

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

VCDToken VCDLexer::lexValue(const char *tokStart) {
  char firstChar = tokStart[0];

  // Handle scalar values
  if (firstChar == '0' || firstChar == '1' || firstChar == 'x' ||
      firstChar == 'X' || firstChar == 'z' || firstChar == 'Z') {
    return formToken(VCDToken::scalar_value, tokStart);
  }

  // Handle vector values (b... or B...)
  if (firstChar == 'b' || firstChar == 'B') {
    while (*curPtr == '0' || *curPtr == '1' || *curPtr == 'x' ||
           *curPtr == 'X' || *curPtr == 'z' || *curPtr == 'Z')
      ++curPtr;
    return formToken(VCDToken::vector_value, tokStart);
  }

  // Handle real values (r... or R...)
  if (firstChar == 'r' || firstChar == 'R') {
    while (llvm::isDigit(*curPtr) || *curPtr == '.' || *curPtr == '-')
      ++curPtr;
    return formToken(VCDToken::real_value, tokStart);
  }

  return emitError(tokStart, "invalid value format");
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

struct VCDParser {
  bool lazyLoadValueChange = false;

  StringAttr getStringAttr(StringRef str) {
    return StringAttr::get(getContext(), str);
  }

  ParseResult parseAsId(StringAttr &id) {
    auto token = lexer.getToken();
    id = getStringAttr(token.getSpelling());
    consumeToken();
    return success();
  }

  ParseResult parseVariableKind(VCDFile::Variable::Kind &kind) {
    auto variableKind = lexer.getToken().getSpelling();
    auto kindOpt =
        llvm::StringSwitch<std::optional<VCDFile::Variable::Kind>>(variableKind)
            .Case("wire", VCDFile::Variable::wire)
            .Case("reg", VCDFile::Variable::reg)
            .Case("integer", VCDFile::Variable::integer)
            .Case("real", VCDFile::Variable::real)
            .Case("time", VCDFile::Variable::time)
            .Default(std::nullopt);
    if (!kindOpt)
      return emitError() << "unexpected variable kind " << variableKind;
    kind = kindOpt.value();
    consumeToken();
    return success();
  }

  ParseResult parseInt(APInt &result) {
    auto token = lexer.getToken();
    if (!token.is(VCDToken::scalar_value))
      return emitError() << "expected bit width";
    if (token.getSpelling().getAsInteger(10, result))
      return emitError() << "invalid integer literal";
    consumeToken();
    return success();
  }

  ParseResult parseVariable(std::unique_ptr<VCDFile::Variable> &variable) {
    if (parseExpectedKeyword(VCDToken::kw_var))
      return failure();
    VCDFile::Variable::Kind kind;
    APInt bitWidth;
    StringAttr id, name;
    ArrayAttr type;
    if (parseVariableKind(kind) || parseInt(bitWidth) || parseAsId(id) ||
        parseAsId(name) || parseStringUntilEnd(type))
      return failure();
    variable = std::make_unique<VCDFile::Variable>(
        kind, bitWidth.getZExtValue(), id, name, type);

    return success();
  }

  ParseResult parseScopeKind(VCDFile::Scope::ScopeType &kind) {
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

  bool isDone() { return lexer.getToken().is(VCDToken::eof); }
  ParseResult parseEnd() { return parseExpectedKeyword(VCDToken::kw_end); }

  ParseResult parseScope(std::unique_ptr<VCDFile::Scope> &result) {
    VCDFile::Scope::ScopeType kind;
    StringAttr nameAttr;

    if (parseExpectedKeyword(VCDToken::kw_scope) || parseScopeKind(kind) ||
        parseAsId(nameAttr) || parseEnd())
      return failure();

    auto scope = std::make_unique<VCDFile::Scope>(nameAttr, kind);

    while (!isDone() && getToken().getKind() != VCDToken::kw_upscope) {
      auto token = getToken();
      if (token.getKind() == VCDToken::kw_scope) {
        std::unique_ptr<VCDFile::Scope> child;
        if (parseScope(child))
          return failure();
        scope->children.insert(std::make_pair(child->name, std::move(child)));
      } else if (token.getKind() == VCDToken::kw_var) {
        std::unique_ptr<VCDFile::Variable> variable;
        if (parseVariable(variable))
          return failure();
        scope->children.insert(
            std::make_pair(variable->name, std::move(variable)));
      } else {
        return emitError() << "expected scope or variable";
      }
    }

    if (parseExpectedKeyword(VCDToken::kw_upscope) || parseEnd())
      return failure();

    result = std::move(scope);
    return success();
  }

  ParseResult parseStringUntilEnd(ArrayAttr &result) {
    SmallVector<Attribute> args;
    while (!isDone() && !lexer.getToken().is(VCDToken::kw_end)) {
      args.push_back(getStringAttr(lexer.getToken().getSpelling()));
      consumeToken();
    }
    if (parseExpectedKeyword(VCDToken::kw_end))
      return failure();
    result = ArrayAttr::get(getContext(), args);
    return success();
  }

  ParseResult parseAsArrayAttr(VCDToken::Kind kind, ArrayAttr &result) {
    if (parseExpectedKeyword(kind) || parseStringUntilEnd(result))
      return failure();
    return success();
  }

  ParseResult parseAsMetadata(VCDToken::Kind kind, ArrayAttr &result) {
    if (parseExpectedKeyword(kind) || parseStringUntilEnd(result))
      return failure();

    return success();
  }

  ParseResult parseHeader(VCDFile::Header &header) {
    LLVM_DEBUG(llvm::dbgs() << "parseHeader\n");
    SmallVector<VCDFile::Metadata> metadata;
    while (true) {
      auto token = getToken();
      switch (token.getKind()) {
      default:
        return success();
      case VCDToken::kw_date:
      case VCDToken::kw_version:
      case VCDToken::kw_timescale:
        consumeToken();
        ArrayAttr attr;
        if (parseStringUntilEnd(attr))
          return failure();
        metadata.push_back(VCDFile::Metadata(token, attr));
        break;
      }
    }
    return success();
  }

  ParseResult parseVCDFile(std::unique_ptr<VCDFile> &file) {
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
        parseExpectedKeyword(VCDToken::kw_enddefinitions))
      return failure();

    // 3. Parse value change section
    file = std::make_unique<VCDFile>(
        std::move(header), std::move(rootScope),
        VCDFile::ValueChange(lexer.remainingBuffer()));
    return success();
  }

  LogicalResult parseVariableDefinition();
  LogicalResult parseValueChange();
  mlir::MLIRContext *context;
  mlir::MLIRContext *getContext() { return context; }

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }

  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {}) {
    auto diag = mlir::emitError(translateLocation(loc), message);

    // If we hit a parse error in response to a lexer error, then the lexer
    // already reported the error.
    if (getToken().is(VCDToken::error))
      diag.abandon();
    return diag;
  }

  void consumeToken() { lexer.lexToken(); }

  mlir::ParseResult parseExpectedKeyword(VCDToken::Kind expected) {
    const auto &token = lexer.getToken();
    if (token.is(expected)) {
      consumeToken();
      return success();
    }

    emitError() << "expected keyword " << getTokenKindName(expected);
    return failure();
  }

  mlir::ParseResult parseId(StringRef &id) {
    const auto &token = lexer.getToken();
    if (!token.is(VCDToken::identifier))
      return emitError() << "expected id";

    id = token.getSpelling();
    consumeToken();
    return success();
  }

  const VCDToken &getToken() const { return lexer.getToken(); }

  // These are pass-through but defined for understandability.
  LogicalResult convertValueChange();
  LogicalResult convertScope();
  const VCDToken &getNextToken() {
    consumeToken();
    return getToken();
  }

  // SmallVector<hw::HWModuleOp> moduleStack;
  // igraph::InstanceGraph &instanceGraph;
  VCDLexer &lexer;

  VCDParser(mlir::MLIRContext *context, VCDLexer &lexer)
      : context(context), lexer(lexer) {}

  friend raw_ostream &operator<<(raw_ostream &os, const VCDToken &token) {
    if (token.is(VCDToken::eof))
      return os << 0;
    if (token.is(VCDToken::command))
      os << '$';
    return os << token.getSpelling();
  }

  ParseResult parse(VCDFile &file);
};

struct VCDEmitter {};
struct SignalMapping {
  mlir::ModuleOp moduleOp;
  VCDFile::Scope *topScope;
  FlatSymbolRefAttr topModuleName;
  const VCDFile &file;
  igraph::InstanceGraph &instanceGraph;
  igraph::InstancePathCache &instancePathCache;
  llvm::DenseMap<VCDFile::Variable *, circt::igraph::InstancePath> signalMap;
  llvm::DenseMap<StringAttr, DenseMap<StringAttr, Operation *>>
      verilogNameToOperation;

  SignalMapping(mlir::ModuleOp moduleOp, const VCDFile &file,
                igraph::InstanceGraph &instanceGraph,
                igraph::InstancePathCache &instancePathCache,
                ArrayRef<StringAttr> pathToTop, VCDFile::Scope *topScope,
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

  LogicalResult run() {
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
            verilogNameToOperation[module][scope->name]);
        if (!instanceOp)
          continue;
        auto newPath = instancePathCache.appendInstance(path, instanceOp);
        auto newModule = instanceOp.getReferencedModuleNameAttr();
        for (auto &[_, child] : scope->children) {
          worklist.push_back(std::make_tuple(child.get(), newPath, newModule));
        }
      }
    }
  }
};

std::unique_ptr<VCDFile> importVCDFile(llvm::SourceMgr &sourceMgr,
                                       MLIRContext *context) {
  VCDLexer lexer(sourceMgr, context);
  VCDParser parser(context, lexer);
  std::unique_ptr<VCDFile> file;
  if (parser.parseVCDFile(file))
    return nullptr;

  return std::move(file);
}
} // namespace vcd
} // namespace circt

#endif // CIRCT_SUPPORT_VCD_H