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
#include "circt/Dialect/SV/SVOps.h"
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

struct VCDFile {
  struct Node {
    enum Kind {
      metadata,
      scope,
      variable,
    };
    Kind kind;
    Kind getKind() const;

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
    llvm::MapVector<StringAttr, std::unique_ptr<Node>> children;
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

  struct Header {
    // TODO: Parse header peroperly
    SmallVector<std::unique_ptr<Node>> metadata;

    void printVCD(mlir::raw_indented_ostream &os) const {
      for (auto &data : metadata)
        data->printVCD(os);
    }
  } header;
  struct ValueChange {
    // For lazy loading.
    StringRef remainingBuffer;
    ValueChange(StringRef remainingBuffer) : remainingBuffer(remainingBuffer) {}
    void printVCD(mlir::raw_indented_ostream &os) const;
    void dump(mlir::raw_indented_ostream &os) const;
  } valueChange;

  VCDFile(VCDFile::Header header, std::unique_ptr<Scope> rootScope,
          ValueChange valueChange);

  void dump(mlir::raw_indented_ostream &os) const;
  void printVCD(mlir::raw_indented_ostream &os) const;

  std::unique_ptr<Scope> rootScope;
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
  VCDToken lexValue(const char *tokStart);
  void skipWhitespace();

  const llvm::SourceMgr &sourceMgr;
  mlir::MLIRContext *context;
  const mlir::StringAttr bufferNameIdentifier;
  StringRef curBuffer;
  const char *curPtr;
  VCDToken curToken;
};

struct VCDParser {
  VCDParser(mlir::MLIRContext *context, VCDLexer &lexer);
  ParseResult parseVCDFile(std::unique_ptr<VCDFile> &file);

private:
  // ... (keep the private methods declarations)
  mlir::MLIRContext *context;
  VCDLexer &lexer;
};

struct SignalMapping {
  SignalMapping(mlir::ModuleOp moduleOp, const VCDFile &file,
                igraph::InstanceGraph &instanceGraph,
                igraph::InstancePathCache &instancePathCache,
                ArrayRef<StringAttr> pathToTop, VCDFile::Scope *topScope,
                FlatSymbolRefAttr topModuleName);

  LogicalResult run();

private:
  static mlir::StringAttr getVariableName(Operation *op);

  mlir::ModuleOp moduleOp;
  VCDFile::Scope *topScope;
  FlatSymbolRefAttr topModuleName;
  const VCDFile &file;
  igraph::InstanceGraph &instanceGraph;
  igraph::InstancePathCache &instancePathCache;
  llvm::DenseMap<VCDFile::Variable *, circt::igraph::InstancePath> signalMap;
  llvm::DenseMap<StringAttr, DenseMap<StringAttr, Operation *>>
      verilogNameToOperation;
};

std::unique_ptr<VCDFile> importVCDFile(llvm::SourceMgr &sourceMgr,
                                       MLIRContext *context);

} // namespace vcd
} // namespace circt

#endif // CIRCT_SUPPORT_VCD_H