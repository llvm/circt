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

  struct Metadata;
  struct Scope;
  struct Variable;
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