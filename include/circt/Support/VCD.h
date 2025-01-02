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

struct VCDFile {
  struct Node {
    // LLVM RTTI kind.
    enum Kind {
      metadata,
      scope,
      variable,
    };
    Kind kind;
    Kind getKind() const;
    Node(Kind kind) : kind(kind) {}

    virtual ~Node() = default;
    // Dump the node to the stream.
    virtual void dump(mlir::raw_indented_ostream &os) const = 0;
    // Print it in VCD format.
    virtual void printVCD(mlir::raw_indented_ostream &os) const = 0;
  };

  struct Metadata : Node {
    // Represent generic metadata command.
    // Used for $date, $version, $timescale and $comment.
    enum MetadataKind {
      date,
      version,
      timescale,
      comment,
    };

    // Implement LLVM RTTI.
    static bool classof(const Node *e);
    Metadata(StringAttr command, ArrayAttr values);
    void printVCD(mlir::raw_indented_ostream &os) const override;
    void dump(mlir::raw_indented_ostream &os) const override;

  private:
    StringAttr command;
    ArrayAttr values;
  };

  struct Scope : Node {
    // Represent a scope container which corresponds to
    // `$scope name $end {...} $upscope $end`

    // SV 21.7.2.1
    enum ScopeType {
      module,
      begin,
    };

    // Implement LLVM RTTI.
    static bool classof(const Node *e);
    Scope(StringAttr name, ScopeType kind)
        : Node(Node::Kind::scope), name(name), kind(kind) {}
    void dump(mlir::raw_indented_ostream &os) const override;
    void printVCD(mlir::raw_indented_ostream &os) const override;
    void setName(StringAttr name) { this->name = name; }
    StringAttr getName() const { return name; }
    auto &getChildren() { return children; }

  private:
    StringAttr name;

    ScopeType kind;
    llvm::MapVector<StringAttr, std::unique_ptr<Node>> children;
  };

  struct Variable : Node {
    // Represent a variable which corresponds to
    // `$var type width id name[$type] $end`
    enum VariableType {
      wire,
      reg,
      integer,
      real,
      time,
    };

    // Implement LLVM RTTI.
    static bool classof(const Node *e);

    Variable(VariableType kind, int64_t bitWidth, StringAttr id,
             StringAttr name, ArrayAttr type)
        : Node(Node::Kind::variable), kind(kind), bitWidth(bitWidth), id(id),
          name(name), type(type) {}
    void dump(mlir::raw_indented_ostream &os) const override;
    void printVCD(mlir::raw_indented_ostream &os) const override;
    StringAttr getName() const { return name; }
    StringAttr getId() const { return id; }
    int64_t getBitWidth() const { return bitWidth; }
    VariableType getKind() const { return kind; }

  private:
    VariableType kind;
    int64_t bitWidth;
    StringAttr id;
    StringAttr name;
    ArrayAttr type; // TODO: Parse type properly
  };

  struct Header {
    // TODO: Parse header peroperly
    SmallVector<std::unique_ptr<Node>> metadata;

    void printVCD(mlir::raw_indented_ostream &os) const;
  };

  struct ValueChange {
    // For lazy loading.
    StringRef remainingBuffer;
    ValueChange(StringRef remainingBuffer) : remainingBuffer(remainingBuffer) {}
    void printVCD(mlir::raw_indented_ostream &os) const;
    void dump(mlir::raw_indented_ostream &os) const;
  };

  VCDFile(VCDFile::Header header, std::unique_ptr<Scope> rootScope,
          ValueChange valueChange);

  void dump(mlir::raw_indented_ostream &os) const;
  void printVCD(mlir::raw_indented_ostream &os) const;

private:
  Header header;
  std::unique_ptr<Scope> rootScope;
  ValueChange valueChange;
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