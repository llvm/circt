//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// VerilogIndex
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGINDEX_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGINDEX_H_

#include "slang/ast/Compilation.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace lsp {

using VerilogIndexSymbol =
    llvm::PointerUnion<const slang::ast::Symbol *, mlir::Attribute>;
using ReferenceMap = llvm::SmallDenseMap<const slang::ast::Symbol *,
                                         llvm::SmallVector<slang::SourceRange>>;
using SlangBufferPointer = char const *;
using MapT =
    llvm::IntervalMap<SlangBufferPointer, VerilogIndexSymbol,
                      llvm::IntervalMapImpl::NodeSizer<
                          SlangBufferPointer, VerilogIndexSymbol>::LeafSize,
                      llvm::IntervalMapHalfOpenInfo<const SlangBufferPointer>>;

class VerilogIndex {
public:
  VerilogIndex(const slang::BufferID &slangBufferID,
               const slang::SourceManager &sourceManager)
      : mlirContext(mlir::MLIRContext::Threading::DISABLED),
        intervalMap(allocator), bufferId(slangBufferID),
        sourceManager(sourceManager) {}

  /// Initialize the index with the given compilation unit.
  void initialize(slang::ast::Compilation &compilation);

  /// Register a reference to a symbol `symbol` from `from`.
  void insertSymbol(const slang::ast::Symbol *symbol, slang::SourceRange from,
                    bool isDefinition = false);
  void insertSymbolDefinition(const slang::ast::Symbol *symbol);

  const slang::SourceManager &getSlangSourceManager() const {
    return sourceManager;
  }
  const slang::BufferID &getBufferId() const { return bufferId; }

  // Half open interval map containing paired pointers into Slang's buffer
  MapT &getIntervalMap() { return intervalMap; }

  /// A mapping from a symbol to their references.
  const ReferenceMap &getReferences() const { return references; }

private:
  /// Parse source location emitted by ExportVerilog.
  void parseSourceLocation();
  void parseSourceLocation(llvm::StringRef toParse);

  // MLIR context used for generating location attr.
  mlir::MLIRContext mlirContext;

  /// An allocator for the interval map.
  MapT::Allocator allocator;

  /// An interval map containing a corresponding definition mapped to a source
  /// interval.
  MapT intervalMap;

  /// References of symbols.
  ReferenceMap references;

  // The parent document's buffer ID.
  const slang::BufferID &bufferId;
  // The parent document's source manager.
  const slang::SourceManager &sourceManager;
};
}; // namespace lsp
}; // namespace circt

#endif
