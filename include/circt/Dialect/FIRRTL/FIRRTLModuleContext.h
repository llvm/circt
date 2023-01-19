//===- FIRRTLModuleContext.h - Module context base --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides context information that is global to the module.
//
// The pure virtual class `FIRRTLModuleContext` is inherited and implemented in
// `FIRParser.cpp` and `FIRRTLFFIContext.cpp`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLMODULECONTEXT_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLMODULECONTEXT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"
#include <string>

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// FIRRTLModuleContext
//===----------------------------------------------------------------------===//

class FIRRTLModuleContext {
public:
  // Entries in a symbol table are either an mlir::Value for the operation that
  // defines the value or an unbundled ID tracking the index in the
  // UnbundledValues list.
  using UnbundledID = llvm::PointerEmbeddedInt<unsigned, 31>;
  using SymbolValueEntry = llvm::PointerUnion<Value, UnbundledID>;

  using ModuleSymbolTable =
      llvm::StringMap<std::pair<llvm::SMLoc, SymbolValueEntry>,
                      llvm::BumpPtrAllocator>;
  using ModuleSymbolTableEntry =
      llvm::StringMapEntry<std::pair<llvm::SMLoc, SymbolValueEntry>>;

  using UnbundledValueEntry = SmallVector<std::pair<Attribute, Value>>;
  using UnbundledValuesList = std::vector<UnbundledValueEntry>;

  using SubaccessCache = llvm::DenseMap<std::pair<Value, unsigned>, Value>;

  explicit FIRRTLModuleContext(std::string moduleTarget);
  virtual ~FIRRTLModuleContext() = default;

  virtual MLIRContext *getContext() const = 0;

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  virtual InFlightDiagnostic emitError(const Twine &message = {}) = 0;
  virtual InFlightDiagnostic emitError(llvm::SMLoc loc,
                                       const Twine &message = {}) = 0;

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  virtual Location translateLocation(llvm::SMLoc loc) = 0;

  /// This is the module target used by annotations referring to this module.
  std::string moduleTarget;

  // The expression-oriented nature of firrtl syntax produces tons of constant
  // nodes which are obviously redundant.  Instead of literally producing them
  // in the parser, do an implicit CSE to reduce parse time and silliness in the
  // resulting IR.
  llvm::DenseMap<std::pair<Attribute, Type>, Value> constantCache;

  //===--------------------------------------------------------------------===//
  // SubaccessCache

  /// This returns a reference with the assumption that the caller will fill in
  /// the cached value. We keep track of inserted subaccesses so that we can
  /// remove them when we exit a scope.
  Value &getCachedSubaccess(Value value, unsigned index);

  //===--------------------------------------------------------------------===//
  // SymbolTable

  /// Add a symbol entry with the specified name, returning failure if the name
  /// is already defined.
  ///
  /// When 'insertNameIntoGlobalScope' is true, we don't allow the name to be
  /// popped.  This is a workaround for (firrtl scala bug) that should
  /// eventually be fixed.
  ParseResult addSymbolEntry(StringRef name, SymbolValueEntry entry,
                             llvm::SMLoc loc,
                             bool insertNameIntoGlobalScope = false);

  /// Look up the specified name, emitting an error and returning null if the
  /// name is unknown.
  ParseResult addSymbolEntry(StringRef name, Value value, llvm::SMLoc loc,
                             bool insertNameIntoGlobalScope = false);

  /// Resolved a symbol table entry to a value.  Emission of error is optional.
  ParseResult resolveSymbolEntry(Value &result, SymbolValueEntry &entry,
                                 llvm::SMLoc loc, bool fatal = true);

  /// Resolved a symbol table entry if it is an expanded bundle e.g. from an
  /// instance.  Emission of error is optional.
  ParseResult resolveSymbolEntry(Value &result, SymbolValueEntry &entry,
                                 StringRef field, llvm::SMLoc loc);

  /// Look up the specified name, emitting an error and returning failure if the
  /// name is unknown.
  ParseResult lookupSymbolEntry(SymbolValueEntry &result, StringRef name,
                                llvm::SMLoc loc);

  UnbundledValueEntry &getUnbundledEntry(unsigned index);

  /// This contains one entry for each value in FIRRTL that is represented as a
  /// bundle type in the FIRRTL spec but for which we represent as an exploded
  /// set of elements in the FIRRTL dialect.
  UnbundledValuesList unbundledValues;

  /// Provide a symbol table scope that automatically pops all the entries off
  /// the symbol table when the scope is exited.
  struct ContextScope {
    friend class FIRRTLModuleContext;
    ContextScope(FIRRTLModuleContext &moduleContext, Block *block);
    ~ContextScope();

  private:
    void operator=(const ContextScope &) = delete;
    ContextScope(const ContextScope &) = delete;

    FIRRTLModuleContext &moduleContext;
    Block *block;
    ContextScope *previousScope;
    std::vector<ModuleSymbolTableEntry *> scopedDecls;
    std::vector<std::pair<Value, unsigned>> scopedSubaccesses;
  };

private:
  /// This symbol table holds the names of ports, wires, and other local decls.
  /// This is scoped because conditional statements introduce subscopes.
  ModuleSymbolTable symbolTable;

  /// This is a cache of subindex and subfield operations so we don't constantly
  /// recreate large chains of them.  This maps a bundle value + index to the
  /// subaccess result.
  SubaccessCache subaccessCache;

  /// This maps a block to related ContextScope.
  DenseMap<Block *, ContextScope *> scopeMap;

  /// If non-null, all new entries added to the symbol table are added to this
  /// list.  This allows us to "pop" the entries by resetting them to null when
  /// scope is exited.
  ContextScope *currentScope = nullptr;
};
} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRMODULECONTEXTBASE_H
