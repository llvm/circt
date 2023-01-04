//===- FIRRTLFFIContext.h - .fir to FIRRTL dialect parser -------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defined the context for CIRCT FIRRTL FFI.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLFFICONTEXT_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLFFICONTEXT_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <memory>
#include <optional>
#include <variant>

struct FirrtlType;
struct FirrtlParameter;
struct FirrtlStatement;
struct FirrtlStatementAttach;

namespace circt {
namespace chirrtl {

class FFIContext;

namespace details {

// This class hides all the member functions of `std::optional`, so that we
// have to access the underlying data by macro `RA_EXPECT`.
template <class T>
struct RequireAssigned {
  using Type = T;

  RequireAssigned() = default;
  inline RequireAssigned(T right) : underlying{std::move(right)} {}

  std::optional<T> underlying;
};

enum class ModuleKind {
  Module,
  ExtModule,
};

using UnbundledID = llvm::PointerEmbeddedInt<unsigned, 31>;
using SymbolValueEntry = llvm::PointerUnion<Value, UnbundledID>;
using ModuleSymbolTable =
    llvm::StringMap<SymbolValueEntry, llvm::BumpPtrAllocator>;
using ModuleSymbolTableEntry = llvm::StringMapEntry<SymbolValueEntry>;
using UnbundledValueEntry = SmallVector<std::pair<Attribute, Value>>;
using UnbundledValuesList = std::vector<UnbundledValueEntry>;
using SubaccessCache = llvm::DenseMap<std::pair<Value, unsigned>, Value>;

// Extract from `class FIRModuleContext` in `FIRParser.cpp`.
class ModuleContext {
public:
  ModuleContext(FFIContext &ctx, ModuleKind kind);

  Value &getCachedSubaccess(Value value, unsigned int index);

  bool addSymbolEntry(StringRef name, SymbolValueEntry entry,
                      bool insertNameIntoGlobalScope = false);
  bool addSymbolEntry(StringRef name, Value value,
                      bool insertNameIntoGlobalScope = false);
  bool lookupSymbolEntry(SymbolValueEntry &result, StringRef name);
  bool resolveSymbolEntry(Value &result, SymbolValueEntry &entry, bool fatal);
  bool resolveSymbolEntry(Value &result, SymbolValueEntry &entry,
                          StringRef fieldName);

  inline bool isExtModule() const { return kind == ModuleKind::ExtModule; }

  class ContextScope {
    friend struct ModuleContext;

    ContextScope(ModuleContext &moduleContext, Block *block);
    ~ContextScope();

  private:
    void operator=(const ContextScope &) = delete;
    ContextScope(const ContextScope &) = delete;

    ModuleContext &moduleContext;
    Block *block;
    ContextScope *previousScope;
    std::vector<ModuleSymbolTableEntry *> scopedDecls;
    std::vector<std::pair<Value, unsigned int>> scopedSubaccesses;
  };

private:
  FFIContext &ffiCtx;
  ModuleKind kind;
  ModuleSymbolTable symbolTable;
  SubaccessCache subaccessCache;
  UnbundledValuesList unbundledValues;
  DenseMap<Block *, ContextScope *> scopeMap;
  ContextScope *currentScope = nullptr;
};
} // namespace details

class FFIContext {
public:
  using Direction = firrtl::Direction;

  FFIContext();

  void setErrorHandler(std::function<void(std::string_view message)> handler);
  void emitError(std::string_view message, bool recoverable = false) const;

  void visitCircuit(StringRef name);
  void visitModule(StringRef name);
  void visitExtModule(StringRef name, StringRef defName);
  void visitParameter(StringRef name, const FirrtlParameter &param);
  void visitPort(StringRef name, Direction direction, const FirrtlType &type);
  void visitStatement(const FirrtlStatement &stmt);

  void exportFIRRTL(llvm::raw_ostream &os) const;

private:
  using BodyOpBuilder = mlir::ImplicitLocOpBuilder;

  friend details::ModuleContext;

  std::function<void(std::string_view message)> errorHandler;

  std::unique_ptr<MLIRContext> mlirCtx;
  std::unique_ptr<mlir::ModuleOp> module;
  std::unique_ptr<mlir::OpBuilder> opBuilder;
  details::RequireAssigned<firrtl::CircuitOp> circuitOp;
  std::variant<details::RequireAssigned<firrtl::FModuleOp>,
               details::RequireAssigned<firrtl::FExtModuleOp>>
      moduleOp;
  SmallPtrSet<StringAttr, 8> seenParamNames;
  details::RequireAssigned<details::ModuleContext> moduleContext;

  Location mockLoc() const;
  StringAttr stringRefToAttr(StringRef stringRef);
  std::optional<mlir::Attribute>
  ffiParamToFirParam(const FirrtlParameter &param);
  std::optional<firrtl::FIRRTLType> ffiTypeToFirType(const FirrtlType &type);
  std::optional<mlir::Value> resolveModuleRefExpr(BodyOpBuilder &bodyOpBuilder,
                                                  StringRef refExpr);

  bool visitStmtAttach(BodyOpBuilder &bodyOpBuilder,
                       const FirrtlStatementAttach &stmt);
};

} // namespace chirrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLFFICONTEXT_H
