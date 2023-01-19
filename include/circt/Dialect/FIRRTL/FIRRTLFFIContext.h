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

#include "circt/Dialect/FIRRTL/FIRRTLModuleContext.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <memory>
#include <optional>
#include <stack>
#include <variant>

struct FirrtlType;
struct FirrtlParameter;
struct FirrtlExpr;
struct FirrtlPrim;
struct FirrtlStatement;
struct FirrtlStatementAttach;
struct FirrtlStatementSeqMemory;
struct FirrtlStatementNode;
struct FirrtlStatementWire;
struct FirrtlStatementInvalid;
struct FirrtlStatementWhenBegin;
struct FirrtlStatementElse;
struct FirrtlStatementWhenEnd;
struct FirrtlStatementConnect;
struct FirrtlStatementMemPort;
struct FirrtlStatementPrintf;
struct FirrtlStatementSkip;
struct FirrtlStatementStop;
struct FirrtlStatementAssert;
struct FirrtlStatementAssume;
struct FirrtlStatementCover;

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

class ModuleContext final : public firrtl::FIRRTLModuleContext {
public:
  ModuleContext(FFIContext &ctx, ModuleKind kind, std::string moduleTarget);

  inline bool isExtModule() const { return kind == ModuleKind::ExtModule; }

  MLIRContext *getContext() const override;

  InFlightDiagnostic emitError(const Twine &message = {}) override;
  InFlightDiagnostic emitError(llvm::SMLoc loc,
                               const Twine &message = {}) override;

  Location translateLocation(llvm::SMLoc loc) override;

private:
  FFIContext &ffiCtx;
  ModuleKind kind;
};

class WhenContext {
public:
  WhenContext(ModuleContext &moduleCtx, firrtl::WhenOp whenOp);

  bool hasElseRegion();
  void createElseRegion();
  Block &currentBlock();

private:
  ModuleContext &moduleCtx;
  firrtl::WhenOp whenOp;

  struct UnbundledValueRestorer {
    ModuleContext::UnbundledValuesList &list;
    size_t startingSize;

    inline UnbundledValueRestorer(ModuleContext::UnbundledValuesList &list)
        : list(list) {
      startingSize = list.size();
    }
    inline ~UnbundledValueRestorer() { list.resize(startingSize); }
  };

  struct Scope {
    inline Scope(ModuleContext &moduleContext, Block *block,
                 const UnbundledValueRestorer &uvr)
        : suiteScope{moduleContext, block}, uvr{uvr} {}
    ModuleContext::ContextScope suiteScope;
    UnbundledValueRestorer uvr;
  };

  std::optional<Scope> scope;

  void newScope();
};

} // namespace details

class FFIContext {
public:
  using Direction = firrtl::Direction;
  using BodyOpBuilder = mlir::ImplicitLocOpBuilder;

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
  friend details::ModuleContext;

  std::function<void(std::string_view message)> errorHandler;

  std::unique_ptr<MLIRContext> mlirCtx;
  std::unique_ptr<mlir::ModuleOp> module;
  std::unique_ptr<mlir::OpBuilder> opBuilder;
  details::RequireAssigned<firrtl::CircuitOp> circuitOp;
  details::RequireAssigned<std::string> circuitTarget;
  std::variant<details::RequireAssigned<firrtl::FModuleOp>,
               details::RequireAssigned<firrtl::FExtModuleOp>>
      moduleOp;
  SmallPtrSet<StringAttr, 8> seenParamNames;
  details::RequireAssigned<details::ModuleContext> moduleContext;
  std::stack<details::WhenContext> whenStack;

  Location mockLoc() const;
  llvm::SMLoc mockSMLoc() const;

  ArrayAttr emptyArrayAttr();

  StringAttr stringRefToAttr(StringRef stringRef);
  std::optional<mlir::Attribute>
  ffiParamToFirParam(const FirrtlParameter &param);
  std::optional<firrtl::FIRRTLType> ffiTypeToFirType(const FirrtlType &type);

  bool checkFinal() const;

  std::optional<mlir::Value> resolveRef(BodyOpBuilder &bodyOpBuilder,
                                        StringRef refExpr,
                                        bool invalidate = false);
  std::optional<mlir::Value> resolvePrim(BodyOpBuilder &bodyOpBuilder,
                                         const FirrtlPrim &prim);
  std::optional<mlir::Value> resolveExpr(BodyOpBuilder &bodyOpBuilder,
                                         const FirrtlExpr &expr);

  bool visitStmtAttach(BodyOpBuilder &bodyOpBuilder,
                       const FirrtlStatementAttach &stmt);
  bool visitStmtSeqMemory(BodyOpBuilder &bodyOpBuilder,
                          const FirrtlStatementSeqMemory &stmt);
  bool visitStmtNode(BodyOpBuilder &bodyOpBuilder,
                     const FirrtlStatementNode &stmt);
  bool visitStmtWire(BodyOpBuilder &bodyOpBuilder,
                     const FirrtlStatementWire &stmt);
  bool visitStmtInvalid(BodyOpBuilder &bodyOpBuilder,
                        const FirrtlStatementInvalid &stmt);
  bool visitStmtWhenBegin(BodyOpBuilder &bodyOpBuilder,
                          const FirrtlStatementWhenBegin &stmt);
  bool visitStmtElse(BodyOpBuilder &bodyOpBuilder,
                     const FirrtlStatementElse &stmt);
  bool visitStmtWhenEnd(BodyOpBuilder &bodyOpBuilder,
                        const FirrtlStatementWhenEnd &stmt);
  bool visitStmtConnect(BodyOpBuilder &bodyOpBuilder,
                        const FirrtlStatementConnect &stmt);
  bool visitStmtMemPort(BodyOpBuilder &bodyOpBuilder,
                        const FirrtlStatementMemPort &stmt);
  bool visitStmtPrintf(BodyOpBuilder &bodyOpBuilder,
                       const FirrtlStatementPrintf &stmt);
  bool visitStmtSkip(BodyOpBuilder &bodyOpBuilder,
                     const FirrtlStatementSkip &stmt);
  bool visitStmtStop(BodyOpBuilder &bodyOpBuilder,
                     const FirrtlStatementStop &stmt);
  bool visitStmtAssert(BodyOpBuilder &bodyOpBuilder,
                       const FirrtlStatementAssert &stmt);
  bool visitStmtAssume(BodyOpBuilder &bodyOpBuilder,
                       const FirrtlStatementAssume &stmt);
  bool visitStmtCover(BodyOpBuilder &bodyOpBuilder,
                      const FirrtlStatementCover &stmt);
};

} // namespace chirrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLFFICONTEXT_H
