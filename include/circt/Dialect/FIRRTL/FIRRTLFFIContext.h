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

#include "circt/Dialect/FIRRTL/FIRModuleContextBase.h"
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
struct FirrtlStatementSeqMemory;

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

class ModuleContext final : public firrtl::FIRModuleContextBase {
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
  details::RequireAssigned<std::string> circuitTarget;
  std::variant<details::RequireAssigned<firrtl::FModuleOp>,
               details::RequireAssigned<firrtl::FExtModuleOp>>
      moduleOp;
  SmallPtrSet<StringAttr, 8> seenParamNames;
  details::RequireAssigned<details::ModuleContext> moduleContext;

  Location mockLoc() const;
  llvm::SMLoc mockSMLoc() const;
  StringAttr stringRefToAttr(StringRef stringRef);
  std::optional<mlir::Attribute>
  ffiParamToFirParam(const FirrtlParameter &param);
  std::optional<firrtl::FIRRTLType> ffiTypeToFirType(const FirrtlType &type);
  std::optional<mlir::Value> resolveModuleRefExpr(BodyOpBuilder &bodyOpBuilder,
                                                  StringRef refExpr);

  bool visitStmtAttach(BodyOpBuilder &bodyOpBuilder,
                       const FirrtlStatementAttach &stmt);
  bool visitStmtSeqMemory(BodyOpBuilder &bodyOpBuilder,
                          const FirrtlStatementSeqMemory &stmt);
};

} // namespace chirrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLFFICONTEXT_H
