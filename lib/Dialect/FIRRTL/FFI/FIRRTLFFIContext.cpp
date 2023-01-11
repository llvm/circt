//===- FIRRTLFFIContext.cpp - .fir to FIRRTL dialect parser ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements FFI for CIRCT FIRRTL.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLFFIContext.h"
#include "circt-c/Dialect/FIRRTL.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::SMLoc;

using ModuleSymbolTable = FIRModuleContextBase::ModuleSymbolTable;
using ModuleSymbolTableEntry = FIRModuleContextBase::ModuleSymbolTableEntry;
using SubaccessCache = FIRModuleContextBase::SubaccessCache;
using SymbolValueEntry = FIRModuleContextBase::SymbolValueEntry;
using UnbundledID = FIRModuleContextBase::UnbundledID;
using UnbundledValueEntry = FIRModuleContextBase::UnbundledValueEntry;
using UnbundledValuesList = FIRModuleContextBase::UnbundledValuesList;

namespace {
StringRef attrToStringRef(const Attribute &attr) {
  return llvm::dyn_cast<StringAttr>(attr);
}
} // namespace

// This macro returns the underlying value of a `RequireAssigned`, which
// requires that the value has been set previously, otherwise it will emit an
// error and return in the current function.
#define RA_EXPECT(var, ra, ...)                                                \
  if (!(ra).underlying.has_value()) {                                          \
    this->emitError("expected `" #ra "` to be set");                           \
    return __VA_ARGS__;                                                        \
  }                                                                            \
  var = (ra).underlying.value(); // NOLINT(bugprone-macro-parentheses)

namespace circt::chirrtl::details {

ModuleContext::ModuleContext(FFIContext &ctx, ModuleKind kind,
                             std::string moduleTarget)
    : FIRModuleContextBase{std::move(moduleTarget)}, ffiCtx{ctx}, kind{kind} {}

MLIRContext *ModuleContext::getContext() const { return ffiCtx.mlirCtx.get(); }

InFlightDiagnostic ModuleContext::emitError(const Twine &message) {
  return emitError(ffiCtx.mockSMLoc(), message);
}
InFlightDiagnostic ModuleContext::emitError(SMLoc loc, const Twine &message) {
  ffiCtx.emitError(message.str());
  auto err = mlir::emitError(translateLocation(loc), message);
  err.abandon();
  return err;
}

Location ModuleContext::translateLocation(llvm::SMLoc loc) {
  (void)loc;
  return ffiCtx.mockLoc();
}

} // namespace circt::chirrtl::details

FFIContext::FFIContext() : mlirCtx{std::make_unique<MLIRContext>()} {
  mlirCtx->loadDialect<CHIRRTLDialect>();
  mlirCtx->loadDialect<FIRRTLDialect, hw::HWDialect>();

  module = std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(mockLoc()));
  opBuilder = std::make_unique<mlir::OpBuilder>(module->getBodyRegion());
}

void FFIContext::setErrorHandler(
    std::function<void(std::string_view message)> handler) {
  errorHandler = std::move(handler);
}

void FFIContext::emitError(std::string_view message, bool recoverable) const {
  if (errorHandler) {
    errorHandler(message);
  }
  // TODO: handle the `recoverable` parameter
}

void FFIContext::visitCircuit(StringRef name) {
  moduleContext.underlying.reset();

  circuitOp = opBuilder->create<CircuitOp>(mockLoc(), stringRefToAttr(name));
  circuitTarget = ("~" + name).str();
}

void FFIContext::visitModule(StringRef name) {
  RA_EXPECT(auto &circuitOp, this->circuitOp);
  RA_EXPECT(auto &circuitTarget, this->circuitTarget);

  auto builder = circuitOp.getBodyBuilder();
  moduleOp =
      builder.create<FModuleOp>(mockLoc(), stringRefToAttr(name),
                                ArrayRef<PortInfo>{} /* TODO: annotations */);

  auto moduleTarget = (circuitTarget + "|" + name).str();
  moduleContext.underlying.emplace(*this, details::ModuleKind::Module,
                                   std::move(moduleTarget));
}

void FFIContext::visitExtModule(StringRef name, StringRef defName) {
  RA_EXPECT(auto &circuitOp, this->circuitOp);
  RA_EXPECT(auto &circuitTarget, this->circuitTarget);

  seenParamNames.clear();

  auto builder = circuitOp.getBodyBuilder();
  moduleOp = builder.create<FExtModuleOp>(mockLoc(), stringRefToAttr(name),
                                          ArrayRef<PortInfo>{}, defName
                                          /* TODO: annotations */);

  auto moduleTarget = (circuitTarget + "|" + name).str();
  moduleContext.underlying.emplace(*this, details::ModuleKind::ExtModule,
                                   std::move(moduleTarget));
}

void FFIContext::visitParameter(StringRef name, const FirrtlParameter &param) {
  RA_EXPECT(auto &circuitOp, this->circuitOp);

  auto *moduleOpPtr =
      std::get_if<details::RequireAssigned<firrtl::FExtModuleOp>>(
          &this->moduleOp);

  if (moduleOpPtr == nullptr) {
    emitError("parameter can only be declare under an `extmodule`");
    return;
  }

  auto &moduleOp = *moduleOpPtr;
  RA_EXPECT(auto &lastModuleOp, moduleOp);

  auto firParam = ffiParamToFirParam(param);
  if (!firParam.has_value()) {
    return;
  }

  auto nameId = stringRefToAttr(name);
  if (!seenParamNames.insert(nameId).second) {
    emitError(("redefinition of parameter '" + name + "'").str());
    return;
  }

  auto newParam = ParamDeclAttr::get(nameId, *firParam);
  auto builder = circuitOp.getBodyBuilder();

  auto previous = lastModuleOp->getAttr("parameters");
  if (previous) {
    auto preArr = llvm::cast<ArrayAttr>(previous);

    SmallVector<Attribute> params;
    params.reserve(preArr.size() + 1);
    params.append(preArr.begin(), preArr.end());
    params.push_back(newParam);
    lastModuleOp->setAttr("parameters", builder.getArrayAttr(params));
  } else {
    lastModuleOp->setAttr(
        "parameters", builder.getArrayAttr(SmallVector<Attribute>{newParam}));
  }
}

void FFIContext::visitPort(StringRef name, Direction direction,
                           const FirrtlType &type) {
  std::visit(
      [&](auto &moduleOp) {
        RA_EXPECT(auto &lastModuleOp, moduleOp);
        RA_EXPECT(auto &lastModuleCtx, this->moduleContext);

        auto existedNames = lastModuleOp.getPortNames();
        for (const auto &existedName : existedNames) {
          if (attrToStringRef(existedName) == name) {
            emitError(("redefinition of port name '" + name + "'").str());
            return;
          }
        }

        auto firType = ffiTypeToFirType(type);
        if (!firType.has_value()) {
          return;
        }
        auto info = PortInfo{stringRefToAttr(name), *firType, direction};

        auto index = lastModuleOp.getNumPorts();

        // If the performance of this function is very poor, we can try to cache
        // all ports and finally create `FModuleOp` at once.
        lastModuleOp.insertPorts({std::make_pair(index, info)});

        if (!lastModuleCtx.isExtModule()) {
          auto arg = lastModuleOp.getBody().getArgument(index);
          (void)lastModuleCtx.addSymbolEntry(name, std::move(arg), mockSMLoc());
        }
      },
      moduleOp);
}

void FFIContext::visitStatement(const FirrtlStatement &stmt) {
  auto *moduleOpPtr =
      std::get_if<details::RequireAssigned<firrtl::FModuleOp>>(&this->moduleOp);
  if (moduleOpPtr == nullptr) {
    emitError("statement cannot be placed under an `extmodule`");
    return;
  }
  auto &moduleOp = *moduleOpPtr;
  RA_EXPECT(auto &lastModuleOp, moduleOp);

  auto bodyOpBuilder = mlir::ImplicitLocOpBuilder::atBlockEnd(
      mockLoc(), lastModuleOp.getBodyBlock());

  switch (stmt.kind) {
  case FIRRTL_STATEMENT_KIND_ATTACH:
    visitStmtAttach(bodyOpBuilder, stmt.u.attach);
    break;
  case FIRRTL_STATEMENT_KIND_SEQ_MEMORY:
    visitStmtSeqMemory(bodyOpBuilder, stmt.u.seqMem);
    break;
  default:
    emitError("unknown statement kind");
    break;
  }
}

void FFIContext::exportFIRRTL(llvm::raw_ostream &os) const {
  // TODO: check states first, otherwise a sigsegv will probably happen.

  auto result = exportFIRFile(*module, os);
  if (result.failed()) {
    emitError("failed to export FIRRTL");
  }
}

Location FFIContext::mockLoc() const {
  // no location info available
  return mlir::UnknownLoc::get(mlirCtx.get());
}

SMLoc FFIContext::mockSMLoc() const {
  // no location info available
  return SMLoc::getFromPointer("no location info available");
}

StringAttr FFIContext::stringRefToAttr(StringRef stringRef) {
  return StringAttr::get(mlirCtx.get(), stringRef);
}

std::optional<Attribute>
FFIContext::ffiParamToFirParam(const FirrtlParameter &param) {
  RA_EXPECT(auto &circuitOp, this->circuitOp, std::nullopt);

  auto builder = circuitOp.getBodyBuilder();

  switch (param.kind) {
  case FIRRTL_PARAMETER_KIND_INT: {
    APInt result;
    result = param.u.int_.value;

    // If the integer parameter is less than 32-bits, sign extend this to a
    // 32-bit value.  This needs to eventually emit as a 32-bit value in
    // Verilog and we want to get the size correct immediately.
    if (result.getBitWidth() < 32) {
      result = result.sext(32);
    }

    return builder.getIntegerAttr(
        builder.getIntegerType(result.getBitWidth(), result.isSignBitSet()),
        result);
  }
  case FIRRTL_PARAMETER_KIND_DOUBLE:
    return builder.getF64FloatAttr(param.u.double_.value);
  case FIRRTL_PARAMETER_KIND_STRING:
    return builder.getStringAttr(unwrap(param.u.string.value));
  case FIRRTL_PARAMETER_KIND_RAW:
    return builder.getStringAttr(unwrap(param.u.raw.value));
  }

  emitError("unknown parameter kind");
  return std::nullopt;
}

// NOLINTNEXTLINE(misc-no-recursion)
std::optional<FIRRTLType> FFIContext::ffiTypeToFirType(const FirrtlType &type) {
  auto *mlirCtx = this->mlirCtx.get();

  FIRRTLType firType;

  switch (type.kind) {
  case FIRRTL_TYPE_KIND_UINT:
    firType = UIntType::get(mlirCtx, type.u.uint.width);
    break;
  case FIRRTL_TYPE_KIND_SINT:
    firType = SIntType::get(mlirCtx, type.u.sint.width);
    break;
  case FIRRTL_TYPE_KIND_CLOCK:
    firType = ClockType::get(mlirCtx);
    break;
  case FIRRTL_TYPE_KIND_RESET:
    firType = ResetType::get(mlirCtx);
    break;
  case FIRRTL_TYPE_KIND_ASYNC_RESET:
    firType = AsyncResetType::get(mlirCtx);
    break;
  case FIRRTL_TYPE_KIND_ANALOG:
    firType = AnalogType::get(mlirCtx, type.u.analog.width);
    break;
  case FIRRTL_TYPE_KIND_VECTOR: {
    auto elementType = ffiTypeToFirType(*type.u.vector.type);
    if (!elementType.has_value()) {
      return std::nullopt;
    }
    auto baseType = elementType->dyn_cast<FIRRTLBaseType>();
    if (!baseType) {
      emitError("element must be base type");
      return std::nullopt;
    }

    firType = FVectorType::get(baseType, type.u.vector.count);
    break;
  }
  case FIRRTL_TYPE_KIND_BUNDLE: {
    SmallVector<BundleType::BundleElement, 4> fields;
    fields.reserve(type.u.bundle.count);

    for (size_t i = 0; i < type.u.bundle.count; i++) {
      const auto &field = type.u.bundle.fields[i];

      auto fieldType = ffiTypeToFirType(*field.type);
      if (!fieldType.has_value()) {
        return std::nullopt;
      }
      auto baseType = fieldType->dyn_cast<FIRRTLBaseType>();
      if (!baseType) {
        emitError("field must be base type");
        return std::nullopt;
      }

      fields.emplace_back(stringRefToAttr(unwrap(field.name)), field.flip,
                          baseType);
    }
    firType = BundleType::get(mlirCtx, fields);
    break;
  }
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    emitError("unknown type kind");
    return std::nullopt;
  }

  return firType;
}

std::optional<mlir::Value>
FFIContext::resolveModuleRefExpr(BodyOpBuilder &bodyOpBuilder,
                                 StringRef refExpr) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, {});

  SmallVector<StringRef, 2> refExprParts;
  refExpr.split(refExprParts, ".");

  if (refExprParts[0].empty()) {
    emitError("expected an identifier");
    return {};
  }

  SymbolValueEntry symtabEntry;
  if (lastModuleCtx.lookupSymbolEntry(symtabEntry, refExprParts[0],
                                      mockSMLoc())) {
    return {};
  }

  mlir::Value result;
  if (!lastModuleCtx.resolveSymbolEntry(result, symtabEntry, mockSMLoc(),
                                        false)) {

    // Handle fields
    for (size_t i = 1; i < refExprParts.size(); i++) {
      const auto &fieldName = refExprParts[i];
      if (fieldName.empty()) {
        emitError("expected an field name after '.'");
        return {};
      }

      auto bundle = result.getType().dyn_cast<BundleType>();
      if (!bundle) {
        emitError("subfield requires bundle operand");
        return {};
      }

      auto indexV = bundle.getElementIndex(fieldName);
      if (!indexV.has_value()) {
        std::string typeStr;
        llvm::raw_string_ostream stream{typeStr};
        result.getType().print(stream);
        emitError(
            ("unknown field '" + fieldName + "' in bundle type " + typeStr)
                .str());
        return {};
      }

      auto indexNo = *indexV;

      NamedAttribute attrs = {StringAttr::get(mlirCtx.get(), "fieldIndex"),
                              bodyOpBuilder.getI32IntegerAttr(indexNo)};
      auto resultType = SubfieldOp::inferReturnType({result}, attrs, {});
      if (!resultType) {
        emitError("failed to infer the result type of field");
        return {};
      }

      auto &value = lastModuleCtx.getCachedSubaccess(result, indexNo);
      if (!value) {
        OpBuilder::InsertionGuard guard{bodyOpBuilder};
        bodyOpBuilder.setInsertionPointAfterValue(result);
        auto op = bodyOpBuilder.create<SubfieldOp>(resultType, result, attrs);
        value = op.getResult();
      }

      result = value;
    }

    return result;
  }

  assert(symtabEntry.is<UnbundledID>() && "should be an instance");

  assert(false && "UnbundledID unimplemented"); // TODO

  return {};
}

bool FFIContext::visitStmtAttach(BodyOpBuilder &bodyOpBuilder,
                                 const FirrtlStatementAttach &stmt) {
  SmallVector<Value, 4> operands;
  operands.reserve(stmt.count);

  for (size_t i = 0; i < stmt.count; i++) {
    const auto &ffiOperand = stmt.operands[i];
    auto operand = resolveModuleRefExpr(bodyOpBuilder, unwrap(ffiOperand.expr));
    if (!operand.has_value()) {
      return false;
    }
    operands.push_back(*operand);
  }

  bodyOpBuilder.create<AttachOp>(mockLoc(), operands);
  return true;
}

bool FFIContext::visitStmtSeqMemory(BodyOpBuilder &bodyOpBuilder,
                                    const FirrtlStatementSeqMemory &stmt) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  RUWAttr ruw;
  switch (stmt.readUnderWrite) {
  case FIRRTL_READ_UNDER_WRITE_UNDEFINED:
    ruw = RUWAttr::Undefined;
    break;
  case FIRRTL_READ_UNDER_WRITE_OLD:
    ruw = RUWAttr::Old;
    break;
  case FIRRTL_READ_UNDER_WRITE_NEW:
    ruw = RUWAttr::New;
    break;
  default:
    emitError("unknown RUW value");
    return false;
  }

  auto firType = ffiTypeToFirType(stmt.type);
  if (!firType.has_value()) {
    return false;
  }

  // Transform the parsed vector type into a memory type.
  auto vectorType = (*firType).dyn_cast<FVectorType>();
  if (!vectorType) {
    emitError("smem requires vector type");
    return false;
  }

  auto name = unwrap(stmt.name);
  auto annotations = ArrayAttr::get(mlirCtx.get(), {});
  StringAttr sym = {};
  auto result = bodyOpBuilder.create<SeqMemOp>(
      vectorType.getElementType(), vectorType.getNumElements(), ruw, name,
      NameKindEnum::InterestingName, annotations, sym);
  return !lastModuleCtx.addSymbolEntry(name, result, mockSMLoc());
}

#undef RA_EXPECT
