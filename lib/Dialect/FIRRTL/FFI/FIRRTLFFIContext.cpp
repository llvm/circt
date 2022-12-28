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
  circuitOp = opBuilder->create<CircuitOp>(mockLoc(), stringRefToAttr(name));
}

void FFIContext::visitModule(StringRef name) {
  RA_EXPECT(auto &circuitOp, this->circuitOp);

  auto builder = circuitOp.getBodyBuilder();
  moduleOp =
      builder.create<FModuleOp>(mockLoc(), stringRefToAttr(name),
                                ArrayRef<PortInfo>{} /* TODO: annotations */);
}

void FFIContext::visitExtModule(StringRef name, StringRef defName) {
  RA_EXPECT(auto &circuitOp, this->circuitOp);

  seenParamNames.clear();

  auto builder = circuitOp.getBodyBuilder();
  moduleOp = builder.create<FExtModuleOp>(mockLoc(), stringRefToAttr(name),
                                          ArrayRef<PortInfo>{}, defName
                                          /* TODO: annotations */);
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

  auto newParam = ParamDeclAttr::get(nameId, firParam.value());
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

        // If the performance of this function is very poor, we can try to cache
        // all ports and finally create `FModuleOp` at once.
        lastModuleOp.insertPorts(
            {std::make_pair(lastModuleOp.getNumPorts(), info)});
      },
      moduleOp);
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

#undef RA_EXPECT
