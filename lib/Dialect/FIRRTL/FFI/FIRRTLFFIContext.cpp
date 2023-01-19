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
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::SMLoc;

using ModuleSymbolTable = FIRRTLModuleContext::ModuleSymbolTable;
using ModuleSymbolTableEntry = FIRRTLModuleContext::ModuleSymbolTableEntry;
using SubaccessCache = FIRRTLModuleContext::SubaccessCache;
using SymbolValueEntry = FIRRTLModuleContext::SymbolValueEntry;
using UnbundledID = FIRRTLModuleContext::UnbundledID;
using UnbundledValueEntry = FIRRTLModuleContext::UnbundledValueEntry;
using UnbundledValuesList = FIRRTLModuleContext::UnbundledValuesList;

namespace {
StringRef attrToStringRef(const Attribute &attr) {
  return llvm::dyn_cast<StringAttr>(attr);
}

std::string typeToString(const mlir::Type &type) {
  std::string typeStr;
  llvm::raw_string_ostream stream{typeStr};
  type.print(stream);
  return typeStr;
}

// Extracted from `FIRParser.cpp`
// NOLINTNEXTLINE(misc-no-recursion)
void emitInvalidate(details::ModuleContext &lastModuleCtx,
                    FFIContext::BodyOpBuilder &bodyOpBuilder, Value val,
                    Flow flow) {
  auto tpe = val.getType().cast<FIRRTLBaseType>();

  auto props = tpe.getRecursiveTypeProperties();
  if (props.isPassive && !props.containsAnalog) {
    if (flow == Flow::Source)
      return;
    if (props.hasUninferredWidth)
      bodyOpBuilder.create<ConnectOp>(
          val, bodyOpBuilder.create<InvalidValueOp>(tpe));
    else
      bodyOpBuilder.create<StrictConnectOp>(
          val, bodyOpBuilder.create<InvalidValueOp>(tpe));
    return;
  }

  // Recurse until we hit passive leaves.  Connect any leaves which have sink or
  // duplex flow.
  //
  // TODO: This is very similar to connect expansion in the LowerTypes pass
  // works.  Find a way to unify this with methods common to LowerTypes or to
  // have LowerTypes to the actual work here, e.g., emitting a partial connect
  // to only the leaf sources.
  TypeSwitch<FIRRTLType>(tpe)
      .Case<BundleType>([&](auto tpe) {
        for (size_t i = 0, e = tpe.getNumElements(); i < e; ++i) {
          auto &subfield = lastModuleCtx.getCachedSubaccess(val, i);
          if (!subfield) {
            OpBuilder::InsertionGuard guard(bodyOpBuilder);
            bodyOpBuilder.setInsertionPointAfterValue(val);
            subfield = bodyOpBuilder.create<SubfieldOp>(val, i);
          }
          emitInvalidate(lastModuleCtx, bodyOpBuilder, subfield,
                         tpe.getElement(i).isFlip ? swapFlow(flow) : flow);
        }
      })
      .Case<FVectorType>([&](auto tpe) {
        auto tpex = tpe.getElementType();
        for (size_t i = 0, e = tpe.getNumElements(); i != e; ++i) {
          auto &subindex = lastModuleCtx.getCachedSubaccess(val, i);
          if (!subindex) {
            OpBuilder::InsertionGuard guard(bodyOpBuilder);
            bodyOpBuilder.setInsertionPointAfterValue(val);
            subindex = bodyOpBuilder.create<SubindexOp>(tpex, val, i);
          }
          emitInvalidate(lastModuleCtx, bodyOpBuilder, subindex, flow);
        }
      });
}

void emitInvalidate(details::ModuleContext &lastModuleCtx,
                    FFIContext::BodyOpBuilder &bodyOpBuilder, Value val) {
  emitInvalidate(lastModuleCtx, bodyOpBuilder, val, firrtl::foldFlow(val));
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
    : FIRRTLModuleContext{std::move(moduleTarget)}, ffiCtx{ctx}, kind{kind} {}

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

//
// WhenContext
//

WhenContext::WhenContext(ModuleContext &moduleCtx, firrtl::WhenOp whenOp)
    : moduleCtx{moduleCtx}, whenOp{whenOp} {
  newScope();
}

bool WhenContext::hasElseRegion() { return whenOp.hasElseRegion(); }

void WhenContext::createElseRegion() {
  whenOp.createElseRegion();
  newScope();
}

Block &WhenContext::currentBlock() {
  if (!whenOp.hasElseRegion()) {
    return whenOp.getThenBlock();
  }
  return whenOp.getElseBlock();
}

void WhenContext::newScope() {
  scope.reset();
  scope.emplace(
      // Declarations within the suite are scoped to within the suite.
      moduleCtx, &currentBlock(),
      // After parsing the when region, we can release any new entries
      // in unbundledValues since the symbol table entries that refer
      // to them will be gone.
      moduleCtx.unbundledValues);
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

void FFIContext::visitDeclaration(const FirrtlDeclaration &decl) {
  auto currMod = currentModule();
  if (!currMod.has_value()) {
    return;
  }
  auto &[lastModuleOp, blockToInsertInto] = *currMod;

  auto bodyOpBuilder =
      mlir::ImplicitLocOpBuilder::atBlockEnd(mockLoc(), blockToInsertInto);

  switch (decl.kind) {
  case FIRRTL_DECLARATION_KIND_INSTANCE:
    visitDeclInstance(bodyOpBuilder, decl.u.instance);
    break;
  case FIRRTL_DECLARATION_KIND_SEQ_MEMORY:
    visitDeclSeqMemory(bodyOpBuilder, decl.u.seqMem);
    break;
  case FIRRTL_DECLARATION_KIND_NODE:
    visitDeclNode(bodyOpBuilder, decl.u.node);
    break;
  case FIRRTL_DECLARATION_KIND_WIRE:
    visitDeclWire(bodyOpBuilder, decl.u.wire);
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    emitError("unknown declaration kind");
    break;
  }
}

void FFIContext::visitStatement(const FirrtlStatement &stmt) {
  auto currMod = currentModule();
  if (!currMod.has_value()) {
    return;
  }
  auto &[lastModuleOp, blockToInsertInto] = *currMod;

  auto bodyOpBuilder =
      mlir::ImplicitLocOpBuilder::atBlockEnd(mockLoc(), blockToInsertInto);

  switch (stmt.kind) {
  case FIRRTL_STATEMENT_KIND_ATTACH:
    visitStmtAttach(bodyOpBuilder, stmt.u.attach);
    break;
  case FIRRTL_STATEMENT_KIND_INVALID:
    visitStmtInvalid(bodyOpBuilder, stmt.u.invalid);
    break;
  case FIRRTL_STATEMENT_KIND_WHEN_BEGIN:
    visitStmtWhenBegin(bodyOpBuilder, stmt.u.whenBegin);
    break;
  case FIRRTL_STATEMENT_KIND_ELSE:
    visitStmtElse(bodyOpBuilder, stmt.u.else_);
    break;
  case FIRRTL_STATEMENT_KIND_WHEN_END:
    visitStmtWhenEnd(bodyOpBuilder, stmt.u.whenEnd);
    break;
  case FIRRTL_STATEMENT_KIND_CONNECT:
    visitStmtConnect(bodyOpBuilder, stmt.u.connect);
    break;
  case FIRRTL_STATEMENT_KIND_MEM_PORT:
    visitStmtMemPort(bodyOpBuilder, stmt.u.memPort);
    break;
  case FIRRTL_STATEMENT_KIND_PRINTF:
    visitStmtPrintf(bodyOpBuilder, stmt.u.printf);
    break;
  case FIRRTL_STATEMENT_KIND_SKIP:
    visitStmtSkip(bodyOpBuilder, stmt.u.skip);
    break;
  case FIRRTL_STATEMENT_KIND_STOP:
    visitStmtStop(bodyOpBuilder, stmt.u.stop);
    break;
  case FIRRTL_STATEMENT_KIND_ASSERT:
    visitStmtAssert(bodyOpBuilder, stmt.u.assert);
    break;
  case FIRRTL_STATEMENT_KIND_ASSUME:
    visitStmtAssume(bodyOpBuilder, stmt.u.assume);
    break;
  case FIRRTL_STATEMENT_KIND_COVER:
    visitStmtCover(bodyOpBuilder, stmt.u.cover);
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    emitError("unknown statement kind");
    break;
  }
}

void FFIContext::exportFIRRTL(llvm::raw_ostream &os) const {
  if (!checkFinal()) {
    return;
  }

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

ArrayAttr FFIContext::emptyArrayAttr() {
  return ArrayAttr::get(mlirCtx.get(), {});
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

bool FFIContext::checkFinal() const {
  if (!whenStack.empty()) {
    emitError("expected " + std::to_string(whenStack.size()) + " WhenEnd");
    return false;
  }
  return true;
}

std::optional<std::pair<firrtl::FModuleOp &, Block *>>
FFIContext::currentModule() {
  auto *moduleOpPtr =
      std::get_if<details::RequireAssigned<firrtl::FModuleOp>>(&this->moduleOp);
  if (moduleOpPtr == nullptr) {
    emitError("declaration or statement cannot be placed under an `extmodule`");
    return std::nullopt;
  }
  auto &moduleOp = *moduleOpPtr;
  RA_EXPECT(auto &lastModuleOp, moduleOp, std::nullopt);

  Block *blockToInsertInto;
  if (!whenStack.empty()) {
    blockToInsertInto = &whenStack.top().currentBlock();
  } else {
    blockToInsertInto = lastModuleOp.getBodyBlock();
  }

  return std::make_pair(std::ref(lastModuleOp), blockToInsertInto);
}

std::optional<mlir::Value> FFIContext::resolveRef(BodyOpBuilder &bodyOpBuilder,
                                                  StringRef refExpr,
                                                  bool invalidate) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, std::nullopt);

  SmallVector<StringRef, 2> unresolvedFields;
  refExpr.split(unresolvedFields, ".");
  std::reverse(unresolvedFields.begin(), unresolvedFields.end());

  const auto &consumeField = [&]() -> std::optional<StringRef> {
    if (unresolvedFields.empty()) {
      return std::nullopt;
    }
    return unresolvedFields.pop_back_val();
  };

  auto field = consumeField();
  if (!field.has_value()) {
    emitError("expected an identifier");
    return {};
  }

  SymbolValueEntry symtabEntry;
  if (lastModuleCtx.lookupSymbolEntry(symtabEntry, *field, mockSMLoc())) {
    return {};
  }

  mlir::Value result;
  if (lastModuleCtx.resolveSymbolEntry(result, symtabEntry, mockSMLoc(),
                                       false)) {
    assert(symtabEntry.is<UnbundledID>() && "should be an instance");

    if (invalidate && unresolvedFields.empty()) {
      // Invalidate all of the results of the bundled value.
      auto unbundledId = symtabEntry.get<UnbundledID>() - 1;
      auto &ubEntry = lastModuleCtx.getUnbundledEntry(unbundledId);
      for (auto elt : ubEntry) {
        emitInvalidate(lastModuleCtx, bodyOpBuilder, elt.second);
      }
      return Value{};
    }

    // Handle the normal "instance.x" reference.
    field = consumeField();
    if (!field.has_value()) {
      emitError("expected field name in field reference");
      return {};
    }
    if (lastModuleCtx.resolveSymbolEntry(result, symtabEntry, *field,
                                         mockSMLoc())) {
      return {};
    }
  }

  // Handle optional exp postscript

  // Handle fields

  for (field = consumeField(); field.has_value(); field = consumeField()) {
    const auto &fieldName = *field;
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
      emitError(("unknown field '" + fieldName + "' in bundle type " +
                 typeToString(result.getType()))
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

  // TODO: Handle subindex: exp ::= exp '[' intLit ']' | exp '[' exp ']'

  if (invalidate) {
    emitInvalidate(lastModuleCtx, bodyOpBuilder, result);
    return Value{};
  }

  return result;
}

// NOLINTNEXTLINE(misc-no-recursion)
std::optional<mlir::Value> FFIContext::resolvePrim(BodyOpBuilder &bodyOpBuilder,
                                                   const FirrtlPrim &prim) {
  size_t numOperandsExpected;

  // NOLINTNEXTLINE(bugprone-branch-clone)
  switch (prim.op) {
  case FIRRTL_PRIM_OP_VALIDIF:
    numOperandsExpected = 2;
    break;
#define PRIM_OP(ENUM, CLASS, NUMOPERANDS)                                      \
  case ENUM:                                                                   \
    numOperandsExpected = NUMOPERANDS;                                         \
    break;
#include "FIRRTLPrimOp.def"
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    emitError("unknown primitive operation");
    return std::nullopt;
  }

  // Parse the operands and constant integer arguments.
  SmallVector<Value, 3> operands;
  SmallVector<int64_t, 3> integers;

  for (size_t i = 0; i < prim.argsCount; i++) {
    const auto &arg = prim.args[i];

    switch (arg.kind) {
    case FIRRTL_PRIM_ARG_KIND_INT_LIT:
      // Handle the integer constant case if present.
      integers.emplace_back(arg.u.intLit.value);
      break;

    case FIRRTL_PRIM_ARG_KIND_EXPR: {
      // Otherwise it must be a value operand.  These must all come before the
      // integers.
      if (!integers.empty()) {
        emitError("expected more integer constants");
        return std::nullopt;
      }

      auto operand = resolveExpr(bodyOpBuilder, arg.u.expr.value);
      if (!operand.has_value()) {
        emitError("expected expression in primitive operand");
        return std::nullopt;
      }
      operands.emplace_back(*operand);
      break;
    }
    default: // NOLINT(clang-diagnostic-covered-switch-default)
      emitError("unknown primitive arg kind (1)");
      return std::nullopt;
    }
  }

  SmallVector<FIRRTLType, 3> opTypes;
  for (auto v : operands) {
    opTypes.emplace_back(v.getType().cast<FIRRTLType>());
  }

  SmallVector<StringAttr, 2> attrNames;

  switch (prim.op) {
  case FIRRTL_PRIM_OP_BITS_EXTRACT: {
    static auto hiIdentifier = StringAttr::get(mlirCtx.get(), "lo");
    static auto loIdentifier = StringAttr::get(mlirCtx.get(), "hi");
    attrNames.emplace_back(hiIdentifier); // "hi"
    attrNames.emplace_back(loIdentifier); // "lo"
    break;
  }
  case FIRRTL_PRIM_OP_HEAD:
  case FIRRTL_PRIM_OP_PAD:
  case FIRRTL_PRIM_OP_SHIFT_LEFT:
  case FIRRTL_PRIM_OP_SHIFT_RIGHT:
  case FIRRTL_PRIM_OP_TAIL: {
    static auto amountIdentifier = StringAttr::get(mlirCtx.get(), "amount");
    attrNames.emplace_back(amountIdentifier);
    break;
  }
  default:
    break;
  }

  if (operands.size() != numOperandsExpected) {
    assert(numOperandsExpected <= 3);
    static const char *numberName[] = {"zero", "one", "two", "three"};
    const char *optionalS = &"s"[numOperandsExpected == 1];
    emitError(std::string{"operation requires "} +
              numberName[numOperandsExpected] + " operand" + optionalS);
    return std::nullopt;
  }

  if (integers.size() != attrNames.size()) {
    emitError("expected " + std::to_string(attrNames.size()) +
              " constant arguments");
    return std::nullopt;
  }

  NamedAttrList attrs;
  for (size_t i = 0, e = attrNames.size(); i != e; ++i) {
    attrs.append(attrNames[i], bodyOpBuilder.getI32IntegerAttr(integers[i]));
  }

  switch (prim.op) {
#define PRIM_OP(ENUM, CLASS, NUMOPERANDS)                                      \
  case ENUM: {                                                                 \
    auto resultTy = CLASS::inferReturnType(operands, attrs, {});               \
    if (!resultTy) {                                                           \
      /* only call translateLocation on an error case, it is expensive. */     \
      (void)CLASS::validateAndInferReturnType(operands, attrs, mockLoc());     \
      return std::nullopt;                                                     \
    }                                                                          \
    return bodyOpBuilder.create<CLASS>(resultTy, operands, attrs);             \
  }
#include "FIRRTLPrimOp.def"

  // Expand `validif(a, b)` expressions to simply `b`.  A `validif` expression
  // is converted to a direct connect by the Scala FIRRTL Compiler's
  // `RemoveValidIfs` pass.  We circumvent that and just squash these during
  // parsing.
  case FIRRTL_PRIM_OP_VALIDIF: {
    if (opTypes.size() != 2 || !integers.empty()) {
      emitError("operation requires two operands and no constants");
      return std::nullopt;
    }
    auto lhsUInt = opTypes[0].dyn_cast<UIntType>();
    if (!lhsUInt) {
      emitError("first operand should have UInt type");
      return std::nullopt;
    }
    auto lhsWidth = lhsUInt.getWidthOrSentinel();
    if (lhsWidth != -1 && lhsWidth != 1) {
      emitError("first operand should have 'uint<1>' type");
      return std::nullopt;
    }
    // Skip the `validif` and emit the second, non-condition operand.
    return operands[1];
  }
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    emitError("unknown primitive arg kind (2)");
    return std::nullopt;
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
std::optional<mlir::Value> FFIContext::resolveExpr(BodyOpBuilder &bodyOpBuilder,
                                                   const FirrtlExpr &expr) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, std::nullopt);

  const auto &resolveIntExpr =
      [&](const auto &intExpr) -> std::optional<mlir::Value> {
    using E = std::decay_t<decltype(intExpr)>;

    static_assert(std::is_same_v<E, FirrtlExprSInt> ||
                  std::is_same_v<E, FirrtlExprUInt>);

    constexpr bool isSigned = std::is_same_v<E, FirrtlExprSInt>;

    std::optional<Value> result;

    APInt value;
    value = intExpr.value;
    auto width = intExpr.width;

    // Construct an integer attribute of the right width.
    auto type = IntType::get(bodyOpBuilder.getContext(), isSigned, width);

    IntegerType::SignednessSemantics signedness =
        isSigned ? IntegerType::Signed : IntegerType::Unsigned;
    if (width == 0) {
      if (!value.isZero()) {
        emitError("zero bit constant must be zero");
        return std::nullopt;
      }
      value = value.trunc(0);
    } else if (width != -1) {
      // Convert to the type's width, checking value fits in destination width.
      bool valueFits =
          isSigned ? value.isSignedIntN(width) : value.isIntN(width);
      if (!valueFits) {
        emitError("initializer too wide for declared width");
        return std::nullopt;
      }
      value = isSigned ? value.sextOrTrunc(width) : value.zextOrTrunc(width);
    }

    Type attrType =
        IntegerType::get(type.getContext(), value.getBitWidth(), signedness);
    auto attr = bodyOpBuilder.getIntegerAttr(attrType, value);

    // Check to see if we've already created this constant.  If so, reuse it.
    auto &entry = lastModuleCtx.constantCache[{attr, type}];
    if (entry) {
      // If we already had an entry, reuse it.
      result = entry;
      return result;
    }

    // Make sure to insert constants at the top level of the module to maintain
    // dominance.
    OpBuilder::InsertPoint savedIP;

    auto *parentOp = bodyOpBuilder.getInsertionBlock()->getParentOp();
    if (!isa<FModuleOp>(parentOp)) {
      savedIP = bodyOpBuilder.saveInsertionPoint();
      while (!isa<FModuleOp>(parentOp)) {
        bodyOpBuilder.setInsertionPoint(parentOp);
        parentOp = bodyOpBuilder.getInsertionBlock()->getParentOp();
      }
    }

    auto op = bodyOpBuilder.create<ConstantOp>(type, value);
    entry = op;
    result = op;

    if (savedIP.isSet()) {
      bodyOpBuilder.setInsertionPoint(savedIP.getBlock(), savedIP.getPoint());
    }
    return result;
  };

  switch (expr.kind) {
  case FIRRTL_EXPR_KIND_UINT:
    return resolveIntExpr(expr.u.uint);
  case FIRRTL_EXPR_KIND_SINT:
    return resolveIntExpr(expr.u.sint);
  case FIRRTL_EXPR_KIND_REF:
    return resolveRef(bodyOpBuilder, unwrap(expr.u.ref.value));
  case FIRRTL_EXPR_KIND_PRIM:
    return resolvePrim(bodyOpBuilder, *expr.u.prim.value);
  }

  emitError("unknown expression kind");
  return std::nullopt;
}

bool FFIContext::visitDeclInstance(BodyOpBuilder &bodyOpBuilder,
                                   const FirrtlDeclarationInstance &decl) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  auto name = unwrap(decl.name);
  auto moduleName = unwrap(decl.moduleName);

  // Look up the module that is being referenced.
  auto circuit =
      bodyOpBuilder.getBlock()->getParentOp()->getParentOfType<CircuitOp>();
  auto referencedModule =
      dyn_cast_or_null<FModuleLike>(circuit.lookupSymbol(moduleName));
  if (!referencedModule) {
    emitError(("use of undefined module name '" + moduleName + "' in instance")
                  .str());
    return false;
  }

  SmallVector<PortInfo> modulePorts = referencedModule.getPorts();

  // Make a bundle of the inputs and outputs of the specified module.
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(modulePorts.size());
  SmallVector<std::pair<StringAttr, Type>, 4> resultNamesAndTypes;

  for (auto port : modulePorts) {
    resultTypes.push_back(port.type);
    resultNamesAndTypes.push_back({port.name, port.type});
  }

  auto annotations = emptyArrayAttr();
  SmallVector<Attribute, 4> portAnnotations(modulePorts.size(), annotations);

  StringAttr sym = {};
  auto result = bodyOpBuilder.create<InstanceOp>(
      referencedModule, name, NameKindEnum::InterestingName,
      annotations.getValue(), portAnnotations, false, sym);

  // Since we are implicitly unbundling the instance results, we need to keep
  // track of the mapping from bundle fields to results in the unbundledValues
  // data structure.  Build our entry now.
  UnbundledValueEntry unbundledValueEntry;
  unbundledValueEntry.reserve(modulePorts.size());
  for (size_t i = 0, e = modulePorts.size(); i != e; ++i)
    unbundledValueEntry.push_back({modulePorts[i].name, result.getResult(i)});

  // Add it to unbundledValues and add an entry to the symbol table to remember
  // it.
  lastModuleCtx.unbundledValues.push_back(std::move(unbundledValueEntry));
  auto entryId = UnbundledID(lastModuleCtx.unbundledValues.size());
  return !lastModuleCtx.addSymbolEntry(name, entryId, mockSMLoc());
}

bool FFIContext::visitDeclSeqMemory(BodyOpBuilder &bodyOpBuilder,
                                    const FirrtlDeclarationSeqMemory &decl) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  RUWAttr ruw;
  switch (decl.readUnderWrite) {
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

  auto firType = ffiTypeToFirType(decl.type);
  if (!firType.has_value()) {
    return false;
  }

  // Transform the parsed vector type into a memory type.
  auto vectorType = (*firType).dyn_cast<FVectorType>();
  if (!vectorType) {
    emitError("smem requires vector type");
    return false;
  }

  auto name = unwrap(decl.name);
  auto annotations = emptyArrayAttr();
  StringAttr sym = {};
  auto result = bodyOpBuilder.create<SeqMemOp>(
      vectorType.getElementType(), vectorType.getNumElements(), ruw, name,
      NameKindEnum::InterestingName, annotations, sym);
  return !lastModuleCtx.addSymbolEntry(name, result, mockSMLoc());
}

bool FFIContext::visitDeclNode(BodyOpBuilder &bodyOpBuilder,
                               const FirrtlDeclarationNode &decl) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  auto name = unwrap(decl.name);

  auto optInitializer = resolveExpr(bodyOpBuilder, decl.expr);
  if (!optInitializer.has_value()) {
    emitError("expected expression for node");
    return false;
  }
  auto initializer = *optInitializer;

  // Error out in the following conditions:
  //
  //   1. Node type is Analog (at the top level)
  //   2. Node type is not passive under an optional outer flip
  //      (analog field is okay)
  //
  // Note: (1) is more restictive than normal NodeOp verification, but
  // this is added to align with the SFC. (2) is less restrictive than
  // the SFC to accomodate for situations where the node is something
  // weird like a module output or an instance input.
  auto initializerType = initializer.getType().cast<FIRRTLType>();
  auto initializerBaseType = initializer.getType().dyn_cast<FIRRTLBaseType>();
  if (initializerType.isa<AnalogType>() ||
      !(initializerBaseType && initializerBaseType.isPassive())) {
    emitError(
        "Node cannot be analog and must be passive or passive under a flip" +
        typeToString(initializer.getType()));
    return false;
  }

  auto annotations = emptyArrayAttr();
  auto sym = StringAttr{};
  auto result = bodyOpBuilder.create<NodeOp>(
      initializer.getType(), initializer, name, NameKindEnum::InterestingName,
      annotations, sym);
  return !lastModuleCtx.addSymbolEntry(name, result, mockSMLoc());
}

bool FFIContext::visitDeclWire(BodyOpBuilder &bodyOpBuilder,
                               const FirrtlDeclarationWire &decl) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  auto name = unwrap(decl.name);
  auto type = ffiTypeToFirType(decl.type);
  if (!type.has_value()) {
    return false;
  }

  auto annotations = emptyArrayAttr();
  StringAttr sym = {};

  auto result = bodyOpBuilder.create<WireOp>(
      *type, name, NameKindEnum::InterestingName, annotations,
      sym ? hw::InnerSymAttr::get(sym) : hw::InnerSymAttr());
  return !lastModuleCtx.addSymbolEntry(name, result, mockSMLoc());
}

bool FFIContext::visitStmtAttach(BodyOpBuilder &bodyOpBuilder,
                                 const FirrtlStatementAttach &stmt) {
  SmallVector<Value, 4> operands;
  operands.reserve(stmt.count);

  for (size_t i = 0; i < stmt.count; i++) {
    const auto &ffiOperand = stmt.operands[i];
    auto operand = resolveExpr(bodyOpBuilder, ffiOperand.expr);
    if (!operand.has_value()) {
      return false;
    }
    operands.push_back(*operand);
  }

  bodyOpBuilder.create<AttachOp>(mockLoc(), operands);
  return true;
}

bool FFIContext::visitStmtInvalid(BodyOpBuilder &bodyOpBuilder,
                                  const FirrtlStatementInvalid &stmt) {
  return resolveRef(bodyOpBuilder, unwrap(stmt.ref.value), true).has_value();
}

bool FFIContext::visitStmtWhenBegin(BodyOpBuilder &bodyOpBuilder,
                                    const FirrtlStatementWhenBegin &stmt) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  auto condition = resolveExpr(bodyOpBuilder, stmt.condition);
  if (!condition.has_value()) {
    return false;
  }

  // Create the IR representation for the when.
  auto whenStmt =
      bodyOpBuilder.create<WhenOp>(*condition, /*createElse*/ false);
  whenStack.emplace(lastModuleCtx, whenStmt);
  return true;
}

bool FFIContext::visitStmtElse(BodyOpBuilder &bodyOpBuilder,
                               const FirrtlStatementElse &stmt) {
  if (whenStack.empty()) {
    emitError("expected a WhenBegin before WhenElse");
    return false;
  }

  auto &whenCtx = whenStack.top();

  if (whenCtx.hasElseRegion()) {
    emitError("already has an else region");
    return false;
  }

  // Create an else block to parse into.
  whenCtx.createElseRegion();
  return true;
}

bool FFIContext::visitStmtWhenEnd(BodyOpBuilder &bodyOpBuilder,
                                  const FirrtlStatementWhenEnd &stmt) {
  if (whenStack.empty()) {
    emitError("expected a WhenBegin before WhenEnd");
    return false;
  }
  whenStack.pop();
  return true;
}

bool FFIContext::visitStmtConnect(BodyOpBuilder &bodyOpBuilder,
                                  const FirrtlStatementConnect &stmt) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  auto lhs = resolveRef(bodyOpBuilder, unwrap(stmt.left.value));
  auto rhs = resolveRef(bodyOpBuilder, unwrap(stmt.right.value));
  if (!lhs.has_value() || !rhs.has_value()) {
    return false;
  }

  auto lhsType = lhs->getType().cast<FIRRTLType>();
  auto rhsType = rhs->getType().cast<FIRRTLType>();

  if (!stmt.isPartial) {
    if (!areTypesEquivalent(lhsType, rhsType)) {
      emitError("cannot connect non-equivalent type " + typeToString(rhsType) +
                " to " + typeToString(lhsType));
      return false;
    }
    emitConnect(bodyOpBuilder, *lhs, *rhs);
  } else {
    if (!areTypesWeaklyEquivalent(lhsType, rhsType)) {
      emitError("cannot partially connect non-weakly-equivalent type " +
                typeToString(rhsType) + " to " + typeToString(lhsType));
      return false;
    }
    emitPartialConnect(bodyOpBuilder, *lhs, *rhs, lastModuleCtx);
  }
  return true;
}

bool FFIContext::visitStmtMemPort(BodyOpBuilder &bodyOpBuilder,
                                  const FirrtlStatementMemPort &stmt) {
  RA_EXPECT(auto &lastModuleCtx, this->moduleContext, false);

  MemDirAttr direction;
  switch (stmt.direction) {
  case FIRRTL_MEM_DIRECTION_INFER:
    direction = MemDirAttr::Infer;
    break;
  case FIRRTL_MEM_DIRECTION_READ:
    direction = MemDirAttr::Read;
    break;
  case FIRRTL_MEM_DIRECTION_WRITE:
    direction = MemDirAttr::Write;
    break;
  case FIRRTL_MEM_DIRECTION_READ_WRITE:
    direction = MemDirAttr::ReadWrite;
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    emitError("unknown memory port direction");
    return false;
  }

  auto memory = resolveRef(bodyOpBuilder, unwrap(stmt.memName.value));
  auto indexExpr = resolveExpr(bodyOpBuilder, stmt.memIndex);
  auto clock = resolveExpr(bodyOpBuilder, stmt.clock);
  if (!memory.has_value() || !indexExpr.has_value() || !clock.has_value()) {
    return false;
  }

  auto memVType = memory->getType().dyn_cast<CMemoryType>();
  if (!memVType) {
    emitError("memory port should have behavioral memory type");
    return false;
  }

  auto resultType = memVType.getElementType();
  auto annotations = emptyArrayAttr();
  auto name = unwrap(stmt.name);

  // Create the memory port at the location of the cmemory.
  Value memoryPort, memoryData;
  {
    OpBuilder::InsertionGuard guard(bodyOpBuilder);
    bodyOpBuilder.setInsertionPointAfterValue(*memory);
    auto memoryPortOp = bodyOpBuilder.create<MemoryPortOp>(
        resultType, CMemoryPortType::get(mlirCtx.get()), *memory, direction,
        name, annotations);
    memoryData = memoryPortOp.getResult(0);
    memoryPort = memoryPortOp.getResult(1);
  }

  // Create a memory port access in the current scope.
  bodyOpBuilder.create<MemoryPortAccessOp>(memoryPort, *indexExpr, *clock);
  return !lastModuleCtx.addSymbolEntry(name, memoryData, mockSMLoc(), true);
}

bool FFIContext::visitStmtPrintf(BodyOpBuilder &bodyOpBuilder,
                                 const FirrtlStatementPrintf &stmt) {
  auto clock = resolveExpr(bodyOpBuilder, stmt.clock);
  auto condition = resolveExpr(bodyOpBuilder, stmt.condition);
  if (!clock.has_value() || !condition.has_value()) {
    return false;
  }

  SmallVector<Value, 4> operands;
  operands.reserve(stmt.operandsCount);
  for (size_t i = 0; i < stmt.operandsCount; i++) {
    auto operand = resolveExpr(bodyOpBuilder, stmt.operands[i]);
    if (!operand.has_value()) {
      return false;
    }
    operands.emplace_back(*operand);
  }

  StringAttr name;
  if (stmt.name != nullptr) {
    name = stringRefToAttr(unwrap(*stmt.name));
  } else {
    name = StringAttr::get(mlirCtx.get(), "");
  }

  bodyOpBuilder.create<PrintFOp>(
      *clock, *condition, bodyOpBuilder.getStringAttr(unwrap(stmt.format)),
      operands, name);
  return true;
}

bool FFIContext::visitStmtSkip(BodyOpBuilder &bodyOpBuilder,
                               const FirrtlStatementSkip &stmt) {
  bodyOpBuilder.create<SkipOp>();
  return true;
}

bool FFIContext::visitStmtStop(BodyOpBuilder &bodyOpBuilder,
                               const FirrtlStatementStop &stmt) {
  auto clock = resolveExpr(bodyOpBuilder, stmt.clock);
  auto condition = resolveExpr(bodyOpBuilder, stmt.condition);
  if (!clock.has_value() || !condition.has_value()) {
    return false;
  }

  StringAttr name;
  if (stmt.name != nullptr) {
    name = stringRefToAttr(unwrap(*stmt.name));
  } else {
    name = StringAttr::get(mlirCtx.get(), "");
  }

  bodyOpBuilder.create<StopOp>(
      *clock, *condition, bodyOpBuilder.getI32IntegerAttr(stmt.exitCode), name);
  return true;
}

bool FFIContext::visitStmtAssert(BodyOpBuilder &bodyOpBuilder,
                                 const FirrtlStatementAssert &stmt) {
  auto clock = resolveExpr(bodyOpBuilder, stmt.clock);
  auto predicate = resolveExpr(bodyOpBuilder, stmt.predicate);
  auto enable = resolveExpr(bodyOpBuilder, stmt.enable);
  if (!clock.has_value() || !predicate.has_value() || !enable.has_value()) {
    return false;
  }

  StringAttr name;
  if (stmt.name != nullptr) {
    name = stringRefToAttr(unwrap(*stmt.name));
  } else {
    name = StringAttr::get(mlirCtx.get(), "");
  }

  bodyOpBuilder.create<AssertOp>(*clock, *predicate, *enable,
                                 stringRefToAttr(unwrap(stmt.message)),
                                 ValueRange{}, name.getValue());
  return true;
}

bool FFIContext::visitStmtAssume(BodyOpBuilder &bodyOpBuilder,
                                 const FirrtlStatementAssume &stmt) {
  auto clock = resolveExpr(bodyOpBuilder, stmt.clock);
  auto predicate = resolveExpr(bodyOpBuilder, stmt.predicate);
  auto enable = resolveExpr(bodyOpBuilder, stmt.enable);
  if (!clock.has_value() || !predicate.has_value() || !enable.has_value()) {
    return false;
  }

  StringAttr name;
  if (stmt.name != nullptr) {
    name = stringRefToAttr(unwrap(*stmt.name));
  } else {
    name = StringAttr::get(mlirCtx.get(), "");
  }

  bodyOpBuilder.create<AssumeOp>(*clock, *predicate, *enable,
                                 stringRefToAttr(unwrap(stmt.message)),
                                 ValueRange{}, name.getValue());
  return true;
}

bool FFIContext::visitStmtCover(BodyOpBuilder &bodyOpBuilder,
                                const FirrtlStatementCover &stmt) {

  auto clock = resolveExpr(bodyOpBuilder, stmt.clock);
  auto predicate = resolveExpr(bodyOpBuilder, stmt.predicate);
  auto enable = resolveExpr(bodyOpBuilder, stmt.enable);
  if (!clock.has_value() || !predicate.has_value() || !enable.has_value()) {
    return false;
  }

  StringAttr name;
  if (stmt.name != nullptr) {
    name = stringRefToAttr(unwrap(*stmt.name));
  } else {
    name = StringAttr::get(mlirCtx.get(), "");
  }

  bodyOpBuilder.create<CoverOp>(*clock, *predicate, *enable,
                                stringRefToAttr(unwrap(stmt.message)),
                                ValueRange{}, name.getValue());
  return true;
}

#undef RA_EXPECT
