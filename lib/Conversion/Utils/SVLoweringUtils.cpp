//===- SVLoweringUtils.cpp - Shared helpers for SV lowering ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SVLoweringUtils.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace circt;

namespace {

static constexpr StringLiteral kFileDescriptorGetterSymName =
    "__circt_lib_logging::FileDescriptor::get";
static constexpr StringLiteral kFileDescriptorFragmentSymName =
    "CIRCT_LIB_LOGGING_FRAGMENT";
static constexpr StringLiteral kFileDescriptorMacroName = "__CIRCT_LIB_LOGGING";

StringAttr getFileDescriptorGetterSymName(MLIRContext *context) {
  return StringAttr::get(context, kFileDescriptorGetterSymName);
}

StringAttr getFileDescriptorFragmentSymName(MLIRContext *context) {
  return StringAttr::get(context, kFileDescriptorFragmentSymName);
}

Value createProceduralFileDescriptorGetterCallImpl(OpBuilder &builder,
                                                   Location loc,
                                                   Value fileName) {
  assert(fileName.getType() == hw::StringType::get(builder.getContext()) &&
         "expected !hw.string file name");
  return sv::FuncCallProceduralOp::create(
             builder, loc, TypeRange{builder.getIntegerType(32)},
             getFileDescriptorGetterSymName(builder.getContext()),
             ValueRange{fileName})
      ->getResult(0);
}

} // namespace

FlatSymbolRefAttr sv::getFileDescriptorFragmentRef(MLIRContext *context) {
  return FlatSymbolRefAttr::get(::getFileDescriptorFragmentSymName(context));
}

void sv::emitFileDescriptorRuntime(Operation *fileScopeOp,
                                   ImplicitLocOpBuilder &builder) {
  assert(fileScopeOp && "expected file-level symbol table op");
  assert(fileScopeOp->hasTrait<mlir::OpTrait::SymbolTable>() &&
         "expected fileScopeOp to define a symbol table");
  assert(builder.getInsertionBlock()->getParentOp() == fileScopeOp &&
         "builder must insert directly into fileScopeOp");

  auto getterSymName = ::getFileDescriptorGetterSymName(builder.getContext());
  auto fragmentSymName =
      ::getFileDescriptorFragmentSymName(builder.getContext());
  auto macroSymName = builder.getStringAttr(kFileDescriptorMacroName);

  auto emitGuard = [&](StringRef guard, llvm::function_ref<void(void)> body) {
    sv::IfDefOp::create(
        builder, guard, [] {}, body);
  };

  if (!SymbolTable::lookupSymbolIn(fileScopeOp, getterSymName)) {
    SmallVector<hw::ModulePort> ports;
    ports.push_back({builder.getStringAttr("name"),
                     hw::StringType::get(builder.getContext()),
                     hw::ModulePort::Direction::Input});
    ports.push_back({builder.getStringAttr("fd"), builder.getIntegerType(32),
                     hw::ModulePort::Direction::Output});

    SmallVector<NamedAttribute> explicitReturnAttrs;
    explicitReturnAttrs.push_back(
        {builder.getStringAttr(sv::FuncOp::getExplicitlyReturnedAttrName()),
         builder.getUnitAttr()});
    SmallVector<Attribute> perArgumentAttrs = {
        builder.getDictionaryAttr({}),
        builder.getDictionaryAttr(explicitReturnAttrs)};

    auto func =
        sv::FuncOp::create(builder, getterSymName,
                           hw::ModuleType::get(builder.getContext(), ports),
                           builder.getArrayAttr(perArgumentAttrs), ArrayAttr(),
                           ArrayAttr(), getterSymName);
    func.setPrivate();
  }

  if (!SymbolTable::lookupSymbolIn(fileScopeOp, macroSymName))
    sv::MacroDeclOp::create(builder, macroSymName);

  if (SymbolTable::lookupSymbolIn(fileScopeOp, fragmentSymName))
    return;

  emit::FragmentOp::create(builder, fragmentSymName, [&] {
    emitGuard("SYNTHESIS", [&]() {
      emitGuard(kFileDescriptorMacroName, [&]() {
        sv::VerbatimOp::create(builder, R"(// CIRCT Logging Library
package __circt_lib_logging;
  class FileDescriptor;
    static int global_id [string];
    static function int get(string name);
      if (global_id.exists(name) == 32'h0) begin
        global_id[name] = $fopen(name, "w");
        if (global_id[name] == 32'h0)
          $error("Failed to open file %s", name);
      end
      return global_id[name];
    endfunction
  endclass
endpackage
)");

        sv::MacroDefOp::create(builder, macroSymName, "");
      });
    });
  });
}

Value sv::createProceduralFileDescriptorGetterCall(OpBuilder &builder,
                                                   Location loc,
                                                   Value fileName) {
  return createProceduralFileDescriptorGetterCallImpl(builder, loc, fileName);
}
