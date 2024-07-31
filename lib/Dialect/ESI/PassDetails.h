//===- PassDetails.h - ESI pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different ESI passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_ESI_PASSDETAILS_H
#define DIALECT_ESI_PASSDETAILS_H

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace esi {
namespace detail {

/// Generic pattern for removing an op during pattern conversion.
template <typename OpTy>
struct RemoveOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

StringAttr getTypeID(Type t);
uint64_t getWidth(Type t);

/// Assist the lowering steps for conversions which need to create auxiliary IR.
class ESIHWBuilder : public circt::ImplicitLocOpBuilder {
public:
  ESIHWBuilder(Operation *top);

  ArrayAttr getStageParameterList(Attribute value);

  hw::HWModuleExternOp declareStage(Operation *symTable, PipelineStageOp);
  hw::HWModuleExternOp declareCosimEndpointToHostModule(Operation *symTable);
  hw::HWModuleExternOp declareCosimEndpointFromHostModule(Operation *symTable);
  sv::InterfaceOp getOrConstructInterface(ChannelType);
  sv::InterfaceOp constructInterface(ChannelType);

  // A bunch of constants for use in various places below.
  const StringAttr a, aValid, aReady, x, xValid, xReady;
  const StringAttr dataOutValid, dataOutReady, dataOut, dataInValid,
      dataInReady, dataIn;
  const StringAttr clk, rst;
  const StringAttr width;

  // Various identifier strings. Keep them all here in case we rename them.
  static constexpr char dataStr[] = "data", validStr[] = "valid",
                        readyStr[] = "ready", sourceStr[] = "source",
                        sinkStr[] = "sink";

private:
  /// Construct a type-appropriate name for the interface, making sure it's not
  /// taken in the symbol table.
  StringAttr constructInterfaceName(ChannelType);

  Type getClockType();

  std::optional<hw::HWModuleExternOp> declaredCosimEndpointToHostModule;
  std::optional<hw::HWModuleExternOp> declaredCosimEndpointFromHostModule;
  llvm::DenseMap<Type, hw::HWModuleExternOp> declaredStage;
  llvm::DenseMap<Type, sv::InterfaceOp> portTypeLookup;
};
} // namespace detail
} // namespace esi
} // namespace circt

#endif // DIALECT_ESI_PASSDETAILS_H
