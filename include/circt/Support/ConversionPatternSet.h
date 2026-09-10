//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to collect sets of conversion patterns.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_CONVERSIONPATTERNSET_H
#define CIRCT_SUPPORT_CONVERSIONPATTERNSET_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {

/// Extension of `RewritePatternSet` that allows adding `matchAndRewrite`
/// functions with op adaptors and `ConversionPatternRewriter` as patterns.
class ConversionPatternSet : public RewritePatternSet {
public:
  const TypeConverter &typeConverter;

  ConversionPatternSet(MLIRContext *context, const TypeConverter &typeConverter)
      : RewritePatternSet(context), typeConverter(typeConverter) {}

  // Expose the `add` implementations of `RewritePatternSet`.
  using RewritePatternSet::add;

  /// Add a `matchAndRewrite` function as a conversion pattern to the set.
  /// Extra arguments beyond the op, adaptor, and rewriter are deduced from the
  /// function pointer signature, stored in the pattern, and forwarded on each
  /// invocation.
  template <class Op, typename... ExtraArgs>
  ConversionPatternSet &add(LogicalResult (*implFn)(Op, typename Op::Adaptor,
                                                    ConversionPatternRewriter &,
                                                    ExtraArgs...),
                            llvm::type_identity_t<ExtraArgs>... args) {

    struct FnPattern final : public OpConversionPattern<Op> {
      LogicalResult (*implFn)(Op, typename Op::Adaptor,
                              ConversionPatternRewriter &, ExtraArgs...);
      std::tuple<ExtraArgs...> extraArgs;

      FnPattern(const TypeConverter &tc, MLIRContext *ctx,
                LogicalResult (*implFn)(Op, typename Op::Adaptor,
                                        ConversionPatternRewriter &,
                                        ExtraArgs...),
                ExtraArgs... args)
          : OpConversionPattern<Op>(tc, ctx), implFn(implFn),
            extraArgs(args...) {}

      LogicalResult
      matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        return std::apply(
            implFn, std::tuple_cat(std::tie(op, adaptor, rewriter), extraArgs));
      }
    };

    add(std::make_unique<FnPattern>(typeConverter, getContext(), implFn,
                                    args...));
    return *this;
  }

  /// Add a `matchAndRewrite` function as a conversion pattern to the set. The
  /// pattern's type converter is automatically forwarded to the function,
  /// followed by any extra arguments.
  template <class Op, typename... ExtraArgs>
  ConversionPatternSet &add(LogicalResult (*implFn)(Op, typename Op::Adaptor,
                                                    ConversionPatternRewriter &,
                                                    const TypeConverter &,
                                                    ExtraArgs...),
                            llvm::type_identity_t<ExtraArgs>... args) {

    struct FnPattern final : public OpConversionPattern<Op> {
      LogicalResult (*implFn)(Op, typename Op::Adaptor,
                              ConversionPatternRewriter &,
                              const TypeConverter &, ExtraArgs...);
      std::tuple<ExtraArgs...> extraArgs;

      FnPattern(const TypeConverter &tc, MLIRContext *ctx,
                LogicalResult (*implFn)(Op, typename Op::Adaptor,
                                        ConversionPatternRewriter &,
                                        const TypeConverter &, ExtraArgs...),
                ExtraArgs... args)
          : OpConversionPattern<Op>(tc, ctx), implFn(implFn),
            extraArgs(args...) {}

      LogicalResult
      matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        return std::apply(
            implFn, std::tuple_cat(std::tie(op, adaptor, rewriter),
                                   std::tie(*this->typeConverter), extraArgs));
      }
    };

    add(std::make_unique<FnPattern>(typeConverter, getContext(), implFn,
                                    args...));
    return *this;
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_CONVERSIONPATTERNSET_H
