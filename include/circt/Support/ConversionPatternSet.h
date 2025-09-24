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
  template <class Op, typename... Args>
  ConversionPatternSet &add(LogicalResult (*implFn)(Op, typename Op::Adaptor,
                                                    ConversionPatternRewriter &,
                                                    Args...),
                            Args &&...args) {

    struct FnPattern final : public OpConversionPattern<Op> {
      typedef LogicalResult Fn(Op, typename Op::Adaptor,
                               ConversionPatternRewriter &, Args...);
      Fn *implFn;
      std::tuple<Args...> args;
      FnPattern(const TypeConverter &converter, MLIRContext *context,
                Fn *implFn, std::tuple<Args...> &&args)
          : OpConversionPattern<Op>(converter, context), implFn(implFn),
            args(args) {}

      LogicalResult
      matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        return std::apply(
            [&](Args &&...args) {
              return implFn(op, adaptor, rewriter, args...);
            },
            args);
      }
    };

    auto pattern = std::make_unique<FnPattern>(
        typeConverter, getContext(), implFn, std::forward_as_tuple(args...));
    add(std::move(pattern));
    return *this;
  }

  /// Add a `matchAndRewrite` function as a conversion pattern to the set.
  template <class Op>
  ConversionPatternSet &add(LogicalResult (*implFn)(Op, typename Op::Adaptor,
                                                    ConversionPatternRewriter &,
                                                    const TypeConverter &)) {

    struct FnPattern final : public OpConversionPattern<Op> {
      using OpConversionPattern<Op>::OpConversionPattern;
      LogicalResult (*implFn)(Op, typename Op::Adaptor,
                              ConversionPatternRewriter &,
                              const TypeConverter &);

      LogicalResult
      matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override {
        return implFn(op, adaptor, rewriter, *this->typeConverter);
      }
    };

    auto pattern = std::make_unique<FnPattern>(typeConverter, getContext());
    pattern->implFn = implFn;
    add(std::move(pattern));
    return *this;
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_CONVERSIONPATTERNSET_H
