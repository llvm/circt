//===- HLSDialect.h - HLS dialect declaration -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HLS dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HLS_HLSDIALECT_H
#define CIRCT_DIALECT_HLS_HLSDIALECT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// Pull in the Dialect definition.
#include "circt/Dialect/HLS/HLSDialect.h.inc"

namespace circt {
namespace hls {
/// Predeclare  the HLS directive implementations
class PipelineDirective;
class PipelineStyleAttr;
class FunctionInstantiateDirective;
} // namespace hls
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HLS/HLSAttributes.h.inc"
#include "circt/Dialect/HLS/HLSEnums.h.inc"

namespace circt {
namespace hls {

/// HLS Directive storage is a wrapper around a TableGen'erated HLS directive
/// implementation TDirectiveImpl, used to implement a dialect attribute.
namespace detail {
template <typename TDirectiveImpl>
struct HLSDirectiveStorage : public mlir::AttributeStorage {
public:
  using KeyTy = TDirectiveImpl;
  HLSDirectiveStorage(KeyTy key) : key(key) {}

  bool operator==(const KeyTy &otherKey) const { return key == otherKey; }
  static unsigned hashKey(const KeyTy &key) { return hash_value(key); }
  static HLSDirectiveStorage *
  construct(mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<HLSDirectiveStorage>())
        HLSDirectiveStorage(key);
  }

  /// Returns the underlying TableGen'erated implementation
  TDirectiveImpl impl() { return key; };

private:
  /// Since we're just wrapping a TableGen'erated attribute, the key is both our
  /// data store and uniqueness identifier.
  KeyTy key;
};

} // namespace detail

/// The HLSDirective class is a wrapper for creating a dialect attribute from a
/// TableGen'erated attribute. By wrapping a TableGen'erated attribute into a
/// dialect attributes we force the parser to delegate parsing and printing to
/// the dialect. Through this, we may instantiate the actual attribute rather
/// than letting the parser default to parsing a dictionary, independently of
/// where in the IR we place an HLS directive.
template <typename TDirectiveImpl>
struct HLSDirective : public mlir::Attribute::AttrBase<
                          HLSDirective<TDirectiveImpl>, mlir::Attribute,
                          detail::HLSDirectiveStorage<TDirectiveImpl>> {
  using Base =
      mlir::Attribute::AttrBase<HLSDirective<TDirectiveImpl>, mlir::Attribute,
                                detail::HLSDirectiveStorage<TDirectiveImpl>>;
  using Base::Base;

  /// Get a new instance of an HLS attribute.
  static HLSDirective get(TDirectiveImpl impl) {
    return Base::get(impl.getContext(), impl);
  }
  /// Returns the underlying TableGen'erated implementation of this directive.
  TDirectiveImpl impl() { return this->getImpl()->impl(); }

  /// Returns the keyword for this directive.
  static constexpr llvm::StringLiteral getKeyword();
};

/// Keywords for the available directives.
template <>
constexpr llvm::StringLiteral HLSDirective<PipelineDirective>::getKeyword() {
  return "pipeline";
}
template <>
constexpr llvm::StringLiteral
HLSDirective<FunctionInstantiateDirective>::getKeyword() {
  return "function_instantiate";
}

} // namespace hls
} // namespace circt

#endif // CIRCT_DIALECT_HLS_HLSDIALECT_H
