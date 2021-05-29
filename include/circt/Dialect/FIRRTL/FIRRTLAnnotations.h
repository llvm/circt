//===- FIRRTLAnnotations.h - Code for working with Annotations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helpers for working with FIRRTL annotations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_ANNOTATIONS_H
#define CIRCT_DIALECT_FIRRTL_ANNOTATIONS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace circt {
namespace firrtl {

/// This class provides a read-only projection over the MLIR attributes that
/// represent a set of annotations.  It is intended to make this work less
/// stringly typed and fiddly for clients.
///
class AnnotationSet {
public:
  /// Get an annotation set for the specified operation.
  explicit AnnotationSet(Operation *op);

  explicit AnnotationSet(ArrayRef<Attribute> annotations)
      : annotations(annotations) {}

  /// Return all the raw annotations that exist.
  ArrayRef<Attribute> getRaw() const { return annotations; }

  /// Return true if we have an annotation with the specified class name.
  bool hasAnnotation(StringRef className) const {
    return !annotations.empty() && hasAnnotationImpl(className);
  }

  /// If this annotation set has an annotation with the specified class name,
  /// return it.  Otherwise return a null DictionaryAttr.
  DictionaryAttr getAnnotation(StringRef className) const {
    if (annotations.empty())
      return {};
    return getAnnotationImpl(className);
  }

  // Support for widely used annotations.

  /// firrtl.transforms.DontTouchAnnotation
  bool hasDontTouch() const;

private:
  bool hasAnnotationImpl(StringRef className) const;
  DictionaryAttr getAnnotationImpl(StringRef className) const;

  ArrayRef<Attribute> annotations;
};

/// Return the
AnnotationSet getAnnotations(Operation *op);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATIONS_H
