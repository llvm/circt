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
#include "llvm/ADT/STLExtras.h"

namespace circt {
namespace firrtl {

class Annotation;
class AnnotationSetIterator;

/// This class provides a read-only projection over the MLIR attributes that
/// represent a set of annotations.  It is intended to make this work less
/// stringly typed and fiddly for clients.
///
class AnnotationSet {
public:
  /// Form an annotation set with a non-null ArrayAttr.
  explicit AnnotationSet(MLIRContext *context)
      : annotations(ArrayAttr::get(context, {})) {}

  /// Form an annotation set with a non-null ArrayAttr.
  explicit AnnotationSet(ArrayAttr annotations) : annotations(annotations) {
    assert(annotations && "Cannot use null attribute set");
  }

  /// Form an annotation set with a possibly-null ArrayAttr.
  explicit AnnotationSet(ArrayAttr annotations, MLIRContext *context);

  /// Get an annotation set for the specified operation.
  explicit AnnotationSet(Operation *op);

  /// Get an annotation set for the specified module port.
  static AnnotationSet forPort(Operation *module, size_t portNo);

  /// Return all the raw annotations that exist.
  ArrayRef<Attribute> getArray() const { return annotations.getValue(); }

  /// Return this annotation set as an ArrayAttr.
  ArrayAttr getArrayAttr() const { return annotations; }

  /// Return this annotation set as an argument attribute dictionary for a port.
  DictionaryAttr
  getArgumentAttrDict(ArrayRef<NamedAttribute> otherPortAttrs = {}) const;

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

  using iterator = AnnotationSetIterator;
  iterator begin() const;
  iterator end() const;

  /// Return the MLIRContext corresponding to this AnnotationSet.
  MLIRContext *getContext() const { return annotations.getContext(); }

  // Support for widely used annotations.

  /// firrtl.transforms.DontTouchAnnotation
  bool hasDontTouch() const;

  bool operator==(const AnnotationSet &other) const {
    return annotations == other.annotations;
  }
  bool operator!=(const AnnotationSet &other) const {
    return !(*this == other);
  }

  bool empty() const { return annotations.empty(); }

  size_t size() const { return annotations.size(); }

  /// Add more annotations to this AttributeSet.
  void addAnnotations(ArrayAttr annotations);

private:
  bool hasAnnotationImpl(StringRef className) const;
  DictionaryAttr getAnnotationImpl(StringRef className) const;

  ArrayAttr annotations;
};

/// This class provides a read-only projection of an annotation.
class Annotation {
public:
  Annotation(DictionaryAttr attrDict) : attrDict(attrDict) {
    assert(attrDict && "null dictionaries not allowed");
  }

  DictionaryAttr getDict() const { return attrDict; }

  /// Return the 'class' that this annotation is representing.
  StringRef getClass() const;

  /// Return true if this annotation matches the specified class name.
  bool isClass(StringRef name) const { return getClass() == name; }

  template <typename AttrClass = Attribute>
  AttrClass getMember(StringRef name) {
    return attrDict.getAs<AttrClass>(name);
  }

private:
  DictionaryAttr attrDict;
};

// Iteration over the annotation set.
class AnnotationSetIterator
    : public llvm::indexed_accessor_iterator<AnnotationSetIterator,
                                             AnnotationSet, Annotation> {
public:
  // Index into this iterator.
  Annotation operator*() const {
    return Annotation(
        this->getBase().getArray()[this->getIndex()].cast<DictionaryAttr>());
  }

private:
  AnnotationSetIterator(AnnotationSet owner, ptrdiff_t curIndex)
      : llvm::indexed_accessor_iterator<AnnotationSetIterator, AnnotationSet,
                                        Annotation>(owner, curIndex) {}
  friend llvm::indexed_accessor_iterator<AnnotationSetIterator, AnnotationSet,
                                         Annotation>;
  friend class AnnotationSet;
};

inline auto AnnotationSet::begin() const -> iterator {
  return AnnotationSetIterator(*this, 0);
}
inline auto AnnotationSet::end() const -> iterator {
  return iterator(*this, annotations.size());
}

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATIONS_H
