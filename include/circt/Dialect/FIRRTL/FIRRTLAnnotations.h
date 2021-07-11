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
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace circt {
namespace firrtl {

class Annotation;
class AnnotationSetIterator;
class FModuleOp;

/// Return the name of the attribute used for annotations on FIRRTL ops.
inline StringRef getAnnotationAttrName() { return "annotations"; }

/// Return the name of the dialect-prefixed attribute used for annotations.
inline StringRef getDialectAnnotationAttrName() { return "firrtl.annotations"; }

/// This class provides a read-only projection over the MLIR attributes that
/// represent a set of annotations.  It is intended to make this work less
/// stringly typed and fiddly for clients.
///
class AnnotationSet {
public:
  /// Form an empty annotation set.
  explicit AnnotationSet(MLIRContext *context)
      : annotations(ArrayAttr::get(context, {})) {}

  /// Form an annotation set from an array of annotation attributes.
  explicit AnnotationSet(ArrayRef<Attribute> annotations, MLIRContext *context);

  /// Form an annotation set from an array of annotations.
  explicit AnnotationSet(ArrayRef<Annotation> annotations,
                         MLIRContext *context);

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

  /// Get an annotation set for the specified module port, as well as other
  /// argument attributes.
  static AnnotationSet
  forPort(Operation *module, size_t portNo,
          SmallVectorImpl<NamedAttribute> &otherAttributes);

  /// Get an annotation set for the specified value.
  static AnnotationSet get(Value v);

  /// Return all the raw annotations that exist.
  ArrayRef<Attribute> getArray() const { return annotations.getValue(); }

  /// Return this annotation set as an ArrayAttr.
  ArrayAttr getArrayAttr() const { return annotations; }

  /// Return this annotation set as an argument attribute dictionary for a port.
  DictionaryAttr
  getArgumentAttrDict(ArrayRef<NamedAttribute> otherPortAttrs = {}) const;

  /// Store the annotations in this set in an operation's `annotations`
  /// attribute, overwriting any existing annotations. Removes the `annotations`
  /// attribute if the set is empty. Returns true if the operation was modified,
  /// false otherwise.
  bool applyToOperation(Operation *op) const;

  /// Store the annotations in this set in a `NamedAttrList` as an array
  /// attribute with the name `annotations`. Overwrites existing annotations.
  /// Removes the `annotations` attribute if the set is empty. Returns true if
  /// the list was modified, false otherwise.
  ///
  /// This function is useful if you are in the process of modifying an
  /// operation's attributes as a `NamedAttrList`, or you are preparing the
  /// attributes of a operation yet to be created. In that case
  /// `applyToAttrList` allows you to set the `annotations` attribute in that
  /// list to the contents of this set.
  bool applyToAttrList(NamedAttrList &attrs) const;

  /// Store the annotations in this set in a `NamedAttrList` as an array
  /// attribute with the name `firrtl.annotations`. Overwrites existing
  /// annotations. Removes the `firrtl.annotations` attribute if the set is
  /// empty. Returns true if the list was modified, false otherwise.
  ///
  /// This function is useful if you are in the process of modifying a port's
  /// attributes as a `NamedAttrList`, or you are preparing the attributes of a
  /// port yet to be created as part of an operation. In that case
  /// `applyToPortAttrList` allows you to set the `firrtl.annotations` attribute
  /// in that list to the contents of this set.
  bool applyToPortAttrList(NamedAttrList &attrs) const;

  /// Insert this annotation set into a `DictionaryAttr` under the `annotations`
  /// key. Overwrites any existing attribute stored under `annotations`. Removes
  /// the `annotations` attribute in the dictionary if the set is empty. Returns
  /// the updated dictionary.
  ///
  /// This function is useful if you hold an operation's attributes dictionary
  /// and want to set the `annotations` key in the dictionary to the contents of
  /// this set.
  DictionaryAttr applyToDictionaryAttr(DictionaryAttr attrs) const;
  DictionaryAttr applyToDictionaryAttr(ArrayRef<NamedAttribute> attrs) const;

  /// Insert this annotation set into a `DictionaryAttr` under the
  /// `firrtl.annotations` key. Overwrites any existing attribute stored under
  /// `firrtl.annotations`. Removes the `firrtl.annotations` attribute in the
  /// dictionary if the set is empty. Returns the updated dictionary.
  ///
  /// This function is useful if you hold a port's attributes dictionary and
  /// want to set the `firrtl.annotations` key in the dictionary to the contents
  /// of this set.
  DictionaryAttr applyToPortDictionaryAttr(DictionaryAttr attrs) const;
  DictionaryAttr
  applyToPortDictionaryAttr(ArrayRef<NamedAttribute> attrs) const;

  /// Return true if we have an annotation with the specified class name.
  bool hasAnnotation(StringRef className) const {
    return !annotations.empty() && hasAnnotationImpl(className);
  }
  bool hasAnnotation(StringAttr className) const {
    return !annotations.empty() && hasAnnotationImpl(className);
  }

  /// If this annotation set has an annotation with the specified class name,
  /// return it.  Otherwise return a null DictionaryAttr.
  DictionaryAttr getAnnotation(StringRef className) const {
    if (annotations.empty())
      return {};
    return getAnnotationImpl(className);
  }
  DictionaryAttr getAnnotation(StringAttr className) const {
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

  /// Add more annotations to this annotation set.
  void addAnnotations(ArrayRef<Annotation> annotations);
  void addAnnotations(ArrayRef<Attribute> annotations);
  void addAnnotations(ArrayAttr annotations);

  /// Remove an annotation from this annotation set. Returns true if any were
  /// removed, false otherwise.
  bool removeAnnotation(Annotation anno);
  bool removeAnnotation(Attribute anno);

  /// Remove all annotations from this annotation set for which `predicate`
  /// returns true. The predicate is guaranteed to be called on every
  /// annotation, such that this method can be used to partition the set by
  /// extracting and removing annotations at the same time. Returns true if any
  /// annotations were removed, false otherwise.
  bool removeAnnotations(llvm::function_ref<bool(Annotation)> predicate);

  /// Remove all annotations with one of the given classes from this annotation
  /// set.
  template <typename... Args>
  bool removeAnnotationsWithClass(Args... names);

  /// Remove all annotations from an operation for which `predicate` returns
  /// true. The predicate is guaranteed to be called on every annotation, such
  /// that this method can be used to partition the set by extracting and
  /// removing annotations at the same time. Returns true if any annotations
  /// were removed, false otherwise.
  static bool removeAnnotations(Operation *op,
                                llvm::function_ref<bool(Annotation)> predicate);
  static bool removeAnnotations(Operation *op, StringRef className);

  /// Remove all port annotations from a module for which `predicate` returns
  /// true. The predicate is guaranteed to be called on every annotation, such
  /// that this method can be used to partition a module's port annotations by
  /// extracting and removing annotations at the same time. Returns true if any
  /// annotations were removed, false otherwise.
  static bool removePortAnnotations(
      FModuleOp module,
      llvm::function_ref<bool(unsigned, Annotation)> predicate);

private:
  bool hasAnnotationImpl(StringAttr className) const;
  bool hasAnnotationImpl(StringRef className) const;
  DictionaryAttr getAnnotationImpl(StringAttr className) const;
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
  StringAttr getClassAttr() const;
  StringRef getClass() const;

  /// Return true if this annotation matches any of the specified class names.
  template <typename... Args>
  bool isClass(Args... names) const {
    return ClassIsa{getClassAttr()}(names...);
  }

  /// Return a member of the annotation.
  template <typename AttrClass = Attribute>
  AttrClass getMember(StringAttr name) const {
    // TODO: Once https://reviews.llvm.org/D103822 lands, the `const_cast` can
    // go away.
    return const_cast<DictionaryAttr &>(attrDict).getAs<AttrClass>(name);
  }
  template <typename AttrClass = Attribute>
  AttrClass getMember(StringRef name) const {
    // TODO: Once https://reviews.llvm.org/D103822 lands, the `const_cast` can
    // go away.
    return const_cast<DictionaryAttr &>(attrDict).getAs<AttrClass>(name);
  }

  bool operator==(const Annotation &other) const {
    return attrDict == other.attrDict;
  }
  bool operator!=(const Annotation &other) const { return !(*this == other); }

private:
  DictionaryAttr attrDict;

  /// Helper struct to perform variadic class equality check.
  struct ClassIsa {
    StringAttr cls;

    bool operator()() const { return false; }
    template <typename T, typename... Rest>
    bool operator()(T name, Rest... rest) const {
      return compare(name) || (*this)(rest...);
    }

  private:
    bool compare(StringAttr name) const { return cls == name; }
    bool compare(StringRef name) const { return cls && cls.getValue() == name; }
  };
};

// Out-of-line impl since we need `Annotation` to be fully defined.
template <typename... Args>
bool AnnotationSet::removeAnnotationsWithClass(Args... names) {
  return removeAnnotations(
      [&](Annotation anno) { return anno.isClass(names...); });
}

// Iteration over the annotation set.
class AnnotationSetIterator
    : public llvm::indexed_accessor_iterator<AnnotationSetIterator,
                                             AnnotationSet, Annotation> {
public:
  // Index into this iterator.
  Annotation operator*() const;

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

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

namespace llvm {

/// Make `Annotation` behave like a `Attribute` in terms of pointer-likeness.
template <>
struct PointerLikeTypeTraits<circt::firrtl::Annotation>
    : PointerLikeTypeTraits<mlir::Attribute> {
  using Annotation = circt::firrtl::Annotation;
  static inline void *getAsVoidPointer(Annotation v) {
    return const_cast<void *>(v.getDict().getAsOpaquePointer());
  }
  static inline Annotation getFromVoidPointer(void *p) {
    return Annotation(mlir::DictionaryAttr::getFromOpaquePointer(p));
  }
};

/// Make `Annotation` hash just like `Attribute`.
template <>
struct DenseMapInfo<circt::firrtl::Annotation> {
  using Annotation = circt::firrtl::Annotation;
  static Annotation getEmptyKey() {
    return Annotation(
        mlir::DictionaryAttr(static_cast<mlir::Attribute::ImplType *>(
            DenseMapInfo<void *>::getEmptyKey())));
  }
  static Annotation getTombstoneKey() {
    return Annotation(
        mlir::DictionaryAttr(static_cast<mlir::Attribute::ImplType *>(
            llvm::DenseMapInfo<void *>::getTombstoneKey())));
  }
  static unsigned getHashValue(Annotation val) {
    return mlir::hash_value(val.getDict());
  }
  static bool isEqual(Annotation LHS, Annotation RHS) { return LHS == RHS; }
};

} // namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATIONS_H
