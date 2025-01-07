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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace circt {
namespace hw {
struct InnerSymbolNamespace;
} // namespace hw
namespace firrtl {

class AnnotationSetIterator;
class FModuleOp;
class FModuleLike;
class MemOp;
class InstanceOp;
class FIRRTLType;

/// Return the name of the attribute used for annotations on FIRRTL ops.
inline StringRef getAnnotationAttrName() { return "annotations"; }

/// Return the name of the attribute used for port annotations on FIRRTL ops.
inline StringRef getPortAnnotationAttrName() { return "portAnnotations"; }

inline ArrayAttr getAnnotationsIfPresent(Operation *op) {
  return op->getAttrOfType<ArrayAttr>(getAnnotationAttrName());
}

/// This class provides a read-only projection of an annotation.
class Annotation {
public:
  Annotation() = default;

  explicit Annotation(Attribute attr) : attr(attr) {
    assert(attr && "null attributes not allowed");
  }

  // Make a new annotation which is a clone of anno with a new fieldID.
  Annotation(Annotation anno, uint64_t fieldID) : attr(anno.attr) {
    auto oldFieldID = anno.getMember<IntegerAttr>("circt.fieldID");
    if (oldFieldID && !fieldID) {
      removeMember("circt.fieldID");
      return;
    }
    if (fieldID)
      setMember("circt.fieldID",
                IntegerAttr::get(IntegerType::get(anno.attr.getContext(), 32),
                                 APInt(32, fieldID)));
  }

  /// Get the data dictionary of this attribute.
  DictionaryAttr getDict() const;

  /// Set the data dictionary of this attribute.
  void setDict(DictionaryAttr dict);

  /// Get the field id this attribute targets.
  unsigned getFieldID() const;

  /// Get the underlying attribute.
  Attribute getAttr() const { return attr; }

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
    return getDict().getAs<AttrClass>(name);
  }
  template <typename AttrClass = Attribute>
  AttrClass getMember(StringRef name) const {
    return getDict().getAs<AttrClass>(name);
  }

  /// Add or set a member of the annotation to a value.
  void setMember(StringAttr name, Attribute value);
  void setMember(StringRef name, Attribute value);

  /// Remove a member of the annotation.
  void removeMember(StringAttr name);
  void removeMember(StringRef name);

  using iterator = llvm::ArrayRef<NamedAttribute>::iterator;
  iterator begin() const { return getDict().begin(); }
  iterator end() const { return getDict().end(); }

  bool operator==(const Annotation &other) const { return attr == other.attr; }
  bool operator!=(const Annotation &other) const { return !(*this == other); }
  explicit operator bool() const { return bool(attr); }
  bool operator!() const { return attr == nullptr; }

  void dump();

private:
  Attribute attr;

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

/// This class provides a read-only projection over the MLIR attributes that
/// represent a set of annotations.  It is intended to make this work less
/// stringly typed and fiddly for clients.
///
class AnnotationSet {
public:
  using ElementType = Annotation;

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

  /// Get an annotation set for the specified port.
  static AnnotationSet forPort(FModuleLike op, size_t portNo);
  static AnnotationSet forPort(MemOp op, size_t portNo);

  /// Get an annotation set for the specified value.
  static AnnotationSet get(Value v);

  /// Return all the raw annotations that exist.
  ArrayRef<Attribute> getArray() const { return annotations.getValue(); }

  /// Return this annotation set as an ArrayAttr.
  ArrayAttr getArrayAttr() const {
    assert(annotations && "Cannot use null attribute set");
    return annotations;
  }

  /// Store the annotations in this set in an operation's `annotations`
  /// attribute, overwriting any existing annotations. Removes the `annotations`
  /// attribute if the set is empty. Returns true if the operation was modified,
  /// false otherwise.
  bool applyToOperation(Operation *op) const;

  /// Store the annotations in this set in an operation's `portAnnotations`
  /// attribute, overwriting any existing annotations for this port. Returns
  /// true if the operation was modified, false otherwise.
  bool applyToPort(FModuleLike op, size_t portNo) const;
  bool applyToPort(MemOp op, size_t portNo) const;

  /// Return true if we have an annotation with the specified class name.
  bool hasAnnotation(StringRef className) const {
    return !annotations.empty() && hasAnnotationImpl(className);
  }
  bool hasAnnotation(StringAttr className) const {
    return !annotations.empty() && hasAnnotationImpl(className);
  }

  /// Return true if we have an annotation with the specified class name.
  template <typename... Args>
  static bool hasAnnotation(Operation *op, Args... args) {
    auto annosArray = getAnnotationsIfPresent(op);
    if (!annosArray)
      return false;
    AnnotationSet annotations(annosArray);
    return !annotations.empty() && (... || annotations.hasAnnotationImpl(args));
  }

  /// If this annotation set has an annotation with the specified class name,
  /// return it.  Otherwise return a null DictionaryAttr.
  Annotation getAnnotation(StringRef className) const {
    if (annotations.empty())
      return {};
    return getAnnotationImpl(className);
  }
  Annotation getAnnotation(StringAttr className) const {
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
  bool setDontTouch(bool dontTouch);
  bool addDontTouch();
  bool removeDontTouch();
  static bool hasDontTouch(Operation *op);
  static bool setDontTouch(Operation *op, bool dontTouch);
  static bool addDontTouch(Operation *op);
  static bool removeDontTouch(Operation *op);

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
  bool removeAnnotation(StringRef className);

  /// Remove all annotations from this annotation set for which `predicate`
  /// returns true. The predicate is guaranteed to be called on every
  /// annotation, such that this method can be used to partition the set by
  /// extracting and removing annotations at the same time. Returns true if any
  /// annotations were removed, false otherwise.
  bool removeAnnotations(llvm::function_ref<bool(Annotation)> predicate);

  /// Remove all annotations with one of the given classes from this annotation
  /// set.
  template <typename... Args>
  bool removeAnnotationsWithClass(Args... names) {
    return removeAnnotations(
        [&](Annotation anno) { return anno.isClass(names...); });
  }

  /// Remove all annotations from an operation for which `predicate` returns
  /// true. The predicate is guaranteed to be called on every annotation, such
  /// that this method can be used to partition the set by extracting and
  /// removing annotations at the same time. Returns true if any annotations
  /// were removed, false otherwise.
  static bool removeAnnotations(Operation *op,
                                llvm::function_ref<bool(Annotation)> predicate);
  static bool removeAnnotations(Operation *op, StringRef className);

  /// Remove all port annotations from a module or extmodule for which
  /// `predicate` returns true. The predicate is guaranteed to be called on
  /// every annotation, such that this method can be used to partition a
  /// module's port annotations by extracting and removing annotations at the
  /// same time. Returns true if any annotations were removed, false otherwise.
  static bool removePortAnnotations(
      Operation *module,
      llvm::function_ref<bool(unsigned, Annotation)> predicate);

private:
  bool hasAnnotationImpl(StringAttr className) const;
  bool hasAnnotationImpl(StringRef className) const;
  Annotation getAnnotationImpl(StringAttr className) const;
  Annotation getAnnotationImpl(StringRef className) const;

  ArrayAttr annotations;
};

// Iteration over the annotation set.
class AnnotationSetIterator
    : public llvm::indexed_accessor_iterator<AnnotationSetIterator,
                                             AnnotationSet, Annotation,
                                             Annotation, Annotation> {
public:
  // Index into this iterator.
  Annotation operator*() const;

private:
  AnnotationSetIterator(AnnotationSet owner, ptrdiff_t curIndex)
      : llvm::indexed_accessor_iterator<AnnotationSetIterator, AnnotationSet,
                                        Annotation, Annotation, Annotation>(
            owner, curIndex) {}
  friend llvm::indexed_accessor_iterator<AnnotationSetIterator, AnnotationSet,
                                         Annotation, Annotation, Annotation>;
  friend class AnnotationSet;
};

inline auto AnnotationSet::begin() const -> iterator {
  return AnnotationSetIterator(*this, 0);
}
inline auto AnnotationSet::end() const -> iterator {
  return iterator(*this, annotations.size());
}

//===----------------------------------------------------------------------===//
// AnnoTarget
//===----------------------------------------------------------------------===//

namespace detail {
struct AnnoTargetImpl {
  /* implicit */ AnnoTargetImpl(Operation *op) : op(op), portNo(~0UL) {}

  AnnoTargetImpl(Operation *op, unsigned portNo) : op(op), portNo(portNo) {}

  operator bool() const { return getOp(); }
  bool operator==(const AnnoTargetImpl &other) const {
    return op == other.op && portNo == other.portNo;
  }
  bool operator!=(const AnnoTargetImpl &other) const {
    return !(*this == other);
  }

  bool isPort() const { return op && portNo != ~0UL; }
  bool isOp() const { return op && portNo == ~0UL; }

  Operation *getOp() const { return op; }
  void setOp(Operation *op) { this->op = op; }

  unsigned getPortNo() const { return portNo; }
  void setPortNo(unsigned portNo) { this->portNo = portNo; }

protected:
  Operation *op;
  size_t portNo;
};
} // namespace detail

/// An annotation target is used to keep track of something that is targeted by
/// an Annotation.
struct AnnoTarget {
  AnnoTarget(detail::AnnoTargetImpl impl = nullptr) : impl(impl){};

  operator bool() const { return impl; }
  bool operator==(const AnnoTarget &other) const { return impl == other.impl; }
  bool operator!=(const AnnoTarget &other) const { return !(*this == other); }

  Operation *getOp() const { return getImpl().getOp(); }
  void setOp(Operation *op) { getImpl().setOp(op); }

  /// Get the annotations associated with the target.
  AnnotationSet getAnnotations() const;

  /// Set the annotations associated with the target.
  void setAnnotations(AnnotationSet annotations) const;

  /// Get the parent module of the target.
  FModuleLike getModule() const;

  /// Get a reference to this target suitable for use in an NLA.
  Attribute getNLAReference(hw::InnerSymbolNamespace &moduleNamespace) const;

  /// Get the type of the target.
  FIRRTLType getType() const;

  detail::AnnoTargetImpl getImpl() const { return impl; }

protected:
  detail::AnnoTargetImpl impl;
};

/// This represents an annotation targeting a specific operation.
struct OpAnnoTarget : public AnnoTarget {
  using AnnoTarget::AnnoTarget;

  OpAnnoTarget(Operation *op) : AnnoTarget(op) {}

  AnnotationSet getAnnotations() const;
  void setAnnotations(AnnotationSet annotations) const;
  Attribute getNLAReference(hw::InnerSymbolNamespace &moduleNamespace) const;
  FIRRTLType getType() const;

  static bool classof(const AnnoTarget &annoTarget) {
    return annoTarget.getImpl().isOp();
  }
};

/// This represents an annotation targeting a specific port of a module, memory,
/// or instance.
struct PortAnnoTarget : public AnnoTarget {
  using AnnoTarget::AnnoTarget;

  PortAnnoTarget(FModuleLike op, unsigned portNo);
  PortAnnoTarget(MemOp op, unsigned portNo);

  unsigned getPortNo() const { return getImpl().getPortNo(); }
  void setPortNo(unsigned portNo) { getImpl().setPortNo(portNo); }

  AnnotationSet getAnnotations() const;
  void setAnnotations(AnnotationSet annotations) const;
  Attribute getNLAReference(hw::InnerSymbolNamespace &moduleNamespace) const;
  FIRRTLType getType() const;

  static bool classof(const AnnoTarget &annoTarget) {
    return annoTarget.getImpl().isPort();
  }
};

} // namespace firrtl
} // namespace circt

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

namespace llvm {

/// Add support for llvm style casts to AnnoTarget.
template <typename To, typename From>
struct CastInfo<
    To, From,
    std::enable_if_t<std::is_base_of_v<::circt::firrtl::AnnoTarget, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(From target) {
    // Allow constant upcasting.  This also gets around the fact that AnnoTarget
    // does not implement classof.
    if constexpr (std::is_base_of_v<To, From>)
      return true;
    else
      return To::classof(target);
  }
  static inline To doCast(From target) { return To(target.getImpl()); }
};

/// Make `Annotation` behave like a `Attribute` in terms of pointer-likeness.
template <>
struct PointerLikeTypeTraits<circt::firrtl::Annotation>
    : PointerLikeTypeTraits<mlir::Attribute> {
  using Annotation = circt::firrtl::Annotation;
  static inline void *getAsVoidPointer(Annotation v) {
    return const_cast<void *>(v.getAttr().getAsOpaquePointer());
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
    return mlir::hash_value(val.getAttr());
  }
  static bool isEqual(Annotation lhs, Annotation rhs) { return lhs == rhs; }
};

/// Make `AnnoTarget` hash.
template <>
struct DenseMapInfo<circt::firrtl::AnnoTarget> {
  using AnnoTarget = circt::firrtl::AnnoTarget;
  using AnnoTargetImpl = circt::firrtl::detail::AnnoTargetImpl;
  static AnnoTarget getEmptyKey() {
    auto *o = DenseMapInfo<mlir::Operation *>::getEmptyKey();
    auto i = DenseMapInfo<unsigned>::getEmptyKey();
    return AnnoTarget(AnnoTargetImpl(o, i));
  }
  static AnnoTarget getTombstoneKey() {
    auto *o = DenseMapInfo<mlir::Operation *>::getTombstoneKey();
    auto i = DenseMapInfo<unsigned>::getTombstoneKey();
    return AnnoTarget(AnnoTargetImpl(o, i));
  }
  static unsigned getHashValue(AnnoTarget val) {
    auto impl = val.getImpl();
    return hash_combine(impl.getOp(), impl.getPortNo());
  }
  static bool isEqual(AnnoTarget lhs, AnnoTarget rhs) { return lhs == rhs; }
};

} // namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATIONS_H
