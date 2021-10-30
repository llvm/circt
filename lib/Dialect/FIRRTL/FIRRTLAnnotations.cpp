//===- FIRRTLAnnotations.cpp - Code for working with Annotations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helpers for working with FIRRTL annotations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Operation.h"

using mlir::function_like_impl::getArgAttrDict;
using mlir::function_like_impl::setAllArgAttrDicts;

using namespace circt;
using namespace firrtl;

static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annots = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annots;
  return ArrayAttr::get(op->getContext(), {});
}

static ArrayAttr getAnnotationsFrom(ArrayRef<Annotation> annotations,
                                    MLIRContext *context) {
  if (annotations.empty())
    return ArrayAttr::get(context, {});
  SmallVector<Attribute> attrs;
  attrs.reserve(annotations.size());
  for (auto anno : annotations)
    attrs.push_back(anno.getDict());
  return ArrayAttr::get(context, attrs);
}

/// Form an annotation set from an array of annotation attributes.
AnnotationSet::AnnotationSet(ArrayRef<Attribute> annotations,
                             MLIRContext *context)
    : annotations(ArrayAttr::get(context, annotations)) {}

/// Form an annotation set from an array of annotations.
AnnotationSet::AnnotationSet(ArrayRef<Annotation> annotations,
                             MLIRContext *context)
    : annotations(getAnnotationsFrom(annotations, context)) {}

/// Form an annotation set with a possibly-null ArrayAttr.
AnnotationSet::AnnotationSet(ArrayAttr annotations, MLIRContext *context)
    : AnnotationSet(annotations ? annotations : ArrayAttr::get(context, {})) {}

/// Get an annotation set for the specified operation.
AnnotationSet::AnnotationSet(Operation *op)
    : AnnotationSet(getAnnotationsFrom(op)) {}

/// Get an annotation set for the specified module port.
AnnotationSet AnnotationSet::forPort(FModuleLike module, size_t portNo) {
  auto ports = module->getAttr("portAnnotations").dyn_cast_or_null<ArrayAttr>();
  if (ports && !ports.empty())
    return AnnotationSet(ports[portNo].cast<ArrayAttr>());
  return AnnotationSet(ArrayAttr::get(module->getContext(), {}));
}

/// Get an annotation set for the specified value.
AnnotationSet AnnotationSet::get(Value v) {
  if (auto op = v.getDefiningOp())
    return AnnotationSet(op);
  // If its not an Operation, then must be a block argument.
  auto arg = v.dyn_cast<BlockArgument>();
  auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
  return forPort(module, arg.getArgNumber());
}

/// Store the annotations in this set in an operation's `annotations` attribute,
/// overwriting any existing annotations.
bool AnnotationSet::applyToOperation(Operation *op) const {
  auto before = op->getAttrDictionary();
  op->setAttr(getAnnotationAttrName(), getArrayAttr());
  return op->getAttrDictionary() != before;
}

static bool applyToAttrListImpl(const AnnotationSet &annoSet, StringRef key,
                                NamedAttrList &attrs) {
  if (annoSet.empty())
    return bool(attrs.erase(key));
  else {
    auto attr = annoSet.getArrayAttr();
    return attrs.set(key, attr) != attr;
  }
}

/// Store the annotations in this set in a `NamedAttrList` as an array attribute
/// with the name `annotations`.
bool AnnotationSet::applyToAttrList(NamedAttrList &attrs) const {
  return applyToAttrListImpl(*this, getAnnotationAttrName(), attrs);
}

/// Store the annotations in this set in a `NamedAttrList` as an array attribute
/// with the name `firrtl.annotations`.
bool AnnotationSet::applyToPortAttrList(NamedAttrList &attrs) const {
  return applyToAttrListImpl(*this, getDialectAnnotationAttrName(), attrs);
}

static DictionaryAttr applyToDictionaryAttrImpl(const AnnotationSet &annoSet,
                                                StringRef key,
                                                ArrayRef<NamedAttribute> attrs,
                                                bool sorted,
                                                DictionaryAttr originalDict) {
  // Find the location in the dictionary where the entry would go.
  ArrayRef<NamedAttribute>::iterator it;
  if (sorted) {
    it = llvm::lower_bound(attrs, key);
    if (it != attrs.end() && it->first != key)
      it = attrs.end();
  } else {
    it = llvm::find_if(
        attrs, [key](NamedAttribute attr) { return attr.first == key; });
  }

  // Fast path in case there are no annotations in the dictionary and we are not
  // supposed to add any.
  if (it == attrs.end() && annoSet.empty())
    return originalDict;

  // Fast path in case there already is an entry in the dictionary, it matches
  // the set, and, in the case we're supposed to remove empty sets, we're not
  // leaving an empty entry in the dictionary.
  if (it != attrs.end() && it->second == annoSet.getArrayAttr() &&
      !annoSet.empty())
    return originalDict;

  // If we arrive here, we are supposed to assemble a new dictionary.
  SmallVector<NamedAttribute> newAttrs;
  newAttrs.reserve(attrs.size() + 1);
  newAttrs.append(attrs.begin(), it);
  if (!annoSet.empty())
    newAttrs.push_back(
        {Identifier::get(key, annoSet.getContext()), annoSet.getArrayAttr()});
  if (it != attrs.end())
    newAttrs.append(it + 1, attrs.end());
  return sorted ? DictionaryAttr::getWithSorted(annoSet.getContext(), newAttrs)
                : DictionaryAttr::get(annoSet.getContext(), newAttrs);
}

/// Update the attribute dictionary of an operation to contain this annotation
/// set.
DictionaryAttr
AnnotationSet::applyToDictionaryAttr(DictionaryAttr attrs) const {
  return applyToDictionaryAttrImpl(*this, getAnnotationAttrName(),
                                   attrs.getValue(), true, attrs);
}

DictionaryAttr
AnnotationSet::applyToDictionaryAttr(ArrayRef<NamedAttribute> attrs) const {
  return applyToDictionaryAttrImpl(*this, getAnnotationAttrName(), attrs, false,
                                   {});
}

/// Update the attribute dictionary of a port to contain this annotation set.
DictionaryAttr
AnnotationSet::applyToPortDictionaryAttr(DictionaryAttr attrs) const {
  return applyToDictionaryAttrImpl(*this, getDialectAnnotationAttrName(),
                                   attrs.getValue(), true, attrs);
}

DictionaryAttr
AnnotationSet::applyToPortDictionaryAttr(ArrayRef<NamedAttribute> attrs) const {
  return applyToDictionaryAttrImpl(*this, getDialectAnnotationAttrName(), attrs,
                                   false, {});
}

DictionaryAttr AnnotationSet::getAnnotationImpl(StringAttr className) const {
  for (auto annotation : annotations) {
    DictionaryAttr annotDict;
    if (auto dict = annotation.dyn_cast<DictionaryAttr>())
      annotDict = dict;
    else
      annotDict = annotation.cast<SubAnnotationAttr>().getAnnotations();
    if (auto annotClass = annotDict.get("class"))
      if (annotClass == className)
        return annotDict;
  }
  return {};
}

DictionaryAttr AnnotationSet::getAnnotationImpl(StringRef className) const {
  for (auto annotation : annotations) {
    DictionaryAttr annotDict;
    if (auto dict = annotation.dyn_cast<DictionaryAttr>())
      annotDict = dict;
    else
      annotDict = annotation.cast<SubAnnotationAttr>().getAnnotations();
    if (auto annotClass = annotDict.get("class"))
      if (auto annotClassString = annotClass.dyn_cast<StringAttr>())
        if (annotClassString.getValue() == className)
          return annotDict;
  }
  return {};
}

bool AnnotationSet::hasAnnotationImpl(StringAttr className) const {
  return getAnnotationImpl(className) != DictionaryAttr();
}

bool AnnotationSet::hasAnnotationImpl(StringRef className) const {
  return getAnnotationImpl(className) != DictionaryAttr();
}

bool AnnotationSet::hasDontTouch() const {
  return hasAnnotation(dontTouchAnnoClass);
}

bool AnnotationSet::setDontTouch(bool dontTouch) {
  if (dontTouch)
    return addDontTouch();
  else
    return removeDontTouch();
}

bool AnnotationSet::addDontTouch() {
  if (hasDontTouch())
    return false;
  addAnnotations(DictionaryAttr::get(
      getContext(), {{Identifier::get("class", getContext()),
                      StringAttr::get(getContext(), dontTouchAnnoClass)}}));
  return true;
}

bool AnnotationSet::removeDontTouch() {
  return removeAnnotation(dontTouchAnnoClass);
}

bool AnnotationSet::hasDontTouch(Operation *op) {
  return AnnotationSet(op).hasDontTouch();
}

bool AnnotationSet::setDontTouch(Operation *op, bool dontTouch) {
  if (dontTouch)
    return addDontTouch(op);
  else
    return removeDontTouch(op);
}

bool AnnotationSet::addDontTouch(Operation *op) {
  AnnotationSet annos(op);
  auto changed = annos.addDontTouch();
  if (changed)
    annos.applyToOperation(op);
  return changed;
}

bool AnnotationSet::removeDontTouch(Operation *op) {
  AnnotationSet annos(op);
  auto changed = annos.removeDontTouch();
  if (changed)
    annos.applyToOperation(op);
  return changed;
}

/// Add more annotations to this AttributeSet.
void AnnotationSet::addAnnotations(ArrayRef<Annotation> newAnnotations) {
  if (newAnnotations.empty())
    return;

  SmallVector<Attribute> annotationVec;
  annotationVec.reserve(annotations.size() + newAnnotations.size());
  annotationVec.append(annotations.begin(), annotations.end());
  for (auto anno : newAnnotations)
    annotationVec.push_back(anno.getDict());
  annotations = ArrayAttr::get(getContext(), annotationVec);
}

void AnnotationSet::addAnnotations(ArrayRef<Attribute> newAnnotations) {
  if (newAnnotations.empty())
    return;

  if (empty()) {
    annotations = ArrayAttr::get(getContext(), newAnnotations);
    return;
  }

  SmallVector<Attribute> annotationVec;
  annotationVec.reserve(annotations.size() + newAnnotations.size());
  annotationVec.append(annotations.begin(), annotations.end());
  annotationVec.append(newAnnotations.begin(), newAnnotations.end());
  annotations = ArrayAttr::get(getContext(), annotationVec);
}

void AnnotationSet::addAnnotations(ArrayAttr newAnnotations) {
  if (!newAnnotations)
    return;

  if (empty()) {
    annotations = newAnnotations;
    return;
  }

  SmallVector<Attribute> annotationVec;
  annotationVec.reserve(annotations.size() + newAnnotations.size());
  annotationVec.append(annotations.begin(), annotations.end());
  annotationVec.append(newAnnotations.begin(), newAnnotations.end());
  annotations = ArrayAttr::get(getContext(), annotationVec);
}

/// Remove an annotation from this annotation set. Returns true if any were
/// removed, false otherwise.
bool AnnotationSet::removeAnnotation(Annotation anno) {
  return removeAnnotations([&](Annotation other) { return other == anno; });
}

/// Remove an annotation from this annotation set. Returns true if any were
/// removed, false otherwise.
bool AnnotationSet::removeAnnotation(Attribute anno) {
  return removeAnnotations(
      [&](Annotation other) { return other.getDict() == anno; });
}

/// Remove an annotation from this annotation set. Returns true if any were
/// removed, false otherwise.
bool AnnotationSet::removeAnnotation(StringRef className) {
  return removeAnnotations(
      [&](Annotation other) { return other.getClass() == className; });
}

/// Remove all annotations from this annotation set for which `predicate`
/// returns true.
bool AnnotationSet::removeAnnotations(
    llvm::function_ref<bool(Annotation)> predicate) {
  // Fast path for empty sets.
  auto attr = getArrayAttr();
  if (!attr)
    return false;

  // Return an Annotation built from an attribute which may be either a
  // DictionaryAttr or a SubAnnotationAttr.
  auto buildAnnotation = [](const Attribute *a) -> Annotation {
    if (auto b = a->dyn_cast<DictionaryAttr>())
      return Annotation(b);
    else
      return Annotation(a->cast<SubAnnotationAttr>().getAnnotations());
  };

  // Search for the first match.
  ArrayRef<Attribute> annos = getArrayAttr().getValue();
  auto it = annos.begin();
  while (it != annos.end() && !predicate(buildAnnotation(it)))
    ++it;

  // Fast path for sets where the predicate never matched.
  if (it == annos.end())
    return false;

  // Build a filtered list of annotations.
  SmallVector<Attribute> filteredAnnos;
  filteredAnnos.reserve(annos.size());
  filteredAnnos.append(annos.begin(), it);
  ++it;
  while (it != annos.end()) {
    if (!predicate(buildAnnotation(it)))
      filteredAnnos.push_back(*it);
    ++it;
  }
  annotations = ArrayAttr::get(getContext(), filteredAnnos);
  return true;
}

/// Remove all annotations from an operation for which `predicate` returns true.
bool AnnotationSet::removeAnnotations(
    Operation *op, llvm::function_ref<bool(Annotation)> predicate) {
  AnnotationSet annos(op);
  if (!annos.empty() && annos.removeAnnotations(predicate)) {
    annos.applyToOperation(op);
    return true;
  }
  return false;
}

bool AnnotationSet::removeAnnotations(Operation *op, StringRef className) {
  return removeAnnotations(
      op, [&](Annotation a) { return (a.getClass() == className); });
}

/// Remove all port annotations from a module or extmodule for which `predicate`
/// returns true.
bool AnnotationSet::removePortAnnotations(
    Operation *module,
    llvm::function_ref<bool(unsigned, Annotation)> predicate) {
  auto ports = module->getAttr("portAnnotations").dyn_cast_or_null<ArrayAttr>();
  if (!ports || ports.empty())
    return false;

  // Collect results
  SmallVector<Attribute> newAnnos;

  // Filter the annotations on each argument.
  bool changed = false;
  for (unsigned argNum = 0, argNumEnd = ports.size(); argNum < argNumEnd;
       ++argNum) {
    AnnotationSet annos(AnnotationSet(ports[argNum].cast<ArrayAttr>()));

    // Go through all annotations on this port and extract the interesting
    // ones. If any modifications were done, keep a reduced set of attributes
    // around for the port, otherwise just stick with the existing ones.
    if (!annos.empty())
      changed |= annos.removeAnnotations(
          [&](Annotation anno) { return predicate(argNum, anno); });
    newAnnos.push_back(annos.getArrayAttr());
  }

  // If we have made any changes, apply them to the operation.
  if (changed)
    module->setAttr("portAnnotations",
                    ArrayAttr::get(module->getContext(), newAnnos));
  return changed;
}

//===----------------------------------------------------------------------===//
// Annotation
//===----------------------------------------------------------------------===//

/// Return the 'class' that this annotation is representing.
StringAttr Annotation::getClassAttr() const {
  return ((DictionaryAttr)attrDict).getAs<StringAttr>("class");
}

/// Return the 'class' that this annotation is representing.
StringRef Annotation::getClass() const {
  if (auto classAttr = getClassAttr())
    return classAttr.getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// AnnotationSetIterator
//===----------------------------------------------------------------------===//

Annotation AnnotationSetIterator::operator*() const {
  auto attr = this->getBase().getArray()[this->getIndex()];
  if (auto dictAttr = attr.dyn_cast<DictionaryAttr>())
    return Annotation(dictAttr);
  else
    return Annotation(attr.cast<SubAnnotationAttr>().getAnnotations());
}

//===----------------------------------------------------------------------===//
// Annotation Details
//===----------------------------------------------------------------------===//

/// Check if an OMIR type is a string-encoded value that the FIRRTL dialect
/// simply passes through as a string without any decoding.
bool circt::firrtl::isOMIRStringEncodedPassthrough(StringRef type) {
  return type == "OMID" || type == "OMReference" || type == "OMBigInt" ||
         type == "OMLong" || type == "OMString" || type == "OMDouble" ||
         type == "OMBigDecimal" || type == "OMDeleted" || type == "OMConstant";
}
