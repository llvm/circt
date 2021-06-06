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
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Operation.h"
using namespace circt;
using namespace firrtl;

static ArrayRef<Attribute> getAnnotationsFrom(Operation *op) {
  if (ArrayAttr array = op->getAttrOfType<ArrayAttr>("annotations"))
    return array.getValue();
  return {};
}

/// Get an annotation set for the specified operation.
AnnotationSet::AnnotationSet(Operation *op)
    : annotations(getAnnotationsFrom(op)) {}

/// Get an annotation set for the specified module port.
AnnotationSet AnnotationSet::forPort(Operation *module, size_t portNo) {
  for (auto a : mlir::function_like_impl::getArgAttrs(module, portNo)) {
    if (a.first == "firrtl.annotations")
      return AnnotationSet(a.second.cast<ArrayAttr>());
  }
  return AnnotationSet(ArrayRef<Attribute>());
}

/// Return this annotation set as an argument attribute dictionary for a port.
DictionaryAttr AnnotationSet::getArgumentAttrDict(
    MLIRContext *context, ArrayRef<NamedAttribute> otherPortAttrs) const {
  if (empty())
    return DictionaryAttr::get(context, otherPortAttrs);

  SmallVector<NamedAttribute> allPortAttrs(otherPortAttrs.begin(),
                                           otherPortAttrs.end());
  // Annotations are stored as under the "firrtl.annotations" key.
  allPortAttrs.push_back(
      {Identifier::get("firrtl.annotations", context), getArrayAttr(context)});
  return DictionaryAttr::get(context, allPortAttrs);
}

DictionaryAttr AnnotationSet::getAnnotationImpl(StringRef className) const {
  for (auto annotation : annotations) {
    auto annotDict = annotation.cast<DictionaryAttr>();
    if (auto annotClass = annotDict.get("class"))
      if (auto annotClassString = annotClass.dyn_cast<StringAttr>())
        if (annotClassString.getValue() == className)
          return annotDict;
  }
  return {};
}

bool AnnotationSet::hasAnnotationImpl(StringRef className) const {
  return getAnnotationImpl(className) != DictionaryAttr();
}

bool AnnotationSet::hasDontTouch() const {
  return hasAnnotation("firrtl.transforms.DontTouchAnnotation");
}

/// Return the 'class' that this annotation is representing.
StringRef Annotation::getClass() const {
  if (auto classAttr = ((DictionaryAttr)attrDict).getAs<StringAttr>("class"))
    return classAttr.getValue();
  return {};
}