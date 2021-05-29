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
