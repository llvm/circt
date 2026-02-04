//===- AnnotationDetails.h - common annotation logic ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains private APIs for dealing with annotations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_ANNOTATIONDETAILS_H
#define CIRCT_DIALECT_FIRRTL_ANNOTATIONDETAILS_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// Common strings related to annotations
//===----------------------------------------------------------------------===//

constexpr const char *rawAnnotations = "rawAnnotations";

//===----------------------------------------------------------------------===//
// Auto-generated Annotation Class Names
//===----------------------------------------------------------------------===//

// Include auto-generated annotation class name constants
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationDetails.h.inc"

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATIONDETAILS_H
