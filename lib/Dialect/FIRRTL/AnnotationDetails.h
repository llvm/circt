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

#ifndef DIALECT_FIRRTL_ANNOTATIONDETAILS_H
#define DIALECT_FIRRTL_ANNOTATIONDETAILS_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

/// Check if an OMIR type is a string-encoded value that the FIRRTL dialect
/// simply passes through as a string without any decoding.
bool isOMIRStringEncodedPassthrough(StringRef type);

//===----------------------------------------------------------------------===//
// Annotation Class Names
//===----------------------------------------------------------------------===//

constexpr const char *dontTouchAnnoClass =
    "firrtl.transforms.DontTouchAnnotation";

constexpr const char *omirAnnoClass =
    "freechips.rocketchip.objectmodel.OMIRAnnotation";
constexpr const char *omirFileAnnoClass =
    "freechips.rocketchip.objectmodel.OMIRFileAnnotation";
constexpr const char *omirTrackerAnnoClass =
    "freechips.rocketchip.objectmodel.OMIRTracker";

// Grand Central Annotations
constexpr const char *dataTapsClass =
    "sifive.enterprise.grandcentral.DataTapsAnnotation";
constexpr const char *memTapClass =
    "sifive.enterprise.grandcentral.MemTapAnnotation";
constexpr const char *deletedKeyClass =
    "sifive.enterprise.grandcentral.DeletedDataTapKey";
constexpr const char *literalKeyClass =
    "sifive.enterprise.grandcentral.LiteralDataTapKey";
constexpr const char *referenceKeyClass =
    "sifive.enterprise.grandcentral.ReferenceDataTapKey";
constexpr const char *internalKeyClass =
    "sifive.enterprise.grandcentral.DataTapModuleSignalKey";

} // namespace firrtl
} // namespace circt

#endif // DIALECT_FIRRTL_ANNOTATIONDETAILS_H
