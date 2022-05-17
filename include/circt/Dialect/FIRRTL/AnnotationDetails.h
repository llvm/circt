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

constexpr const char *blackBoxTargetDirAnnoClass =
    "firrtl.transforms.BlackBoxTargetDirAnno";
constexpr const char *blackBoxResourceFileNameAnnoClass =
    "firrtl.transforms.BlackBoxResourceFileNameAnno";
constexpr const char *extractCoverageAnnoClass =
    "sifive.enterprise.firrtl.ExtractCoverageAnnotation";
constexpr const char *testBenchDirAnnoClass =
    "sifive.enterprise.firrtl.TestBenchDirAnnotation";

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
constexpr const char *extractGrandCentralClass =
    "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation";

// SiFive specific Annotations
constexpr const char *dutAnnoClass =
    "sifive.enterprise.firrtl.MarkDUTAnnotation";
constexpr const char *testbenchDirAnnoClass =
    "sifive.enterprise.firrtl.TestBenchDirAnnotation";
constexpr const char *injectDUTHierarchyAnnoClass =
    "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation";

// Instance Extraction
constexpr const char *extractBlackBoxAnnoClass =
    "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation";
constexpr const char *extractClockGatesAnnoClass =
    "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation";
constexpr const char *extractSeqMemsAnnoClass =
    "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation";

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATIONDETAILS_H
