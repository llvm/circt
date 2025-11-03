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

constexpr const char *conventionAnnoClass = "circt.ConventionAnnotation";
constexpr const char *typeLoweringAnnoClass =
    "circt.BodyTypeLoweringAnnotation";
constexpr const char *dontTouchAnnoClass =
    "firrtl.transforms.DontTouchAnnotation";
constexpr const char *enumComponentAnnoClass =
    "chisel3.experimental.EnumAnnotations$EnumComponentAnnotation";
constexpr const char *enumDefAnnoClass =
    "chisel3.experimental.EnumAnnotations$EnumDefAnnotation";
constexpr const char *enumVecAnnoClass =
    "chisel3.experimental.EnumAnnotations$EnumVecAnnotation";
constexpr const char *forceNameAnnoClass =
    "chisel3.util.experimental.ForceNameAnnotation";
constexpr const char *decodeTableAnnotation =
    "chisel3.util.experimental.decode.DecodeTableAnnotation";
constexpr const char *flattenAnnoClass = "firrtl.transforms.FlattenAnnotation";
constexpr const char *inlineAnnoClass = "firrtl.passes.InlineAnnotation";
constexpr const char *traceNameAnnoClass =
    "chisel3.experimental.Trace$TraceNameAnnotation";
constexpr const char *traceAnnoClass =
    "chisel3.experimental.Trace$TraceAnnotation";

constexpr const char *blackBoxInlineAnnoClass =
    "firrtl.transforms.BlackBoxInlineAnno";
constexpr const char *blackBoxPathAnnoClass =
    "firrtl.transforms.BlackBoxPathAnno";
constexpr const char *blackBoxTargetDirAnnoClass =
    "firrtl.transforms.BlackBoxTargetDirAnno";
constexpr const char *blackBoxAnnoClass =
    "firrtl.transforms.BlackBox"; // Not in SFC
constexpr const char *verbatimBlackBoxAnnoClass = "circt.VerbatimBlackBoxAnno";
constexpr const char *mustDedupAnnoClass =
    "firrtl.transforms.MustDeduplicateAnnotation";
constexpr const char *runFIRRTLTransformAnnoClass =
    "firrtl.stage.RunFirrtlTransformAnnotation";
constexpr const char *extractAssertAnnoClass =
    "sifive.enterprise.firrtl.ExtractAssertionsAnnotation";
constexpr const char *extractAssumeAnnoClass =
    "sifive.enterprise.firrtl.ExtractAssumptionsAnnotation";
constexpr const char *extractCoverageAnnoClass =
    "sifive.enterprise.firrtl.ExtractCoverageAnnotation";
constexpr const char *testBenchDirAnnoClass =
    "sifive.enterprise.firrtl.TestBenchDirAnnotation";
constexpr const char *moduleHierAnnoClass =
    "sifive.enterprise.firrtl.ModuleHierarchyAnnotation";
constexpr const char *outputDirAnnoClass = "circt.OutputDirAnnotation";
constexpr const char *testHarnessHierAnnoClass =
    "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation";
constexpr const char *retimeModulesFileAnnoClass =
    "sifive.enterprise.firrtl.RetimeModulesAnnotation";
constexpr const char *retimeModuleAnnoClass =
    "freechips.rocketchip.util.RetimeModuleAnnotation";
constexpr const char *verifBlackBoxAnnoClass =
    "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation";
constexpr const char *metadataDirectoryAttrName =
    "sifive.enterprise.firrtl.MetadataDirAnnotation";
constexpr const char *noDedupAnnoClass = "firrtl.transforms.NoDedupAnnotation";
constexpr const char *dedupGroupAnnoClass =
    "firrtl.transforms.DedupGroupAnnotation";

// Grand Central Annotations
constexpr const char *serializedViewAnnoClass =
    "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation";
constexpr const char *viewAnnoClass =
    "sifive.enterprise.grandcentral.ViewAnnotation";
constexpr const char *companionAnnoClass =
    "sifive.enterprise.grandcentral.ViewAnnotation.companion"; // not in SFC
constexpr const char *augmentedGroundTypeClass =
    "sifive.enterprise.grandcentral.AugmentedGroundType"; // not an annotation
constexpr const char *augmentedBundleTypeClass =
    "sifive.enterprise.grandcentral.AugmentedBundleType"; // not an annotation
constexpr const char *augmentedVectorTypeClass =
    "sifive.enterprise.grandcentral.AugmentedVectorType"; // not an annotation
constexpr const char *memTapClass =
    "sifive.enterprise.grandcentral.MemTapAnnotation";
constexpr const char *extractGrandCentralClass =
    "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation";
constexpr const char *grandCentralHierarchyFileAnnoClass =
    "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation";

// SiFive specific Annotations
constexpr const char *dutAnnoClass =
    "sifive.enterprise.firrtl.MarkDUTAnnotation";
constexpr const char *injectDUTHierarchyAnnoClass =
    "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation";
constexpr const char *sitestBlackBoxAnnoClass =
    "sifive.enterprise.firrtl.SitestBlackBoxAnnotation";
constexpr const char *sitestTestHarnessBlackBoxAnnoClass =
    "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation";
constexpr const char *sitestBlackBoxLibrariesAnnoClass =
    "sifive.enterprise.firrtl.SitestBlackBoxLibrariesAnnotation";
constexpr const char *dontObfuscateModuleAnnoClass =
    "sifive.enterprise.firrtl.DontObfuscateModuleAnnotation";
constexpr const char *elaborationArtefactsDirectoryAnnoClass =
    "sifive.enterprise.firrtl.ElaborationArtefactsDirectory";
constexpr const char *testHarnessPathAnnoClass =
    "sifive.enterprise.firrtl.TestHarnessPathAnnotation";
/// Annotation that marks a reset (port or wire) and domain.
constexpr const char *fullResetAnnoClass = "circt.FullResetAnnotation";
/// Annotation that marks a module as not belonging to any reset domain.
constexpr const char *excludeFromFullResetAnnoClass =
    "circt.ExcludeFromFullResetAnnotation";
/// Annotation that marks a reset (port or wire) and domain.
constexpr const char *fullAsyncResetAnnoClass =
    "sifive.enterprise.firrtl.FullAsyncResetAnnotation";
/// Annotation that marks a module as not belonging to any reset domain.
constexpr const char *ignoreFullAsyncResetAnnoClass =
    "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation";

// MemToRegOfVec Annotations
constexpr const char *convertMemToRegOfVecAnnoClass =
    "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$";

// Instance Extraction
constexpr const char *extractBlackBoxAnnoClass =
    "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation";
constexpr const char *extractClockGatesAnnoClass =
    "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation";
constexpr const char *extractSeqMemsAnnoClass =
    "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation";

// AddSeqMemPort Annotations
constexpr const char *addSeqMemPortAnnoClass =
    "sifive.enterprise.firrtl.AddSeqMemPortAnnotation";
constexpr const char *addSeqMemPortsFileAnnoClass =
    "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation";

// Memory file loading annotations.
constexpr const char *loadMemoryFromFileAnnoClass =
    "firrtl.annotations.LoadMemoryAnnotation";
constexpr const char *loadMemoryFromFileInlineAnnoClass =
    "firrtl.annotations.MemoryFileInlineAnnotation";

// WiringTransform Annotations
constexpr const char *wiringSinkAnnoClass =
    "firrtl.passes.wiring.SinkAnnotation";
constexpr const char *wiringSourceAnnoClass =
    "firrtl.passes.wiring.SourceAnnotation";

// Attribute annotations.
constexpr const char *attributeAnnoClass = "firrtl.AttributeAnnotation";

// Module Prefix Annotations.
constexpr const char *modulePrefixAnnoClass = "chisel3.ModulePrefixAnnotation";

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_ANNOTATIONDETAILS_H
