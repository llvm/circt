//===- GrandCentralTaps.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GrandCentralTaps pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/InstanceGraphBase.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;
using hw::HWInstanceLike;
using hw::InnerRefAttr;
using hw::InstancePath;
using mlir::FailureOr;

//===----------------------------------------------------------------------===//
// PointerLikeTypeTraits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct PointerLikeTypeTraits<InstanceOp> : PointerLikeTypeTraits<Operation *> {
public:
  static inline void *getAsVoidPointer(InstanceOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline InstanceOp getFromVoidPointer(void *p) {
    return InstanceOp::getFromOpaquePointer(p);
  }
};
} // end namespace llvm

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {

/// A port annotated with a data tap key or mem tap.
struct AnnotatedPort {
  unsigned portNum;
  Annotation anno;
};

/// An extmodule that has annotated ports.
struct AnnotatedExtModule {
  FExtModuleOp extModule;
  SmallVector<AnnotatedPort, 4> portAnnos;
  // Module annotations without data tap stuff.
  ArrayAttr filteredModuleAnnos;
  /// Port annotations without data tap stuff.
  SmallVector<AnnotationSet, 8> filteredPortAnnos;
};

/// A value annotated to be tapped.
struct TappedValue {
  Value value;
  Annotation anno;
};

/// A parsed integer literal.
struct Literal {
  IntegerAttr value;
  FIRRTLType type;
  operator bool() const { return value && type; }
};

/// Necessary information to wire up a port with tapped data or memory location.
struct PortWiring {
  using Target = hw::HWSymbolCache::Item;

  unsigned portNum;
  /// The different instance paths that lead to this port. If the NLA field is
  /// set, these are the different instance paths to the root of the NLA path.
  ArrayRef<InstancePath> prefices;
  /// The operation or module port being wire to this data tap module port.
  Target target;
  /// The fieldID which specifies the element we want to connect in the port.
  unsigned targetFieldID;
  /// An additional string suffix to append to the hierarchical name.
  SmallString<16> suffix;
  /// If set, the port should output a constant literal.
  Literal literal;
  /// The non-local anchor further specifying where to connect.
  HierPathOp nla;
  /// True if the tapped target is known to be zero-width.  This indicates that
  /// the port should not be wired.  The port will be removed by LowerToHW.
  bool zeroWidth = false;

  PortWiring() : target(nullptr) {}
};

} // namespace

/// Return a version of `path` that skips all front instances it has in common
/// with `other`.
static InstancePath stripCommonPrefix(InstancePath path, InstancePath other) {
  while (!path.empty() && !other.empty() && path.front() == other.front()) {
    path = path.drop_front();
    other = other.drop_front();
  }
  return path;
}

/// A reference data tap key, identified by the annotation id and port id.
using Key = std::pair<Attribute, Attribute>;

/// A tapped port, described as the module/extmodule operation and the port
/// number.
struct Port : std::pair<Operation *, unsigned> {
  using std::pair<Operation *, unsigned>::pair;
  operator bool() const { return bool(first); }
};

// Allow printing of `Key` through `<<`.
template <typename T>
static T &operator<<(T &os, Key key) {
  return os << "[" << key.first << ", " << key.second << "]";
}

/// Map an annotation to a `Key`.
static Key getKey(Annotation anno) {
  auto id = anno.getMember("id");
  auto portID = anno.getMember("portID");
  return {id, portID};
}

/// Parse a FIRRTL `UInt`/`SInt` literal.
static FailureOr<Literal> parseIntegerLiteral(MLIRContext *context,
                                              StringRef literal, Location loc) {
  auto initial = literal; // used for error reporting
  auto consumed = [&]() {
    return initial.take_front(initial.size() - literal.size());
  };
  auto bail = [&](const Twine &message) {
    mlir::emitError(loc, "expected ")
        << message << " in literal after `" << consumed() << "`";
    return failure();
  };
  auto consume = [&](StringRef text) {
    if (!literal.consume_front(text)) {
      (void)bail(Twine("`") + text + "`");
      return false;
    }
    return true;
  };

  // Parse the leading keyword.
  bool isSigned;
  if (literal.consume_front("UInt")) {
    isSigned = false;
  } else if (literal.consume_front("SInt")) {
    isSigned = true;
  } else {
    mlir::emitError(loc, "expected leading `UInt` or `SInt` in literal");
    return failure();
  }

  // Parse the optional width.
  Optional<APInt> width = {};
  if (literal.consume_front("<")) {
    auto widthLiteral = literal.take_while(llvm::isDigit);
    APInt parsedWidth;
    if (widthLiteral.getAsInteger(10, parsedWidth))
      return bail("integer width");
    literal = literal.drop_front(widthLiteral.size());
    if (!literal.consume_front(">"))
      return bail("closing `>`");
    width = parsedWidth;
  }

  // Parse the opening parenthesis.
  if (!consume("("))
    return failure();

  // Parse the opening quotes and base specifier.
  unsigned base = 10;
  bool hasQuotes = false;
  if (literal.consume_front("\"")) {
    hasQuotes = true;
    if (literal.consume_front("h"))
      base = 16;
    else if (literal.consume_front("o"))
      base = 8;
    else if (literal.consume_front("b"))
      base = 2;
    else
      return bail("base specifier (`h`, `o`, or `b`)");
  }

  // Parse the optional sign.
  bool isNegative = false;
  if (literal.consume_front("-"))
    isNegative = true;
  else if (literal.consume_front("+"))
    isNegative = false;

  // Parse the actual value.
  APInt parsedValue;
  auto valueLiteral = literal.take_while(llvm::isHexDigit);
  if (valueLiteral.getAsInteger(base, parsedValue))
    return bail("integer value");
  literal = literal.drop_front(valueLiteral.size());

  // Parse the closing quotes.
  if (hasQuotes && !literal.consume_front("\""))
    return bail("closing quotes");

  // Parse the closing parenthesis.
  if (!consume(")"))
    return failure();

  // Ensure that there's no junk afterwards.
  if (!literal.empty()) {
    mlir::emitError(loc, "extraneous `")
        << literal << "` in literal after `" << consumed() << "`";
    return failure();
  }

  // Ensure we have a 0 bit at the top to properly hold a negative value.
  if (parsedValue.isNegative())
    parsedValue = parsedValue.zext(parsedValue.getBitWidth() + 1);
  if (isNegative)
    parsedValue = -parsedValue;

  // Ensure the width is sound.
  int32_t saneWidth = -1;
  if (width) {
    saneWidth = (int32_t)width->getLimitedValue(INT32_MAX);
    if (saneWidth != *width) {
      mlir::emitError(loc, "width of literal `")
          << initial << "` is too big to handle";
      return failure();
    }
    parsedValue = isSigned ? parsedValue.sextOrTrunc(saneWidth)
                           : parsedValue.zextOrTrunc(saneWidth);
  }

  // Assemble the literal.
  Literal lit;
  lit.type = IntType::get(context, isSigned, saneWidth);
  lit.value = IntegerAttr::get(
      IntegerType::get(context, parsedValue.getBitWidth(),
                       isSigned ? IntegerType::Signed : IntegerType::Unsigned),
      parsedValue);
  return lit;
}

Value lowerInternalPathAnno(AnnoPathValue &srcTarget,
                            const AnnoPathValue &moduleTarget,
                            const AnnoPathValue &target,
                            StringAttr internalPathAttr, FIRRTLType targetType,
                            ApplyState &state) {
  Value sendVal;
  FModuleLike mod = cast<FModuleLike>(moduleTarget.ref.getOp());
  InstanceOp modInstance;
  if (!moduleTarget.instances.empty()) {
    modInstance = moduleTarget.instances.back();
  } else {
    auto *node = state.instancePathCache.instanceGraph.lookup(
        cast<hw::HWModuleLike>((Operation *)mod));
    if (!node->hasOneUse()) {
      mod->emitOpError(
          "cannot be used for DataTaps, it is instantiated multiple times");
      return nullptr;
    }
    modInstance = cast<InstanceOp>((*node->uses().begin())->getInstance());
  }
  ImplicitLocOpBuilder builder(modInstance.getLoc(), modInstance);
  builder.setInsertionPointAfter(modInstance);
  auto portRefType = RefType::get(targetType.cast<FIRRTLBaseType>());
  StringRef refName("ref");
  // Add RefType ports corresponding to this "internalPath" to the external
  // module. This also updates all the instances of the external module.
  // This removes and replaces the instance, and returns the updated
  // instance.
  modInstance = addPortsToModule(
      mod, modInstance, portRefType, Direction::Out, refName,
      state.instancePathCache,
      [&](FModuleLike mod) -> ModuleNamespace & {
        return state.getNamespace(mod);
      },
      &state.targetCaches);
  // Since the intance op genenerates the RefType output, no need of another
  // RefSendOp.
  sendVal = modInstance.getResults().back();
  // Now set the instance as the source for the final datatap xmr.
  srcTarget = AnnoPathValue(modInstance);
  if (auto extMod = dyn_cast<FExtModuleOp>((Operation *)mod)) {
    // The extern module can have other internal paths attached to it,
    // append this to them.
    SmallVector<Attribute> paths(extMod.getInternalPathsAttr().getValue());
    paths.push_back(internalPathAttr);
    extMod.setInternalPathsAttr(builder.getArrayAttr(paths));
  } else if (auto intMod = dyn_cast<FModuleOp>((Operation *)mod)) {
    auto builder = ImplicitLocOpBuilder::atBlockEnd(
        intMod.getLoc(), &intMod.getBody().getBlocks().back());
    auto pathStr = builder.create<VerbatimExprOp>(
        portRefType.getType(), internalPathAttr.getValue(), ValueRange{});
    auto sendPath = builder.create<RefSendOp>(pathStr);
    builder.create<StrictConnectOp>(intMod.getArguments().back(),
                                    sendPath.getResult());
  }

  if (!moduleTarget.instances.empty())
    srcTarget.instances = moduleTarget.instances;
  else {
    auto path = state.instancePathCache
                    .getAbsolutePaths(modInstance->getParentOfType<FModuleOp>())
                    .back();
    srcTarget.instances.append(path.begin(), path.end());
  }
  return sendVal;
}

LogicalResult static applyNoBlackBoxStyleDataTaps(const AnnoPathValue &target,
                                                  DictionaryAttr anno,
                                                  ApplyState &state) {
  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  // Process all the taps.
  auto keyAttr = tryGetAs<ArrayAttr>(anno, anno, "keys", loc, dataTapsClass);
  if (!keyAttr)
    return failure();
  auto noDedupAnnoClassName = StringAttr::get(context, noDedupAnnoClass);
  auto noDedupAnno = DictionaryAttr::get(
      context, {{StringAttr::get(context, "class"), noDedupAnnoClassName}});
  for (size_t i = 0, e = keyAttr.size(); i != e; ++i) {
    auto b = keyAttr[i];
    auto path = ("keys[" + Twine(i) + "]").str();
    auto bDict = b.cast<DictionaryAttr>();
    auto classAttr =
        tryGetAs<StringAttr>(bDict, anno, "class", loc, dataTapsClass, path);
    if (!classAttr)
      return failure();
    // Can only handle ReferenceDataTapKey and DataTapModuleSignalKey
    if (classAttr.getValue() != referenceKeyClass &&
        classAttr.getValue() != internalKeyClass)
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' with path '" + path + ".class" +
                                      +"' contained an unknown/unimplemented "
                                       "DataTapKey class '" +
                                      classAttr.getValue() + "'.")
                 .attachNote()
             << "The full Annotation is reprodcued here: " << anno << "\n";

    auto sinkNameAttr =
        tryGetAs<StringAttr>(bDict, anno, "sink", loc, dataTapsClass, path);
    std::string wirePathStr;
    if (sinkNameAttr)
      wirePathStr = canonicalizeTarget(sinkNameAttr.getValue());
    if (!wirePathStr.empty())
      if (!tokenizePath(wirePathStr))
        wirePathStr.clear();
    Optional<AnnoPathValue> wireTarget = None;
    if (!wirePathStr.empty())
      wireTarget = resolvePath(wirePathStr, state.circuit, state.symTbl,
                               state.targetCaches);
    if (!wireTarget)
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' with wire path '" + wirePathStr +
                                      "' couldnot be resolved.");
    if (!wireTarget->ref.getImpl().isOp())
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' with path '" + path + ".class" +
                                      +"' cannot specify a port for sink.");
    // Extract the name of the wire, used for datatap.
    auto tapName = StringAttr::get(
        context, wirePathStr.substr(wirePathStr.find_last_of('>') + 1));
    Optional<AnnoPathValue> srcTarget = None;
    Value sendVal;
    if (classAttr.getValue() == internalKeyClass) {
      // For DataTapModuleSignalKey, the source is encoded as a string, that
      // should exist inside the specified module. This source string is used as
      // a suffix to the instance name for the module inside a VerbatimExprOp.
      // This verbatim represents an intermediate xmr, which is then used by a
      // ref.send to be read remotely.
      auto internalPathAttr = tryGetAs<StringAttr>(bDict, anno, "internalPath",
                                                   loc, dataTapsClass, path);
      auto moduleAttr =
          tryGetAs<StringAttr>(bDict, anno, "module", loc, dataTapsClass, path);
      if (!internalPathAttr || !moduleAttr)
        return failure();
      auto moduleTargetStr = canonicalizeTarget(moduleAttr.getValue());
      if (!tokenizePath(moduleTargetStr))
        return failure();
      Optional<AnnoPathValue> moduleTarget = resolvePath(
          moduleTargetStr, state.circuit, state.symTbl, state.targetCaches);
      if (!moduleTarget)
        return failure();
      AnnoPathValue internalPathSrc;
      sendVal = lowerInternalPathAnno(
          internalPathSrc, *moduleTarget, target, internalPathAttr,
          wireTarget->ref.getOp()->getResult(0).getType().cast<FIRRTLType>(),
          state);
      if (!sendVal)
        return failure();
      srcTarget = internalPathSrc;
    } else {
      // Now handle ReferenceDataTapKey. Get the source from annotation.
      auto sourceAttr =
          tryGetAs<StringAttr>(bDict, anno, "source", loc, dataTapsClass, path);
      if (!sourceAttr)
        return failure();
      auto sourcePathStr = canonicalizeTarget(sourceAttr.getValue());
      if (!tokenizePath(sourcePathStr))
        return failure();
      LLVM_DEBUG(llvm::dbgs() << "\n Drill xmr path from :" << sourcePathStr
                              << " to " << wirePathStr);
      srcTarget = resolvePath(sourcePathStr, state.circuit, state.symTbl,
                              state.targetCaches);
    }
    if (!srcTarget)
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' source path couldnot be resolved.");

    auto wireModule =
        cast<FModuleOp>(wireTarget->ref.getModule().getOperation());

    if (auto extMod = dyn_cast<FExtModuleOp>(srcTarget->ref.getOp())) {
      // If the source is a port on extern module, then move the source to the
      // instance port for the ext module.
      auto portNo = srcTarget->ref.getImpl().getPortNo();
      auto lastInst = srcTarget->instances.pop_back_val();
      auto builder = ImplicitLocOpBuilder::atBlockEnd(lastInst.getLoc(),
                                                      lastInst->getBlock());
      builder.setInsertionPointAfter(lastInst);
      // Instance port cannot be used as an annotation target, so use a NodeOp.
      auto node = builder.create<NodeOp>(lastInst.getType(portNo),
                                         lastInst.getResult(portNo));
      AnnotationSet::addDontTouch(node);
      srcTarget->ref = AnnoTarget(circt::firrtl::detail::AnnoTargetImpl(node));
    }

    SmallVector<InstanceOp> pathFromSrcToWire;
    FModuleOp lcaModule;
    // Find the lca and get the path from source to wire through that lca.
    if (findLCAandSetPath(*srcTarget, *wireTarget, pathFromSrcToWire, lcaModule,
                          state)
            .failed())
      return mlir::emitError(
          loc, "Annotation '" + Twine(dataTapsClass) + "' with path '" + path +
                   ".class" +
                   +"' failed to find a uinque path from source to wire.");
    LLVM_DEBUG(llvm::dbgs() << "\n lca :" << lcaModule.getNameAttr();
               for (auto i
                    : pathFromSrcToWire) llvm::dbgs()
               << "\n"
               << i->getParentOfType<FModuleOp>().getNameAttr() << ">"
               << i.getNameAttr(););
    // The RefSend value can be either generated by the instance of an external
    // module or a RefSendOp.
    if (!sendVal) {
      auto srcModule =
          dyn_cast<FModuleOp>(srcTarget->ref.getModule().getOperation());

      Value refSendBase;
      ImplicitLocOpBuilder refSendBuilder(srcModule.getLoc(), srcModule);
      // Set the insertion point for the RefSend, it should be dominated by the
      // srcTarget value. If srcTarget is a port, then insert the RefSend
      // at the beggining of the module, else define the RefSend at the end of
      // the block that contains the srcTarget Op.
      if (srcTarget->ref.getImpl().isOp()) {
        refSendBase = srcTarget->ref.getImpl().getOp()->getResult(0);
        refSendBuilder.setInsertionPointAfter(srcTarget->ref.getOp());
      } else if (srcTarget->ref.getImpl().isPort()) {
        refSendBase =
            srcModule.getArgument(srcTarget->ref.getImpl().getPortNo());
        refSendBuilder.setInsertionPointToStart(srcModule.getBodyBlock());
      }
      // If the target value is a field of an aggregate create the
      // subfield/subaccess into it.
      refSendBase =
          getValueByFieldID(refSendBuilder, refSendBase, srcTarget->fieldIdx);
      // Note: No DontTouch added to refSendTarget, it can be constantprop'ed or
      // CSE'ed.
      sendVal = refSendBuilder.create<RefSendOp>(refSendBase);
    }
    // Now drill ports to connect the `sendVal` to the `wireTarget`.
    auto remoteXMR = borePortsOnPath(
        pathFromSrcToWire, lcaModule, sendVal, tapName.getValue(),
        state.instancePathCache,
        [&](FModuleLike mod) -> ModuleNamespace & {
          return state.getNamespace(mod);
        },
        &state.targetCaches);
    ImplicitLocOpBuilder refResolveBuilder(wireModule.getLoc(), wireModule);
    AnnotationSet annos(wireModule);
    if (!annos.hasAnnotation(noDedupAnnoClassName)) {
      annos.addAnnotations(noDedupAnno);
      annos.applyToOperation(wireModule);
    }

    if (remoteXMR.isa<BlockArgument>())
      refResolveBuilder.setInsertionPointToStart(wireModule.getBodyBlock());
    else
      refResolveBuilder.setInsertionPointAfter(remoteXMR.getDefiningOp());
    auto refResolve = refResolveBuilder.create<RefResolveOp>(remoteXMR);
    refResolveBuilder.setInsertionPointToEnd(
        wireTarget->ref.getOp()->getBlock());
    auto wireType = wireTarget->ref.getOp()
                        ->getResult(0)
                        .getType()
                        .cast<FIRRTLType>()
                        .cast<FIRRTLBaseType>();
    Value resolveResult = refResolve.getResult();
    if (refResolve.getType().isResetType() &&
        refResolve.getType().getWidthlessType() !=
            wireType.getWidthlessType()) {
      if (wireType.dyn_cast<IntType>())
        resolveResult =
            refResolveBuilder.create<AsUIntPrimOp>(wireType, refResolve);
      else if (wireType.isa<AsyncResetType>())
        resolveResult =
            refResolveBuilder.create<AsAsyncResetPrimOp>(refResolve);
    }
    refResolveBuilder.create<ConnectOp>(
        getValueByFieldID(refResolveBuilder,
                          wireTarget->ref.getOp()->getResult(0),
                          wireTarget->fieldIdx),
        resolveResult);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Code related to handling Grand Central Data/Mem Taps annotations
//===----------------------------------------------------------------------===//

// Describes tap points into the design.  This has the following structure:
//   blackBox: ModuleTarget
//   keys: Seq[DataTapKey]
// DataTapKey has multiple implementations:
//   - ReferenceDataTapKey: (tapping a point which exists in the FIRRTL)
//       portName: ReferenceTarget
//       source: ReferenceTarget
//   - DataTapModuleSignalKey: (tapping a point, by name, in a blackbox)
//       portName: ReferenceTarget
//       module: IsModule
//       internalPath: String
//   - DeletedDataTapKey: (not implemented here)
//       portName: ReferenceTarget
//   - LiteralDataTapKey: (not implemented here)
//       portName: ReferenceTarget
//       literal: Literal
// A Literal is a FIRRTL IR literal serialized to a string.  For now, just
// store the string.
// TODO: Parse the literal string into a UInt or SInt literal.
LogicalResult circt::firrtl::applyGCTDataTaps(const AnnoPathValue &target,
                                              DictionaryAttr anno,
                                              ApplyState &state) {

  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  auto id = state.newID();
  NamedAttrList attrs;
  attrs.append("class", StringAttr::get(context, dataTapsBlackboxClass));
  // The new DataTaps donot have blackbox field. Lower them directly to RefType.
  if (!anno.contains("blackBox"))
    return applyNoBlackBoxStyleDataTaps(target, anno, state);
  auto blackBoxAttr =
      tryGetAs<StringAttr>(anno, anno, "blackBox", loc, dataTapsClass);
  if (!blackBoxAttr)
    return failure();
  auto canonicalTarget = canonicalizeTarget(blackBoxAttr.getValue());
  if (!tokenizePath(canonicalTarget))
    return failure();
  attrs.append("target", StringAttr::get(context, canonicalTarget));
  state.addToWorklistFn(DictionaryAttr::getWithSorted(context, attrs));

  // Process all the taps.
  auto keyAttr = tryGetAs<ArrayAttr>(anno, anno, "keys", loc, dataTapsClass);
  if (!keyAttr)
    return failure();
  for (size_t i = 0, e = keyAttr.size(); i != e; ++i) {
    auto b = keyAttr[i];
    auto path = ("keys[" + Twine(i) + "]").str();
    auto bDict = b.cast<DictionaryAttr>();
    auto classAttr =
        tryGetAs<StringAttr>(bDict, anno, "class", loc, dataTapsClass, path);
    if (!classAttr)
      return failure();

    // The "portName" field is common across all sub-types of DataTapKey.
    NamedAttrList port;
    auto portNameAttr =
        tryGetAs<StringAttr>(bDict, anno, "portName", loc, dataTapsClass, path);
    if (!portNameAttr)
      return failure();
    auto portTarget = canonicalizeTarget(portNameAttr.getValue());
    if (!tokenizePath(portTarget))
      return failure();

    if (classAttr.getValue() == referenceKeyClass) {
      NamedAttrList source;
      auto portID = state.newID();
      source.append("class", StringAttr::get(context, referenceKeySourceClass));
      source.append("id", id);
      source.append("portID", portID);
      auto sourceAttr =
          tryGetAs<StringAttr>(bDict, anno, "source", loc, dataTapsClass, path);
      if (!sourceAttr)
        return failure();
      auto sourceTarget = canonicalizeTarget(sourceAttr.getValue());
      if (!tokenizePath(sourceTarget))
        return failure();

      source.append("target", StringAttr::get(context, sourceTarget));

      state.addToWorklistFn(DictionaryAttr::get(context, source));

      // Annotate the data tap module port.
      port.append("class", StringAttr::get(context, referenceKeyPortClass));
      port.append("id", id);
      port.append("portID", portID);
      port.append("target", StringAttr::get(context, portTarget));
      state.addToWorklistFn(DictionaryAttr::getWithSorted(context, port));
      continue;
    }

    if (classAttr.getValue() == internalKeyClass) {
      NamedAttrList module;
      auto portID = state.newID();
      module.append("class", StringAttr::get(context, internalKeySourceClass));
      module.append("id", id);
      auto internalPathAttr = tryGetAs<StringAttr>(bDict, anno, "internalPath",
                                                   loc, dataTapsClass, path);
      auto moduleAttr =
          tryGetAs<StringAttr>(bDict, anno, "module", loc, dataTapsClass, path);
      if (!internalPathAttr || !moduleAttr)
        return failure();
      module.append("internalPath", internalPathAttr);
      module.append("portID", portID);
      auto moduleTarget = canonicalizeTarget(moduleAttr.getValue());
      if (!tokenizePath(moduleTarget))
        return failure();

      module.append("target", StringAttr::get(context, moduleTarget));
      state.addToWorklistFn(DictionaryAttr::getWithSorted(context, module));

      // Annotate the data tap module port.
      port.append("class", StringAttr::get(context, internalKeyPortClass));
      port.append("id", id);
      port.append("portID", portID);
      port.append("target", StringAttr::get(context, portTarget));
      state.addToWorklistFn(DictionaryAttr::get(context, port));
      continue;
    }

    if (classAttr.getValue() == deletedKeyClass) {
      // Annotate the data tap module port.
      port.append("class", classAttr);
      port.append("id", id);
      port.append("target", StringAttr::get(context, portTarget));
      state.addToWorklistFn(DictionaryAttr::get(context, port));
      continue;
    }

    if (classAttr.getValue() == literalKeyClass) {
      auto literalAttr = tryGetAs<StringAttr>(bDict, anno, "literal", loc,
                                              dataTapsClass, path);
      if (!literalAttr)
        return failure();

      // Annotate the data tap module port.
      port.append("class", classAttr);
      port.append("id", id);
      port.append("literal", literalAttr);
      port.append("target", StringAttr::get(context, portTarget));
      state.addToWorklistFn(DictionaryAttr::get(context, port));
      continue;
    }

    mlir::emitError(
        loc, "Annotation '" + Twine(dataTapsClass) + "' with path '" + path +
                 ".class" +
                 +"' contained an unknown/unimplemented DataTapKey class '" +
                 classAttr.getValue() + "'.")
            .attachNote()
        << "The full Annotation is reprodcued here: " << anno << "\n";
    return failure();
  }

  return success();
}

LogicalResult applyGCTMemTapsWithWires(const AnnoPathValue &target,
                                       DictionaryAttr anno,
                                       std::string &sourceTargetStr,
                                       ApplyState &state) {
  auto loc = state.circuit.getLoc();
  Value memDbgPort;
  Optional<AnnoPathValue> srcTarget = resolvePath(
      sourceTargetStr, state.circuit, state.symTbl, state.targetCaches);
  if (!srcTarget)
    return mlir::emitError(loc, "cannot resolve source target path '")
           << sourceTargetStr << "'";
  auto tapsAttr = tryGetAs<ArrayAttr>(anno, anno, "sink", loc, memTapClass);
  if (!tapsAttr || tapsAttr.empty())
    return mlir::emitError(loc, "sink must have at least one entry");
  if (auto combMem = dyn_cast<chirrtl::CombMemOp>(srcTarget->ref.getOp())) {
    if (!combMem.getType().getElementType().isGround())
      return combMem.emitOpError(
          "cannot generate MemTap to a memory with aggregate data type");
    ImplicitLocOpBuilder builder(combMem->getLoc(), combMem);
    builder.setInsertionPointAfter(combMem);
    // Construct the type for the debug port.
    auto debugType =
        RefType::get(FVectorType::get(combMem.getType().getElementType(),
                                      combMem.getType().getNumElements()));

    auto debugPort = builder.create<chirrtl::MemoryDebugPortOp>(
        debugType, combMem,
        state.getNamespace(srcTarget->ref.getModule()).newName("memTap"));

    memDbgPort = debugPort.getResult();
    if (srcTarget->instances.empty()) {
      auto path = state.instancePathCache.getAbsolutePaths(
          combMem->getParentOfType<FModuleOp>());
      if (path.size() > 1)
        return combMem.emitOpError(
            "cannot be resolved as source for MemTap, multiple paths from top "
            "exist and unique instance cannot be resolved");
      srcTarget->instances.append(path.back().begin(), path.back().end());
    }
    if (tapsAttr.size() != combMem.getType().getNumElements())
      return mlir::emitError(
          loc, "sink cannot specify more taps than the depth of the memory");
  } else
    return srcTarget->ref.getOp()->emitOpError(
        "unsupported operation, only CombMem can be used as the source of "
        "MemTap");

  auto tap = tapsAttr[0].dyn_cast_or_null<StringAttr>();
  if (!tap) {
    return mlir::emitError(
               loc, "Annotation '" + Twine(memTapClass) +
                        "' with path '.taps[0" +
                        "]' contained an unexpected type (expected a string).")
               .attachNote()
           << "The full Annotation is reprodcued here: " << anno << "\n";
  }
  auto wireTargetStr = canonicalizeTarget(tap.getValue());
  if (!tokenizePath(wireTargetStr))
    return failure();
  Optional<AnnoPathValue> wireTarget = resolvePath(
      wireTargetStr, state.circuit, state.symTbl, state.targetCaches);
  SmallVector<InstanceOp> pathFromSrcToWire;
  FModuleOp lcaModule;
  // Find the lca and get the path from source to wire through that lca.
  if (findLCAandSetPath(*srcTarget, *wireTarget, pathFromSrcToWire, lcaModule,
                        state)
          .failed())
    return mlir::emitError(loc,
                           "Failed to find a uinque path from source to wire.");
  LLVM_DEBUG(llvm::dbgs() << "\n lca :" << lcaModule.getNameAttr();
             for (auto i
                  : pathFromSrcToWire) llvm::dbgs()
             << "\n"
             << i->getParentOfType<FModuleOp>().getNameAttr() << ">"
             << i.getNameAttr(););
  auto srcModule =
      dyn_cast<FModuleOp>(srcTarget->ref.getModule().getOperation());
  ImplicitLocOpBuilder refSendBuilder(srcModule.getLoc(), srcModule);
  auto sendVal = memDbgPort;
  // Now drill ports to connect the `sendVal` to the `wireTarget`.
  auto remoteXMR = borePortsOnPath(
      pathFromSrcToWire, lcaModule, sendVal, "memTap", state.instancePathCache,
      [&](FModuleLike mod) -> ModuleNamespace & {
        return state.getNamespace(mod);
      },
      &state.targetCaches);
  auto wireModule = cast<FModuleOp>(wireTarget->ref.getModule());
  ImplicitLocOpBuilder refResolveBuilder(wireModule.getLoc(), wireModule);
  if (remoteXMR.isa<BlockArgument>())
    refResolveBuilder.setInsertionPointToStart(wireModule.getBodyBlock());
  else
    refResolveBuilder.setInsertionPointAfter(remoteXMR.getDefiningOp());
  auto refResolve = refResolveBuilder.create<RefResolveOp>(remoteXMR);
  refResolveBuilder.setInsertionPointToEnd(wireTarget->ref.getOp()->getBlock());
  if (wireTarget->ref.getOp()->getResult(0).getType() != refResolve.getType())
    return wireTarget->ref.getOp()->emitError(
        "cannot generate the MemTap, wiretap Type does not match the memory "
        "type");
  refResolveBuilder.create<StrictConnectOp>(
      wireTarget->ref.getOp()->getResult(0), refResolve.getResult());
  return success();
}

LogicalResult circt::firrtl::applyGCTMemTaps(const AnnoPathValue &target,
                                             DictionaryAttr anno,
                                             ApplyState &state) {

  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  auto id = state.newID();
  NamedAttrList attrs;
  auto sourceAttr =
      tryGetAs<StringAttr>(anno, anno, "source", loc, memTapClass);
  if (!sourceAttr)
    return failure();
  auto sourceTarget = canonicalizeTarget(sourceAttr.getValue());
  if (!tokenizePath(sourceTarget))
    return failure();
  attrs.append("class", StringAttr::get(context, memTapSourceClass));
  attrs.append("id", id);
  attrs.append("target", StringAttr::get(context, sourceTarget));

  if (!anno.contains("taps"))
    return applyGCTMemTapsWithWires(target, anno, sourceTarget, state);
  auto tapsAttr = tryGetAs<ArrayAttr>(anno, anno, "taps", loc, memTapClass);
  state.addToWorklistFn(DictionaryAttr::get(context, attrs));
  StringSet<> memTapBlackboxes;
  for (size_t i = 0, e = tapsAttr.size(); i != e; ++i) {
    auto tap = tapsAttr[i].dyn_cast_or_null<StringAttr>();
    if (!tap) {
      mlir::emitError(
          loc, "Annotation '" + Twine(memTapClass) + "' with path '.taps[" +
                   Twine(i) +
                   "]' contained an unexpected type (expected a string).")
              .attachNote()
          << "The full Annotation is reprodcued here: " << anno << "\n";
      return failure();
    }
    NamedAttrList port;
    port.append("class", StringAttr::get(context, memTapPortClass));
    port.append("id", id);
    port.append("portID", IntegerAttr::get(IntegerType::get(context, 64), i));
    auto canonTarget = canonicalizeTarget(tap.getValue());
    if (!tokenizePath(canonTarget))
      return failure();
    port.append("target", StringAttr::get(context, canonTarget));
    state.addToWorklistFn(DictionaryAttr::get(context, port));

    auto blackboxTarget = tokenizePath(canonTarget).value();
    blackboxTarget.name = {};
    blackboxTarget.component.clear();
    auto blackboxTargetStr = blackboxTarget.str();
    if (!memTapBlackboxes.insert(blackboxTargetStr).second)
      continue;

    NamedAttrList blackbox;
    blackbox.append("class", StringAttr::get(context, memTapBlackboxClass));
    blackbox.append("target", StringAttr::get(context, blackboxTargetStr));
    state.addToWorklistFn(DictionaryAttr::getWithSorted(context, blackbox));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

class GrandCentralTapsPass : public GrandCentralTapsBase<GrandCentralTapsPass> {
  void runOnOperation() override;
  void gatherAnnotations(Operation *op);
  void processAnnotation(AnnotatedPort &portAnno, AnnotatedExtModule &blackBox,
                         InstancePathCache &instancePaths);

  // Helpers to simplify collecting taps on the various things.
  void gatherTap(Annotation anno, Port port) {
    auto key = getKey(anno);
    annos.insert({key, anno});
    assert(!tappedPorts.count(key) && "ambiguous tap annotation");
    auto portType = cast<FModuleLike>(port.first).getPortType(port.second);
    auto firrtlPortType = portType.dyn_cast<FIRRTLType>();
    if (!firrtlPortType) {
      port.first->emitError("data tap cannot target port with non-FIRRTL type ")
          << portType;
      return signalPassFailure();
    }

    auto portWidth =
        firrtlPortType.cast<FIRRTLBaseType>().getBitWidthOrSentinel();
    // If the port width is non-zero, process it normally.  Otherwise, record it
    // as being zero-width.
    if (portWidth)
      tappedPorts.insert({key, port});
    else
      zeroWidthTaps.insert(key);
  }
  void gatherTap(Annotation anno, Operation *op) {
    auto key = getKey(anno);
    annos.insert({key, anno});
    assert(!tappedOps.count(key) && "ambiguous tap annotation");
    // If the port width is targeting a module (e.g., a blackbox) or if it has a
    // non-zero width, process it normally.  Otherwise, record it as being
    // zero-width.
    if (isa<FModuleLike>(op) || op->getResult(0)
                                    .getType()
                                    .cast<FIRRTLBaseType>()
                                    .getBitWidthOrSentinel())
      tappedOps.insert({key, op});
    else
      zeroWidthTaps.insert(key);
  }

  /// Returns a port's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(FModuleLike module, size_t portIdx);
  /// Obtain an inner reference to an operation, possibly adding an `inner_sym`
  /// to that operation.
  InnerRefAttr getInnerRefTo(Operation *op);
  /// Obtain an inner reference to a module port, possibly adding an `inner_sym`
  /// to that port.
  InnerRefAttr getInnerRefTo(FModuleLike module, size_t portIdx);

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  DenseMap<Key, Annotation> annos;
  DenseMap<Key, Operation *> tappedOps;
  DenseSet<Key> zeroWidthTaps;
  DenseMap<Key, Port> tappedPorts;
  SmallVector<PortWiring, 8> portWiring;

  /// The name of the directory where data and mem tap modules should be
  /// output.
  StringAttr maybeExtractDirectory = {};

  /// Cached module namespaces.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  /// The circuit symbol table, used to look up NLAs.
  SymbolTable *circuitSymbols;
};

void GrandCentralTapsPass::runOnOperation() {
  auto circuitOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Running the GCT Data Taps pass\n");
  SymbolTable symtbl(circuitOp);
  circuitSymbols = &symtbl;
  InnerSymbolTableCollection innerSymTblCol(circuitOp);
  InnerRefNamespace innerRefNS{symtbl, innerSymTblCol};

  // Here's a rough idea of what the Scala code is doing:
  // - Gather the `source` of all `keys` of all `DataTapsAnnotation`s throughout
  //   the design.
  // - Convert the sources, which are specified on modules, to the absolute
  //   paths in all instances. E.g. module M tap x will produce a.x and b.x if
  //   there are two instances a and b of module M in the design.
  // - All data tap keys are specified on black box ports.
  // - The code then processes every DataTapsAnnotation separately as follows
  //   (with the targeted blackbox and keys):
  // - Check for collisions between SV keywords and key port names (skip this).
  // - Find all instances of the blackbox, but then just pick the first one (we
  //   should probably do this for each?)
  // - Process each key independently as follows:
  // - Look up the absolute path of the source in the map. Ensure it exists and
  //   is unambiguous. Make it relative to the blackbox instance path.
  // - Look up the port on the black box.
  // - Create a hierarchical SV name and store this as an assignment to the
  //   blackbox port.
  //   - DeletedDataTapKey: skip and don't create a wiring
  //   - LiteralDataTapKey: just the literal
  //   - ReferenceDataTapKey: relative path with "." + source name
  //   - DataTapModuleSignalKey: relative path with "." + internal path
  // - Generate a body for the blackbox module with the signal mapping

  AnnotationSet circuitAnnotations(circuitOp);
  if (auto extractAnno =
          circuitAnnotations.getAnnotation(extractGrandCentralClass)) {
    auto directory = extractAnno.getMember<StringAttr>("directory");
    if (!directory) {
      circuitOp->emitError()
          << "contained an invalid 'ExtractGrandCentralAnnotation' that does "
             "not contain a 'directory' field: "
          << extractAnno.getDict();
      return signalPassFailure();
    }
    maybeExtractDirectory = directory;
  }

  // Build a generator for absolute module and instance paths in the design.
  InstancePathCache instancePaths(getAnalysis<InstanceGraph>());

  // Gather a list of extmodules that have data or mem tap annotations to be
  // expanded.
  SmallVector<AnnotatedExtModule, 4> modules;
  for (auto extModule : llvm::make_early_inc_range(
           circuitOp.getBodyBlock()->getOps<FExtModuleOp>())) {

    // If the external module indicates that it is a data or mem tap, but does
    // not actually contain any taps (it has no ports), then delete the module
    // and all instantiations of it.
    AnnotationSet annotations(extModule);
    if (annotations.hasAnnotation(dataTapsBlackboxClass) &&
        !extModule.getNumPorts()) {
      LLVM_DEBUG(llvm::dbgs() << "Extmodule " << extModule.getName()
                              << " is a data/memtap that has no ports and "
                                 "will be deleted\n";);
      for (auto *use : instancePaths.instanceGraph[extModule]->uses())
        use->getInstance().erase();
      extModule.erase();
      continue;
    }

    // Go through the module ports and collect the annotated ones.
    AnnotatedExtModule result{extModule, {}, {}, {}};
    result.filteredPortAnnos.reserve(extModule.getNumPorts());
    for (unsigned argNum = 0, e = extModule.getNumPorts(); argNum < e;
         ++argNum) {
      // Go through all annotations on this port and add the data tap key and
      // mem tap ones to the list.
      auto annos = AnnotationSet::forPort(extModule, argNum);
      annos.removeAnnotations([&](Annotation anno) {
        if (anno.isClass(memTapPortClass, deletedKeyClass, literalKeyClass,
                         internalKeyPortClass, referenceKeyPortClass)) {
          result.portAnnos.push_back({argNum, anno});
          return true;
        }
        return false;
      });
      result.filteredPortAnnos.push_back(annos);
    }

    // If there are data tap annotations on the module, which is likely the
    // case, create a filtered array of annotations with them removed.
    AnnotationSet annos(extModule.getOperation());
    annos.removeAnnotations(
        [&](Annotation anno) { return anno.isClass(dataTapsBlackboxClass); });
    result.filteredModuleAnnos = annos.getArrayAttr();

    if (!result.portAnnos.empty())
      modules.push_back(std::move(result));
  }

  LLVM_DEBUG({
    for (auto m : modules) {
      llvm::dbgs() << "Extmodule " << m.extModule.getName() << " has "
                   << m.portAnnos.size() << " port annotations\n";
    }
  });

  // Fast path if there's nothing to do.
  if (modules.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  // Gather the annotated ports and operations throughout the design that we are
  // supposed to tap in one way or another.
  tappedPorts.clear();
  tappedOps.clear();
  circuitOp.walk([&](Operation *op) { gatherAnnotations(op); });

  LLVM_DEBUG({
    llvm::dbgs() << "Tapped ports:\n";
    for (auto it : tappedPorts)
      llvm::dbgs() << "- " << it.first << ": "
                   << it.second.first->getAttr(SymbolTable::getSymbolAttrName())
                   << " port #" << it.second.second << "\n";
    llvm::dbgs() << "Tapped ops:\n";
    for (auto it : tappedOps)
      llvm::dbgs() << "- " << it.first << ": " << *it.second << "\n";
  });

  // Process each black box independently.
  for (auto blackBox : modules) {
    LLVM_DEBUG(llvm::dbgs() << "Generating impls for "
                            << blackBox.extModule.getName() << "\n");

    // As a first step, gather a list of all absolute paths to instances of
    // this black box.
    auto paths = instancePaths.getAbsolutePaths(blackBox.extModule);
    LLVM_DEBUG({
      for (auto path : paths)
        llvm::dbgs() << "- " << path << "\n";
    });

    // Go through the port annotations of the tap module and generate a
    // hierarchical path for each.
    portWiring.clear();
    portWiring.reserve(blackBox.portAnnos.size());
    for (auto portAnno : blackBox.portAnnos) {
      processAnnotation(portAnno, blackBox, instancePaths);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "- Wire up as follows:\n";
      for (auto wiring : portWiring) {
        llvm::dbgs() << "- Port " << wiring.portNum << ":\n";
        for (auto path : wiring.prefices) {
          llvm::dbgs() << "  - " << path;
          if (wiring.nla)
            llvm::dbgs() << "." << wiring.nla.getNamepathAttr();
          if (!wiring.suffix.empty())
            llvm::dbgs() << " $ " << wiring.suffix;
          llvm::dbgs() << "\n";
        }
      }
    });

    // Now we have an awkward mapping problem. We have multiple data tap
    // module instances, which reference things in modules that in turn have
    // multiple instances. This is a side-effect of how Grand Central
    // annotates things on modules rather than instances. (However in practice
    // these will have a one-to-one correspondence due to CHIRRTL having fully
    // uniquified instances.) To solve this issue, create a dedicated
    // implementation for every data tap instance, and among the possible
    // targets for the data taps choose the one with the shortest relative
    // path to the data tap instance.
    ImplicitLocOpBuilder builder(blackBox.extModule->getLoc(),
                                 blackBox.extModule);
    unsigned implIdx = 0;
    for (auto path : paths) {
      builder.setInsertionPointAfter(blackBox.extModule);

      // Get the list of ports from the original extmodule, and update the
      // annotations such that they no longer contain any data/mem taps.
      auto ports = blackBox.extModule.getPorts();
      for (auto port : llvm::zip(ports, blackBox.filteredPortAnnos)) {
        std::get<0>(port).annotations = std::get<1>(port);
      }

      // Create a new firrtl.module that implements the data tap.
      auto name =
          StringAttr::get(&getContext(), Twine(blackBox.extModule.getName()) +
                                             "_impl_" + Twine(implIdx++));
      LLVM_DEBUG(llvm::dbgs()
                 << "Implementing " << name << " ("
                 << blackBox.extModule.getName() << " for " << path << ")\n");
      auto impl =
          builder.create<FModuleOp>(name, ports, blackBox.filteredModuleAnnos);
      SymbolTable::setSymbolVisibility(
          impl, SymbolTable::getSymbolVisibility(blackBox.extModule));

      // If extraction information was provided via an
      // `ExtractGrandCentralAnnotation`, put the created data or memory taps
      // inside this directory.
      if (maybeExtractDirectory)
        impl->setAttr("output_file",
                      hw::OutputFileAttr::getFromDirectoryAndFilename(
                          &getContext(), maybeExtractDirectory.getValue(),
                          impl.getName() + ".sv"));
      impl->setAttr("comment",
                    builder.getStringAttr("VCS coverage exclude_file"));
      builder.setInsertionPointToEnd(impl.getBodyBlock());

      // Connect the output ports to the appropriate tapped object.
      for (auto port : portWiring) {
        LLVM_DEBUG(llvm::dbgs() << "- Wiring up port " << port.portNum << "\n");

        // Ignore the port if it is marked for deletion.
        if (port.zeroWidth)
          continue;

        // Handle literals. We send the literal string off to the FIRParser to
        // translate into whatever ops are necessary. This yields a handle on
        // value we're supposed to drive.
        if (port.literal) {
          LLVM_DEBUG(llvm::dbgs() << "  - Connecting literal "
                                  << port.literal.value << "\n");
          auto literal =
              builder.create<ConstantOp>(port.literal.type, port.literal.value);
          auto arg = impl.getArgument(port.portNum);
          builder.create<ConnectOp>(arg, literal);
          continue;
        }
        // The code tries to come up with a relative path from the data tap
        // instance (path being the absolute path to that instance) to the
        // tapped thing (prefix being the path to the tapped port, wire, or
        // memory) by calling stripCommonPrefix(prefix, path).  If the tapped
        // thing includes an NLA, then the NLA path is appended to the rest of
        // the path before the common prefix stripping is done.

        // Determine the shortest hierarchical prefix from this black box
        // instance to the tapped object.
        Optional<SmallVector<HWInstanceLike>> shortestPrefix;
        for (auto prefix : port.prefices) {

          // Append the NLA path to the instance graph-determined path.
          SmallVector<HWInstanceLike> prefixWithNLA(prefix.begin(),
                                                    prefix.end());
          if (port.nla) {
            for (auto segment : port.nla.getNamepath().getValue().drop_back())
              if (auto ref = segment.dyn_cast<InnerRefAttr>()) {
                prefixWithNLA.push_back(
                    cast<HWInstanceLike>(innerRefNS.lookupOp(ref)));
              }
          }

          auto relative = stripCommonPrefix(prefixWithNLA, path);
          if (!shortestPrefix || relative.size() < shortestPrefix->size())
            shortestPrefix.emplace(relative.begin(), relative.end());
        }
        if (!shortestPrefix) {
          LLVM_DEBUG(llvm::dbgs() << "  - Has no prefix, skipping\n");
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Shortest prefix " << *shortestPrefix << "\n");

        // Determine the module at which the hierarchical name should start.
        FModuleLike rootModule;
        if (shortestPrefix->empty()) {
          if (port.target.hasPort())
            rootModule = cast<FModuleLike>(port.target.getOp());
          else
            rootModule = port.target.getOp()->getParentOfType<FModuleLike>();
        } else
          rootModule = shortestPrefix->front()->getParentOfType<FModuleLike>();

        SmallVector<Attribute> symbols;
        SmallString<128> hname;
        auto addSymbol = [&](Attribute symbol) {
          auto id = symbols.size();
          symbols.push_back(symbol);
          if (!hname.empty())
            hname += '.';
          ("{{" + Twine(id) + "}}").toVector(hname);
        };

        // Concatenate the prefix into a proper full hierarchical name.
        addSymbol(
            FlatSymbolRefAttr::get(SymbolTable::getSymbolName(rootModule)));
        for (auto inst : *shortestPrefix)
          addSymbol(getInnerRefTo(inst));
        FIRRTLBaseType tpe;
        if (port.target.getOp()) {
          Attribute leaf;
          if (port.target.hasPort()) {
            leaf = getInnerRefTo(port.target.getOp(), port.target.getPort());
            tpe = cast<FModuleLike>(port.target.getOp())
                      .getPortType(port.target.getPort())
                      .cast<FIRRTLBaseType>();
          } else {
            leaf = getInnerRefTo(port.target.getOp());
            tpe = port.target.getOp()
                      ->getResult(0)
                      .getType()
                      .cast<FIRRTLBaseType>();
          }
          addSymbol(leaf);
        }
        auto fieldID = port.targetFieldID;
        while (fieldID) {
          TypeSwitch<FIRRTLType>(tpe)
              .template Case<FVectorType>([&](FVectorType vector) {
                unsigned index = vector.getIndexForFieldID(fieldID);
                tpe = vector.getElementType();
                fieldID -= vector.getFieldID(index);
                hname += ("[" + Twine(index) + "]").str();
              })
              .template Case<BundleType>([&](BundleType bundle) {
                unsigned index = bundle.getIndexForFieldID(fieldID);
                tpe = bundle.getElementType(index);
                fieldID -= bundle.getFieldID(index);
                // FIXME: Invalid verilog names (e.g. "begin", "reg", .. ) will
                // be renamed at ExportVerilog so the path constructed here
                // might become invalid. We can use an inner name ref to encode
                // a reference to a subfield.

                hname += "." + bundle.getElement(index).name.getValue().str();
              })
              .Default([&](auto) {
                blackBox.extModule.emitError()
                    << "ReferenceDataTapKey on port has invalid field ID";
                signalPassFailure();
                fieldID = 0;
              });
        }

        if (!port.suffix.empty()) {
          hname += '.';
          hname += port.suffix;
        }
        LLVM_DEBUG({
          llvm::dbgs() << "  - Connecting as " << hname;
          if (!symbols.empty()) {
            llvm::dbgs() << " (";
            llvm::interleave(
                symbols, llvm::dbgs(),
                [&](Attribute sym) { llvm::dbgs() << sym; }, ".");
            llvm::dbgs() << ")";
          }
          llvm::dbgs() << "\n";
        });

        // Add a verbatim op that assigns this module port.
        auto arg = impl.getArgument(port.portNum);
        auto hnameExpr = builder.create<VerbatimExprOp>(
            arg.getType().cast<FIRRTLType>(), hname, ValueRange{}, symbols);
        builder.create<ConnectOp>(arg, hnameExpr);
      }

      // Switch the instance from the original extmodule to this
      // implementation. CAVEAT: If the same black box data tap module is
      // instantiated in a parent module that itself is instantiated in
      // different locations, this will pretty arbitrarily pick one of those
      // locations.
      path.back()->setAttr("moduleName", SymbolRefAttr::get(name));
    }

    // Drop the original black box module.
    blackBox.extModule.erase();
  }
}

/// Gather the annotations on ports and operations into the `tappedPorts` and
/// `tappedOps` maps.
void GrandCentralTapsPass::gatherAnnotations(Operation *op) {
  if (isa<FModuleOp, FExtModuleOp>(op)) {
    // Handle port annotations on module/extmodule ops.
    auto gather = [&](unsigned argNum, Annotation anno) {
      if (anno.isClass(referenceKeySourceClass)) {
        gatherTap(anno, Port{op, argNum});
        return true;
      }
      return false;
    };
    AnnotationSet::removePortAnnotations(op, gather);

    // Handle internal data taps.
    // Note that these work for both extmodules AND regular modules.
    // Note also that we do NOT currently check that the String target of an
    // internalKeySourceClass actually corresponds to anything in regular
    // modules.
    if (isa<FModuleOp, FExtModuleOp>(op)) {
      auto gather = [&](Annotation anno) {
        if (anno.isClass(internalKeySourceClass)) {
          gatherTap(anno, op);
          return true;
        }
        return false;
      };
      AnnotationSet::removeAnnotations(op, gather);
    }

    return;
  }

  // Go through all annotations on this op and extract the interesting
  // ones. Note that the way tap annotations are scattered to their
  // targets, we should never see multiple values or memories annotated
  // with the exact same annotation (hence the asserts).
  AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
    if (anno.isClass(memTapSourceClass, referenceKeySourceClass)) {
      gatherTap(anno, op);
      return true;
    }
    return false;
  });
}

void GrandCentralTapsPass::processAnnotation(AnnotatedPort &portAnno,
                                             AnnotatedExtModule &blackBox,
                                             InstancePathCache &instancePaths) {
  LLVM_DEBUG(llvm::dbgs() << "- Processing port " << portAnno.portNum
                          << " anno " << portAnno.anno.getDict() << "\n");
  auto key = getKey(portAnno.anno);
  auto portName = blackBox.extModule.getPortNameAttr(portAnno.portNum);
  PortWiring wiring;
  wiring.portNum = portAnno.portNum;

  // Lookup the sibling annotation no the target. This may not exist, e.g. in
  // the case of a `LiteralDataTapKey`, in which use the annotation on the
  // data tap module port again.
  auto targetAnnoIt = annos.find(key);
  if (portAnno.anno.isClass(memTapPortClass) && targetAnnoIt == annos.end())
    targetAnnoIt = annos.find({key.first, {}});
  auto targetAnno =
      targetAnnoIt != annos.end() ? targetAnnoIt->second : portAnno.anno;
  LLVM_DEBUG(llvm::dbgs() << "  Target anno " << targetAnno.getDict() << "\n");

  // NOTE:
  // - portAnno holds the "*.port" flavor of the annotation
  // - targetAnno holds the "*.source" flavor of the annotation

  // If the annotation is non-local, look up the corresponding NLA operation to
  // find the exact instance path. We basically make the `wiring.prefices` array
  // of instance paths list all the possible paths to the beginning of the NLA
  // path. During wiring of the ports, we generate hierarchical names of the
  // form `<prefix>.<nla-path>.<suffix>`. If we don't have an NLA, we leave it
  // to the key-class-specific code below to come up with the possible prefices.
  HierPathOp nla;
  if (auto nlaSym = targetAnno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
    nla = dyn_cast<HierPathOp>(circuitSymbols->lookup(nlaSym.getAttr()));
    assert(nla);
    // Find all paths to the root of the NLA.
    Operation *root = circuitSymbols->lookup(nla.root());
    wiring.nla = nla;
    wiring.prefices = instancePaths.getAbsolutePaths(root);
  }

  wiring.targetFieldID = targetAnno.getFieldID();
  if (portAnno.anno.getFieldID()) {
    blackBox.extModule.emitError(
        "external module ports must have ground types");
    signalPassFailure();
  }

  // Handle data taps on signals and ports.
  if (targetAnno.isClass(referenceKeySourceClass)) {
    // Handle ports.
    if (auto port = tappedPorts.lookup(key)) {
      if (!nla)
        wiring.prefices = instancePaths.getAbsolutePaths(port.first);
      wiring.target = PortWiring::Target(port.first, port.second);
      portWiring.push_back(std::move(wiring));
      return;
    }

    // Handle operations.
    if (auto op = tappedOps.lookup(key)) {
      // We require the target to be a wire or node, such that it gets a name
      // during Verilog emission.
      if (!isa<WireOp, NodeOp, RegOp, RegResetOp>(op)) {
        auto diag = blackBox.extModule.emitError("ReferenceDataTapKey on port ")
                    << portName << " must be a wire, node, or reg";
        diag.attachNote(op->getLoc()) << "referenced operation is here:";
        signalPassFailure();
        return;
      }

      if (!nla)
        wiring.prefices =
            instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
      wiring.target = PortWiring::Target(op);

      // If the tapped operation is trivially driven by a constant, set
      // information about the literal so that this can later be used instead of
      // an XMR.
      if (auto driver = getDriverFromConnect(op->getResult(0)))
        if (auto constant =
                dyn_cast_or_null<ConstantOp>(driver.getDefiningOp()))
          wiring.literal = {constant.getValueAttr(), constant.getType()};

      portWiring.push_back(std::move(wiring));
      return;
    }

    // If the port is zero-width, then mark it as
    if (zeroWidthTaps.contains(key)) {
      wiring.zeroWidth = true;
      portWiring.push_back(std::move(wiring));
      return;
    }

    // The annotation scattering must have placed this annotation on some
    // target operation or block argument, which we should have picked up in
    // the tapped args or ops maps.
    blackBox.extModule.emitOpError(
        "ReferenceDataTapKey annotation was not scattered to "
        "an operation: ")
        << targetAnno.getDict();
    signalPassFailure();
    return;
  }

  // Handle data taps on black boxes.
  if (targetAnno.isClass(internalKeySourceClass)) {
    auto op = tappedOps.lookup(key);
    if (!op) {
      blackBox.extModule.emitOpError(
          "DataTapModuleSignalKey annotation was not scattered to "
          "an operation: ")
          << targetAnno.getDict();
      signalPassFailure();
      return;
    }

    // Extract the internal path we're supposed to append.
    auto internalPath = targetAnno.getMember<StringAttr>("internalPath");
    if (!internalPath) {
      blackBox.extModule.emitError("DataTapModuleSignalKey annotation on port ")
          << portName << " missing \"internalPath\" attribute";
      signalPassFailure();
      return;
    }

    if (!nla)
      wiring.prefices = instancePaths.getAbsolutePaths(op);
    wiring.suffix = internalPath.getValue();
    portWiring.push_back(std::move(wiring));
    return;
  }

  // Handle data taps with literals.
  if (targetAnno.isClass(literalKeyClass)) {
    auto literal = portAnno.anno.getMember<StringAttr>("literal");
    if (!literal) {
      blackBox.extModule.emitError("LiteralDataTapKey annotation on port ")
          << portName << " missing \"literal\" attribute";
      signalPassFailure();
      return;
    }

    // Parse the literal.
    auto parsed =
        parseIntegerLiteral(blackBox.extModule.getContext(), literal.getValue(),
                            blackBox.extModule.getLoc());
    if (failed(parsed)) {
      blackBox.extModule.emitError("LiteralDataTapKey annotation on port ")
          << portName << " has invalid literal \"" << literal.getValue()
          << "\"";
      signalPassFailure();
      return;
    }

    wiring.literal = *parsed;
    portWiring.push_back(std::move(wiring));
    return;
  }

  // Handle memory taps.
  if (targetAnno.isClass(memTapSourceClass)) {

    // Handle operations.
    if (auto *op = tappedOps.lookup(key)) {
      // We require the target to be a wire or node, such that it gets a name
      // during Verilog emission.
      if (!isa<WireOp, NodeOp, RegOp, RegResetOp>(op)) {
        auto diag = blackBox.extModule.emitError("MemTapAnnotation on port ")
                    << portName << " must be a wire, node, or reg";
        diag.attachNote(op->getLoc()) << "referenced operation is here:";
        signalPassFailure();
        return;
      }

      if (!nla)
        wiring.prefices =
            instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
      wiring.target = PortWiring::Target(op);
      portWiring.push_back(std::move(wiring));
      return;
    }
    // This handles the case when the memtap is on a MemOp, which shouldn't have
    // the PortID attribute. Lookup the op without the portID key.
    if (auto *op = tappedOps.lookup({key.first, {}})) {

      // Extract the memory location we're supposed to access.
      auto word = portAnno.anno.getMember<IntegerAttr>("portID");
      if (!word) {
        blackBox.extModule.emitError("MemTapAnnotation annotation on port ")
            << portName << " missing \"word\" attribute";
        signalPassFailure();
        return;
      }
      // Formulate a hierarchical reference into the memory.
      // CAVEAT: This just assumes that the memory will map to something that
      // can be indexed in the final Verilog. If the memory gets turned into
      // an instance of some sort, we lack the information necessary to go in
      // and access individual elements of it. This will break horribly since
      // we generate memory impls out-of-line already, and memories coming
      // from an external generator are even worse. This needs a special node
      // in the IR that can properly inject the memory array on emission.
      if (!nla)
        wiring.prefices =
            instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
      wiring.target = PortWiring::Target(op);
      ("Memory[" + Twine(word.getValue().getLimitedValue()) + "]")
          .toVector(wiring.suffix);
      portWiring.push_back(std::move(wiring));
      return;
    }
    blackBox.extModule.emitOpError(
        "MemTapAnnotation annotation was not scattered to "
        "an operation: ")
        << targetAnno.getDict();
    signalPassFailure();
    return;
  }

  // We never arrive here since the above code that populates the portAnnos
  // list only adds annotations that we handle in one of the if statements
  // above.
  llvm_unreachable("portAnnos is never populated with unsupported annos");
}

InnerRefAttr GrandCentralTapsPass::getInnerRefTo(Operation *op) {
  return ::getInnerRefTo(op, "gct_sym",
                         [&](FModuleOp mod) -> ModuleNamespace & {
                           return getModuleNamespace(mod);
                         });
}

InnerRefAttr GrandCentralTapsPass::getInnerRefTo(FModuleLike module,
                                                 size_t portIdx) {
  return ::getInnerRefTo(module, portIdx, "gct_sym",
                         [&](FModuleLike mod) -> ModuleNamespace & {
                           return getModuleNamespace(mod);
                         });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralTapsPass() {
  return std::make_unique<GrandCentralTapsPass>();
}
