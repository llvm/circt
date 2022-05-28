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
using hw::InnerRefAttr;
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
  /// An additional string suffix to append to the hierarchical name.
  SmallString<16> suffix;
  /// If set, the port should output a constant literal.
  Literal literal;
  /// The non-local anchor further specifying where to connect.
  NonLocalAnchor nla;

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
LogicalResult circt::firrtl::applyGCTDataTaps(AnnoPathValue target,
                                              DictionaryAttr anno,
                                              ApplyState &state) {

  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  auto dontTouch = [&](StringRef targetStr) {
    NamedAttrList anno;
    anno.append("class", StringAttr::get(context, dontTouchAnnoClass));
    anno.append("target", StringAttr::get(context, targetStr));
    state.addToWorklistFn(DictionaryAttr::get(context, anno));
  };

  auto id = state.newID();
  NamedAttrList attrs;
  attrs.append("class", StringAttr::get(context, dataTapsBlackboxClass));
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
      auto sourceTargetPath = canonicalizeTarget(sourceAttr.getValue());
      auto sourceTarget = tokenizePath(sourceTargetPath);
      if (!sourceTarget)
        return failure();

      // If this refers to a module port, create a "tap" wire to observe the
      // value of the port, which can unblock optimizations by not pointing at
      // the port directly.
      AnnoPathValue path =
          resolvePath(sourceTargetPath, state.circuit, state.symTbl).getValue();
      auto portSourceTarget = path.ref.dyn_cast_or_null<PortAnnoTarget>();
      if (portSourceTarget && isa<FModuleOp>(portSourceTarget.getOp())) {
        auto module = cast<FModuleOp>(portSourceTarget.getOp());
        Value value = module.getArgument(portSourceTarget.getPortNo());
        auto builder = ImplicitLocOpBuilder::atBlockBegin(value.getLoc(),
                                                          module.getBody());
        auto type = portSourceTarget.getType()
                        .getFinalTypeByFieldID(path.fieldIdx)
                        .getPassiveType();
        auto tap = builder.create<WireOp>(
            type, state.getNamespace(module).newName("_gctTap"));
        value = getValueByFieldID(builder, value, path.fieldIdx);
        emitConnect(builder, tap, value);
        sourceTarget->name = tap.name();
        sourceTarget->component.clear(); // resolved in getValueByFieldID
      } else if (!portSourceTarget) {
        ImplicitLocOpBuilder builder(path.ref.getOp()->getLoc(),
                                     path.ref.getOp());
        builder.setInsertionPointAfter(path.ref.getOp());
        Value value = path.ref.getOp()->getResult(0);
        auto type = path.ref.getType()
                        .getFinalTypeByFieldID(path.fieldIdx)
                        .getPassiveType();
        auto tap = builder.create<WireOp>(
            type, state.getNamespace(path.ref.getModule()).newName("_gctTap"));
        value = getValueByFieldID(builder, value, path.fieldIdx);
        emitConnect(builder, tap, value);
        sourceTarget->name = tap.name();
        sourceTarget->component.clear(); // resolved in getValueByFieldID
      }

      sourceTargetPath = sourceTarget->str();
      source.append("target", StringAttr::get(context, sourceTargetPath));

      state.addToWorklistFn(DictionaryAttr::get(context, source));
      dontTouch(sourceTargetPath);

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

LogicalResult circt::firrtl::applyGCTMemTaps(AnnoPathValue target,
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
  state.addToWorklistFn(DictionaryAttr::get(context, attrs));

  auto tapsAttr = tryGetAs<ArrayAttr>(anno, anno, "taps", loc, memTapClass);
  if (!tapsAttr)
    return failure();
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

    auto blackboxTarget = tokenizePath(canonTarget).getValue();
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
    tappedPorts.insert({key, port});
    if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
      deadNLAs.insert(sym.getAttr());
  }
  void gatherTap(Annotation anno, Operation *op) {
    auto key = getKey(anno);
    annos.insert({key, anno});
    assert(!tappedOps.count(key) && "ambiguous tap annotation");
    tappedOps.insert({key, op});
    if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
      deadNLAs.insert(sym.getAttr());
  }

  /// Returns an operation's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(Operation *op);
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
  DenseMap<Key, Port> tappedPorts;
  SmallVector<PortWiring, 8> portWiring;

  /// The name of the directory where data and mem tap modules should be
  /// output.
  StringAttr maybeExtractDirectory = {};

  /// Cached module namespaces.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  /// NLAs which were removed while this pass ran.  These will be garbage
  /// collected before the pass exits.
  DenseSet<StringAttr> deadNLAs;

  /// The circuit symbol table, used to look up NLAs.
  SymbolTable *circuitSymbols;
};

void GrandCentralTapsPass::runOnOperation() {
  auto circuitOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Running the GCT Data Taps pass\n");
  SymbolTable symtbl(circuitOp);
  circuitSymbols = &symtbl;

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
           circuitOp.getBody()->getOps<FExtModuleOp>())) {

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
            llvm::dbgs() << "." << wiring.nla.namepathAttr();
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
      // If extraction information was provided via an
      // `ExtractGrandCentralAnnotation`, put the created data or memory taps
      // inside this directory.
      if (maybeExtractDirectory)
        impl->setAttr("output_file",
                      hw::OutputFileAttr::getFromDirectoryAndFilename(
                          &getContext(), maybeExtractDirectory.getValue(),
                          impl.getName() + ".sv"));
      builder.setInsertionPointToEnd(impl.getBody());

      // Connect the output ports to the appropriate tapped object.
      for (auto port : portWiring) {
        LLVM_DEBUG(llvm::dbgs() << "- Wiring up port " << port.portNum << "\n");

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
        // memory) by calling stripCommonPrefix(prefix, path). Bug 2767 was
        // caused because, the path in the NLA was not considered for this
        // common prefix stripping (prefix is the path to the root of the NLA).
        // This leads to overly pessimistic paths like
        // Top.dut.submodule_1.bar.Memory[0], which should rather be
        // DUT.submodule_1.bar.Memory[0]. To properly fix this, the path in the
        // NLA should be inlined in the prefix such that it becomes part of the
        // prefix stripping operation.

        // Determine the shortest hierarchical prefix from this black box
        // instance to the tapped object.
        Optional<InstancePath> shortestPrefix;
        for (auto prefix : port.prefices) {
          auto relative = stripCommonPrefix(prefix, path);
          if (!shortestPrefix.hasValue() ||
              relative.size() < shortestPrefix->size())
            shortestPrefix = relative;
        }
        if (!shortestPrefix.hasValue()) {
          LLVM_DEBUG(llvm::dbgs() << "  - Has no prefix, skipping\n");
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Shortest prefix " << *shortestPrefix << "\n");

        // Determine the module at which the hierarchical name should start.
        Operation *opInRootModule =
            shortestPrefix->empty() ? path.front() : shortestPrefix->front();
        auto rootModule = opInRootModule->getParentOfType<FModuleLike>();

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
        if (port.nla && shortestPrefix->empty() &&
            port.nla.root() != rootModule.moduleNameAttr()) {
          // This handles the case when nla is not considered for common prefix
          // stripping.
          auto rootMod = port.nla.root();
          for (auto p : path) {
            if (rootMod == p->getParentOfType<FModuleOp>().getNameAttr())
              break;
            addSymbol(getInnerRefTo(p));
          }
        }
        for (auto inst : shortestPrefix.getValue())
          addSymbol(getInnerRefTo(inst));
        if (port.nla) {
          for (auto sym : port.nla.namepath())
            addSymbol(sym);
        } else if (port.target.getOp()) {
          if (port.target.hasPort())
            addSymbol(
                getInnerRefTo(port.target.getOp(), port.target.getPort()));
          else
            addSymbol(getInnerRefTo(port.target.getOp()));
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

  // Garbage collect NLAs which were removed.
  for (auto &op :
       llvm::make_early_inc_range(circuitOp.getBody()->getOperations())) {
    // Remove NLA anchors whose leaf annotations were removed.
    if (auto nla = dyn_cast<NonLocalAnchor>(op)) {
      if (deadNLAs.contains(nla.getNameAttr()))
        nla.erase();
      continue;
    }

    // Remove NLA paths on instances associated with removed NLA
    // leaves.
    auto module = dyn_cast<FModuleOp>(op);
    if (!module)
      continue;
    for (auto instance : module.getBody()->getOps<InstanceOp>()) {
      AnnotationSet annotations(instance);
      if (annotations.empty())
        continue;
      bool modified = false;
      annotations.removeAnnotations([&](Annotation anno) {
        if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
          if (deadNLAs.contains(sym.getAttr()))
            return modified = true;
        return false;
      });
      if (modified)
        annotations.applyToOperation(instance);
    }
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

    // Handle internal data taps on extmodule ops.
    if (isa<FExtModuleOp>(op)) {
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
  NonLocalAnchor nla;
  if (auto nlaSym = targetAnno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
    nla = dyn_cast<NonLocalAnchor>(circuitSymbols->lookup(nlaSym.getAttr()));
    assert(nla);
    // Find all paths to the root of the NLA.
    Operation *root = circuitSymbols->lookup(nla.root());
    wiring.nla = nla;
    wiring.prefices = instancePaths.getAbsolutePaths(root);
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

      // Set the literal member if this wire or node is driven by a constant.
      auto driver = getDriverFromConnect(op->getResult(0));
      if (driver)
        if (auto constant =
                dyn_cast_or_null<ConstantOp>(driver.getDefiningOp())) {
          wiring.literal = {
              IntegerAttr::get(constant.getContext(), constant.value()),
              constant.getType()};
          // Delete any NLAs on the dead wire tap if as we are going to delete
          // the symbol.  This deals with the situation where there is a
          // non-local DontTouchAnnotation.
          for (auto anno : AnnotationSet(op))
            if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
              deadNLAs.insert(sym.getAttr());
          op->removeAttr("inner_sym");
        }

      wiring.target = PortWiring::Target(op);
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

StringAttr GrandCentralTapsPass::getOrAddInnerSym(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("inner_sym");
  if (attr)
    return attr;
  auto module = op->getParentOfType<FModuleOp>();
  auto name = getModuleNamespace(module).newName("gct_sym");
  attr = StringAttr::get(op->getContext(), name);
  op->setAttr("inner_sym", attr);
  return attr;
}

StringAttr GrandCentralTapsPass::getOrAddInnerSym(FModuleLike module,
                                                  size_t portIdx) {
  auto attr = module.getPortSymbolAttr(portIdx);
  if (attr && !attr.getValue().empty())
    return attr;
  auto name = getModuleNamespace(module).newName("gct_sym");
  attr = StringAttr::get(module.getContext(), name);
  module.setPortSymbolAttr(portIdx, attr);
  return attr;
}

InnerRefAttr GrandCentralTapsPass::getInnerRefTo(Operation *op) {
  return InnerRefAttr::get(
      SymbolTable::getSymbolName(op->getParentOfType<FModuleOp>()),
      getOrAddInnerSym(op));
}

InnerRefAttr GrandCentralTapsPass::getInnerRefTo(FModuleLike module,
                                                 size_t portIdx) {
  return InnerRefAttr::get(SymbolTable::getSymbolName(module),
                           getOrAddInnerSym(module, portIdx));
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralTapsPass() {
  return std::make_unique<GrandCentralTapsPass>();
}
