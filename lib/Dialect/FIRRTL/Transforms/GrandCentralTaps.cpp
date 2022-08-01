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

// Allow printing of `PortWiring` through `<<`.
template <typename T>
static T &operator<<(T &os, PortWiring w) {
  os << "PortWiring(";
  //  os << w.target << ",";
  bool first = true;
  for (auto path : w.prefices) {
    os << path;
    if (w.nla)
      os << "." << w.nla; //.getNamepathAttr();
    if (!w.suffix.empty())
      os << " $ " << w.suffix;
    if (first)
      first = false;
    else
      os << ",";
  }
  os << ")";

  return os;
}

/// Map an annotation to a `Key`.
static Key getKey(Annotation anno) {
  auto id = anno.getMember("id");
  auto tapID = anno.getMember("tapID");
  return {id, tapID};
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
LogicalResult circt::firrtl::applyGCTDataTaps(const AnnoPathValue &target,
                                              DictionaryAttr anno,
                                              ApplyState &state) {

  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  auto id = state.newID();

  NamedAttrList attrs;

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

    // The "wireName" field is common across all sub-types of DataTapKey.
    NamedAttrList wire;
    auto wireNameAttr =
        tryGetAs<StringAttr>(bDict, anno, "wireName", loc, dataTapsClass, path);
    if (!wireNameAttr)
      return failure();
    auto wireTarget = canonicalizeTarget(wireNameAttr.getValue());
    if (!tokenizePath(wireTarget))
      return failure();

    if (classAttr.getValue() == referenceKeyClass) {
      NamedAttrList source;
      auto tapID = state.newID();
      source.append("class", StringAttr::get(context, referenceKeySourceClass));
      source.append("id", id);
      source.append("tapID", tapID);
      auto sourceAttr =
          tryGetAs<StringAttr>(bDict, anno, "source", loc, dataTapsClass, path);
      if (!sourceAttr)
        return failure();
      auto sourceTarget = canonicalizeTarget(sourceAttr.getValue());
      if (!tokenizePath(sourceTarget))
        return failure();

      source.append("target", StringAttr::get(context, sourceTarget));

      state.addToWorklistFn(DictionaryAttr::get(context, source));

      // Annotate the data tap wire.
      wire.append("class", StringAttr::get(context, referenceKeyWireClass));
      wire.append("id", id);
      wire.append("tapID", tapID);
      wire.append("target", StringAttr::get(context, wireTarget));
      state.addToWorklistFn(DictionaryAttr::getWithSorted(context, wire));
      continue;
    }

    if (classAttr.getValue() == internalKeyClass) {
      NamedAttrList module;
      auto tapID = state.newID();
      module.append("class", StringAttr::get(context, internalKeySourceClass));
      module.append("id", id);
      auto internalPathAttr = tryGetAs<StringAttr>(bDict, anno, "internalPath",
                                                   loc, dataTapsClass, path);
      auto moduleAttr =
          tryGetAs<StringAttr>(bDict, anno, "module", loc, dataTapsClass, path);
      if (!internalPathAttr || !moduleAttr)
        return failure();
      module.append("internalPath", internalPathAttr);
      module.append("tapID", tapID);
      auto moduleTarget = canonicalizeTarget(moduleAttr.getValue());
      if (!tokenizePath(moduleTarget))
        return failure();

      module.append("target", StringAttr::get(context, moduleTarget));
      state.addToWorklistFn(DictionaryAttr::getWithSorted(context, module));

      // Annotate the data tap module port.
      wire.append("class", StringAttr::get(context, internalKeyPortClass));
      wire.append("id", id);
      wire.append("tapID", tapID);
      wire.append("target", StringAttr::get(context, wireTarget));
      state.addToWorklistFn(DictionaryAttr::get(context, wire));
      continue;
    }

    if (classAttr.getValue() == deletedKeyClass) {
      // Annotate the data tap module port.
      wire.append("class", classAttr);
      wire.append("id", id);
      wire.append("target", StringAttr::get(context, wireTarget));
      state.addToWorklistFn(DictionaryAttr::get(context, wire));
      continue;
    }

    if (classAttr.getValue() == literalKeyClass) {
      auto literalAttr = tryGetAs<StringAttr>(bDict, anno, "literal", loc,
                                              dataTapsClass, path);
      if (!literalAttr)
        return failure();

      // Annotate the data tap module port.
      wire.append("class", classAttr);
      wire.append("id", id);
      wire.append("literal", literalAttr);
      wire.append("target", StringAttr::get(context, wireTarget));
      state.addToWorklistFn(DictionaryAttr::get(context, wire));
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
  llvm::Optional<PortWiring>
  processAnnotation(const Annotation &sinkAnno, WireOp sinkOp,
                    InstancePathCache &instancePaths);

  // Helpers to simplify collecting taps on the various things.
  void gatherTap(Annotation anno, Port port) {
    auto key = getKey(anno);
    annos.insert({key, anno});
    assert(!tappedPorts.count(key) && "ambiguous tap annotation");
    auto portWidth = cast<FModuleLike>(port.first)
                         .getPortType(port.second)
                         .cast<FIRRTLBaseType>()
                         .getBitWidthOrSentinel();
    // If the port width is non-zero, process it normally.  Otherwise, record it
    // as being zero-width.
    if (portWidth)
      tappedPorts.insert({key, port});
    else
      zeroWidthTaps.insert(key);
    if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
      deadNLAs.insert(sym.getAttr());
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
    if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
      deadNLAs.insert(sym.getAttr());
  }
  // Helper to get a sink annotation from a wire if it exists
  llvm::Optional<Annotation> getSinkAnno(WireOp wire) {
    llvm::Optional<Annotation> result =
        llvm::Optional<Annotation>(); // How do I say "None"?
    auto gather = [&](Annotation anno) {
      if (anno.isClass(referenceKeyWireClass)) {
        assert(!result.hasValue() &&
               "A Wire should only have 1 sink annotation!");
        result = anno;
        return true;
      }
      return false;
    };
    AnnotationSet::removeAnnotations(wire, gather);

    return result;
  }

  void findAndDeleteDriver(WireOp wire) {
    StrictConnectOp connect;
    for (auto *user : wire->getUsers()) {
      if (auto op = dyn_cast<StrictConnectOp>(user)) {
        if (op.getDest() == wire) {
          connect = op;
          break;
        }
      }
    }
    if (connect) {
      LLVM_DEBUG(llvm::dbgs() << "Deleting " << connect << "\n");
      connect.erase();
    } else {
      // Should we just error here?
      LLVM_DEBUG(llvm::dbgs() << "No driver found for " << wire << "\n");
    }
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
  // TODO should we keep the Annotation here or just the Key?
  SmallVector<std::pair<WireOp, Annotation>, 8> tapSinks;

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
    llvm::dbgs() << "Tap sinks:\n";
    for (auto it : tapSinks) {
      auto key = getKey(it.second);
      llvm::dbgs() << "- " << key << ": " << it.first << "\n";
    }
  });

  // TODO should we group these by Module?
  for (auto sinkPair : tapSinks) {
    auto wire = sinkPair.first;
    auto sinkAnno = sinkPair.second;

    auto module = wire->getParentOfType<FModuleOp>();

    LLVM_DEBUG({
      llvm::dbgs() << "Wiring up sink '" << wire << "' in module"
                   << module.moduleName() << "\n"
                   << "- " << sinkAnno.getDict() << "\n";
    });

    // TODO is it bad to make a builder for a module a bunch of times?
    ImplicitLocOpBuilder builder(module->getLoc(), module);
    builder.setInsertionPointToEnd(module.getBodyBlock());

    // Wire the sink up
    auto wiringOpt = processAnnotation(sinkAnno, wire, instancePaths);
    if (!wiringOpt) {
      wire->emitError("No Tap source found for this wire!");
      return signalPassFailure();
    }
    auto wiring = wiringOpt.value();

    // Connect the output ports to the appropriate tapped object.
    LLVM_DEBUG(llvm::dbgs() << "- Wiring up " << wiring << "\n");

    // Ignore the port if it is marked for deletion.
    if (wiring.zeroWidth)
      continue;

    // Handle literals. We send the literal string off to the FIRParser to
    // translate into whatever ops are necessary. This yields a handle on
    // value we're supposed to drive.
    if (wiring.literal) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  - Connecting literal " << wiring.literal.value << "\n");
      auto literal =
          builder.create<ConstantOp>(wiring.literal.type, wiring.literal.value);
      findAndDeleteDriver(wire);
      builder.create<StrictConnectOp>(wire, literal);
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
    Optional<SmallVector<InstanceOp>> shortestPrefix;
    for (auto prefix : wiring.prefices) {

      // Append the NLA path to the instance graph-determined path.
      SmallVector<InstanceOp> prefixWithNLA(prefix.begin(), prefix.end());
      if (wiring.nla) {
        for (auto segment : wiring.nla.getNamepath().getValue().drop_back())
          if (auto ref = segment.dyn_cast<InnerRefAttr>()) {
            prefixWithNLA.push_back(cast<InstanceOp>(innerRefNS.lookupOp(ref)));
          }
      }
      auto path = instancePaths.getAbsolutePaths(
          module)[0]; // Should this be memoized above? How do we deal with
                      // multiple paths?

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
      if (wiring.target.hasPort())
        rootModule = cast<FModuleLike>(wiring.target.getOp());
      else
        rootModule = wiring.target.getOp()->getParentOfType<FModuleLike>();
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
    addSymbol(FlatSymbolRefAttr::get(SymbolTable::getSymbolName(rootModule)));
    for (auto inst : *shortestPrefix)
      addSymbol(getInnerRefTo(inst));

    if (wiring.target.getOp()) {
      Attribute leaf;
      if (wiring.target.hasPort())
        leaf = getInnerRefTo(wiring.target.getOp(), wiring.target.getPort());
      else
        leaf = getInnerRefTo(wiring.target.getOp());
      addSymbol(leaf);
    }

    if (!wiring.suffix.empty()) {
      hname += '.';
      hname += wiring.suffix;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "  - Connecting as " << hname;
      if (!symbols.empty()) {
        llvm::dbgs() << " (";
        llvm::interleave(
            symbols, llvm::dbgs(), [&](Attribute sym) { llvm::dbgs() << sym; },
            ".");
        llvm::dbgs() << ")";
      }
      llvm::dbgs() << "\n";
    });

    // Add a verbatim op that assigns the sink.
    auto hnameExpr = builder.create<VerbatimExprOp>(
        wire.getType().cast<FIRRTLType>(), hname, ValueRange{}, symbols);
    findAndDeleteDriver(wire);
    builder.create<StrictConnectOp>(wire, hnameExpr);
  }

  // Garbage collect NLAs which were removed.
  for (auto &op :
       llvm::make_early_inc_range(circuitOp.getBodyBlock()->getOperations())) {
    // Remove NLA anchors whose leaf annotations were removed.
    if (auto nla = dyn_cast<HierPathOp>(op)) {
      if (deadNLAs.contains(nla.getNameAttr()))
        nla.erase();
      continue;
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

  // Collet all of the sinks which are always on Wires
  if (isa<WireOp>(op)) {
    auto gather = [&](Annotation anno) {
      if (anno.isClass(referenceKeyWireClass)) {
        tapSinks.push_back({dyn_cast<WireOp>(op), anno});
        return true;
      }
      return false;
    };
    AnnotationSet::removeAnnotations(op, gather);
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

llvm::Optional<PortWiring>
GrandCentralTapsPass::processAnnotation(const Annotation &sinkAnno,
                                        WireOp sinkOp,
                                        InstancePathCache &instancePaths) {

  auto key = getKey(sinkAnno);

  PortWiring wiring;
  wiring.portNum = 0;

  // // Lookup the sibling annotation no the target. This may not exist, e.g.
  // in
  // // the case of a `LiteralDataTapKey`, in which use the annotation on the
  // // data tap module port again.
  // auto targetAnnoIt = annos.find(key);
  // if (portAnno.anno.isClass(memTapPortClass) && targetAnnoIt ==
  // annos.end())
  //   targetAnnoIt = annos.find({key.first, {}});
  // auto targetAnno =
  //     targetAnnoIt != annos.end() ? targetAnnoIt->second : portAnno.anno;

  // NOTE: target anno renamed to sourceAnno
  auto sourceAnno = annos[key];
  LLVM_DEBUG(llvm::dbgs() << "  Target anno " << sourceAnno.getDict() << "\n");

  // NOTE:
  // - portAnno holds the "*.port" flavor of the annotation
  // - targetAnno holds the "*.source" flavor of the annotation
  // - New is below
  // - sinkAnno holds the ".wire" flavor of the annotation
  // - sourceAnno holds the "*.source" flavor of the annotation

  // If the annotation is non-local, look up the corresponding NLA operation
  // to find the exact instance path. We basically make the `wiring.prefices`
  // array of instance paths list all the possible paths to the beginning of
  // the NLA path. During wiring of the ports, we generate hierarchical names
  // of the form `<prefix>.<nla-path>.<suffix>`. If we don't have an NLA, we
  // leave it to the key-class-specific code below to come up with the
  // possible prefices.
  HierPathOp nla;
  if (auto nlaSym = sourceAnno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
    nla = dyn_cast<HierPathOp>(circuitSymbols->lookup(nlaSym.getAttr()));
    assert(nla);
    // Find all paths to the root of the NLA.
    Operation *root = circuitSymbols->lookup(nla.root());
    wiring.nla = nla;
    wiring.prefices = instancePaths.getAbsolutePaths(root);
  }

  // Handle data taps on signals and ports.
  if (sourceAnno.isClass(referenceKeySourceClass)) {
    // Handle ports.
    if (auto port = tappedPorts.lookup(key)) {
      if (!nla)
        wiring.prefices = instancePaths.getAbsolutePaths(port.first);
      wiring.target = PortWiring::Target(port.first, port.second);
      return wiring;
    }

    // Handle operations.
    if (auto op = tappedOps.lookup(key)) {
      // We require the target to be a wire or node, such that it gets a name
      // during Verilog emission.
      if (!isa<WireOp, NodeOp, RegOp, RegResetOp>(op)) {
        // auto diag = blackBox.extModule.emitError("ReferenceDataTapKey on
        // port
        // ")
        //             << portName << " must be a wire, node, or reg";
        auto diag = op->emitError("Tap must be a wire, node or reg");
        diag.attachNote(op->getLoc()) << "referenced operation is here:";
        signalPassFailure();
        return None;
      }

      if (!nla)
        wiring.prefices =
            instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
      wiring.target = PortWiring::Target(op);

      // If the tapped operation is trivially driven by a constant, set
      // information about the literal so that this can later be used instead
      // of an XMR.
      if (auto driver = getDriverFromConnect(op->getResult(0)))
        if (auto constant =
                dyn_cast_or_null<ConstantOp>(driver.getDefiningOp()))
          wiring.literal = {constant.getValueAttr(), constant.getType()};

      return wiring;
    }

    // If the port is zero-width, then mark it as
    if (zeroWidthTaps.contains(key)) {
      wiring.zeroWidth = true;
      return wiring;
    }

    // The annotation scattering must have placed this annotation on some
    // target operation or block argument, which we should have picked up in
    // the tapped args or ops maps.
    // blackBox.extModule.emitOpError(
    //     "ReferenceDataTapKey annotation was not scattered to "
    //     "an operation: ")
    //     << sourceAnno.getDict();
    // TODO add some error reporting
    signalPassFailure();
    return wiring;
  }

  // Handle data taps on black boxes.
  if (sourceAnno.isClass(internalKeySourceClass)) {
    auto op = tappedOps.lookup(key);
    if (!op) {
      // TODO fix
      // blackBox.extModule.emitOpError(
      //     "DataTapModuleSignalKey annotation was not scattered to "
      //     "an operation: ")
      //     << sourceAnno.getDict();
      signalPassFailure();
      return None;
    }

    // Extract the internal path we're supposed to append.
    auto internalPath = sourceAnno.getMember<StringAttr>("internalPath");
    if (!internalPath) {
      // TODO fix
      // blackBox.extModule.emitError("DataTapModuleSignalKey annotation on
      // port
      // ")
      //     << portName << " missing \"internalPath\" attribute";
      signalPassFailure();
      return None;
    }

    if (!nla)
      wiring.prefices = instancePaths.getAbsolutePaths(op);
    wiring.suffix = internalPath.getValue();
    //    portWiring.push_back(std::move(wiring));
    return wiring;
  }

  // Handle data taps with literals.
  if (sourceAnno.isClass(literalKeyClass)) {
    auto literal = sourceAnno.getMember<StringAttr>("literal");
    if (!literal) {
      // TODO fix
      // blackBox.extModule.emitError("LiteralDataTapKey annotation on port ")
      //     << portName << " missing \"literal\" attribute";
      signalPassFailure();
      return None;
    }

    // Parse the literal.
    auto parsed = parseIntegerLiteral(sinkOp->getContext(), literal.getValue(),
                                      sinkOp.getLoc());
    if (failed(parsed)) {
      // TODO fix
      // blackBox.extModule.emitError("LiteralDataTapKey annotation on port ")
      //     << portName << " has invalid literal \"" << literal.getValue()
      //     << "\"";
      signalPassFailure();
      return None;
    }

    wiring.literal = *parsed;
    return wiring;
  }

  // TODO fix
  // // Handle memory taps.
  // if (sourceAnno.isClass(memTapSourceClass)) {

  //   // Handle operations.
  //   if (auto *op = tappedOps.lookup(key)) {
  //     // We require the target to be a wire or node, such that it gets a
  //     name
  //     // during Verilog emission.
  //     if (!isa<WireOp, NodeOp, RegOp, RegResetOp>(op)) {
  //       auto diag = blackBox.extModule.emitError("MemTapAnnotation on port
  //       ")
  //                   << portName << " must be a wire, node, or reg";
  //       diag.attachNote(op->getLoc()) << "referenced operation is here:";
  //       signalPassFailure();
  //       return;
  //     }

  //     if (!nla)
  //       wiring.prefices =
  //           instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
  //     wiring.target = PortWiring::Target(op);
  //     portWiring.push_back(std::move(wiring));
  //     return;
  //   }
  //   // This handles the case when the memtap is on a MemOp, which shouldn't
  //   have
  //   // the PortID attribute. Lookup the op without the portID key.
  //   if (auto *op = tappedOps.lookup({key.first, {}})) {

  //     // Extract the memory location we're supposed to access.
  //     auto word = portAnno.anno.getMember<IntegerAttr>("portID");
  //     if (!word) {
  //       blackBox.extModule.emitError("MemTapAnnotation annotation on port
  //       ")
  //           << portName << " missing \"word\" attribute";
  //       signalPassFailure();
  //       return;
  //     }
  //     // Formulate a hierarchical reference into the memory.
  //     // CAVEAT: This just assumes that the memory will map to something
  //     that
  //     // can be indexed in the final Verilog. If the memory gets turned
  //     into
  //     // an instance of some sort, we lack the information necessary to go
  //     in
  //     // and access individual elements of it. This will break horribly
  //     since
  //     // we generate memory impls out-of-line already, and memories coming
  //     // from an external generator are even worse. This needs a special
  //     node
  //     // in the IR that can properly inject the memory array on emission.
  //     if (!nla)
  //       wiring.prefices =
  //           instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
  //     wiring.target = PortWiring::Target(op);
  //     ("Memory[" + Twine(word.getValue().getLimitedValue()) + "]")
  //         .toVector(wiring.suffix);
  //     portWiring.push_back(std::move(wiring));
  //     return;
  //   }
  //   blackBox.extModule.emitOpError(
  //       "MemTapAnnotation annotation was not scattered to "
  //       "an operation: ")
  //       << sourceAnno.getDict();
  //   signalPassFailure();
  //   return;
  // }

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
