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

#include "../AnnotationDetails.h"
#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
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
using mlir::FailureOr;
using mlir::function_like_impl::getArgAttrDict;
using mlir::function_like_impl::setAllArgAttrDicts;

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
  unsigned portNum;
  ArrayRef<InstancePath> prefices;
  SmallString<16> suffix;
  /// If set, the port should output a constant literal.
  Literal literal;
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

/// Check if an annotation is a `ReferenceDataTapKey`, and that it has a `type`
/// field with a given content.
static bool isReferenceDataTapOfType(Annotation anno, StringRef type) {
  if (!anno.isClass(referenceKeyClass))
    return false;
  auto typeAttr = anno.getMember<StringAttr>("type");
  if (!typeAttr)
    return false;
  return typeAttr.getValue() == type;
}

/// Check if an annotation is a `ReferenceDataTapKey` with `source` type.
static bool isReferenceDataTapSource(Annotation anno) {
  return isReferenceDataTapOfType(anno, "source");
}

/// Check if an annotation is a `ReferenceDataTapKey` with `portName` type.
static bool isReferenceDataTapPortName(Annotation anno) {
  return isReferenceDataTapOfType(anno, "portName");
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
    auto it = tappedPorts.insert({key, port});
    assert(it.second && "ambiguous tap annotation");
  }
  void gatherTap(Annotation anno, Operation *op) {
    auto key = getKey(anno);
    annos.insert({key, anno});
    auto it = tappedOps.insert({key, op});
    assert(it.second && "ambiguous tap annotation");
  }

  DenseMap<Key, Annotation> annos;
  DenseMap<Key, Operation *> tappedOps;
  DenseMap<Key, Port> tappedPorts;
  SmallVector<PortWiring, 8> portWiring;

  /// The name of the directory where data and mem tap modules should be
  /// output.
  StringAttr maybeExtractDirectory = {};
};

void GrandCentralTapsPass::runOnOperation() {
  auto circuitOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Running the GCT Data Taps pass\n");

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
  if (auto dict = circuitAnnotations.getAnnotation(extractGrandCentralClass)) {
    auto directory = dict.getAs<StringAttr>("directory");
    if (!directory) {
      circuitOp->emitError()
          << "contained an invalid 'ExtractGrandCentralAnnotation' that does "
             "not contain a 'directory' field: "
          << dict;
      return signalPassFailure();
    }
    maybeExtractDirectory = directory;
  }

  // Gather a list of extmodules that have data or mem tap annotations to be
  // expanded.
  SmallVector<AnnotatedExtModule, 4> modules;
  for (auto &op : *circuitOp.getBody()) {
    auto extModule = dyn_cast<FExtModuleOp>(&op);
    if (!extModule)
      continue;

    // Go through the module ports and collect the annotated ones.
    AnnotatedExtModule result{extModule, {}, {}, {}};
    result.filteredPortAnnos.reserve(extModule.getNumPorts());
    for (unsigned argNum = 0, e = extModule.getNumPorts(); argNum < e;
         ++argNum) {
      // Go through all annotations on this port and add the data tap key and
      // mem tap ones to the list.
      auto annos = AnnotationSet::forPort(extModule, argNum);
      annos.removeAnnotations([&](Annotation anno) {
        if (anno.isClass(memTapClass, deletedKeyClass, literalKeyClass,
                         internalKeyClass) ||
            isReferenceDataTapPortName(anno)) {
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
        [&](Annotation anno) { return anno.isClass(dataTapsClass); });
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
          llvm::dbgs() << "  - " << path << "." << wiring.suffix << "\n";
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

        // Concatenate the prefix into a proper full hierarchical name.
        SmallString<128> hname;
        for (auto inst : shortestPrefix.getValue()) {
          if (!hname.empty())
            hname += '.';
          hname += inst.name();
        }
        if (!hname.empty())
          hname += '.';
        hname += port.suffix;
        LLVM_DEBUG(llvm::dbgs() << "  - Connecting as " << hname << "\n");

        // Add a verbatim op that assigns this module port.
        auto arg = impl.getArgument(port.portNum);
        auto hnameExpr = builder.create<VerbatimExprOp>(
            arg.getType().cast<FIRRTLType>(), hname);
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
      if (isReferenceDataTapSource(anno)) {
        gatherTap(anno, Port{op, argNum});
        return true;
      }
      return false;
    };
    AnnotationSet::removePortAnnotations(op, gather);

    // Handle internal data taps on extmodule ops.
    if (isa<FExtModuleOp>(op)) {
      auto gather = [&](Annotation anno) {
        if (anno.isClass(internalKeyClass)) {
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
    if (anno.isClass(memTapClass) || isReferenceDataTapSource(anno)) {
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
  PortWiring wiring = {portAnno.portNum, {}, {}, {}};

  // Lookup the sibling annotation no the target. This may not exist, e.g. in
  // the case of a `LiteralDataTapKey`, in which use the annotation on the
  // data tap module port again.
  auto targetAnnoIt = annos.find(key);
  auto targetAnno =
      targetAnnoIt != annos.end() ? targetAnnoIt->second : portAnno.anno;

  // Handle data taps on signals and ports.
  if (targetAnno.isClass(referenceKeyClass)) {
    // Handle ports.
    if (auto port = tappedPorts.lookup(key)) {
      wiring.prefices = instancePaths.getAbsolutePaths(port.first);
      wiring.suffix = cast<FModuleLike>(port.first).getPortName(port.second);
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

      // We currently require the target to be named.
      // TODO: If we were to use proper cross-module reference ops in the IR
      // then this could be anonymous, with ExportVerilog resolving the name
      // at the last moment.
      auto name = op->getAttrOfType<StringAttr>("name");
      if (!name) {
        auto diag =
            op->emitError("declaration targeted by data tap must have a name");
        diag.attachNote(blackBox.extModule->getLoc())
            << "used by ReferenceDataTapKey on port " << portName << " here:";
        signalPassFailure();
        return;
      }

      wiring.prefices =
          instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
      wiring.suffix = name.getValue();
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
  if (targetAnno.isClass(internalKeyClass)) {
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
  if (targetAnno.isClass(memTapClass)) {
    auto op = tappedOps.lookup(key);
    if (!op) {
      blackBox.extModule.emitOpError(
          "MemTapAnnotation annotation was not scattered to "
          "an operation: ")
          << targetAnno.getDict();
      signalPassFailure();
      return;
    }

    // Extract the name of the memory.
    // TODO: This would be better handled through a proper cross-module
    // reference preserved in the IR, such that ExportVerilog can insert a
    // proper name here at the last moment.
    auto name = op->getAttrOfType<StringAttr>("name");
    if (!name) {
      auto diag = op->emitError("target of memory tap must have a name");
      diag.attachNote(blackBox.extModule->getLoc())
          << "used by MemTapAnnotation on port " << portName << " here:";
      signalPassFailure();
      return;
    }

    // Extract the memory location we're supposed to access.
    auto word = portAnno.anno.getMember<IntegerAttr>("word");
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
    wiring.prefices =
        instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
    (Twine(name.getValue()) + ".Memory[" +
     llvm::utostr(word.getValue().getLimitedValue()) + "]")
        .toVector(wiring.suffix);
    portWiring.push_back(std::move(wiring));
    return;
  }

  // We never arrive here since the above code that populates the portAnnos
  // list only adds annotations that we handle in one of the if statements
  // above.
  llvm_unreachable("portAnnos is never populated with unsupported annos");
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralTapsPass() {
  return std::make_unique<GrandCentralTapsPass>();
}
