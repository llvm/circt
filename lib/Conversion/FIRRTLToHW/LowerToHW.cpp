//===- LowerToHW.cpp - FIRRTL to HW/SV Lowering Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main FIRRTL to HW/SV Lowering Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"

#define DEBUG_TYPE "lower-to-hw"

namespace circt {
#define GEN_PASS_DEF_LOWERFIRRTLTOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace firrtl;
using circt::comb::ICmpPredicate;

/// Attribute that indicates that the module hierarchy starting at the
/// annotated module should be dumped to a file.
static const char moduleHierarchyFileAttrName[] = "firrtl.moduleHierarchyFile";

/// Return true if the specified type is a sized FIRRTL type (Int or Analog)
/// with zero bits.
static bool isZeroBitFIRRTLType(Type type) {
  auto ftype = dyn_cast<FIRRTLBaseType>(type);
  return ftype && ftype.getPassiveType().getBitWidthOrSentinel() == 0;
}

// Return a single source value in the operands of the given attach op if
// exists.
static Value getSingleNonInstanceOperand(AttachOp op) {
  Value singleSource;
  for (auto operand : op.getAttached()) {
    if (isZeroBitFIRRTLType(operand.getType()) ||
        operand.getDefiningOp<InstanceOp>())
      continue;
    // If it is used by other than attach op or there is already a source
    // value, bail out.
    if (!operand.hasOneUse() || singleSource)
      return {};
    singleSource = operand;
  }
  return singleSource;
}

/// This verifies that the target operation has been lowered to a legal
/// operation.  This checks that the operation recursively has no FIRRTL
/// operations or types.
static LogicalResult verifyOpLegality(Operation *op) {
  auto checkTypes = [](Operation *op) -> WalkResult {
    // Check that this operation is not a FIRRTL op.
    if (isa_and_nonnull<FIRRTLDialect>(op->getDialect()))
      return op->emitError("Found unhandled FIRRTL operation '")
             << op->getName() << "'";

    // Helper to check a TypeRange for any FIRRTL types.
    auto checkTypeRange = [&](TypeRange types) -> LogicalResult {
      if (llvm::any_of(types, [](Type type) {
            return isa<FIRRTLDialect>(type.getDialect());
          }))
        return op->emitOpError("found unhandled FIRRTL type");
      return success();
    };

    // Check operand and result types.
    if (failed(checkTypeRange(op->getOperandTypes())) ||
        failed(checkTypeRange(op->getResultTypes())))
      return WalkResult::interrupt();

    // Check the block argument types.
    for (auto &region : op->getRegions())
      for (auto &block : region)
        if (failed(checkTypeRange(block.getArgumentTypes())))
          return WalkResult::interrupt();

    // Continue to the next operation.
    return WalkResult::advance();
  };

  if (checkTypes(op).wasInterrupted() || op->walk(checkTypes).wasInterrupted())
    return failure();
  return success();
}

/// Given two FIRRTL integer types, return the widest one.
static IntType getWidestIntType(Type t1, Type t2) {
  auto t1c = type_cast<IntType>(t1), t2c = type_cast<IntType>(t2);
  return t2c.getWidth() > t1c.getWidth() ? t2c : t1c;
}

/// Cast a value to a desired target type. This will insert struct casts and
/// unrealized conversion casts as necessary.
static Value castToFIRRTLType(Value val, Type type,
                              ImplicitLocOpBuilder &builder) {
  // Use HWStructCastOp for a bundle type.
  if (BundleType bundle = dyn_cast<BundleType>(type))
    val = builder.createOrFold<HWStructCastOp>(bundle.getPassiveType(), val);

  if (type != val.getType())
    val = mlir::UnrealizedConversionCastOp::create(builder, type, val)
              .getResult(0);

  return val;
}

/// Cast from a FIRRTL type (potentially with a flip) to a standard type.
static Value castFromFIRRTLType(Value val, Type type,
                                ImplicitLocOpBuilder &builder) {

  if (hw::StructType structTy = dyn_cast<hw::StructType>(type)) {
    // Strip off Flip type if needed.
    val = mlir::UnrealizedConversionCastOp::create(
              builder,
              type_cast<FIRRTLBaseType>(val.getType()).getPassiveType(), val)
              .getResult(0);
    val = builder.createOrFold<HWStructCastOp>(type, val);
    return val;
  }

  val =
      mlir::UnrealizedConversionCastOp::create(builder, type, val).getResult(0);

  return val;
}

/// Move a ExtractTestCode related annotation from annotations to an attribute.
static void moveVerifAnno(ModuleOp top, AnnotationSet &annos,
                          StringRef annoClass, StringRef attrBase) {
  auto anno = annos.getAnnotation(annoClass);
  auto *ctx = top.getContext();
  if (!anno)
    return;
  if (auto dir = anno.getMember<StringAttr>("directory")) {
    SmallVector<NamedAttribute> old;
    for (auto i : top->getAttrs())
      old.push_back(i);
    old.emplace_back(
        StringAttr::get(ctx, attrBase),
        hw::OutputFileAttr::getAsDirectory(ctx, dir.getValue(), true, true));
    top->setAttrs(old);
  }
  if (auto file = anno.getMember<StringAttr>("filename")) {
    SmallVector<NamedAttribute> old;
    for (auto i : top->getAttrs())
      old.push_back(i);
    old.emplace_back(StringAttr::get(ctx, attrBase + ".bindfile"),
                     hw::OutputFileAttr::getFromFilename(
                         ctx, file.getValue(), /*excludeFromFileList=*/true));
    top->setAttrs(old);
  }
}

static unsigned getBitWidthFromVectorSize(unsigned size) {
  return size == 1 ? 1 : llvm::Log2_64_Ceil(size);
}

// Try moving a name from an firrtl expression to a hw expression as a name
// hint.  Dont' overwrite an existing name.
static void tryCopyName(Operation *dst, Operation *src) {
  if (auto attr = src->getAttrOfType<StringAttr>("name"))
    if (!dst->hasAttr("sv.namehint") && !dst->hasAttr("name"))
      dst->setAttr("sv.namehint", attr);
}

namespace {

// A helper strutc to hold information about output file descriptor.
class FileDescriptorInfo {
public:
  FileDescriptorInfo(StringAttr outputFileName, mlir::ValueRange substitutions)
      : outputFileFormat(outputFileName), substitutions(substitutions) {
    assert(outputFileName ||
           substitutions.empty() &&
               "substitutions must be empty when output file name is empty");
  }

  FileDescriptorInfo() = default;

  // Substitution is required if substitution oprends are not empty.
  bool isSubstitutionRequired() const { return !substitutions.empty(); }

  // If the output file is not specified, the default file descriptor is used.
  bool isDefaultFd() const { return !outputFileFormat; }

  StringAttr getOutputFileFormat() const { return outputFileFormat; }
  mlir::ValueRange getSubstitutions() const { return substitutions; }

private:
  // "Verilog" format string for the output file.
  StringAttr outputFileFormat = {};

  // "FIRRTL" pre-lowered operands.
  mlir::ValueRange substitutions;
};

} // namespace

//===----------------------------------------------------------------------===//
// firrtl.module Lowering Pass
//===----------------------------------------------------------------------===//
namespace {

struct FIRRTLModuleLowering;

/// This is state shared across the parallel module lowering logic.
struct CircuitLoweringState {
  // Flags indicating whether the circuit uses certain header fragments.
  std::atomic<bool> usedPrintf{false};
  std::atomic<bool> usedAssertVerboseCond{false};
  std::atomic<bool> usedStopCond{false};
  std::atomic<bool> usedFileDescriptorLib{false};

  CircuitLoweringState(CircuitOp circuitOp, bool enableAnnotationWarning,
                       firrtl::VerificationFlavor verificationFlavor,
                       InstanceGraph &instanceGraph, NLATable *nlaTable)
      : circuitOp(circuitOp), instanceGraph(instanceGraph),
        enableAnnotationWarning(enableAnnotationWarning),
        verificationFlavor(verificationFlavor), nlaTable(nlaTable) {
    auto *context = circuitOp.getContext();

    // Get the testbench output directory.
    if (auto tbAnno =
            AnnotationSet(circuitOp).getAnnotation(testBenchDirAnnoClass)) {
      auto dirName = tbAnno.getMember<StringAttr>("dirname");
      testBenchDirectory = hw::OutputFileAttr::getAsDirectory(
          context, dirName.getValue(), false, true);
    }

    for (auto &op : *circuitOp.getBodyBlock()) {
      if (auto module = dyn_cast<FModuleLike>(op))
        if (AnnotationSet::removeAnnotations(module, dutAnnoClass))
          dut = module;
    }

    // Figure out which module is the DUT and TestHarness.  If there is no
    // module marked as the DUT, the top module is the DUT. If the DUT and the
    // test harness are the same, then there is no test harness.
    testHarness = instanceGraph.getTopLevelModule();
    if (!dut) {
      dut = testHarness;
      testHarness = nullptr;
    } else if (dut == testHarness) {
      testHarness = nullptr;
    }

    // Pre-populate the dutModules member with a list of all modules that are
    // determined to be under the DUT.
    auto inDUT = [&](igraph::ModuleOpInterface child) {
      auto isPhony = [](igraph::InstanceRecord *instRec) {
        if (auto inst = instRec->getInstance<InstanceOp>())
          return inst.getLowerToBind() || inst.getDoNotPrint();
        return false;
      };
      if (auto parent = dyn_cast<igraph::ModuleOpInterface>(*dut))
        return getInstanceGraph().isAncestor(child, parent, isPhony);
      return dut == child;
    };
    circuitOp->walk([&](FModuleLike moduleOp) {
      if (inDUT(moduleOp))
        dutModules.insert(moduleOp);
    });
  }

  Operation *getNewModule(Operation *oldModule) {
    auto it = oldToNewModuleMap.find(oldModule);
    return it != oldToNewModuleMap.end() ? it->second : nullptr;
  }

  Operation *getOldModule(Operation *newModule) {
    auto it = newToOldModuleMap.find(newModule);
    return it != newToOldModuleMap.end() ? it->second : nullptr;
  }

  void recordModuleMapping(Operation *oldFMod, Operation *newHWMod) {
    oldToNewModuleMap[oldFMod] = newHWMod;
    newToOldModuleMap[newHWMod] = oldFMod;
  }

  // Process remaining annotations and emit warnings on unprocessed annotations
  // still remaining in the annoSet.
  void processRemainingAnnotations(Operation *op, const AnnotationSet &annoSet);

  CircuitOp circuitOp;

  // Safely add a BindOp to global mutable state.  This will acquire a lock to
  // do this safely.
  void addBind(sv::BindOp op) {
    std::lock_guard<std::mutex> lock(bindsMutex);
    binds.push_back(op);
  }

  /// For a given Type Alias, return the corresponding AliasType. Create and
  /// record the AliasType, if it doesn't exist.
  hw::TypeAliasType getTypeAlias(Type rawType, BaseTypeAliasType firAliasType,
                                 Location typeLoc) {

    auto hwAlias = typeAliases.getTypedecl(firAliasType);
    if (hwAlias)
      return hwAlias;
    assert(!typeAliases.isFrozen() &&
           "type aliases cannot be generated after its frozen");
    return typeAliases.addTypedecl(rawType, firAliasType, typeLoc);
  }

  FModuleLike getDut() { return dut; }
  FModuleLike getTestHarness() { return testHarness; }

  // Return true if this module is the DUT or is instantiated by the DUT.
  // Returns false if the module is not instantiated by the DUT or is
  // instantiated under a bind.  This will accept either an old FIRRTL module or
  // a new HW module.
  bool isInDUT(igraph::ModuleOpInterface child) {
    if (auto hwModule = dyn_cast<hw::HWModuleOp>(child.getOperation()))
      child = cast<igraph::ModuleOpInterface>(getOldModule(hwModule));
    return dutModules.contains(child);
  }

  hw::OutputFileAttr getTestBenchDirectory() { return testBenchDirectory; }

  // Return true if this module is instantiated by the Test Harness.  Returns
  // false if the module is not instantiated by the Test Harness or if the Test
  // Harness is not known.
  bool isInTestHarness(igraph::ModuleOpInterface mod) { return !isInDUT(mod); }

  InstanceGraph &getInstanceGraph() { return instanceGraph; }

  /// Given a type, return the corresponding lowered type for the HW dialect.
  ///  A wrapper to the FIRRTLUtils::lowerType, required to ensure safe addition
  ///  of TypeScopeOp for all the TypeDecls.
  Type lowerType(Type type, Location loc) {
    return ::lowerType(type, loc,
                       [&](Type rawType, BaseTypeAliasType firrtlType,
                           Location typeLoc) -> hw::TypeAliasType {
                         return getTypeAlias(rawType, firrtlType, typeLoc);
                       });
  }

private:
  friend struct FIRRTLModuleLowering;
  friend struct FIRRTLLowering;
  CircuitLoweringState(const CircuitLoweringState &) = delete;
  void operator=(const CircuitLoweringState &) = delete;

  /// Mapping of FModuleOp to HWModuleOp
  DenseMap<Operation *, Operation *> oldToNewModuleMap;

  /// Mapping of HWModuleOp to FModuleOp
  DenseMap<Operation *, Operation *> newToOldModuleMap;

  /// Cache of module symbols.  We need to test hirarchy-based properties to
  /// lower annotaitons.
  InstanceGraph &instanceGraph;

  /// The set of old FIRRTL modules that are instantiated under the DUT.  This
  /// is precomputed as a module being under the DUT may rely on knowledge of
  /// properties of the instance and is not suitable for querying in the
  /// parallel execution region of this pass when the backing instances may
  /// already be erased.
  DenseSet<igraph::ModuleOpInterface> dutModules;

  // Record the set of remaining annotation classes. This is used to warn only
  // once about any annotation class.
  StringSet<> pendingAnnotations;
  const bool enableAnnotationWarning;
  std::mutex annotationPrintingMtx;

  const firrtl::VerificationFlavor verificationFlavor;

  // Records any sv::BindOps that are found during the course of execution.
  // This is unsafe to access directly and should only be used through addBind.
  SmallVector<sv::BindOp> binds;

  // Control access to binds.
  std::mutex bindsMutex;

  // The design-under-test (DUT), if it is found.  This will be set if a
  // "sifive.enterprise.firrtl.MarkDUTAnnotation" exists.
  FModuleLike dut;

  // If there is a module marked as the DUT and it is not the top level module,
  // this will be set.
  FModuleLike testHarness;

  // If there is a testbench output directory, this will be set.
  hw::OutputFileAttr testBenchDirectory;

  /// A mapping of instances to their forced instantiation names (if
  /// applicable).
  DenseMap<std::pair<Attribute, Attribute>, Attribute> instanceForceNames;

  /// The set of guard macros to emit declarations for.
  SetVector<StringAttr> macroDeclNames;
  std::mutex macroDeclMutex;

  void addMacroDecl(StringAttr name) {
    std::unique_lock<std::mutex> lock(macroDeclMutex);
    macroDeclNames.insert(name);
  }

  /// The list of fragments on which the modules rely. Must be set outside the
  /// parallelized module lowering since module type reads access it.
  DenseMap<hw::HWModuleOp, SetVector<Attribute>> fragments;
  llvm::sys::SmartMutex<true> fragmentsMutex;

  void addFragment(hw::HWModuleOp module, StringRef fragment) {
    llvm::sys::SmartScopedLock<true> lock(fragmentsMutex);
    fragments[module].insert(
        FlatSymbolRefAttr::get(circuitOp.getContext(), fragment));
  }

  /// Cached nla table analysis.
  NLATable *nlaTable = nullptr;

  /// FIRRTL::BaseTypeAliasType is lowered to hw::TypeAliasType, which requires
  /// TypedeclOp inside a single global TypeScopeOp. This structure
  /// maintains a map of FIRRTL alias types to HW alias type, which is populated
  /// in the sequential phase and accessed during the read-only phase when its
  /// frozen.
  /// This structure ensures that
  /// all TypeAliases are lowered as a prepass, before lowering all the modules
  /// in parallel. Lowering of TypeAliases must be done sequentially to ensure
  /// deteministic TypeDecls inside the global TypeScopeOp.
  struct RecordTypeAlias {

    RecordTypeAlias(CircuitOp c) : circuitOp(c) {}

    hw::TypeAliasType getTypedecl(BaseTypeAliasType firAlias) const {
      auto iter = firrtlTypeToAliasTypeMap.find(firAlias);
      if (iter != firrtlTypeToAliasTypeMap.end())
        return iter->second;
      return {};
    }

    bool isFrozen() { return frozen; }

    void freeze() { frozen = true; }

    hw::TypeAliasType addTypedecl(Type rawType, BaseTypeAliasType firAlias,
                                  Location typeLoc) {
      assert(!frozen && "Record already frozen, cannot be updated");

      if (!typeScope) {
        auto b = ImplicitLocOpBuilder::atBlockBegin(
            circuitOp.getLoc(),
            &circuitOp->getParentRegion()->getBlocks().back());
        typeScope = hw::TypeScopeOp::create(
            b, b.getStringAttr(circuitOp.getName() + "__TYPESCOPE_"));
        typeScope.getBodyRegion().push_back(new Block());
      }
      auto typeName = firAlias.getName();
      // Get a unique typedecl name.
      // The bundleName can conflict with other symbols, but must be unique
      // within the TypeScopeOp.
      typeName =
          StringAttr::get(typeName.getContext(),
                          typeDeclNamespace.newName(typeName.getValue()));

      auto typeScopeBuilder =
          ImplicitLocOpBuilder::atBlockEnd(typeLoc, typeScope.getBodyBlock());
      auto typeDecl = hw::TypedeclOp::create(typeScopeBuilder, typeLoc,
                                             typeName, rawType, nullptr);
      auto hwAlias = hw::TypeAliasType::get(
          SymbolRefAttr::get(typeScope.getSymNameAttr(),
                             {FlatSymbolRefAttr::get(typeDecl)}),
          rawType);
      auto insert = firrtlTypeToAliasTypeMap.try_emplace(firAlias, hwAlias);
      assert(insert.second && "Entry already exists, insert failed");
      return insert.first->second;
    }

  private:
    bool frozen = false;
    /// Global typescope for all the typedecls in this module.
    hw::TypeScopeOp typeScope;

    /// Map of FIRRTL type to the lowered AliasType.
    DenseMap<Type, hw::TypeAliasType> firrtlTypeToAliasTypeMap;

    /// Set to keep track of unique typedecl names.
    Namespace typeDeclNamespace;

    CircuitOp circuitOp;
  };

  RecordTypeAlias typeAliases = RecordTypeAlias(circuitOp);
};

void CircuitLoweringState::processRemainingAnnotations(
    Operation *op, const AnnotationSet &annoSet) {
  if (!enableAnnotationWarning || annoSet.empty())
    return;
  std::lock_guard<std::mutex> lock(annotationPrintingMtx);

  for (auto a : annoSet) {
    auto inserted = pendingAnnotations.insert(a.getClass());
    if (!inserted.second)
      continue;

    // The following annotations are okay to be silently dropped at this point.
    // This can occur for example if an annotation marks something in the IR as
    // not to be processed by a pass, but that pass hasn't run anyway.
    if (a.isClass(
            // If the class is `circt.nonlocal`, it's not really an annotation,
            // but part of a path specifier for another annotation which is
            // non-local.  We can ignore these path specifiers since there will
            // be a warning produced for the real annotation.
            "circt.nonlocal",
            // The following are either consumed by a pass running before
            // LowerToHW, or they have no effect if the pass doesn't run at all.
            // If the accompanying pass runs on the HW dialect, then LowerToHW
            // should have consumed and processed these into an attribute on the
            // output.
            dontObfuscateModuleAnnoClass, noDedupAnnoClass,
            // The following are inspected (but not consumed) by FIRRTL/GCT
            // passes that have all run by now. Since no one is responsible for
            // consuming these, they will linger around and can be ignored.
            dutAnnoClass, metadataDirectoryAttrName,
            elaborationArtefactsDirectoryAnnoClass, testBenchDirAnnoClass,
            // This annotation is used to mark which external modules are
            // imported blackboxes from the BlackBoxReader pass.
            blackBoxAnnoClass,
            // This annotation is used by several GrandCentral passes.
            extractGrandCentralClass,
            // The following will be handled while lowering the verification
            // ops.
            extractAssertAnnoClass, extractAssumeAnnoClass,
            extractCoverageAnnoClass,
            // The following will be handled after lowering FModule ops, since
            // they are still needed on the circuit until after lowering
            // FModules.
            moduleHierAnnoClass, testHarnessHierAnnoClass,
            blackBoxTargetDirAnnoClass))
      continue;

    mlir::emitWarning(op->getLoc(), "unprocessed annotation:'" + a.getClass() +
                                        "' still remaining after LowerToHW");
  }
}
} // end anonymous namespace

namespace {
struct FIRRTLModuleLowering
    : public circt::impl::LowerFIRRTLToHWBase<FIRRTLModuleLowering> {

  void runOnOperation() override;
  void setEnableAnnotationWarning() { enableAnnotationWarning = true; }

  using LowerFIRRTLToHWBase<FIRRTLModuleLowering>::verificationFlavor;

private:
  void lowerFileHeader(CircuitOp op, CircuitLoweringState &loweringState);
  LogicalResult lowerPorts(ArrayRef<PortInfo> firrtlPorts,
                           SmallVectorImpl<hw::PortInfo> &ports,
                           Operation *moduleOp, StringRef moduleName,
                           CircuitLoweringState &loweringState);
  bool handleForceNameAnnos(FModuleLike oldModule, AnnotationSet &annos,
                            CircuitLoweringState &loweringState);
  hw::HWModuleOp lowerModule(FModuleOp oldModule, Block *topLevelModule,
                             CircuitLoweringState &loweringState);
  hw::HWModuleExternOp lowerExtModule(FExtModuleOp oldModule,
                                      Block *topLevelModule,
                                      CircuitLoweringState &loweringState);
  hw::HWModuleExternOp lowerMemModule(FMemModuleOp oldModule,
                                      Block *topLevelModule,
                                      CircuitLoweringState &loweringState);

  LogicalResult
  lowerModulePortsAndMoveBody(FModuleOp oldModule, hw::HWModuleOp newModule,
                              CircuitLoweringState &loweringState);
  LogicalResult lowerModuleBody(hw::HWModuleOp module,
                                CircuitLoweringState &loweringState);
  LogicalResult lowerFormalBody(verif::FormalOp formalOp,
                                CircuitLoweringState &loweringState);
  LogicalResult lowerSimulationBody(verif::SimulationOp simulationOp,
                                    CircuitLoweringState &loweringState);
  LogicalResult lowerFileBody(emit::FileOp op);
  LogicalResult lowerBody(Operation *op, CircuitLoweringState &loweringState);
};

} // end anonymous namespace

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::createLowerFIRRTLToHWPass(
    bool enableAnnotationWarning,
    firrtl::VerificationFlavor verificationFlavor) {
  auto pass = std::make_unique<FIRRTLModuleLowering>();
  if (enableAnnotationWarning)
    pass->setEnableAnnotationWarning();
  pass->verificationFlavor = verificationFlavor;
  return pass;
}

/// Run on the firrtl.circuit operation, lowering any firrtl.module operations
/// it contains.
void FIRRTLModuleLowering::runOnOperation() {

  // We run on the top level modules in the IR blob.  Start by finding the
  // firrtl.circuit within it.  If there is none, then there is nothing to do.
  auto *topLevelModule = getOperation().getBody();

  // Find the single firrtl.circuit in the module.
  CircuitOp circuit;
  for (auto &op : *topLevelModule) {
    if ((circuit = dyn_cast<CircuitOp>(&op)))
      break;
  }

  if (!circuit)
    return;

  auto *circuitBody = circuit.getBodyBlock();

  // Keep track of the mapping from old to new modules.  The result may be null
  // if lowering failed.
  CircuitLoweringState state(circuit, enableAnnotationWarning,
                             verificationFlavor, getAnalysis<InstanceGraph>(),
                             &getAnalysis<NLATable>());

  SmallVector<Operation *, 32> opsToProcess;

  AnnotationSet circuitAnno(circuit);
  moveVerifAnno(getOperation(), circuitAnno, extractAssertAnnoClass,
                "firrtl.extract.assert");
  moveVerifAnno(getOperation(), circuitAnno, extractAssumeAnnoClass,
                "firrtl.extract.assume");
  moveVerifAnno(getOperation(), circuitAnno, extractCoverageAnnoClass,
                "firrtl.extract.cover");
  circuitAnno.removeAnnotationsWithClass(
      extractAssertAnnoClass, extractAssumeAnnoClass, extractCoverageAnnoClass);

  state.processRemainingAnnotations(circuit, circuitAnno);
  // Iterate through each operation in the circuit body, transforming any
  // FModule's we come across. If any module fails to lower, return early.
  for (auto &op : make_early_inc_range(circuitBody->getOperations())) {
    auto result =
        TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<FModuleOp>([&](auto module) {
              auto loweredMod = lowerModule(module, topLevelModule, state);
              if (!loweredMod)
                return failure();

              state.recordModuleMapping(&op, loweredMod);
              opsToProcess.push_back(loweredMod);
              // Lower all the alias types.
              module.walk([&](Operation *op) {
                for (auto res : op->getResults()) {
                  if (auto aliasType =
                          type_dyn_cast<BaseTypeAliasType>(res.getType()))
                    state.lowerType(aliasType, op->getLoc());
                }
              });
              return lowerModulePortsAndMoveBody(module, loweredMod, state);
            })
            .Case<FExtModuleOp>([&](auto extModule) {
              auto loweredMod =
                  lowerExtModule(extModule, topLevelModule, state);
              if (!loweredMod)
                return failure();
              state.recordModuleMapping(&op, loweredMod);
              return success();
            })
            .Case<FMemModuleOp>([&](auto memModule) {
              auto loweredMod =
                  lowerMemModule(memModule, topLevelModule, state);
              if (!loweredMod)
                return failure();
              state.recordModuleMapping(&op, loweredMod);
              return success();
            })
            .Case<FormalOp>([&](auto oldOp) {
              auto builder = OpBuilder::atBlockEnd(topLevelModule);
              auto newOp = verif::FormalOp::create(builder, oldOp.getLoc(),
                                                   oldOp.getNameAttr(),
                                                   oldOp.getParametersAttr());
              newOp.getBody().emplaceBlock();
              state.recordModuleMapping(oldOp, newOp);
              opsToProcess.push_back(newOp);
              return success();
            })
            .Case<SimulationOp>([&](auto oldOp) {
              auto loc = oldOp.getLoc();
              auto builder = OpBuilder::atBlockEnd(topLevelModule);
              auto newOp = verif::SimulationOp::create(
                  builder, loc, oldOp.getNameAttr(), oldOp.getParametersAttr());
              auto &body = newOp.getRegion().emplaceBlock();
              body.addArgument(seq::ClockType::get(builder.getContext()), loc);
              body.addArgument(builder.getI1Type(), loc);
              state.recordModuleMapping(oldOp, newOp);
              opsToProcess.push_back(newOp);
              return success();
            })
            .Case<emit::FileOp>([&](auto fileOp) {
              fileOp->moveBefore(topLevelModule, topLevelModule->end());
              opsToProcess.push_back(fileOp);
              return success();
            })
            .Default([&](Operation *op) {
              // We don't know what this op is.  If it has no illegal FIRRTL
              // types, we can forward the operation.  Otherwise, we emit an
              // error and drop the operation from the circuit.
              if (succeeded(verifyOpLegality(op)))
                op->moveBefore(topLevelModule, topLevelModule->end());
              else
                return failure();
              return success();
            });
    if (failed(result))
      return signalPassFailure();
  }
  // Ensure no more TypeDecl can be added to the global TypeScope.
  state.typeAliases.freeze();
  // Handle the creation of the module hierarchy metadata.

  // Collect the two sets of hierarchy files from the circuit. Some of them will
  // be rooted at the test harness, the others will be rooted at the DUT.
  SmallVector<Attribute> dutHierarchyFiles;
  SmallVector<Attribute> testHarnessHierarchyFiles;
  circuitAnno.removeAnnotations([&](Annotation annotation) {
    if (annotation.isClass(moduleHierAnnoClass)) {
      auto file = hw::OutputFileAttr::getFromFilename(
          &getContext(),
          annotation.getMember<StringAttr>("filename").getValue(),
          /*excludeFromFileList=*/true);
      dutHierarchyFiles.push_back(file);
      return true;
    }
    if (annotation.isClass(testHarnessHierAnnoClass)) {
      auto file = hw::OutputFileAttr::getFromFilename(
          &getContext(),
          annotation.getMember<StringAttr>("filename").getValue(),
          /*excludeFromFileList=*/true);
      // If there is no testHarness, we print the hiearchy for this file
      // starting at the DUT.
      if (state.getTestHarness())
        testHarnessHierarchyFiles.push_back(file);
      else
        dutHierarchyFiles.push_back(file);
      return true;
    }
    return false;
  });
  // Attach the lowered form of these annotations.
  if (!dutHierarchyFiles.empty())
    state.getNewModule(state.getDut())
        ->setAttr(moduleHierarchyFileAttrName,
                  ArrayAttr::get(&getContext(), dutHierarchyFiles));
  if (!testHarnessHierarchyFiles.empty())
    state.getNewModule(state.getTestHarness())
        ->setAttr(moduleHierarchyFileAttrName,
                  ArrayAttr::get(&getContext(), testHarnessHierarchyFiles));

  // Lower all module and formal op bodies.
  auto result =
      mlir::failableParallelForEach(&getContext(), opsToProcess, [&](auto op) {
        return lowerBody(op, state);
      });
  if (failed(result))
    return signalPassFailure();

  // Move binds from inside modules to outside modules.
  for (auto bind : state.binds) {
    bind->moveBefore(bind->getParentOfType<hw::HWModuleOp>());
  }

  // Fix up fragment attributes.
  for (auto &[module, fragments] : state.fragments)
    module->setAttr(emit::getFragmentsAttrName(),
                    ArrayAttr::get(&getContext(), fragments.getArrayRef()));

  // Finally delete all the old modules.
  for (auto oldNew : state.oldToNewModuleMap)
    oldNew.first->erase();

  if (!state.macroDeclNames.empty()) {
    ImplicitLocOpBuilder b(UnknownLoc::get(&getContext()), circuit);
    for (auto name : state.macroDeclNames) {
      sv::MacroDeclOp::create(b, name);
    }
  }

  // Emit all the macros and preprocessor gunk at the start of the file.
  lowerFileHeader(circuit, state);

  // Now that the modules are moved over, remove the Circuit.
  circuit.erase();
}

/// Emit the file header that defines a bunch of macros.
void FIRRTLModuleLowering::lowerFileHeader(CircuitOp op,
                                           CircuitLoweringState &state) {
  // Intentionally pass an UnknownLoc here so we don't get line number
  // comments on the output of this boilerplate in generated Verilog.
  ImplicitLocOpBuilder b(UnknownLoc::get(&getContext()), op);

  // Helper function to emit a "#ifdef guard" with a `define in the then and
  // optionally in the else branch.
  auto emitGuardedDefine = [&](StringRef guard, StringRef defName,
                               StringRef defineTrue = "",
                               StringRef defineFalse = StringRef()) {
    if (!defineFalse.data()) {
      assert(defineTrue.data() && "didn't define anything");
      sv::IfDefOp::create(
          b, guard, [&]() { sv::MacroDefOp::create(b, defName, defineTrue); });
    } else {
      sv::IfDefOp::create(
          b, guard,
          [&]() {
            if (defineTrue.data())
              sv::MacroDefOp::create(b, defName, defineTrue);
          },
          [&]() { sv::MacroDefOp::create(b, defName, defineFalse); });
    }
  };

  // Helper function to emit #ifndef guard.
  auto emitGuard = [&](const char *guard, llvm::function_ref<void(void)> body) {
    sv::IfDefOp::create(
        b, guard, []() {}, body);
  };

  if (state.usedFileDescriptorLib) {
    // Define a type for the file descriptor getter.
    SmallVector<hw::ModulePort> ports;

    // Input port for filename
    hw::ModulePort namePort;
    namePort.name = b.getStringAttr("name");
    namePort.type = hw::StringType::get(b.getContext());
    namePort.dir = hw::ModulePort::Direction::Input;
    ports.push_back(namePort);

    // Output port for file descriptor
    hw::ModulePort fdPort;
    fdPort.name = b.getStringAttr("fd");
    fdPort.type = b.getIntegerType(32);
    fdPort.dir = hw::ModulePort::Direction::Output;
    ports.push_back(fdPort);

    // Create module type with the ports
    auto moduleType = hw::ModuleType::get(b.getContext(), ports);

    SmallVector<NamedAttribute> perArgumentsAttr;
    perArgumentsAttr.push_back(
        {sv::FuncOp::getExplicitlyReturnedAttrName(), b.getUnitAttr()});

    SmallVector<Attribute> argumentAttr = {
        DictionaryAttr::get(b.getContext(), {}),
        DictionaryAttr::get(b.getContext(), perArgumentsAttr)};

    // Create the function declaration
    auto func = sv::FuncOp::create(
        b, /*sym_name=*/
        "__circt_lib_logging::FileDescriptor::get", moduleType,
        /*perArgumentAttrs=*/
        b.getArrayAttr(
            {b.getDictionaryAttr({}), b.getDictionaryAttr(perArgumentsAttr)}),
        /*inputLocs=*/
        ArrayAttr(),
        /*resultLocs=*/
        ArrayAttr(),
        /*verilogName=*/
        b.getStringAttr("__circt_lib_logging::FileDescriptor::get"));
    func.setPrivate();

    sv::MacroDeclOp::create(b, "__CIRCT_LIB_LOGGING");
    // Create the fragment containing the FileDescriptor class.
    emit::FragmentOp::create(b, "CIRCT_LIB_LOGGING_FRAGMENT", [&] {
      emitGuard("SYNTHESIS", [&]() {
        emitGuard("__CIRCT_LIB_LOGGING", [&]() {
          sv::VerbatimOp::create(b, R"(// CIRCT Logging Library
package __circt_lib_logging;
  class FileDescriptor;
    static int global_id [string];
    static function int get(string name);
      if (global_id.exists(name) == 32'h0) begin
        global_id[name] = $fopen(name);
        if (global_id[name] == 32'h0)
          $error("Failed to open file %s", name);
      end
      return global_id[name];
    endfunction
  endclass
endpackage
)");

          sv::MacroDefOp::create(b, "__CIRCT_LIB_LOGGING", "");
        });
      });
    });
  }

  if (state.usedPrintf) {
    sv::MacroDeclOp::create(b, "PRINTF_COND");
    sv::MacroDeclOp::create(b, "PRINTF_COND_");
    emit::FragmentOp::create(b, "PRINTF_COND_FRAGMENT", [&] {
      sv::VerbatimOp::create(
          b, "\n// Users can define 'PRINTF_COND' to add an extra gate to "
             "prints.");
      emitGuard("PRINTF_COND_", [&]() {
        emitGuardedDefine("PRINTF_COND", "PRINTF_COND_", "(`PRINTF_COND)", "1");
      });
    });
  }

  if (state.usedAssertVerboseCond) {
    sv::MacroDeclOp::create(b, "ASSERT_VERBOSE_COND");
    sv::MacroDeclOp::create(b, "ASSERT_VERBOSE_COND_");
    emit::FragmentOp::create(b, "ASSERT_VERBOSE_COND_FRAGMENT", [&] {
      sv::VerbatimOp::create(
          b, "\n// Users can define 'ASSERT_VERBOSE_COND' to add an extra "
             "gate to assert error printing.");
      emitGuard("ASSERT_VERBOSE_COND_", [&]() {
        emitGuardedDefine("ASSERT_VERBOSE_COND", "ASSERT_VERBOSE_COND_",
                          "(`ASSERT_VERBOSE_COND)", "1");
      });
    });
  }

  if (state.usedStopCond) {
    sv::MacroDeclOp::create(b, "STOP_COND");
    sv::MacroDeclOp::create(b, "STOP_COND_");
    emit::FragmentOp::create(b, "STOP_COND_FRAGMENT", [&] {
      sv::VerbatimOp::create(
          b, "\n// Users can define 'STOP_COND' to add an extra gate "
             "to stop conditions.");
      emitGuard("STOP_COND_", [&]() {
        emitGuardedDefine("STOP_COND", "STOP_COND_", "(`STOP_COND)", "1");
      });
    });
  }
}

LogicalResult
FIRRTLModuleLowering::lowerPorts(ArrayRef<PortInfo> firrtlPorts,
                                 SmallVectorImpl<hw::PortInfo> &ports,
                                 Operation *moduleOp, StringRef moduleName,
                                 CircuitLoweringState &loweringState) {
  ports.reserve(firrtlPorts.size());
  size_t numArgs = 0;
  size_t numResults = 0;
  for (auto e : llvm::enumerate(firrtlPorts)) {
    PortInfo firrtlPort = e.value();
    size_t portNo = e.index();
    hw::PortInfo hwPort;
    hwPort.name = firrtlPort.name;
    hwPort.type = loweringState.lowerType(firrtlPort.type, firrtlPort.loc);
    if (firrtlPort.sym)
      if (firrtlPort.sym.size() > 1 ||
          (firrtlPort.sym.size() == 1 && !firrtlPort.sym.getSymName()))
        return emitError(firrtlPort.loc)
               << "cannot lower aggregate port " << firrtlPort.name
               << " with field sensitive symbols, HW dialect does not support "
                  "per field symbols yet.";
    hwPort.setSym(firrtlPort.sym, moduleOp->getContext());
    bool hadDontTouch = firrtlPort.annotations.removeDontTouch();
    if (hadDontTouch && !hwPort.getSym()) {
      if (hwPort.type.isInteger(0)) {
        if (enableAnnotationWarning) {
          mlir::emitWarning(firrtlPort.loc)
              << "zero width port " << hwPort.name
              << " has dontTouch annotation, removing anyway";
        }
        continue;
      }

      hwPort.setSym(
          hw::InnerSymAttr::get(StringAttr::get(
              moduleOp->getContext(),
              Twine("__") + moduleName + Twine("__DONTTOUCH__") +
                  Twine(portNo) + Twine("__") + firrtlPort.name.strref())),
          moduleOp->getContext());
    }

    // We can't lower all types, so make sure to cleanly reject them.
    if (!hwPort.type) {
      moduleOp->emitError("cannot lower this port type to HW");
      return failure();
    }

    // If this is a zero bit port, just drop it.  It doesn't matter if it is
    // input, output, or inout.  We don't want these at the HW level.
    if (hwPort.type.isInteger(0)) {
      auto sym = hwPort.getSym();
      if (sym && !sym.empty()) {
        return mlir::emitError(firrtlPort.loc)
               << "zero width port " << hwPort.name
               << " is referenced by name [" << sym
               << "] (e.g. in an XMR) but must be removed";
      }
      continue;
    }

    // Figure out the direction of the port.
    if (firrtlPort.isOutput()) {
      hwPort.dir = hw::ModulePort::Direction::Output;
      hwPort.argNum = numResults++;
    } else if (firrtlPort.isInput()) {
      hwPort.dir = hw::ModulePort::Direction::Input;
      hwPort.argNum = numArgs++;
    } else {
      // If the port is an inout bundle or contains an analog type, then it is
      // implicitly inout.
      hwPort.type = hw::InOutType::get(hwPort.type);
      hwPort.dir = hw::ModulePort::Direction::InOut;
      hwPort.argNum = numArgs++;
    }
    hwPort.loc = firrtlPort.loc;
    ports.push_back(hwPort);
    loweringState.processRemainingAnnotations(moduleOp, firrtlPort.annotations);
  }
  return success();
}

/// Map the parameter specifier on the specified extmodule into the HWModule
/// representation for parameters.  If `ignoreValues` is true, all the values
/// are dropped.
static ArrayAttr getHWParameters(FExtModuleOp module, bool ignoreValues) {
  auto params = llvm::map_range(module.getParameters(), [](Attribute a) {
    return cast<ParamDeclAttr>(a);
  });
  if (params.empty())
    return {};

  Builder builder(module);

  // Map the attributes over from firrtl attributes to HW attributes
  // directly.  MLIR's DictionaryAttr always stores keys in the dictionary
  // in sorted order which is nicely stable.
  SmallVector<Attribute> newParams;
  for (const ParamDeclAttr &entry : params) {
    auto name = entry.getName();
    auto type = entry.getType();
    auto value = ignoreValues ? Attribute() : entry.getValue();
    auto paramAttr =
        hw::ParamDeclAttr::get(builder.getContext(), name, type, value);
    newParams.push_back(paramAttr);
  }
  return builder.getArrayAttr(newParams);
}

bool FIRRTLModuleLowering::handleForceNameAnnos(
    FModuleLike oldModule, AnnotationSet &annos,
    CircuitLoweringState &loweringState) {
  bool failed = false;
  // Remove ForceNameAnnotations by generating verilogNames on instances.
  annos.removeAnnotations([&](Annotation anno) {
    if (!anno.isClass(forceNameAnnoClass))
      return false;

    auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
    // This must be a non-local annotation due to how the Chisel API is
    // implemented.
    //
    // TODO: handle this in some sensible way based on what the SFC does with
    // a local annotation.
    if (!sym) {
      auto diag = oldModule.emitOpError()
                  << "contains a '" << forceNameAnnoClass
                  << "' that is not a non-local annotation";
      diag.attachNote() << "the erroneous annotation is '" << anno.getDict()
                        << "'\n";
      failed = true;
      return false;
    }

    auto nla = loweringState.nlaTable->getNLA(sym.getAttr());
    // The non-local anchor must exist.
    //
    // TODO: handle this with annotation verification.
    if (!nla) {
      auto diag = oldModule.emitOpError()
                  << "contains a '" << forceNameAnnoClass
                  << "' whose non-local symbol, '" << sym
                  << "' does not exist in the circuit";
      diag.attachNote() << "the erroneous annotation is '" << anno.getDict();
      failed = true;
      return false;
    }

    // Add the forced name to global state (keyed by a pseudo-inner name ref).
    // Error out if this key is alredy in use.
    //
    // TODO: this error behavior can be relaxed to always overwrite with the
    // new forced name (the bug-compatible behavior of the Chisel
    // implementation) or fixed to duplicate modules such that the naming can
    // be applied.
    auto inst =
        cast<hw::InnerRefAttr>(nla.getNamepath().getValue().take_back(2)[0]);
    auto inserted = loweringState.instanceForceNames.insert(
        {{inst.getModule(), inst.getName()}, anno.getMember("name")});
    if (!inserted.second &&
        (anno.getMember("name") != (inserted.first->second))) {
      auto diag = oldModule.emitError()
                  << "contained multiple '" << forceNameAnnoClass
                  << "' with different names: " << inserted.first->second
                  << " was not " << anno.getMember("name");
      diag.attachNote() << "the erroneous annotation is '" << anno.getDict()
                        << "'";
      failed = true;
      return false;
    }
    return true;
  });
  return failed;
}

hw::HWModuleExternOp
FIRRTLModuleLowering::lowerExtModule(FExtModuleOp oldModule,
                                     Block *topLevelModule,
                                     CircuitLoweringState &loweringState) {
  // Map the ports over, lowering their types as we go.
  SmallVector<PortInfo> firrtlPorts = oldModule.getPorts();
  SmallVector<hw::PortInfo, 8> ports;
  if (failed(lowerPorts(firrtlPorts, ports, oldModule, oldModule.getName(),
                        loweringState)))
    return {};

  StringRef verilogName;
  if (auto defName = oldModule.getDefname())
    verilogName = defName.value();

  // Build the new hw.module op.
  auto builder = OpBuilder::atBlockEnd(topLevelModule);
  auto nameAttr = builder.getStringAttr(oldModule.getName());
  // Map over parameters if present.  Drop all values as we do so, so there are
  // no known default values in the extmodule.  This ensures that the
  // hw.instance will print all the parameters when generating verilog.
  auto parameters = getHWParameters(oldModule, /*ignoreValues=*/true);
  auto newModule = hw::HWModuleExternOp::create(
      builder, oldModule.getLoc(), nameAttr, ports, verilogName, parameters);
  SymbolTable::setSymbolVisibility(newModule,
                                   SymbolTable::getSymbolVisibility(oldModule));

  bool hasOutputPort =
      llvm::any_of(firrtlPorts, [&](auto p) { return p.isOutput(); });
  if (!hasOutputPort &&
      AnnotationSet::removeAnnotations(oldModule, verifBlackBoxAnnoClass) &&
      loweringState.isInDUT(oldModule))
    newModule->setAttr("firrtl.extract.cover.extra", builder.getUnitAttr());

  AnnotationSet annos(oldModule);
  if (handleForceNameAnnos(oldModule, annos, loweringState))
    return {};

  loweringState.processRemainingAnnotations(oldModule, annos);
  return newModule;
}

hw::HWModuleExternOp
FIRRTLModuleLowering::lowerMemModule(FMemModuleOp oldModule,
                                     Block *topLevelModule,
                                     CircuitLoweringState &loweringState) {
  // Map the ports over, lowering their types as we go.
  SmallVector<PortInfo> firrtlPorts = oldModule.getPorts();
  SmallVector<hw::PortInfo, 8> ports;
  if (failed(lowerPorts(firrtlPorts, ports, oldModule, oldModule.getName(),
                        loweringState)))
    return {};

  // Build the new hw.module op.
  auto builder = OpBuilder::atBlockEnd(topLevelModule);
  auto newModule = hw::HWModuleExternOp::create(
      builder, oldModule.getLoc(), oldModule.getModuleNameAttr(), ports,
      oldModule.getModuleNameAttr());
  loweringState.processRemainingAnnotations(oldModule,
                                            AnnotationSet(oldModule));
  return newModule;
}

/// Run on each firrtl.module, creating a basic hw.module for the firrtl module.
hw::HWModuleOp
FIRRTLModuleLowering::lowerModule(FModuleOp oldModule, Block *topLevelModule,
                                  CircuitLoweringState &loweringState) {
  // Map the ports over, lowering their types as we go.
  SmallVector<PortInfo> firrtlPorts = oldModule.getPorts();
  SmallVector<hw::PortInfo, 8> ports;
  if (failed(lowerPorts(firrtlPorts, ports, oldModule, oldModule.getName(),
                        loweringState)))
    return {};

  // Build the new hw.module op.
  auto builder = OpBuilder::atBlockEnd(topLevelModule);
  auto nameAttr = builder.getStringAttr(oldModule.getName());
  auto newModule =
      hw::HWModuleOp::create(builder, oldModule.getLoc(), nameAttr, ports);

  if (auto comment = oldModule->getAttrOfType<StringAttr>("comment"))
    newModule.setCommentAttr(comment);

  // Copy over any attributes which are not required for FModuleOp.
  SmallVector<StringRef, 12> attrNames = {
      "annotations",   "convention",      "layers",
      "portNames",     "sym_name",        "portDirections",
      "portTypes",     "portAnnotations", "portSymbols",
      "portLocations", "parameters",      SymbolTable::getVisibilityAttrName()};

  DenseSet<StringRef> attrSet(attrNames.begin(), attrNames.end());
  SmallVector<NamedAttribute> newAttrs(newModule->getAttrs());
  for (auto i :
       llvm::make_filter_range(oldModule->getAttrs(), [&](auto namedAttr) {
         return !attrSet.count(namedAttr.getName()) &&
                !newModule->getAttrDictionary().contains(namedAttr.getName());
       }))
    newAttrs.push_back(i);

  newModule->setAttrs(newAttrs);

  // If the circuit has an entry point, set all other modules private.
  // Otherwise, mark all modules as public.
  SymbolTable::setSymbolVisibility(newModule,
                                   SymbolTable::getSymbolVisibility(oldModule));

  // Transform module annotations
  AnnotationSet annos(oldModule);

  if (annos.removeAnnotation(verifBlackBoxAnnoClass))
    newModule->setAttr("firrtl.extract.cover.extra", builder.getUnitAttr());

  // If this is in the test harness, make sure it goes to the test directory.
  // Do not update output file information if it is already present.
  if (auto testBenchDir = loweringState.getTestBenchDirectory())
    if (loweringState.isInTestHarness(oldModule)) {
      if (!newModule->hasAttr("output_file"))
        newModule->setAttr("output_file", testBenchDir);
      newModule->setAttr("firrtl.extract.do_not_extract",
                         builder.getUnitAttr());
      newModule.setCommentAttr(
          builder.getStringAttr("VCS coverage exclude_file"));
    }

  if (handleForceNameAnnos(oldModule, annos, loweringState))
    return {};

  loweringState.processRemainingAnnotations(oldModule, annos);
  return newModule;
}

/// Given a value of analog type, check to see the only use of it is an
/// attach. If so, remove the attach and return the value being attached to
/// it, converted to an HW inout type.  If this isn't a situation we can
/// handle, just return null.
static Value tryEliminatingAttachesToAnalogValue(Value value,
                                                 Operation *insertPoint) {
  if (!value.hasOneUse())
    return {};

  auto attach = dyn_cast<AttachOp>(*value.user_begin());
  if (!attach || attach.getNumOperands() != 2)
    return {};

  // Don't optimize zero bit analogs.
  auto loweredType = lowerType(value.getType());
  if (loweredType.isInteger(0))
    return {};

  // Check to see if the attached value dominates the insertion point.  If
  // not, just fail.
  auto attachedValue = attach.getOperand(attach.getOperand(0) == value);
  auto *op = attachedValue.getDefiningOp();
  if (op && op->getBlock() == insertPoint->getBlock() &&
      !op->isBeforeInBlock(insertPoint))
    return {};

  attach.erase();

  ImplicitLocOpBuilder builder(insertPoint->getLoc(), insertPoint);
  return castFromFIRRTLType(attachedValue, hw::InOutType::get(loweredType),
                            builder);
}

/// Given a value of flip type, check to see if all of the uses of it are
/// connects.  If so, remove the connects and return the value being connected
/// to it, converted to an HW type.  If this isn't a situation we can handle,
/// just return null.
///
/// This can happen when there are no connects to the value.  The 'mergePoint'
/// location is where a 'hw.merge' operation should be inserted if needed.
static Value
tryEliminatingConnectsToValue(Value flipValue, Operation *insertPoint,
                              CircuitLoweringState &loweringState) {
  // Handle analog's separately.
  if (type_isa<AnalogType>(flipValue.getType()))
    return tryEliminatingAttachesToAnalogValue(flipValue, insertPoint);

  Operation *connectOp = nullptr;
  for (auto &use : flipValue.getUses()) {
    // We only know how to deal with connects where this value is the
    // destination.
    if (use.getOperandNumber() != 0)
      return {};
    if (!isa<ConnectOp, MatchingConnectOp>(use.getOwner()))
      return {};

    // We only support things with a single connect.
    if (connectOp)
      return {};
    connectOp = use.getOwner();
  }

  // We don't have an HW equivalent of "poison" so just don't special case
  // the case where there are no connects other uses of an output.
  if (!connectOp)
    return {}; // TODO: Emit an sv.constant here since it is unconnected.

  // Don't special case zero-bit results.
  auto loweredType =
      loweringState.lowerType(flipValue.getType(), flipValue.getLoc());
  if (loweredType.isInteger(0))
    return {};

  // Convert each connect into an extended version of its operand being
  // output.
  ImplicitLocOpBuilder builder(insertPoint->getLoc(), insertPoint);

  auto connectSrc = connectOp->getOperand(1);

  // Directly forward foreign types.
  if (!isa<FIRRTLType>(connectSrc.getType())) {
    connectOp->erase();
    return connectSrc;
  }

  // Convert fliped sources to passive sources.
  if (!type_cast<FIRRTLBaseType>(connectSrc.getType()).isPassive())
    connectSrc =
        mlir::UnrealizedConversionCastOp::create(
            builder,
            type_cast<FIRRTLBaseType>(connectSrc.getType()).getPassiveType(),
            connectSrc)
            .getResult(0);

  // We know it must be the destination operand due to the types, but the
  // source may not match the destination width.
  auto destTy = type_cast<FIRRTLBaseType>(flipValue.getType()).getPassiveType();

  if (destTy != connectSrc.getType() &&
      (isa<BaseTypeAliasType>(connectSrc.getType()) ||
       isa<BaseTypeAliasType>(destTy))) {
    connectSrc =
        builder.createOrFold<BitCastOp>(flipValue.getType(), connectSrc);
  }
  if (!destTy.isGround()) {
    // If types are not ground type and they don't match, we give up.
    if (destTy != type_cast<FIRRTLType>(connectSrc.getType()))
      return {};
  } else if (destTy.getBitWidthOrSentinel() !=
             type_cast<FIRRTLBaseType>(connectSrc.getType())
                 .getBitWidthOrSentinel()) {
    // The only type mismatchs we care about is due to integer width
    // differences.
    auto destWidth = destTy.getBitWidthOrSentinel();
    assert(destWidth != -1 && "must know integer widths");
    connectSrc = builder.createOrFold<PadPrimOp>(destTy, connectSrc, destWidth);
  }

  // Remove the connect and use its source as the value for the output.
  connectOp->erase();

  // Convert from FIRRTL type to builtin type.
  return castFromFIRRTLType(connectSrc, loweredType, builder);
}

static SmallVector<SubfieldOp> getAllFieldAccesses(Value structValue,
                                                   StringRef field) {
  SmallVector<SubfieldOp> accesses;
  for (auto *op : structValue.getUsers()) {
    assert(isa<SubfieldOp>(op));
    auto fieldAccess = cast<SubfieldOp>(op);
    auto elemIndex =
        fieldAccess.getInput().getType().base().getElementIndex(field);
    if (elemIndex && *elemIndex == fieldAccess.getFieldIndex())
      accesses.push_back(fieldAccess);
  }
  return accesses;
}

/// Now that we have the operations for the hw.module's corresponding to the
/// firrtl.module's, we can go through and move the bodies over, updating the
/// ports and output op.
LogicalResult FIRRTLModuleLowering::lowerModulePortsAndMoveBody(
    FModuleOp oldModule, hw::HWModuleOp newModule,
    CircuitLoweringState &loweringState) {
  ImplicitLocOpBuilder bodyBuilder(oldModule.getLoc(), newModule.getBody());

  // Use a placeholder instruction be a cursor that indicates where we want to
  // move the new function body to.  This is important because we insert some
  // ops at the start of the function and some at the end, and the body is
  // currently empty to avoid iterator invalidation.
  auto cursor = hw::ConstantOp::create(bodyBuilder, APInt(1, 1));
  bodyBuilder.setInsertionPoint(cursor);

  // Insert argument casts, and re-vector users in the old body to use them.
  SmallVector<PortInfo> firrtlPorts = oldModule.getPorts();
  assert(oldModule.getBody().getNumArguments() == firrtlPorts.size() &&
         "port count mismatch");

  SmallVector<Value, 4> outputs;

  // This is the terminator in the new module.
  auto *outputOp = newModule.getBodyBlock()->getTerminator();
  ImplicitLocOpBuilder outputBuilder(oldModule.getLoc(), outputOp);

  unsigned nextHWInputArg = 0;
  int hwPortIndex = -1;
  for (auto [firrtlPortIndex, port] : llvm::enumerate(firrtlPorts)) {
    // Inputs and outputs are both modeled as arguments in the FIRRTL level.
    auto oldArg = oldModule.getBody().getArgument(firrtlPortIndex);

    bool isZeroWidth =
        type_isa<FIRRTLBaseType>(port.type) &&
        type_cast<FIRRTLBaseType>(port.type).getBitWidthOrSentinel() == 0;
    if (!isZeroWidth)
      ++hwPortIndex;

    if (!port.isOutput() && !isZeroWidth) {
      // Inputs and InOuts are modeled as arguments in the result, so we can
      // just map them over.  We model zero bit outputs as inouts.
      Value newArg = newModule.getBody().getArgument(nextHWInputArg++);

      // Cast the argument to the old type, reintroducing sign information in
      // the hw.module body.
      newArg = castToFIRRTLType(newArg, oldArg.getType(), bodyBuilder);
      // Switch all uses of the old operands to the new ones.
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }

    // We lower zero width inout and outputs to a wire that isn't connected to
    // anything outside the module.  Inputs are lowered to zero.
    if (isZeroWidth && port.isInput()) {
      Value newArg =
          WireOp::create(bodyBuilder, port.type,
                         "." + port.getName().str() + ".0width_input")
              .getResult();
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }

    if (auto value =
            tryEliminatingConnectsToValue(oldArg, outputOp, loweringState)) {
      // If we were able to find the value being connected to the output,
      // directly use it!
      outputs.push_back(value);
      assert(oldArg.use_empty() && "should have removed all uses of oldArg");
      continue;
    }

    // Outputs need a temporary wire so they can be connect'd to, which we
    // then return.
    auto newArg = WireOp::create(bodyBuilder, port.type,
                                 "." + port.getName().str() + ".output");

    // Switch all uses of the old operands to the new ones.
    oldArg.replaceAllUsesWith(newArg.getResult());

    // Don't output zero bit results or inouts.
    auto resultHWType = loweringState.lowerType(port.type, port.loc);
    if (!resultHWType.isInteger(0)) {
      auto output =
          castFromFIRRTLType(newArg.getResult(), resultHWType, outputBuilder);
      outputs.push_back(output);

      // If output port has symbol, move it to this wire.
      if (auto sym = newModule.getPort(hwPortIndex).getSym()) {
        newArg.setInnerSymAttr(sym);
        newModule.setPortSymbolAttr(hwPortIndex, {});
      }
    }
  }

  // Update the hw.output terminator with the list of outputs we have.
  outputOp->setOperands(outputs);

  // Finally splice the body over, don't move the old terminator over though.
  auto &oldBlockInstList = oldModule.getBodyBlock()->getOperations();
  auto &newBlockInstList = newModule.getBodyBlock()->getOperations();
  newBlockInstList.splice(Block::iterator(cursor), oldBlockInstList,
                          oldBlockInstList.begin(), oldBlockInstList.end());

  // We are done with our cursor op.
  cursor.erase();

  return success();
}

/// Run on each `verif.formal` to populate its body based on the original
/// `firrtl.formal` operation.
LogicalResult
FIRRTLModuleLowering::lowerFormalBody(verif::FormalOp newOp,
                                      CircuitLoweringState &loweringState) {
  auto builder = OpBuilder::atBlockEnd(&newOp.getBody().front());

  // Find the module targeted by the `firrtl.formal` operation. The `FormalOp`
  // verifier guarantees the module exists and that it is an `FModuleOp`. This
  // we can then translate to the corresponding `HWModuleOp`.
  auto oldOp = cast<FormalOp>(loweringState.getOldModule(newOp));
  auto moduleName = oldOp.getModuleNameAttr().getAttr();
  auto oldModule = cast<FModuleOp>(
      loweringState.getInstanceGraph().lookup(moduleName)->getModule());
  auto newModule = cast<hw::HWModuleOp>(loweringState.getNewModule(oldModule));

  // Create a symbolic input for every input of the lowered module.
  SmallVector<Value> symbolicInputs;
  for (auto arg : newModule.getBody().getArguments())
    symbolicInputs.push_back(
        verif::SymbolicValueOp::create(builder, arg.getLoc(), arg.getType()));

  // Instantiate the module with the given symbolic inputs.
  hw::InstanceOp::create(builder, newOp.getLoc(), newModule,
                         newModule.getNameAttr(), symbolicInputs);
  return success();
}

/// Run on each `verif.simulation` to populate its body based on the original
/// `firrtl.simulation` operation.
LogicalResult
FIRRTLModuleLowering::lowerSimulationBody(verif::SimulationOp newOp,
                                          CircuitLoweringState &loweringState) {
  auto builder = OpBuilder::atBlockEnd(newOp.getBody());

  // Find the module targeted by the `firrtl.simulation` operation.
  auto oldOp = cast<SimulationOp>(loweringState.getOldModule(newOp));
  auto moduleName = oldOp.getModuleNameAttr().getAttr();
  auto oldModule = cast<FModuleLike>(
      *loweringState.getInstanceGraph().lookup(moduleName)->getModule());
  auto newModule =
      cast<hw::HWModuleLike>(loweringState.getNewModule(oldModule));

  // Instantiate the module with the simulation op's block arguments as inputs,
  // and yield the module's outputs.
  SmallVector<Value> inputs(newOp.getBody()->args_begin(),
                            newOp.getBody()->args_end());
  auto instOp = hw::InstanceOp::create(builder, newOp.getLoc(), newModule,
                                       newModule.getNameAttr(), inputs);
  verif::YieldOp::create(builder, newOp.getLoc(), instOp.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// Module Body Lowering Pass
//===----------------------------------------------------------------------===//

namespace {

struct FIRRTLLowering : public FIRRTLVisitor<FIRRTLLowering, LogicalResult> {

  FIRRTLLowering(hw::HWModuleOp module, CircuitLoweringState &circuitState)
      : theModule(module), circuitState(circuitState),
        builder(module.getLoc(), module.getContext()), moduleNamespace(module),
        backedgeBuilder(builder, module.getLoc()) {}

  LogicalResult run();

  // Helpers.
  Value getOrCreateClockConstant(seq::ClockConst clock);
  Value getOrCreateIntConstant(const APInt &value);
  Value getOrCreateIntConstant(unsigned numBits, uint64_t val,
                               bool isSigned = false) {
    return getOrCreateIntConstant(APInt(numBits, val, isSigned));
  }
  Attribute getOrCreateAggregateConstantAttribute(Attribute value, Type type);
  Value getOrCreateXConstant(unsigned numBits);
  Value getOrCreateZConstant(Type type);
  Value getPossiblyInoutLoweredValue(Value value);
  Value getLoweredValue(Value value);
  Value getLoweredNonClockValue(Value value);
  Value getLoweredAndExtendedValue(Value value, Type destType);
  Value getLoweredAndExtOrTruncValue(Value value, Type destType);
  LogicalResult setLowering(Value orig, Value result);
  LogicalResult setPossiblyFoldedLowering(Value orig, Value result);
  template <typename ResultOpType, typename... CtorArgTypes>
  LogicalResult setLoweringTo(Operation *orig, CtorArgTypes... args);
  template <typename ResultOpType, typename... CtorArgTypes>
  LogicalResult setLoweringToLTL(Operation *orig, CtorArgTypes... args);
  Backedge createBackedge(Location loc, Type type);
  Backedge createBackedge(Value orig, Type type);
  bool updateIfBackedge(Value dest, Value src);

  /// Returns true if the lowered operation requires an inner symbol on it.
  bool requiresInnerSymbol(hw::InnerSymbolOpInterface op) {
    if (AnnotationSet::removeDontTouch(op))
      return true;
    if (!hasDroppableName(op))
      return true;
    if (auto forceable = dyn_cast<Forceable>(op.getOperation()))
      if (forceable.isForceable())
        return true;
    return false;
  }

  /// Gets the lowered InnerSymAttr of this operation.  If the operation is
  /// DontTouched, has a non-droppable name, or is forceable, then we will
  /// ensure that the InnerSymAttr has a symbol with fieldID zero.
  hw::InnerSymAttr lowerInnerSymbol(hw::InnerSymbolOpInterface op) {
    auto attr = op.getInnerSymAttr();
    // TODO: we should be checking for symbol collisions here and renaming as
    // neccessary. As well, we should record the renamings in a map so that we
    // can update any InnerRefAttrs that we find.
    if (requiresInnerSymbol(op))
      std::tie(attr, std::ignore) = getOrAddInnerSym(
          op.getContext(), attr, 0,
          [&]() -> hw::InnerSymbolNamespace & { return moduleNamespace; });
    return attr;
  }

  void runWithInsertionPointAtEndOfBlock(const std::function<void(void)> &fn,
                                         Region &region);

  /// Return a read value for the specified inout value, auto-uniquing them.
  Value getReadValue(Value v);
  /// Return an `i1` value for the specified value, auto-uniqueing them.
  Value getNonClockValue(Value v);

  void addToAlwaysBlock(sv::EventControl clockEdge, Value clock,
                        sv::ResetType resetStyle, sv::EventControl resetEdge,
                        Value reset, const std::function<void(void)> &body = {},
                        const std::function<void(void)> &resetBody = {});
  void addToAlwaysBlock(Value clock,
                        const std::function<void(void)> &body = {}) {
    addToAlwaysBlock(sv::EventControl::AtPosEdge, clock, sv::ResetType(),
                     sv::EventControl(), Value(), body,
                     std::function<void(void)>());
  }

  LogicalResult emitGuards(Location loc, ArrayRef<Attribute> guards,
                           std::function<void(void)> emit);
  void addToIfDefBlock(StringRef cond, std::function<void(void)> thenCtor,
                       std::function<void(void)> elseCtor = {});
  void addToInitialBlock(std::function<void(void)> body);
  void addIfProceduralBlock(Value cond, std::function<void(void)> thenCtor,
                            std::function<void(void)> elseCtor = {});
  Value getExtOrTruncAggregateValue(Value array, FIRRTLBaseType sourceType,
                                    FIRRTLBaseType destType,
                                    bool allowTruncate);
  Value createArrayIndexing(Value array, Value index);
  Value createValueWithMuxAnnotation(Operation *op, bool isMux2);

  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitExpr;
  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitDecl;
  using FIRRTLVisitor<FIRRTLLowering, LogicalResult>::visitStmt;

  // Lowering hooks.
  enum UnloweredOpResult { AlreadyLowered, NowLowered, LoweringFailure };
  UnloweredOpResult handleUnloweredOp(Operation *op);
  LogicalResult visitExpr(ConstantOp op);
  LogicalResult visitExpr(SpecialConstantOp op);
  LogicalResult visitExpr(SubindexOp op);
  LogicalResult visitExpr(SubaccessOp op);
  LogicalResult visitExpr(SubfieldOp op);
  LogicalResult visitExpr(VectorCreateOp op);
  LogicalResult visitExpr(BundleCreateOp op);
  LogicalResult visitExpr(FEnumCreateOp op);
  LogicalResult visitExpr(AggregateConstantOp op);
  LogicalResult visitExpr(IsTagOp op);
  LogicalResult visitExpr(SubtagOp op);
  LogicalResult visitExpr(TagExtractOp op);
  LogicalResult visitUnhandledOp(Operation *op) { return failure(); }
  LogicalResult visitInvalidOp(Operation *op) {
    if (auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(op))
      return visitUnrealizedConversionCast(castOp);
    return failure();
  }

  // Declarations.
  LogicalResult visitDecl(WireOp op);
  LogicalResult visitDecl(NodeOp op);
  LogicalResult visitDecl(RegOp op);
  LogicalResult visitDecl(RegResetOp op);
  LogicalResult visitDecl(MemOp op);
  LogicalResult visitDecl(InstanceOp oldInstance);
  LogicalResult visitDecl(VerbatimWireOp op);
  LogicalResult visitDecl(ContractOp op);

  // Unary Ops.
  LogicalResult lowerNoopCast(Operation *op);
  LogicalResult visitExpr(AsSIntPrimOp op);
  LogicalResult visitExpr(AsUIntPrimOp op);
  LogicalResult visitExpr(AsClockPrimOp op);
  LogicalResult visitExpr(AsAsyncResetPrimOp op) { return lowerNoopCast(op); }

  LogicalResult visitExpr(HWStructCastOp op);
  LogicalResult visitExpr(BitCastOp op);
  LogicalResult
  visitUnrealizedConversionCast(mlir::UnrealizedConversionCastOp op);
  LogicalResult visitExpr(CvtPrimOp op);
  LogicalResult visitExpr(NotPrimOp op);
  LogicalResult visitExpr(NegPrimOp op);
  LogicalResult visitExpr(PadPrimOp op);
  LogicalResult visitExpr(XorRPrimOp op);
  LogicalResult visitExpr(AndRPrimOp op);
  LogicalResult visitExpr(OrRPrimOp op);

  // Binary Ops.
  template <typename ResultUnsignedOpType,
            typename ResultSignedOpType = ResultUnsignedOpType>
  LogicalResult lowerBinOp(Operation *op);
  template <typename ResultOpType>
  LogicalResult lowerBinOpToVariadic(Operation *op);

  template <typename ResultOpType>
  LogicalResult lowerElementwiseLogicalOp(Operation *op);

  LogicalResult lowerCmpOp(Operation *op, ICmpPredicate signedOp,
                           ICmpPredicate unsignedOp);
  template <typename SignedOp, typename UnsignedOp>
  LogicalResult lowerDivLikeOp(Operation *op);

  LogicalResult visitExpr(CatPrimOp op);

  LogicalResult visitExpr(AndPrimOp op) {
    return lowerBinOpToVariadic<comb::AndOp>(op);
  }
  LogicalResult visitExpr(OrPrimOp op) {
    return lowerBinOpToVariadic<comb::OrOp>(op);
  }
  LogicalResult visitExpr(XorPrimOp op) {
    return lowerBinOpToVariadic<comb::XorOp>(op);
  }
  LogicalResult visitExpr(ElementwiseOrPrimOp op) {
    return lowerElementwiseLogicalOp<comb::OrOp>(op);
  }
  LogicalResult visitExpr(ElementwiseAndPrimOp op) {
    return lowerElementwiseLogicalOp<comb::AndOp>(op);
  }
  LogicalResult visitExpr(ElementwiseXorPrimOp op) {
    return lowerElementwiseLogicalOp<comb::XorOp>(op);
  }
  LogicalResult visitExpr(AddPrimOp op) {
    return lowerBinOpToVariadic<comb::AddOp>(op);
  }
  LogicalResult visitExpr(EQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::eq, ICmpPredicate::eq);
  }
  LogicalResult visitExpr(NEQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::ne, ICmpPredicate::ne);
  }
  LogicalResult visitExpr(LTPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::slt, ICmpPredicate::ult);
  }
  LogicalResult visitExpr(LEQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::sle, ICmpPredicate::ule);
  }
  LogicalResult visitExpr(GTPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::sgt, ICmpPredicate::ugt);
  }
  LogicalResult visitExpr(GEQPrimOp op) {
    return lowerCmpOp(op, ICmpPredicate::sge, ICmpPredicate::uge);
  }

  LogicalResult visitExpr(SubPrimOp op) { return lowerBinOp<comb::SubOp>(op); }
  LogicalResult visitExpr(MulPrimOp op) {
    return lowerBinOpToVariadic<comb::MulOp>(op);
  }
  LogicalResult visitExpr(DivPrimOp op) {
    return lowerDivLikeOp<comb::DivSOp, comb::DivUOp>(op);
  }
  LogicalResult visitExpr(RemPrimOp op) {
    return lowerDivLikeOp<comb::ModSOp, comb::ModUOp>(op);
  }

  // Intrinsic Operations
  LogicalResult visitExpr(IsXIntrinsicOp op);
  LogicalResult visitExpr(PlusArgsTestIntrinsicOp op);
  LogicalResult visitExpr(PlusArgsValueIntrinsicOp op);
  LogicalResult visitStmt(FPGAProbeIntrinsicOp op);
  LogicalResult visitExpr(ClockInverterIntrinsicOp op);
  LogicalResult visitExpr(ClockDividerIntrinsicOp op);
  LogicalResult visitExpr(SizeOfIntrinsicOp op);
  LogicalResult visitExpr(ClockGateIntrinsicOp op);
  LogicalResult visitExpr(LTLAndIntrinsicOp op);
  LogicalResult visitExpr(LTLOrIntrinsicOp op);
  LogicalResult visitExpr(LTLIntersectIntrinsicOp op);
  LogicalResult visitExpr(LTLDelayIntrinsicOp op);
  LogicalResult visitExpr(LTLConcatIntrinsicOp op);
  LogicalResult visitExpr(LTLRepeatIntrinsicOp op);
  LogicalResult visitExpr(LTLGoToRepeatIntrinsicOp op);
  LogicalResult visitExpr(LTLNonConsecutiveRepeatIntrinsicOp op);
  LogicalResult visitExpr(LTLNotIntrinsicOp op);
  LogicalResult visitExpr(LTLImplicationIntrinsicOp op);
  LogicalResult visitExpr(LTLUntilIntrinsicOp op);
  LogicalResult visitExpr(LTLEventuallyIntrinsicOp op);
  LogicalResult visitExpr(LTLClockIntrinsicOp op);

  template <typename TargetOp, typename IntrinsicOp>
  LogicalResult lowerVerifIntrinsicOp(IntrinsicOp op);
  LogicalResult visitStmt(VerifAssertIntrinsicOp op);
  LogicalResult visitStmt(VerifAssumeIntrinsicOp op);
  LogicalResult visitStmt(VerifCoverIntrinsicOp op);
  LogicalResult visitStmt(VerifRequireIntrinsicOp op);
  LogicalResult visitStmt(VerifEnsureIntrinsicOp op);
  LogicalResult visitExpr(HasBeenResetIntrinsicOp op);
  LogicalResult visitStmt(UnclockedAssumeIntrinsicOp op);

  // Other Operations
  LogicalResult visitExpr(BitsPrimOp op);
  LogicalResult visitExpr(InvalidValueOp op);
  LogicalResult visitExpr(HeadPrimOp op);
  LogicalResult visitExpr(ShlPrimOp op);
  LogicalResult visitExpr(ShrPrimOp op);
  LogicalResult visitExpr(DShlPrimOp op) {
    return lowerDivLikeOp<comb::ShlOp, comb::ShlOp>(op);
  }
  LogicalResult visitExpr(DShrPrimOp op) {
    return lowerDivLikeOp<comb::ShrSOp, comb::ShrUOp>(op);
  }
  LogicalResult visitExpr(DShlwPrimOp op) {
    return lowerDivLikeOp<comb::ShlOp, comb::ShlOp>(op);
  }
  LogicalResult visitExpr(TailPrimOp op);
  LogicalResult visitExpr(MuxPrimOp op);
  LogicalResult visitExpr(Mux2CellIntrinsicOp op);
  LogicalResult visitExpr(Mux4CellIntrinsicOp op);
  LogicalResult visitExpr(MultibitMuxOp op);
  LogicalResult visitExpr(VerbatimExprOp op);
  LogicalResult visitExpr(XMRRefOp op);
  LogicalResult visitExpr(XMRDerefOp op);

  // Format String Operations
  LogicalResult visitExpr(TimeOp op);
  LogicalResult visitExpr(HierarchicalModuleNameOp op);

  // Statements
  LogicalResult lowerVerificationStatement(
      Operation *op, StringRef labelPrefix, Value clock, Value predicate,
      Value enable, StringAttr messageAttr, ValueRange operands,
      StringAttr nameAttr, bool isConcurrent, EventControl eventControl);

  LogicalResult visitStmt(SkipOp op);

  FailureOr<bool> lowerConnect(Value dest, Value srcVal);
  LogicalResult visitStmt(ConnectOp op);
  LogicalResult visitStmt(MatchingConnectOp op);
  LogicalResult visitStmt(ForceOp op);

  std::optional<Value> getLoweredFmtOperand(Value operand);
  LogicalResult loweredFmtOperands(ValueRange operands,
                                   SmallVectorImpl<Value> &loweredOperands);
  FailureOr<Value> callFileDescriptorLib(const FileDescriptorInfo &info);
  // Lower statemens that use file descriptors such as printf, fprintf and
  // fflush. `fn` is a function that takes a file descriptor and build an always
  // and if-procedural block.
  LogicalResult lowerStatementWithFd(
      const FileDescriptorInfo &fileDescriptorInfo, Value clock, Value cond,
      const std::function<LogicalResult(Value)> &fn, bool usePrintfCond);
  // Lower a printf-like operation. `fileDescriptorInfo` is a pair of the
  // file name and whether it requires format string substitution.
  template <class T>
  LogicalResult visitPrintfLike(T op,
                                const FileDescriptorInfo &fileDescriptorInfo,
                                bool usePrintfCond);
  LogicalResult visitStmt(PrintFOp op) { return visitPrintfLike(op, {}, true); }
  LogicalResult visitStmt(FPrintFOp op);
  LogicalResult visitStmt(FFlushOp op);
  LogicalResult visitStmt(StopOp op);
  LogicalResult visitStmt(AssertOp op);
  LogicalResult visitStmt(AssumeOp op);
  LogicalResult visitStmt(CoverOp op);
  LogicalResult visitStmt(AttachOp op);
  LogicalResult visitStmt(RefForceOp op);
  LogicalResult visitStmt(RefForceInitialOp op);
  LogicalResult visitStmt(RefReleaseOp op);
  LogicalResult visitStmt(RefReleaseInitialOp op);
  LogicalResult visitStmt(BindOp op);

  FailureOr<Value> lowerSubindex(SubindexOp op, Value input);
  FailureOr<Value> lowerSubaccess(SubaccessOp op, Value input);
  FailureOr<Value> lowerSubfield(SubfieldOp op, Value input);

  LogicalResult fixupLTLOps();

  Type lowerType(Type type) {
    return circuitState.lowerType(type, builder.getLoc());
  }

private:
  /// The module we're lowering into.
  hw::HWModuleOp theModule;

  /// Global state.
  CircuitLoweringState &circuitState;

  /// This builder is set to the right location for each visit call.
  ImplicitLocOpBuilder builder;

  /// Each value lowered (e.g. operation result) is kept track in this map.
  /// The key should have a FIRRTL type, the result will have an HW dialect
  /// type.
  DenseMap<Value, Value> valueMapping;

  /// Mapping from clock values to corresponding non-clock values converted
  /// via a deduped `seq.from_clock` op.
  DenseMap<Value, Value> fromClockMapping;

  /// This keeps track of constants that we have created so we can reuse them.
  /// This is populated by the getOrCreateIntConstant method.
  DenseMap<Attribute, Value> hwConstantMap;
  DenseMap<std::pair<Attribute, Type>, Attribute> hwAggregateConstantMap;

  /// This keeps track of constant X that we have created so we can reuse them.
  /// This is populated by the getOrCreateXConstant method.
  DenseMap<unsigned, Value> hwConstantXMap;
  DenseMap<Type, Value> hwConstantZMap;

  /// We auto-unique "ReadInOut" ops from wires and regs, enabling
  /// optimizations and CSEs of the read values to be more obvious.  This
  /// caches a known ReadInOutOp for the given value and is managed by
  /// `getReadValue(v)`.
  DenseMap<Value, Value> readInOutCreated;

  /// This keeps track of the file descriptors for each file name.
  DenseMap<StringAttr, sv::RegOp> fileNameToFileDescriptor;

  // We auto-unique graph-level blocks to reduce the amount of generated
  // code and ensure that side effects are properly ordered in FIRRTL.
  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value,
                                   sv::ResetType, sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;
  llvm::SmallDenseMap<std::pair<Block *, Attribute>, sv::IfDefOp> ifdefBlocks;
  llvm::SmallDenseMap<Block *, sv::InitialOp> initialBlocks;

  /// A namespace that can be used to generate new symbol names that are unique
  /// within this module.
  hw::InnerSymbolNamespace moduleNamespace;

  /// A backedge builder to directly materialize values during the lowering
  /// without requiring temporary wires.
  BackedgeBuilder backedgeBuilder;
  /// Currently unresolved backedges. More precisely, a mapping from the
  /// backedge value to the value it will be replaced with. We use a MapVector
  /// so that a combinational cycles of backedges, the one backedge that gets
  /// replaced with an undriven wire is consistent.
  llvm::MapVector<Value, Value> backedges;

  /// A collection of values generated by the lowering process that may have
  /// become obsolete through subsequent parts of the lowering. This covers the
  /// values of wires that may be overridden by subsequent connects; or
  /// subaccesses that appear only as destination of a connect, and thus gets
  /// obsoleted by the connect directly updating the wire or register.
  DenseSet<Operation *> maybeUnusedValues;

  void maybeUnused(Operation *op) { maybeUnusedValues.insert(op); }
  void maybeUnused(Value value) {
    if (auto *op = value.getDefiningOp())
      maybeUnused(op);
  }

  /// A worklist of LTL operations that don't have their final type yet. The
  /// FIRRTL intrinsics for LTL ops all use `uint<1>` types, but the actual LTL
  /// ops themselves have more precise `!ltl.sequence` and `!ltl.property`
  /// types. After all LTL ops have been lowered, this worklist is used to
  /// compute their actual types (re-inferring return types) and push the
  /// updated types to their users. This also drops any `hw.wire`s in between
  /// the LTL ops, which were necessary to go from the def-before-use FIRRTL
  /// dialect to the graph-like HW dialect.
  SetVector<Operation *> ltlOpFixupWorklist;

  /// A worklist of operation ranges to be lowered. Parnet operations can push
  /// their nested operations onto this worklist to be processed after the
  /// parent operation has handled the region, blocks, and block arguments.
  SmallVector<std::pair<Block::iterator, Block::iterator>> worklist;

  void addToWorklist(Block &block) {
    worklist.push_back({block.begin(), block.end()});
  }
  void addToWorklist(Region &region) {
    for (auto &block : llvm::reverse(region))
      addToWorklist(block);
  }
};
} // end anonymous namespace

LogicalResult
FIRRTLModuleLowering::lowerModuleBody(hw::HWModuleOp module,
                                      CircuitLoweringState &loweringState) {
  return FIRRTLLowering(module, loweringState).run();
}

LogicalResult FIRRTLModuleLowering::lowerFileBody(emit::FileOp fileOp) {
  OpBuilder b(&getContext());
  fileOp->walk([&](Operation *op) {
    if (auto bindOp = dyn_cast<BindOp>(op)) {
      b.setInsertionPointAfter(bindOp);
      sv::BindOp::create(b, bindOp.getLoc(), bindOp.getInstanceAttr());
      bindOp->erase();
    }
  });
  return success();
}

LogicalResult
FIRRTLModuleLowering::lowerBody(Operation *op,
                                CircuitLoweringState &loweringState) {
  if (auto moduleOp = dyn_cast<hw::HWModuleOp>(op))
    return lowerModuleBody(moduleOp, loweringState);
  if (auto formalOp = dyn_cast<verif::FormalOp>(op))
    return lowerFormalBody(formalOp, loweringState);
  if (auto simulationOp = dyn_cast<verif::SimulationOp>(op))
    return lowerSimulationBody(simulationOp, loweringState);
  if (auto fileOp = dyn_cast<emit::FileOp>(op))
    return lowerFileBody(fileOp);
  return failure();
}

// This is the main entrypoint for the lowering pass.
LogicalResult FIRRTLLowering::run() {
  // Mark the module's block arguments are already lowered. This will allow
  // `getLoweredValue` to return the block arguments as they are.
  for (auto arg : theModule.getBodyBlock()->getArguments())
    if (failed(setLowering(arg, arg)))
      return failure();

  // Add the operations in the body to the worklist and lower all operations
  // until the worklist is empty. Operations may push their own nested
  // operations onto the worklist to lower them in turn. The `builder` is
  // positioned ahead of each operation as it is being lowered.
  addToWorklist(theModule.getBody());
  SmallVector<Operation *, 16> opsToRemove;

  while (!worklist.empty()) {
    auto &[opsIt, opsEnd] = worklist.back();
    if (opsIt == opsEnd) {
      worklist.pop_back();
      continue;
    }
    Operation *op = &*opsIt++;

    builder.setInsertionPoint(op);
    builder.setLoc(op->getLoc());
    auto done = succeeded(dispatchVisitor(op));
    circuitState.processRemainingAnnotations(op, AnnotationSet(op));
    if (done)
      opsToRemove.push_back(op);
    else {
      switch (handleUnloweredOp(op)) {
      case AlreadyLowered:
        break;         // Something like hw.output, which is already lowered.
      case NowLowered: // Something handleUnloweredOp removed.
        opsToRemove.push_back(op);
        break;
      case LoweringFailure:
        backedgeBuilder.abandon();
        return failure();
      }
    }
  }

  // Replace all backedges with uses of their regular values.  We process them
  // after the module body since the lowering table is too hard to keep up to
  // date.  Multiple operations may be lowered to the same backedge when values
  // are folded, which means we would have to scan the entire lowering table to
  // safely replace a backedge.
  for (auto &[backedge, value] : backedges) {
    SmallVector<Location> driverLocs;
    // In the case where we have backedges connected to other backedges, we have
    // to find the value that actually drives the group.
    while (true) {
      // If we find the original backedge we have some undriven logic or
      // a combinatorial loop. Bail out and provide information on the nodes.
      if (backedge == value) {
        Location edgeLoc = backedge.getLoc();
        if (driverLocs.empty()) {
          mlir::emitError(edgeLoc, "sink does not have a driver");
        } else {
          auto diag = mlir::emitError(edgeLoc, "sink in combinational loop");
          for (auto loc : driverLocs)
            diag.attachNote(loc) << "through driver here";
        }
        backedgeBuilder.abandon();
        return failure();
      }
      // If the value is not another backedge, we have found the driver.
      auto *it = backedges.find(value);
      if (it == backedges.end())
        break;
      // Find what is driving the next backedge.
      driverLocs.push_back(value.getLoc());
      value = it->second;
    }
    if (auto *defOp = backedge.getDefiningOp())
      maybeUnusedValues.erase(defOp);
    backedge.replaceAllUsesWith(value);
  }

  // Now that all of the operations that can be lowered are, remove th
  // original values.  We know that any lowered operations will be dead (if
  // removed in reverse order) at this point - any users of them from
  // unremapped operations will be changed to use the newly lowered ops.
  hw::ConstantOp zeroI0;
  while (!opsToRemove.empty()) {
    auto *op = opsToRemove.pop_back_val();

    // We remove zero-width values when lowering FIRRTL ops. We can't remove
    // such a value if it escapes to a foreign op. In that case, create an
    // `hw.constant 0 : i0` to pass along.
    for (auto result : op->getResults()) {
      if (!isZeroBitFIRRTLType(result.getType()))
        continue;
      if (!zeroI0) {
        auto builder = OpBuilder::atBlockBegin(theModule.getBodyBlock());
        zeroI0 = hw::ConstantOp::create(builder, op->getLoc(),
                                        builder.getIntegerType(0), 0);
        maybeUnusedValues.insert(zeroI0);
      }
      result.replaceAllUsesWith(zeroI0);
    }

    if (!op->use_empty()) {
      auto d = op->emitOpError(
          "still has uses; should remove ops in reverse order of visitation");
      SmallPtrSet<Operation *, 2> visited;
      for (auto *user : op->getUsers())
        if (visited.insert(user).second)
          d.attachNote(user->getLoc())
              << "used by " << user->getName() << " op";
      return d;
    }
    maybeUnusedValues.erase(op);
    op->erase();
  }

  // Prune operations that may have become unused throughout the lowering.
  while (!maybeUnusedValues.empty()) {
    auto it = maybeUnusedValues.begin();
    auto *op = *it;
    maybeUnusedValues.erase(it);
    if (!isOpTriviallyDead(op))
      continue;
    for (auto operand : op->getOperands())
      if (auto *defOp = operand.getDefiningOp())
        maybeUnusedValues.insert(defOp);
    op->erase();
  }

  // Determine the actual types of lowered LTL operations and remove any
  // intermediate wires among them.
  if (failed(fixupLTLOps()))
    return failure();

  return backedgeBuilder.clearOrEmitError();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create uniqued constant clocks.
Value FIRRTLLowering::getOrCreateClockConstant(seq::ClockConst clock) {
  auto attr = seq::ClockConstAttr::get(theModule.getContext(), clock);

  auto &entry = hwConstantMap[attr];
  if (entry)
    return entry;

  OpBuilder entryBuilder(&theModule.getBodyBlock()->front());
  entry = seq::ConstClockOp::create(entryBuilder, builder.getLoc(), attr);
  return entry;
}

/// Check to see if we've already lowered the specified constant.  If so,
/// return it.  Otherwise create it and put it in the entry block for reuse.
Value FIRRTLLowering::getOrCreateIntConstant(const APInt &value) {
  auto attr = builder.getIntegerAttr(
      builder.getIntegerType(value.getBitWidth()), value);

  auto &entry = hwConstantMap[attr];
  if (entry)
    return entry;

  OpBuilder entryBuilder(&theModule.getBodyBlock()->front());
  entry = hw::ConstantOp::create(entryBuilder, builder.getLoc(), attr);
  return entry;
}

/// Check to see if we've already created the specified aggregate constant
/// attribute. If so, return it.  Otherwise create it.
Attribute FIRRTLLowering::getOrCreateAggregateConstantAttribute(Attribute value,
                                                                Type type) {
  // Base case.
  if (hw::type_isa<IntegerType>(type))
    return builder.getIntegerAttr(type, cast<IntegerAttr>(value).getValue());

  auto cache = hwAggregateConstantMap.lookup({value, type});
  if (cache)
    return cache;

  // Recursively construct elements.
  SmallVector<Attribute> values;
  for (auto e : llvm::enumerate(cast<ArrayAttr>(value))) {
    Type subType;
    if (auto array = hw::type_dyn_cast<hw::ArrayType>(type))
      subType = array.getElementType();
    else if (auto structType = hw::type_dyn_cast<hw::StructType>(type))
      subType = structType.getElements()[e.index()].type;
    else
      assert(false && "type must be either array or struct");

    values.push_back(getOrCreateAggregateConstantAttribute(e.value(), subType));
  }

  // FIRRTL and HW have a different operand ordering for arrays.
  if (hw::type_isa<hw::ArrayType>(type))
    std::reverse(values.begin(), values.end());

  auto &entry = hwAggregateConstantMap[{value, type}];
  entry = builder.getArrayAttr(values);
  return entry;
}

/// Zero bit operands end up looking like failures from getLoweredValue.  This
/// helper function invokes the closure specified if the operand was actually
/// zero bit, or returns failure() if it was some other kind of failure.
static LogicalResult handleZeroBit(Value failedOperand,
                                   const std::function<LogicalResult()> &fn) {
  assert(failedOperand && "Should be called on the failed operand");
  if (!isZeroBitFIRRTLType(failedOperand.getType()))
    return failure();
  return fn();
}

/// Check to see if we've already lowered the specified constant.  If so,
/// return it.  Otherwise create it and put it in the entry block for reuse.
Value FIRRTLLowering::getOrCreateXConstant(unsigned numBits) {

  auto &entry = hwConstantXMap[numBits];
  if (entry)
    return entry;

  OpBuilder entryBuilder(&theModule.getBodyBlock()->front());
  entry = sv::ConstantXOp::create(entryBuilder, builder.getLoc(),
                                  entryBuilder.getIntegerType(numBits));
  return entry;
}

Value FIRRTLLowering::getOrCreateZConstant(Type type) {
  auto &entry = hwConstantZMap[type];
  if (!entry) {
    OpBuilder entryBuilder(&theModule.getBodyBlock()->front());
    entry = sv::ConstantZOp::create(entryBuilder, builder.getLoc(), type);
  }
  return entry;
}

/// Return the lowered HW value corresponding to the specified original value.
/// This returns a null value for FIRRTL values that haven't be lowered, e.g.
/// unknown width integers.  This returns hw::inout type values if present, it
/// does not implicitly read from them.
Value FIRRTLLowering::getPossiblyInoutLoweredValue(Value value) {
  // If we lowered this value, then return the lowered value, otherwise fail.
  if (auto lowering = valueMapping.lookup(value)) {
    assert(!isa<FIRRTLType>(lowering.getType()) &&
           "Lowered value should be a non-FIRRTL value");
    return lowering;
  }
  return Value();
}

/// Return the lowered value corresponding to the specified original value.
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredValue(Value value) {
  auto result = getPossiblyInoutLoweredValue(value);
  if (!result)
    return result;

  // If we got an inout value, implicitly read it.  FIRRTL allows direct use
  // of wires and other things that lower to inout type.
  if (isa<hw::InOutType>(result.getType()))
    return getReadValue(result);

  return result;
}

/// Return the lowered value, converting `seq.clock` to `i1.
Value FIRRTLLowering::getLoweredNonClockValue(Value value) {
  auto result = getLoweredValue(value);
  if (!result)
    return result;

  if (hw::type_isa<seq::ClockType>(result.getType()))
    return getNonClockValue(result);

  return result;
}

/// Return the lowered aggregate value whose type is converted into
/// `destType`. We have to care about the extension/truncation/signedness of
/// each element.
Value FIRRTLLowering::getExtOrTruncAggregateValue(Value array,
                                                  FIRRTLBaseType sourceType,
                                                  FIRRTLBaseType destType,
                                                  bool allowTruncate) {
  SmallVector<Value> resultBuffer;

  // Helper function to cast each element of array to dest type.
  auto cast = [&](Value value, FIRRTLBaseType sourceType,
                  FIRRTLBaseType destType) {
    auto srcWidth = firrtl::type_cast<IntType>(sourceType).getWidthOrSentinel();
    auto destWidth = firrtl::type_cast<IntType>(destType).getWidthOrSentinel();
    auto resultType = builder.getIntegerType(destWidth);

    if (srcWidth == destWidth)
      return value;

    if (srcWidth > destWidth) {
      if (allowTruncate)
        return builder.createOrFold<comb::ExtractOp>(resultType, value, 0);

      builder.emitError("operand should not be a truncation");
      return Value();
    }

    if (firrtl::type_cast<IntType>(sourceType).isSigned())
      return comb::createOrFoldSExt(value, resultType, builder);
    auto zero = getOrCreateIntConstant(destWidth - srcWidth, 0);
    return builder.createOrFold<comb::ConcatOp>(zero, value);
  };

  // This recursive function constructs the output array.
  std::function<LogicalResult(Value, FIRRTLBaseType, FIRRTLBaseType)> recurse =
      [&](Value src, FIRRTLBaseType srcType,
          FIRRTLBaseType destType) -> LogicalResult {
    return TypeSwitch<FIRRTLBaseType, LogicalResult>(srcType)
        .Case<FVectorType>([&](auto srcVectorType) {
          auto destVectorType = firrtl::type_cast<FVectorType>(destType);
          unsigned size = resultBuffer.size();
          unsigned indexWidth =
              getBitWidthFromVectorSize(srcVectorType.getNumElements());
          for (size_t i = 0, e = std::min(srcVectorType.getNumElements(),
                                          destVectorType.getNumElements());
               i != e; ++i) {
            auto iIdx = getOrCreateIntConstant(indexWidth, i);
            auto arrayIndex = hw::ArrayGetOp::create(builder, src, iIdx);
            if (failed(recurse(arrayIndex, srcVectorType.getElementType(),
                               destVectorType.getElementType())))
              return failure();
          }
          SmallVector<Value> temp(resultBuffer.begin() + size,
                                  resultBuffer.end());
          auto array = builder.createOrFold<hw::ArrayCreateOp>(temp);
          resultBuffer.resize(size);
          resultBuffer.push_back(array);
          return success();
        })
        .Case<BundleType>([&](BundleType srcStructType) {
          auto destStructType = firrtl::type_cast<BundleType>(destType);
          unsigned size = resultBuffer.size();

          // TODO: We don't support partial connects for bundles for now.
          if (destStructType.getNumElements() != srcStructType.getNumElements())
            return failure();

          for (auto elem : llvm::enumerate(destStructType)) {
            auto structExtract =
                hw::StructExtractOp::create(builder, src, elem.value().name);
            if (failed(recurse(structExtract,
                               srcStructType.getElementType(elem.index()),
                               destStructType.getElementType(elem.index()))))
              return failure();
          }
          SmallVector<Value> temp(resultBuffer.begin() + size,
                                  resultBuffer.end());
          auto newStruct = builder.createOrFold<hw::StructCreateOp>(
              lowerType(destStructType), temp);
          resultBuffer.resize(size);
          resultBuffer.push_back(newStruct);
          return success();
        })
        .Case<IntType>([&](auto) {
          if (auto result = cast(src, srcType, destType)) {
            resultBuffer.push_back(result);
            return success();
          }
          return failure();
        })
        .Default([&](auto) { return failure(); });
  };

  if (failed(recurse(array, sourceType, destType)))
    return Value();

  assert(resultBuffer.size() == 1 &&
         "resultBuffer must only contain a result array if `success` is true");
  return resultBuffer[0];
}

/// Return the lowered value corresponding to the specified original value and
/// then extend it to match the width of destType if needed.
///
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredAndExtendedValue(Value src, Type target) {
  auto srcType = cast<FIRRTLBaseType>(src.getType());
  auto dstType = cast<FIRRTLBaseType>(target);
  auto loweredSrc = getLoweredValue(src);

  // We only know how to extend integer types with known width.
  auto dstWidth = dstType.getBitWidthOrSentinel();
  if (dstWidth == -1)
    return {};

  // Handle zero width FIRRTL values which have been removed.
  if (!loweredSrc) {
    // If this was a zero bit operand being extended, then produce a zero of
    // the right result type.  If it is just a failure, fail.
    if (!isZeroBitFIRRTLType(src.getType()))
      return {};
    // Zero bit results have to be returned as null.  The caller can handle
    // this if they want to.
    if (dstWidth == 0)
      return {};
    // Otherwise, FIRRTL semantics is that an extension from a zero bit value
    // always produces a zero value in the destination width.
    return getOrCreateIntConstant(dstWidth, 0);
  }

  auto loweredSrcType = loweredSrc.getType();
  auto loweredDstType = lowerType(dstType);

  // If the two types are the same we do not have to extend.
  if (loweredSrcType == loweredDstType)
    return loweredSrc;

  // Handle type aliases.
  if (dstWidth == srcType.getBitWidthOrSentinel()) {
    // Lookup the lowered type of dest.
    if (loweredSrcType != loweredDstType &&
        (isa<hw::TypeAliasType>(loweredSrcType) ||
         isa<hw::TypeAliasType>(loweredDstType))) {
      return builder.createOrFold<hw::BitcastOp>(loweredDstType, loweredSrc);
    }
  }

  // Aggregates values.
  if (isa<hw::ArrayType, hw::StructType>(loweredSrcType))
    return getExtOrTruncAggregateValue(loweredSrc, srcType, dstType,
                                       /* allowTruncate */ false);

  if (isa<seq::ClockType>(loweredSrcType)) {
    builder.emitError("cannot use clock type as an integer");
    return {};
  }

  auto intSourceType = dyn_cast<IntegerType>(loweredSrcType);
  if (!intSourceType) {
    builder.emitError("operand of type ")
        << loweredSrcType << " cannot be used as an integer";
    return {};
  }

  auto loweredSrcWidth = intSourceType.getWidth();
  if (loweredSrcWidth == unsigned(dstWidth))
    return loweredSrc;

  if (loweredSrcWidth > unsigned(dstWidth)) {
    builder.emitError("operand should not be a truncation");
    return {};
  }

  // Extension follows the sign of the src value, not the destination.
  auto valueFIRType = type_cast<FIRRTLBaseType>(src.getType()).getPassiveType();
  if (type_cast<IntType>(valueFIRType).isSigned())
    return comb::createOrFoldSExt(loweredSrc, loweredDstType, builder);

  auto zero = getOrCreateIntConstant(dstWidth - loweredSrcWidth, 0);
  return builder.createOrFold<comb::ConcatOp>(zero, loweredSrc);
}

/// Return the lowered value corresponding to the specified original value and
/// then extended or truncated to match the width of destType if needed.
///
/// This returns a null value for FIRRTL values that cannot be lowered, e.g.
/// unknown width integers.
Value FIRRTLLowering::getLoweredAndExtOrTruncValue(Value value, Type destType) {
  assert(type_isa<FIRRTLBaseType>(value.getType()) &&
         type_isa<FIRRTLBaseType>(destType) &&
         "input/output value should be FIRRTL");

  // We only know how to adjust integer types with known width.
  auto destWidth = type_cast<FIRRTLBaseType>(destType).getBitWidthOrSentinel();
  if (destWidth == -1)
    return {};

  auto result = getLoweredValue(value);
  if (!result) {
    // If this was a zero bit operand being extended, then produce a zero of
    // the right result type.  If it is just a failure, fail.
    if (!isZeroBitFIRRTLType(value.getType()))
      return {};
    // Zero bit results have to be returned as null.  The caller can handle
    // this if they want to.
    if (destWidth == 0)
      return {};
    // Otherwise, FIRRTL semantics is that an extension from a zero bit value
    // always produces a zero value in the destination width.
    return getOrCreateIntConstant(destWidth, 0);
  }

  // Aggregates values
  if (isa<hw::ArrayType, hw::StructType>(result.getType())) {
    // Types already match.
    if (destType == value.getType())
      return result;

    return getExtOrTruncAggregateValue(
        result, type_cast<FIRRTLBaseType>(value.getType()),
        type_cast<FIRRTLBaseType>(destType),
        /* allowTruncate */ true);
  }

  auto srcWidth = type_cast<IntegerType>(result.getType()).getWidth();
  if (srcWidth == unsigned(destWidth))
    return result;

  if (destWidth == 0)
    return {};

  if (srcWidth > unsigned(destWidth)) {
    auto resultType = builder.getIntegerType(destWidth);
    return builder.createOrFold<comb::ExtractOp>(resultType, result, 0);
  }

  auto resultType = builder.getIntegerType(destWidth);

  // Extension follows the sign of the source value, not the destination.
  auto valueFIRType =
      type_cast<FIRRTLBaseType>(value.getType()).getPassiveType();
  if (type_cast<IntType>(valueFIRType).isSigned())
    return comb::createOrFoldSExt(result, resultType, builder);

  auto zero = getOrCreateIntConstant(destWidth - srcWidth, 0);
  return builder.createOrFold<comb::ConcatOp>(zero, result);
}

/// Return a lowered version of 'operand' suitable for use with substitution /
/// format strings. There are three possible results:
///
///   1. Does not contain a value if no lowering is set.  This is an error.
///   2. The lowering contains an empty value.  This means that the operand
///      should be dropped.
///   3. The lowering contains a value.  This means the operand should be used.
///
/// Zero bit operands are rewritten as one bit zeros and signed integers are
/// wrapped in $signed().
std::optional<Value> FIRRTLLowering::getLoweredFmtOperand(Value operand) {
  // Handle special substitutions.
  if (type_isa<FStringType>(operand.getType())) {
    if (isa<TimeOp>(operand.getDefiningOp()))
      return sv::TimeOp::create(builder);
    if (isa<HierarchicalModuleNameOp>(operand.getDefiningOp()))
      return {nullptr};
  }

  auto loweredValue = getLoweredValue(operand);
  if (!loweredValue) {
    // If this is a zero bit operand, just pass a one bit zero.
    if (!isZeroBitFIRRTLType(operand.getType()))
      return {};
    loweredValue = getOrCreateIntConstant(1, 0);
  }

  // If the operand was an SInt, we want to give the user the option to print
  // it as signed decimal and have to wrap it in $signed().
  if (auto intTy = firrtl::type_cast<IntType>(operand.getType()))
    if (intTy.isSigned())
      loweredValue = sv::SystemFunctionOp::create(
          builder, loweredValue.getType(), "signed", loweredValue);

  return loweredValue;
}

LogicalResult
FIRRTLLowering::loweredFmtOperands(mlir::ValueRange operands,
                                   SmallVectorImpl<Value> &loweredOperands) {
  for (auto operand : operands) {
    std::optional<Value> loweredValue = getLoweredFmtOperand(operand);
    if (!loweredValue)
      return failure();
    // Skip if the lowered value is null.
    if (*loweredValue)
      loweredOperands.push_back(*loweredValue);
  }
  return success();
}

LogicalResult FIRRTLLowering::lowerStatementWithFd(
    const FileDescriptorInfo &fileDescriptor, Value clock, Value cond,
    const std::function<LogicalResult(Value)> &fn, bool usePrintfCond) {
  // Emit an "#ifndef SYNTHESIS" guard into the always block.
  bool failed = false;
  circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
  addToIfDefBlock("SYNTHESIS", std::function<void()>(), [&]() {
    addToAlwaysBlock(clock, [&]() {
      // TODO: This is not printf specific anymore. Replace "Printf" with "FD"
      // or similar but be aware that changing macro name breaks existing uses.
      circuitState.usedPrintf = true;
      if (usePrintfCond)
        circuitState.addFragment(theModule, "PRINTF_COND_FRAGMENT");

      // Emit an "sv.if '`PRINTF_COND_ & cond' into the #ifndef.
      Value ifCond = cond;
      if (usePrintfCond) {
        ifCond =
            sv::MacroRefExprOp::create(builder, cond.getType(), "PRINTF_COND_");
        ifCond = builder.createOrFold<comb::AndOp>(ifCond, cond, true);
      }

      addIfProceduralBlock(ifCond, [&]() {
        // `fd`represents a file decriptor. Use the stdout or the one opened
        // using $fopen.
        Value fd;
        if (fileDescriptor.isDefaultFd()) {
          // Emit the sv.fwrite, writing to stderr by default.
          fd = hw::ConstantOp::create(builder, APInt(32, 0x80000002));
        } else {
          // Call the library function to get the FD.
          auto fdOrError = callFileDescriptorLib(fileDescriptor);
          if (llvm::failed(fdOrError)) {
            failed = true;
            return;
          }
          fd = *fdOrError;
        }
        failed = llvm::failed(fn(fd));
      });
    });
  });
  return failure(failed);
}

FailureOr<Value>
FIRRTLLowering::callFileDescriptorLib(const FileDescriptorInfo &info) {
  circuitState.usedFileDescriptorLib = true;
  circuitState.addFragment(theModule, "CIRCT_LIB_LOGGING_FRAGMENT");

  Value fileName;
  if (info.isSubstitutionRequired()) {
    SmallVector<Value> fileNameOperands;
    if (failed(loweredFmtOperands(info.getSubstitutions(), fileNameOperands)))
      return failure();

    fileName = sv::SFormatFOp::create(builder, info.getOutputFileFormat(),
                                      fileNameOperands)
                   .getResult();
  } else {
    // If substitution is not required, just use the output file name.
    fileName = sv::ConstantStrOp::create(builder, info.getOutputFileFormat())
                   .getResult();
  }

  return sv::FuncCallProceduralOp::create(
             builder, mlir::TypeRange{builder.getIntegerType(32)},
             builder.getStringAttr("__circt_lib_logging::FileDescriptor::get"),
             ValueRange{fileName})
      ->getResult(0);
}

/// Set the lowered value of 'orig' to 'result', remembering this in a map.
/// This always returns success() to make it more convenient in lowering code.
///
/// Note that result may be null here if we're lowering orig to a zero-bit
/// value.
///
LogicalResult FIRRTLLowering::setLowering(Value orig, Value result) {
  if (auto origType = dyn_cast<FIRRTLType>(orig.getType())) {
    assert((!result || !type_isa<FIRRTLType>(result.getType())) &&
           "Lowering didn't turn a FIRRTL value into a non-FIRRTL value");

#ifndef NDEBUG
    auto baseType = getBaseType(origType);
    auto srcWidth = baseType.getPassiveType().getBitWidthOrSentinel();

    // Caller should pass null value iff this was a zero bit value.
    if (srcWidth != -1) {
      if (result)
        assert((srcWidth != 0) &&
               "Lowering produced value for zero width source");
      else
        assert((srcWidth == 0) &&
               "Lowering produced null value but source wasn't zero width");
    }
#endif
  } else {
    assert(result && "Lowering of foreign type produced null value");
  }

  auto &slot = valueMapping[orig];
  assert(!slot && "value lowered multiple times");
  slot = result;
  return success();
}

/// Set the lowering for a value to the specified result.  This came from a
/// possible folding, so check to see if we need to handle a constant.
LogicalResult FIRRTLLowering::setPossiblyFoldedLowering(Value orig,
                                                        Value result) {
  // If this is a constant, check to see if we have it in our unique mapping:
  // it could have come from folding an operation.
  if (auto cst = dyn_cast_or_null<hw::ConstantOp>(result.getDefiningOp())) {
    auto &entry = hwConstantMap[cst.getValueAttr()];
    if (entry == cst) {
      // We're already using an entry in the constant map, nothing to do.
    } else if (entry) {
      // We already had this constant, reuse the one we have instead of the
      // one we just folded.
      result = entry;
      cst->erase();
    } else {
      // This is a new constant.  Remember it!
      entry = cst;
      cst->moveBefore(&theModule.getBodyBlock()->front());
    }
  }

  return setLowering(orig, result);
}

/// Create a new operation with type ResultOpType and arguments CtorArgTypes,
/// then call setLowering with its result.
template <typename ResultOpType, typename... CtorArgTypes>
LogicalResult FIRRTLLowering::setLoweringTo(Operation *orig,
                                            CtorArgTypes... args) {
  auto result = builder.createOrFold<ResultOpType>(args...);
  if (auto *op = result.getDefiningOp())
    tryCopyName(op, orig);
  return setPossiblyFoldedLowering(orig->getResult(0), result);
}

/// Create a new LTL operation with type ResultOpType and arguments
/// CtorArgTypes, then call setLowering with its result. Also add the operation
/// to the worklist of LTL ops that need to have their types fixed-up after the
/// lowering.
template <typename ResultOpType, typename... CtorArgTypes>
LogicalResult FIRRTLLowering::setLoweringToLTL(Operation *orig,
                                               CtorArgTypes... args) {
  auto result = builder.createOrFold<ResultOpType>(args...);
  if (auto *op = result.getDefiningOp())
    ltlOpFixupWorklist.insert(op);
  return setPossiblyFoldedLowering(orig->getResult(0), result);
}

/// Creates a backedge of the specified result type. A backedge represents a
/// placeholder to be filled in later by a lowered value. If the backedge is not
/// updated with a real value by the end of the pass, it will be replaced with
/// an undriven wire.  Backedges are allowed to be updated to other backedges.
/// If a chain of backedges forms a combinational loop, they will be replaced
/// with an undriven wire.
Backedge FIRRTLLowering::createBackedge(Location loc, Type type) {
  auto backedge = backedgeBuilder.get(type, loc);
  backedges.insert({backedge, backedge});
  return backedge;
}

/// Sets the lowering for a value to a backedge of the specified result type.
/// This is useful for lowering types which cannot pass through a wire, or to
/// directly materialize values in operations that violate the SSA dominance
/// constraint.
Backedge FIRRTLLowering::createBackedge(Value orig, Type type) {
  auto backedge = createBackedge(orig.getLoc(), type);
  (void)setLowering(orig, backedge);
  return backedge;
}

/// If the `from` value is in fact a backedge, record that the backedge will
/// be replaced by the value.  Return true if the destination is a backedge.
bool FIRRTLLowering::updateIfBackedge(Value dest, Value src) {
  auto backedgeIt = backedges.find(dest);
  if (backedgeIt == backedges.end())
    return false;
  backedgeIt->second = src;
  return true;
}

/// Switch the insertion point of the current builder to the end of the
/// specified block and run the closure.  This correctly handles the case
/// where the closure is null, but the caller needs to make sure the block
/// exists.
void FIRRTLLowering::runWithInsertionPointAtEndOfBlock(
    const std::function<void(void)> &fn, Region &region) {
  if (!fn)
    return;

  auto oldIP = builder.saveInsertionPoint();

  builder.setInsertionPointToEnd(&region.front());
  fn();
  builder.restoreInsertionPoint(oldIP);
}

/// Return a read value for the specified inout operation, auto-uniquing them.
Value FIRRTLLowering::getReadValue(Value v) {
  Value result = readInOutCreated.lookup(v);
  if (result)
    return result;

  // Make sure to put the read value at the correct scope so it dominates all
  // future uses.
  auto oldIP = builder.saveInsertionPoint();
  if (auto *vOp = v.getDefiningOp()) {
    builder.setInsertionPointAfter(vOp);
  } else {
    // For reads of ports, just set the insertion point at the top of the
    // module.
    builder.setInsertionPoint(&theModule.getBodyBlock()->front());
  }

  // Instead of creating `ReadInOutOp` for `ArrayIndexInOutOp`, create
  // `ArrayGetOp` for root arrays.
  if (auto arrayIndexInout = v.getDefiningOp<sv::ArrayIndexInOutOp>()) {
    result = getReadValue(arrayIndexInout.getInput());
    result = builder.createOrFold<hw::ArrayGetOp>(result,
                                                  arrayIndexInout.getIndex());
  } else {
    // Otherwise, create a read inout operation.
    result = builder.createOrFold<sv::ReadInOutOp>(v);
  }
  builder.restoreInsertionPoint(oldIP);
  readInOutCreated.insert({v, result});
  return result;
}

Value FIRRTLLowering::getNonClockValue(Value v) {
  auto it = fromClockMapping.try_emplace(v, Value{});
  if (it.second) {
    ImplicitLocOpBuilder builder(v.getLoc(), v.getContext());
    builder.setInsertionPointAfterValue(v);
    it.first->second = seq::FromClockOp::create(builder, v);
  }
  return it.first->second;
}

void FIRRTLLowering::addToAlwaysBlock(
    sv::EventControl clockEdge, Value clock, sv::ResetType resetStyle,
    sv::EventControl resetEdge, Value reset,
    const std::function<void(void)> &body,
    const std::function<void(void)> &resetBody) {
  AlwaysKeyType key{builder.getBlock(), clockEdge, clock,
                    resetStyle,         resetEdge, reset};
  sv::AlwaysOp alwaysOp;
  sv::IfOp insideIfOp;
  std::tie(alwaysOp, insideIfOp) = alwaysBlocks.lookup(key);

  if (!alwaysOp) {
    if (reset) {
      assert(resetStyle != sv::ResetType::NoReset);
      // Here, we want to create the folloing structure with sv.always and
      // sv.if. If `reset` is async, we need to add `reset` to a sensitivity
      // list.
      //
      // sv.always @(clockEdge or reset) {
      //   sv.if (reset) {
      //     resetBody
      //   } else {
      //     body
      //   }
      // }

      auto createIfOp = [&]() {
        // It is weird but intended. Here we want to create an empty sv.if
        // with an else block.
        insideIfOp = sv::IfOp::create(
            builder, reset, []() {}, []() {});
      };
      if (resetStyle == sv::ResetType::AsyncReset) {
        sv::EventControl events[] = {clockEdge, resetEdge};
        Value clocks[] = {clock, reset};

        alwaysOp = sv::AlwaysOp::create(builder, events, clocks, [&]() {
          if (resetEdge == sv::EventControl::AtNegEdge)
            llvm_unreachable("negative edge for reset is not expected");
          createIfOp();
        });
      } else {
        alwaysOp = sv::AlwaysOp::create(builder, clockEdge, clock, createIfOp);
      }
    } else {
      assert(!resetBody);
      alwaysOp = sv::AlwaysOp::create(builder, clockEdge, clock);
      insideIfOp = nullptr;
    }
    alwaysBlocks[key] = {alwaysOp, insideIfOp};
  }

  if (reset) {
    assert(insideIfOp && "reset body must be initialized before");
    runWithInsertionPointAtEndOfBlock(resetBody, insideIfOp.getThenRegion());
    runWithInsertionPointAtEndOfBlock(body, insideIfOp.getElseRegion());
  } else {
    runWithInsertionPointAtEndOfBlock(body, alwaysOp.getBody());
  }

  // Move the earlier always block(s) down to where the last would have been
  // inserted.  This ensures that any values used by the always blocks are
  // defined ahead of the uses, which leads to better generated Verilog.
  alwaysOp->moveBefore(builder.getInsertionBlock(),
                       builder.getInsertionPoint());
}

LogicalResult FIRRTLLowering::emitGuards(Location loc,
                                         ArrayRef<Attribute> guards,
                                         std::function<void(void)> emit) {
  if (guards.empty()) {
    emit();
    return success();
  }
  auto guard = dyn_cast<StringAttr>(guards[0]);
  if (!guard)
    return mlir::emitError(loc,
                           "elements in `guards` array must be `StringAttr`");

  // Record the guard macro to emit a declaration for it.
  circuitState.addMacroDecl(builder.getStringAttr(guard.getValue()));
  LogicalResult result = LogicalResult::failure();
  addToIfDefBlock(guard.getValue(), [&]() {
    result = emitGuards(loc, guards.drop_front(), emit);
  });
  return result;
}

void FIRRTLLowering::addToIfDefBlock(StringRef cond,
                                     std::function<void(void)> thenCtor,
                                     std::function<void(void)> elseCtor) {
  auto condAttr = builder.getStringAttr(cond);
  auto op = ifdefBlocks.lookup({builder.getBlock(), condAttr});
  if (op) {
    runWithInsertionPointAtEndOfBlock(thenCtor, op.getThenRegion());
    runWithInsertionPointAtEndOfBlock(elseCtor, op.getElseRegion());

    // Move the earlier #ifdef block(s) down to where the last would have been
    // inserted.  This ensures that any values used by the #ifdef blocks are
    // defined ahead of the uses, which leads to better generated Verilog.
    op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
  } else {
    ifdefBlocks[{builder.getBlock(), condAttr}] =
        sv::IfDefOp::create(builder, condAttr, thenCtor, elseCtor);
  }
}

void FIRRTLLowering::addToInitialBlock(std::function<void(void)> body) {
  auto op = initialBlocks.lookup(builder.getBlock());
  if (op) {
    runWithInsertionPointAtEndOfBlock(body, op.getBody());

    // Move the earlier initial block(s) down to where the last would have
    // been inserted.  This ensures that any values used by the initial blocks
    // are defined ahead of the uses, which leads to better generated Verilog.
    op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
  } else {
    initialBlocks[builder.getBlock()] = sv::InitialOp::create(builder, body);
  }
}

void FIRRTLLowering::addIfProceduralBlock(Value cond,
                                          std::function<void(void)> thenCtor,
                                          std::function<void(void)> elseCtor) {
  // Check to see if we already have an if on this condition immediately
  // before the insertion point.  If so, extend it.
  auto insertIt = builder.getInsertionPoint();
  if (insertIt != builder.getBlock()->begin())
    if (auto ifOp = dyn_cast<sv::IfOp>(*--insertIt)) {
      if (ifOp.getCond() == cond) {
        runWithInsertionPointAtEndOfBlock(thenCtor, ifOp.getThenRegion());
        runWithInsertionPointAtEndOfBlock(elseCtor, ifOp.getElseRegion());
        return;
      }
    }

  sv::IfOp::create(builder, cond, thenCtor, elseCtor);
}

//===----------------------------------------------------------------------===//
// Special Operations
//===----------------------------------------------------------------------===//

/// Handle the case where an operation wasn't lowered.  When this happens, the
/// operands should just be unlowered non-FIRRTL values.  If the operand was
/// not lowered then leave it alone, otherwise we have a problem with
/// lowering.
///
FIRRTLLowering::UnloweredOpResult
FIRRTLLowering::handleUnloweredOp(Operation *op) {
  // FIRRTL operations must explicitly handle their regions.
  if (!op->getRegions().empty() && isa<FIRRTLDialect>(op->getDialect())) {
    op->emitOpError("must explicitly handle its regions");
    return LoweringFailure;
  }

  // Simply pass through non-FIRRTL operations and consider them already
  // lowered. This allows us to handled partially lowered inputs, and also allow
  // other FIRRTL operations to spawn additional already-lowered operations,
  // like `hw.output`.
  if (!isa<FIRRTLDialect>(op->getDialect())) {
    // Push nested operations onto the worklist such that they are lowered.
    for (auto &region : op->getRegions())
      addToWorklist(region);
    for (auto &operand : op->getOpOperands())
      if (auto lowered = getPossiblyInoutLoweredValue(operand.get()))
        operand.set(lowered);
    for (auto result : op->getResults())
      (void)setLowering(result, result);
    return AlreadyLowered;
  }

  // Ok, at least one operand got lowered, so this operation is using a FIRRTL
  // value, but wasn't itself lowered.  This is because the lowering is
  // incomplete. This is either a bug or incomplete implementation.
  //
  // There is one aspect of incompleteness we intentionally expect: we allow
  // primitive operations that produce a zero bit result to be ignored by the
  // lowering logic.  They don't have side effects, and handling this corner
  // case just complicates each of the lowering hooks. Instead, we just handle
  // them all right here.
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    if (type_isa<FIRRTLBaseType>(resultType) &&
        isZeroBitFIRRTLType(resultType) &&
        (isExpression(op) || isa<mlir::UnrealizedConversionCastOp>(op))) {
      // Zero bit values lower to the null Value.
      (void)setLowering(op->getResult(0), Value());
      return NowLowered;
    }
  }
  op->emitOpError("LowerToHW couldn't handle this operation");
  return LoweringFailure;
}

LogicalResult FIRRTLLowering::visitExpr(ConstantOp op) {
  // Zero width values must be lowered to nothing.
  if (isZeroBitFIRRTLType(op.getType()))
    return setLowering(op, Value());

  return setLowering(op, getOrCreateIntConstant(op.getValue()));
}

LogicalResult FIRRTLLowering::visitExpr(SpecialConstantOp op) {
  Value cst;
  if (isa<ClockType>(op.getType())) {
    cst = getOrCreateClockConstant(op.getValue() ? seq::ClockConst::High
                                                 : seq::ClockConst::Low);
  } else {
    cst = getOrCreateIntConstant(APInt(/*bitWidth*/ 1, op.getValue()));
  }
  return setLowering(op, cst);
}

FailureOr<Value> FIRRTLLowering::lowerSubindex(SubindexOp op, Value input) {
  auto iIdx = getOrCreateIntConstant(
      getBitWidthFromVectorSize(
          firrtl::type_cast<FVectorType>(op.getInput().getType())
              .getNumElements()),
      op.getIndex());

  // If the input has an inout type, we need to lower to ArrayIndexInOutOp;
  // otherwise hw::ArrayGetOp.
  Value result;
  if (isa<sv::InOutType>(input.getType()))
    result = builder.createOrFold<sv::ArrayIndexInOutOp>(input, iIdx);
  else
    result = builder.createOrFold<hw::ArrayGetOp>(input, iIdx);
  if (auto *definingOp = result.getDefiningOp())
    tryCopyName(definingOp, op);
  return result;
}

FailureOr<Value> FIRRTLLowering::lowerSubaccess(SubaccessOp op, Value input) {
  Value valueIdx = getLoweredAndExtOrTruncValue(
      op.getIndex(),
      UIntType::get(op->getContext(),
                    getBitWidthFromVectorSize(
                        firrtl::type_cast<FVectorType>(op.getInput().getType())
                            .getNumElements())));
  if (!valueIdx) {
    op->emitError() << "input lowering failed";
    return failure();
  }

  // If the input has an inout type, we need to lower to ArrayIndexInOutOp;
  // otherwise, lower the op to array indexing.
  Value result;
  if (isa<sv::InOutType>(input.getType()))
    result = builder.createOrFold<sv::ArrayIndexInOutOp>(input, valueIdx);
  else
    result = createArrayIndexing(input, valueIdx);
  if (auto *definingOp = result.getDefiningOp())
    tryCopyName(definingOp, op);
  return result;
}

FailureOr<Value> FIRRTLLowering::lowerSubfield(SubfieldOp op, Value input) {
  auto resultType = lowerType(op->getResult(0).getType());
  if (!resultType || !input) {
    op->emitError() << "subfield type lowering failed";
    return failure();
  }

  // If the input has an inout type, we need to lower to StructFieldInOutOp;
  // otherwise, StructExtractOp.
  auto field = firrtl::type_cast<BundleType>(op.getInput().getType())
                   .getElementName(op.getFieldIndex());
  Value result;
  if (isa<sv::InOutType>(input.getType()))
    result = builder.createOrFold<sv::StructFieldInOutOp>(input, field);
  else
    result = builder.createOrFold<hw::StructExtractOp>(input, field);
  if (auto *definingOp = result.getDefiningOp())
    tryCopyName(definingOp, op);
  return result;
}

LogicalResult FIRRTLLowering::visitExpr(SubindexOp op) {
  if (isZeroBitFIRRTLType(op.getType()))
    return setLowering(op, Value());

  auto input = getPossiblyInoutLoweredValue(op.getInput());
  if (!input)
    return op.emitError() << "input lowering failed";

  auto result = lowerSubindex(op, input);
  if (failed(result))
    return failure();
  return setLowering(op, *result);
}

LogicalResult FIRRTLLowering::visitExpr(SubaccessOp op) {
  if (isZeroBitFIRRTLType(op.getType()))
    return setLowering(op, Value());

  auto input = getPossiblyInoutLoweredValue(op.getInput());
  if (!input)
    return op.emitError() << "input lowering failed";

  auto result = lowerSubaccess(op, input);
  if (failed(result))
    return failure();
  return setLowering(op, *result);
}

LogicalResult FIRRTLLowering::visitExpr(SubfieldOp op) {
  // firrtl.mem lowering lowers some SubfieldOps.  Zero-width can leave
  // invalid subfield accesses
  if (getLoweredValue(op) || !op.getInput())
    return success();

  if (isZeroBitFIRRTLType(op.getType()))
    return setLowering(op, Value());

  auto input = getPossiblyInoutLoweredValue(op.getInput());
  if (!input)
    return op.emitError() << "input lowering failed";

  auto result = lowerSubfield(op, input);
  if (failed(result))
    return failure();
  return setLowering(op, *result);
}

LogicalResult FIRRTLLowering::visitExpr(VectorCreateOp op) {
  auto resultType = lowerType(op.getResult().getType());
  SmallVector<Value> operands;
  // NOTE: The operand order must be inverted.
  for (auto oper : llvm::reverse(op.getOperands())) {
    auto val = getLoweredValue(oper);
    if (!val)
      return failure();
    operands.push_back(val);
  }
  return setLoweringTo<hw::ArrayCreateOp>(op, resultType, operands);
}

LogicalResult FIRRTLLowering::visitExpr(BundleCreateOp op) {
  auto resultType = lowerType(op.getResult().getType());
  SmallVector<Value> operands;
  for (auto oper : op.getOperands()) {
    auto val = getLoweredValue(oper);
    if (!val)
      return failure();
    operands.push_back(val);
  }
  return setLoweringTo<hw::StructCreateOp>(op, resultType, operands);
}

LogicalResult FIRRTLLowering::visitExpr(FEnumCreateOp op) {
  // Zero width values must be lowered to nothing.
  if (isZeroBitFIRRTLType(op.getType()))
    return setLowering(op, Value());

  auto input = getLoweredValue(op.getInput());
  auto tagName = op.getFieldNameAttr();
  auto oldType = op.getType().base();
  auto newType = lowerType(oldType);
  auto element = *oldType.getElement(op.getFieldNameAttr());

  if (auto structType = dyn_cast<hw::StructType>(newType)) {
    auto tagType = structType.getFieldType("tag");
    auto tagValue = IntegerAttr::get(tagType, element.value.getValue());
    auto tag = sv::LocalParamOp::create(builder, op.getLoc(), tagType, tagValue,
                                        tagName);
    auto bodyType = structType.getFieldType("body");
    auto body = hw::UnionCreateOp::create(builder, bodyType, tagName, input);
    SmallVector<Value> operands = {tag.getResult(), body.getResult()};
    return setLoweringTo<hw::StructCreateOp>(op, structType, operands);
  }
  auto tagValue = IntegerAttr::get(newType, element.value.getValue());
  return setLoweringTo<sv::LocalParamOp>(op, newType, tagValue, tagName);
}

LogicalResult FIRRTLLowering::visitExpr(AggregateConstantOp op) {
  auto resultType = lowerType(op.getResult().getType());
  auto attr =
      getOrCreateAggregateConstantAttribute(op.getFieldsAttr(), resultType);

  return setLoweringTo<hw::AggregateConstantOp>(op, resultType,
                                                cast<ArrayAttr>(attr));
}

LogicalResult FIRRTLLowering::visitExpr(IsTagOp op) {

  auto tagName = op.getFieldNameAttr();
  auto lhs = getLoweredValue(op.getInput());
  if (isa<hw::StructType>(lhs.getType()))
    lhs = hw::StructExtractOp::create(builder, lhs, "tag");

  auto index = op.getFieldIndex();
  auto enumType = op.getInput().getType().base();
  auto tagValue = enumType.getElementValueAttr(index);
  auto tagValueType = IntegerType::get(op.getContext(), enumType.getTagWidth());
  auto loweredTagValue = IntegerAttr::get(tagValueType, tagValue.getValue());
  auto rhs = sv::LocalParamOp::create(builder, op.getLoc(), tagValueType,
                                      loweredTagValue, tagName);

  Type resultType = builder.getIntegerType(1);
  return setLoweringTo<comb::ICmpOp>(op, resultType, ICmpPredicate::eq, lhs,
                                     rhs, true);
}

LogicalResult FIRRTLLowering::visitExpr(SubtagOp op) {
  // Zero width values must be lowered to nothing.
  if (isZeroBitFIRRTLType(op.getType()))
    return setLowering(op, Value());

  auto tagName = op.getFieldNameAttr();
  auto input = getLoweredValue(op.getInput());
  auto field = hw::StructExtractOp::create(builder, input, "body");
  return setLoweringTo<hw::UnionExtractOp>(op, field, tagName);
}

LogicalResult FIRRTLLowering::visitExpr(TagExtractOp op) {
  // Zero width values must be lowered to nothing.
  if (isZeroBitFIRRTLType(op.getType()))
    return setLowering(op, Value());

  auto input = getLoweredValue(op.getInput());
  if (!input)
    return failure();

  // If the lowered enum is a struct (has both tag and body), extract the tag
  // field.
  if (isa<hw::StructType>(input.getType())) {
    return setLoweringTo<hw::StructExtractOp>(op, input, "tag");
  }

  // If the lowered enum is just the tag (simple enum with no data), return it
  // directly.
  return setLowering(op, input);
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitDecl(WireOp op) {
  auto origResultType = op.getResult().getType();

  // Foreign types lower to a backedge that needs to be resolved by a later
  // connect op.
  if (!type_isa<FIRRTLType>(origResultType)) {
    createBackedge(op.getResult(), origResultType);
    return success();
  }

  auto resultType = lowerType(origResultType);
  if (!resultType)
    return failure();

  if (resultType.isInteger(0)) {
    if (op.getInnerSym())
      return op.emitError("zero width wire is referenced by name [")
             << *op.getInnerSym() << "] (e.g. in an XMR) but must be removed";
    return setLowering(op.getResult(), Value());
  }

  // Name attr is required on sv.wire but optional on firrtl.wire.
  auto innerSym = lowerInnerSymbol(op);
  auto name = op.getNameAttr();
  // This is not a temporary wire created by the compiler, so attach a symbol
  // name.
  auto wire = hw::WireOp::create(
      builder, op.getLoc(), getOrCreateZConstant(resultType), name, innerSym);

  if (auto svAttrs = sv::getSVAttributes(op))
    sv::setSVAttributes(wire, svAttrs);

  return setLowering(op.getResult(), wire);
}

LogicalResult FIRRTLLowering::visitDecl(VerbatimWireOp op) {
  auto resultTy = lowerType(op.getType());
  if (!resultTy)
    return failure();
  resultTy = sv::InOutType::get(op.getContext(), resultTy);

  SmallVector<Value, 4> operands;
  operands.reserve(op.getSubstitutions().size());
  for (auto operand : op.getSubstitutions()) {
    auto lowered = getLoweredValue(operand);
    if (!lowered)
      return failure();
    operands.push_back(lowered);
  }

  ArrayAttr symbols = op.getSymbolsAttr();
  if (!symbols)
    symbols = ArrayAttr::get(op.getContext(), {});

  return setLoweringTo<sv::VerbatimExprSEOp>(op, resultTy, op.getTextAttr(),
                                             operands, symbols);
}

LogicalResult FIRRTLLowering::visitDecl(NodeOp op) {
  auto operand = getLoweredValue(op.getInput());
  if (!operand)
    return handleZeroBit(op.getInput(), [&]() -> LogicalResult {
      if (op.getInnerSym())
        return op.emitError("zero width node is referenced by name [")
               << *op.getInnerSym()
               << "] (e.g. in an XMR) but must be "
                  "removed";
      return setLowering(op.getResult(), Value());
    });

  // Node operations are logical noops, but may carry annotations or be
  // referred to through an inner name. If a don't touch is present, ensure
  // that we have a symbol name so we can keep the node as a wire.
  auto name = op.getNameAttr();
  auto innerSym = lowerInnerSymbol(op);

  if (innerSym)
    operand = hw::WireOp::create(builder, operand, name, innerSym);

  // Move SV attributes.
  if (auto svAttrs = sv::getSVAttributes(op)) {
    if (!innerSym)
      operand = hw::WireOp::create(builder, operand, name);
    sv::setSVAttributes(operand.getDefiningOp(), svAttrs);
  }

  return setLowering(op.getResult(), operand);
}

LogicalResult FIRRTLLowering::visitDecl(RegOp op) {
  auto resultType = lowerType(op.getResult().getType());
  if (!resultType)
    return failure();
  if (resultType.isInteger(0))
    return setLowering(op.getResult(), Value());

  Value clockVal = getLoweredValue(op.getClockVal());
  if (!clockVal)
    return failure();

  // Create a reg op, wiring itself to its input.
  auto innerSym = lowerInnerSymbol(op);
  Backedge inputEdge = backedgeBuilder.get(resultType);
  auto reg = seq::FirRegOp::create(builder, inputEdge, clockVal,
                                   op.getNameAttr(), innerSym);

  // Pass along the start and end random initialization bits for this register.
  if (auto randomRegister = op->getAttr("firrtl.random_init_register"))
    reg->setAttr("firrtl.random_init_register", randomRegister);
  if (auto randomStart = op->getAttr("firrtl.random_init_start"))
    reg->setAttr("firrtl.random_init_start", randomStart);
  if (auto randomEnd = op->getAttr("firrtl.random_init_end"))
    reg->setAttr("firrtl.random_init_end", randomEnd);

  // Move SV attributes.
  if (auto svAttrs = sv::getSVAttributes(op))
    sv::setSVAttributes(reg, svAttrs);

  inputEdge.setValue(reg);
  (void)setLowering(op.getResult(), reg);
  return success();
}

LogicalResult FIRRTLLowering::visitDecl(RegResetOp op) {
  auto resultType = lowerType(op.getResult().getType());
  if (!resultType)
    return failure();
  if (resultType.isInteger(0))
    return setLowering(op.getResult(), Value());

  Value clockVal = getLoweredValue(op.getClockVal());
  Value resetSignal = getLoweredValue(op.getResetSignal());
  // Reset values may be narrower than the register.  Extend appropriately.
  Value resetValue = getLoweredAndExtOrTruncValue(
      op.getResetValue(), type_cast<FIRRTLBaseType>(op.getResult().getType()));

  if (!clockVal || !resetSignal || !resetValue)
    return failure();

  // Create a reg op, wiring itself to its input.
  auto innerSym = lowerInnerSymbol(op);
  bool isAsync = type_isa<AsyncResetType>(op.getResetSignal().getType());
  Backedge inputEdge = backedgeBuilder.get(resultType);
  auto reg =
      seq::FirRegOp::create(builder, inputEdge, clockVal, op.getNameAttr(),
                            resetSignal, resetValue, innerSym, isAsync);

  // Pass along the start and end random initialization bits for this register.
  if (auto randomRegister = op->getAttr("firrtl.random_init_register"))
    reg->setAttr("firrtl.random_init_register", randomRegister);
  if (auto randomStart = op->getAttr("firrtl.random_init_start"))
    reg->setAttr("firrtl.random_init_start", randomStart);
  if (auto randomEnd = op->getAttr("firrtl.random_init_end"))
    reg->setAttr("firrtl.random_init_end", randomEnd);

  // Move SV attributes.
  if (auto svAttrs = sv::getSVAttributes(op))
    sv::setSVAttributes(reg, svAttrs);

  inputEdge.setValue(reg);
  (void)setLowering(op.getResult(), reg);

  return success();
}

LogicalResult FIRRTLLowering::visitDecl(MemOp op) {
  // TODO: Remove this restriction and preserve aggregates in
  // memories.
  if (type_isa<BundleType>(op.getDataType()))
    return op.emitOpError(
        "should have already been lowered from a ground type to an aggregate "
        "type using the LowerTypes pass. Use "
        "'firtool --lower-types' or 'circt-opt "
        "--pass-pipeline='firrtl.circuit(firrtl-lower-types)' "
        "to run this.");

  FirMemory memSummary = op.getSummary();

  // Create the memory declaration.
  auto memType = seq::FirMemType::get(
      op.getContext(), memSummary.depth, memSummary.dataWidth,
      memSummary.isMasked ? std::optional<uint32_t>(memSummary.maskBits)
                          : std::optional<uint32_t>());

  seq::FirMemInitAttr memInit;
  if (auto init = op.getInitAttr())
    memInit = seq::FirMemInitAttr::get(init.getContext(), init.getFilename(),
                                       init.getIsBinary(), init.getIsInline());

  auto memDecl = seq::FirMemOp::create(
      builder, memType, memSummary.readLatency, memSummary.writeLatency,
      memSummary.readUnderWrite, memSummary.writeUnderWrite, op.getNameAttr(),
      op.getInnerSymAttr(), memInit, op.getPrefixAttr(), Attribute{});

  if (auto parent = op->getParentOfType<hw::HWModuleOp>()) {
    if (auto file = parent->getAttrOfType<hw::OutputFileAttr>("output_file")) {
      auto dir = file;
      if (!file.isDirectory())
        dir = hw::OutputFileAttr::getAsDirectory(builder.getContext(),
                                                 file.getDirectory());
      memDecl.setOutputFileAttr(dir);
    }
  }

  // Memories return multiple structs, one for each port, which means we
  // have two layers of type to split apart.
  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {

    auto addOutput = [&](StringRef field, size_t width, Value value) {
      for (auto &a : getAllFieldAccesses(op.getResult(i), field)) {
        if (width > 0)
          (void)setLowering(a, value);
        else
          a->eraseOperand(0);
      }
    };

    auto addInput = [&](StringRef field, Value backedge) {
      for (auto a : getAllFieldAccesses(op.getResult(i), field)) {
        if (cast<FIRRTLBaseType>(a.getType())
                .getPassiveType()
                .getBitWidthOrSentinel() > 0)
          (void)setLowering(a, backedge);
        else
          a->eraseOperand(0);
      }
    };

    auto addInputPort = [&](StringRef field, size_t width) -> Value {
      // If the memory is 0-width, do not materialize any connections to it.
      // However, `seq.firmem` now requires a 1-bit input, so materialize
      // a dummy x value to provide it with.
      Value backedge, portValue;
      if (width == 0) {
        portValue = getOrCreateXConstant(1);
      } else {
        auto portType = IntegerType::get(op.getContext(), width);
        backedge = portValue = createBackedge(builder.getLoc(), portType);
      }
      addInput(field, backedge);
      return portValue;
    };

    auto addClock = [&](StringRef field) -> Value {
      Type clockTy = seq::ClockType::get(op.getContext());
      Value portValue = createBackedge(builder.getLoc(), clockTy);
      addInput(field, portValue);
      return portValue;
    };

    auto memportKind = op.getPortKind(i);
    if (memportKind == MemOp::PortKind::Read) {
      auto addr = addInputPort("addr", op.getAddrBits());
      auto en = addInputPort("en", 1);
      auto clk = addClock("clk");
      auto data = seq::FirMemReadOp::create(builder, memDecl, addr, clk, en);
      addOutput("data", memSummary.dataWidth, data);
    } else if (memportKind == MemOp::PortKind::ReadWrite) {
      auto addr = addInputPort("addr", op.getAddrBits());
      auto en = addInputPort("en", 1);
      auto clk = addClock("clk");
      // If maskBits =1, then And the mask field with enable, and update the
      // enable. Else keep mask port.
      auto mode = addInputPort("wmode", 1);
      if (!memSummary.isMasked)
        mode = builder.createOrFold<comb::AndOp>(mode, addInputPort("wmask", 1),
                                                 true);
      auto wdata = addInputPort("wdata", memSummary.dataWidth);
      // Ignore mask port, if maskBits =1
      Value mask;
      if (memSummary.isMasked)
        mask = addInputPort("wmask", memSummary.maskBits);
      auto rdata = seq::FirMemReadWriteOp::create(builder, memDecl, addr, clk,
                                                  en, wdata, mode, mask);
      addOutput("rdata", memSummary.dataWidth, rdata);
    } else {
      auto addr = addInputPort("addr", op.getAddrBits());
      // If maskBits =1, then And the mask field with enable, and update the
      // enable. Else keep mask port.
      auto en = addInputPort("en", 1);
      if (!memSummary.isMasked)
        en = builder.createOrFold<comb::AndOp>(en, addInputPort("mask", 1),
                                               true);
      auto clk = addClock("clk");
      auto data = addInputPort("data", memSummary.dataWidth);
      // Ignore mask port, if maskBits =1
      Value mask;
      if (memSummary.isMasked)
        mask = addInputPort("mask", memSummary.maskBits);
      seq::FirMemWriteOp::create(builder, memDecl, addr, clk, en, data, mask);
    }
  }

  return success();
}

LogicalResult FIRRTLLowering::visitDecl(InstanceOp oldInstance) {
  Operation *oldModule =
      oldInstance.getReferencedModule(circuitState.getInstanceGraph());

  auto *newModule = circuitState.getNewModule(oldModule);
  if (!newModule) {
    oldInstance->emitOpError("could not find module [")
        << oldInstance.getModuleName() << "] referenced by instance";
    return failure();
  }

  // If this is a referenced to a parameterized extmodule, then bring the
  // parameters over to this instance.
  ArrayAttr parameters;
  if (auto oldExtModule = dyn_cast<FExtModuleOp>(oldModule))
    parameters = getHWParameters(oldExtModule, /*ignoreValues=*/false);

  // Decode information about the input and output ports on the referenced
  // module.
  SmallVector<PortInfo, 8> portInfo = cast<FModuleLike>(oldModule).getPorts();

  // Build an index from the name attribute to an index into portInfo, so we
  // can do efficient lookups.
  llvm::SmallDenseMap<Attribute, unsigned> portIndicesByName;
  for (unsigned portIdx = 0, e = portInfo.size(); portIdx != e; ++portIdx)
    portIndicesByName[portInfo[portIdx].name] = portIdx;

  // Ok, get ready to create the new instance operation.  We need to prepare
  // input operands.
  SmallVector<Value, 8> operands;
  for (size_t portIndex = 0, e = portInfo.size(); portIndex != e; ++portIndex) {
    auto &port = portInfo[portIndex];
    auto portType = lowerType(port.type);
    if (!portType) {
      oldInstance->emitOpError("could not lower type of port ") << port.name;
      return failure();
    }

    // Drop zero bit input/inout ports.
    if (portType.isInteger(0))
      continue;

    // We wire outputs up after creating the instance.
    if (port.isOutput())
      continue;

    auto portResult = oldInstance.getResult(portIndex);
    assert(portResult && "invalid IR, couldn't find port");

    // Replace the input port with a backedge.  If it turns out that this port
    // is never driven, an uninitialized wire will be materialized at the end.
    if (port.isInput()) {
      operands.push_back(createBackedge(portResult, portType));
      continue;
    }

    // If the result has an analog type and is used only by attach op, try
    // eliminating a temporary wire by directly using an attached value.
    if (type_isa<AnalogType>(portResult.getType()) && portResult.hasOneUse()) {
      if (auto attach = dyn_cast<AttachOp>(*portResult.getUsers().begin())) {
        if (auto source = getSingleNonInstanceOperand(attach)) {
          auto loweredResult = getPossiblyInoutLoweredValue(source);
          operands.push_back(loweredResult);
          (void)setLowering(portResult, loweredResult);
          continue;
        }
      }
    }

    // Create a wire for each inout operand, so there is something to connect
    // to. The instance becomes the sole driver of this wire.
    auto wire = sv::WireOp::create(builder, portType,
                                   "." + port.getName().str() + ".wire");

    // Know that the argument FIRRTL value is equal to this wire, allowing
    // connects to it to be lowered.
    (void)setLowering(portResult, wire);

    operands.push_back(wire);
  }

  // If this instance is destined to be lowered to a bind, generate a symbol
  // for it and generate a bind op.  Enter the bind into global
  // CircuitLoweringState so that this can be moved outside of module once
  // we're guaranteed to not be a parallel context.
  auto innerSym = oldInstance.getInnerSymAttr();
  if (oldInstance.getLowerToBind()) {
    if (!innerSym)
      std::tie(innerSym, std::ignore) = getOrAddInnerSym(
          oldInstance.getContext(), oldInstance.getInnerSymAttr(), 0,
          [&]() -> hw::InnerSymbolNamespace & { return moduleNamespace; });

    auto bindOp = sv::BindOp::create(builder, theModule.getNameAttr(),
                                     innerSym.getSymName());
    // If the lowered op already had output file information, then use that.
    // Otherwise, generate some default bind information.
    if (auto outputFile = oldInstance->getAttr("output_file"))
      bindOp->setAttr("output_file", outputFile);
    // Add the bind to the circuit state.  This will be moved outside of the
    // encapsulating module after all modules have been processed in parallel.
    circuitState.addBind(bindOp);
  }

  // Create the new hw.instance operation.
  auto newInstance =
      hw::InstanceOp::create(builder, newModule, oldInstance.getNameAttr(),
                             operands, parameters, innerSym);

  if (oldInstance.getLowerToBind() || oldInstance.getDoNotPrint())
    newInstance.setDoNotPrintAttr(builder.getUnitAttr());

  if (newInstance.getInnerSymAttr())
    if (auto forceName = circuitState.instanceForceNames.lookup(
            {newInstance->getParentOfType<hw::HWModuleOp>().getNameAttr(),
             newInstance.getInnerNameAttr()}))
      newInstance->setAttr("hw.verilogName", forceName);

  // Now that we have the new hw.instance, we need to remap all of the users
  // of the outputs/results to the values returned by the instance.
  unsigned resultNo = 0;
  for (size_t portIndex = 0, e = portInfo.size(); portIndex != e; ++portIndex) {
    auto &port = portInfo[portIndex];
    if (!port.isOutput() || isZeroBitFIRRTLType(port.type))
      continue;

    Value resultVal = newInstance.getResult(resultNo);

    auto oldPortResult = oldInstance.getResult(portIndex);
    (void)setLowering(oldPortResult, resultVal);
    ++resultNo;
  }
  return success();
}

LogicalResult FIRRTLLowering::visitDecl(ContractOp oldOp) {
  SmallVector<Value> inputs;
  SmallVector<Type> types;
  for (auto input : oldOp.getInputs()) {
    auto lowered = getLoweredValue(input);
    if (!lowered)
      return failure();
    inputs.push_back(lowered);
    types.push_back(lowered.getType());
  }

  auto newOp = verif::ContractOp::create(builder, types, inputs);
  newOp->setDiscardableAttrs(oldOp->getDiscardableAttrDictionary());
  auto &body = newOp.getBody().emplaceBlock();

  for (auto [newResult, oldResult, oldArg] :
       llvm::zip(newOp.getResults(), oldOp.getResults(),
                 oldOp.getBody().getArguments())) {
    if (failed(setLowering(oldResult, newResult)))
      return failure();
    if (failed(setLowering(oldArg, newResult)))
      return failure();
  }

  body.getOperations().splice(body.end(),
                              oldOp.getBody().front().getOperations());
  addToWorklist(body);

  return success();
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Lower a cast that is a noop at the HW level.
LogicalResult FIRRTLLowering::lowerNoopCast(Operation *op) {
  auto operand = getPossiblyInoutLoweredValue(op->getOperand(0));
  if (!operand)
    return failure();

  // Noop cast.
  return setLowering(op->getResult(0), operand);
}

LogicalResult FIRRTLLowering::visitExpr(AsSIntPrimOp op) {
  if (isa<ClockType>(op.getInput().getType()))
    return setLowering(op->getResult(0),
                       getLoweredNonClockValue(op.getInput()));
  return lowerNoopCast(op);
}

LogicalResult FIRRTLLowering::visitExpr(AsUIntPrimOp op) {
  if (isa<ClockType>(op.getInput().getType()))
    return setLowering(op->getResult(0),
                       getLoweredNonClockValue(op.getInput()));
  return lowerNoopCast(op);
}

LogicalResult FIRRTLLowering::visitExpr(AsClockPrimOp op) {
  return setLoweringTo<seq::ToClockOp>(op, getLoweredValue(op.getInput()));
}

LogicalResult FIRRTLLowering::visitUnrealizedConversionCast(
    mlir::UnrealizedConversionCastOp op) {
  // General lowering for non-unary casts.
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return failure();

  auto operand = op.getOperand(0);
  auto result = op.getResult(0);

  // FIRRTL -> FIRRTL
  if (type_isa<FIRRTLType>(operand.getType()) &&
      type_isa<FIRRTLType>(result.getType()))
    return lowerNoopCast(op);

  // other -> FIRRTL
  // other -> other
  if (!type_isa<FIRRTLType>(operand.getType())) {
    if (type_isa<FIRRTLType>(result.getType()))
      return setLowering(result, getPossiblyInoutLoweredValue(operand));
    return failure(); // general foreign op lowering for other -> other
  }

  // FIRRTL -> other
  // Otherwise must be a conversion from FIRRTL type to standard type.
  auto loweredResult = getLoweredValue(operand);
  if (!loweredResult) {
    // If this is a conversion from a zero bit HW type to firrtl value, then
    // we want to successfully lower this to a null Value.
    if (operand.getType().isSignlessInteger(0)) {
      return setLowering(result, Value());
    }
    return failure();
  }

  // We lower builtin.unrealized_conversion_cast converting from a firrtl type
  // to a standard type into the lowered operand.
  result.replaceAllUsesWith(loweredResult);
  return success();
}

LogicalResult FIRRTLLowering::visitExpr(HWStructCastOp op) {
  // Conversions from hw struct types to FIRRTL types are lowered as the
  // input operand.
  if (auto opStructType = dyn_cast<hw::StructType>(op.getOperand().getType()))
    return setLowering(op, op.getOperand());

  // Otherwise must be a conversion from FIRRTL bundle type to hw struct
  // type.
  auto result = getLoweredValue(op.getOperand());
  if (!result)
    return failure();

  // We lower firrtl.stdStructCast converting from a firrtl bundle to an hw
  // struct type into the lowered operand.
  op.replaceAllUsesWith(result);
  return success();
}

LogicalResult FIRRTLLowering::visitExpr(BitCastOp op) {
  auto operand = getLoweredValue(op.getOperand());
  if (!operand)
    return failure();
  auto resultType = lowerType(op.getType());
  if (!resultType)
    return failure();

  return setLoweringTo<hw::BitcastOp>(op, resultType, operand);
}

LogicalResult FIRRTLLowering::visitExpr(CvtPrimOp op) {
  auto operand = getLoweredValue(op.getOperand());
  if (!operand) {
    return handleZeroBit(op.getOperand(), [&]() {
      // Unsigned zero bit to Signed is 1b0.
      if (type_cast<IntType>(op.getOperand().getType()).isUnsigned())
        return setLowering(op, getOrCreateIntConstant(1, 0));
      // Signed->Signed is a zero bit value.
      return setLowering(op, Value());
    });
  }

  // Signed to signed is a noop.
  if (type_cast<IntType>(op.getOperand().getType()).isSigned())
    return setLowering(op, operand);

  // Otherwise prepend a zero bit.
  auto zero = getOrCreateIntConstant(1, 0);
  return setLoweringTo<comb::ConcatOp>(op, zero, operand);
}

LogicalResult FIRRTLLowering::visitExpr(NotPrimOp op) {
  auto operand = getLoweredValue(op.getInput());
  if (!operand)
    return failure();
  // ~x  ---> x ^ 0xFF
  auto allOnes = getOrCreateIntConstant(
      APInt::getAllOnes(operand.getType().getIntOrFloatBitWidth()));
  return setLoweringTo<comb::XorOp>(op, operand, allOnes, true);
}

LogicalResult FIRRTLLowering::visitExpr(NegPrimOp op) {
  // FIRRTL negate always adds a bit.
  // -x ---> 0-sext(x) or 0-zext(x)
  auto operand = getLoweredAndExtendedValue(op.getInput(), op.getType());
  if (!operand)
    return failure();

  auto resultType = lowerType(op.getType());

  auto zero = getOrCreateIntConstant(resultType.getIntOrFloatBitWidth(), 0);
  return setLoweringTo<comb::SubOp>(op, zero, operand, true);
}

// Pad is a noop or extension operation.
LogicalResult FIRRTLLowering::visitExpr(PadPrimOp op) {
  auto operand = getLoweredAndExtendedValue(op.getInput(), op.getType());
  if (!operand)
    return failure();
  return setLowering(op, operand);
}

LogicalResult FIRRTLLowering::visitExpr(XorRPrimOp op) {
  auto operand = getLoweredValue(op.getInput());
  if (!operand) {
    return handleZeroBit(op.getInput(), [&]() {
      return setLowering(op, getOrCreateIntConstant(1, 0));
    });
    return failure();
  }

  return setLoweringTo<comb::ParityOp>(op, builder.getIntegerType(1), operand,
                                       true);
}

LogicalResult FIRRTLLowering::visitExpr(AndRPrimOp op) {
  auto operand = getLoweredValue(op.getInput());
  if (!operand) {
    return handleZeroBit(op.getInput(), [&]() {
      return setLowering(op, getOrCreateIntConstant(1, 1));
    });
  }

  // Lower AndR to == -1
  return setLoweringTo<comb::ICmpOp>(
      op, ICmpPredicate::eq, operand,
      getOrCreateIntConstant(
          APInt::getAllOnes(operand.getType().getIntOrFloatBitWidth())),
      true);
}

LogicalResult FIRRTLLowering::visitExpr(OrRPrimOp op) {
  auto operand = getLoweredValue(op.getInput());
  if (!operand) {
    return handleZeroBit(op.getInput(), [&]() {
      return setLowering(op, getOrCreateIntConstant(1, 0));
    });
    return failure();
  }

  // Lower OrR to != 0
  return setLoweringTo<comb::ICmpOp>(
      op, ICmpPredicate::ne, operand,
      getOrCreateIntConstant(operand.getType().getIntOrFloatBitWidth(), 0),
      true);
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

template <typename ResultOpType>
LogicalResult FIRRTLLowering::lowerBinOpToVariadic(Operation *op) {
  auto resultType = op->getResult(0).getType();
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);
  if (!lhs || !rhs)
    return failure();

  return setLoweringTo<ResultOpType>(op, lhs, rhs, true);
}

/// Element-wise logical operations can be lowered into bitcast and normal comb
/// operations. Eventually we might want to introduce elementwise operations
/// into HW/SV level as well.
template <typename ResultOpType>
LogicalResult FIRRTLLowering::lowerElementwiseLogicalOp(Operation *op) {
  auto resultType = op->getResult(0).getType();
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);

  if (!lhs || !rhs)
    return failure();
  auto bitwidth = firrtl::getBitWidth(type_cast<FIRRTLBaseType>(resultType));

  if (!bitwidth)
    return failure();

  // TODO: Introduce elementwise operations to HW dialect instead of abusing
  // bitcast operations.
  auto intType = builder.getIntegerType(*bitwidth);
  auto retType = lhs.getType();
  lhs = builder.createOrFold<hw::BitcastOp>(intType, lhs);
  rhs = builder.createOrFold<hw::BitcastOp>(intType, rhs);
  auto result = builder.createOrFold<ResultOpType>(lhs, rhs, /*twoState=*/true);
  return setLoweringTo<hw::BitcastOp>(op, retType, result);
}

/// lowerBinOp extends each operand to the destination type, then performs the
/// specified binary operator.
template <typename ResultUnsignedOpType, typename ResultSignedOpType>
LogicalResult FIRRTLLowering::lowerBinOp(Operation *op) {
  // Extend the two operands to match the destination type.
  auto resultType = op->getResult(0).getType();
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);
  if (!lhs || !rhs)
    return failure();

  // Emit the result operation.
  if (type_cast<IntType>(resultType).isSigned())
    return setLoweringTo<ResultSignedOpType>(op, lhs, rhs, true);
  return setLoweringTo<ResultUnsignedOpType>(op, lhs, rhs, true);
}

/// lowerCmpOp extends each operand to the longest type, then performs the
/// specified binary operator.
LogicalResult FIRRTLLowering::lowerCmpOp(Operation *op, ICmpPredicate signedOp,
                                         ICmpPredicate unsignedOp) {
  // Extend the two operands to match the longest type.
  auto lhsIntType = type_cast<IntType>(op->getOperand(0).getType());
  auto rhsIntType = type_cast<IntType>(op->getOperand(1).getType());
  if (!lhsIntType.hasWidth() || !rhsIntType.hasWidth())
    return failure();

  auto cmpType = getWidestIntType(lhsIntType, rhsIntType);
  if (cmpType.getWidth() == 0) // Handle 0-width inputs by promoting to 1 bit.
    cmpType = UIntType::get(builder.getContext(), 1);
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), cmpType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), cmpType);
  if (!lhs || !rhs)
    return failure();

  // Emit the result operation.
  Type resultType = builder.getIntegerType(1);
  return setLoweringTo<comb::ICmpOp>(
      op, resultType, lhsIntType.isSigned() ? signedOp : unsignedOp, lhs, rhs,
      true);
}

/// Lower a divide or dynamic shift, where the operation has to be performed
/// in the widest type of the result and two inputs then truncated down.
template <typename SignedOp, typename UnsignedOp>
LogicalResult FIRRTLLowering::lowerDivLikeOp(Operation *op) {
  // hw has equal types for these, firrtl doesn't.  The type of the firrtl
  // RHS may be wider than the LHS, and we cannot truncate off the high bits
  // (because an overlarge amount is supposed to shift in sign or zero bits).
  auto opType = type_cast<IntType>(op->getResult(0).getType());
  if (opType.getWidth() == 0)
    return setLowering(op->getResult(0), Value());

  auto resultType = getWidestIntType(opType, op->getOperand(1).getType());
  resultType = getWidestIntType(resultType, op->getOperand(0).getType());
  auto lhs = getLoweredAndExtendedValue(op->getOperand(0), resultType);
  auto rhs = getLoweredAndExtendedValue(op->getOperand(1), resultType);
  if (!lhs || !rhs)
    return failure();

  Value result;
  if (opType.isSigned())
    result = builder.createOrFold<SignedOp>(lhs, rhs, true);
  else
    result = builder.createOrFold<UnsignedOp>(lhs, rhs, true);

  if (auto *definingOp = result.getDefiningOp())
    tryCopyName(definingOp, op);

  if (resultType == opType)
    return setLowering(op->getResult(0), result);
  return setLoweringTo<comb::ExtractOp>(op, lowerType(opType), result, 0);
}

LogicalResult FIRRTLLowering::visitExpr(CatPrimOp op) {
  // Handle the case of no operands - should result in a 0-bit value
  if (op.getInputs().empty())
    return setLowering(op, Value());

  SmallVector<Value> loweredOperands;

  // Lower all operands, filtering out zero-bit values
  for (auto operand : op.getInputs()) {
    auto loweredOperand = getLoweredValue(operand);
    if (loweredOperand) {
      loweredOperands.push_back(loweredOperand);
    } else {
      // Check if this is a zero-bit operand, which we can skip
      auto result = handleZeroBit(operand, [&]() { return success(); });
      if (failed(result))
        return failure();
      // Zero-bit operands are skipped (not added to loweredOperands)
    }
  }

  // If no non-zero operands, return 0-bit value
  if (loweredOperands.empty())
    return setLowering(op, Value());

  // Use comb.concat
  return setLoweringTo<comb::ConcatOp>(op, loweredOperands);
}

//===----------------------------------------------------------------------===//
// Verif Operations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitExpr(IsXIntrinsicOp op) {
  auto input = getLoweredNonClockValue(op.getArg());
  if (!input)
    return failure();

  if (!isa<IntType>(input.getType())) {
    auto srcType = op.getArg().getType();
    auto bitwidth = firrtl::getBitWidth(type_cast<FIRRTLBaseType>(srcType));
    assert(bitwidth && "Unknown width");
    auto intType = builder.getIntegerType(*bitwidth);
    input = builder.createOrFold<hw::BitcastOp>(intType, input);
  }

  return setLoweringTo<comb::ICmpOp>(
      op, ICmpPredicate::ceq, input,
      getOrCreateXConstant(input.getType().getIntOrFloatBitWidth()), true);
}

LogicalResult FIRRTLLowering::visitStmt(FPGAProbeIntrinsicOp op) {
  auto operand = getLoweredValue(op.getInput());
  hw::WireOp::create(builder, operand);
  return success();
}

LogicalResult FIRRTLLowering::visitExpr(PlusArgsTestIntrinsicOp op) {
  return setLoweringTo<sim::PlusArgsTestOp>(op, builder.getIntegerType(1),
                                            op.getFormatStringAttr());
}

LogicalResult FIRRTLLowering::visitExpr(PlusArgsValueIntrinsicOp op) {
  auto type = lowerType(op.getResult().getType());
  if (!type)
    return failure();

  auto valueOp = sim::PlusArgsValueOp::create(
      builder, builder.getIntegerType(1), type, op.getFormatStringAttr());
  if (failed(setLowering(op.getResult(), valueOp.getResult())))
    return failure();
  if (failed(setLowering(op.getFound(), valueOp.getFound())))
    return failure();
  return success();
}

LogicalResult FIRRTLLowering::visitExpr(SizeOfIntrinsicOp op) {
  op.emitError("SizeOf should have been resolved.");
  return failure();
}

LogicalResult FIRRTLLowering::visitExpr(ClockGateIntrinsicOp op) {
  Value testEnable;
  if (op.getTestEnable())
    testEnable = getLoweredValue(op.getTestEnable());
  return setLoweringTo<seq::ClockGateOp>(
      op, getLoweredValue(op.getInput()), getLoweredValue(op.getEnable()),
      testEnable, /*inner_sym=*/hw::InnerSymAttr{});
}

LogicalResult FIRRTLLowering::visitExpr(ClockInverterIntrinsicOp op) {
  auto operand = getLoweredValue(op.getInput());
  return setLoweringTo<seq::ClockInverterOp>(op, operand);
}

LogicalResult FIRRTLLowering::visitExpr(ClockDividerIntrinsicOp op) {
  auto operand = getLoweredValue(op.getInput());
  return setLoweringTo<seq::ClockDividerOp>(op, operand, op.getPow2());
}

LogicalResult FIRRTLLowering::visitExpr(LTLAndIntrinsicOp op) {
  return setLoweringToLTL<ltl::AndOp>(
      op,
      ValueRange{getLoweredValue(op.getLhs()), getLoweredValue(op.getRhs())});
}

LogicalResult FIRRTLLowering::visitExpr(LTLOrIntrinsicOp op) {
  return setLoweringToLTL<ltl::OrOp>(
      op,
      ValueRange{getLoweredValue(op.getLhs()), getLoweredValue(op.getRhs())});
}

LogicalResult FIRRTLLowering::visitExpr(LTLIntersectIntrinsicOp op) {
  return setLoweringToLTL<ltl::IntersectOp>(
      op,
      ValueRange{getLoweredValue(op.getLhs()), getLoweredValue(op.getRhs())});
}

LogicalResult FIRRTLLowering::visitExpr(LTLDelayIntrinsicOp op) {
  return setLoweringToLTL<ltl::DelayOp>(op, getLoweredValue(op.getInput()),
                                        op.getDelayAttr(), op.getLengthAttr());
}

LogicalResult FIRRTLLowering::visitExpr(LTLConcatIntrinsicOp op) {
  return setLoweringToLTL<ltl::ConcatOp>(
      op,
      ValueRange{getLoweredValue(op.getLhs()), getLoweredValue(op.getRhs())});
}

LogicalResult FIRRTLLowering::visitExpr(LTLRepeatIntrinsicOp op) {
  return setLoweringToLTL<ltl::RepeatOp>(op, getLoweredValue(op.getInput()),
                                         op.getBaseAttr(), op.getMoreAttr());
}

LogicalResult FIRRTLLowering::visitExpr(LTLGoToRepeatIntrinsicOp op) {
  return setLoweringToLTL<ltl::GoToRepeatOp>(
      op, getLoweredValue(op.getInput()), op.getBaseAttr(), op.getMoreAttr());
}

LogicalResult FIRRTLLowering::visitExpr(LTLNonConsecutiveRepeatIntrinsicOp op) {
  return setLoweringToLTL<ltl::NonConsecutiveRepeatOp>(
      op, getLoweredValue(op.getInput()), op.getBaseAttr(), op.getMoreAttr());
}

LogicalResult FIRRTLLowering::visitExpr(LTLNotIntrinsicOp op) {
  return setLoweringToLTL<ltl::NotOp>(op, getLoweredValue(op.getInput()));
}

LogicalResult FIRRTLLowering::visitExpr(LTLImplicationIntrinsicOp op) {
  return setLoweringToLTL<ltl::ImplicationOp>(
      op,
      ValueRange{getLoweredValue(op.getLhs()), getLoweredValue(op.getRhs())});
}

LogicalResult FIRRTLLowering::visitExpr(LTLUntilIntrinsicOp op) {
  return setLoweringToLTL<ltl::UntilOp>(
      op,
      ValueRange{getLoweredValue(op.getLhs()), getLoweredValue(op.getRhs())});
}

LogicalResult FIRRTLLowering::visitExpr(LTLEventuallyIntrinsicOp op) {
  return setLoweringToLTL<ltl::EventuallyOp>(op,
                                             getLoweredValue(op.getInput()));
}

LogicalResult FIRRTLLowering::visitExpr(LTLClockIntrinsicOp op) {
  return setLoweringToLTL<ltl::ClockOp>(op, getLoweredValue(op.getInput()),
                                        ltl::ClockEdge::Pos,
                                        getLoweredNonClockValue(op.getClock()));
}

template <typename TargetOp, typename IntrinsicOp>
LogicalResult FIRRTLLowering::lowerVerifIntrinsicOp(IntrinsicOp op) {
  auto property = getLoweredValue(op.getProperty());
  auto enable = op.getEnable() ? getLoweredValue(op.getEnable()) : Value();
  TargetOp::create(builder, property, enable, op.getLabelAttr());
  return success();
}

LogicalResult FIRRTLLowering::visitStmt(VerifAssertIntrinsicOp op) {
  return lowerVerifIntrinsicOp<verif::AssertOp>(op);
}

LogicalResult FIRRTLLowering::visitStmt(VerifAssumeIntrinsicOp op) {
  return lowerVerifIntrinsicOp<verif::AssumeOp>(op);
}

LogicalResult FIRRTLLowering::visitStmt(VerifCoverIntrinsicOp op) {
  return lowerVerifIntrinsicOp<verif::CoverOp>(op);
}

LogicalResult FIRRTLLowering::visitStmt(VerifRequireIntrinsicOp op) {
  if (!isa<verif::ContractOp>(op->getParentOp()))
    return lowerVerifIntrinsicOp<verif::AssertOp>(op);
  return lowerVerifIntrinsicOp<verif::RequireOp>(op);
}

LogicalResult FIRRTLLowering::visitStmt(VerifEnsureIntrinsicOp op) {
  if (!isa<verif::ContractOp>(op->getParentOp()))
    return lowerVerifIntrinsicOp<verif::AssertOp>(op);
  return lowerVerifIntrinsicOp<verif::EnsureOp>(op);
}

LogicalResult FIRRTLLowering::visitExpr(HasBeenResetIntrinsicOp op) {
  auto clock = getLoweredNonClockValue(op.getClock());
  auto reset = getLoweredValue(op.getReset());
  if (!clock || !reset)
    return failure();
  auto resetType = op.getReset().getType();
  auto uintResetType = dyn_cast<UIntType>(resetType);
  auto isSync = uintResetType && uintResetType.getWidth() == 1;
  auto isAsync = isa<AsyncResetType>(resetType);
  if (!isAsync && !isSync) {
    auto d = op.emitError("uninferred reset passed to 'has_been_reset'; "
                          "requires sync or async reset");
    d.attachNote() << "reset is of type " << resetType
                   << ", should be '!firrtl.uint<1>' or '!firrtl.asyncreset'";
    return failure();
  }
  return setLoweringTo<verif::HasBeenResetOp>(op, clock, reset, isAsync);
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitExpr(BitsPrimOp op) {
  auto input = getLoweredValue(op.getInput());
  if (!input)
    return failure();

  Type resultType = builder.getIntegerType(op.getHi() - op.getLo() + 1);
  return setLoweringTo<comb::ExtractOp>(op, resultType, input, op.getLo());
}

LogicalResult FIRRTLLowering::visitExpr(InvalidValueOp op) {
  auto resultTy = lowerType(op.getType());
  if (!resultTy)
    return failure();

  // Values of analog type always need to be lowered to something with inout
  // type.  We do that by lowering to a wire and return that.  As with the
  // SFC, we do not connect anything to this, because it is bidirectional.
  if (type_isa<AnalogType>(op.getType()))
    // This is a locally visible, private wire created by the compiler, so do
    // not attach a symbol name.
    return setLoweringTo<sv::WireOp>(op, resultTy, ".invalid_analog");

  // We don't allow aggregate values which contain values of analog types.
  if (type_cast<FIRRTLBaseType>(op.getType()).containsAnalog())
    return failure();

  // We lower invalid to 0.  TODO: the FIRRTL spec mentions something about
  // lowering it to a random value, we should see if this is what we need to
  // do.
  if (auto bitwidth =
          firrtl::getBitWidth(type_cast<FIRRTLBaseType>(op.getType()))) {
    if (*bitwidth == 0) // Let the caller handle zero width values.
      return failure();

    auto constant = getOrCreateIntConstant(*bitwidth, 0);
    // If the result is an aggregate value, we have to bitcast the constant.
    if (!type_isa<IntegerType>(resultTy))
      constant = hw::BitcastOp::create(builder, resultTy, constant);
    return setLowering(op, constant);
  }

  // Invalid for bundles isn't supported.
  op.emitOpError("unsupported type");
  return failure();
}

LogicalResult FIRRTLLowering::visitExpr(HeadPrimOp op) {
  auto input = getLoweredValue(op.getInput());
  if (!input)
    return failure();
  auto inWidth = type_cast<IntegerType>(input.getType()).getWidth();
  if (op.getAmount() == 0)
    return setLowering(op, Value());
  Type resultType = builder.getIntegerType(op.getAmount());
  return setLoweringTo<comb::ExtractOp>(op, resultType, input,
                                        inWidth - op.getAmount());
}

LogicalResult FIRRTLLowering::visitExpr(ShlPrimOp op) {
  auto input = getLoweredValue(op.getInput());
  if (!input) {
    return handleZeroBit(op.getInput(), [&]() {
      if (op.getAmount() == 0)
        return failure();
      return setLowering(op, getOrCreateIntConstant(op.getAmount(), 0));
    });
  }

  // Handle the degenerate case.
  if (op.getAmount() == 0)
    return setLowering(op, input);

  auto zero = getOrCreateIntConstant(op.getAmount(), 0);
  return setLoweringTo<comb::ConcatOp>(op, input, zero);
}

LogicalResult FIRRTLLowering::visitExpr(ShrPrimOp op) {
  auto input = getLoweredValue(op.getInput());
  if (!input)
    return failure();

  // Handle the special degenerate cases.
  auto inWidth = type_cast<IntegerType>(input.getType()).getWidth();
  auto shiftAmount = op.getAmount();
  if (shiftAmount >= inWidth) {
    // Unsigned shift by full width returns a single-bit zero.
    if (type_cast<IntType>(op.getInput().getType()).isUnsigned())
      return setLowering(op, {});

    // Signed shift by full width is equivalent to extracting the sign bit.
    shiftAmount = inWidth - 1;
  }

  Type resultType = builder.getIntegerType(inWidth - shiftAmount);
  return setLoweringTo<comb::ExtractOp>(op, resultType, input, shiftAmount);
}

LogicalResult FIRRTLLowering::visitExpr(TailPrimOp op) {
  auto input = getLoweredValue(op.getInput());
  if (!input)
    return failure();

  auto inWidth = type_cast<IntegerType>(input.getType()).getWidth();
  if (inWidth == op.getAmount())
    return setLowering(op, Value());
  Type resultType = builder.getIntegerType(inWidth - op.getAmount());
  return setLoweringTo<comb::ExtractOp>(op, resultType, input, 0);
}

LogicalResult FIRRTLLowering::visitExpr(MuxPrimOp op) {
  auto cond = getLoweredValue(op.getSel());
  auto ifTrue = getLoweredAndExtendedValue(op.getHigh(), op.getType());
  auto ifFalse = getLoweredAndExtendedValue(op.getLow(), op.getType());
  if (!cond || !ifTrue || !ifFalse)
    return failure();

  if (isa<ClockType>(op.getType()))
    return setLoweringTo<seq::ClockMuxOp>(op, cond, ifTrue, ifFalse);
  return setLoweringTo<comb::MuxOp>(op, ifTrue.getType(), cond, ifTrue, ifFalse,
                                    true);
}

LogicalResult FIRRTLLowering::visitExpr(Mux2CellIntrinsicOp op) {
  auto cond = getLoweredValue(op.getSel());
  auto ifTrue = getLoweredAndExtendedValue(op.getHigh(), op.getType());
  auto ifFalse = getLoweredAndExtendedValue(op.getLow(), op.getType());
  if (!cond || !ifTrue || !ifFalse)
    return failure();

  auto val = comb::MuxOp::create(builder, ifTrue.getType(), cond, ifTrue,
                                 ifFalse, true);
  return setLowering(op, createValueWithMuxAnnotation(val, true));
}

LogicalResult FIRRTLLowering::visitExpr(Mux4CellIntrinsicOp op) {
  auto sel = getLoweredValue(op.getSel());
  auto v3 = getLoweredAndExtendedValue(op.getV3(), op.getType());
  auto v2 = getLoweredAndExtendedValue(op.getV2(), op.getType());
  auto v1 = getLoweredAndExtendedValue(op.getV1(), op.getType());
  auto v0 = getLoweredAndExtendedValue(op.getV0(), op.getType());
  if (!sel || !v3 || !v2 || !v1 || !v0)
    return failure();
  Value array[] = {v3, v2, v1, v0};
  auto create = hw::ArrayCreateOp::create(builder, array);
  auto val = hw::ArrayGetOp::create(builder, create, sel);
  return setLowering(op, createValueWithMuxAnnotation(val, false));
}

// Construct a value with vendor specific pragmas to utilize MUX cells.
// Specifically we annotate pragmas in the following form.
//
// For an array indexing:
// ```
//   wire GEN;
//   /* synopsys infer_mux_override */
//   assign GEN = array[index] /* cadence map_to_mux */;
// ```
//
// For a mux:
// ```
//   wire GEN;
//   /* synopsys infer_mux_override */
//   assign GEN = sel ? /* cadence map_to_mux */ high : low;
// ```
Value FIRRTLLowering::createValueWithMuxAnnotation(Operation *op, bool isMux2) {
  assert(op->getNumResults() == 1 && "only expect a single result");
  auto val = op->getResult(0);
  auto valWire = sv::WireOp::create(builder, val.getType());
  // Use SV attributes to annotate pragmas.
  circt::sv::setSVAttributes(
      op, sv::SVAttributeAttr::get(builder.getContext(), "cadence map_to_mux",
                                   /*emitAsComment=*/true));

  // For operands, create temporary wires with optimization blockers(inner
  // symbols) so that the AST structure will never be destoyed in the later
  // pipeline.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    StringRef namehint = isMux2 ? "mux2cell_in" : "mux4cell_in";
    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      auto [innerSym, _] = getOrAddInnerSym(
          op->getContext(), /*attr=*/nullptr, 0,
          [&]() -> hw::InnerSymbolNamespace & { return moduleNamespace; });
      auto wire =
          hw::WireOp::create(builder, operand, namehint + Twine(idx), innerSym);
      op->setOperand(idx, wire);
    }
  }

  auto assignOp = sv::AssignOp::create(builder, valWire, val);
  sv::setSVAttributes(assignOp,
                      sv::SVAttributeAttr::get(builder.getContext(),
                                               "synopsys infer_mux_override",
                                               /*emitAsComment=*/true));
  return sv::ReadInOutOp::create(builder, valWire);
}

Value FIRRTLLowering::createArrayIndexing(Value array, Value index) {

  auto size = hw::type_cast<hw::ArrayType>(array.getType()).getNumElements();
  // Extend to power of 2.  FIRRTL semantics say out-of-bounds access result in
  // an indeterminate value.  Existing chisel code depends on this behavior
  // being "return index 0".  Ideally, we would tail extend the array to improve
  // optimization.
  if (!llvm::isPowerOf2_64(size)) {
    auto extElem = getOrCreateIntConstant(APInt(llvm::Log2_64_Ceil(size), 0));
    auto extValue = hw::ArrayGetOp::create(builder, array, extElem);
    SmallVector<Value> temp(llvm::NextPowerOf2(size) - size, extValue);
    auto ext = hw::ArrayCreateOp::create(builder, temp);
    Value temp2[] = {ext.getResult(), array};
    array = hw::ArrayConcatOp::create(builder, temp2);
  }

  Value inBoundsRead = hw::ArrayGetOp::create(builder, array, index);

  return inBoundsRead;
}

LogicalResult FIRRTLLowering::visitExpr(MultibitMuxOp op) {
  // Lower and resize to the index width.
  auto index = getLoweredAndExtOrTruncValue(
      op.getIndex(),
      UIntType::get(op.getContext(),
                    getBitWidthFromVectorSize(op.getInputs().size())));

  if (!index)
    return failure();
  SmallVector<Value> loweredInputs;
  loweredInputs.reserve(op.getInputs().size());
  for (auto input : op.getInputs()) {
    auto lowered = getLoweredAndExtendedValue(input, op.getType());
    if (!lowered)
      return failure();
    loweredInputs.push_back(lowered);
  }

  Value array = hw::ArrayCreateOp::create(builder, loweredInputs);
  return setLowering(op, createArrayIndexing(array, index));
}

LogicalResult FIRRTLLowering::visitExpr(VerbatimExprOp op) {
  auto resultTy = lowerType(op.getType());
  if (!resultTy)
    return failure();

  SmallVector<Value, 4> operands;
  operands.reserve(op.getSubstitutions().size());
  for (auto operand : op.getSubstitutions()) {
    auto lowered = getLoweredValue(operand);
    if (!lowered)
      return failure();
    operands.push_back(lowered);
  }

  ArrayAttr symbols = op.getSymbolsAttr();
  if (!symbols)
    symbols = ArrayAttr::get(op.getContext(), {});

  return setLoweringTo<sv::VerbatimExprOp>(op, resultTy, op.getTextAttr(),
                                           operands, symbols);
}

LogicalResult FIRRTLLowering::visitExpr(XMRRefOp op) {
  // This XMR is accessed solely by FIRRTL statements that mutate the probe.
  // To avoid the use of clock wires, create an `i1` wire and ensure that
  // all connections are also of the `i1` type.
  Type baseType = op.getType().getType();

  Type xmrType;
  if (isa<ClockType>(baseType))
    xmrType = builder.getIntegerType(1);
  else
    xmrType = lowerType(baseType);

  return setLoweringTo<sv::XMRRefOp>(op, sv::InOutType::get(xmrType),
                                     op.getRef(), op.getVerbatimSuffixAttr());
}

LogicalResult FIRRTLLowering::visitExpr(XMRDerefOp op) {
  // When an XMR targets a clock wire, replace it with an `i1` wire, but
  // introduce a clock-typed read op into the design afterwards.
  Type xmrType;
  if (isa<ClockType>(op.getType()))
    xmrType = builder.getIntegerType(1);
  else
    xmrType = lowerType(op.getType());

  auto xmr = sv::XMRRefOp::create(builder, sv::InOutType::get(xmrType),
                                  op.getRef(), op.getVerbatimSuffixAttr());
  auto readXmr = getReadValue(xmr);
  if (!isa<ClockType>(op.getType()))
    return setLowering(op, readXmr);
  return setLoweringTo<seq::ToClockOp>(op, readXmr);
}

// Do nothing when lowering fstring operations.  These need to be handled at
// their usage sites (at the PrintfOps).
LogicalResult FIRRTLLowering::visitExpr(TimeOp op) { return success(); }
LogicalResult FIRRTLLowering::visitExpr(HierarchicalModuleNameOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

LogicalResult FIRRTLLowering::visitStmt(SkipOp op) {
  // Nothing!  We could emit an comment as a verbatim op if there were a
  // reason to.
  return success();
}

/// Resolve a connection to `destVal`, an `hw::WireOp` or `seq::FirRegOp`, by
/// updating the input operand to be `srcVal`. Returns true if the update was
/// made and the connection can be considered lowered. Returns false if the
/// destination isn't a wire or register with an input operand to be updated.
/// Returns failure if the destination is a subaccess operation. These should be
/// transposed to the right-hand-side by a pre-pass.
FailureOr<bool> FIRRTLLowering::lowerConnect(Value destVal, Value srcVal) {
  auto srcType = srcVal.getType();
  auto dstType = destVal.getType();
  if (srcType != dstType &&
      (isa<hw::TypeAliasType>(srcType) || isa<hw::TypeAliasType>(dstType))) {
    srcVal = hw::BitcastOp::create(builder, destVal.getType(), srcVal);
  }
  return TypeSwitch<Operation *, FailureOr<bool>>(destVal.getDefiningOp())
      .Case<hw::WireOp>([&](auto op) {
        maybeUnused(op.getInput());
        op.getInputMutable().assign(srcVal);
        return true;
      })
      .Case<seq::FirRegOp>([&](auto op) {
        maybeUnused(op.getNext());
        op.getNextMutable().assign(srcVal);
        return true;
      })
      .Case<hw::StructExtractOp, hw::ArrayGetOp>([](auto op) {
        // NOTE: msvc thinks `return op.emitOpError(...);` is ambiguous. So
        // return `failure()` separately.
        op.emitOpError("used as connect destination");
        return failure();
      })
      .Default([](auto) { return false; });
}

LogicalResult FIRRTLLowering::visitStmt(ConnectOp op) {
  auto dest = op.getDest();
  // The source can be a smaller integer, extend it as appropriate if so.
  auto destType = type_cast<FIRRTLBaseType>(dest.getType()).getPassiveType();
  auto srcVal = getLoweredAndExtendedValue(op.getSrc(), destType);
  if (!srcVal)
    return handleZeroBit(op.getSrc(), []() { return success(); });

  auto destVal = getPossiblyInoutLoweredValue(dest);
  if (!destVal)
    return failure();

  auto result = lowerConnect(destVal, srcVal);
  if (failed(result))
    return failure();
  if (*result)
    return success();

  // If this connect is driving a value that is currently a backedge, record
  // that the source is the value of the backedge.
  if (updateIfBackedge(destVal, srcVal))
    return success();

  if (!isa<hw::InOutType>(destVal.getType()))
    return op.emitError("destination isn't an inout type");

  sv::AssignOp::create(builder, destVal, srcVal);
  return success();
}

LogicalResult FIRRTLLowering::visitStmt(MatchingConnectOp op) {
  auto dest = op.getDest();
  auto srcVal = getLoweredValue(op.getSrc());
  if (!srcVal)
    return handleZeroBit(op.getSrc(), []() { return success(); });

  auto destVal = getPossiblyInoutLoweredValue(dest);
  if (!destVal)
    return failure();

  auto result = lowerConnect(destVal, srcVal);
  if (failed(result))
    return failure();
  if (*result)
    return success();

  // If this connect is driving a value that is currently a backedge, record
  // that the source is the value of the backedge.
  if (updateIfBackedge(destVal, srcVal))
    return success();

  if (!isa<hw::InOutType>(destVal.getType()))
    return op.emitError("destination isn't an inout type");

  sv::AssignOp::create(builder, destVal, srcVal);
  return success();
}

LogicalResult FIRRTLLowering::visitStmt(ForceOp op) {
  auto srcVal = getLoweredValue(op.getSrc());
  if (!srcVal)
    return failure();

  auto destVal = getPossiblyInoutLoweredValue(op.getDest());
  if (!destVal)
    return failure();

  if (!isa<hw::InOutType>(destVal.getType()))
    return op.emitError("destination isn't an inout type");

  // #ifndef SYNTHESIS
  circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
  addToIfDefBlock("SYNTHESIS", std::function<void()>(), [&]() {
    addToInitialBlock([&]() { sv::ForceOp::create(builder, destVal, srcVal); });
  });
  return success();
}

LogicalResult FIRRTLLowering::visitStmt(RefForceOp op) {
  auto src = getLoweredNonClockValue(op.getSrc());
  auto clock = getLoweredNonClockValue(op.getClock());
  auto pred = getLoweredValue(op.getPredicate());
  if (!src || !clock || !pred)
    return failure();

  auto destVal = getPossiblyInoutLoweredValue(op.getDest());
  if (!destVal)
    return failure();

  // #ifndef SYNTHESIS
  circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
  addToIfDefBlock("SYNTHESIS", std::function<void()>(), [&]() {
    addToAlwaysBlock(clock, [&]() {
      addIfProceduralBlock(
          pred, [&]() { sv::ForceOp::create(builder, destVal, src); });
    });
  });
  return success();
}
LogicalResult FIRRTLLowering::visitStmt(RefForceInitialOp op) {
  auto src = getLoweredNonClockValue(op.getSrc());
  auto pred = getLoweredValue(op.getPredicate());
  if (!src || !pred)
    return failure();

  auto destVal = getPossiblyInoutLoweredValue(op.getDest());
  if (!destVal)
    return failure();

  // #ifndef SYNTHESIS
  circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
  addToIfDefBlock("SYNTHESIS", std::function<void()>(), [&]() {
    addToInitialBlock([&]() {
      addIfProceduralBlock(
          pred, [&]() { sv::ForceOp::create(builder, destVal, src); });
    });
  });
  return success();
}
LogicalResult FIRRTLLowering::visitStmt(RefReleaseOp op) {
  auto clock = getLoweredNonClockValue(op.getClock());
  auto pred = getLoweredValue(op.getPredicate());
  if (!clock || !pred)
    return failure();

  auto destVal = getPossiblyInoutLoweredValue(op.getDest());
  if (!destVal)
    return failure();

  // #ifndef SYNTHESIS
  circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
  addToIfDefBlock("SYNTHESIS", std::function<void()>(), [&]() {
    addToAlwaysBlock(clock, [&]() {
      addIfProceduralBlock(pred,
                           [&]() { sv::ReleaseOp::create(builder, destVal); });
    });
  });
  return success();
}
LogicalResult FIRRTLLowering::visitStmt(RefReleaseInitialOp op) {
  auto destVal = getPossiblyInoutLoweredValue(op.getDest());
  auto pred = getLoweredValue(op.getPredicate());
  if (!destVal || !pred)
    return failure();

  // #ifndef SYNTHESIS
  circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
  addToIfDefBlock("SYNTHESIS", std::function<void()>(), [&]() {
    addToInitialBlock([&]() {
      addIfProceduralBlock(pred,
                           [&]() { sv::ReleaseOp::create(builder, destVal); });
    });
  });
  return success();
}

// Replace FIRRTL "special" substitutions {{..}} with verilog equivalents.
static LogicalResult resolveFormatString(Location loc,
                                         StringRef originalFormatString,
                                         mlir::OperandRange operands,
                                         StringAttr &result) {
  // Update the format string to replace "special" substitutions based on
  // substitution type and lower normal substitusion.
  SmallString<32> formatString;
  for (size_t i = 0, e = originalFormatString.size(), subIdx = 0; i != e; ++i) {
    char c = originalFormatString[i];
    switch (c) {
    // Maybe a "%?" normal substitution.
    case '%': {
      formatString.push_back(c);

      // Parse the width specifier.
      SmallString<6> width;
      c = originalFormatString[++i];
      while (isdigit(c)) {
        width.push_back(c);
        c = originalFormatString[++i];
      }

      // Parse the radix.
      switch (c) {
      // A normal substitution.  If this is a radix specifier, include the width
      // if one exists.
      case 'b':
      case 'd':
      case 'x':
        if (!width.empty())
          formatString.append(width);
        [[fallthrough]];
      case 'c':
        ++subIdx;
        [[fallthrough]];
      default:
        formatString.push_back(c);
      }
      break;
    }
    // Maybe a "{{}}" special substitution.
    case '{': {
      // Not a special substituion.
      if (originalFormatString.slice(i, i + 4) != "{{}}") {
        formatString.push_back(c);
        break;
      }
      // Special substitution.  Look at the defining op to know how to lower it.
      auto substitution = operands[subIdx++];
      assert(type_isa<FStringType>(substitution.getType()) &&
             "the operand for a '{{}}' substitution must be an 'fstring' type");
      auto result =
          TypeSwitch<Operation *, LogicalResult>(substitution.getDefiningOp())
              .template Case<TimeOp>([&](auto) {
                formatString.append("%0t");
                return success();
              })
              .template Case<HierarchicalModuleNameOp>([&](auto) {
                formatString.append("%m");
                return success();
              })
              .Default([&](auto) {
                emitError(loc, "has a substitution with an unimplemented "
                               "lowering")
                        .attachNote(substitution.getLoc())
                    << "op with an unimplemented lowering is here";
                return failure();
              });
      if (failed(result))
        return failure();
      i += 3;
      break;
    }
    // Default is to let characters through.
    default:
      formatString.push_back(c);
    }
  }

  result = StringAttr::get(loc->getContext(), formatString);
  return success();
}

// Printf/FPrintf is a macro op that lowers to an sv.ifdef.procedural, an sv.if,
// and an sv.fwrite all nested together.
template <class T>
LogicalResult FIRRTLLowering::visitPrintfLike(
    T op, const FileDescriptorInfo &fileDescriptorInfo, bool usePrintfCond) {
  auto clock = getLoweredNonClockValue(op.getClock());
  auto cond = getLoweredValue(op.getCond());
  if (!clock || !cond)
    return failure();

  StringAttr formatString;
  if (failed(resolveFormatString(op.getLoc(), op.getFormatString(),
                                 op.getSubstitutions(), formatString)))
    return failure();

  auto fn = [&](Value fd) {
    SmallVector<Value> operands;
    if (failed(loweredFmtOperands(op.getSubstitutions(), operands)))
      return failure();
    sv::FWriteOp::create(builder, op.getLoc(), fd, formatString, operands);
    return success();
  };

  return lowerStatementWithFd(fileDescriptorInfo, clock, cond, fn,
                              usePrintfCond);
}

LogicalResult FIRRTLLowering::visitStmt(FPrintFOp op) {
  StringAttr outputFileAttr;
  if (failed(resolveFormatString(op.getLoc(), op.getOutputFileAttr(),
                                 op.getOutputFileSubstitutions(),
                                 outputFileAttr)))
    return failure();

  FileDescriptorInfo outputFile(outputFileAttr,
                                op.getOutputFileSubstitutions());
  return visitPrintfLike(op, outputFile, false);
}

// FFlush lowers into $fflush statement.
LogicalResult FIRRTLLowering::visitStmt(FFlushOp op) {
  auto clock = getLoweredNonClockValue(op.getClock());
  auto cond = getLoweredValue(op.getCond());
  if (!clock || !cond)
    return failure();

  auto fn = [&](Value fd) {
    sv::FFlushOp::create(builder, op.getLoc(), fd);
    return success();
  };

  if (!op.getOutputFileAttr())
    return lowerStatementWithFd({}, clock, cond, fn, false);

  // If output file is specified, resolve the format string and lower it with a
  // file descriptor associated with the output file.
  StringAttr outputFileAttr;
  if (failed(resolveFormatString(op.getLoc(), op.getOutputFileAttr(),
                                 op.getOutputFileSubstitutions(),
                                 outputFileAttr)))
    return failure();

  return lowerStatementWithFd(
      FileDescriptorInfo(outputFileAttr, op.getOutputFileSubstitutions()),
      clock, cond, fn, false);
}

// Stop lowers into a nested series of behavioral statements plus $fatal
// or $finish.
LogicalResult FIRRTLLowering::visitStmt(StopOp op) {
  auto clock = getLoweredValue(op.getClock());
  auto cond = getLoweredValue(op.getCond());
  if (!clock || !cond)
    return failure();

  circuitState.usedStopCond = true;
  circuitState.addFragment(theModule, "STOP_COND_FRAGMENT");

  Value stopCond =
      sv::MacroRefExprOp::create(builder, cond.getType(), "STOP_COND_");
  Value exitCond = builder.createOrFold<comb::AndOp>(stopCond, cond, true);

  if (op.getExitCode())
    sim::FatalOp::create(builder, clock, exitCond);
  else
    sim::FinishOp::create(builder, clock, exitCond);

  return success();
}

/// Helper function to build an immediate assert operation based on the
/// original FIRRTL operation name. This reduces code duplication in
/// `lowerVerificationStatement`.
template <typename... Args>
static Operation *buildImmediateVerifOp(ImplicitLocOpBuilder &builder,
                                        StringRef opName, Args &&...args) {
  if (opName == "assert")
    return sv::AssertOp::create(builder, std::forward<Args>(args)...);
  if (opName == "assume")
    return sv::AssumeOp::create(builder, std::forward<Args>(args)...);
  if (opName == "cover")
    return sv::CoverOp::create(builder, std::forward<Args>(args)...);
  llvm_unreachable("unknown verification op");
}

/// Helper function to build a concurrent assert operation based on the
/// original FIRRTL operation name. This reduces code duplication in
/// `lowerVerificationStatement`.
template <typename... Args>
static Operation *buildConcurrentVerifOp(ImplicitLocOpBuilder &builder,
                                         StringRef opName, Args &&...args) {
  if (opName == "assert")
    return sv::AssertConcurrentOp::create(builder, std::forward<Args>(args)...);
  if (opName == "assume")
    return sv::AssumeConcurrentOp::create(builder, std::forward<Args>(args)...);
  if (opName == "cover")
    return sv::CoverConcurrentOp::create(builder, std::forward<Args>(args)...);
  llvm_unreachable("unknown verification op");
}

/// Template for lowering verification statements from type A to
/// type B.
///
/// For example, lowering the "foo" op to the "bar" op would start
/// with:
///
///     foo(clock, condition, enable, "message")
///
/// This becomes a Verilog clocking block with the "bar" op guarded
/// by an if enable:
///
///     always @(posedge clock) begin
///       if (enable) begin
///         bar(condition);
///       end
///     end
/// The above can also be reduced into a concurrent verification statement
/// sv.assert.concurrent posedge %clock (condition && enable)
LogicalResult FIRRTLLowering::lowerVerificationStatement(
    Operation *op, StringRef labelPrefix, Value opClock, Value opPredicate,
    Value opEnable, StringAttr opMessageAttr, ValueRange opOperands,
    StringAttr opNameAttr, bool isConcurrent, EventControl opEventControl) {
  StringRef opName = op->getName().stripDialect();

  // The attribute holding the compile guards
  ArrayRef<Attribute> guards{};
  if (auto guardsAttr = op->template getAttrOfType<ArrayAttr>("guards"))
    guards = guardsAttr.getValue();

  auto isCover = isa<CoverOp>(op);
  auto clock = getLoweredNonClockValue(opClock);
  auto enable = getLoweredValue(opEnable);
  auto predicate = getLoweredValue(opPredicate);
  if (!clock || !enable || !predicate)
    return failure();

  StringAttr label;
  if (opNameAttr && !opNameAttr.getValue().empty())
    label = opNameAttr;
  StringAttr prefixedLabel;
  if (label)
    prefixedLabel =
        StringAttr::get(builder.getContext(), labelPrefix + label.getValue());

  StringAttr message;
  SmallVector<Value> messageOps;
  VerificationFlavor flavor = circuitState.verificationFlavor;

  // For non-assertion, rollback to per-op configuration.
  if (flavor == VerificationFlavor::IfElseFatal && !isa<AssertOp>(op))
    flavor = VerificationFlavor::None;

  if (flavor == VerificationFlavor::None) {
    // TODO: This should *not* be part of the op, but rather a lowering
    // option that the user of this pass can choose.

    auto format = op->getAttrOfType<StringAttr>("format");
    // if-else-fatal iff concurrent and the format is specified.
    if (isConcurrent && format && format.getValue() == "ifElseFatal") {
      if (!isa<AssertOp>(op))
        return op->emitError()
               << "ifElseFatal format cannot be used for non-assertions";
      flavor = VerificationFlavor::IfElseFatal;
    } else if (isConcurrent)
      flavor = VerificationFlavor::SVA;
    else
      flavor = VerificationFlavor::Immediate;
  }

  if (!isCover && opMessageAttr && !opMessageAttr.getValue().empty()) {
    message = opMessageAttr;
    if (failed(loweredFmtOperands(opOperands, messageOps)))
      return failure();

    if (flavor == VerificationFlavor::SVA) {
      // For SVA assert/assume statements, wrap any message ops in $sampled() to
      // guarantee that these will print with the same value as when the
      // assertion triggers.  (See SystemVerilog 2017 spec section 16.9.3 for
      // more information.)
      for (auto &loweredValue : messageOps)
        loweredValue = sv::SampledOp::create(builder, loweredValue);
    }
  }

  auto emit = [&]() {
    switch (flavor) {
    case VerificationFlavor::Immediate: {
      // Handle the purely procedural flavor of the operation.
      auto deferImmediate = circt::sv::DeferAssertAttr::get(
          builder.getContext(), circt::sv::DeferAssert::Immediate);
      addToAlwaysBlock(clock, [&]() {
        addIfProceduralBlock(enable, [&]() {
          buildImmediateVerifOp(builder, opName, predicate, deferImmediate,
                                prefixedLabel, message, messageOps);
        });
      });
      return;
    }
    case VerificationFlavor::IfElseFatal: {
      assert(isa<AssertOp>(op) && "only assert is expected");
      // Handle the `ifElseFatal` format, which does not emit an SVA but
      // rather a process that uses $error and $fatal to perform the checks.
      auto boolType = IntegerType::get(builder.getContext(), 1);
      predicate = comb::createOrFoldNot(predicate, builder, /*twoState=*/true);
      predicate = builder.createOrFold<comb::AndOp>(enable, predicate, true);

      circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
      addToIfDefBlock("SYNTHESIS", {}, [&]() {
        addToAlwaysBlock(clock, [&]() {
          addIfProceduralBlock(predicate, [&]() {
            circuitState.usedStopCond = true;
            circuitState.addFragment(theModule, "STOP_COND_FRAGMENT");

            circuitState.usedAssertVerboseCond = true;
            circuitState.addFragment(theModule, "ASSERT_VERBOSE_COND_FRAGMENT");

            addIfProceduralBlock(
                sv::MacroRefExprOp::create(builder, boolType,
                                           "ASSERT_VERBOSE_COND_"),
                [&]() { sv::ErrorOp::create(builder, message, messageOps); });
            addIfProceduralBlock(
                sv::MacroRefExprOp::create(builder, boolType, "STOP_COND_"),
                [&]() { sv::FatalOp::create(builder); });
          });
        });
      });
      return;
    }
    case VerificationFlavor::SVA: {
      // Formulate the `enable -> predicate` as `!enable | predicate`.
      // Except for covers, combine them: enable & predicate
      if (!isCover) {
        auto notEnable =
            comb::createOrFoldNot(enable, builder, /*twoState=*/true);
        predicate =
            builder.createOrFold<comb::OrOp>(notEnable, predicate, true);
      } else {
        predicate = builder.createOrFold<comb::AndOp>(enable, predicate, true);
      }

      // Handle the regular SVA case.
      sv::EventControl event;
      switch (opEventControl) {
      case EventControl::AtPosEdge:
        event = circt::sv::EventControl::AtPosEdge;
        break;
      case EventControl::AtEdge:
        event = circt::sv::EventControl::AtEdge;
        break;
      case EventControl::AtNegEdge:
        event = circt::sv::EventControl::AtNegEdge;
        break;
      }

      buildConcurrentVerifOp(
          builder, opName,
          circt::sv::EventControlAttr::get(builder.getContext(), event), clock,
          predicate, prefixedLabel, message, messageOps);
      return;
    }
    case VerificationFlavor::None:
      llvm_unreachable(
          "flavor `None` must be converted into one of concreate flavors");
    }
  };

  // Wrap the verification statement up in the optional preprocessor
  // guards. This is a bit awkward since we want to translate an array of
  // guards  into a recursive call to `addToIfDefBlock`.
  return emitGuards(op->getLoc(), guards, emit);
}

// Lower an assert to SystemVerilog.
LogicalResult FIRRTLLowering::visitStmt(AssertOp op) {
  return lowerVerificationStatement(
      op, "assert__", op.getClock(), op.getPredicate(), op.getEnable(),
      op.getMessageAttr(), op.getSubstitutions(), op.getNameAttr(),
      op.getIsConcurrent(), op.getEventControl());
}

// Lower an assume to SystemVerilog.
LogicalResult FIRRTLLowering::visitStmt(AssumeOp op) {
  return lowerVerificationStatement(
      op, "assume__", op.getClock(), op.getPredicate(), op.getEnable(),
      op.getMessageAttr(), op.getSubstitutions(), op.getNameAttr(),
      op.getIsConcurrent(), op.getEventControl());
}

// Lower a cover to SystemVerilog.
LogicalResult FIRRTLLowering::visitStmt(CoverOp op) {
  return lowerVerificationStatement(
      op, "cover__", op.getClock(), op.getPredicate(), op.getEnable(),
      op.getMessageAttr(), op.getSubstitutions(), op.getNameAttr(),
      op.getIsConcurrent(), op.getEventControl());
}

// Lower an UNR only assume to a specific style of SV assume.
LogicalResult FIRRTLLowering::visitStmt(UnclockedAssumeIntrinsicOp op) {
  // TODO : Need to figure out if there is a cleaner way to get the string which
  // indicates the assert is UNR only. Or better - not rely on this at all -
  // ideally there should have been some other attribute which indicated that
  // this assert for UNR only.
  auto guardsAttr = op->getAttrOfType<mlir::ArrayAttr>("guards");
  ArrayRef<Attribute> guards =
      guardsAttr ? guardsAttr.getValue() : ArrayRef<Attribute>();

  auto label = op.getNameAttr();
  StringAttr assumeLabel;
  if (label && !label.empty())
    assumeLabel =
        StringAttr::get(builder.getContext(), "assume__" + label.getValue());
  auto predicate = getLoweredValue(op.getPredicate());
  auto enable = getLoweredValue(op.getEnable());
  auto notEnable = comb::createOrFoldNot(enable, builder, /*twoState=*/true);
  predicate = builder.createOrFold<comb::OrOp>(notEnable, predicate, true);

  SmallVector<Value> messageOps;
  for (auto operand : op.getSubstitutions()) {
    auto loweredValue = getLoweredValue(operand);
    if (!loweredValue) {
      // If this is a zero bit operand, just pass a one bit zero.
      if (!isZeroBitFIRRTLType(operand.getType()))
        return failure();
      loweredValue = getOrCreateIntConstant(1, 0);
    }
    messageOps.push_back(loweredValue);
  }
  return emitGuards(op.getLoc(), guards, [&]() {
    sv::AlwaysOp::create(
        builder, ArrayRef(sv::EventControl::AtEdge), ArrayRef(predicate),
        [&]() {
          if (op.getMessageAttr().getValue().empty())
            buildImmediateVerifOp(
                builder, "assume", predicate,
                circt::sv::DeferAssertAttr::get(
                    builder.getContext(), circt::sv::DeferAssert::Immediate),
                assumeLabel);
          else
            buildImmediateVerifOp(
                builder, "assume", predicate,
                circt::sv::DeferAssertAttr::get(
                    builder.getContext(), circt::sv::DeferAssert::Immediate),
                assumeLabel, op.getMessageAttr(), messageOps);
        });
  });
}

LogicalResult FIRRTLLowering::visitStmt(AttachOp op) {
  // Don't emit anything for a zero or one operand attach.
  if (op.getAttached().size() < 2)
    return success();

  SmallVector<Value, 4> inoutValues;
  for (auto v : op.getAttached()) {
    inoutValues.push_back(getPossiblyInoutLoweredValue(v));
    if (!inoutValues.back()) {
      // Ignore zero bit values.
      if (!isZeroBitFIRRTLType(v.getType()))
        return failure();
      inoutValues.pop_back();
      continue;
    }

    if (!isa<hw::InOutType>(inoutValues.back().getType()))
      return op.emitError("operand isn't an inout type");
  }

  if (inoutValues.size() < 2)
    return success();

  // If the op has a single source value, the value is used as a lowering result
  // of other values. Therefore we can delete the attach op here.
  if (getSingleNonInstanceOperand(op))
    return success();

  // If all operands of the attach are internal to this module (none of them
  // are ports), then they can all be replaced with a single wire, and we can
  // delete the attach op.
  bool isAttachInternalOnly =
      llvm::none_of(inoutValues, [](auto v) { return isa<BlockArgument>(v); });

  if (isAttachInternalOnly) {
    auto v0 = inoutValues.front();
    for (auto v : inoutValues) {
      if (v == v0)
        continue;
      v.replaceAllUsesWith(v0);
    }
    return success();
  }

  // If the attach operands contain a port, then we can't do anything to
  // simplify the attach operation.
  circuitState.addMacroDecl(builder.getStringAttr("SYNTHESIS"));
  circuitState.addMacroDecl(builder.getStringAttr("VERILATOR"));
  addToIfDefBlock(
      "SYNTHESIS",
      // If we're doing synthesis, we emit an all-pairs assign complex.
      [&]() {
        SmallVector<Value, 4> values;
        for (auto inoutValue : inoutValues)
          values.push_back(getReadValue(inoutValue));

        for (size_t i1 = 0, e = inoutValues.size(); i1 != e; ++i1) {
          for (size_t i2 = 0; i2 != e; ++i2)
            if (i1 != i2)
              sv::AssignOp::create(builder, inoutValues[i1], values[i2]);
        }
      },
      // In the non-synthesis case, we emit a SystemVerilog alias
      // statement.
      [&]() {
        sv::IfDefOp::create(
            builder, "VERILATOR",
            [&]() {
              sv::VerbatimOp::create(
                  builder,
                  "`error \"Verilator does not support alias and thus "
                  "cannot "
                  "arbitrarily connect bidirectional wires and ports\"");
            },
            [&]() { sv::AliasOp::create(builder, inoutValues); });
      });

  return success();
}

LogicalResult FIRRTLLowering::visitStmt(BindOp op) {
  sv::BindOp::create(builder, op.getInstanceAttr());
  return success();
}

LogicalResult FIRRTLLowering::fixupLTLOps() {
  if (ltlOpFixupWorklist.empty())
    return success();
  LLVM_DEBUG(llvm::dbgs() << "Fixing up " << ltlOpFixupWorklist.size()
                          << " LTL ops\n");

  // Add wire users into the worklist.
  for (unsigned i = 0, e = ltlOpFixupWorklist.size(); i != e; ++i)
    for (auto *user : ltlOpFixupWorklist[i]->getUsers())
      if (isa<hw::WireOp>(user))
        ltlOpFixupWorklist.insert(user);

  // Re-infer LTL op types and remove wires.
  while (!ltlOpFixupWorklist.empty()) {
    auto *op = ltlOpFixupWorklist.pop_back_val();

    // Update the operation's return type by re-running type inference.
    if (auto opIntf = dyn_cast_or_null<mlir::InferTypeOpInterface>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "- Update " << *op << "\n");
      SmallVector<Type, 2> types;
      auto result = opIntf.inferReturnTypes(
          op->getContext(), op->getLoc(), op->getOperands(),
          op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
          types);
      if (failed(result))
        return failure();
      assert(types.size() == op->getNumResults());

      // Update the result types and add the dependent ops into the worklist if
      // the type changed.
      for (auto [result, type] : llvm::zip(op->getResults(), types)) {
        if (result.getType() == type)
          continue;
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Result #" << result.getResultNumber() << " from "
                   << result.getType() << " to " << type << "\n");
        result.setType(type);
        for (auto *user : result.getUsers())
          if (user != op)
            ltlOpFixupWorklist.insert(user);
      }
    }

    // Remove LTL-typed wires.
    if (auto wireOp = dyn_cast<hw::WireOp>(op)) {
      if (isa<ltl::SequenceType, ltl::PropertyType>(wireOp.getType())) {
        wireOp.replaceAllUsesWith(wireOp.getInput());
        LLVM_DEBUG(llvm::dbgs() << "- Remove " << wireOp << "\n");
        if (wireOp.use_empty())
          wireOp.erase();
      }
      continue;
    }

    // Ensure that the operation has no users outside of LTL operations.
    SmallPtrSet<Operation *, 4> usersReported;
    for (auto *user : op->getUsers()) {
      if (!usersReported.insert(user).second)
        continue;
      if (isa<ltl::LTLDialect, verif::VerifDialect>(user->getDialect()))
        continue;
      if (isa<hw::WireOp>(user))
        continue;
      auto d = op->emitError(
          "verification operation used in a non-verification context");
      d.attachNote(user->getLoc())
          << "leaking outside verification context here";
      return d;
    }
  }

  return success();
}
