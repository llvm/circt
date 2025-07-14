#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Regex.h"
#include <numeric>

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWEXPUNGEMODULE
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

namespace {
struct HWExpungeModulePass
    : circt::hw::impl::HWExpungeModuleBase<HWExpungeModulePass> {
  void runOnOperation() override;
};

struct InstPathSeg {
  llvm::StringRef seg;

  InstPathSeg(llvm::StringRef seg) : seg(seg) {}
  const llvm::StringRef &getSeg() const { return seg; }
  operator llvm::StringRef() const { return seg; }

  void Profile(llvm::FoldingSetNodeID &ID) const { ID.AddString(seg); }
};
using InstPath = llvm::ImmutableList<InstPathSeg>;
std::string defaultPrefix(InstPath path) {
  std::string accum;
  while (!path.isEmpty()) {
    accum += path.getHead().getSeg();
    accum += "_";
    path = path.getTail();
  }
  accum += "_";
  return std::move(accum);
}

// The regex for port prefix specification
// "^([@#a-zA-Z0-9_]+):([a-zA-Z0-9_]+)(\\.[a-zA-Z0-9_]+)*=([a-zA-Z0-9_]+)$"
// Unfortunately, the LLVM Regex cannot capture repeating capture groups, so
// manually parse the spec This parser may accept identifiers with invalid
// characters

std::variant<std::tuple<llvm::StringRef, InstPath, llvm::StringRef>,
             std::string>
parsePrefixSpec(llvm::StringRef in, InstPath::Factory &listFac) {
  auto [l, r] = in.split("=");
  if (r == "")
    return "No '=' found in input";
  auto [ll, lr] = l.split(":");
  if (lr == "")
    return "No ':' found before '='";
  llvm::SmallVector<llvm::StringRef, 4> segs;
  while (lr != "") {
    auto [seg, rest] = lr.split(".");
    segs.push_back(seg);
    lr = rest;
  }
  InstPath path;
  for (auto &seg : llvm::reverse(segs))
    path = listFac.add(seg, path);
  return std::make_tuple(ll, path, r);
}
} // namespace

void HWExpungeModulePass::runOnOperation() {
  auto root = getOperation();
  llvm::DenseMap<mlir::StringRef, circt::hw::HWModuleLike> allModules;
  root.walk(
      [&](circt::hw::HWModuleLike mod) { allModules[mod.getName()] = mod; });

  // The instance graph. We only use this graph to traverse the hierarchy in
  // post order. The order does not change throught out the operation, onlygets
  // weakened, but still valid. So we keep this cached instance graph throughout
  // the pass.
  auto &instanceGraph = getAnalysis<circt::hw::InstanceGraph>();

  // Instance path.
  InstPath::Factory pathFactory;

  // Process port prefix specifications
  // (Module name, Instance path) -> Prefix
  llvm::DenseMap<std::pair<mlir::StringRef, InstPath>, mlir::StringRef>
      designatedPrefixes;
  bool containsFailure = false;
  for (const auto &raw : portPrefixes) {
    auto matched = parsePrefixSpec(raw, pathFactory);
    if (std::holds_alternative<std::string>(matched)) {
      llvm::errs() << "Invalid port prefix specification: " << raw << "\n";
      llvm::errs() << "Error: " << std::get<std::string>(matched) << "\n";
      containsFailure = true;
      continue;
    }

    auto [module, path, prefix] =
        std::get<std::tuple<llvm::StringRef, InstPath, llvm::StringRef>>(
            matched);
    if (!allModules.contains(module)) {
      llvm::errs() << "Module not found in port prefix specification: "
                   << module << "\n";
      llvm::errs() << "From specification: " << raw << "\n";
      containsFailure = true;
      continue;
    }

    // Skip checking instance paths' existence. Non-existent paths are ignored
    designatedPrefixes.insert({{module, path}, prefix});
  }

  if (containsFailure)
    return signalPassFailure();

  // Instance path * prefix name
  using ReplacedDescendent = std::pair<InstPath, std::string>;
  // This map holds the expunged descendents of a module
  llvm::DenseMap<llvm::StringRef, llvm::SmallVector<ReplacedDescendent>>
      expungedDescendents;
  for (auto &expunging : this->modules) {
    // Clear expungedDescendents
    for (auto &it : expungedDescendents)
      it.getSecond().clear();

    auto expungingMod = allModules.lookup(expunging);
    if (!expungingMod)
      continue; // Ignored missing modules
    auto expungingModTy = expungingMod.getHWModuleType();
    auto expungingModPorts = expungingModTy.getPorts();

    auto createPortsOn = [&expungingModPorts](circt::hw::HWModuleOp mod,
                                              const std::string &prefix,
                                              auto genOutput, auto emitInput) {
      mlir::OpBuilder builder(mod);
      // Create ports using *REVERSE* direction of their definitions
      for (auto &port : expungingModPorts) {
        auto defaultName = prefix + port.name.getValue();
        auto finalName = defaultName;
        if (port.dir == circt::hw::PortInfo::Input) {
          auto val = genOutput(port);
          assert(val.getType() == port.type);
          mod.appendOutput(finalName, val);
        } else if (port.dir == circt::hw::PortInfo::Output) {
          auto [_, arg] = mod.appendInput(finalName, port.type);
          emitInput(port, arg);
        }
      }
    };

    for (auto &instGraphNode : llvm::post_order(&instanceGraph)) {
      // Skip extmodule and intmodule because they cannot contain anything
      circt::hw::HWModuleOp processing =
          llvm::dyn_cast_if_present<circt::hw::HWModuleOp>(
              instGraphNode->getModule().getOperation());
      if (!processing)
        continue;

      std::optional<decltype(expungedDescendents.lookup("")) *>
          outerExpDescHold = {};
      auto getOuterExpDesc = [&]() -> decltype(**outerExpDescHold) {
        if (!outerExpDescHold.has_value())
          outerExpDescHold = {
              &expungedDescendents.insert({processing.getName(), {}})
                   .first->getSecond()};
        return **outerExpDescHold;
      };

      mlir::OpBuilder outerBuilder(processing);

      processing.walk([&](circt::hw::InstanceOp inst) {
        auto instName = inst.getInstanceName();
        auto instMod = allModules.lookup(inst.getModuleName());

        if (instMod.getOutputNames().size() != inst.getResults().size() ||
            instMod.getNumInputPorts() != inst.getInputs().size()) {
          // Module have been modified during this pass, create new instances
          assert(instMod.getNumOutputPorts() >= inst.getResults().size());
          assert(instMod.getNumInputPorts() >= inst.getInputs().size());

          auto instModInTypes = instMod.getInputTypes();

          llvm::SmallVector<mlir::Value> newInputs;
          newInputs.reserve(instMod.getNumInputPorts());

          outerBuilder.setInsertionPointAfter(inst);

          // Appended inputs are at the end of the input list
          for (size_t i = 0; i < instMod.getNumInputPorts(); ++i) {
            mlir::Value input;
            if (i < inst.getNumInputPorts()) {
              input = inst.getInputs()[i];
              if (auto existingName = inst.getInputName(i))
                assert(existingName == instMod.getInputName(i));
            } else {
              input =
                  outerBuilder
                      .create<mlir::UnrealizedConversionCastOp>(
                          inst.getLoc(), instModInTypes[i], mlir::ValueRange{})
                      .getResult(0);
            }
            newInputs.push_back(input);
          }

          auto newInst = outerBuilder.create<circt::hw::InstanceOp>(
              inst.getLoc(), instMod, inst.getInstanceNameAttr(), newInputs,
              inst.getParameters(),
              inst.getInnerSym().value_or<circt::hw::InnerSymAttr>({}));

          for (size_t i = 0; i < inst.getNumResults(); ++i)
            assert(inst.getOutputName(i) == instMod.getOutputName(i));
          inst.replaceAllUsesWith(
              newInst.getResults().slice(0, inst.getNumResults()));
          inst.erase();
          inst = newInst;
        }

        llvm::StringMap<mlir::Value> instOMap;
        llvm::StringMap<mlir::Value> instIMap;
        assert(instMod.getOutputNames().size() == inst.getResults().size());
        for (auto [oname, oval] :
             llvm::zip(instMod.getOutputNames(), inst.getResults()))
          instOMap[llvm::cast<mlir::StringAttr>(oname).getValue()] = oval;
        assert(instMod.getInputNames().size() == inst.getInputs().size());
        for (auto [iname, ival] :
             llvm::zip(instMod.getInputNames(), inst.getInputs()))
          instIMap[llvm::cast<mlir::StringAttr>(iname).getValue()] = ival;

        // Get outer expunged descendent first because it may modify the map and
        // invalidate iterators.
        auto &outerExpDesc = getOuterExpDesc();
        auto instExpDesc = expungedDescendents.find(inst.getModuleName());

        if (inst.getModuleName() == expunging) {
          // Handle the directly expunged module
          // input maps also useful for directly expunged instance

          auto singletonPath = pathFactory.create(instName);

          auto designatedPrefix =
              designatedPrefixes.find({processing.getName(), singletonPath});
          std::string prefix = designatedPrefix != designatedPrefixes.end()
                                   ? designatedPrefix->getSecond().str()
                                   : (instName + "__").str();

          // Port name collision is still possible, but current relying on MLIR
          // to automatically rename input arguments.
          // TODO: name collision detect

          createPortsOn(
              processing, prefix,
              [&](circt::hw::ModulePort port) {
                // Generate output for outer module, so input for us
                return instIMap.at(port.name);
              },
              [&](circt::hw::ModulePort port, mlir::Value val) {
                // Generated input for outer module, replace inst results
                assert(instOMap.contains(port.name));
                instOMap[port.name].replaceAllUsesWith(val);
              });

          outerExpDesc.emplace_back(singletonPath, prefix);

          assert(instExpDesc == expungedDescendents.end() ||
                 instExpDesc->getSecond().size() == 0);
          inst.erase();
        } else if (instExpDesc != expungedDescendents.end()) {
          // Handle all transitive descendents
          if (instExpDesc->second.size() == 0)
            return;
          llvm::DenseMap<llvm::StringRef, mlir::Value> newInputs;
          for (const auto &exp : instExpDesc->second) {
            auto newPath = pathFactory.add(instName, exp.first);
            auto designatedPrefix =
                designatedPrefixes.find({processing.getName(), newPath});
            std::string prefix = designatedPrefix != designatedPrefixes.end()
                                     ? designatedPrefix->getSecond().str()
                                     : defaultPrefix(newPath);

            // TODO: name collision detect

            createPortsOn(
                processing, prefix,
                [&](circt::hw::ModulePort port) {
                  // Generate output for outer module, directly forward from
                  // inner inst
                  return instOMap.at((exp.second + port.name.getValue()).str());
                },
                [&](circt::hw::ModulePort port, mlir::Value val) {
                  // Generated input for outer module, replace inst results.
                  // The operand in question has to be an backedge
                  auto in =
                      instIMap.at((exp.second + port.name.getValue()).str());
                  auto inDef = in.getDefiningOp();
                  assert(llvm::isa<mlir::UnrealizedConversionCastOp>(inDef));
                  in.replaceAllUsesWith(val);
                  inDef->erase();
                });

            outerExpDesc.emplace_back(newPath, prefix);
          }
        }
      });
    }
  }
}

std::unique_ptr<mlir::Pass> circt::hw::createHWExpungeModulePass() {
  return std::make_unique<HWExpungeModulePass>();
}