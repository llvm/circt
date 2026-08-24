//===- ClockSpecJSONHandler.cpp - Generate Clock Spec JSON from domains ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A handler for generating "Clock Spec JSON" format from domain information
// contained in a final MLIR blob (likely compiled with `firtool`).
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "Handler.h"

using namespace llvm;

namespace {
namespace options {

cl::OptionCategory cat{"SiFive Clock Spec JSON Options"};
cl::list<std::string> sifiveClockDomainAsync{
    "sifive-clock-domain-async",
    cl::desc("indicate that a specific clock domain should be treated as "
             "'async' indicating that any of its associations should be "
             "put under the 'asynchronous_ports' grouping"),
    cl::Prefix, cl::cat(cat)};
cl::list<std::string> sifiveClockDomainStatic{
    "sifive-clock-domain-static",
    cl::desc("indicate that a specific clock domain should be treated as "
             "'static' indicating that any of its associations should be "
             "put under the 'static_ports' grouping"),
    cl::Prefix, cl::cat(cat)};

} // namespace options
} // namespace

namespace circt {
namespace handlers {

/// The kind of relationship between two clock domains.
enum class RelationshipKind {
  /// A synchronous relationship (integer frequency ratio, same source).
  Synchronous,
  /// A rational relationship (non-integer rational frequency ratio, same
  /// source).
  Rational,
  /// An asynchronous relationship (no deterministic phase relationship).
  Async,
  /// Relationship is not yet determined.
  Inferred
};

struct Relationship {
  StringAttr namePattern;
  RelationshipKind relationship;
};

struct Clock {
  StringAttr namePattern;
  SmallVector<Relationship> relationships;
  /// Paths to clock-gate cells accumulated in the domain's clockGates registry.
  SmallVector<StringAttr> clockGates;
};

struct SynchronousData {
  StringAttr namePattern;
  SmallVector<StringAttr> portPatterns;
  std::optional<StringAttr> comment;
};

/// A handler that generates Clock Spec JSON output from Clock Domain
/// information.  This is an internal format that is an input to further tooling
/// that generates Synopsys Design Constraint (SDC) files.
///
/// Note: this is intended to be replaced by a handler which directly generates
/// SDC files.
class ClockSpecJSON : public Handler {

public:
  bool shouldHandle(Type type) override {
    auto classType = dyn_cast<om::ClassType>(type);
    return classType && classType.getClassName().getValue() == "ClockDomain";
  }

  /// Extract path refs from an evaluator path list, reporting type errors.
  /// Associations use getRef() (leaf names).  Registry assets use
  /// getAsString() so hierarchical basepaths are preserved.
  static LogicalResult
  collectPathRefs(ArrayRef<om::evaluator::EvaluatorValuePtr> values,
                  SmallVectorImpl<StringAttr> &out, bool &failed,
                  bool hierarchical = false) {
    for (auto &value : values) {
      if (auto *p = dyn_cast<om::evaluator::PathValue>(value.get())) {
        out.push_back(hierarchical ? p->getAsString() : p->getRef());
        continue;
      }
      emitError(value->getLoc())
          << "expected path, but got " << value->getType();
      failed = true;
    }
    return success(failed == false);
  }

  LogicalResult handle(const ObjectMap &objectMap) override {
    // For a variety of reasons, this could fail.  Track failure, reporting all
    // errors before finally exiting.
    bool failed = false;
    for (auto &[objectValue, lists] : objectMap) {
      auto &associations = lists.associations;
      auto name = cast<StringAttr>(cast<om::evaluator::AttributeValue>(
                                       objectValue->getField("name_out")->get())
                                       ->getAttr());

      // Insert this domain into its own equivalence class.  Then, if this
      // domain specifies a "source" relationship (either synchronous or
      // rational), merge it into the same equivalence class as its source.
      syncEquivalenceClasses.insert(name);

      auto source =
          cast<StringAttr>(cast<om::evaluator::AttributeValue>(
                               objectValue->getField("source_out")->get())
                               ->getAttr());
      auto relationship =
          cast<StringAttr>(cast<om::evaluator::AttributeValue>(
                               objectValue->getField("relationship_out")->get())
                               ->getAttr());

      // Track the relationship kind for this domain.
      RelationshipKind kind = RelationshipKind::Inferred;
      if (relationship.getValue() == "synchronous")
        kind = RelationshipKind::Synchronous;
      else if (relationship.getValue() == "rational")
        kind = RelationshipKind::Rational;

      // If this domain specifies a source, merge it into that source's
      // equivalence class and record the relationship.
      if (!source.getValue().empty()) {
        syncEquivalenceClasses.insert(source);
        syncEquivalenceClasses.unionSets(source, name);
        domainRelationships[name] = {source, kind};
      }

      // Collect optional clockGates registry paths for this domain.
      // Use hierarchical stringification so instance prefixes are retained.
      SmallVector<StringAttr> clockGatePaths;
      if (auto it = lists.registries.find(
              StringAttr::get(name.getContext(), "clockGates"));
          it != lists.registries.end()) {
        bool pathFailed = false;
        (void)collectPathRefs(it->second, clockGatePaths, pathFailed,
                              /*hierarchical=*/true);
        failed |= pathFailed;
      }

      // Add to async ports if the name matches a provided option.
      bool isAsync =
          llvm::any_of(options::sifiveClockDomainAsync, [&](auto asyncName) {
            return asyncName == name.getValue();
          });
      if (isAsync) {
        bool pathFailed = false;
        (void)collectPathRefs(associations, asyncPorts, pathFailed);
        failed |= pathFailed;
        continue;
      }

      // Add to static ports if the name matches a provided option.
      bool isStatic =
          llvm::any_of(options::sifiveClockDomainStatic, [&](auto staticName) {
            return staticName == name.getValue();
          });
      if (isStatic) {
        bool pathFailed = false;
        (void)collectPathRefs(associations, staticPorts, pathFailed);
        failed |= pathFailed;
        continue;
      }

      // Otherwise, this is a normal clock association.  Add the clock and
      // populate the associations and clock gates.
      clocks.push_back({/*namePattern=*/name,
                        /*relationships=*/{},
                        /*clockGates=*/std::move(clockGatePaths)});

      for (auto &association : associations) {
        if (auto *p = dyn_cast<om::evaluator::PathValue>(association.get())) {
          // TODO: Add checks that path is empty.
          syncPorts.try_emplace(name, SynchronousData{name, {}, {}})
              .first->second.portPatterns.push_back(p->getRef());
          continue;
        }
        emitError(association->getLoc())
            << "expected associations to be a path, but got "
            << association->getType();
        failed = true;
      }
    }

    if (failed)
      return failure();

    // Populate clock relationships based on equivalence class leaders
    populateClockRelationships();

    return success();
  }

  LogicalResult emit(raw_ostream &os) override {
    json::OStream json(os, /*indentSize=*/2);
    json.object([&] {
      json.attributeArray("clocks", [&] {
        for (auto &clock : clocks) {
          json.object([&] {
            auto name = clock.namePattern.getValue();
            json.attribute("name_pattern", name);
            json.attribute("define_period",
                           (Twine(name.upper()) + "_PERIOD").str());
            json.attributeArray("clock_relationships", [&] {
              for (auto &rel : clock.relationships) {
                json.object([&] {
                  json.attribute("name_pattern", rel.namePattern.getValue());
                  // Both synchronous and rational are treated as "sync" in the
                  // output JSON.  They both imply a deterministic, bounded
                  // frequency relationship derived from a common source.
                  json.attribute("relationship", "sync");
                });
              }
            });
            json.attributeArray("clock_gates", [&] {
              for (auto gate : clock.clockGates)
                json.value(gate.getValue());
            });
          });
        }
      });
      json.attributeArray("static_ports", [&] {
        for (auto port : staticPorts)
          json.value(port.getValue());
      });
      json.attributeArray("asynchronous_ports", [&] {
        for (auto port : asyncPorts)
          json.value(port.getValue());
      });
      json.attributeArray("synchronous_ports", [&] {
        for (auto &[_, syncPort] : syncPorts) {
          auto name = syncPort.namePattern.getValue();
          auto &ports = syncPort.portPatterns;
          auto &comment = syncPort.comment;
          json.object([&] {
            json.attribute("name_pattern", name);
            json.attributeArray("port_patterns", [&] {
              for (auto port : ports)
                json.value(port.getValue());
            });
            json.attribute("comment", comment);
          });
        }
      });
    });

    return success();
  }

  void clear() override {
    clocks.clear();
    asyncPorts.clear();
    staticPorts.clear();
    syncPorts.clear();
    domainRelationships.clear();
    syncEquivalenceClasses = EquivalenceClasses<StringAttr>();
  };

private:
  /// Populate clock relationships based on equivalence class leaders.  If a
  /// domain's leader is not itself, it has a sync (or rational, treated the
  /// same in output) relationship to the leader.
  void populateClockRelationships() {
    for (auto &clock : clocks) {
      auto &name = clock.namePattern;

      // Find the leader of this domain's equivalence class
      auto leaderIter = syncEquivalenceClasses.findLeader(name);
      if (leaderIter == syncEquivalenceClasses.member_end())
        continue;

      StringAttr leader = *leaderIter;

      // If this domain is not the leader, add a relationship to the leader.
      // Look up the recorded relationship kind for this domain and fall back to
      // Synchronous if no explicit entry exists.
      if (name != leader) {
        RelationshipKind kind = RelationshipKind::Synchronous;
        auto it = domainRelationships.find(name);
        if (it != domainRelationships.end())
          kind = it->second.relationship;
        clock.relationships.push_back({leader, kind});
      }
    }
  }

  SmallVector<Clock> clocks;
  SmallVector<StringAttr> asyncPorts;
  SmallVector<StringAttr> staticPorts;
  MapVector<StringAttr, SynchronousData> syncPorts;

  /// Map from a domain name to its recorded source relationship (source name +
  /// kind).  Populated during handle() when a domain declares a non-empty
  /// source.
  DenseMap<StringAttr, Relationship> domainRelationships;

  // Equivalence classes tracking all domains that are related to each other
  // (synchronous or rational) through the transitive closure of source
  // relationships.  The leader of each equivalence class represents the root
  // domain.
  EquivalenceClasses<StringAttr> syncEquivalenceClasses;
};

static bool registeredClockSpecJSONHandler = [] {
  HandlerRegistry::get().registerHandler(std::make_unique<ClockSpecJSON>());
  return true;
}();

} // namespace handlers
} // namespace circt
