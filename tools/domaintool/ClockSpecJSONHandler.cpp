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

enum class RelationshipKind { Sync, Async, Inferred };

struct Relationship {
  StringAttr namePattern;
  RelationshipKind relationship;
};

struct Clock {
  StringAttr namePattern;
  SmallVector<Relationship> relationships;
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

  LogicalResult handle(const ObjectMap &objectMap) override {
    // For a variety of reasons, this could fail.  Track failure, reporting all
    // errors before finally exiting.
    bool failed = false;
    for (auto &[objectValue, associations] : objectMap) {
      auto name = cast<StringAttr>(cast<om::evaluator::AttributeValue>(
                                       objectValue->getField("name_out")->get())
                                       ->getAttr());

      // Insert this domain into its own equivalence class.  Then, if this has a
      // specified "synchronousTo" relationship to a different domain, merge
      // this domain into that different domain.
      syncEquivalenceClasses.insert(name);
      auto synchronousTo = cast<StringAttr>(
          cast<om::evaluator::AttributeValue>(
              objectValue->getField("synchronousTo_out")->get())
              ->getAttr());
      if (!synchronousTo.getValue().empty()) {
        syncEquivalenceClasses.insert(synchronousTo);
        syncEquivalenceClasses.unionSets(synchronousTo, name);
      }

      // Add to async ports if the name matches a provided option.
      bool isAsync =
          llvm::any_of(options::sifiveClockDomainAsync, [&](auto asyncName) {
            return asyncName == name.getValue();
          });
      if (isAsync) {
        for (auto &association : associations) {
          if (auto *p = dyn_cast<om::evaluator::PathValue>(association.get())) {
            // TODO: Add checks that path is empty.
            asyncPorts.push_back(p->getRef());
            continue;
          }
          emitError(association->getLoc())
              << "expected associations to be a path, but got "
              << association->getType();
          failed = true;
        }
        continue;
      }

      // Add to static ports if the name matches a provided option.
      bool isStatic =
          llvm::any_of(options::sifiveClockDomainStatic, [&](auto staticName) {
            return staticName == name.getValue();
          });
      if (isStatic) {
        for (auto &association : associations) {
          if (auto *p = dyn_cast<om::evaluator::PathValue>(association.get())) {
            // TODO: Add checks that path is empty.
            staticPorts.push_back(p->getRef());
            continue;
          }
          emitError(association->getLoc())
              << "expected associations to be a path, but got "
              << association->getType();
          failed = true;
        }
        continue;
      }

      // Otherwise, this is a normal clock association.  Add the clock and
      // populate the associations.
      clocks.push_back({/*namePattern=*/name,
                        /*relationships=*/{}});

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
                  json.attribute("relationship", "sync");
                });
              }
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
    syncEquivalenceClasses = EquivalenceClasses<StringAttr>();
  };

private:
  /// Populate clock relationships based on equivalence class leaders.  If a
  /// domain's leader is not itself, it has a sync relationship to the leader.
  void populateClockRelationships() {
    for (auto &clock : clocks) {
      auto &name = clock.namePattern;

      // Find the leader of this domain's equivalence class
      auto leaderIter = syncEquivalenceClasses.findLeader(name);
      if (leaderIter == syncEquivalenceClasses.member_end())
        continue;

      StringAttr leader = *leaderIter;

      // If this domain is not the leader, add a sync relationship to the leader
      if (name != leader)
        clock.relationships.push_back({leader, RelationshipKind::Sync});
    }
  }

  SmallVector<Clock> clocks;
  SmallVector<StringAttr> asyncPorts;
  SmallVector<StringAttr> staticPorts;
  MapVector<StringAttr, SynchronousData> syncPorts;

  // Equivalence classes tracking all domains that are synchronous to each other
  // through the transitive closure of synchronousTo relationships.  The leader
  // of each equivalence class represents the root synchronous domain.
  EquivalenceClasses<StringAttr> syncEquivalenceClasses;
};

static bool registeredClockSpecJSONHandler = [] {
  HandlerRegistry::get().registerHandler(std::make_unique<ClockSpecJSON>());
  return true;
}();

} // namespace handlers
} // namespace circt
