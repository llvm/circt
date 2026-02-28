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
  StringRef namePattern;
  RelationshipKind relationship;
};

struct Clock {
  StringRef namePattern;
  uint64_t definePeriod;
  SmallVector<Relationship> relationships;
};

struct SynchronousData {
  StringRef namePattern;
  SmallVector<StringRef> portPatterns;
  std::optional<StringRef> comment;
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
                                       ->getAttr())
                      .getValue();
      // Add to async ports if the name matches a provided option.
      bool isAsync =
          llvm::any_of(options::sifiveClockDomainAsync,
                       [&](auto asyncName) { return asyncName == name; });
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
          llvm::any_of(options::sifiveClockDomainStatic,
                       [&](auto staticName) { return staticName == name; });
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
      clocks.push_back(
          {/*namePattern=*/name,
           /*define_period=*/
           cast<om::IntegerAttr>(cast<om::evaluator::AttributeValue>(
                                     objectValue->getField("period_out")->get())
                                     ->getAttr())
               .getValue()
               .getValue()
               .getZExtValue(),
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

    return success();
  }

  LogicalResult emit(raw_ostream &os) override {
    json::OStream json(os, /*indentSize=*/2);
    json.object([&] {
      json.attributeArray("clocks", [&] {
        for (auto clock : clocks) {
          json.object([&] {
            json.attribute("name_pattern", clock.namePattern);
            json.attribute("define_period", clock.definePeriod);
            json.attributeArray("clock_relationships", [&] {
              // TODO: Implement this.
            });
          });
        }
      });
      json.attributeArray("static_ports", [&] {
        for (auto port : staticPorts)
          json.value(port);
      });
      json.attributeArray("asynchronous_ports", [&] {
        for (auto port : asyncPorts)
          json.value(port);
      });
      json.attributeArray("synchronous_ports", [&] {
        for (auto &[_, syncPort] : syncPorts) {
          auto &name = syncPort.namePattern;
          auto &ports = syncPort.portPatterns;
          auto &comment = syncPort.comment;
          json.object([&] {
            json.attribute("name_pattern", name);
            json.attributeArray("port_patterns", [&] {
              for (auto port : ports)
                json.value(port);
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
  };

private:
  SmallVector<Clock> clocks;
  SmallVector<StringRef> asyncPorts;
  SmallVector<StringRef> staticPorts;
  MapVector<StringRef, SynchronousData> syncPorts;
};

static bool registeredClockSpecJSONHandler = [] {
  HandlerRegistry::get().registerHandler(std::make_unique<ClockSpecJSON>());
  return true;
}();

} // namespace handlers
} // namespace circt
