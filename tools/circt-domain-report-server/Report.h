//===- Report.h - Indexed FIRRTL domain inference report -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_DOMAIN_REPORT_REPORT_H
#define CIRCT_TOOLS_DOMAIN_REPORT_REPORT_H

#include "llvm/Support/JSON.h"
#include <cstdint>
#include <functional>
#include <istream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace circt::domain_report {

class Report {
public:
  static constexpr uint32_t none = std::numeric_limits<uint32_t>::max();

  bool load(std::istream &input, std::string &error,
            std::function<void(uint64_t)> progress = {});
  void setSourceRoots(const std::vector<std::string> &roots);
  llvm::json::Value request(llvm::StringRef method,
                            const llvm::json::Object &params,
                            std::string &error) const;
  llvm::json::Object summary() const;

private:
  struct Assignment {
    int64_t domainType = -1;
    int64_t domainValue = -1;
    bool inferred = false;
  };
  struct Source {
    std::string file;
    uint32_t line = 0;
    uint32_t column = 0;
    uint32_t endLine = 0;
    uint32_t endColumn = 0;
    bool range = false;
  };
  struct Location {
    std::string display;
    std::vector<Source> sources;
  };
  struct Domain {
    int64_t id = -1;
    std::string name;
    uint32_t location = none;
  };
  struct Value {
    int64_t id = -1;
    std::string name;
    std::string kind;
    std::string definition;
    std::string direction;
    uint32_t module = none;
    uint32_t type = none;
    uint32_t location = none;
    int64_t domainType = -1;
    int64_t instance = -1;
    int32_t portIndex = -1;
    int32_t instancePortIndex = -1;
    uint32_t assignmentBegin = 0;
    uint32_t assignmentCount = 0;
  };
  struct PortAssignment {
    int64_t domainType = -1;
    int64_t domainPortValue = -1;
    int32_t domainPortIndex = -1;
    bool inferred = false;
  };
  struct Port {
    int32_t index = -1;
    int64_t value = -1;
    std::string name;
    std::string direction;
    uint32_t type = none;
    uint32_t location = none;
    std::vector<PortAssignment> assignments;
  };
  struct Binding {
    int64_t targetModule = -1;
    int32_t portIndex = -1;
    int64_t portValue = -1;
    int64_t domainType = -1;
    int32_t domainPortIndex = -1;
    int64_t effectiveDomainValue = -1;
    std::string effectiveDomainValueName;
    uint32_t location = none;
  };
  struct Instance {
    int64_t id = -1;
    std::string name;
    uint32_t module = none;
    uint32_t location = none;
    std::vector<std::string> targets;
    std::vector<Binding> bindings;
  };
  struct Module {
    int64_t id = -1;
    std::string name;
    std::string kind;
    std::vector<Port> ports;
    std::vector<uint32_t> values;
    std::vector<uint32_t> instances;
  };
  struct Edge {
    uint32_t owner = none;
    uint32_t kind = none;
    uint32_t domain = none;
    uint32_t location = none;
    uint32_t lhs = none;
    uint32_t rhs = none;
    uint32_t operation = none;
    uint32_t instance = none;
    uint32_t target = none;
    uint32_t flags = 0;
  };
  struct Crossing {
    uint32_t owner = none;
    uint32_t domain = none;
    uint32_t location = none;
    uint32_t operation = none;
    uint32_t lhs = none;
    uint32_t rhs = none;
    uint32_t lhsDomain = none;
    uint32_t rhsDomain = none;
    uint32_t lhsSource = none;
    uint32_t rhsSource = none;
  };
  struct SourceHit {
    uint32_t location;
    uint32_t sourceIndex;
  };
  struct DomainAssociation {
    uint32_t value = none;
    uint32_t module = none;
    uint32_t port = none;
    int64_t domainValue = -1;
    int32_t domainPortIndex = -1;
  };
  struct DomainComponent {
    std::vector<uint32_t> nodes;
    std::vector<uint32_t> edges;
    std::vector<uint32_t> associations;
    std::vector<uint32_t> crossings;
  };

  bool complete = false;
  bool hasComplete = false;
  bool hasFormat = false;
  bool hasVersion = false;
  int64_t version = -1;
  uint64_t skippedEdges = 0;
  std::vector<std::string> types;
  std::vector<std::string> operationKinds;
  std::vector<std::string> edgeKinds;
  std::vector<std::string> edgeFields;
  std::vector<Location> locations;
  std::vector<Domain> domains;
  std::vector<Module> modules;
  std::vector<Value> values;
  std::vector<Assignment> assignments;
  std::vector<Instance> instances;
  std::vector<Edge> edges;
  std::vector<Crossing> crossings;
  std::vector<std::vector<uint32_t>> crossingsByDomain;
  bool hasIllegalCrossings = false;
  std::unordered_map<int64_t, uint32_t> domainById;
  std::unordered_map<int64_t, uint32_t> moduleById;
  std::unordered_map<int64_t, uint32_t> valueById;
  std::unordered_map<int64_t, uint32_t> instanceById;
  std::vector<std::vector<uint32_t>> valuesByLocation;
  std::vector<uint32_t> adjacencyOffsets;
  std::vector<uint32_t> adjacencyEdges;
  std::vector<std::vector<uint32_t>> domainNodes;
  std::vector<std::vector<uint32_t>> domainEdges;
  std::vector<std::vector<DomainAssociation>> domainAssociations;
  std::vector<std::vector<DomainComponent>> domainComponents;
  std::vector<std::vector<uint32_t>> unmappedAssociations;
  std::vector<std::vector<uint32_t>> unmappedCrossings;
  std::unordered_map<std::string, std::vector<SourceHit>> sourceIndex;
  std::vector<std::string> sourceRoots;
  int inferredFlag = 1;
  int summarizedFlag = 2;

  llvm::json::Object locationJSON(uint32_t id) const;
  llvm::json::Object valueJSON(uint32_t index, bool details = false) const;
  llvm::json::Object instanceJSON(uint32_t index, bool details = false) const;
  llvm::json::Object edgeJSON(uint32_t index) const;
  llvm::json::Object crossingJSON(uint32_t index) const;
  llvm::json::Value crossingPath(const llvm::json::Object &params,
                                 std::string &error) const;
  llvm::json::Value trace(const llvm::json::Object &params,
                          std::string &error) const;
  llvm::json::Value neighbors(const llvm::json::Object &params,
                              std::string &error) const;
  llvm::json::Value domainItems(const llvm::json::Object &params,
                                std::string &error) const;
  llvm::json::Value minCut(const llvm::json::Object &params,
                           std::string &error) const;
};

} // namespace circt::domain_report

#endif
