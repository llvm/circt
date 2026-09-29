//===- ReportTest.cpp - Domain report reader tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Report.h"
#include "gtest/gtest.h"
#include <climits>
#include <random>
#include <sstream>
#include <tuple>
#include <unordered_set>

using namespace circt::domain_report;
using llvm::json::Object;

static std::string exampleReport() {
  // The field table deliberately differs from the documentation's order.
  return R"json({
    "format":"circt-domain-inference", "version":3, "complete":true,
    "types":["!firrtl.uint<1>","!firrtl.domain<@D()>"],
    "locations":[{"display":"fused source", "sources":[
      {"file":"/tmp/a.scala","line":2,"column":3},
      {"file":"/tmp/b.scala","start_line":4,"start_column":5,
       "end_line":4,"end_column":9}]}],
    "operation_kinds":["firrtl.matchingconnect"],
    "domains":[{"id":42,"name":"D","location_id":0}],
    "provenance_edge_kinds":["constraint","association","instance_binding"],
    "provenance_edge_fields":["rhs_value_id","lhs_value_id","flags",
      "location_id","domain_type_id","kind_id","owner_module_id",
      "operation_kind_id","instance_id","target_module_id"],
    "provenance_edge_flags":{"inferred":1,"summarized":2},
    "instance_binding_direction":"parent_instance_to_module_template",
    "modules":[
      {"id":10,"name":"Parent","kind":"module",
       "ports":[{"index":0,"value_id":100}],
       "values":[
         {"id":100,"name":"signal","kind":"hardware","type_id":0,
          "location_id":0,"domain_assignments":[{"domain_type_id":42,
          "domain_value_id":300,"inferred":false}]},
         {"id":300,"name":"clockDomain","kind":"domain","type_id":1,
          "location_id":0,"domain_type_id":42}],
       "instances":[{"id":900,"name":"child","location_id":0,
         "targets":[11],"effective_domain_bindings":[{
         "target_module_id":11,"port_index":0,"port_value_id":100,
         "domain_type_id":42,"domain_port_index":0,
         "effective_domain_value_id":300,
         "effective_domain_value_name":"clockDomain","location_id":0}]}]},
      {"id":11,"name":"Child","kind":"module",
       "ports":[{"index":0,"value_id":200}],
       "values":[
         {"id":200,"name":"childSignal","kind":"hardware","type_id":0,
          "location_id":0,"domain_assignments":[{"domain_type_id":42,
          "domain_value_id":201,"inferred":true}]},
         {"id":201,"name":"childDomain","kind":"domain","type_id":1,
          "location_id":0,"domain_type_id":42}],"instances":[]},
      {"id":12,"name":"External","kind":"extmodule",
       "ports":[{"index":0,"name":"ext","type_id":0,"direction":"in",
         "location_id":0}],"values":[],"instances":[]}],
    "provenance_edges":[
      [300,100,0,0,42,1,10,0,null,null],
      [200,100,1,0,42,2,10,0,900,11],
      [200,200,1,0,42,0,11,0,null,null]]
  })json";
}

static std::string
graphReport(const std::vector<std::tuple<int, int, int, int>> &edgeRows) {
  std::ostringstream edges;
  for (size_t i = 0; i < edgeRows.size(); ++i) {
    auto [lhs, rhs, kind, flags] = edgeRows[i];
    if (i)
      edges << ',';
    edges << "[10," << kind << ",42,0," << flags << ',' << lhs << ',' << rhs
          << ",0,null,null]";
  }
  return std::string(R"json({
    "format":"circt-domain-inference", "version":3, "complete":true,
    "types":["!firrtl.uint<1>","!firrtl.domain<@D()>"],
    "locations":[{"display":"test", "sources":[]}],
    "operation_kinds":["firrtl.connect"],
    "domains":[{"id":42,"name":"D","location_id":0}],
    "provenance_edge_kinds":["constraint","association"],
    "provenance_edge_fields":["owner_module_id","kind_id",
      "domain_type_id","location_id","flags","lhs_value_id","rhs_value_id",
      "operation_kind_id","instance_id","target_module_id"],
    "provenance_edge_flags":{"inferred":1,"summarized":2},
    "instance_binding_direction":"parent_instance_to_module_template",
    "modules":[{"id":10,"name":"M","kind":"module",
      "ports":[{"index":0,"value_id":1,"domain_assignments":[{
        "domain_type_id":42,"domain_port_index":1,
        "domain_port_value_id":5,"inferred":false}]},
        {"index":1,"value_id":5}],
      "values":[
        {"id":1,"name":"portA","kind":"hardware","type_id":0,
         "location_id":0,"port_index":0,"port_direction":"in",
         "domain_assignments":[{"domain_type_id":42,
           "domain_value_id":5,"inferred":false}]},
        {"id":2,"name":"portB","kind":"hardware","type_id":0,
         "location_id":0,"domain_assignments":[{"domain_type_id":42,
           "domain_value_id":5,"inferred":true}]},
        {"id":3,"name":"wireC","kind":"hardware","type_id":0,
         "location_id":0},
        {"id":4,"name":"wireD","kind":"hardware","type_id":0,
         "location_id":0},
        {"id":5,"name":"dom","kind":"domain","type_id":1,
         "location_id":0,"domain_type_id":42}
      ]},
      {"id":11,"name":"Ext","kind":"extmodule","ports":[
        {"index":0,"name":"extPort","type_id":0,"direction":"in",
         "location_id":0,"domain_assignments":[{
           "domain_type_id":42,"domain_port_index":1,
           "domain_port_value_id":null,"inferred":false}]},
        {"index":1,"name":"extDom","type_id":1,"direction":"in",
         "location_id":0}],"values":[],"instances":[]}
    ],
    "provenance_edges":[)json") +
         edges.str() + R"json(]
  })json";
}

static std::string crossingReport() {
  return R"json({
    "format":"circt-domain-inference", "version":3, "complete":false,
    "types":["!firrtl.uint<1>","!firrtl.domain<@D()>"],
    "locations":[{"display":"crossing", "sources":[]}],
    "operation_kinds":["firrtl.matchingconnect"],
    "domains":[{"id":42,"name":"D","location_id":0}],
    "provenance_edge_kinds":["constraint","association"],
    "provenance_edge_fields":["owner_module_id","kind_id",
      "domain_type_id","location_id","flags","lhs_value_id","rhs_value_id",
      "operation_kind_id","instance_id","target_module_id"],
    "provenance_edge_flags":{"inferred":1,"summarized":2},
    "instance_binding_direction":"parent_instance_to_module_template",
    "modules":[{"id":10,"name":"M","kind":"module","ports":[],
      "values":[
        {"id":1,"name":"A","kind":"domain","type_id":1,
         "location_id":0,"domain_type_id":42},
        {"id":2,"name":"B","kind":"domain","type_id":1,
         "location_id":0,"domain_type_id":42},
        {"id":3,"name":"left","kind":"hardware","type_id":0,
         "location_id":0},
        {"id":4,"name":"right","kind":"hardware","type_id":0,
         "location_id":0},
        {"id":5,"name":"middle","kind":"hardware","type_id":0,
         "location_id":0}],"instances":[]}],
    "provenance_edges":[
      [10,1,42,0,0,1,3,0,null,null],
      [10,1,42,0,1,1,5,0,null,null],
      [10,0,42,0,0,5,3,0,null,null],
      [10,1,42,0,0,4,2,0,null,null]],
    "illegal_crossings":[{
      "owner_module_id":10,"domain_type_id":42,"location_id":0,
      "operation_kind_id":0,"lhs_value_id":3,"rhs_value_id":4,
      "lhs_domain_value_id":1,"rhs_domain_value_id":2,
      "lhs_source_value_id":1,"rhs_source_value_id":2}]
  })json";
}

TEST(DomainReport, ExplainsIllegalCrossingWithShortestPath) {
  std::istringstream input(crossingReport());
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  EXPECT_EQ(report.summary().getInteger("illegalCrossings"), 1);
  EXPECT_EQ(report.summary().getBoolean("hasIllegalCrossings"), true);
  auto domains = report.request("listDomains", Object{}, error);
  ASSERT_TRUE(domains.getAsObject()) << error;
  const auto *domain =
      domains.getAsObject()->getArray("items")->front().getAsObject();
  ASSERT_TRUE(domain);
  EXPECT_EQ(domain->getInteger("illegalCrossings"), 1);
  auto components = report.request("listDomainComponents",
                                   Object{{"domainTypeId", "42"}}, error);
  ASSERT_TRUE(components.getAsObject()) << error;
  EXPECT_EQ(components.getAsObject()->getInteger("total"), 2);
  for (const auto &item : *components.getAsObject()->getArray("items"))
    EXPECT_EQ(item.getAsObject()->getInteger("illegalCrossings"), 1);
  auto leftComponent =
      report.request("componentForValue",
                     Object{{"domainTypeId", "42"}, {"valueId", "3"}}, error);
  ASSERT_TRUE(leftComponent.getAsObject()) << error;
  EXPECT_EQ(leftComponent.getAsObject()->getInteger("index"), 0);
  auto rightComponent =
      report.request("componentForValue",
                     Object{{"domainTypeId", "42"}, {"valueId", "4"}}, error);
  ASSERT_TRUE(rightComponent.getAsObject()) << error;
  EXPECT_EQ(rightComponent.getAsObject()->getInteger("index"), 1);

  auto listed = report.request("listIllegalCrossings",
                               Object{{"domainTypeId", "42"}}, error);
  ASSERT_TRUE(listed.getAsObject()) << error;
  const auto *crossing =
      listed.getAsObject()->getArray("items")->front().getAsObject();
  ASSERT_TRUE(crossing);
  EXPECT_EQ(crossing->getString("lhsSource"), "A");
  EXPECT_EQ(crossing->getString("rhsSource"), "B");
  for (int componentIndex = 0; componentIndex < 2; ++componentIndex) {
    auto involved = report.request(
        "listIllegalCrossings",
        Object{{"domainTypeId", "42"}, {"componentIndex", componentIndex}},
        error);
    ASSERT_TRUE(involved.getAsObject()) << error;
    EXPECT_EQ(involved.getAsObject()->getInteger("total"), 1);
  }

  auto path =
      report.request("crossingPath", Object{{"crossingIndex", 0}}, error);
  ASSERT_TRUE(path.getAsObject()) << error;
  EXPECT_EQ(path.getAsObject()->getBoolean("found"), true);
  const auto *steps = path.getAsObject()->getArray("steps");
  ASSERT_TRUE(steps);
  ASSERT_EQ(steps->size(), 3u);
  EXPECT_EQ((*steps)[0].getAsObject()->getString("kind"), "association");
  EXPECT_EQ((*steps)[0].getAsObject()->getString("fromId"), "1");
  EXPECT_EQ((*steps)[1].getAsObject()->getString("kind"), "failed_constraint");
  EXPECT_EQ((*steps)[1].getAsObject()->getString("fromId"), "3");
  EXPECT_EQ((*steps)[1].getAsObject()->getString("toId"), "4");
  EXPECT_EQ((*steps)[2].getAsObject()->getString("kind"), "association");
  EXPECT_EQ((*steps)[2].getAsObject()->getString("toId"), "2");

  auto connections = report.request(
      "domainItems",
      Object{{"domainTypeId", "42"}, {"category", "connections"}}, error);
  ASSERT_TRUE(connections.getAsObject()) << error;
  EXPECT_EQ(connections.getAsObject()->getInteger("total"), 4);
  auto leftConnections = report.request("domainItems",
                                        Object{{"domainTypeId", "42"},
                                               {"componentIndex", 0},
                                               {"category", "connections"}},
                                        error);
  ASSERT_TRUE(leftConnections.getAsObject()) << error;
  EXPECT_EQ(leftConnections.getAsObject()->getInteger("total"), 3);

  auto missingSource = crossingReport();
  const std::string sourceField = "\"lhs_source_value_id\":1";
  auto position = missingSource.find(sourceField);
  ASSERT_NE(position, std::string::npos);
  missingSource.replace(position, sourceField.size(),
                        "\"lhs_source_value_id\":null");
  std::istringstream partial(missingSource);
  ASSERT_TRUE(report.load(partial, error)) << error;
  auto unavailable =
      report.request("crossingPath", Object{{"crossingIndex", 0}}, error);
  ASSERT_TRUE(unavailable.getAsObject()) << error;
  EXPECT_EQ(unavailable.getAsObject()->getBoolean("found"), false);
}

TEST(DomainReport, ListsDomainAssociationsConnectionsAndNodes) {
  std::istringstream input(graphReport({{1, 5, 1, 0},
                                        {1, 2, 0, 0},
                                        {2, 3, 0, 0},
                                        {3, 4, 0, 0},
                                        {4, 1, 0, 0},
                                        {2, 4, 0, 0}}));
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  auto domains = report.request("listDomains", Object{}, error);
  ASSERT_TRUE(domains.getAsObject());
  const auto *domain =
      domains.getAsObject()->getArray("items")->front().getAsObject();
  ASSERT_TRUE(domain);
  EXPECT_EQ(domain->getInteger("nodes"), 5);
  EXPECT_EQ(domain->getInteger("connections"), 6);
  EXPECT_EQ(domain->getInteger("explicitAssociations"), 2);
  EXPECT_EQ(domain->getInteger("components"), 1);

  auto components = report.request("listDomainComponents",
                                   Object{{"domainTypeId", "42"}}, error);
  ASSERT_TRUE(components.getAsObject());
  EXPECT_EQ(components.getAsObject()->getInteger("total"), 2);
  EXPECT_EQ(components.getAsObject()
                ->getArray("items")
                ->front()
                .getAsObject()
                ->getInteger("nodes"),
            5);
  EXPECT_EQ(components.getAsObject()
                ->getArray("items")
                ->front()
                .getAsObject()
                ->getInteger("explicitAssociations"),
            1);
  const auto *unmapped =
      components.getAsObject()->getArray("items")->back().getAsObject();
  ASSERT_TRUE(unmapped);
  EXPECT_EQ(unmapped->getBoolean("unmapped"), true);
  EXPECT_EQ(unmapped->getInteger("explicitAssociations"), 1);

  auto associations = report.request(
      "domainItems",
      Object{{"domainTypeId", "42"}, {"category", "associations"}}, error);
  ASSERT_TRUE(associations.getAsObject());
  EXPECT_EQ(associations.getAsObject()->getInteger("total"), 2);
  const auto *first =
      associations.getAsObject()->getArray("items")->front().getAsObject();
  ASSERT_TRUE(first);
  EXPECT_EQ(first->getString("valueId"), "1");
  const auto *external =
      associations.getAsObject()->getArray("items")->back().getAsObject();
  ASSERT_TRUE(external);
  EXPECT_EQ(external->getString("name"), "extPort");
  EXPECT_EQ(external->getString("domainValue"), "extDom");
  auto componentAssociations =
      report.request("domainItems",
                     Object{{"domainTypeId", "42"},
                            {"componentIndex", 0},
                            {"category", "associations"}},
                     error);
  ASSERT_TRUE(componentAssociations.getAsObject()) << error;
  EXPECT_EQ(componentAssociations.getAsObject()->getInteger("total"), 1);
  auto unmappedAssociations =
      report.request("domainItems",
                     Object{{"domainTypeId", "42"},
                            {"unmapped", true},
                            {"category", "associations"}},
                     error);
  ASSERT_TRUE(unmappedAssociations.getAsObject()) << error;
  EXPECT_EQ(unmappedAssociations.getAsObject()->getInteger("total"), 1);

  auto nodes = report.request(
      "domainItems",
      Object{{"domainTypeId", "42"}, {"category", "nodes"}, {"query", "wire"}},
      error);
  ASSERT_TRUE(nodes.getAsObject());
  EXPECT_EQ(nodes.getAsObject()->getInteger("total"), 2);
}

TEST(DomainReport, CutsAssociationAndFindsMinimumBetweenValues) {
  std::istringstream input(graphReport({{1, 5, 1, 0},
                                        {1, 2, 0, 0},
                                        {2, 3, 0, 0},
                                        {3, 4, 0, 0},
                                        {4, 1, 0, 0},
                                        {2, 4, 0, 0}}));
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  auto global = report.request("minCut", Object{{"domainTypeId", "42"}}, error);
  ASSERT_TRUE(global.getAsObject()) << error;
  EXPECT_EQ(global.getAsObject()->getInteger("count"), 1);
  const auto *edge =
      global.getAsObject()->getArray("edges")->front().getAsObject();
  ASSERT_TRUE(edge);
  EXPECT_EQ(edge->getString("kind"), "association");

  auto between = report.request(
      "minCut",
      Object{{"domainTypeId", "42"}, {"sourceId", "1"}, {"targetId", "3"}},
      error);
  ASSERT_TRUE(between.getAsObject()) << error;
  EXPECT_EQ(between.getAsObject()->getInteger("count"), 2);
  EXPECT_EQ(between.getAsObject()->getArray("edges")->size(), 2u);
  std::unordered_set<int64_t> removed;
  for (const auto &item : *between.getAsObject()->getArray("edges")) {
    const auto *cutEdge = item.getAsObject();
    ASSERT_TRUE(cutEdge);
    removed.insert(*cutEdge->getInteger("index"));
  }
  const std::pair<int, int> originalEdges[] = {{1, 5}, {1, 2}, {2, 3},
                                               {3, 4}, {4, 1}, {2, 4}};
  std::unordered_set<int> reached{1};
  bool changed;
  do {
    changed = false;
    for (size_t i = 0; i < std::size(originalEdges); ++i) {
      if (removed.count(i))
        continue;
      auto [lhs, rhs] = originalEdges[i];
      if (reached.count(lhs))
        changed |= reached.insert(rhs).second;
      if (reached.count(rhs))
        changed |= reached.insert(lhs).second;
    }
  } while (changed);
  EXPECT_FALSE(reached.count(3));
}

TEST(DomainReport, CountsParallelEdgesAndPrunesFloatingFragments) {
  std::istringstream parallel(
      graphReport({{1, 5, 1, 0}, {1, 2, 0, 0}, {1, 2, 0, 0}}));
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(parallel, error)) << error;
  auto between = report.request(
      "minCut",
      Object{{"domainTypeId", "42"}, {"sourceId", "1"}, {"targetId", "2"}},
      error);
  ASSERT_TRUE(between.getAsObject()) << error;
  EXPECT_EQ(between.getAsObject()->getInteger("count"), 2);

  std::istringstream disconnected(graphReport({{1, 5, 1, 0}, {3, 4, 0, 0}}));
  ASSERT_TRUE(report.load(disconnected, error)) << error;
  auto domains = report.request("listDomains", Object{}, error);
  ASSERT_TRUE(domains.getAsObject());
  const auto *domain =
      domains.getAsObject()->getArray("items")->front().getAsObject();
  ASSERT_TRUE(domain);
  EXPECT_EQ(domain->getInteger("nodes"), 3);
  EXPECT_EQ(domain->getInteger("connections"), 1);
  EXPECT_EQ(domain->getInteger("components"), 2);
  auto components = report.request("listDomainComponents",
                                   Object{{"domainTypeId", "42"}}, error);
  ASSERT_TRUE(components.getAsObject());
  EXPECT_EQ(components.getAsObject()->getInteger("total"), 3);
  EXPECT_EQ(components.getAsObject()
                ->getArray("items")
                ->front()
                .getAsObject()
                ->getInteger("nodes"),
            2);
  auto componentCut = report.request(
      "minCut", Object{{"domainTypeId", "42"}, {"componentIndex", 0}}, error);
  ASSERT_TRUE(componentCut.getAsObject()) << error;
  EXPECT_EQ(componentCut.getAsObject()->getInteger("count"), 1);
  auto isolated = report.request(
      "minCut", Object{{"domainTypeId", "42"}, {"componentIndex", 1}}, error);
  ASSERT_TRUE(isolated.getAsObject()) << error;
  EXPECT_EQ(isolated.getAsObject()->getBoolean("available"), false);
  auto global = report.request("minCut", Object{{"domainTypeId", "42"}}, error);
  ASSERT_TRUE(global.getAsObject()) << error;
  EXPECT_EQ(global.getAsObject()->getInteger("count"), 0);
  between = report.request(
      "minCut",
      Object{{"domainTypeId", "42"}, {"sourceId", "1"}, {"targetId", "2"}},
      error);
  ASSERT_TRUE(between.getAsObject()) << error;
  EXPECT_EQ(between.getAsObject()->getInteger("count"), 0);
}

TEST(DomainReport, ComputesExactGlobalCutWithoutBridges) {
  std::vector<std::tuple<int, int, int, int>> rows;
  for (int lhs = 1; lhs <= 5; ++lhs)
    for (int rhs = lhs + 1; rhs <= 5; ++rhs)
      rows.push_back({lhs, rhs, lhs == 1 && rhs == 5 ? 1 : 0, 0});
  std::istringstream input(graphReport(rows));
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  auto global = report.request("minCut", Object{{"domainTypeId", "42"}}, error);
  ASSERT_TRUE(global.getAsObject()) << error;
  EXPECT_EQ(global.getAsObject()->getInteger("count"), 4);
}

TEST(DomainReport, CutCountsAgreeWithSmallGraphEnumeration) {
  std::minstd_rand random(17);
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<std::tuple<int, int, int, int>> rows{
        {3, 3, 0, 0}, {4, 4, 0, 0}, {1, 3, 0, 0}, {1, 4, 0, 0}};
    for (int lhs = 1; lhs <= 5; ++lhs)
      for (int rhs = lhs + 1; rhs <= 5; ++rhs)
        for (unsigned count = random() % 3; count; --count)
          rows.push_back({lhs, rhs, lhs == 1 && rhs == 5 ? 1 : 0, 0});
    std::istringstream input(graphReport(rows));
    Report report;
    std::string error;
    ASSERT_TRUE(report.load(input, error)) << error;

    int globalMinimum = INT_MAX, pairMinimum = INT_MAX;
    for (unsigned side = 1; side < (1u << 5) - 1; ++side) {
      int count = 0;
      for (auto [lhs, rhs, kind, flags] : rows)
        count += ((side >> (lhs - 1)) & 1) != ((side >> (rhs - 1)) & 1);
      globalMinimum = std::min(globalMinimum, count);
      if (((side >> 0) & 1) != ((side >> 2) & 1))
        pairMinimum = std::min(pairMinimum, count);
    }
    auto global =
        report.request("minCut", Object{{"domainTypeId", "42"}}, error);
    ASSERT_TRUE(global.getAsObject()) << error;
    EXPECT_EQ(global.getAsObject()->getInteger("count"), globalMinimum)
        << "trial " << trial;
    auto between = report.request(
        "minCut",
        Object{{"domainTypeId", "42"}, {"sourceId", "1"}, {"targetId", "3"}},
        error);
    ASSERT_TRUE(between.getAsObject()) << error;
    EXPECT_EQ(between.getAsObject()->getInteger("count"), pairMinimum)
        << "trial " << trial;
  }
}

TEST(DomainReport, IndexesSparseIDsAndSourceLocations) {
  std::istringstream input(exampleReport());
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  report.setSourceRoots({});
  auto summary = report.summary();
  EXPECT_EQ(summary.getInteger("modules"), 3);
  EXPECT_EQ(summary.getInteger("values"), 4);
  auto modules = report.request("listModules", Object{}, error);
  ASSERT_TRUE(modules.getAsObject());
  EXPECT_EQ(modules.getAsObject()->getInteger("total"), 3);
  auto externalPorts = report.request(
      "moduleItems", Object{{"moduleId", "12"}, {"category", "ports"}}, error);
  ASSERT_TRUE(externalPorts.getAsObject());
  EXPECT_EQ(externalPorts.getAsObject()->getInteger("total"), 1);
  EXPECT_EQ(externalPorts.getAsObject()
                ->getArray("items")
                ->front()
                .getAsObject()
                ->getString("name"),
            "ext");
  auto source = report.request(
      "sourceMatches",
      Object{{"path", "/tmp/b.scala"}, {"line", 4}, {"column", 6}}, error);
  ASSERT_TRUE(source.getAsObject());
  EXPECT_EQ(source.getAsObject()->getInteger("total"), 4);
  auto afterRange = report.request(
      "sourceMatches",
      Object{{"path", "/tmp/b.scala"}, {"line", 4}, {"column", 9}}, error);
  EXPECT_EQ(afterRange.getAsObject()->getInteger("total"), 0);
  auto point = report.request(
      "sourceMatches",
      Object{{"path", "/tmp/a.scala"}, {"line", 2}, {"column", 4}}, error);
  EXPECT_EQ(point.getAsObject()->getInteger("total"), 0);
}

TEST(DomainReport, RequiresInstanceContextForTemplateTrace) {
  std::istringstream input(exampleReport());
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  Object params{{"valueId", "200"}, {"domainTypeId", "42"}};
  auto withoutContext = report.request("trace", params, error);
  ASSERT_TRUE(withoutContext.getAsObject());
  EXPECT_EQ(withoutContext.getAsObject()->getBoolean("found"), false);
  EXPECT_EQ(withoutContext.getAsObject()->getBoolean("contextRequired"), true);
  params["instanceId"] = "900";
  auto withContext = report.request("trace", params, error);
  ASSERT_TRUE(withContext.getAsObject());
  EXPECT_EQ(withContext.getAsObject()->getBoolean("found"), true);
  EXPECT_EQ(withContext.getAsObject()->getArray("steps")->size(), 2u);
}

TEST(DomainReport, ReadsEdgesBeforeTheirTables) {
  std::string text = exampleReport();
  auto edgeStart = text.find("    \"provenance_edges\":[");
  auto edgeEnd = text.find("\n  }", edgeStart);
  ASSERT_NE(edgeStart, std::string::npos);
  ASSERT_NE(edgeEnd, std::string::npos);
  std::string edgeBlock = text.substr(edgeStart, edgeEnd - edgeStart);
  text.erase(edgeStart, edgeEnd - edgeStart);
  // The edge array was the final field, so remove its preceding comma too.
  ASSERT_GE(edgeStart, 2u);
  ASSERT_EQ(text[edgeStart - 2], ',');
  text.erase(edgeStart - 2, 1);
  auto insertAt = text.find("    \"provenance_edge_fields\":[");
  ASSERT_NE(insertAt, std::string::npos);
  text.insert(insertAt, edgeBlock + ",\n");

  std::istringstream input(text);
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  EXPECT_EQ(report.summary().getInteger("provenanceEdges"), 3);
  auto trace = report.request(
      "trace",
      Object{{"valueId", "200"}, {"domainTypeId", "42"}, {"instanceId", "900"}},
      error);
  ASSERT_TRUE(trace.getAsObject());
  EXPECT_EQ(trace.getAsObject()->getBoolean("found"), true);
}

TEST(DomainReport, ChoosesShortestRouteIncludingSummaryEdges) {
  std::string text = exampleReport();
  auto finalEdge = text.find("[200,200,1,0,42,0,11,0,null,null]");
  ASSERT_NE(finalEdge, std::string::npos);
  text.insert(finalEdge, "[201,200,2,0,42,0,11,0,null,null],");
  std::istringstream input(text);
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  auto trace = report.request(
      "trace",
      Object{{"valueId", "200"}, {"domainTypeId", "42"}, {"instanceId", "900"}},
      error);
  ASSERT_TRUE(trace.getAsObject());
  EXPECT_EQ(trace.getAsObject()->getBoolean("found"), true);
  EXPECT_EQ(trace.getAsObject()->getArray("steps")->size(), 1u);
  EXPECT_EQ(trace.getAsObject()->getBoolean("usesSummarizedEdges"), true);
}

TEST(DomainReport, PartialReportSkipsMissingEdges) {
  std::string text = exampleReport();
  auto complete = text.find("\"complete\":true");
  ASSERT_NE(complete, std::string::npos);
  text.replace(complete, 15, "\"complete\":false");
  auto finalEdge = text.find("[200,200,1,0,42,0,11,0,null,null]");
  ASSERT_NE(finalEdge, std::string::npos);
  text.insert(finalEdge, "[9999,200,1,0,42,0,11,0,null,null],");
  std::istringstream input(text);
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  EXPECT_EQ(report.summary().getBoolean("complete"), false);
  EXPECT_EQ(report.summary().getInteger("skippedEdges"), 1);
}

TEST(DomainReport, RejectsUnsupportedVersionAndMalformedJSON) {
  std::string text = exampleReport();
  auto version = text.find("\"version\":3");
  ASSERT_NE(version, std::string::npos);
  text.replace(version, 11, "\"version\":4");
  std::istringstream unsupported(text);
  Report report;
  std::string error;
  EXPECT_FALSE(report.load(unsupported, error));
  std::istringstream malformed("{\"format\":");
  EXPECT_FALSE(report.load(malformed, error));
}

TEST(DomainReport, CompleteReportRejectsMissingReferences) {
  std::string text = exampleReport();
  auto finalEdge = text.find("[200,200,1,0,42,0,11,0,null,null]");
  ASSERT_NE(finalEdge, std::string::npos);
  text.insert(finalEdge, "[9999,200,1,0,42,0,11,0,null,null],");
  std::istringstream input(text);
  Report report;
  std::string error;
  EXPECT_FALSE(report.load(input, error));
  EXPECT_NE(error.find("missing references"), std::string::npos);
}

TEST(DomainReport, ParsesStringAcrossInputBuffers) {
  std::string text = exampleReport();
  auto display = text.find("\"display\":\"fused source\"");
  ASSERT_NE(display, std::string::npos);
  std::string replacement =
      "\"display\":\"" + std::string(65520, 'x') + "\\\"quote\"";
  text.replace(display, 24, replacement);
  std::istringstream input(text);
  Report report;
  std::string error;
  ASSERT_TRUE(report.load(input, error)) << error;
  auto value = report.request("getValue", Object{{"id", "100"}}, error);
  ASSERT_TRUE(value.getAsObject());
  auto *location = value.getAsObject()->getObject("location");
  ASSERT_TRUE(location);
  EXPECT_EQ(location->getString("display")->size(), 65526u);
  EXPECT_TRUE(location->getString("display")->contains("\"quote"));
}
