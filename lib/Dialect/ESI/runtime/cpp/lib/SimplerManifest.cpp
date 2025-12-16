//===- SimplerManifest.cpp - Simplified metadata for accelerators ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// The entirety of this file (beyond the stubs for the Impl class) was generated
// by Claude Opus 4.5 with the prompt below. I generated the example complex
// manifest by running a pycde integration test with the cosim backend. I also
// wrote the `simpler_manifest_test.cpp` tool by hand first so that the AI had
// an example of what I wanted. If you want a function exposed, AI should be
// able to figure it out if you just add a usage example to
// simpler_manifest_test.cpp.
//
// "Implement the SimplerManifest parser and accelerator creation. The complex
// parser and accelerator creation is implemented in Manifest.cpp. An example of
// a cosim-based esi system manifest is esi_system_manifest.json. The example of
// the simpler manifest I want you to implement is located in
// simpler_manifest_test.cpp. Ignore types. Make it cosim-only. Pay special
// attention to the cosim engine in the complex manifest example -- it contains
// the specification which you need to create for the simpler manifest to
// support cosim.""
//
// Then when it didn't work:
// "I've started the server in the background. I'm running the client with
// `simpler_manifest_test cosim localhost:1234` and getting the following error:
//
// ```
// simpler_manifest_test:
// /workspace/circt/lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:188:
// virtual void (anonymous namespace)::WriteCosimChannelPort::connectImpl(const
// ChannelPort::ConnectOptions &): Assertion `desc.name() == name' failed.
// Aborted (core dumped)
// ``` "
//
// It also detected an omission in the simpler manifest format -- the cosim
// channel was missing a direction -- so it added it.
//
// I've not read or reviewed it. A quick glance implies that it's doing some
// unnecessary things regarding the types. I've verified that it works.
//===----------------------------------------------------------------------===//

#include "esi/SimplerManifest.h"
#include "esi/Accelerator.h"
#include "esi/Services.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#endif
#include <nlohmann/json.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <sstream>

using namespace ::esi;

// This is a proxy class to the manifest JSON. It is used to avoid having to
// include the JSON parser in the header. Forward references don't work since
// nlohmann::json is a rather complex template.
//
// Plus, it allows us to hide some implementation functions from the header
// file.
class SimplerManifest::Impl {
  friend class ::esi::SimplerManifest;

public:
  Impl(Context &ctxt, const std::string &jsonManifest) : ctxt(ctxt) {
    manifestJson = nlohmann::json::parse(jsonManifest);
  }

  auto at(const std::string &key) const { return manifestJson.at(key); }

  /// Build a dynamic API for the Accelerator connection 'acc' based on the
  /// manifest stored herein.
  std::unique_ptr<Accelerator> buildAccelerator(AcceleratorConnection &acc);

private:
  Context &ctxt;

  // The parsed json.
  nlohmann::json manifestJson;
};

std::unique_ptr<Accelerator>
SimplerManifest::Impl::buildAccelerator(AcceleratorConnection &acc) {
  // Build the client details for the cosim engine from the ports section.
  // Each port in the manifest maps to a channel assignment for the engine.
  HWClientDetails clientDetails;

  auto portsIter = manifestJson.find("ports");
  if (portsIter != manifestJson.end()) {
    for (const auto &port : portsIter.value()) {
      HWClientDetail clientDetail;

      // Get the port name (which becomes the AppID).
      std::string portName = port.at("name").get<std::string>();
      clientDetail.relPath = {AppID(portName)};

      // Get the cosim channel name.
      std::string cosimChannelName =
          port.at("cosim_channel_name").get<std::string>();

      // Determine direction - check if port has a direction field, otherwise
      // default based on common patterns.
      std::string direction = "to_host"; // Default
      if (port.contains("direction")) {
        direction = port.at("direction").get<std::string>();
      }

      // For simple ports, we assume a single "data" channel in the bundle.
      // The channel assignment maps the bundle's channel to a cosim channel.
      ChannelAssignment chanAssign;
      chanAssign.type = "cosim";
      chanAssign.implOptions["name"] = cosimChannelName;

      // Determine the channel name within the bundle. For simple from_host
      // ports, it's typically "data". For function ports, it could be
      // "arg"/"result".
      std::string bundleChannelName = "data";
      if (port.contains("bundle_channel")) {
        bundleChannelName = port.at("bundle_channel").get<std::string>();
      }
      clientDetail.channelAssignments[bundleChannelName] = chanAssign;

      // Set the service port info.
      if (direction == "from_host" || direction == "to") {
        clientDetail.port = {"@_ChannelServiceDecl", "from_host"};
      } else {
        clientDetail.port = {"@_ChannelServiceDecl", "to_host"};
      }

      clientDetails.push_back(clientDetail);
    }
  }

  // Create the cosim engine with the client details.
  // The engine AppID is "cosim" by convention.
  ServiceImplDetails engineDetails;
  acc.createEngine("cosim", {AppID("cosim")}, engineDetails, clientDetails);

  // Build bundle ports for each port in the manifest.
  std::vector<std::unique_ptr<BundlePort>> ports;

  if (portsIter != manifestJson.end()) {
    for (const auto &port : portsIter.value()) {
      std::string portName = port.at("name").get<std::string>();
      AppIDPath idPath = {AppID(portName)};

      // Get the engine map for this port.
      const BundleEngineMap &engineMap = acc.getEngineMapFor(idPath);

      // Determine direction for bundle type creation.
      std::string direction = "to_host";
      if (port.contains("direction")) {
        direction = port.at("direction").get<std::string>();
      }

      // Create a simple bundle type with a single channel.
      // For simplicity, we use a void/any type since we're ignoring types.
      BundleType::Direction bundleDir =
          (direction == "from_host" || direction == "to")
              ? BundleType::Direction::To
              : BundleType::Direction::From;

      // Get or create a bundle type. We use nullptr for the inner type
      // since we're ignoring types.
      std::string bundleChannelName = "data";
      if (port.contains("bundle_channel")) {
        bundleChannelName = port.at("bundle_channel").get<std::string>();
      }

      // Create the bundle type ID string.
      std::string bundleTypeId =
          "!esi.bundle<[!esi.channel<!esi.any> " +
          std::string(bundleDir == BundleType::Direction::To ? "to" : "from") +
          " \"" + bundleChannelName + "\"]>";

      // Get or create the bundle type in the context.
      const Type *anyType = ctxt.getType("!esi.any").value_or(nullptr);
      if (!anyType) {
        auto *newAnyType = new AnyType("!esi.any");
        ctxt.registerType(newAnyType);
        anyType = newAnyType;
      }

      BundleType::ChannelVector channels = {
          {bundleChannelName, bundleDir, anyType}};
      const BundleType *bundleType = dynamic_cast<const BundleType *>(
          ctxt.getType(bundleTypeId).value_or(nullptr));
      if (!bundleType) {
        auto *newBundleType = new BundleType(bundleTypeId, channels);
        ctxt.registerType(newBundleType);
        bundleType = newBundleType;
      }

      // Request ports from the engine map.
      PortMap channelPorts = engineMap.requestPorts(idPath, bundleType);

      // Create the bundle port.
      ports.push_back(std::make_unique<BundlePort>(AppID(portName), bundleType,
                                                   std::move(channelPorts)));
    }
  }

  // Create and return the Accelerator with no children and no services
  // (cosim-only).
  return std::make_unique<Accelerator>(
      std::nullopt,                             // No module info
      std::vector<std::unique_ptr<Instance>>{}, // No children
      std::vector<services::Service *>{},       // No services
      std::move(ports));
}

SimplerManifest::SimplerManifest(Context &ctxt,
                                 const std::string &jsonManifest) {
  impl = std::make_unique<Impl>(ctxt, jsonManifest);
}
SimplerManifest::~SimplerManifest() {}

Accelerator *
SimplerManifest::buildAccelerator(AcceleratorConnection &acc) const {
  auto accel = impl->buildAccelerator(acc);
  return acc.takeOwnership(std::move(accel));
}
