//===- esiquery.cpp - ESI accelerator system query tool -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT
// (lib/dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp).
//
//===----------------------------------------------------------------------===//

#include "esi/Accelerator.h"
#include "esi/Manifest.h"
#include "esi/StdServices.h"

#include <iostream>
#include <map>
#include <stdexcept>

using namespace std;

using namespace esi;

void printInfo(ostream &os, Accelerator &acc);

int main(int argc, const char *argv[]) {
  // TODO: find a command line parser library rather than doing this by hand.
  if (argc < 3) {
    cerr << "Expected usage: " << argv[0]
         << " <backend> <connection specifier> [command]" << endl;
    return -1;
  }

  const char *backend = argv[1];
  const char *conn = argv[2];
  string cmd;
  if (argc > 3)
    cmd = argv[3];

  try {
    unique_ptr<Accelerator> acc = registry::connect(backend, conn);
    const auto &info = *acc->getService<services::SysInfo>();

    if (cmd == "version")
      cout << "ESI system version: " << info.getEsiVersion() << endl;
    else if (cmd == "json_manifest")
      cout << info.getJsonManifest() << endl;
    else if (cmd == "info")
      printInfo(cout, *acc);
    // TODO: add a command to print out the instance hierarchy.
    else {
      cout << "Connection successful." << endl;
      if (!cmd.empty()) {
        cerr << "Unknown command: " << cmd << endl;
        return 1;
      }
    }
    return 0;
  } catch (exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }
}

void printInfo(ostream &os, Accelerator &acc) {
  string jsonManifest = acc.getService<services::SysInfo>()->getJsonManifest();
  Manifest m(jsonManifest);
  os << "API version: " << m.getApiVersion() << endl << endl;
  os << "********************************" << endl;
  os << "* Design information" << endl;
  os << "********************************" << endl;
  os << endl;
  for (ModuleInfo mod : m.getModuleInfos())
    os << mod << endl;

  os << "********************************" << endl;
  os << "* Type table" << endl;
  os << "********************************" << endl;
  os << endl;
  size_t i = 0;
  for (Type t : m.getTypeTable())
    os << "  " << i++ << ": " << t.getID() << endl;
}
