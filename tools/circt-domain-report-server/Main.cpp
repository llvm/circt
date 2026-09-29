//===- Main.cpp - VS Code domain report helper ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Report.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace circt::domain_report;
using llvm::json::Object;
using llvm::json::Value;

static void send(Value value) {
  llvm::outs() << llvm::formatv("{0}", value) << '\n';
  llvm::outs().flush();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: circt-domain-report-server <report.json>\n";
    return 2;
  }
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) {
    send(Object{{"event", "error"}, {"message", "could not open report"}});
    return 1;
  }
  std::error_code fileError;
  uint64_t size = std::filesystem::file_size(argv[1], fileError);
  if (fileError)
    size = 0;
  Report report;
  std::string error;
  bool loaded = report.load(input, error, [&](uint64_t bytes) {
    send(Object{{"event", "progress"},
                {"bytes", static_cast<int64_t>(bytes)},
                {"total", static_cast<int64_t>(size)}});
  });
  if (!loaded) {
    send(Object{{"event", "error"}, {"message", error}});
    return 1;
  }
  send(Object{{"event", "ready"}, {"summary", report.summary()}});

  std::string line;
  while (std::getline(std::cin, line)) {
    auto parsed = llvm::json::parse(line);
    if (!parsed || !parsed->getAsObject()) {
      send(Object{{"event", "protocolError"},
                  {"message", "invalid request JSON"}});
      continue;
    }
    const auto &request = *parsed->getAsObject();
    auto id = request.getInteger("id");
    auto method = request.getString("method");
    const auto *params = request.getObject("params");
    if (!id || !method || !params) {
      send(Object{{"event", "protocolError"},
                  {"message", "request needs id, method, and params"}});
      continue;
    }
    Value result(nullptr);
    error.clear();
    if (*method == "setSourceRoots") {
      std::vector<std::string> roots;
      auto *items = params->getArray("roots");
      if (!items)
        error = "expected source roots array";
      else {
        for (const auto &item : *items) {
          auto root = item.getAsString();
          if (!root) {
            error = "source roots must be paths";
            break;
          }
          roots.push_back(root->str());
        }
      }
      if (error.empty()) {
        report.setSourceRoots(roots);
        result = Object{{"ok", true}};
      }
    } else {
      result = report.request(*method, *params, error);
    }
    if (!error.empty())
      send(Object{{"id", *id}, {"error", error}});
    else
      send(Object{{"id", *id}, {"result", std::move(result)}});
  }
  return 0;
}
