// Shared probe-runner framework for ESI integration tests. Each test binary
// defines a set of named probe functions and registers them via
// ESI_PROBE_REGISTRY. The runner provides:
//
//   * ``--probe NAME`` to run a single probe (used by pytest for isolation)
//   * ``--probe all`` or no ``--probe`` to run every probe in order
//   * Connection setup, manifest loading, and error handling boilerplate
//
// Usage:
//
//   static int myProbe(esi::Accelerator *accel) { ... return 0; }
//
//   ESI_PROBE_REGISTRY("my_binary", "Description.",
//     {"probe_a", &myProbe},
//     {"probe_b", &otherProbe},
//   );

#pragma once

#include "esi/Accelerator.h"
#include "esi/CLI.h"
#include "esi/Manifest.h"
#include "esi/Services.h"

#ifdef _MSC_VER
#include <crtdbg.h>
#include <cstdlib>
#endif
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace esi_test {

using ProbeFn = int (*)(esi::Accelerator *);
using ProbeEntry = std::pair<std::string, ProbeFn>;

inline void configureMsvcDebugReports() {
#if defined(_MSC_VER) && defined(_DEBUG)
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
}

/// Run the probe-runner main loop. Returns 0 on success, nonzero on failure.
inline int runProbes(int argc, const char *argv[], const char *name,
                     const char *description,
                     const std::vector<ProbeEntry> &probes) {
  esi::CliParser cli(name);
  cli.description(description);
  std::string probeName;
  cli.add_option("--probe", probeName,
                 "Run only the named probe. Without this flag, every probe "
                 "runs in sequence.");
  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  esi::Context &ctxt = cli.getContext();
  esi::AcceleratorConnection *conn = cli.connect();
  try {
    const auto &info = *conn->getService<esi::services::SysInfo>();
    esi::Manifest manifest(ctxt, info.getJsonManifest());
    esi::Accelerator *accel = manifest.buildAccelerator(*conn);
    conn->getServiceThread()->addPoll(*accel);

    int rc = 0;
    if (probeName.empty()) {
      for (const auto &[pname, fn] : probes) {
        rc = fn(accel);
        if (rc)
          break;
      }
    } else {
      ProbeFn fn = nullptr;
      for (const auto &[pname, candidate] : probes) {
        if (pname == probeName) {
          fn = candidate;
          break;
        }
      }
      if (!fn) {
        std::cerr << name << ": unknown probe '" << probeName << "'\n";
        conn->disconnect();
        return 2;
      }
      rc = fn(accel);
    }

    conn->disconnect();
    return rc;
  } catch (std::exception &e) {
    ctxt.getLogger().error(name, e.what());
    conn->disconnect();
    return 1;
  }
}

} // namespace esi_test

/// Convenience macro: defines ``main()`` with a probe registry.
/// The variadic arguments are ``{"name", &fn}`` pairs.
#define ESI_PROBE_REGISTRY(name, description, ...)                             \
  int main(int argc, const char *argv[]) {                                     \
    esi_test::configureMsvcDebugReports();                                     \
    static const std::vector<esi_test::ProbeEntry> kProbes = {__VA_ARGS__};    \
    return esi_test::runProbes(argc, argv, name, description, kProbes);        \
  }
