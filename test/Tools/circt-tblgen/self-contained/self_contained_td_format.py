#!/usr/bin/env python3
"""Lit test format for verifying TableGen files are self-contained.

Discovers .td files from configured source directories and creates one lit
test per file. Each test runs circt-tblgen --print-records on a single .td
file to verify it parses without errors when processed in isolation. This
means no missing includes and no unresolved references.
"""

import os
import subprocess
import lit.formats
import lit.Test


class SelfContainedTDFormat(lit.formats.TestFormat):
  """Lit test format that checks TableGen files are self-contained."""

  def __init__(self, circt_tblgen):
    super().__init__()
    self._circt_tblgen = circt_tblgen
    self._dir_specs = []
    self._seen = set()

  def add_dir_spec(self, tablegen_includes, td_dirs):
    """Register a group of include paths and source directories."""
    cmd_prefix = [self._circt_tblgen, "--print-records"]
    for inc in tablegen_includes:
      cmd_prefix += ["-I", inc]
    self._dir_specs.append((cmd_prefix, td_dirs))

  def _walk_td_files(self):
    """Yield (absolute_path, cmd_prefix) for each .td file."""
    for cmd_prefix, td_dirs in self._dir_specs:
      for root in td_dirs:
        if not os.path.isdir(root):
          continue
        for dirpath, dirnames, filenames in os.walk(root):
          dirnames[:] = [d for d in dirnames if not d.startswith(".")]
          for f in sorted(filenames):
            if f.endswith(".td"):
              full = os.path.join(dirpath, f)
              if full not in self._seen:
                self._seen.add(full)
                yield full, cmd_prefix

  def getTestsInDirectory(self, tests, path_in_suite, litConfig, localConfig):
    circt_src_root = localConfig.circt_src_root
    for td_file, cmd_prefix in self._walk_td_files():
      rel_path = os.path.relpath(td_file, circt_src_root)
      test = lit.Test.Test(tests, path_in_suite + tuple(rel_path.split(os.sep)),
                           localConfig)
      test.td_file = td_file
      test.cmd_prefix = cmd_prefix
      yield test

  def execute(self, test, litConfig):
    if litConfig.noExecute:
      return (lit.Test.PASS, "")

    td_file = test.td_file
    cmd = [*test.cmd_prefix, td_file]
    timeout = test.config.maxIndividualTestTime or None
    env = test.config.environment
    try:
      result = subprocess.run(cmd,
                              stderr=subprocess.PIPE,
                              stdout=subprocess.DEVNULL,
                              text=True,
                              timeout=timeout,
                              env=env)
      if result.returncode == 0:
        return (lit.Test.PASS, "")
      return (
          lit.Test.FAIL,
          f"{td_file} is not self-contained\n" + (result.stderr or ""),
      )
    except subprocess.TimeoutExpired:
      return (lit.Test.TIMEOUT, f"checking {td_file} timed out")
    except Exception as exc:
      return (
          lit.Test.UNRESOLVED,
          f"could not run circt-tblgen on {td_file}: {exc}",
      )
