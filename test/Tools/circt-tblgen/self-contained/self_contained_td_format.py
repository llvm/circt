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

  def __init__(self, circt_tblgen, tablegen_includes, td_dirs, timeout):
    super().__init__()
    self.td_dirs = td_dirs
    # Build the fixed part of the command.
    self._cmd_prefix = [circt_tblgen, "--print-records"]
    for inc in tablegen_includes:
      self._cmd_prefix += ["-I", inc]
    self.timeout = timeout
    self._seen = set()

  def _walk_td_files(self):
    """Yield (absolute_path, rel_path_from_circt_root) for each .td file."""
    for root in self.td_dirs:
      if not os.path.isdir(root):
        continue
      for dirpath, dirnames, filenames in os.walk(root):
        # Prevent walking into hidden directories, llvm directories,
        # and build* directories.
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and
            not d.startswith("build") and d != "llvm"
        ]
        for f in sorted(filenames):
          if f.endswith(".td"):
            full = os.path.join(dirpath, f)
            if full not in self._seen:
              self._seen.add(full)
              yield full

  def getTestsInDirectory(self, tests, path_in_suite, litConfig, localConfig):
    for td_file in self._walk_td_files():
      yield lit.Test.Test(tests, (td_file,), localConfig)

  def execute(self, test, litConfig):
    td_file = test.getSourcePath()
    cmd = [*self._cmd_prefix, td_file]
    try:
      result = subprocess.run(cmd,
                              capture_output=True,
                              text=True,
                              timeout=self.timeout)
      if result.returncode == 0:
        return (lit.Test.PASS, "")
      return (
          lit.Test.FAIL,
          f"{td_file} is not self-contained\n" + (result.stderr or ""),
      )
    except subprocess.TimeoutExpired:
      return (
          lit.Test.FAIL,
          f"checking {td_file} took longer than {self.timeout} seconds",
      )
    except Exception as exc:
      return (
          lit.Test.FAIL,
          f"could not run circt-tblgen on {td_file}: {exc}",
      )
