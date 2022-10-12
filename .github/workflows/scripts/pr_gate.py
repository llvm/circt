import sys
import subprocess
import argparse
import re
import os
import git

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=
      """Check if any of the files in the current diff match the provided pattern.
     If so, prints 1, else prints 0.""")
  parser.add_argument("pattern", help="Pattern to match", type=str)
  parser.add_argument("base",
                      help="SHA of the base commit to compare against",
                      type=str)
  args = parser.parse_args()
  repo_path = "./"
  repo = git.Repo(repo_path, search_parent_directories=True)

  # Get files changed in the diff.
  command = f"git diff --name-only --diff-filter=ADMR {args.base}..HEAD"
  output = subprocess.check_output(command, shell=True).decode("utf-8")
  pattern = re.compile(args.pattern)
  for file in output.splitlines():
    if re.search(pattern, file):
      print(f"File '{file}' matches pattern '{args.pattern}'", file=sys.stderr)
      print("1")
      sys.exit(0)

  print("No file in the diff matched the provided pattern", file=sys.stderr)
  print("0")
