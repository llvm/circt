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
  args = parser.parse_args()

  print("=== PR Gate ===", file=sys.stderr)

  repo_path = "./"
  repo = git.Repo(repo_path, search_parent_directories=True)

  def get_env_sha(env):
    ref_name = os.getenv(env)
    sha = repo.git.rev_parse(ref_name)
    print(f"{env} = {ref_name} = {sha}", file=sys.stderr)
    return sha

  head_ref = get_env_sha("GITHUB_HEAD_REF")
  base_ref = get_env_sha("GITHUB_BASE_REF")

  if base_ref == head_ref:
    print("Base and head refs are the same, skipping", file=sys.stderr)
    print("0")
    sys.exit(0)

  # Get files changed in the diff.
  command = f"git diff --name-only --diff-filter=ADMR {base_ref}..{head_ref}"
  output = subprocess.check_output(command, shell=True).decode("utf-8")
  pattern = re.compile(args.pattern)
  for file in output.splitlines():
    if re.search(pattern, file):
      print(f"File '{file}' matches pattern '{args.pattern}'", file=sys.stderr)
      print("1")
      sys.exit(0)

  print("No file in the diff matched the provided pattern", file=sys.stderr)
  print("0")
