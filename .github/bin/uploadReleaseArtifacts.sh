#!/usr/bin/env bash

set -eo pipefail

usage() {
  cat << EOF
USAGE: $0 -o <os> -e <github-event> -b <build-type> [-t -a]

Generate a JSON payload to drive unifiedBuildAndTest* scripts.

OPTIONS:
    -a                    Enable assertions
    -b                    The CMAKE build type: [release, relwithdebinfo, debug]
    -e                    GitHub event name: [workflow_dispatch, schedule]
    -h                    Display available options
    -o                    OS to target: [linux, macos, windows]
    -t                    Run tests
EOF
}

#-------------------------------------------------------------------------------
# Option parsing
#-------------------------------------------------------------------------------

OPT_ASSERTIONS=OFF
OPT_CMAKE_BUILD_TYPE=
OPT_GITHUB_EVENT_NAME=
OPT_OS=()
OPT_RUN_TESTS=false
while getopts "ab:e:ho:t" option; do
  case $option in
    a)
      OPT_ASSERTIONS=ON
      ;;
    b)
      case "$OPTARG" in
        "release" | "relwithdebinfo" | "debug")
          OPT_CMAKE_BUILD_TYPE=$OPTARG
        ;;
        *)
          echo "unknown argument '$OPTARG' for '-b', must be one of ['release', 'relwithdebinfo', 'debug']"
          exit 1
          ;;
      esac
      ;;
    e)
      case "$OPTARG" in
        "release" | "schedule" | "workflow_dispatch")
          OPT_GITHUB_EVENT_NAME=$OPTARG
          ;;
        *)
          echo "unsupported GitHub event '$OPTARG', must be one of ['release', 'schedule', 'workflow_dispatch']"
          exit 1
          ;;
      esac
      ;;
    h)
      usage
      exit 0
      ;;
    o)
      case "$OPTARG" in
        "linux" | "macos" | "windows")
          OPT_OS+=("$OPTARG")
          ;;
        *)
          echo "unknown argument '$OPTARG' for '-o', must be one of ['linux', 'macos', 'windows']"
          exit 1
          ;;
      esac
      ;;
    t)
      OPT_RUN_TESTS=true
      ;;
    *)
      echo "uknown option"
      usage
      exit 1
      ;;
  esac
done

#-------------------------------------------------------------------------------
# Option validation, default handling
#-------------------------------------------------------------------------------

# A GitHub event name is mandatory.
if [[ ! $OPT_GITHUB_EVENT_NAME ]] ; then
  echo "missing mandatory '-e <github-event>' option"
  usage
  exit 1
fi

# Certain GitHub events have pre-programmed options.  For "release", these are
# the settings used for the artifacts we will publish and intend for every
# downstream project to use.  For "schedule", these artifacts will be used by
# nightly testing.  For "workflow_dispatch", everything is in the user's
# control and _not_ defaults are set.
case "$OPT_GITHUB_EVENT_NAME" in
  "release")
    OPT_ASSERTIONS=OFF
    OPT_CMAKE_BUILD_TYPE=release
    OPT_OS=("linux" "macos" "windows")
    OPT_RUN_TESTS=true
    ;;
  "schedule")
    OPT_ASSERTIONS=ON
    OPT_CMAKE_BUILD_TYPE=release
    OPT_OS=("linux")
    ;;
  "workflow_dispatch")
    ;;
esac

# After defaults have been applied, do a final check of options.
if [[ ! $OPT_CMAKE_BUILD_TYPE ]]; then
  echo "missing mandatory '-b <build-type>' option"
  usage
  exit 1
fi
if [[ "${#OPT_OS[@]}" == 0 ]] ; then
  echo "at least one '-o <os>' options must be specified"
  usage
  exit 1
fi

# Unique the OS arguments.
mapfile -t OPT_OS < <(echo "${OPT_OS[@]}" | tr ' ' '\n' | sort -u)

#-------------------------------------------------------------------------------
# JSON snippets used to configure downstream workflows
#-------------------------------------------------------------------------------

# For non-full installs, install the following tools.
binaries=(
    circt-opt
    circt-reduce
    circt-synth
    circt-test
    circt-translate
    circt-verilog
    firtool
    om-linker
)
binaryTargets=
for bin in "${binaries[@]}"; do
  binaryTargets="$binaryTargets install-$bin"
done

# Configuration for a run of the static UBTI script.
configStatic=$(cat <<EOF
[
  {
    "cmake_build_type": "$OPT_CMAKE_BUILD_TYPE",
    "llvm_enable_assertions": "$OPT_ASSERTIONS",
    "llvm_force_enable_stats": "ON",
    "run_tests": $OPT_RUN_TESTS,
    "install_target": "$binaryTargets",
    "package_name_prefix": "firrtl-bin"
  }
]
EOF
)

# Configuration snippets for a run of the native runner UBTI script.
configLinuxRunner=$(cat <<EOF
[
  {
    "runner": "ubuntu-24.04",
    "cmake_c_compiler": "clang",
    "cmake_cxx_compiler": "clang++"
  }
]
EOF
)
configMacOsRunner=$(cat <<EOF
[
  {
    "runner": "macos-13",
    "cmake_c_compiler": "clang",
    "cmake_cxx_compiler": "clang++"
  }
]
EOF
)
configWindowsRunner=$(cat <<EOF
[
  {
    "runner": "windows-2022",
    "cmake_c_compiler": "cl",
    "cmake_cxx_compiler": "cl"
  }
]
EOF
)

# Configuration snippets for building something on the native UBTI workflow.
configNativeFullShared=$(cat <<EOF
[
  {
    "name":"CIRCT-full shared",
    "install_target":"install",
    "package_name_prefix":"circt-full-shared",
    "cmake_build_type":"$OPT_CMAKE_BUILD_TYPE",
    "llvm_enable_assertions":"$OPT_ASSERTIONS",
    "build_shared_libs":"ON",
    "llvm_force_enable_stats":"ON",
    "run_tests": $OPT_RUN_TESTS
  }
]
EOF
)
configNativeFullStatic=$(cat <<EOF
[
  {
    "name":"CIRCT-full static",
    "install_target":"install",
    "package_name_prefix":"circt-full-static",
    "cmake_build_type":"$OPT_CMAKE_BUILD_TYPE",
    "llvm_enable_assertions":"$OPT_ASSERTIONS",
    "build_shared_libs":"OFF",
    "llvm_force_enable_stats":"ON",
    "run_tests": $OPT_RUN_TESTS
  }
]
EOF
)
configNativeFirtool=$(cat <<EOF
[
  {
    "name": "firtool",
    "install_target": "$binaryTargets",
    "package_name_prefix": "firrtl-bin",
    "cmake_build_type":"$OPT_CMAKE_BUILD_TYPE",
    "llvm_enable_assertions":"$OPT_ASSERTIONS",
    "build_shared_libs":"OFF",
    "llvm_force_enable_stats":"ON",
    "run_tests": $OPT_RUN_TESTS
  }
]
EOF
)

#-------------------------------------------------------------------------------
# Build the JSON payload the will be used to configure UBTI and UBTI-static.
#-------------------------------------------------------------------------------

# This is the shape of the JSON payload.  This is a dictionary with two keys:
# static and native.  Static will be used to configure the UBTI-static workflow.
# Native will be used to configure the UBTI workflow.
config=$(cat <<EOF
{
  "static": [],
  "native": []
}
EOF
)
for os in "${OPT_OS[@]}"; do
  case "$os" in
    # Linux gets:
    #   1. Static firtool binary
    #   2. Native full shared
    #   3. Native full static
    "linux")
      # Static:
      config=$(echo "$config" | jq '.static += $a' --argjson a "$configStatic")
      # Native:
      native=$(echo "$configNativeFullShared" "$configNativeFullStatic" | jq -s 'add')
      native=$(echo "$native" "$configLinuxRunner" | jq -s '[combinations | add]')
      config=$(echo "$config" | jq '.native += $a' --argjson a "$native")
      ;;
    # MacOS gets:
    #   1. Native firtool binary
    #   2. Native full shared
    #   3. Native full static
    "macos")
      native=$(echo "$configNativeFullShared" "$configNativeFullStatic" "$configNativeFirtool" | jq -s 'add')
      native=$(echo "$native" "$configMacOsRunner" | jq -s '[combinations | add]')
      config=$(echo "$config" | jq '.native += $a' --argjson a "$native")
      ;;
    # Windows gets:
    #   1. Native firtool binary
    #   2. Native full static
    #
    # Note: Windows cannot handle full shared.
    "windows")
      native=$(echo "$configNativeFullStatic" "$configNativeFirtool" | jq -s 'add')
      native=$(echo "$native" "$configWindowsRunner" | jq -s '[combinations | add]')
      config=$(echo "$config" | jq '.native += $a' --argjson a "$native")
      ;;
    *)
      echo "unknown os '$os'"
      exit 1
      ;;
  esac
done

# Return the final `config` JSON.
echo "$config"
