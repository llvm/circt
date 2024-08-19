#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Code generation from ESI manifests to source code. C++ header support included
# with the runtime, though it is intended to be extensible for other languages.

from typing import List, TextIO, Type, Optional
from .accelerator import AcceleratorConnection
from .esiCppAccel import ModuleInfo
from . import types

import argparse
from pathlib import Path
import textwrap
import sys

_thisdir = Path(__file__).absolute().resolve().parent


class Generator:
  """Base class for all generators."""

  language: Optional[str] = None

  def __init__(self, conn: AcceleratorConnection):
    self.manifest = conn.manifest()

  def generate(self, output_dir: Path, system_name: str):
    raise NotImplementedError("Generator.generate() must be overridden")


class CppGenerator(Generator):
  """Generate C++ headers from an ESI manifest."""

  language = "C++"

  # Supported bit widths for lone integer types.
  int_width_support = set([8, 16, 32, 64])

  def get_type_str(self, type: types.ESIType) -> str:
    """Get the textual code for the storage class of a type.

    Examples: uint32_t, int64_t, CustomStruct."""

    if isinstance(type, (types.BitsType, types.IntType)):
      if type.bit_width not in self.int_width_support:
        raise ValueError(f"Unsupported integer width: {type.bit_width}")
      if isinstance(type, (types.BitsType, types.UIntType)):
        return f"uint{type.bit_width}_t"
      return f"int{type.bit_width}_t"
    raise NotImplementedError(f"Type '{type}' not supported for C++ generation")

  def get_consts_str(self, module_info: ModuleInfo) -> str:
    """Get the C++ code for a constant in a module."""
    const_strs: List[str] = [
        f"static constexpr {self.get_type_str(const.type)} "
        f"{name} = 0x{const.value:x};"
        for name, const in module_info.constants.items()
    ]
    return "\n".join(const_strs)

  def write_modules(self, output_dir: Path, system_name: str):
    """Write the C++ header. One for each module in the manifest."""

    for module_info in self.manifest.module_infos:
      s = f"""
      /// Generated header for {system_name} module {module_info.name}.
      #pragma once
      #include "types.h"

      namespace {system_name} {{
      class {module_info.name} {{
      public:
        {self.get_consts_str(module_info)}
      }};
      }} // namespace {system_name}
      """

      hdr_file = output_dir / f"{module_info.name}.h"
      with open(hdr_file, "w") as hdr:
        hdr.write(textwrap.dedent(s))

  def write_type(self, hdr: TextIO, type: types.ESIType):
    if isinstance(type, (types.BitsType, types.IntType)):
      # Bit vector types use standard C++ types.
      return
    raise NotImplementedError(f"Type '{type}' not supported for C++ generation")

  def write_types(self, output_dir: Path, system_name: str):
    hdr_file = output_dir / "types.h"
    with open(hdr_file, "w") as hdr:
      hdr.write(
          textwrap.dedent(f"""
      // Generated header for {system_name} types.
      #pragma once

      #include <cstdint>

      namespace {system_name} {{
      """))

      for type in self.manifest.type_table:
        try:
          self.write_type(hdr, type)
        except NotImplementedError:
          sys.stderr.write(
              f"Warning: type '{type}' not supported for C++ generation\n")

      hdr.write(
          textwrap.dedent(f"""
      }} // namespace {system_name}
      """))

  def generate(self, output_dir: Path, system_name: str):
    self.write_types(output_dir, system_name)
    self.write_modules(output_dir, system_name)


def run(generator: Type[Generator] = CppGenerator,
        cmdline_args=sys.argv) -> int:
  """Create and run a generator reading options from the command line."""

  argparser = argparse.ArgumentParser(
      description=f"Generate {generator.language} headers from an ESI manifest",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent("""
        Can read the manifest from either a file OR a running accelerator.

        Usage examples:
          # To read the manifest from a file:
          esi-cppgen --file /path/to/manifest.json

          # To read the manifest from a running accelerator:
          esi-cppgen --platform cosim --connection localhost:1234
      """))

  argparser.add_argument("--file",
                         type=str,
                         default=None,
                         help="Path to the manifest file.")
  argparser.add_argument(
      "--platform",
      type=str,
      help="Name of platform for live accelerator connection.")
  argparser.add_argument(
      "--connection",
      type=str,
      help="Connection string for live accelerator connection.")
  argparser.add_argument(
      "--output-dir",
      type=str,
      default="esi",
      help="Output directory for generated files. Recommend adding either `esi`"
      " or the system name to the end of the path so as to avoid header name"
      "conflicts. Defaults to `esi`")
  argparser.add_argument(
      "--system-name",
      type=str,
      default="esi_system",
      help="Name of the ESI system. For C++, this will be the namespace.")

  if (len(cmdline_args) <= 1):
    argparser.print_help()
    return 1
  args = argparser.parse_args(cmdline_args[1:])

  if args.file is not None and args.platform is not None:
    print("Cannot specify both --file and --platform")
    return 1

  conn: AcceleratorConnection
  if args.file is not None:
    conn = AcceleratorConnection("trace", f"-:{args.file}")
  elif args.platform is not None:
    if args.connection is None:
      print("Must specify --connection with --platform")
      return 1
    conn = AcceleratorConnection(args.platform, args.connection)
  else:
    print("Must specify either --file or --platform")
    return 1

  output_dir = Path(args.output_dir)
  if output_dir.exists() and not output_dir.is_dir():
    print(f"Output directory {output_dir} is not a directory")
    return 1
  if not output_dir.exists():
    output_dir.mkdir(parents=True)

  gen = generator(conn)
  gen.generate(output_dir, args.system_name)
  return 0


if __name__ == '__main__':
  sys.exit(run())
