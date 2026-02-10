#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Code generation from ESI manifests to source code.

Uses a two-pass approach for C++: first collect and name all reachable types,
then emit structs/aliases in a dependency-ordered sequence so headers are
standalone and deterministic.
"""

# C++ header support included with the runtime, though it is intended to be
# extensible for other languages.

from typing import Dict, List, Set, TextIO, Tuple, Type, Optional
from .accelerator import AcceleratorConnection, Context
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

  def __init__(self, conn: AcceleratorConnection):
    super().__init__(conn)
    self.type_planner = CppTypePlanner(self.manifest.type_table)
    self.type_emitter = CppTypeEmitter(self.type_planner)

  def get_consts_str(self, module_info: ModuleInfo) -> str:
    """Get the C++ code for a constant in a module."""
    const_strs: List[str] = [
        f"static constexpr {self.type_emitter.type_identifier(const.type)} "
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

  def generate(self, output_dir: Path, system_name: str):
    self.type_emitter.write_header(output_dir, system_name)
    self.write_modules(output_dir, system_name)


class CppTypePlanner:
  """Plan C++ type naming and ordering from an ESI manifest."""

  def __init__(self, type_table) -> None:
    """Initialize the generator with the manifest and target namespace."""
    # Map manifest type ids to their preferred C++ names.
    self.type_id_map: Dict[types.ESIType, str] = {}
    # Track all names already taken to avoid collisions. True => alias-based.
    self.used_names: Dict[str, bool] = {}
    # Track alias base names to warn on collisions.
    self.alias_base_names: Set[str] = set()
    self.ordered_types: List[types.ESIType] = []
    self.has_cycle = False
    self._prepare_types(type_table)

  def _prepare_types(self, type_table) -> None:
    """Name the types and prepare for emission by registering all reachable
    types and assigning."""
    visited: Set[str] = set()
    for t in type_table:
      self._collect_aliases(t, visited)

    visited = set()
    for t in type_table:
      self._collect_structs(t, visited)

    self.ordered_types, self.has_cycle = self._ordered_emit_types()

  def _sanitize_name(self, name: str) -> str:
    """Create a C++-safe identifier from the manifest-provided name."""
    name = name.replace("::", "_")
    if name.startswith("@"):
      name = name[1:]
    sanitized = []
    for ch in name:
      if ch.isalnum() or ch == "_":
        sanitized.append(ch)
      else:
        sanitized.append("_")
    if not sanitized:
      return "Type"
    if sanitized[0].isdigit():
      sanitized.insert(0, "_")
    return "".join(sanitized)

  def _reserve_name(self, base: str, is_alias: bool) -> str:
    """Reserve a globally unique identifier using the sanitized base name."""
    base = self._sanitize_name(base)
    if is_alias and base in self.alias_base_names:
      sys.stderr.write(
          f"Warning: duplicate alias name '{base}' detected; disambiguating.\n")
    if is_alias:
      self.alias_base_names.add(base)
    name = base
    idx = 1
    while name in self.used_names:
      name = f"{base}_{idx}"
      idx += 1
    self.used_names[name] = is_alias
    return name

  def _auto_struct_name(self, struct_type: types.StructType) -> str:
    """Derive a deterministic name for anonymous structs from their fields."""
    parts = ["_struct"]
    for field_name, field_type in struct_type.fields:
      parts.append(field_name)
      parts.append(self._sanitize_name(field_type.id))
    return self._reserve_name("_".join(parts), is_alias=False)

  def _iter_type_children(self, t: types.ESIType) -> List[types.ESIType]:
    """Return child types in a stable order for traversal."""
    if isinstance(t, types.TypeAlias):
      return [t.inner_type] if t.inner_type is not None else []
    if isinstance(t, types.BundleType):
      return [channel.type for channel in t.channels]
    if isinstance(t, types.ChannelType):
      return [t.inner]
    if isinstance(t, types.StructType):
      return [field_type for _, field_type in t.fields]
    if isinstance(t, types.ArrayType):
      return [t.element_type]
    return []

  def _visit_types(self, t: types.ESIType, visited: Set[str], visit_fn) -> None:
    """Traverse types with alphabetical child ordering in post-order."""
    if not isinstance(t, types.ESIType):
      raise TypeError(f"Expected ESIType, got {type(t)}")
    tid = t.id
    if tid in visited:
      return
    visited.add(tid)
    children = sorted(self._iter_type_children(t), key=lambda child: child.id)
    for child in children:
      self._visit_types(child, visited, visit_fn)
    visit_fn(t)

  def _collect_aliases(self, t: types.ESIType, visited: Set[str]) -> None:
    """Scan for aliases and reserve their names (recursive)."""

    # Visit callback: reserve alias names and map aliases to identifiers.
    def visit(alias_type: types.ESIType) -> None:
      if not isinstance(alias_type, types.TypeAlias):
        return
      if alias_type not in self.type_id_map:
        alias_name = self._reserve_name(alias_type.name, is_alias=True)
        self.type_id_map[alias_type] = alias_name

    self._visit_types(t, visited, visit)

  def _collect_structs(self, t: types.ESIType, visited: Set[str]) -> None:
    """Scan for structs needing auto-names and reserve them (recursive)."""

    # Visit callback: assign auto-names to unnamed structs.
    def visit(struct_type: types.ESIType) -> None:
      if not isinstance(struct_type, types.StructType):
        return
      if struct_type in self.type_id_map:
        return
      struct_name = self._auto_struct_name(struct_type)
      self.type_id_map[struct_type] = struct_name

    self._visit_types(t, visited, visit)

  def _collect_decls_from_type(self,
                               wrapped: types.ESIType) -> Set[types.ESIType]:
    """Collect types that require top-level declarations for a given type."""
    deps: Set[types.ESIType] = set()

    # Visit callback: collect structs and non-struct aliases used by a type.
    def visit(current: types.ESIType) -> None:
      if isinstance(current, types.TypeAlias):
        inner = current.inner_type
        if inner is not None and isinstance(inner, types.StructType):
          deps.add(inner)
        else:
          deps.add(current)
      elif isinstance(current, types.StructType):
        deps.add(current)

    self._visit_types(wrapped, set(), visit)
    return deps

  def _ordered_emit_types(self) -> Tuple[List[types.ESIType], bool]:
    """Collect and order types for deterministic emission."""
    emit_types: List[types.ESIType] = []
    for esi_type in self.type_id_map.keys():
      if isinstance(esi_type, (types.StructType, types.TypeAlias)):
        emit_types.append(esi_type)

    # Prefer alias-reserved names first, then lexicographic for determinism.
    name_to_type = {self.type_id_map[t]: t for t in emit_types}
    sorted_names = sorted(name_to_type.keys(),
                          key=lambda name:
                          (0 if self.used_names.get(name, False) else 1, name))

    ordered: List[types.ESIType] = []
    visited: Set[types.ESIType] = set()
    visiting: Set[types.ESIType] = set()
    has_cycle = False

    # Visit callback: DFS to emit dependencies before their users.
    def visit(current: types.ESIType) -> None:
      nonlocal has_cycle
      if current in visited:
        return
      if current in visiting:
        has_cycle = True
        return
      visiting.add(current)

      deps: Set[types.ESIType] = set()
      if isinstance(current, types.TypeAlias):
        inner = current.inner_type
        if inner is not None:
          deps.update(self._collect_decls_from_type(inner))
      elif isinstance(current, types.StructType):
        for _, field_type in current.fields:
          deps.update(self._collect_decls_from_type(field_type))
      for dep in sorted(deps, key=lambda dep: self.type_id_map[dep]):
        visit(dep)

      visiting.remove(current)
      visited.add(current)
      ordered.append(current)

    for name in sorted_names:
      visit(name_to_type[name])

    return ordered, has_cycle


class CppTypeEmitter:
  """Emit C++ headers from precomputed type ordering."""

  def __init__(self, planner: CppTypePlanner) -> None:
    self.type_id_map = planner.type_id_map
    self.ordered_types = planner.ordered_types
    self.has_cycle = planner.has_cycle

  def type_identifier(self, type: types.ESIType) -> str:
    """Get the C++ type string for an ESI type."""
    return self._cpp_type(type)

  def _get_bitvector_str(self, type: types.ESIType) -> str:
    """Get the textual code for the storage class of an integer type."""
    assert isinstance(type, (types.BitsType, types.IntType))

    if type.bit_width == 1:
      return "bool"
    elif type.bit_width <= 8:
      storage_width = 8
    elif type.bit_width <= 16:
      storage_width = 16
    elif type.bit_width <= 32:
      storage_width = 32
    elif type.bit_width <= 64:
      storage_width = 64
    else:
      raise ValueError(f"Unsupported integer width: {type.bit_width}")

    if isinstance(type, (types.BitsType, types.UIntType)):
      return f"uint{storage_width}_t"
    return f"int{storage_width}_t"

  def _array_base_and_suffix(self,
                             array_type: types.ArrayType) -> Tuple[str, str]:
    """Return the base C++ type and bracket suffix for a nested array."""
    dims: List[int] = []
    inner: types.ESIType = array_type
    while isinstance(inner, types.ArrayType):
      dims.append(inner.size)
      inner = inner.element_type
    base_cpp = self._cpp_type(inner)
    suffix = "".join([f"[{d}]" for d in dims])
    return base_cpp, suffix

  def _format_array_type(self, array_type: types.ArrayType) -> str:
    """Return the C++ type string for a nested array alias."""
    base_cpp, suffix = self._array_base_and_suffix(array_type)
    return f"{base_cpp}{suffix}"

  def _cpp_type(self, wrapped: types.ESIType) -> str:
    """Resolve an ESI type to its C++ identifier."""
    if isinstance(wrapped, (types.TypeAlias, types.StructType)):
      return self.type_id_map[wrapped]
    if isinstance(wrapped, types.BundleType):
      return "void"
    if isinstance(wrapped, types.ChannelType):
      return self._cpp_type(wrapped.inner)
    if isinstance(wrapped, types.VoidType):
      return "void"
    if isinstance(wrapped, types.AnyType):
      return "std::any"
    if isinstance(wrapped, (types.BitsType, types.IntType)):
      return self._get_bitvector_str(wrapped)
    if isinstance(wrapped, types.ArrayType):
      return self._format_array_type(wrapped)
    if type(wrapped) is types.ESIType:
      return "std::any"
    raise NotImplementedError(
        f"Type '{wrapped}' not supported for C++ generation")

  def _unwrap_aliases(self, wrapped: types.ESIType) -> types.ESIType:
    """Strip alias wrappers to reach the underlying type."""
    while isinstance(wrapped, types.TypeAlias):
      wrapped = wrapped.inner_type
    return wrapped

  def _format_array_decl(self, array_type: types.ArrayType, name: str) -> str:
    """Emit a field declaration for a multi-dimensional array member.

    The declaration flattens nested arrays into repeated bracketed sizes for C++.
    """
    base_cpp, suffix = self._array_base_and_suffix(array_type)
    return f"{base_cpp} {name}{suffix};"

  def _emit_struct(self, hdr: TextIO, struct_type: types.StructType) -> None:
    """Emit a packed struct declaration plus its type id string."""
    fields = list(struct_type.fields)
    if struct_type.cpp_type.reverse:
      fields = list(reversed(fields))
    field_decls: List[str] = []
    for field_name, field_type in fields:
      if isinstance(field_type, types.ArrayType):
        field_decls.append(self._format_array_decl(field_type, field_name))
      else:
        field_cpp = self._cpp_type(field_type)
        wrapped = self._unwrap_aliases(field_type)
        if isinstance(wrapped, (types.BitsType, types.IntType)):
          # TODO: Bitfield layout is implementation-defined; consider
          # byte-aligned storage with explicit pack/unpack helpers.
          bitfield_width = wrapped.bit_width
          if bitfield_width >= 0:
            field_decls.append(f"{field_cpp} {field_name} : {bitfield_width};")
          else:
            field_decls.append(f"{field_cpp} {field_name};")
        else:
          field_decls.append(f"{field_cpp} {field_name};")
    hdr.write(f"struct {self.type_id_map[struct_type]} {{\n")
    for decl in field_decls:
      hdr.write(f"  {decl}\n")
    hdr.write("\n")
    hdr.write(
        f"  static constexpr std::string_view _ESI_ID = \"{struct_type.id}\";\n"
    )
    hdr.write("};\n\n")

  def _emit_alias(self, hdr: TextIO, alias_type: types.TypeAlias) -> None:
    """Emit a using alias when the alias targets a different C++ type."""
    inner_wrapped = alias_type.inner_type
    alias_name = self.type_id_map[alias_type]
    inner_cpp = None
    if inner_wrapped is not None:
      inner_cpp = self._cpp_type(inner_wrapped)
    if inner_cpp is None:
      inner_cpp = self.type_id_map[alias_type]
    if inner_cpp != alias_name:
      hdr.write(f"using {alias_name} = {inner_cpp};\n\n")

  def write_header(self, output_dir: Path, system_name: str) -> None:
    """Emit the fully ordered types.h header into the output directory."""
    hdr_file = output_dir / "types.h"
    with open(hdr_file, "w") as hdr:
      hdr.write(
          textwrap.dedent(f"""
        // Generated header for {system_name} types.
        #pragma once

        #include <cstdint>
        #include <any>
        #include <string_view>

        namespace {system_name} {{
        #pragma pack(push, 1)

      """))
      if self.has_cycle:
        sys.stderr.write("Warning: cyclic type dependencies detected.\n")
        sys.stderr.write("  Logically this should not be possible.\n")
        sys.stderr.write(
            "  Emitted code may fail to compile due to ordering issues.\n")

      for emit_type in self.ordered_types:
        if isinstance(emit_type, types.StructType):
          self._emit_struct(hdr, emit_type)
        elif isinstance(emit_type, types.TypeAlias):
          self._emit_alias(hdr, emit_type)

      hdr.write(
          textwrap.dedent(f"""
    #pragma pack(pop)
    }} // namespace {system_name}
    """))


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
    conn = Context.default().connect("trace", f"-:{args.file}")
  elif args.platform is not None:
    if args.connection is None:
      print("Must specify --connection with --platform")
      return 1
    conn = Context.default().connect(args.platform, args.connection)
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
