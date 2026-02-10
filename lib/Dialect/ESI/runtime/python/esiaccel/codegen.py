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
    self.type_generator = CppTypeGenerator(self.manifest.type_table)

  def get_consts_str(self, module_info: ModuleInfo) -> str:
    """Get the C++ code for a constant in a module."""
    const_strs: List[str] = [
        f"static constexpr {self.type_generator.type_identifier(const.type)} "
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
    self.type_generator.write_header(output_dir, system_name)
    self.write_modules(output_dir, system_name)


class CppTypeGenerator:
  """Produce a deterministic types.h from an ESI manifest."""

  def __init__(self, type_table) -> None:
    """Initialize the generator with the manifest and target namespace."""
    self.type_by_id: Dict[str, types.ESIType] = {}
    # Map manifest type ids to their preferred C++ names.
    self.type_id_map: Dict[str, str] = {}
    # Reserve unique C++ names for type aliases.
    self.alias_name_by_id: Dict[str, str] = {}
    # Reserve unique C++ names for struct types.
    self.struct_name_by_id: Dict[str, str] = {}
    # Track all names already taken to avoid collisions.
    self.used_names: Set[str] = set()
    # Track structs that were named via alias to keep emission order stable.
    self.aliased_structs: Set[str] = set()
    # Monotonic counter for insertion order during emission ordering.
    self.emit_counter = 0
    self._precompute(type_table)

  def _precompute(self, type_table) -> None:
    for t in type_table:
      self._register_type(t)
    self._prepare_aliases()
    self._assign_struct_names()

    # Ensure every manifest type gets a mapped C++ identifier.
    for esi_type in type_table:
      self._assign_single_type_name(esi_type)

  def type_identifier(self, type: types.ESIType) -> str:
    """Get the C++ type string for an ESI type."""
    return self.type_id_map[type.id]

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

  def _reserve_name(self, base: str) -> str:
    """Reserve a globally unique identifier using the sanitized base name."""
    base = self._sanitize_name(base)
    name = base
    idx = 1
    while name in self.used_names:
      name = f"{base}_{idx}"
      idx += 1
    self.used_names.add(name)
    return name

  def _auto_struct_name(self, struct_type: types.StructType) -> str:
    """Derive a deterministic name for anonymous structs from their fields."""
    parts = ["_struct"]
    for field_name, field_type in struct_type.fields:
      parts.append(field_name)
      parts.append(self._sanitize_name(field_type.id))
    return self._reserve_name("_".join(parts))

  def _get_alias_info(
      self,
      wrapped: types.ESIType) -> Optional[Tuple[str, Optional[types.ESIType]]]:
    """Return alias metadata (name and inner type) if the type is a TypeAlias."""
    if isinstance(wrapped, types.TypeAlias):
      return (wrapped.name, wrapped.inner_type)
    return None

  def _is_opaque_any(self, wrapped: types.ESIType) -> bool:
    """Detect legacy opaque types that should be treated as std::any."""
    return wrapped.id == "!esi.any"

  def _register_type(self, t: object) -> None:
    """Collect reachable types by recursively exploring bundles, channels, and
    fields."""
    if not isinstance(t, types.ESIType):
      raise TypeError(f"Expected ESIType, got {type(t)}")
    # Depth-first traversal ensures all referenced types are registered.
    wrapped = t
    tid = wrapped.id
    if tid in self.type_by_id:
      return
    self.type_by_id[tid] = wrapped

    if isinstance(wrapped, types.BundleType):
      for channel in wrapped.channels:
        self._register_type(channel.type)
      return

    if isinstance(wrapped, types.ChannelType):
      self._register_type(wrapped.inner)
      return

    if isinstance(wrapped, types.StructType):
      for _, field_type in wrapped.fields:
        self._register_type(field_type)
      return

    if isinstance(wrapped, types.ArrayType):
      self._register_type(wrapped.element_type)
      return

  def _prepare_aliases(self) -> None:
    """Map raw alias identifiers to unique C++ names and remember aliased structs.

    The second pass records which structs adopted their alias name so we can
    emit them immediately after the alias for readability.
    """
    for tid, t in self.type_by_id.items():
      alias_info = self._get_alias_info(t)
      if alias_info is None:
        continue
      alias_raw, inner_wrapped = alias_info
      alias_name = self.alias_name_by_id.get(tid)
      if alias_name is None:
        alias_name = self._reserve_name(alias_raw)
        self.alias_name_by_id[tid] = alias_name
      if inner_wrapped is not None and isinstance(inner_wrapped,
                                                  types.StructType):
        self.struct_name_by_id.setdefault(inner_wrapped.id, alias_name)
    # Keep track of structs that gained their name via an alias so we can
    # emit them immediately after the alias that owns their name.
    for tid, t in self.type_by_id.items():
      alias_info = self._get_alias_info(t)
      if alias_info is None:
        continue
      _, inner_wrapped = alias_info
      if inner_wrapped is not None and isinstance(inner_wrapped,
                                                  types.StructType):
        self.aliased_structs.add(inner_wrapped.id)

  def _assign_struct_names(self) -> None:
    """Generate names for any structs that were not named via aliases.

    Anonymous structs are labeled using their field names so the selector is
    deterministic across runs.
    """
    for tid, t in self.type_by_id.items():
      if isinstance(t, types.StructType) and tid not in self.struct_name_by_id:
        self.struct_name_by_id[tid] = self._auto_struct_name(t)

  def _get_bitvector_str(self, type: types.ESIType) -> str:
    """Get the textual code for the storage class of a type.

    Examples: uint32_t, int64_t, CustomStruct."""

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

  def _assign_single_type_name(self, t: object) -> None:
    """Recursively assign a C++ type name for a single manifest entry.

    Alias types propagate their chosen name down to structs, while arrays and
    bundles call down to their element types before assigning the composite
    string.
    """
    if not isinstance(t, types.ESIType):
      raise TypeError(f"Expected ESIType, got {type(t)}")
    wrapped = t
    tid = wrapped.id
    if tid in self.type_id_map:
      return
    # Aliases define preferred names and may propagate to structs.
    alias_info = self._get_alias_info(wrapped)
    if alias_info is not None:
      alias_name, inner_wrapped = alias_info
      if tid not in self.alias_name_by_id:
        self.alias_name_by_id[tid] = self._reserve_name(alias_name)
      alias_name = self.alias_name_by_id[tid]
      if inner_wrapped is not None:
        if isinstance(inner_wrapped, types.StructType):
          inner_tid = inner_wrapped.id
          self.struct_name_by_id[inner_tid] = alias_name
          self.type_id_map[inner_tid] = alias_name
          self.type_id_map[tid] = alias_name
          for _, field_type in inner_wrapped.fields:
            self._assign_single_type_name(field_type)
          return
        self._assign_single_type_name(inner_wrapped)
      self.type_id_map[tid] = alias_name
      return
    # Bundles/channels forward to their inner payload types.
    if isinstance(wrapped, types.BundleType):
      for channel in wrapped.channels:
        self._assign_single_type_name(channel.type)
      self.type_id_map[tid] = "void"
      return

    if isinstance(wrapped, types.ChannelType):
      self._assign_single_type_name(wrapped.inner)
      self.type_id_map[tid] = self.type_id_map[wrapped.inner.id]
      return
    # Scalar, array, and struct cases map directly to concrete C++ types.
    if isinstance(wrapped, types.VoidType):
      self.type_id_map[tid] = "void"
      return

    if isinstance(wrapped, types.AnyType):
      self.type_id_map[tid] = "std::any"
      return

    if self._is_opaque_any(wrapped):
      self.type_id_map[tid] = "std::any"
      return

    if isinstance(wrapped, (types.BitsType, types.IntType)):
      self.type_id_map[tid] = self._get_bitvector_str(wrapped)
      return

    if isinstance(wrapped, types.ArrayType):
      self._assign_single_type_name(wrapped.element_type)
      self.type_id_map[tid] = self._format_array_type(wrapped)
      return

    if isinstance(wrapped, types.StructType):
      for _, field_type in wrapped.fields:
        self._assign_single_type_name(field_type)
      self.type_id_map[tid] = self.struct_name_by_id[tid]
      return

    if type(wrapped) is types.ESIType:
      self.type_id_map[tid] = "std::any"
      return
    raise NotImplementedError(
        f"Type '{wrapped}' not supported for C++ generation")

  def _format_array_decl(self, array_type: types.ArrayType, name: str) -> str:
    """Emit a field declaration for a multi-dimensional array member.

    The declaration flattens nested arrays into repeated bracketed sizes for C++.
    """
    dims: List[int] = []
    inner: types.ESIType = array_type
    while isinstance(inner, types.ArrayType):
      dims.append(inner.size)
      inner = inner.element_type
    base_cpp = self.type_id_map[inner.id]
    suffix = "".join([f"[{d}]" for d in dims])
    return f"{base_cpp} {name}{suffix};"

  def _format_array_type(self, array_type: types.ArrayType) -> str:
    """Return the C++ type string for a nested array alias.

    The suffix mirrors the multi-dimensional declaration produced for fields.
    """
    dims: List[int] = []
    inner: types.ESIType = array_type
    while isinstance(inner, types.ArrayType):
      dims.append(inner.size)
      inner = inner.element_type
    base_cpp = self.type_id_map[inner.id]
    suffix = "".join([f"[{d}]" for d in dims])
    return f"{base_cpp}{suffix}"

  def _dep_nodes_for_type(self, wrapped: types.ESIType) -> Set[Tuple[str, str]]:
    """Return the dependency nodes required before emitting this type.

    Dependencies help the topological sort emit structs and aliases in order.
    """
    if isinstance(wrapped, types.StructType):
      return {("struct", wrapped.id)}
    if isinstance(wrapped, types.TypeAlias):
      inner = wrapped.inner_type
      if isinstance(inner, types.StructType):
        return {("struct", inner.id)}
      return {("alias", wrapped.id)}
    if isinstance(wrapped, types.ArrayType):
      return self._dep_nodes_for_type(wrapped.element_type)
    return set()

  def _add_node(self, kind: str, wrapped: types.ESIType,
                emit_nodes: Dict[Tuple[str, str], types.ESIType],
                emit_deps: Dict[Tuple[str, str], Set[Tuple[str, str]]],
                emit_index: Dict[Tuple[str, str], int]) -> Tuple[str, str]:
    """Register a struct or alias for emission ordering.

    The insertion index keeps the order deterministic when dependencies tie.
    """
    tid = wrapped.id
    key = (kind, tid)
    if key not in emit_nodes:
      emit_nodes[key] = wrapped
      emit_deps[key] = set()
      emit_index[key] = self.emit_counter
      self.emit_counter += 1
    return key

  def _visit_type(self, t: object, emit_nodes: Dict[Tuple[str, str],
                                                    types.ESIType],
                  emit_deps: Dict[Tuple[str, str], Set[Tuple[str, str]]],
                  emit_index: Dict[Tuple[str, str], int]) -> None:
    """Walk the type structure to collect emit nodes and their dependencies.

    Bundles and channels unwrap into their inner types while structs gather
    per-field edges needed later during emission.
    """
    if not isinstance(t, types.ESIType):
      raise TypeError(f"Expected ESIType, got {type(t)}")
    wrapped = t
    tid = wrapped.id
    # Alias nodes must appear before anything that references them.
    alias_info = self._get_alias_info(wrapped)
    if alias_info is not None:
      _, inner_wrapped = alias_info
      if inner_wrapped is not None and isinstance(inner_wrapped,
                                                  types.StructType):
        self._visit_type(inner_wrapped, emit_nodes, emit_deps, emit_index)
        return
      if inner_wrapped is not None:
        self._visit_type(inner_wrapped, emit_nodes, emit_deps, emit_index)
      key = self._add_node("alias", wrapped, emit_nodes, emit_deps, emit_index)
      if inner_wrapped is not None:
        emit_deps[key].update(self._dep_nodes_for_type(inner_wrapped))
      return
    # Unwrap bundles/channels to their element types.
    if isinstance(wrapped, types.BundleType):
      for channel in wrapped.channels:
        self._visit_type(channel.type, emit_nodes, emit_deps, emit_index)
      return

    if isinstance(wrapped, types.ChannelType):
      self._visit_type(wrapped.inner, emit_nodes, emit_deps, emit_index)
      return
    # Struct fields introduce dependencies on their member types.
    if isinstance(wrapped, types.StructType):
      key = self._add_node("struct", wrapped, emit_nodes, emit_deps, emit_index)
      for _, field_type in wrapped.fields:
        self._visit_type(field_type, emit_nodes, emit_deps, emit_index)
        emit_deps[key].update(self._dep_nodes_for_type(field_type))
      emit_deps[key].discard(key)
      return

    if isinstance(wrapped, types.ArrayType):
      self._visit_type(wrapped.element_type, emit_nodes, emit_deps, emit_index)
      return

    if self._is_opaque_any(wrapped):
      return

    if isinstance(wrapped, types.AnyType):
      return

  def _emit_struct(self, hdr: TextIO, struct_type: types.StructType) -> None:
    """Emit a packed struct declaration plus its type id string."""
    tid = struct_type.id
    fields = list(struct_type.fields)
    if struct_type.cpp_type.reverse:
      fields = list(reversed(fields))
    field_decls: List[str] = []
    for field_name, field_type in fields:
      if isinstance(field_type, types.ArrayType):
        field_decls.append(self._format_array_decl(field_type, field_name))
      else:
        field_cpp = self.type_id_map[field_type.id]
        field_decls.append(f"{field_cpp} {field_name};")
    hdr.write(f"struct {self.type_id_map[tid]} {{\n")
    for decl in field_decls:
      hdr.write(f"  {decl}\n")
    hdr.write("\n")
    hdr.write(
        f"  static constexpr std::string_view __ESI_ID = \"{struct_type.id}\";\n"
    )
    hdr.write("};\n\n")

  def _emit_alias(self, hdr: TextIO, alias_type: types.ESIType) -> None:
    """Emit a using alias when the alias targets a different C++ type."""
    tid = alias_type.id
    alias_info = self._get_alias_info(alias_type)
    if alias_info is None:
      return
    alias_name, inner_wrapped = alias_info
    alias_name = self.alias_name_by_id.get(tid, alias_name)
    inner_cpp = None
    if inner_wrapped is not None:
      inner_cpp = self.type_id_map[inner_wrapped.id]
    if inner_cpp is None:
      inner_cpp = self.type_id_map[tid]
    if inner_cpp != alias_name:
      hdr.write(f"using {alias_name} = {inner_cpp};\n\n")

  def _ordered_emit_types(
      self,
      types_to_emit: List[types.ESIType]) -> Tuple[List[types.ESIType], bool]:
    """Collect and order types for deterministic emission."""
    self.emit_counter = 0
    emit_nodes: Dict[Tuple[str, str], types.ESIType] = {}
    emit_deps: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    emit_index: Dict[Tuple[str, str], int] = {}
    # First pass: build the dependency graph.
    for esi_type in types_to_emit:
      try:
        self._visit_type(esi_type, emit_nodes, emit_deps, emit_index)
      except NotImplementedError:
        sys.stderr.write(
            f"Warning: type '{esi_type}' not supported for C++ generation\n")

    # Second pass: compute a stable topological ordering.
    indegree: Dict[Tuple[str, str], int] = {key: 0 for key in emit_nodes}
    for key, deps in emit_deps.items():
      for dep in deps:
        if dep in indegree:
          indegree[key] += 1
    ready = [key for key, deg in indegree.items() if deg == 0]
    ordered: List[Tuple[str, str]] = []

    def sort_name(k: Tuple[str, str]) -> str:
      kind, tid = k
      if kind == "alias":
        return self.alias_name_by_id.get(tid, tid)
      if kind == "struct":
        return self.struct_name_by_id.get(tid, tid)
      return tid

    def sort_key(k: Tuple[str, str]) -> Tuple[int, str, int]:
      alias_or_aliased = k[0] == "alias" or (k[0] == "struct" and
                                             k[1] in self.aliased_structs)
      return (0 if alias_or_aliased else 1, sort_name(k), emit_index[k])

    while ready:
      ready.sort(key=sort_key)
      key = ready.pop(0)
      ordered.append(key)
      for other_key, deps in emit_deps.items():
        if key in deps:
          deps.remove(key)
          indegree[other_key] -= 1
          if indegree[other_key] == 0:
            ready.append(other_key)

    has_cycle = len(ordered) != len(emit_nodes)
    if has_cycle:
      remaining = [key for key in emit_nodes if key not in ordered]
      remaining.sort(key=sort_key)
      ordered.extend(remaining)

    return [emit_nodes[key] for key in ordered], has_cycle

  def write_header(self, output_dir: Path, system_name: str) -> None:
    """Emit the fully ordered types.h header into the output directory.

    Types are drained via a dependency queue so aliases and structs appear in
    an order that satisfies their nested references.
    """
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
      ordered_types, has_cycle = self._ordered_emit_types(
          list(self.type_by_id.values()))
      if has_cycle:
        sys.stderr.write("Warning: cyclic type dependencies detected.\n")
        sys.stderr.write("  Logically this should not be possible.\n")
        sys.stderr.write(
            "  Emitted code may fail to compile due to ordering issues.\n")

      for emit_type in ordered_types:
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
