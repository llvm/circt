#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Port-kind strategy table for the C++ module generator.

Centralises everything that varies between the ESI port kinds (function,
callback, to-host / from-host channel, MMIO region, telemetry metric, plain
bundle) into a single `_PortKind` record per kind plus two pure rendering
functions, so the generator carries one table instead of ~8 parallel
`isinstance(port, ...)` ladders. The renderers are pure functions of a
`_PortKind`, so they are golden-tested without a live accelerator.
"""

from dataclasses import dataclass, field as _dc_field
from typing import List, Optional, Tuple

from ..types import (FunctionPort as _FunctionPort, CallbackPort as
                     _CallbackPort, ToHostPort as _ToHostPort, FromHostPort as
                     _FromHostPort, MMIORegion as _MMIORegionPort, MetricPort as
                     _MetricPort)


@dataclass
class _PortGroup:
  """All strings needed to emit one port slot (member + ctor + find) in Connected."""
  struct_decls: List[str] = _dc_field(default_factory=list)
  member_decl: str = ""
  ctor_params: List[str] = _dc_field(default_factory=list)
  init_entry: str = ""
  find_code: str = ""
  make_unique_args: List[str] = _dc_field(default_factory=list)
  post_connect: str = ""
  using_aliases: List[Tuple[str, str]] = _dc_field(default_factory=list)


@dataclass(frozen=True)
class _PortKind:
  """Per-port-kind constants for generating a module's typed port slots.

  Centralises what were ~8 parallel `isinstance(port, ...)` ladders in
  `CppGenerator`. `member_template` (typed channel/func ports) and
  `member_ref` (service / bundle ports) are mutually exclusive. `find_style`
  / `indexed_find` select the `connect()` resolution snippet for scalar and
  indexed ports respectively; `alias_kind` selects which `using` aliases to
  emit for the typed template parameters.
  """
  service_type: Optional[
      str]  # findPortAs<...> service type (None = plain bundle)
  find_style: str  # 'ptr' | 'read_chan' | 'write_chan' | 'deref' | 'bundle'
  indexed_find: str  # 'as_ptr' | 'chan_read' | 'chan_write' | 'bundle_addr'
  param_type: str  # ctor parameter type
  param_suffix: str  # '_port' | '_chan' | '_svc'
  member_template: Optional[
      str]  # typed member, e.g. "esi::TypedFunction<{p}Args, {p}Result>"
  member_ref: Optional[str]  # non-typed member / ctor ref type
  indexed_elem: Optional[
      str]  # IndexedPorts<...> element (None => use member_template)
  alias_kind: Optional[str]  # None | 'func' (Args/Result) | 'chan' (Data)
  connectable: bool  # whether generated connect() calls .connect()


# Ordered list of (runtime port class, kind). `_port_kind` returns the first
# match, falling back to `_BUNDLE_KIND` for any unrecognised service port.
_PORT_KINDS: List[Tuple[type, _PortKind]] = [
    (_FunctionPort,
     _PortKind(service_type="esi::services::FuncService::Function",
               find_style="ptr",
               indexed_find="as_ptr",
               param_type="esi::services::FuncService::Function *",
               param_suffix="_port",
               member_template="esi::TypedFunction<{p}Args, {p}Result>",
               member_ref=None,
               indexed_elem=None,
               alias_kind="func",
               connectable=True)),
    (_CallbackPort,
     _PortKind(service_type="esi::services::CallService::Callback",
               find_style="ptr",
               indexed_find="as_ptr",
               param_type="esi::services::CallService::Callback *",
               param_suffix="_port",
               member_template="esi::TypedCallback<{p}Args, {p}Result>",
               member_ref=None,
               indexed_elem=None,
               alias_kind="func",
               connectable=False)),
    (_ToHostPort,
     _PortKind(service_type="esi::services::ChannelService::ToHost",
               find_style="read_chan",
               indexed_find="chan_read",
               param_type="esi::ReadChannelPort &",
               param_suffix="_chan",
               member_template="esi::TypedReadPort<{p}Data>",
               member_ref=None,
               indexed_elem=None,
               alias_kind="chan",
               connectable=True)),
    (_FromHostPort,
     _PortKind(service_type="esi::services::ChannelService::FromHost",
               find_style="write_chan",
               indexed_find="chan_write",
               param_type="esi::WriteChannelPort &",
               param_suffix="_chan",
               member_template="esi::TypedWritePort<{p}Data>",
               member_ref=None,
               indexed_elem=None,
               alias_kind="chan",
               connectable=True)),
    (_MMIORegionPort,
     _PortKind(service_type="esi::services::MMIO::MMIORegion",
               find_style="deref",
               indexed_find="as_ptr",
               param_type="esi::services::MMIO::MMIORegion &",
               param_suffix="_svc",
               member_template=None,
               member_ref="esi::services::MMIO::MMIORegion &",
               indexed_elem="esi::services::MMIO::MMIORegion *",
               alias_kind=None,
               connectable=False)),
    (_MetricPort,
     _PortKind(service_type="esi::services::TelemetryService::Metric",
               find_style="deref",
               indexed_find="as_ptr",
               param_type="esi::services::TelemetryService::Metric &",
               param_suffix="_svc",
               member_template=None,
               member_ref="esi::services::TelemetryService::Metric &",
               indexed_elem="esi::services::TelemetryService::Metric *",
               alias_kind=None,
               connectable=True)),
]

# Fallback for any service port that doesn't match a standard specialization
# (e.g. a custom `@esi.ServiceDecl`-defined service).
_BUNDLE_KIND = _PortKind(service_type=None,
                         find_style="bundle",
                         indexed_find="bundle_addr",
                         param_type="esi::BundlePort &",
                         param_suffix="_port",
                         member_template=None,
                         member_ref="esi::BundlePort &",
                         indexed_elem="esi::BundlePort *",
                         alias_kind=None,
                         connectable=False)


def _port_kind(port) -> _PortKind:
  """Return the `_PortKind` describing `port`, or `_BUNDLE_KIND` as fallback."""
  for cls, kind in _PORT_KINDS:
    if isinstance(port, cls):
      return kind
  return _BUNDLE_KIND


def _render_scalar_find(kind: _PortKind, v: str, appid_expr: str) -> str:
  """Render the connect()-body snippet that resolves one scalar port into the
  local `v`. Pure function of the kind so it can be golden-tested without a
  live accelerator (see `test_port_find_code_golden`)."""
  if kind.find_style == "ptr":
    return (f"auto *{v} =\n"
            f"    esi::findPortAsOrThrow<{kind.service_type}>(\n"
            f"        rawModule, {appid_expr});")
  if kind.find_style == "read_chan":
    return (f"auto &{v} =\n"
            f"    esi::findPortAsOrThrow<{kind.service_type}>(\n"
            f'        rawModule, {appid_expr})->getRawRead("data");')
  if kind.find_style == "write_chan":
    return (f"auto &{v} =\n"
            f"    esi::findPortAsOrThrow<{kind.service_type}>(\n"
            f'        rawModule, {appid_expr})->getRawWrite("data");')
  if kind.find_style == "deref":
    return (f"auto &{v} =\n"
            f"    *esi::findPortAsOrThrow<{kind.service_type}>(\n"
            f"        rawModule, {appid_expr});")
  # plain BundlePort fallback
  return f"auto &{v} = esi::findPortOrThrow(rawModule, {appid_expr});"


def _render_indexed_find(kind: _PortKind, map_var: str, appid_expr: str) -> str:
  """Render the per-index `try_emplace` body for an `IndexedPorts<T>` group.
  Pure function of the kind (golden-tested alongside `_render_scalar_find`)."""
  if kind.indexed_find == "as_ptr":
    return (f"  {map_var}.try_emplace(\n"
            f"      static_cast<int>(idx),\n"
            f"      esi::findPortAsOrThrow<{kind.service_type}>(\n"
            f"          rawModule, {appid_expr}));")
  if kind.indexed_find == "chan_read":
    return (f"  auto *svc =\n"
            f"      esi::findPortAsOrThrow<{kind.service_type}>(\n"
            f"          rawModule, {appid_expr});\n"
            f"  {map_var}.try_emplace(\n"
            f"      static_cast<int>(idx),\n"
            f'      svc->getRawRead("data"));')
  if kind.indexed_find == "chan_write":
    return (f"  auto *svc =\n"
            f"      esi::findPortAsOrThrow<{kind.service_type}>(\n"
            f"          rawModule, {appid_expr});\n"
            f"  {map_var}.try_emplace(\n"
            f"      static_cast<int>(idx),\n"
            f'      svc->getRawWrite("data"));')
  # bundle_addr
  return (f"  {map_var}.try_emplace(\n"
          f"      static_cast<int>(idx),\n"
          f"      &esi::findPortOrThrow(rawModule, {appid_expr}));")
