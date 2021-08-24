#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import mlir.ir as ir

from typing import Tuple

# TODO: consider moving this functionality into C++.


class AppID:
  """AppID models the instance hierarchy which the architect cares about.
  Specifically, AppIDs can skip levels in the instances hierarchy and persist
  through hierarchy changes."""

  def __init__(self, *appid: Tuple[str]):
    assert len(appid) > 0
    self._parts = list()
    for p in appid:
      assert isinstance(p, str)
      self._parts.extend(p.split("."))

  @property
  def head(self):
    return self._parts[0]

  @property
  def tail(self):
    if len(self._parts) > 1:
      return AppID(*self._parts[1:])
    return None

  def __eq__(self, o: object) -> bool:
    return isinstance(o, AppID) and o._parts == self._parts

  def __add__(self, part: str) -> AppID:
    return AppID(*self._parts, part)


class AppIDIndex(dict):
  """Model the AppID hierarchy. Provides the ability to attach attributes to
  AppIDs rather than instances, to get applied once the design is fully
  generated."""

  def __init__(self):
    self._children: dict[str, AppIDIndex] = dict()

  def lookup(self, appid: AppID) -> AppIDIndex:
    if appid is None:
      return self
    if appid.head not in self._children:
      self._children[appid.head] = AppIDIndex()
    return self._children[appid.head].lookup(appid.tail)

  def add_attribute(self, attr: Tuple[str, ir.Attribute]) -> None:
    self[attr[0]] = attr[1]

  @property
  def apply_attributes_visitor(self):
    from .instance import Instance

    def _visit(idx, inst: Instance):
      attrs = idx.lookup(inst.appid)
      for (akey, attr) in attrs.items():
        inst.attach_attribute(akey, attr)

    return lambda i, idx=self: _visit(idx, i)
