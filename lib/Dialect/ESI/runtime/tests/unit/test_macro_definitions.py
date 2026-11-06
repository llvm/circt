"""Unit tests for RTL macro definition loading and propagation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

_accel_mock = MagicMock()
sys.modules["esiaccel.esiCppAccel"] = _accel_mock

from esiaccel.cosim import simulator  # noqa: E402
from esiaccel.cosim import pytest as cosim_pytest  # noqa: E402


def _load_cosim_cli():
  script = (Path(__file__).parents[2] / "cosim_dpi_server" / "esi-cosim.py")
  spec = importlib.util.spec_from_file_location("esi_cosim", script)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class TestLoadMacroDefinitions:

  def test_loads_and_normalizes_scalar_values(self, tmp_path):
    definitions_file = tmp_path / "macros.json"
    definitions_file.write_text(
        json.dumps({
            "STRING": "value",
            "INTEGER": 42,
            "FLOAT": 1.5,
            "BOOLEAN": True,
            "FLAG": None,
        }))

    assert simulator.load_macro_definitions(definitions_file) == {
        "STRING": "value",
        "INTEGER": "42",
        "FLOAT": "1.5",
        "BOOLEAN": "True",
        "FLAG": None,
    }

  def test_missing_file_raises(self, tmp_path):
    with pytest.raises(FileNotFoundError,
                       match="Macro definitions file not found"):
      simulator.load_macro_definitions(tmp_path / "missing.json")

  def test_invalid_json_raises(self, tmp_path):
    definitions_file = tmp_path / "macros.json"
    definitions_file.write_text("{")

    with pytest.raises(ValueError, match="is not valid JSON"):
      simulator.load_macro_definitions(definitions_file)

  @pytest.mark.parametrize("contents", ["[]", '"macro"', "null"])
  def test_non_object_json_raises(self, tmp_path, contents):
    definitions_file = tmp_path / "macros.json"
    definitions_file.write_text(contents)

    with pytest.raises(ValueError, match="must contain a JSON object"):
      simulator.load_macro_definitions(definitions_file)

  def test_non_scalar_value_raises(self, tmp_path):
    definitions_file = tmp_path / "macros.json"
    definitions_file.write_text(json.dumps({"BAD": ["value"]}))

    with pytest.raises(ValueError, match="expected a scalar or null"):
      simulator.load_macro_definitions(definitions_file)


class TestPytestMacroDefinitions:

  def test_resolves_relative_file_and_explicit_overrides(self, tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "macros.json").write_text(
        json.dumps({
            "FROM_FILE": "file",
            "OVERRIDE": "file",
        }))
    config = cosim_pytest.CosimPytestConfig(
        source_generator=lambda _config, _tmp_dir: sources_dir,
        macro_definitions={
            "OVERRIDE": "explicit",
            "FROM_CONFIG": None
        },
        macro_definitions_file="macros.json",
    )

    assert cosim_pytest._resolve_macro_definitions(config, sources_dir) == {
        "FROM_FILE": "file",
        "OVERRIDE": "explicit",
        "FROM_CONFIG": None,
    }

  def test_resolves_absolute_file(self, tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    definitions_file = tmp_path / "macros.json"
    definitions_file.write_text(json.dumps({"FROM_FILE": "absolute"}))
    config = cosim_pytest.CosimPytestConfig(
        source_generator=lambda _config, _tmp_dir: sources_dir,
        macro_definitions_file=definitions_file,
    )

    assert cosim_pytest._resolve_macro_definitions(config, sources_dir) == {
        "FROM_FILE": "absolute",
    }

  def test_create_simulator_passes_resolved_macros(self, tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "macros.json").write_text(json.dumps({"FROM_FILE": "1"}))
    config = cosim_pytest.CosimPytestConfig(
        source_generator=lambda _config, _tmp_dir: sources_dir,
        macro_definitions={"FROM_CONFIG": None},
        macro_definitions_file="macros.json",
    )
    expected_simulator = object()

    with patch.object(cosim_pytest,
                      "get_simulator",
                      return_value=expected_simulator) as get_simulator:
      result = cosim_pytest._create_simulator(config, sources_dir,
                                              tmp_path / "run")

    assert result is expected_simulator
    assert get_simulator.call_args.args[-1] == {
        "FROM_FILE": "1",
        "FROM_CONFIG": None,
    }


class TestCosimCliMacroDefinitions:

  def test_define_file_is_overridden_by_command_line_definitions(
      self, tmp_path):
    cli = _load_cosim_cli()
    source_dir = tmp_path / "hw"
    source_dir.mkdir()
    definitions_file = tmp_path / "macros.json"
    definitions_file.write_text(
        json.dumps({
            "FROM_FILE": "file",
            "OVERRIDE": "file",
        }))
    fake_simulator = MagicMock()
    fake_simulator.compile.return_value = 0
    fake_simulator.run.return_value = 17

    with patch.object(cli, "get_simulator",
                      return_value=fake_simulator) as get_simulator:
      result = cli.__main__([
          "esi-cosim.py",
          "--define-file",
          str(definitions_file),
          "--source",
          str(source_dir),
          "--define",
          "OVERRIDE=command-line",
          "--define",
          "FLAG",
          "--no-compile",
          "inner-command",
          "argument",
      ])

    assert result == 17
    assert get_simulator.call_args.kwargs["macro_definitions"] == {
        "FROM_FILE": "file",
        "OVERRIDE": "command-line",
        "FLAG": None,
    }
    fake_simulator.run.assert_called_once_with(["argument"],
                                               gui=False,
                                               server_only=False)
