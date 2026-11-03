#!/usr/bin/env python3
"""Inventory CIRCT import, lowering, and Verilog-emission failures on LogikBench."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
CIRCT = HOME / 'projwect/CC/circt'
LB = HOME / 'projwect/CC/logikbench'
BENCH = LB / 'logikbench/benchmarks/large'
CORAL_BENCH = LB / 'logikbench/benchmarks/microalchemy/coralnpu'
DEFAULT_OUT = CIRCT / 'cleanedlogik/diagnostics'
DEFAULT_DEFINES = ('SYNTHESIS', 'VERILINT', 'NO_PLI_OR_EMU', 'DESIGNWARE_NOEXIST')


DESIGNS = {
    'axicrossbar': ('axi_crossbar', (
        'arbiter.v', 'axi_crossbar_addr.v', 'axi_crossbar_rd.v',
        'axi_crossbar.v', 'axi_crossbar_wr.v', 'axi_register_rd.v',
        'axi_register_wr.v', 'priority_encoder.v'), {}, ()),
    'blackparrot': ('black_parrot', ('pickled.v', 'lambda.v'), {}, ('SYNTHESIS',)),
    'coralnpu': ('__resolve_from_compile_json__', (), {}, ('SYNTHESIS',)),
    'qr': ('qr', ('qr_cordic.v', 'qr_boundary.v', 'qr_internal.v',
                 'qr_array.v', 'qr.v'), {'N': '16', 'DW': '16'}, ()),
    'rocket': ('rocket', ('rocket.v', 'riscv_bootrom.v'), {}, ()),
    'sonicboom': ('sonicboom_DigitalTop', ('sonicboom.sv', 'sonicboom_srams.v'), {}, ()),
    'tpu': ('tpu', ('tpu_pe.v', 'tpu_array.v', 'tpu.v'), {'N': '128'}, ()),
    'vortex': ('vortex_core_wrap', ('vortex.v',), {}, ()),
}
NEEDS_LAMBDA = frozenset(('blackparrot', 'rocket', 'sonicboom'))
BASE_OPT = ('--hw-cleanup', '--canonicalize', '--cse', '--canonicalize')
LOWER = ('--lower-seq-to-sv', '--lower-sim-to-sv', '--canonicalize')
EXPORT = ('--lower-hw-to-sv', '--hw-legalize-modules', '--hw-cleanup',
          '--export-verilog', '-o', '/dev/null')
OP_RE = re.compile(r'(?<![\w.])(llhd|cf|seq|sim|moore|hw|comb|sv|ub|datapath)\.([A-Za-z_]\w*)')
SEVERITY_RE = re.compile(r'\b(?P<kind>fatal error|error|warning|remark):\s*(?P<msg>.*)$')
NOTE_RE = re.compile(r'\bnote:\s*')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
LOCATION_RE = re.compile(r'(?:^|\s)(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+):\s*(?:fatal error|error|warning|remark):')


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(p: Path, obj: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def library_dirs() -> list[str]:
    python = LB / 'banana/bin/python'
    cmd = [str(python), '-c',
           'import lambdalib; from pathlib import Path; '
           'print(Path(lambdalib.__file__).resolve().parent)']
    root = Path(subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE).strip())
    return sorted(str(p.resolve()) for p in root.rglob('rtl') if p.is_dir())


def resolve_coral_top(root: Path) -> str:
    """Use CORAL_TOP if supplied; otherwise infer the top from compile.json."""
    explicit = os.environ.get('CORAL_TOP', '').strip()
    if explicit:
        return explicit
    meta_path = root / 'compile.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'Missing {meta_path}; set CORAL_TOP explicitly')
    data = json.loads(meta_path.read_text())
    top_keys = {'top', 'top_module', 'topmodule', 'top_module_name',
                'topmodulename', 'topmodules', 'tops', 'top_modules'}
    candidates: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower().replace('-', '_') in top_keys:
                    vals = item if isinstance(item, list) else [item]
                    for v in vals:
                        if isinstance(v, str) and re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', v):
                            candidates.add(v)
                if key in ('command', 'arguments'):
                    import shlex
                    if isinstance(item, str):
                        try:
                            args = shlex.split(item)
                        except ValueError:
                            args = []
                    elif isinstance(item, list):
                        args = item
                    else:
                        args = []
                    for i, arg in enumerate(args[:-1]):
                        if arg in ('-top', '--top', '-top-module', '--top-module'):
                            candidates.add(str(args[i + 1]))
                walk(item)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(data)
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ValueError(
        f'Could not identify one unambiguous top from {meta_path}; '
        f'candidates={sorted(candidates)}. Set CORAL_TOP to the actual benchmark top.')


def manifest(name: str, libs: list[str] | None) -> dict:
    top, filenames, parameters, defines = DESIGNS[name]
    if name == 'coralnpu':
        root = CORAL_BENCH.resolve()
        filelist = root / 'filelist.f'
        if not filelist.is_file():
            raise FileNotFoundError(f'Missing CoralNPU filelist: {filelist}')
        top = resolve_coral_top(root)
        return {
            'top': top,
            'files': [],
            'filelist': str(filelist),
            'cwd': str(root),
            'includes': [str(root)],
            'libraries': [],
            'defines': list(dict.fromkeys((*DEFAULT_DEFINES, *defines))),
            'parameters': dict(parameters),
        }

    root = BENCH / name
    spec = {
        'top': top,
        'files': [str((root / 'rtl' / f).resolve()) for f in filenames],
        'includes': [str((root / 'include').resolve())] if (root / 'include').is_dir() else [],
        'libraries': list(libs or ()) if name in NEEDS_LAMBDA else [],
        'defines': list(dict.fromkeys((*DEFAULT_DEFINES, *defines))),
        'parameters': dict(parameters),
    }
    for folder in ('vlibs', 'rams/synth', 'rams/model'):
        path = root / folder
        if path.is_dir():
            spec['libraries'].append(str(path.resolve()))
    spec['libraries'] = list(dict.fromkeys(spec['libraries']))
    absent = [f for f in spec['files'] if not Path(f).is_file()]
    if absent:
        raise FileNotFoundError('Missing RTL input(s): ' + '; '.join(absent))
    return spec


def frontend_command(tool_bin: Path, spec: dict, mode: str, out_ir: Path,
                     *, ir_kind: str = 'hw', input_files: list[str] | None = None,
                     apply_parameters: bool = True) -> list[str]:
    # CIRCT otherwise cannot infer a format from a command-file-only invocation.
    cmd = [str(tool_bin / 'circt-verilog'), '--format=sv', '--ir-' + ir_kind]
    if mode == 'sroa':
        cmd.append('--sroa')
    cmd += ['--top', spec['top']]
    if apply_parameters:
        for key, value in spec['parameters'].items():
            cmd += ['-G', f'{key}={value}']
    for define in spec['defines']:
        cmd += ['-D', define]
    for include in spec['includes']:
        cmd += ['-I', include]
    for library in spec['libraries']:
        cmd += ['-y', library]
    if input_files is not None:
        cmd += list(input_files)
    elif spec.get('filelist'):
        # CIRCT's -f -C command-file option and is declared cl::Prefix.
        cmd.append('-C' + spec['filelist'])
    else:
        cmd += spec['files']
    cmd += ['-o', str(out_ir)]
    return cmd


def run_step(command: list[str], folder: Path, label: str,
             timeout: float, stdout_override: Path | None = None,
             cwd: Path | None = None) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    stdout = stdout_override if stdout_override is not None else folder / (label + '.stdout')
    stderr = folder / (label + '.stderr')
    command_file = folder / (label + '.command.json')
    save_json(command_file, command)
    start = time.perf_counter()
    status, code = 'launch_error', None
    with stdout.open('wb') as out, stderr.open('wb') as err:
        try:
            child = subprocess.Popen(command, stdout=out, stderr=err,
                                     start_new_session=True, cwd=cwd)
            try:
                code = child.wait(timeout=timeout)
                status = 'ok' if code == 0 else ('crash' if code < 0 else 'error')
            except subprocess.TimeoutExpired:
                status = 'timeout'
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                code = child.returncode
            except KeyboardInterrupt:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                raise
        except OSError as exc:
            err.write((f'Unable to start command: {exc}\n').encode())
    return {
        'status': status, 'returncode': code,
        'seconds': round(time.perf_counter() - start, 3),
        'stdout': str(stdout), 'stderr': str(stderr),
        'command_file': str(command_file),
    }


def operation_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if path.is_file():
        with path.open(errors='replace') as f:
            for line in f:
                for m in OP_RE.finditer(line):
                    counts[m.group(1) + '.' + m.group(2)] += 1
    return dict(counts.most_common())


def parse_diagnostics(path: Path) -> list[dict]:
    """Preserve diagnostic context without truncation; raw stderr remains on disk."""
    if not path.is_file():
        return []
    events: list[dict] = []
    active: dict | None = None
    with path.open(errors='replace') as f:
        for lineno, raw in enumerate(f, 1):
            line = ANSI_RE.sub('', raw.rstrip('\n'))
            m = SEVERITY_RE.search(line)
            if m is not None and NOTE_RE.search(line) is None:
                if active is not None:
                    events.append(active)
                loc = LOCATION_RE.search(line)
                active = {
                    'kind': m.group('kind'),
                    'message': m.group('msg').strip(),
                    'header': line,
                    'stderr_line': lineno,
                    'location': ({'file': loc.group('file'),
                                  'line': int(loc.group('line')),
                                  'column': int(loc.group('col'))} if loc else None),
                    'text': line + '\n',
                }
            elif active is not None:
                active['text'] += line + '\n'
        if active is not None:
            events.append(active)
    return events


def attach(result: dict, label: str, step: dict,
           ir: Path | None = None, output_sv: Path | None = None) -> bool:
    events = parse_diagnostics(Path(step['stderr']))
    diag_path = Path(step['stderr']).with_name(label + '.diagnostics.json')
    save_json(diag_path, events)
    step['diagnostics_file'] = str(diag_path)
    step['error_count'] = sum(e['kind'] in ('error', 'fatal error') for e in events)
    step['warning_count'] = sum(e['kind'] == 'warning' for e in events)
    if ir is not None:
        step['ir'] = str(ir)
        step['ops'] = operation_counts(ir)
        if step['status'] == 'ok' and (not ir.is_file() or ir.stat().st_size == 0):
            step['status'] = 'missing_output'
    if output_sv is not None:
        step['output_sv'] = str(output_sv)
        step['sv_bytes'] = output_sv.stat().st_size if output_sv.is_file() else 0
        if step['status'] == 'ok' and step['sv_bytes'] == 0:
            step['status'] = 'empty_output'
    result['steps'][label] = step
    return step['status'] == 'ok'


def analyze(name: str, mode: str, spec: dict, out: Path, tools: Path,
            timeout: float, reimport: bool) -> dict:
    folder = out / name / mode
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / 'manifest.json', spec)
    result = {'design': name, 'mode': mode, 'top': spec['top'], 'steps': {}}
    imp_ir = folder / 'imported.mlir'
    opt_ir = folder / 'optimized.mlir'
    low_ir = folder / 'lowered.mlir'
    sv = folder / (spec['top'] + '.sv')
    for old in (imp_ir, opt_ir, low_ir, sv, folder / 'import_moore.mlir',
                folder / 'reimported_moore.mlir'):
        old.unlink(missing_ok=True)
    cmd = frontend_command(tools, spec, mode, imp_ir)
    cwd = Path(spec['cwd']) if spec.get('cwd') else None
    imported = attach(result, 'import', run_step(cmd, folder, 'import', timeout, cwd=cwd), imp_ir)
    if not imported:
        # Distinguish slang/SV->Moore errors from later Moore->Core failures.
        moore_ir = folder / 'import_moore.mlir'
        cmd = frontend_command(tools, spec, mode, moore_ir, ir_kind='moore')
        attach(result, 'import_moore',
               run_step(cmd, folder, 'import_moore', timeout, cwd=cwd), moore_ir)
        result['final_status'] = 'failed_import'
        return result
    cmd = [str(tools / 'circt-opt'), str(imp_ir), *BASE_OPT, '-o', str(opt_ir)]
    if not attach(result, 'optimize', run_step(cmd, folder, 'optimize', timeout), opt_ir):
        result['final_status'] = 'failed_optimize'
        return result
    cmd = [str(tools / 'circt-opt'), str(opt_ir), *LOWER, '-o', str(low_ir)]
    if not attach(result, 'lower_seq_sim', run_step(cmd, folder, 'lower_seq_sim', timeout), low_ir):
        result['final_status'] = 'failed_lower_seq_sim'
        return result
    cmd = [str(tools / 'circt-opt'), str(low_ir), *EXPORT]
    if not attach(result, 'export', run_step(cmd, folder, 'export', timeout, sv), output_sv=sv):
        result['final_status'] = 'failed_export'
        return result
    if reimport:
        parsed = folder / 'reimported_moore.mlir'
        cmd = frontend_command(tools, spec, 'default', parsed, ir_kind='moore',
                               input_files=[str(sv)], apply_parameters=False)
        if not attach(result, 'reimport',
                      run_step(cmd, folder, 'reimport', timeout), parsed):
            result['final_status'] = 'failed_reimport'
            return result
    result['final_status'] = 'emitted_unverified'
    return result


def step_label(step: dict | None) -> str:
    if step is None:
        return 'skipped'
    text = step['status']
    if step['status'] == 'crash':
        text += f" (signal {-step['returncode']})" if step['returncode'] is not None else ''
    return text


def md_escape(text: str) -> str:
    return text.replace('|', '\\|').replace('\n', ' ')


def write_report(results: list[dict], setup_errors: dict[str, str],
                 out: Path, reimport: bool) -> None:
    """Report every observed diagnostic without imposing a fixed error taxonomy."""
    report = out / 'problems.md'
    with report.open('w') as w:
        w.write('# CIRCT failures on eight LogikBench designs\n\n')
        w.write('Generated: ' + iso_now() + '\n\n')
        w.write('Raw stderr and individual diagnostic occurrences are preserved under each design/mode. '
                'The list below contains every distinct emitted error/warning message per step, '
                'its count, source locations, and an unabridged example with all accompanying notes. '
                'There is no predefined error taxonomy.\n\n')
        w.write('`--sroa` is experimental; successful export does not establish equivalence.\n\n')
        w.write('## Stage results\n\n')
        w.write('| Design | Mode | Import | Optimize | Lower seq/sim | Export | '
                + ('Reimport |' if reimport else '') + '\n')
        w.write('|---|---|---|---|---|---|'
                + ('---|' if reimport else '') + '\n')
        stages = ['import', 'optimize', 'lower_seq_sim', 'export']
        if reimport:
            stages.append('reimport')
        for design, error in sorted(setup_errors.items()):
            w.write(f'| {design} | setup failed | {md_escape(error)} | '
                    + ' | '.join('skipped' for _ in stages[1:]) + ' |\n')
        for r in results:
            labels = [step_label(r['steps'].get(s)) for s in stages]
            w.write('| ' + ' | '.join([r['design'], r['mode'], *labels]) + ' |\n')
        w.write('\n## All observed diagnostics, by design and stage\n\n')
        if setup_errors:
            w.write('### Setup failures\n\n')
            for name, message in sorted(setup_errors.items()):
                w.write(f'- `{name}`: {message}\n')
            w.write('\n')
        for r in results:
            design, mode = r['design'], r['mode']
            w.write(f'### {design} / {mode}\n\n')
            w.write(f"Final status: `{r['final_status']}`.\n\n")
            for stage, step in r['steps'].items():
                events = parse_diagnostics(Path(step['stderr']))
                errors = [e for e in events if e['kind'] in ('error', 'fatal error')]
                warnings = [e for e in events if e['kind'] == 'warning']
                w.write(f"#### {stage}: {step['status']} "
                        f"(exit {step['returncode']}, {step['seconds']}s)\n\n")
                w.write(f"Command: `{Path(step['command_file']).relative_to(out)}`. "
                        f"Complete stderr: `{Path(step['stderr']).relative_to(out)}`. "
                        f"All diagnostic occurrences: "
                        f"`{Path(step['diagnostics_file']).relative_to(out)}`.\n\n")
                if not errors and not warnings:
                    if step['status'] != 'ok':
                        w.write('No standard error/warning header was recognized. '
                                'The raw stderr and process exit status above are authoritative.\n\n')
                    else:
                        w.write('No warnings or errors reported.\n\n')
                for kind, subset in [('Errors', errors), ('Warnings', warnings)]:
                    if not subset:
                        continue
                    grouped: dict[str, list[dict]] = {}
                    for e in subset:
                        grouped.setdefault(e['kind'] + ': ' + e['message'], []).append(e)
                    w.write(f'**{kind}: {len(subset)} occurrences, '
                            f'{len(grouped)} distinct messages.**\n\n')
                    for i, (message, instances) in enumerate(grouped.items(), 1):
                        w.write(f'{i}. `{md_escape(message)}` '
                                f'(occurrences: {len(instances)})\n\n')
                        locations = [e['location'] for e in instances if e['location']]
                        if locations:
                            loc_counts = Counter(
                                f"{loc['file']}:{loc['line']}:{loc['column']}" for loc in locations)
                            w.write('   Locations: ' + ', '.join(
                                f'`{md_escape(loc)}`' + (f' ×{count}' if count > 1 else '')
                                for loc, count in loc_counts.items()) + '\n\n')
                        w.write('   Full first diagnostic and notes:\n\n')
                        w.write('```text\n' + instances[0]['text'].rstrip('\n') + '\n```\n\n')
            w.write('\n')
        w.write('## Residual IR operations (counts, not failure diagnoses)\n\n')
        w.write('| Design | Mode | llhd.process | llhd.sig | llhd.drv | CF branches | sim.* | seq.firreg |\n')
        w.write('|---|---|---:|---:|---:|---:|---:|---:|\n')
        for r in results:
            step = r['steps'].get('lower_seq_sim')
            if step is None or step['status'] != 'ok':
                continue
            ops = step['ops']
            sim = sum(num for name, num in ops.items() if name.startswith('sim.'))
            cf = sum(num for name, num in ops.items() if name.startswith('cf.'))
            w.write(f"| {r['design']} | {r['mode']} | {ops.get('llhd.process', 0)} | "
                    f"{ops.get('llhd.sig', 0)} | {ops.get('llhd.drv', 0)} | "
                    f"{cf} | {sim} | {ops.get('seq.firreg', 0)} |\n")
        w.write('\nCounts are text occurrences, not independently verified operation counts. '
                'A remaining operation is not automatically an error or nonsynthesizable. '
                'Failed imports are probed separately using `--ir-moore` to distinguish '
                'frontend parsing/elaboration failures from later conversion failures.\n')

    raw_path = out / 'full_failure_stderr.md'
    with raw_path.open('w') as w:
        w.write('# Complete stderr for every failed CIRCT stage\n\n')
        for name, message in sorted(setup_errors.items()):
            w.write(f'## {name} / setup\n\n```text\n{message}\n```\n\n')
        for r in results:
            for stage, step in r['steps'].items():
                if step['status'] == 'ok':
                    continue
                w.write(f"## {r['design']} / {r['mode']} / {stage} "
                        f"({step['status']}; exit {step['returncode']})\n\n")
                w.write(f"Original log: `{Path(step['stderr']).relative_to(out)}`\n\n")
                w.write('```text\n')
                with Path(step['stderr']).open(errors='replace') as src:
                    shutil.copyfileobj(src, w)
                w.write('\n```\n\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', choices=['all', *DESIGNS], default='all')
    parser.add_argument('--timeout', type=float, default=300,
                        help='seconds per tool invocation (default: 300)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--circt-bin', type=Path, default=CIRCT / 'build/bin')
    parser.add_argument('--only-default', action='store_true')
    parser.add_argument('--only-sroa', action='store_true')
    parser.add_argument('--skip-reimport', action='store_true',
                        help='skip parsing emitted SV back through circt-verilog')
    parser.add_argument('--list', action='store_true', help='show exact manifests')
    args = parser.parse_args()
    if args.only_default and args.only_sroa:
        parser.error('choose only one of --only-default / --only-sroa')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.list:
        for name, (top, files, params, _) in DESIGNS.items():
            if name == 'coralnpu':
                top = resolve_coral_top(CORAL_BENCH)
                print(f'{name}: top={top}; filelist={CORAL_BENCH / "filelist.f"}; params={params}')
            else:
                print(f'{name}: top={top}; files={", ".join(files)}; params={params}')
        return 0
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tool_bin = args.circt_bin.expanduser().resolve()
    for tool in ('circt-verilog', 'circt-opt'):
        if not (tool_bin / tool).is_file():
            parser.error(f'Missing tool: {tool_bin / tool}')
    designs = list(DESIGNS) if args.design == 'all' else [args.design]
    modes = ['sroa'] if args.only_sroa else ['default'] if args.only_default else ['default', 'sroa']
    setup_errors: dict[str, str] = {}
    results: list[dict] = []
    libs: list[str] = []
    if any(n in NEEDS_LAMBDA for n in designs):
        try:
            libs = library_dirs()
        except (OSError, subprocess.SubprocessError) as exc:
            for name in designs:
                if name in NEEDS_LAMBDA:
                    setup_errors[name] = 'lambdalib resolution failed: ' + str(exc)

    def persist() -> None:
        save_json(out / 'results.json', {
            'generated': iso_now(), 'circt_bin': str(tool_bin),
            'results': results, 'setup_errors': setup_errors,
        })
        write_report(results, setup_errors, out, not args.skip_reimport)

    persist()
    try:
        for design in designs:
            if design in setup_errors:
                print(f'[{design}] SETUP FAILED: {setup_errors[design]}', flush=True)
                continue
            try:
                spec = manifest(design, libs)
            except (OSError, ValueError) as exc:
                setup_errors[design] = str(exc)
                print(f'[{design}] SETUP FAILED: {exc}', flush=True)
                persist()
                continue
            for mode in modes:
                print(f'[{design}/{mode}] import -> optimize -> lower -> export'
                      + (' -> reimport' if not args.skip_reimport else ''), flush=True)
                r = analyze(design, mode, spec, out, tool_bin,
                            args.timeout, not args.skip_reimport)
                results.append(r)
                print(f'[{design}/{mode}] ' + ', '.join(
                    f'{stage}={step_label(step)}' for stage, step in r['steps'].items()),
                    flush=True)
                persist()
    except KeyboardInterrupt:
        print('\nInterrupted; completed design/mode results have been saved.', file=sys.stderr)
        persist()
        return 130
    persist()
    print(f'Inventory:        {out / "problems.md"}')
    print(f'Every failure log: {out / "full_failure_stderr.md"}')
    print(f'Structured JSON:  {out / "results.json"}')
    print('Emitted SV is not equivalence-proven. SROA is experimental.')
    return 1 if setup_errors or any(r['final_status'] != 'emitted_unverified' for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
