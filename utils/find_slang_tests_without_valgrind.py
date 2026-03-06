#!/usr/bin/env python3
"""Check that test files with "REQUIRES: slang" also have "// UNSUPPORTED: valgrind"."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Check test files with "REQUIRES: slang" have "// UNSUPPORTED: valgrind"'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='test',
        help='Directory to search (default: test)'
    )
    args = parser.parse_args()

    # Find all .sv and .mlir files
    test_files = []
    for ext in ['.sv', '.mlir']:
        test_files.extend(Path(args.directory).rglob(f'*{ext}'))

    # Check each file
    missing = []
    for filepath in sorted(test_files):
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            if 'REQUIRES: slang' in content and 'UNSUPPORTED: valgrind' not in content:
                missing.append(filepath)
        except Exception:
            pass

    # Print results
    if missing:
        print(f"Files with 'REQUIRES: slang' missing '// UNSUPPORTED: valgrind':")
        for f in missing:
            print(f"  {f}")
        return 1

    print(f"All files in '{args.directory}' with 'REQUIRES: slang' have the valgrind marker")
    return 0


if __name__ == '__main__':
    sys.exit(main())

