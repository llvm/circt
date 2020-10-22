#!/bin/bash
# Renders the `docs/dialects.drawio` diagram using the `draw.io` desktop app.

set -e
DOCS_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../docs" && pwd)

# On macOS the app is an actual application package. On Linux it's usually in
# the PATH.
if [[ "$OSTYPE" == "darwin"* ]]; then
	DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io
else
	DRAWIO=draw.io
fi

# Check it's there and resolve the path.
if DRAWIO=`! which $DRAWIO`; then
	echo "error: draw.io not installed" >&2
	exit 1
fi

# Update the rendered diagrams in the docs.
$DRAWIO -x -t -s 2 -o $DOCS_DIR/dialects.png $DOCS_DIR/dialects.drawio
$DRAWIO -x -t -s 2 -o $DOCS_DIR/dialects.svg $DOCS_DIR/dialects.drawio
