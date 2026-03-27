#!/usr/bin/env bash -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

apt update
apt install -y python3-venv
python3 -m venv "$DIR/../env"
source "$DIR/../env/bin/activate"
pip install --upgrade pip
pip install -r "$DIR/../../lib/Dialect/ESI/runtime/requirements.txt"
pip install IPython yapf
