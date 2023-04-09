# Asterix processing tool - python version

## Installation

Install latest git version to virtual environment

```bash
python3 -m venv env
source env/bin/activate
pip install "git+https://github.com/zoranbosnjak/asterix-tool.git#subdirectory=ast-tool-py"
ast-tool-py -h
```

## Development

```bash
nix-shell
./update-from-upstream.sh
python3 ./src/main.py --version
python3 ./src/main.py -h
exit
```

