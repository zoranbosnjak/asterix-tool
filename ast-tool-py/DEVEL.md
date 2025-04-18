# Development environment

```bash
nix-shell

# prettify python source code
autopep8 --in-place --aggressive --aggressive <filename> <filename>...

# run static code check and tests once
mypy
pytest

# monitor changes in .py files, check automatically on any change
find . | grep "\.py" | entr sh -c 'clear && date && mypy && pytest'

./update-from-upstream.sh
python3 ./src/main.py --version
alias ast-tool-py='python3 ./src/main.py'
ast-tool-py --version
exit
```

## Manually publish/update project to pypi

**Note**:
Normally this is not necessary.
The process is already running from github action.

``` bash
nix-shell
# from clean repository
git status
python3 -m build
ls -l dist/*

# upload to testpypi
twine upload --repository testpypi dist/*

# upload to production pypi
twine upload dist/*
```

