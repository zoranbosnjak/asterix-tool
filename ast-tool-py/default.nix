{ sources ? import ../nix/sources.nix
, pkgs ? import sources.nixpkgs {}
, asterixlibRef ? builtins.fromJSON (builtins.readFile ../nix/asterix-libs.json)
, inShell ? null
}:

let

  asterixlibDir = pkgs.fetchgit {
    url = asterixlibRef.url;
    rev = asterixlibRef.rev;
    sha256 = asterixlibRef.sha256;
  };

  asterixLib = pkgs.callPackage
    "${asterixlibDir}/libs/python" {inShell=false; inherit sources pkgs;};

  deps = [
    pkgs.python3
    pkgs.python3Packages.mypy
    pkgs.python3Packages.pytest
    pkgs.python3Packages.hypothesis
    pkgs.python3Packages.autopep8
    pkgs.python3Packages.build
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.scapy
    asterixLib
  ];

  env = pkgs.mkShell {
    packages = [
        (pkgs.python3.withPackages (python-pkgs: deps))
    ];

    shellHook = ''
        export PYTHONPATH=$(pwd)/src:$PYTHONPATH
    '';
    };

  drv = pkgs.python3Packages.buildPythonPackage rec {
    name = "ast-tool-py";
    format = "pyproject";
    src = ./.;
    propagatedBuildInputs = deps;
    shellHook = ''
        ast-tool() { python ./src/main.py "$@"; }
    '';
  };

in if inShell == false
   then drv
   else if pkgs.lib.inNixShell then env else drv
