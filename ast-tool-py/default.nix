{ sources ? import ../nix/sources.nix
, pkgs ? import sources.nixpkgs {}
, asterixlibsRef ? builtins.fromJSON (builtins.readFile ../nix/asterix-libs.json)
, inShell ? null
}:

let

  asterixlibDir = pkgs.fetchgit {
    url = asterixlibsRef.url;
    rev = asterixlibsRef.rev;
    sha256 = asterixlibsRef.sha256;
  };

  libasterix = pkgs.callPackage
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
    libasterix
  ];

  env = pkgs.mkShell {
    packages = [
        (pkgs.python3.withPackages (python-pkgs: deps))
    ];

    shellHook = ''
        export PYTHONPATH=$(pwd)/src:$PYTHONPATH
        ast-tool-py() { python ./src/main.py "$@"; }
    '';
    };

  drv = pkgs.python3Packages.buildPythonPackage rec {
    name = "ast-tool-py";
    format = "pyproject";
    src = ./.;
    propagatedBuildInputs = deps;
    shellHook = ''
    '';
  };

in if inShell == false
   then drv
   else if pkgs.lib.inNixShell then env else drv
