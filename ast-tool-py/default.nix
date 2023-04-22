{ sources ? import ../nix/sources.nix
, packages ? import sources.nixpkgs {}
}:

let
  deps = with packages; [
    python3
    python3Packages.setuptools
    python3Packages.scapy
  ];

in packages.python3Packages.buildPythonPackage rec {
  name = "ast-tool-py";
  format = "pyproject";
  src = ./.;
  propagatedBuildInputs = deps;
}

