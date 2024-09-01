{ gitrev ? "devel"
, sources ? import ./nix/sources.nix
, packages ? import sources.nixpkgs {}
, inShell ? null
}:

let
  env = packages.stdenv.mkDerivation {
    name = "asterix-tool-environment";
    buildInputs = [];
    shellHook = ''
      echo "Run nix-shell inside individual sub-directory!"
      exit 1
    '';
  };

  ast-tool-py = import ./ast-tool-py/default.nix {};

  envVars = ''
  '';

  deps = [
    packages.git
  ];

  drv = packages.stdenv.mkDerivation {
    name = "ast-tool";
    preBuild = envVars;
    src = builtins.filterSource
      (path: type:
        (type != "directory" || baseNameOf path != ".git")
        && (type != "symlink" || baseNameOf path != "result"))
      ./.;
    buildInputs = deps;
    installPhase = ''
      mkdir -p $out/bin

      ln -s ${ast-tool-py}/bin/ast-tool-py $out/bin/ast-tool-py
    '';
  };

in
  if inShell == false
    then drv
    else if packages.lib.inNixShell then env else drv
