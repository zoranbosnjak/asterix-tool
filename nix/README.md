# Update procedures

## sources.json

```bash
# run from parent directory
nix-shell -p niv --run "niv update"
# or
nix-shell -p niv --run "niv update nixpkgs -b master"
# or (check current release here https://nixos.org/download.html)
nix-shell -p niv --run "niv update nixpkgs -b release-..."
```

## asterix-libs

```bash
nix-prefetch-git https://github.com/zoranbosnjak/asterix-libs.git \
    > nix/asterix-libs.json
# or
hsh=...
nix-prefetch-git --rev $hsh https://github.com/zoranbosnjak/asterix-libs.git \
    > nix/asterix-libs.json
```
