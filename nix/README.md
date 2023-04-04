# Update procedures

# sources.json

```bash
# run from parent directory
nix-shell -p niv --run "niv update"
# or
nix-shell -p niv --run "niv update nixpkgs -b master"
# or (check current release here https://nixos.org/download.html)
nix-shell -p niv --run "niv update nixpkgs -b release-22.11"
```

