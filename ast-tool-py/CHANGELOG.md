# Changelog

## 0.20.5 (2025-05-09)

* Global argument rename `--multicast-ttl` to `--ttl`.
* Global argument rename `--multicast-if` to `--local-ip`

## 0.20.0 (2025-03-07)

* With multiple UAPs, try different decoding combinations.

## 0.19.0 (2025-02-03)

* Multicast local interface IP argument is removed from `from-udp --multicast`
  and `to-udp --multicast` commands. This setting is normally not required to
  be explicitely specified.
  Local IP address is still possible to set with the optional `--multicast-if`
  toplevel argument.

## 0.16.0 (2024-09-01)

* Switched to `asterix-libs`.

## 0.12.0 (2023-11-20)

* Intermediate format (between processes in bash pipeline) now include
  (monotonic time, utc time, channel, data).
* Custom script entry function arguments change to `def custom(ast, io, args)`.
* `sender` addribute configuration removed from `vcr` format.
* Command line arguments simplification, `--channel` argument can be appended.
* README.md file extended with examples.

