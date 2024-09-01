# Asterix data processing tool

*A versatile asterix data related command line tool*

This project uses
* [asterix-libs](https://github.com/zoranbosnjak/asterix-libs#readme)
library collection, which in turn uses
[asterix-specs](https://zoranbosnjak.github.io/asterix-specs/)
definition files. To add support for additional asterix
category/edition, please contribute patch to
[asterix-specs repository](https://github.com/zoranbosnjak/asterix-specs)
and this project will inherit the change automatically.

## python implementation

For install instructions, features list, usage details and examples see
[ast-tool-py](ast-tool-py/README.md).

## Subcommand composition with bash

Subcommands (`random`, `decode`, ...) can be composed in different
ways by using `bash` pipe operator. For example:

```bash
ast-tool-py random | head -n 3 | ast-tool-py decode -l 4 --truncate 80
```

### Buffering remark

Some bash commands (like `head`) are using buffering mode when they are
executed in the middle of a pipeline. To force line-buffered mode for such
commands, the `stdbuf -oL` can be used.

Examples:

```bash
# timing to the terminal is correct
ast-tool-py random --sleep 0.2

# timing to the terminal is correct (output of 'head' is a terminal)
ast-tool-py random --sleep 0.2 | head -n 10

# timing to the terminal is NOT correct (delayed)
# output of the first 'head' command is now buffered
ast-tool-py random --sleep 0.2 | head -n 10 | head -n 10

# 'stdbuf' can fix the problem, timing is again correct in this case
ast-tool-py random --sleep 0.2 | stdbuf -oL head -n 10 | head -n 10
```

## Bash script with arguments

For complex commands, a script with arguments can be created. For example:

```bash
#!/usr/bin/env bash
# 'random-udp.sh' script
# composing `random` and `to-udp` subcommands with command line arguments
sleeptime=$1
dst_ip=$2
dst_port=$3
ast-tool-py random --sleep $sleeptime | ast-tool-py to-udp --unicast "*" $dst_ip $dst_port
```

Run script:

```bash
chmod 755 random-udp.sh
./random-udp.sh 0.3 127.0.0.1 59123
```
