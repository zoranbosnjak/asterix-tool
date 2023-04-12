# Asterix data processing tool

A versatile asterix data related command line tool.

This project uses
[asterix-lib-generator](https://zoranbosnjak.github.io/asterix-lib-generator/)
library generator, which in turn uses
[asterix-specs](https://zoranbosnjak.github.io/asterix-specs/)
definition files. To add support for additional asterix
category/edition, please contribute patch to
[asterix-specs repository](https://github.com/zoranbosnjak/asterix-specs)
and this project will inherit the change automatically.

Project dependency graph:
```
asterix-specs -> asterix-lib-generator -> asterix-tool
```

**Note**: Multiple implementations of the same asterix tool are foreseen,
with the same or similar command line interface:

- [python implementation](ast-tool-py/README.md)
- haskell implementation (work in progress)

See individual implementation subdirectory for install instructions
and details about additional commands.

This tutorial is using a `bash` alias to refer to the specific implementation.

```bash
alias ast-tool=ast-tool-py # use 'python' implementation
ast-tool -h                # get help
ast-tool {subcommand} -h   # get help for a particular subcommand
```

## `random` command

*Random asterix data generator*

### Examples

```bash
# generate random data to stdout
ast-tool random

# generate only asterix category 062, edition 1.19
ast-tool manifest # show available editions
ast-tool --empty-selection --cat 062 1.19 random

# limit number of generated samples
ast-tool random | head -n 10

# limit sample generation speed/rate
ast-tool random | while read x; do echo "$x"; sleep 0.5; done
ast-tool random | pv -qL 300

# prepend/append some string to generated samples
ast-tool random | awk '{print "0: "$1}'
```

## `decode` command

*Asterix decoder*

### Examples

```bash
# decode random data
ast-tool random | ast-tool decode

# decode random data, truncate output to 80 chars
ast-tool random | ast-tool decode --truncate 80

# decode random data, parse only up to the level of 'records'
ast-tool random | ast-tool decode --parsing-level 3
```

## `from-udp`, `to-udp` commands

*UDP datagram receiver/transmitter*

### Examples

```bash
# send random data to UDP
ast-tool random | ast-tool to-udp --unicast 127.0.0.1 56780

# forward UDP from one port to another
ast-tool from-udp --unicast 127.0.0.1 56780 | ast-tool to-udp --unicast 127.0.0.1 56781

# decode data from UDP
ast-tool from-udp --unicast 127.0.0.1 56781 | ast-tool decode
```

## `inspect-editions` command

*Detect valid/invalid asterix editions in a stream*

This command inspects a stream and tryes to decode asterix with all defined
asterix category/edition combinations. It runs until the stream is exhausted
or until the process is interrupted.

Example: inspect random samples:

```bash
ast-tool random | head -n 1000 | ast-tool inspect-editions
```

## Tips and tricks

### Use bash pipe

`bash` pipe operator can be used with arbitrary stdin/stdout enabled commands.
For example, use `xxd` and `socat` external tools to send UDP datagrams.

**Note**: This is for demonstration purposes only. `to-udp` command offers a simpler solution.

```bash
ast-tool random | while read x; do echo $x | xxd -r -p | socat -u stdin udp-sendto:127.0.0.1:59123; sleep 0.5; done
```

### Create script with arguments

For complex commands, a script with arguments can be created, for example:

```bash
touch random-udp.sh
chmod 755 random-udp.sh
cat << EOF > random-udp.sh
#!/usr/bin/env bash
sleeptime=\$1
destination=\$2
ast-tool random | while read x; do echo \$x | xxd -r -p | socat -u stdin udp-sendto:\${destination}; sleep \${sleeptime}; done
EOF
```

and run it:

```bash
./random-udp.sh 0.3 127.0.0.1:59123
```

### Run simple self test

Check if the tool can decode it's own random data, ignore decoding results.
This bash pipeline shall run without error until interrupted.

```bash
ast-tool random | ast-tool decode --stop-on-error > /dev/null
<press CTRL-C to interrupt>
```

