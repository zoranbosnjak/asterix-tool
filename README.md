# Asterix processing tool

A versatile asterix data related command line tool.

**Note**: Multiple implementations of the same tool are foreseen,
with the same or similar command line interface. For example:

- `ast-tool-py` - python implementation
- `ast-tool-hs` - haskell implementation

This tutorial is using a `bash` alias to refer to the specific implementation.

```bash
alias ast-tool=ast-tool-py # use 'python' version
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
ast-tool decode -h

# decode random data (deep)
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
# decode received data
ast-tool from-udp from-udp --unicast 127.0.0.1 56789 | ast-tool decode

# forward UDP from one port to another
ast-tool from-udp from-udp --unicast 127.0.0.1 56788 | ast-tool to-udp --unicast 127.0.0.1 56789

# send random data to UDP
ast-tool random | ast-tool to-udp --unicast 127.0.0.1 56789
```

## Tips and tricks

### Use bash pipe

`bash` pipe operator can be used with arbitrary stdin/stdout enabled commands.
For example, use `xxd` and `socat` external tools to send UDP datagrams.
Note: This is for demonstration purposes only. `to-udp` command offers a simpler solution.

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
This bash pipeline shall run untill interrupted.

```bash
ast-tool random | ast-tool decode --stop-on-error > /dev/null
<press CTRL-C to interrupt>
```

