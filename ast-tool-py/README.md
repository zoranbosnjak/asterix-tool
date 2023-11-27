# Asterix processing tool - python version

Features:

- random asterix data generator
- asterix decoder to text output
- UDP receiver and transmitter with multicast mode support
- asterix category/edition detector
- stream recorder and replay
- support for multiple recording file formats
- simple integration with other standard command line tools via stdin/stdout
- user defined asterix data processing with custom script

## Installation

This installation procedures requires `python >= 3.7` and `pip >= 21.3`.
Tested under `ubuntu-22.04` and `ubuntu-20.04`.

Prepare virtual environment:

```bash
sudo apt -y install python3-venv
python3 -m venv env
source env/bin/activate
python3 -m pip install wheel # might be required (e.g. under ubuntu 20.04)
```

Under some older OS versions (like ubuntu-18.04) it might be necessary to upgrade the
the required versions first. In this case, the procedure to prepare the environment
should be something like:

```bash
sudo apt -y install python3.8 python3.8-venv python3-pip
python3.8 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python --version # check version
pip --version # check version
python3 -m pip install wheel
```

Install latest git version of this project and check installation:

```bash
python3 -m pip install "git+https://github.com/zoranbosnjak/asterix-tool.git#subdirectory=ast-tool-py"
ast-tool-py --version
ast-tool-py --help
```

## Development

Use `nix-shell` to setup development environment

```bash
nix-shell
./update-from-upstream.sh
python3 ./src/main.py --version
alias ast-tool-py='python3 ./src/main.py'
ast-tool-py --version
exit
```

## Common arguments

### Category, edition and expansion selection

Some commands (`random`, `decode`...) need selection of asterix categories
and edition. By default all available categories and the latest editions
are used. The following command line arguments are available to adjust the
setting (can be specified multiple times).

- `--empty-selection` - start with empty selection instead of *all latest*
- `--cat CAT EDITION` - add/replace category with explicit edition
- `--ref CAT EDITION` - add/replace expansion with explicit edition
- `--expand CAT ITEM-NAME` - Use expansion definition with selected topitem,
  for example `--expand 62 RE`

### Data flow operating mode

The following options are available when program is used in a
*bash pipeline*:

- `--simple-input` - force simple mode for data input
- `--simple-output` - force simple mode for data output
- `-s` or `--simple` - force simple mode for data input and output
  (this is the same as setting `--simple-input` and `--simple-output`)
- `--no-flush` - do not flush output on each event (use buffering)

Examples:

```bash
# generate random output stream (CTRL-C to terminate)
ast-tool-py random

# generate random output with simple output mode
ast-tool-py --simple-output random

# next process must use '--simple-input', to match
ast-tool-py --simple-output random | ast-tool-py --simple-input decode
# ... or using the short form
ast-tool-py -s random | ast-tool-py -s decode
```

## Getting help

```bash
ast-tool-py --help                   # general help
ast-tool-py {subcommand-name} --help # subcommand help
```

## Subcommands

### `random` command

*Pseudo random asterix data generator*

Examples:

```bash
# generate random data
ast-tool-py random

# limit number of generated samples
ast-tool-py random | stdbuf -oL head -n 10

# random streem is different on each program run
# try multiple times, expect different result
ast-tool-py -s random | stdbuf -oL head -n 10 | sha1sum

# unless a seed value is fixed in which case
# try multiple times, expect same result
ast-tool-py -s random --seed 0 | stdbuf -oL head -n 10 | sha1sum

# generate only asterix category 062, edition 1.19
ast-tool-py manifest | grep "062" # show available editions
ast-tool-py --empty-selection --cat 062 1.19 random

# generate all categories and for cat062, generate 'RE' expansion field too
ast-tool-py --expand 62 RE random

# limit sample generation speed/rate (various options)
ast-tool-py random --sleep 0.5
ast-tool-py random | while read x; do echo "$x"; sleep 0.5; done
ast-tool-py random | pv -qL 300

# prepend/append some string to generated samples in simple format
ast-tool-py -s random | awk '{print "0: "$1}'

# set random channel name for each event, choose from ['ch1', 'ch2']
ast-tool-py random --channel ch1 --channel ch2
```

### `decode` command

*Asterix decoder to plain text*

Examples:

```bash
# decode random data
ast-tool-py random | ast-tool-py decode

# decode random data, truncate output to 80 chars
ast-tool-py random | ast-tool-py decode --truncate 80

# decode, truncate, parse only up to the level of 'records' or 'items
ast-tool-py random | ast-tool-py decode --truncate 80 --parsing-level 3
ast-tool-py random | ast-tool-py decode --truncate 80 --parsing-level 4

# generate and decode 'RE' expansion field too
ast-tool-py --expand 62 RE random | ast-tool-py --expand 62 RE decode
```

Run simple self-test:

Check if the tool can decode it's own random data, ignore decoding results.
This bash pipeline shall run without error until interrupted.

```bash
ast-tool-py random | ast-tool-py decode --stop-on-error > /dev/null
# press CTRL-C to interrupt
```

### `from-udp`, `to-udp` commands

*UDP datagram receiver/transmitter*

Examples:

```bash
# send random data to UDP
ast-tool-py random | ast-tool-py to-udp --unicast "*" 127.0.0.1 56780

# forward UDP from one port to another
ast-tool-py from-udp --unicast "ch1" 127.0.0.1 56780 | \
    ast-tool-py to-udp --unicast "*" 127.0.0.1 56781

# decode data from UDP
ast-tool-py from-udp --unicast "ch1" 127.0.0.1 56781 | ast-tool-py decode

# distribute data by channel name (ch1 -> 56001, ch2 -> 56002)
ast-tool-py random --sleep 0.3 --channel ch1 --channel ch2 |
    ast-tool-py to-udp \
        --unicast "ch1" 127.0.0.1 56001 \
        --unicast "ch2" 127.0.0.1 56002

# monitor result on individual UDP ports
ast-tool-py from-udp --unicast "ch1" 127.0.0.1 56001
ast-tool-py from-udp --unicast "ch2" 127.0.0.1 56002
```

### `inspect` command

*Detect valid/invalid asterix editions in a stream*

This command inspects a stream and tryes to decode asterix with all defined
asterix category/edition combinations. It runs until the stream is exhausted
or until the process is interrupted.

Examples:

```bash
# inspect random samples:
ast-tool-py random | stdbuf -oL head -n 1000 | ast-tool-py inspect

# inspect network traffic (CTRL-C to stop after some observation time)
ast-tool-py from-udp --unicast "ch1" 127.0.0.1 56780 | ast-tool-py inspect
```

### `record`, `replay` commands

*Record/replay data to/from a file*

Examples:

```bash
# save random data
ast-tool-py random --sleep 0.2 | stdbuf -oL head -n 10 | ast-tool-py record
# ... to a file
ast-tool-py random --sleep 0.2 | stdbuf -oL head -n 10 | ast-tool-py record | tee recording.simple
ast-tool-py random --sleep 0.2 | stdbuf -oL head -n 10 | ast-tool-py record > recording.simple

# use binary final file format
ast-tool-py record --help # check supported recording file formats
ast-tool-py random --sleep 0.2 | stdbuf -oL head -n 10 | \
    ast-tool-py record --format final > recording.ff

# replay at normal/full speed
ast-tool-py replay recording.simple
ast-tool-py replay recording.simple --full-speed

# use different replay file format
ast-tool-py replay --help # check supported replay file formats
ast-tool-py replay --format final recording.ff

# replay from gzipped file (not supported with 'pcap' format)
gzip recording.simple
cat recording.simple.gz | gunzip | ast-tool-py replay
zcat recording.simple.gz | ast-tool-py replay # or using 'zcat'
```

### `custom` command

*Running custom python script*

This command dynamically imports a custom `python` script and runs required function
(custom script entry point). The entry point function shall accept the following arguments:

- `ast` - asterix module (encoder/decoder), see:
  [asterix python library](https://zoranbosnjak.github.io/asterix-lib-generator/python.html)
- `io` - standard input/output instance
- `args` - program arguments

Custom script can use:

- `io.rx` to fetch events as data *consumer*,
  for example to decode and display asterix data in any data-serialization
  format (json, xml, bson...)
- `io.tx` to generate events as data *producer*,
  for example: reading and parsing non-standard recording file from disk
- both `io.rx` and `io.tx` to act as custom data *filter* with arbitrary
  data manipulation capabilities

#### Minimal example

```python
# -- custom.py script
def custom(ast, io, args):
    print("Hello from custom script!")
    print(args.args) # explicit arguments
    print(args)      # all program arguments
```

Test:

```bash
ast-tool-py custom --script custom.py --call custom --args "additional arguments, any string"
```

#### Example: transparent filter

Basic filtering loop (transparent filter), use `io.rx` and `io.tx`.

```python
# -- custom.py script
def custom(ast, io, args):
    for event in io.rx():
        io.tx(event)
```

#### Example: Channel filter

Drop events unless `channel == "ch1"`.

```python
# -- custom.py script
def custom(ast, io, args):
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        if channel != "ch1":
            continue
        io.tx(event)
```

```bash
# expect only 'ch1' on output
ast-tool-py random --channel ch1 --channel ch2 --channel ch3 | \
    ast-tool-py custom --script custom.py --call custom
```

#### Example: Make channel name configurable from command line

Use `args`.

```python
# -- custom.py script
def custom(ast, io, args):
    valid_channels = args.args.strip().split()
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        if not channel in valid_channels:
            continue
        io.tx(event)
```

Specify channels in command line argument.

```bash
# expect 'ch1' and 'ch2' on output
ast-tool-py random --channel ch1 --channel ch2 --channel ch3 | \
    ast-tool-py custom --script custom.py --call custom --args "ch1 ch2"
```

### Custom asterix processing examples

Note: This project is using
[python asterix library](https://zoranbosnjak.github.io/asterix-lib-generator/python.html)
for asterix data processing. The `asterix` module is automatically imported and available
in custom script (it does not require separate installation step).

In general, if both `rx` and `tx` are used, custom scripts are in the form similar to
the code snipped below. User might decide to handle exceptions differently.

```python
# -- custom.py script

# custom script entry point
def custom(ast, io, args):
    cfg = setup(ast)
    for event in io.rx():
        try:
            result = handle_event(cfg, event)
        except Exception as e:
            raise Exception('problem: ', event, e)
        io.tx(result)

# prepare configuration for handle_event function
def setup(ast):
    return ast # for example if complete module is required

# actual event handler
def handle_event(cfg, event):
    (t_mono, t_utc, channel, data) = event
    # process data in some way...
    return event # possible modified event
```

#### Example: Print number of datablocks per datagram

```python
# -- custom.py script

# custom script entry point
def custom(ast, io, args):
    cfg = setup(ast)
    for event in io.rx():
        handle_event(cfg, event)

def setup(ast):
    return ast.RawDatablock.parse

def handle_event(parse, event):
    (t_mono, t_utc, channel, data) = event
    raw_datablocks = parse(data)
    print(t_utc, channel, len(raw_datablocks))
```

#### Example: Search for some specific events

... for example *north marker* and *sector crossing* message in category 034.

```python
# -- custom.py script

# custom script entry point
def custom(ast, io, args):
    cfg = setup(ast)
    for event in io.rx():
        handle_event(cfg, event)

def setup(ast):
    opt = ast.ParsingOptions.default()
    return (ast, opt)

def handle_event(cfg, event):
    (ast, opt) = cfg
    (t_mono, t_utc, channel, data) = event
    # parse to raw datablocks
    raw_datablocks = ast.RawDatablock.parse(data)
    Spec = ast.CAT_034_1_29 # use cat 034, explicit edition
    for raw_db in raw_datablocks:
        # focus on one category only
        if raw_db.category != Spec.cat:
            continue
        # fully parse raw datablock
        db = Spec.parse(raw_db, opt)
        for rec in db.records:
            handle_record(t_utc, channel, rec)

def handle_record(t_utc, channel, rec):
    msg_type = rec.get_item('000')
    if msg_type is None:
        return
    x = msg_type.to_uinteger()
    s = None
    if x == 1:
        s = 'north marker'
    elif x == 2:
        s = 'sector crossing'
    if s is not None:
        print(t_utc, channel, s)
```

#### Example: Reverse datablocks in each datagram

```python
# -- custom.py script

# custom script entry point
def custom(ast, io, args):
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        data2 = handle_datagram(ast, data)
        event2 = (t_mono, t_utc, channel, data2)
        io.tx(event2)

def handle_datagram(ast, data):
    raw_datablocks = ast.RawDatablock.parse(data)
    if len(raw_datablocks) <= 1:
        return data
    return b''.join([db.unparse() for db in reversed(raw_datablocks)])

```

#### Example: Filter asterix by category

Accept category number as argument, drop other categories.

```python
# -- custom.py script

# custom script entry point
def custom(ast, io, args):
    cat = int(args.args.strip())
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        lst = ast.RawDatablock.parse(data)
        lst = [db for db in lst if db.category == cat]
        if not lst:
            continue
        data2 = b''.join([db.unparse() for db in lst])
        event2 = (t_mono, t_utc, channel, data2)
        io.tx(event2)
```

Run custom filter on random data, filte out all categories but `62`.

```bash
ast-tool-py random | ast-tool-py custom --script custom.py --call custom --args 62
```

#### Example: Modify category `062`, set `SAC/SIC` codes in item `010` to zero

Keep other items unmodified.

```python
# -- custom.py script

# custom script entry point
def custom(ast, io, args):
    opt = ast.ParsingOptions.default()
    Spec = ast.CAT_062_1_20
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        lst = ast.RawDatablock.parse(data)
        lst = [handle_datablock(opt, Spec, db) for db in lst]
        data2 = b''.join([db.unparse() for db in lst])
        event2 = (t_mono, t_utc, channel, data2)
        io.tx(event2)

def handle_datablock(opt, Spec, raw_db):
    if raw_db.category != Spec.cat:
        return raw_db
    db = Spec.parse(raw_db, opt)
    records = [handle_record(rec) for rec in db.records]
    return Spec.make_datablock(records)

def handle_record(rec):
    return rec.modify_item('010', lambda _old: {'SAC': 0, 'SIC': 0})
```

Check

```bash
ast-tool-py --empty-selection --cat 62 1.20 random | \
    ast-tool-py custom --script custom.py --call custom | \
    ast-tool-py decode | grep \'010\' | grep -E 'SAC|SIC'
```

#### Example: Convert binary asterix to `json` output

This example fully decodes and converts each *event* to `json` format.
Obviously, there are multiple ways to perform such conversion. This is one example.

```python
# -- custom.py script

import json

# custom script entry point
def custom(ast, io, args):
    cfg = setup(ast, args)
    for event in io.rx():
        try:
            obj = handle_event(cfg, event)
        except Exception as e:
            raise Exception('problem: ', event, e)
        print(json.dumps(obj))

def string_to_edition(ed):
    """Convert edition string to a tuple, for example "1.2" -> (1,2)"""
    a,b = ed.split('.')
    return (int(a), int(b))

def get_selection(ast, empty, explicit_cats, explicit_refs):
    """Get category selection."""

    def get_latest(lst):
        return sorted(lst, key=lambda pair: string_to_edition(pair[0]), reverse=True)[0]

    # get latest
    cats = {cat: get_latest(ast.manifest['CATS'][cat].items())[1] for cat in ast.manifest['CATS'].keys()}
    refs = {cat: get_latest(ast.manifest['REFS'][cat].items())[1] for cat in ast.manifest['REFS'].keys()}

    # cleanup if required
    if empty:
        cats = {}
        refs = {}

    # update with explicit editions
    for (a,b,c) in [
        (cats, 'CATS', explicit_cats),
        (refs, 'REFS', explicit_refs),
        ]:
        for (cat,ed) in c:
            cat = int(cat)
            a.update({cat: manifest[b][cat][ed]})
    return {'CATS': cats, 'REFS': refs}

def get_expansions(ast, selection, expansions):
    result = []
    for (cat, name) in expansions:
        cat = int(cat)
        assert cat in selection['REFS'].keys(), 'REF not defined'
        spec = selection['CATS'][cat]
        subitem = spec.variation.spec(name)
        assert issubclass(subitem, ast.Explicit)
        result.append((cat, name))
    return result

def setup(ast, args):
    # use command line arguments for asterix category/edition selection
    sel = get_selection(ast, args.empty_selection, args.cat or [], args.ref or [])
    exp = get_expansions(ast, sel, args.expand or [])
    opt = ast.ParsingOptions.default()
    return (ast, sel, exp, opt)

def handle_event(cfg, event):
    """Turn 'event' to json-serializable object"""
    (t_mono, t_utc, channel, data) = event
    return {
        'tMono':    t_mono,
        'tUtc':     t_utc.isoformat(),
        'channel':  channel,
        'data':     handle_datagram(cfg, data),
    }

def handle_datagram(cfg, data):
    (ast, sel, exp, opt) = cfg
    # parse to raw datablocks
    raw_datablocks = ast.RawDatablock.parse(data)
    return [handle_datablock(cfg, raw_db) for raw_db in raw_datablocks]

def handle_datablock(cfg, raw_db):
    (ast, sel, exp, opt) = cfg
    cat = raw_db.category
    spec = sel['CATS'].get(cat)
    if spec is None: # asterix category unknown
        return raw_db.unparse().hex()
    db = spec.parse(raw_db, opt)
    return {
        'cat': cat,
        'records': [handle_record(cfg, cat, rec) for rec in db.records]
    }

def handle_record(cfg, cat, rec):
    (ast, sel, exp, opt) = cfg

    def handle_variation(var, path):
        cls = var.__class__
        if isinstance(var, ast.Element):
            raw = var.to_uinteger()
            t = 'raw'
            val = None
            if hasattr(var, 'table_value'):
                t = 'table'
                tv = var.table_value
                val = tv if tv is not None else 'undefined value'
            elif hasattr(var, 'to_string'):
                t = 'string'
                val = var.to_string()
            elif hasattr(var, 'to_quantity'):
                t = 'quantity'
                val = (var.to_quantity(), var.__class__.quantity.unit)
            return {
                'raw': raw,
                'type': t,
                'value': val
                }
        elif isinstance (var, ast.Group):
            result = {}
            for i in cls.subitems_list:
                # regular subitem
                if type(i) is tuple:
                    name = i[0]
                    sub = var.get_item(name)
                    result[name] = handle_variation(sub, path+[name])
                # spare subitem
                else:
                    pass
            return result
        elif isinstance (var, ast.Extended):
            result = {}
            for j in cls.subitems_list:
                for k in j:
                    if type(k) is tuple:
                        name = k[0]
                        sub = var.get_item(name)
                        if sub is None:
                            continue
                        result[name] = handle_variation(sub, path+[name])
                    else: # spare
                        pass
            return result
        elif isinstance (var, ast.Repetitive):
            return [handle_variation(sub, path+[cnt]) for (cnt, sub) in enumerate(var)]
        elif isinstance (var, ast.Explicit):
            this_item = (cat, path[0])
            if not this_item in exp:
                # can not parse explicit item, return raw value
                return var.raw.hex()
            sub = sel['REFS'][cat].variation
            bits = ast.Bits.from_bytes(var.raw)
            (val, b) = sub.parse_bits(bits, opt)
            if len(b):
                raise ast.AsterixError('Unexpected remaining bits in explicit item')
            return handle_variation(val, path)
        elif isinstance (var, ast.Compound):
            result = {}
            for i in cls.subitems_list:
                if i is None:
                    continue
                name = i[0]
                sub = var.get_item(name)
                if sub is None:
                    continue
                result[name] = handle_variation(sub, path+[name])
            return result
        else:
            raise Exception('internal error, unexpected variation', var.variation, var)

    return handle_variation(rec, [])
```

As an example:

- use random input data
- use latest edition for each known category
- in addition, decode cat021 'RE' expansion field

```bash
ast-tool-py random | ast-tool-py --expand 21 RE custom --script custom.py --call custom
```

#### Restamp asterix data to current UTC time

Scenario:

- read data from recording file, filter required channels
- filter out non-asterix data
- restamp several asterix categories to current time
- send to udp destination, according to channel name mapping

```python
# -- custom.py script

import datetime

# cleanup entry point
def cleanup(ast, io, args):
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        try:
            raw_datablocks = ast.RawDatablock.parse(data)
        except: # skip non-asterix
            continue
        io.tx(event)

# restamp entry point
def restamp(ast, io, args):
    opt = ast.ParsingOptions.default()
    # for each category specify (edition, item to modify)
    updates = {
        1: (ast.CAT_001_1_4, "141"),
        2: (ast.CAT_002_1_1, '030'),
        19: (ast.CAT_019_1_3, '140'),
        20: (ast.CAT_020_1_10, '140'),
        34: (ast.CAT_034_1_29, '030'),
        48: (ast.CAT_048_1_31, '140'),
        # add more categories here...
    }

    for event in io.rx():
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        (t_mono, t_utc, channel, data) = event
        raw_datablocks = [db for db in ast.RawDatablock.parse(data) if db.category in updates]
        if not raw_datablocks:
            continue
        t2 = seconds_since_midnight(t_utc)
        t3 = seconds_since_midnight(now)
        lst = [handle_datablock(t2, t3, opt, updates, db) for db in raw_datablocks]
        data2 = b''.join([db.unparse() for db in lst])
        event2 = (t_mono, now, channel, data2)
        io.tx(event2)

def seconds_since_midnight(t):
    """Calculate seconds since midnight."""
    midnight = datetime.datetime.combine(t, datetime.time(0), t.tzinfo)
    dt = t - midnight
    return dt.total_seconds()

def handle_datablock(t2, t3, opt, updates, raw_db):
    cat = raw_db.category
    (Spec, name) = updates.get(cat)
    db = Spec.parse(raw_db, opt)
    records = [handle_record(t2, t3, rec, name) for rec in db.records]
    return Spec.make_datablock(records)

def handle_record(t2, t3, rec, name):
    # compensate original delay/jitter from the recording
    def stamp(t1):
        t1 = t1.to_quantity()
        original_delay = t2 - t1
        return t3 - original_delay
    return rec.modify_item(name, stamp)
```

Run command:

```bash
cat recording.ff.gz | gunzip | ast-tool-py replay \
        --channel ch1 \
        --channel ch2 \
        --format final | \
    ast-tool-py custom --script custom.py --call cleanup | \
    ast-tool-py custom --script custom.py --call restamp | \
    ast-tool-py to-udp \
        --unicast ch1 127.0.0.1 56001 \
        --unicast ch2 127.0.0.1 56002
```

