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

Install from python package index:

``` bash
pip install ast-tool-py
```

### Other installation methods and remarks

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
python3 -m pip --version # check version
python3 -m pip install wheel
```

Install latest git version of this project and check installation:

```bash
python3 -m pip install "git+https://github.com/zoranbosnjak/asterix-tool.git#subdirectory=ast-tool-py"
ast-tool-py --version
ast-tool-py --help
```

### Offline installation

If the target server is offline (that is: without internet access), use the
following installation procedure (tested with `ubuntu-22.04`):

- prepare installation bundle on auxilary server with internet access
- transfer files to offline server
- on the offline server, install from local disk

**NOTE**:
At the time of writing, `ubuntu-22.04` contains `pip` version `22.0.2` with a bug.
For proper operation, `pip` needs to be upgraded on both auxilary and target
server. Tested with `pip` version `23.3.1`.

#### Prepare installation bundle

Run on the server with internet access.

```bash
sudo apt -y install python3-pip
mkdir ast-tool-py-bundle
cd ast-tool-py-bundle

# check pip version, upgrade if necessary (see note above)
python3 -m pip --version
python3 -m pip install --upgrade pip
python3 -m pip --version
# download python support packages and 'ast-tool-py' package
python3 -m pip download -d . pip setuptools wheel
python3 -m pip download -d . "git+https://github.com/zoranbosnjak/asterix-tool.git#subdirectory=ast-tool-py"
```

#### Install on offline server

It is assumed that target server has `python`, `pip` and `venv` installed already.
If required, install:

```bash
sudo apt -y install python3-pip python3-venv
```

Manually transfer `ast-tool-py-bundle/` to the target server and install.

```bash
# prepare 'env'
cd
python3 -m venv env
source env/bin/activate

cd ast-tool-py-bundle

# check pip version, upgrade if necessary (see note above)
python3 -m pip --version
python3 -m pip install --upgrade --no-index ./pip*
python3 -m pip --version

# install ast-tool-py package
python3 -m pip install --no-index --find-links=./ ./ast-tool*
ast-tool-py --version
ast-tool-py --help
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
ast-tool-py from-udp --unicast "ch1" 127.0.0.1 56780 \
    | ast-tool-py to-udp --unicast "*" 127.0.0.1 56781

# decode data from UDP
ast-tool-py from-udp --unicast "ch1" 127.0.0.1 56781 | ast-tool-py decode

# distribute data by channel name (ch1 -> 56001, ch2 -> 56002)
ast-tool-py random --sleep 0.3 --channel ch1 --channel ch2 \
    | ast-tool-py to-udp \
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
ast-tool-py random --sleep 0.2 | stdbuf -oL head -n 10 \
    | ast-tool-py record --format final > recording.ff

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

This command dynamically imports a custom `python` script and runs required
function (custom script entry point). The entry point function shall accept
the following arguments:

- `base`, `gen` - base and generated asterix module (encoder/decoder), see
  python asterix library:
  [libasterix](https://github.com/zoranbosnjak/asterix-libs/tree/main/libs/python#readme)
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
def custom(base, gen, io, args):
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
def custom(base, gen, io, args):
    for event in io.rx():
        io.tx(event)
```

Test:

```bash
ast-tool-py random \
    | ast-tool-py custom --script custom.py --call custom \
    | ast-tool-py decode
```

#### Example: Channel filter

Drop events unless `channel == "ch1"`.

```python
# -- custom.py script
def custom(base, gen, io, args):
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        if channel != "ch1":
            continue
        io.tx(event)
```

```bash
# expect only 'ch1' on output
ast-tool-py random --channel ch1 --channel ch2 --channel ch3 \
    | ast-tool-py custom --script custom.py --call custom
```

#### Example: Make channel name configurable from command line

Use `args`.

```python
# -- custom.py script
def custom(base, gen, io, args):
    valid_channels = args.args.strip().split()
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        if not channel in valid_channels:
            continue
        io.tx(event)
```

Specify channels with command line argument.

```bash
# expect 'ch1' and 'ch2' on output
ast-tool-py random --channel ch1 --channel ch2 --channel ch3 \
    | ast-tool-py custom --script custom.py --call custom --args "ch1 ch2"
```

### Custom asterix processing examples

Note: This project is using
[libasterix](https://github.com/zoranbosnjak/asterix-libs/tree/main/libs/python#readme)
for asterix data processing. The `asterix` module is automatically imported
and available in custom script (it does not require separate installation step).

In general, if both `rx` and `tx` are used, custom scripts are in the form similar to
the code snipped below. User might decide to handle exceptions differently.

```python
# -- custom.py script

# custom script entry point
def custom(base, gen, io, args):
    cfg = setup(base, gen)
    for event in io.rx():
        try:
            result = handle_event(cfg, event)
        except Exception as e:
            raise Exception('problem: ', event, e)
        io.tx(result)

# prepare configuration for handle_event function
def setup(base, gen):
    return (base,gen) # for example if complete module is required

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
def custom(base, gen, io, args):
    cfg = setup(base)
    for event in io.rx():
        handle_event(cfg, event)

def setup(base):
    return (base.RawDatablock, base.Bits)

def handle_event(cfg, event):
    RawDatablock, Bits = cfg
    (t_mono, t_utc, channel, data) = event
    raw_datablocks = RawDatablock.parse(Bits.from_bytes(data))
    print(t_utc, channel, len(raw_datablocks))
```

Test:

```bash
ast-tool-py random --channel ch1 --channel ch2 --channel ch3 \
    | ast-tool-py custom --script custom.py --call custom
```

#### Example: Extract SAC/SIC code from item '010' (any category)

... ignore errors.

```python
# -- custom.py script

def custom(base, gen, io, args):
    manifest = gen.manifest
    specs = {cat: manifest['CATS'][cat][-1] for cat in manifest['CATS']}
    for event in io.rx():
        try: handle_event(base, specs, event)
        except: pass

def handle_event(base, specs, event):
    (t_mono, t_utc, channel, data) = event
    bits = base.Bits.from_bytes(data)
    raw_datablocks = base.RawDatablock.parse(bits)
    for raw_db in raw_datablocks:
        cat = raw_db.get_category()
        spec = specs.get(cat)
        records = spec.cv_uap.parse(raw_db.get_raw_records())
        for record in spec.cv_uap.parse(raw_db.get_raw_records()):
            i010 = record.get_item('010').variation
            sac = i010.get_item('SAC').as_uint()
            sic = i010.get_item('SIC').as_uint()
            print(t_utc, channel, cat, (sac, sic))
```

Test:

```bash
ast-tool-py random --channel ch1 --channel ch2 --channel ch3 \
    | ast-tool-py custom --script custom.py --call custom
```

#### Example: Search for some specific events

... for example *north marker* and *sector crossing* message in category 034.

```python
# -- custom.py script

# custom script entry point
def custom(base, gen, io, args):
    cfg = setup((base, gen))
    for event in io.rx():
        handle_event(cfg, event)

def setup(cfg):
    base, gen = cfg
    return (base, gen.Cat_034_1_29) # use cat 034, explicit edition

def handle_event(cfg, event):
    base, Spec = cfg
    (t_mono, t_utc, channel, data) = event
    # parse to raw datablocks
    bits = base.Bits.from_bytes(data)
    raw_datablocks = base.RawDatablock.parse(bits)
    for raw_db in raw_datablocks:
        # focus on one category only
        if raw_db.get_category() != Spec.cv_category:
            continue
        # fully parse raw datablock
        records = Spec.cv_uap.parse(raw_db.get_raw_records())
        for rec in records:
            handle_record(t_utc, channel, rec)

def handle_record(t_utc, channel, rec):
    msg_type = rec.get_item('000')
    if msg_type is None:
        return
    x = msg_type.as_uint()
    s = None
    if x == 1:
        s = 'north marker'
    elif x == 2:
        s = 'sector crossing'
    if s is not None:
        print(t_utc, channel, s)
```

Test:

```bash
ast-tool-py --empty-selection --cat 34 1.29 random --channel ch1 --channel ch2 --channel ch3 \
    | ast-tool-py custom --script custom.py --call custom
```

#### Example: Reverse datablocks in each datagram

```python
# -- custom.py script

# custom script entry point
def custom(base, gen, io, args):
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        data2 = handle_datagram(base, data)
        event2 = (t_mono, t_utc, channel, data2)
        io.tx(event2)

def handle_datagram(base, data):
    bits = base.Bits.from_bytes(data)
    raw_datablocks = base.RawDatablock.parse(bits)
    if len(raw_datablocks) <= 1:
        return data
    return b''.join([db.unparse().to_bytes() for db in reversed(raw_datablocks)])
```

Test:

```bash
ast-tool-py random | ast-tool-py custom --script custom.py --call custom
```

#### Example: Filter asterix by category

Accept category number as argument, drop other categories.

```python
# -- custom.py script

# custom script entry point
def custom(base, gen, io, args):
    cat = int(args.args.strip())
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        bits = base.Bits.from_bytes(data)
        lst = base.RawDatablock.parse(bits)
        lst = [db for db in lst if db.get_category() == cat]
        if not lst:
            continue
        data2 = b''.join([db.unparse().to_bytes() for db in lst])
        event2 = (t_mono, t_utc, channel, data2)
        io.tx(event2)
```

Run custom filter on random data, filte out all categories but `62`.

```bash
ast-tool-py random \
    | ast-tool-py custom --script custom.py --call custom --args 62 \
    | ast-tool-py decode --parsing-level 2 --truncate 80
```

#### Example: Modify category `062`, set `SAC/SIC` codes in item `010` to zero

Keep other items unmodified.

```python
# -- custom.py script

# custom script entry point
def custom(base, gen, io, args):
    Spec = gen.Cat_062_1_20
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        bits = base.Bits.from_bytes(data)
        lst = base.RawDatablock.parse(bits)
        lst = [handle_datablock(Spec, db) for db in lst]
        data2 = b''.join([db.unparse().to_bytes() for db in lst])
        event2 = (t_mono, t_utc, channel, data2)
        io.tx(event2)

def handle_datablock(Spec, raw_db):
    if raw_db.get_category() != Spec.cv_category:
        return raw_db
    records1 = Spec.cv_uap.parse(raw_db.get_raw_records())
    records2 = map(handle_record, records1)
    return Spec.create(records2)

def handle_record(rec):
    return rec.set_item('010', (('SAC', 0), ('SIC', 0)))
```

Test:

```bash
ast-tool-py --empty-selection --cat 62 1.20 random | \
    ast-tool-py custom --script custom.py --call custom | \
    ast-tool-py decode | grep \'010\' | grep -E 'SAC|SIC'
```

#### Example: Convert binary asterix to `json` output

This example fully decodes and converts each *event* to `json` format.
Obviously, there are multiple ways to perform such conversion, depending
on user preferences and information that needs to be preserved.
This is one example.

```python
# -- custom.py script

import json

# custom script entry point
def custom(base, gen, io, args):
    cfg = setup(base, gen, args)
    for event in io.rx():
        try:
            obj = convert_event(cfg, event)
        except Exception as e:
            raise Exception('problem: ', event, e)
        print(json.dumps(obj))

def get_selection(gen, empty, explicit_cats, explicit_refs):
    """Get category selection."""

    manifest = gen.manifest

    # get latest
    cats = {cat: manifest['CATS'][cat][-1] for cat in manifest['CATS']} # type: ignore
    refs = {cat: manifest['REFS'][cat][-1] for cat in manifest['REFS']} # type: ignore

    # cleanup if required
    if empty:
        cats = {}
        refs = {}

    # update with explicit editions
    for (a, b, c) in [
        (cats, manifest['CATS'], explicit_cats),
        (refs, manifest['REFS'], explicit_refs),
    ]:
        for (cat_i, ed_s) in c:
            cat = int(cat_i)
            ed1, ed2 = ed_s.split('.')
            ed = (int(ed1), int(ed2))
            for spec in b[cat]: # type: ignore
                if spec.cv_edition == ed:
                    a.update({cat: spec})
    return {'CATS': cats, 'REFS': refs}

def get_expansions(base, gen, selection, expansions):
    result = []
    for (cat, name) in expansions:
        cat = int(cat)
        assert cat in selection['REFS'].keys(), 'REF not defined'
        spec = selection['CATS'][cat]
        subitem = spec.cv_record.spec(name)
        assert issubclass(subitem.cv_rule.cv_variation, base.Explicit)
        result.append((cat, name))
    return result

def setup(base, gen, args):
    # use command line arguments for asterix category/edition selection
    sel = get_selection(gen, args.empty_selection, args.cat or [], args.ref or [])
    exp = get_expansions(base, gen, sel, args.expand or [])
    return (base, sel, exp)

def convert_event(cfg, event):
    """Turn 'event' to json-serializable object"""
    (t_mono, t_utc, channel, data) = event
    return {
        'tMono':    t_mono,
        'tUtc':     t_utc.isoformat(),
        'channel':  channel,
        'data':     convert_datagram(cfg, data),
    }

def convert_datagram(cfg, data):
    (base, sel, exp) = cfg
    # parse to raw datablocks
    bits = base.Bits.from_bytes(data)
    raw_datablocks = base.RawDatablock.parse(bits)
    return [convert_datablock(cfg, raw_db) for raw_db in raw_datablocks]

def convert_datablock(cfg, raw_db):
    (base, sel, exp) = cfg
    cat = raw_db.get_category()
    spec = sel['CATS'].get(cat)
    if spec is None: # asterix category unknown
        return raw_db.unparse().to_bytes().hex()
    records = spec.cv_uap.parse(raw_db.get_raw_records())
    return {
        'cat': cat,
        'records': [convert_record(cfg, cat, rec) for rec in records]
    }

def convert_record(cfg, cat, rec):
    (base, sel, exp) = cfg

    def convert_content(content):
        if isinstance(content, base.ContentRaw):
            return {'type': 'raw', 'raw': content.as_uint()}
        elif isinstance(content, base.ContentTable):
            return {'type': 'table', 'raw': content.as_uint(), 'str': content.table_value()}
        elif isinstance(content, base.ContentString):
            return {'type': 'string', 'raw': content.as_uint(), 'str': content.as_string()}
        elif isinstance(content, base.ContentInteger):
            return {'type': 'integer', 'raw': content.as_uint(), 'int': content.as_integer()}
        elif isinstance(content, base.ContentQuantity):
            return {'type': 'quantity', 'raw': content.as_uint(), 'float': content._as_quantity()
                    , 'unit': content.__class__.cv_unit}
        elif isinstance(content, base.ContentBds):
            return {'type': 'bds', 'raw': content.as_uint()}
        else:
            raise Exception('internal error, unexpected content', content)

    def convert_rulecontent(rule):
        if isinstance(rule, base.RuleContentContextFree):
            return convert_content(rule.content)
        elif isinstance(rule, base.RuleContentDependent):
            return '(content dependent structure...)'
        else:
            raise Exception('internal error, unexpected rule', rule)

    def convert_variation(var, path):
        if isinstance(var, base.Element):
            return {
                'type': 'Element',
                'content': convert_rulecontent(var.rule),
            }
        elif isinstance (var, base.Group):
            return {
                'type': 'Group',
                'items': [convert_item(i, path) for i in var.arg]
            }
        elif isinstance (var, base.Extended):
            items = []
            for lists in var.arg:
                for i in lists:
                    if i is not None:
                        items.append((convert_item(i, path),))
                    else:
                        items.append('(FX)')
            return {
                'type': 'Extended',
                'items': items,
            }
        elif isinstance (var, base.Repetitive):
            return {
                'type': 'Repetitive',
                'items': [convert_variation(sub, path+[cnt]) for (cnt, sub) in enumerate(var.arg)]
            }
        elif isinstance (var, base.Explicit):
            this_item = (cat, path[0])
            b = var.get_bytes()
            content = b.hex()
            if this_item in exp:
                sub = sel['REFS'][cat].cv_expansion
                bits = base.Bits.from_bytes(b)
                (val, remaining) = sub.parse(bits)
                assert not len(remaining)
                content = convert_expansion(val, path)
            return {
                'type': 'Explicit',
                'content': content,
            }
        elif isinstance (var, base.Compound):
            return {
                'type': 'Compound',
                'items': {name: convert_rulevariation(nsp.rule, path+[name]) for name, nsp in var.arg.items()},
            }
        else:
            raise Exception('internal error, unexpected variation', var.variation, var)

    def convert_item(item, path):
        if isinstance(item, base.Spare):
            return item.as_uint()
        elif isinstance(item, base.Item):
            nsp = item.arg
            name = nsp.__class__.cv_name
            return (name, convert_rulevariation(nsp.rule, path+[name]))
        else:
            raise Exception('internal error, unexpected item', item)

    def convert_rulevariation(rule, path):
        if isinstance(rule, base.RuleVariationContextFree):
            return convert_variation(rule.variation, path)
        elif isinstance(rule, base.RuleVariationDependent):
            return '(content dependent structure...)'
        else:
            raise Exception('internal error, unexpected rule', rule)

    def convert_expansion(var, path):
        return {name: convert_rulevariation(nsp.rule, path+[name]) for name, nsp in var.arg.items()}

    result = {}
    for ui in rec.cv_items_list:
        if not issubclass(ui, base.UapItem):
            continue
        name = ui.cv_non_spare.cv_name
        nsp = rec.get_item(name)
        if nsp is None:
            continue
        result[name] = convert_rulevariation(nsp.rule, [name])
    return result
```

As an example:

- use random input data
- use latest edition for each known category
- in addition, decode cat021 'RE' expansion field

```bash
sudo apt install jq
ast-tool-py --expand 21 RE random --seed 0 --populate-all-items \
    | ast-tool-py --expand 21 RE custom --script custom.py --call custom \
    | jq
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
def cleanup(base, gen, io, args):
    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        bits = base.Bits.from_bytes(data)
        result = base.RawDatablock.parse(bits)
        if isinstance(result, ValueError): # skip non-asterix
            continue
        io.tx(event)

# restamp entry point
def restamp(base, gen, io, args):
    # for each category specify (edition, item to modify)
    updates = {
        1: (gen.Cat_001_1_4, "141"),
        2: (gen.Cat_002_1_1, '030'),
        19: (gen.Cat_019_1_3, '140'),
        20: (gen.Cat_020_1_10, '140'),
        34: (gen.Cat_034_1_29, '030'),
        48: (gen.Cat_048_1_32, '140'),
        # add more categories here...
    }

    for event in io.rx():
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        (t_mono, t_utc, channel, data) = event
        bits = base.Bits.from_bytes(data)
        result = base.RawDatablock.parse(bits)
        if isinstance(result, ValueError):
            continue
        raw_datablocks = [db for db in result if db.get_category() in updates]
        if not raw_datablocks:
            continue
        t2 = seconds_since_midnight(t_utc)
        t3 = seconds_since_midnight(now)
        lst = [handle_datablock(t2, t3, updates, db) for db in raw_datablocks]
        data2 = b''.join([db.unparse().to_bytes() for db in lst])
        event2 = (t_mono, now, channel, data2)
        io.tx(event2)

def seconds_since_midnight(t):
    """Calculate seconds since midnight."""
    midnight = datetime.datetime.combine(t, datetime.time(0), t.tzinfo)
    dt = t - midnight
    return dt.total_seconds()

def handle_datablock(t2, t3, updates, raw_db):
    cat = raw_db.get_category()
    (Spec, name) = updates.get(cat)
    result = Spec.cv_uap.parse(raw_db.get_raw_records())
    assert not isinstance(result, ValueError)
    records = [handle_record(t2, t3, rec, name) for rec in result]
    return Spec.create(records)

def handle_record(t2, t3, rec, name):
    # compensate original delay/jitter from the recording
    def stamp(t1):
        t1 = t1.variation.content.as_quantity()
        original_delay = t2 - t1
        return t3 - original_delay
    x = rec.get_item(name)
    if x is None:
        return rec
    return rec.set_item(name, stamp(x))
```

Test:

```bash
# create recording file
ast-tool-py random --seed 0 --sleep 0.05 --channel ch1 --channel ch2 \
    | stdbuf -oL head -n 100 \
    | ast-tool-py record | tee recording

# inspect recording file, make sure to use valid editions in custom script
cat recording | ast-tool-py replay | ast-tool-py inspect

# replay, restamp, resend
cat recording | ast-tool-py replay \
    | ast-tool-py custom --script custom.py --call cleanup \
    | ast-tool-py custom --script custom.py --call restamp \
    | ast-tool-py to-udp \
        --unicast ch1 127.0.0.1 56001 \
        --unicast ch2 127.0.0.1 56002
```
