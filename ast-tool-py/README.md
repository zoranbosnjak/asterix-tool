# Asterix processing tool - python version

## Installation

Install latest git version to virtual environment

```bash
python3 -m venv env
source env/bin/activate
pip install "git+https://github.com/zoranbosnjak/asterix-tool.git#subdirectory=ast-tool-py"
ast-tool-py -h
```

## Development

```bash
nix-shell
./update-from-upstream.sh
python3 ./src/main.py --version
# or using alias
ast-tool --version
ast-tool -h
exit
```

## Running custom python script

`ast-tool-py custom` command supports dynamic import of custom `python` script.

The `custom` command runs provided python script/function with the following arguments:
- `ast` - generated asterix module, see: [generated python library](https://zoranbosnjak.github.io/asterix-lib-generator/python.html)
- `rx` - stream receiver (used to fetch data)
- `tx` - stream transmit function
- `args` - additional command line arguments (string)

Custom filter example:
- Receive data from UDP unicast
- Restamp asterix data with current time (update items [I019, 140] and [I020, 140]),
  if present.
- Set [I020, 020, TST] bit to '1', if present.

```bash
ast-tool-py from-udp --unicast 127.0.0.1 56780 | ast-tool-py custom --script restamp.py --call restamp
```

```python
# --- restamp.py script
import datetime

def update_tst(x):
    """Set 'TST' bit to '1' if present, ignore previous value."""
    return x.modify_item('TST', lambda y: 1)

def process_record(t, cat, rec):
    """Process single record, category 19 or 20."""
    if cat == 20:
        rec = rec.modify_item('020', update_tst)
    rec = rec.modify_item('140', lambda x: t)
    return rec

def process_datablock(t, ast, raw_db):
    """Process single raw datablock."""
    opt = ast.ParsingOptions.default()

    Spec019 = ast.CAT_019_1_3
    Spec020 = ast.CAT_020_1_10

    if raw_db.category == Spec019.cat:
        Spec = Spec019
    elif raw_db.category == Spec020.cat:
        Spec = Spec020
    else:
        return(raw_db)
    db = Spec.parse(raw_db, opt)
    return Spec.make_datablock([process_record(t, Spec.cat, rec) for rec in db.records])

# custom filter entry point function
def restamp(ast, rx, tx, args):
    for line in rx:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        midnight = datetime.datetime.combine(now, datetime.time(0), now.tzinfo)
        t = now - midnight
        s = bytes.fromhex(line)
        raw_datablocks = ast.RawDatablock.parse(s)
        result = [process_datablock(t.total_seconds(), ast, db) for db in raw_datablocks]
        output = b''.join([db.unparse() for db in result])
        tx(output.hex())
```

