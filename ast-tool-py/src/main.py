#!/usr/bin/env python3

# Asterix data processing tool

from asterix.base import *
import asterix.base as base
import locale
import json
import uuid
import selectors
import struct
import socket
import datetime
import time
import importlib.metadata
import importlib.util
import sys
import os
import random
import fileinput
import argparse
from typing import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# skip some imports (and features) if fast startup is required
fast = len(sys.argv) >= 2 and sys.argv[1] == '--fast-startup'
if fast:
    generated_orig: Optional[Any] = None
else:
    import asterix.generated as generated_orig
    from scapy.all import rdpcap, UDP  # type: ignore

__version__ = "0.27.4"

# Import module from some source path


@no_type_check
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 'Event' in this context is a tuple, containing:
#   - monotonic time
#   - UTC time
#   - channel name
#   - actual data bytes


Event: TypeAlias = Tuple[float,
                         Optional[datetime.datetime],
                         Optional[str],
                         bytes]


class CIO:
    """Input/output helper class, handles broken pipe exception and automatic flush.
    """

    def __init__(self, s_in: bool, s_out: bool, flush: bool) -> None:
        self.simple_input = s_in
        self.simple_output = s_out
        self.flush = flush

    def rx(self) -> Generator[Event, None, None]:
        for line in fileinput.input('-'):
            if self.simple_input:
                t_mono = time.monotonic()
                t_utc = datetime.datetime.now(tz=datetime.timezone.utc)
                channel = None
                data = bytes.fromhex(line.strip())
            else:
                o = json.loads(line)
                t_mono = o['tMono']
                t_utc = datetime.datetime.fromisoformat(o['tUtc'])
                channel = o['channel']
                data = bytes.fromhex(o['data'])
            yield (t_mono, t_utc, channel, data)

    def tx_raw(self, s: str) -> None:
        try:
            print(s, flush=self.flush)
        except BrokenPipeError:
            sys.stdout = None
            sys.exit(0)

    def tx_raw_bin(self, s: bytes) -> None:
        try:
            sys.stdout.buffer.write(s)
            if self.flush:
                sys.stdout.buffer.flush()
        except BrokenPipeError:
            sys.stdout = None
            sys.exit(0)

    def tx(self, event: Event) -> None:
        (t_mono, t_utc, channel, data) = event
        if self.simple_output:
            s = data.hex()
        else:
            s = json.dumps({
                "tMono": t_mono,
                "tUtc": t_utc.isoformat() if t_utc is not None else None,
                "channel": channel,
                "data": data.hex(),
            })
        self.tx_raw(s)

    def tx_simple(self, data: bytes, channel: Optional[str] = None) -> None:
        """Simple wrapper around 'tx' function."""
        t_mono = time.monotonic()
        t_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        event = (t_mono, t_utc, channel, data)
        self.tx(event)


class PCG32:
    """Simple random number generator,
    based on https://www.pcg-random.org/download.html"""

    top64: int = pow(2, 64)
    top32: int = pow(2, 32)

    def mod64(self, val: int) -> int: return val % self.__class__.top64
    def mod32(self, val: int) -> int: return val % self.__class__.top32

    def __init__(self, state: int = 0, inc: int = 0) -> None:
        self.state = self.mod64(state)
        self.inc = self.mod64(inc)

    def next(self) -> int:
        """Generate next 32-bit random value."""
        oldstate = self.state
        self.state = self.mod64(
            oldstate * 6364136223846793005 + (self.inc | 1))
        xorshifted = self.mod32(((oldstate >> 18) ^ oldstate) >> 27)
        rot = oldstate >> 59
        return self.mod32((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))

    def choose(self, lst: List[Any]) -> Any:
        """Choose list element."""
        ix = self.next() % len(lst)
        return lst[ix]

    def bool(self) -> bool:
        """Generate random bool value."""
        return bool(self.next() % 2)

    def bigint(self, bitsize: int) -> int:
        """Generate random number of arbitrary bitsize."""
        val = self.next()
        size = 32
        while True:
            if size >= bitsize:
                return val % pow(2, bitsize)  # type: ignore
            size += 32
            val = val * pow(2, 32) + self.next()


class Fmt:
    """File format base class.
    Each subclass shall define one or both:
        - 'write_event' method (to support recording)
        - 'events' generator (to support replay)
    """
    name: str

    def __init__(self, io: CIO):
        self.io = io
        self.on_init()

    def on_init(self) -> None:
        pass


class FmtSimple(Fmt):
    """Simple line oriented data format:
       {timestamp-iso} {hex data} {channel} {time-mono-ns}"""

    name = 'simple'

    def write_event(self, event: Event) -> None:
        (t_mono, t_utc, channel, data) = event
        if not channel:
            channel = '-'
        t_mono_ns = round(t_mono * 1_000_000_000)
        t_utc_iso = t_utc.isoformat() if t_utc is not None else None
        self.io.tx_raw('{} {} {} {}'.format(t_utc_iso,
                                            data.hex(), channel, t_mono_ns))

    def events(self, infile: str) -> Generator[Event, None, None]:
        for line in fileinput.input(infile or '-'):
            (t, s, ch, t_mono_ns) = line.split()
            t_utc = datetime.datetime.fromisoformat(t)
            data = bytes.fromhex(s)
            channel = None if ch == '-' else ch
            t_mono = int(t_mono_ns) / 1_000_000_000
            yield (t_mono, t_utc, channel, data)


class FmtVcr(Fmt):
    """JSON based file format from the 'vcr' recording/replay project."""
    name = 'vcr'
    period = 0x100000000
    time_format = '%Y-%m-%dT%H:%M:%S.%fZ'

    def on_init(self) -> None:
        self.session = str(uuid.uuid4())[0:8]
        self.channels: Dict[Optional[str], Tuple[str, int]] = {}

    def write_event(self, event: Event) -> None:
        (t_mono, t_utc, channel, data) = event
        (track, sequence) = self.channels.get(
            channel, (str(uuid.uuid4())[0:8], 0))
        tUtc_s = t_utc.strftime(self.__class__.time_format) \
            if t_utc is not None else None
        rec = {
            "channel": channel,
            "tMono": round(t_mono * 1_000_000_000),
            "tUtc": tUtc_s,
            "session": self.session,
            "track": track,
            "sequence": sequence,
            "value": {
                "data": data.hex(),
                "sender": None,
            },
        }
        sequence = (sequence + 1) % self.__class__.period
        self.channels[channel] = (track, sequence)
        self.io.tx_raw(json.dumps(rec))

    def events(self, infile: str) -> Generator[Event, None, None]:
        for line in fileinput.input(infile or '-'):
            o = json.loads(line)
            t_mono = o['tMono'] / 1_000_000_000
            # datetime does not support nanoseconds,
            # keep at most 6 decimal digits
            t_utc, subseconds = o['tUtc'].split('.')
            subseconds = subseconds[:-1]  # remove trailing 'Z'
            subseconds = subseconds[0:6]  # keep the first part
            t_utc = ''.join(t_utc) + '.{}Z'.format(subseconds)
            t_utc = datetime.datetime.strptime(
                t_utc, self.__class__.time_format)
            t_utc = t_utc.replace(tzinfo=datetime.timezone.utc)
            channel = o['channel']
            data = bytes.fromhex(o['value']['data'])
            yield (t_mono, t_utc, channel, data)


class FmtPcap(Fmt):
    """Wireshark/tcpdump file format."""
    name = 'pcap'

    def events(self, infile: str) -> Generator[Event, None, None]:
        if infile is None:
            raise Exception('expecting explicit filename')
        scapy_cap = rdpcap(infile)
        for packet in scapy_cap:
            try:
                data = bytes(packet[UDP].load)
            except IndexError:
                continue
            ts = float(packet.time)
            t_mono = ts
            t_utc = datetime.datetime.utcfromtimestamp(ts)
            t_utc = t_utc.replace(tzinfo=datetime.timezone.utc)
            yield (t_mono, t_utc, None, data)


class FmtBonita(Fmt):
    """Simple line oriented data format: {timestamp.seconds} : {hex data}"""
    name = 'bonita'

    def events(self, infile: str) -> Generator[Event, None, None]:
        fmt = '%a %b %d %H:%M:%S %Y UTC'
        start_time = None
        first_packet_time = None
        for line in fileinput.input(infile or '-'):
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                line = line[1:].lstrip()
                loc = locale.getlocale()
                locale.setlocale(locale.LC_ALL, 'C')
                start_time = datetime.datetime.strptime(line, fmt)
                start_time = start_time.replace(tzinfo=datetime.timezone.utc)
                # problem with setlocale inside nix-shell
                try:
                    locale.setlocale(locale.LC_ALL, loc)
                except BaseException:
                    pass
                continue
            t, s = line.split(':')
            t_mono = float(t.strip())
            if first_packet_time is None:
                first_packet_time = t_mono
            delta = t_mono - first_packet_time
            t_utc = datetime.datetime.now(tz=datetime.timezone.utc)
            if start_time is not None:
                t_utc = start_time + datetime.timedelta(seconds=delta)
            data = bytes.fromhex(s.strip())
            yield (t_mono, t_utc, None, data)


class FmtFinal(Fmt):
    """Final file format from Eurocontrol.
        - 2 bytes total length, including length itself
        - 1 byte 'error code'
        - 1 byte 'board/line number' (used as channel)
        - 1 byte recording day
        - 3 bytes time (LSB=0.01s)
        - ... asterix datablock(s)
        - 4 bytes constant padding (0xa5a5a5a5)
    """
    name = 'final'

    def on_init(self) -> None:
        self.day0: Optional[datetime.date] = None

    def write_event(self, event: Event) -> None:
        (t_mono, _t_utc, channel, data) = event
        t_utc: datetime.datetime = _t_utc  # type: ignore
        try:
            ch2 = int(channel or '')
        except ValueError:
            ch2 = hash(channel)
        ch2 %= 256
        ch = ch2.to_bytes(1, byteorder='big')
        if self.day0 is None:
            self.day0 = t_utc.date()
        total_length = (12 + len(data)).to_bytes(2, byteorder='big')
        err = bytes([0x00])
        day = t_utc.date()
        delta = day - self.day0
        recording_day = delta.days.to_bytes(1, byteorder='big')
        ts_midnight = datetime.datetime(
            t_utc.year, t_utc.month, t_utc.day, tzinfo=t_utc.tzinfo)
        seconds = (t_utc - ts_midnight).total_seconds()
        time = round(seconds * 100).to_bytes(3, byteorder='big')
        padding = bytes([0xa5, 0xa5, 0xa5, 0xa5])
        s = total_length + err + ch + recording_day + time + data + padding
        self.io.tx_raw_bin(s)

    def events(self, infile: str) -> Generator[Event, None, None]:
        def loop(f: Any) -> Generator[Event, None, None]:
            while True:
                n = f.read(2)
                if not n:
                    break
                assert len(n) == 2, str(n)
                n = (n[0] * 256 + n[1]) - 2
                s = f.read(n)
                assert len(s) == n, str(s)
                _err = s[0]
                ch = str(s[1])
                day = s[2]
                t = (s[3] * 256 * 256 + s[4] * 256 + s[5]) * 0.01
                data = s[6:][:-4]
                padding = s[-4:]
                assert padding == b'\xa5\xa5\xa5\xa5', str(padding)
                t_mono = float(day) * 24 * 3600 + t
                # This is the best approximation.
                # UTC time is not stored in this format,
                # but it might be useful to have at least relative times.
                t_utc = datetime.datetime.fromtimestamp(
                    0, tz=datetime.timezone.utc) + datetime.timedelta(seconds=t_mono)
                yield (t_mono, t_utc, ch, data)

        if infile is None:
            infile = '-'
        if infile == '-':
            for i in loop(sys.stdin.buffer):
                yield i
        else:
            with open(infile, "rb") as f:
                for i in loop(f):
                    yield i


format_input = [FmtSimple, FmtVcr, FmtBonita, FmtFinal]
if not fast:
    format_input.append(FmtPcap)
format_output = [FmtSimple, FmtVcr, FmtFinal]


def format_find(lst: List[Any], name: str) -> Any:
    for i in lst:
        if i.name == name:
            return i
    return None


def cmd_show_manifest(generated: Any, io: CIO, args: Any) -> None:

    @no_type_check
    def loop(arg: str, what: str) -> None:
        for cat in sorted(generated.manifest[arg]):
            for spec in reversed(generated.manifest[arg][cat]):
                ed_major, ed_minor = spec.cv_edition
                print(
                    '{} {}, edition {}.{}'.format(
                        what,
                        str(cat).zfill(3),
                        ed_major,
                        ed_minor))

    loop('CATS', 'cat')
    loop('REFS', 'ref')


def get_selection(generated: Any, empty: bool,
                  explicit_cats: List[Tuple[str, str]],
                  explicit_refs: List[Tuple[str, str]]
                  ) -> Dict[str, Dict[int, AstSpec]]:
    """Get category selection."""

    manifest = generated.manifest

    # get latest
    cats = {cat: manifest['CATS'][cat][-1] for cat in manifest['CATS']}
    refs = {cat: manifest['REFS'][cat][-1] for cat in manifest['REFS']}

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
            for spec in b[cat]:
                if spec.cv_edition == ed:
                    a.update({cat: spec})
    return {'CATS': cats, 'REFS': refs}


def get_expansions(selection: Dict[str, Dict[int, AstSpec]],
                   expansions: List[Tuple[str, str]]) -> Any:
    result = []
    for (cat_s, name) in expansions:
        cat = int(cat_s)
        assert cat in selection['REFS'].keys(), 'REF not defined'
        spec = selection['CATS'][cat]
        uap = spec.cv_uap  # type: ignore
        rec = uap.cv_record
        nsp = rec.spec(name)
        rule = nsp.cv_rule
        var = rule.cv_variation
        assert issubclass(var, Explicit)
        result.append((cat, name))
    return result


class AsterixSamples:
    """Asterix sample generator."""

    def __init__(self, gen: PCG32,
                 sel: Dict[Any, Any], exp: Dict[Any, Any],
                 populate_all_items: bool,
                 include_multi_uap: bool,
                 max_datablocks: int,
                 max_records: int,
                 error_bit_flip: Optional[int]) -> None:
        self.gen = gen
        self.expand = set(exp)
        self.populate_all_items = populate_all_items
        self.max_datablocks = max_datablocks
        self.max_records = max_records
        self.error_bit_flip = error_bit_flip

        # for some specs it is not possible to generate valid record,
        # without knowing the UAP, so skip those
        self.valid_specs = {}
        for (cat, spec) in sel['CATS'].items():
            if include_multi_uap or issubclass(spec.cv_uap, UapSingle):
                self.valid_specs[cat] = spec
        err = "non-empty list is required, only single UAP specs are supported"
        assert self.valid_specs, err
        self.refs = sel['REFS']

    def __iter__(self) -> 'AsterixSamples':
        return self

    def random_record(self, cat: int, cls: Type[AstSpec]) -> AstSpec:
        gen = self.gen

        @no_type_check
        def random_var(var, name=None):

            if issubclass(var, Element):
                return gen.bigint(var.cv_bit_size)

            if issubclass(var, Group):
                return tuple([random_item(i)
                             for (i, _size) in var.cv_items_list])

            if issubclass(var, Extended):
                groups = var.cv_items_list
                n = gen.next() % len(groups)
                groups = groups[0:n + 1]
                result1 = []
                for g in groups:
                    result2 = []
                    for x in g:
                        if x is None:
                            result2.append(None)
                        else:
                            (i, _size) = x
                            result2.append(random_item(i))
                    result1.append(tuple(result2))
                return tuple(result1)

            if issubclass(var, Repetitive):
                n = 1 + (gen.next() % 10)
                return [random_var(var.cv_variation) for i in range(n)]

            if issubclass(var, Explicit):
                this_item = (cat, name)
                if this_item not in self.expand:
                    return None
                exp = self.refs[cat].cv_expansion
                d = {}
                for (name, cls) in exp.cv_items_dict.items():
                    populate_this_item = self.populate_all_items or gen.bool()
                    if populate_this_item:
                        rule = random_rule(cls.cv_rule, name)
                        if rule is not None:
                            d[name] = rule
                if not d:
                    return None
                obj = exp.create(d)
                return obj.unparse().to_bytes() or None  # avoid empty

            if issubclass(var, Compound):
                d = {}
                for (name, cls) in var.cv_items_dict.items():
                    populate_this_item = self.populate_all_items or gen.bool()
                    if populate_this_item:
                        rule = random_rule(cls.cv_rule, name)
                        if rule is not None:
                            d[name] = rule
                return d or None  # turn {} into None, to skip this subitem

            raise Exception('internal error, unexpected variation', var)

        @no_type_check
        def random_item(t):
            if issubclass(t, Spare):
                return 0
            if issubclass(t, Item):
                return random_rule(t.cv_non_spare.cv_rule)
            raise Exception('internal error, unexpected type', t)

        @no_type_check
        def random_rule(t, name=None):
            if issubclass(t, RuleVariationContextFree):
                cls = t.cv_variation
            elif issubclass(t, RuleVariationDependent):
                cls = t.cv_default_variation
            else:
                raise Exception('internal error, unexpected type', t)
            return random_var(cls, name)

        @no_type_check
        def random_rec(t):
            d = {}
            for (key, cls) in t.cv_items_dict.items():
                populate = self.populate_all_items or gen.bool()
                if populate:
                    rule = random_rule(cls.cv_rule, key)
                    if rule is not None:
                        d[key] = rule
            return t.create(d)

        uap = cls.cv_uap  # type: ignore
        if issubclass(uap, UapSingle):
            rec_class = uap.cv_record
        elif issubclass(uap, UapMultiple):
            rec_class = self.gen.choose(list(uap.cv_uaps.values()))
        else:
            raise Exception('internal error, unexpected type', uap)
        r = random_rec(rec_class)
        return r  # type: ignore

    def inject_errors(self, bs: bytes) -> bytes:
        n = self.error_bit_flip
        if n is None:
            return bs
        out = b''
        for x in bs:
            flip = self.gen.next() % n
            if flip == 0:
                bit_to_flip = self.gen.next() % 8
                x ^= pow(2, bit_to_flip)
            out += bytes([x])
        return out

    def __next__(self) -> bytes:
        bs = b''
        n1 = self.gen.choose(list(range(self.max_datablocks))) + 1
        n2 = self.gen.choose(list(range(self.max_records))) + 1
        for i in range(n1):
            cat = self.gen.choose(list(self.valid_specs.keys()))
            cls = self.valid_specs[cat]
            records = []
            for j in range(n2):
                rec = self.random_record(cat, cls)
                records.append(rec)
            db = cls.create(records)
            bs += db.unparse().to_bytes()
        return self.inject_errors(bs)


def cmd_gen_random(generated: Any, io: CIO, args: Any) -> None:
    """Generate random samples."""
    sel = get_selection(generated, args.empty_selection,
                        args.cat or [], args.ref or [])
    exp = get_expansions(sel, args.expand or [])
    populate_all_items = args.populate_all_items
    include_multi_uap = args.include_multi_uap
    seed = args.seed
    if seed is None:
        seed = random.randint(0, pow(2, 64) - 1)
    gen = PCG32(seed)
    channel = None
    for data in AsterixSamples(
            gen,
            sel,
            exp,
            populate_all_items,
            include_multi_uap,
            args.max_datablocks,
            args.max_records,
            args.error_bit_flip):
        t_mono = time.monotonic()
        t_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        if args.channel:
            channel = gen.choose(list(args.channel))
        io.tx((t_mono, t_utc, channel, data))
        if args.sleep is not None:
            time.sleep(args.sleep)


def cmd_asterix_decoder(generated: Any, io: CIO, args: Any) -> None:
    sel = get_selection(generated, args.empty_selection,
                        args.cat or [], args.ref or [])
    exp = get_expansions(sel, args.expand or [])
    if args.truncate:
        @no_type_check
        def smax(n, s):
            return s if (len(s) <= n) else (s[0:n] + '|')

        @no_type_check
        def truncate(s): return print(smax(args.truncate, s))
    else:
        @no_type_check
        def truncate(s): return print(s)

    @no_type_check
    def too_deep(i):
        """Parsing level check."""
        max_level = args.parsing_level
        if max_level <= 0:
            return False
        return i > max_level

    @no_type_check
    def handle_variation(cat, i, path, var):
        if too_deep(i):
            return
        if isinstance(var, Element):
            x = var.as_uint()
            dsc = 'value: {} = {} = {}'.format(x, hex(x), oct(x))
            rule = var.rule
            if isinstance(rule, RuleContentContextFree):
                content = rule.content
                if isinstance(content, ContentTable):
                    tv = content.table_value()
                    if tv is None:
                        tv = '(undefined value)'
                    dsc = 'value: {} -> "{}"'.format(x, tv)
                elif isinstance(content, ContentString):
                    s = content.as_string().encode('utf-8')
                    s = repr(s)[2:-1]  # strip off b'...'
                    dsc = 'value: {}, str: "{}"'.format(x, s)
                elif isinstance(content, ContentInteger):
                    val = content.as_integer()
                    dsc = 'value: {}, int: {}'.format(x, val)
                elif isinstance(content, ContentQuantity):
                    dsc = 'value: {}, quantity: {} {}'.format(
                        x, content.as_quantity(), content.__class__.cv_unit)
            elif isinstance(rule, RuleContentDependent):
                dsc += ' (content dependent)'
            else:
                raise Exception('internal error, unexpected type', rule)
            truncate('{}Element: {}'.format('  ' * i, dsc))

        elif isinstance(var, Group):
            for item in var.arg:
                handle_item(cat, i, path, item)

        elif isinstance(var, Extended):
            for lst in var.arg:
                for item in lst:
                    if item is not None:
                        handle_item(cat, i, path, item)

        elif isinstance(var, Repetitive):
            for cnt, sub in enumerate(var.arg):
                s = sub.unparse()
                truncate('{}subitem ({}), len={} bits, bin={}'
                         .format('  ' * i, cnt, len(s), s))
                handle_variation(cat, i + 1, path + ['({})'.format(cnt)], sub)

        elif isinstance(var, Explicit):
            this_item = (cat, path[0])
            if this_item not in exp:
                return
            sub = sel['REFS'][cat]
            bs = Bits.from_bytes(var.get_bytes())
            result = sub.cv_expansion.parse(bs)
            if isinstance(result, ValueError):
                truncate('Error! {}'.format(result))
                truncate('Unable to parse expansion: {}'.format(bs))
                if args.stop_on_error:
                    sys.exit(1)
                return
            obj, bs2 = result
            if len(bs2):
                truncate(
                    'Unexpected remaining bits in explicit item: {}'.format(bs2))
                if args.stop_on_error:
                    sys.exit(1)
                return
            for (name, nsp) in obj.arg.items():
                handle_nonspare(cat, i, path + [name], nsp)

        elif isinstance(var, Compound):
            # fspec is not stored, need to re-parse
            result = Fspec.parse(ParsingMode.StrictParsing,
                                 var.__class__.cv_fspec_max_bytes, var.bs)
            assert not isinstance(result, ValueError)
            (fspec, _remaining) = result
            truncate('{}(FSPEC): {}'.format('  ' * i, fspec.bs))
            for (name, nsp) in var.arg.items():
                handle_nonspare(cat, i, path + [name], nsp)
        else:
            raise Exception('internal error, unexpected variation', var)

    @no_type_check
    def handle_item(cat, i, path, item):
        if too_deep(i):
            return
        if isinstance(item, Spare):
            bs = item.unparse()
            truncate(
                '{}(Spare): len={} bits, bin={}'.format(
                    '  ' * i, len(bs), bs))
            if args.check_spare_bits:
                if bs.to_uinteger() != 0:
                    truncate('Error! Non-zero spare bit(s)')
                    if args.stop_on_error:
                        sys.exit(1)
        elif isinstance(item, Item):
            name = item.arg.__class__.cv_name
            handle_nonspare(cat, i, path + [name], item.arg)
        else:
            raise Exception('internal error, unexpected type', item)

    @no_type_check
    def handle_rulevar(cat, i, path, rule):
        if too_deep(i):
            return
        if isinstance(rule, RuleVariationContextFree):
            handle_variation(cat, i, path, rule.variation)
        elif isinstance(rule, RuleVariationDependent):
            truncate('{}(content dependent structure)'.format('  ' * i))
        else:
            raise Exception('internal error, unexpected type', rule)

    @no_type_check
    def handle_nonspare(cat, i, path, nsp):
        if too_deep(i):
            return
        title = nsp.cv_title
        bs = nsp.unparse()
        truncate(
            '{}{}: "{}", len={} bits, bin={}'.format(
                '  ' * i,
                path,
                title,
                len(bs),
                bs))
        handle_rulevar(cat, i + 1, path, nsp.rule)

    @no_type_check
    def handle_rfs(cat, i, rfs_count, rfs_max, rfs):
        if rfs_max > 1:
            truncate('{}(RFS {}/{})'.format(
                '  ' * i,
                rfs_count + 1,
                rfs_max))
        else:
            truncate('{}(RFS)'.format('  ' * i))
        for (name, nsp) in rfs:
            handle_nonspare(cat, i + 1, [name], nsp)

    @no_type_check
    def handle_record(cat, i, rec, uap=None):
        if too_deep(i):
            return
        raw = rec.unparse().to_bytes()
        truncate(
            '{}record{}: len={} bytes, hex={}'.format(
                '  ' * i,
                '' if uap is None else ' (uap={})'.format(uap),
                len(raw),
                raw.hex()))
        # fspec is not stored, need to re-parse
        result = Fspec.parse(ParsingMode.StrictParsing,
                             rec.__class__.cv_fspec_max_bytes, rec.bs)
        assert not isinstance(result, ValueError)
        (fspec, _remaining) = result
        truncate('{}(FSPEC): {}'.format('  ' * (i + 1), fspec.bs))
        rfs_count = 0
        rfs_max = 0
        for ispec in rec.__class__.cv_items_list:
            if issubclass(ispec, UapItemRFS):
                rfs_max += 1
        for ispec in rec.__class__.cv_items_list:
            if issubclass(ispec, UapItem):
                name = ispec.cv_non_spare.cv_name
                nsp = rec.items_regular.get(name)
                if nsp is not None:
                    handle_nonspare(cat, i + 1, [name], nsp)
            elif issubclass(ispec, UapItemSpare):
                pass
            elif issubclass(ispec, UapItemRFS):
                try:
                    rfs = rec.items_rfs[rfs_count]
                except IndexError:
                    rfs = None
                if rfs is not None:
                    handle_rfs(cat, i + 1, rfs_count, rfs_max, rfs)
                rfs_count += 1
            else:
                raise Exception('internal error, unexpected item_spec', ispec)

    @no_type_check
    def handle_datablock(i, db):
        if too_deep(i):
            return
        cat = db.get_category()
        n = db.get_length()
        bs = db.get_raw_records()
        d = bs.to_bytes().hex()
        truncate(
            '{}datablock: cat={}, len={} bytes, records={}'.format(
                '  ' * i, cat, n, d))
        spec = sel['CATS'].get(cat)
        if spec is None:
            return
        uap = spec.cv_uap
        if issubclass(uap, UapSingle):
            result = uap.parse(bs)
            if isinstance(result, ValueError):
                truncate('Error! {}'.format(result))
                truncate('Unable to parse datablock: {}'.format(d))
                if args.stop_on_error:
                    sys.exit(1)
                return
            for rec in result:
                handle_record(cat, i + 1, rec)
        elif issubclass(uap, UapMultiple):
            def find_uap_by_type(rec):
                for name, cls in uap.cv_uaps.items():
                    if isinstance(rec, cls):
                        return name
                return None
            max_depth = args.multi_uap_max_parsing_depth
            results = spec.cv_uap.parse_any_uap(bs, max_depth)
            if isinstance(results, ValueError):
                truncate('Error! {}'.format(results))
                if args.stop_on_error:
                    sys.exit(1)
            else:
                if len(results) == 0:
                    truncate('Error! Multiple UAP, no possible results')
                    truncate('Unable to parse datablock: {}'.format(d))
                    if args.stop_on_error:
                        sys.exit(1)
                    return
                else:
                    for (n, result) in enumerate(results):
                        truncate(
                            '{}Multiple UAP, guessing possible result ({}/{}):'
                            .format('  ' * (i + 1), n + 1, len(results)))
                        for rec in result:
                            handle_record(cat, i + 2, rec, find_uap_by_type(rec))
        else:
            raise Exception('internal error, unexpected type', uap)

    @no_type_check
    def handle_datagram(i, s):
        if too_deep(i):
            return
        truncate(
            '{}datagram: len={} bytes, hex={}'.format(
                '  ' * i, len(s), s.hex()))
        dbs = RawDatablock.parse(Bits.from_bytes(s))
        if isinstance(dbs, ValueError):
            truncate('Error! {}'.format(dbs))
            truncate('Unable to parse datagram: {}'.format(s.hex()))
            if args.stop_on_error:
                sys.exit(1)
            return
        for db in dbs:
            handle_datablock(i + 1, db)

    @no_type_check
    def handle_event(t, s):
        truncate('timestamp: {}'.format(t if t is not None else '<unknown>'))
        handle_datagram(1, s)

    for event in io.rx():
        (t_mono, t_utc, channel, data) = event
        handle_event(t_utc, data)


def cmd_from_udp(generated: Any, io: CIO, args: Any) -> None:
    sockets = {}
    sel = selectors.DefaultSelector()
    for (channel, ip, port) in args.unicast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, int(port)))
        sock.setblocking(False)
        sel.register(sock, selectors.EVENT_READ)
        sockets[sock] = channel
    for (channel, mcast, port) in args.multicast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((mcast, int(port)))
        if args.local_ip is None:
            mreq = struct.pack(
                "=4sl",
                socket.inet_aton(mcast),
                socket.INADDR_ANY)
        else:
            mreq = struct.pack("=4s4s", socket.inet_aton(mcast),
                               socket.inet_aton(args.local_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setblocking(False)
        sel.register(sock, selectors.EVENT_READ)
        sockets[sock] = channel

    # processing loop
    while True:
        select_events = sel.select()
        for key, mask in select_events:
            t_mono = time.monotonic()
            t_utc = datetime.datetime.now(tz=datetime.timezone.utc)
            f = key.fileobj
            channel = sockets[f]  # type: ignore
            (data, addr) = f.recvfrom(pow(2, 16))  # type: ignore
            io.tx((t_mono, t_utc, channel, data))


def cmd_to_udp(generated: Any, io: CIO, args: Any) -> None:
    sockets = []
    for (channel, ip, port) in args.unicast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockets.append((channel, sock, ip, int(port)))
    for (channel, mcast, port) in args.multicast:
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
            socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, args.ttl)
        if args.local_ip is not None:
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_IF,
                socket.inet_aton(args.local_ip))
        sockets.append((channel, sock, mcast, int(port)))

    # processing loop
    for event in io.rx():
        (t_mono, t_utc, channel1, data) = event
        for (channel2, sock, ip, port) in sockets:
            if channel2 == '*' or channel1 == channel2:
                sock.sendto(data, (ip, port))


def cmd_inspect(generated: Any, io: CIO, args: Any) -> None:
    raw_datablock_errors = 0
    unknown_categories: Set[int] = set()
    processed_categories: Set[int] = set()
    parse_errors: Dict[int, Any] = dict()

    def str_edition(ed: Tuple[int, int]) -> str:
        return '{}.{}'.format(ed[0], ed[1])

    @no_type_check
    def handle_event(s):
        nonlocal raw_datablock_errors, unknown_categories, processed_categories, parse_errors
        raw_datablocks = RawDatablock.parse(Bits.from_bytes(s))
        if isinstance(raw_datablocks, ValueError):
            raw_datablocks = []
            raw_datablock_errors += 1
        for raw_db in raw_datablocks:
            cat = raw_db.get_category()
            specs = generated.manifest['CATS'].get(cat)
            if specs is None:
                unknown_categories.add(cat)
                continue
            processed_categories.add(cat)
            bs = raw_db.get_raw_records()
            for Spec in specs:
                uap = Spec.cv_uap
                if issubclass(uap, UapSingle):
                    result = uap.parse(bs)
                    if isinstance(result, ValueError):
                        problems = parse_errors.get(cat, set())
                        problems.add(Spec.cv_edition)
                        parse_errors[cat] = problems
                elif issubclass(uap, UapMultiple):
                    max_depth = args.multi_uap_max_parsing_depth
                    results = Spec.cv_uap.parse_any_uap(bs, max_depth)
                    if isinstance(results, ValueError) or len(results) == 0:
                        problems = parse_errors.get(cat, set())
                        problems.add(Spec.cv_edition)
                        parse_errors[cat] = problems
                else:
                    raise Exception('internal error, unexpected type', uap)

    try:
        for event in io.rx():
            (t_mono, t_utc, channel, data) = event
            handle_event(data)
    except KeyboardInterrupt:
        pass

    print('done...')
    print('datablock erros: {}'.format(raw_datablock_errors))
    print('unknown categories: {}'.format(sorted(unknown_categories)))
    print('success category/edition:')
    for cat in sorted(processed_categories):
        editions = {x.cv_edition for x in generated.manifest['CATS'][cat]}
        problems = parse_errors.get(cat, set())
        editions.difference_update(problems)
        print('{} -> {}'.format(cat, [str_edition(x)
              for x in sorted(editions)]))
    print('problems category/edition:')
    for cat in sorted(processed_categories):
        problems = parse_errors.get(cat)
        if not problems:
            continue
        print('{} -> {}'.format(cat, [str_edition(x)
              for x in sorted(problems)]))


def cmd_record(generated: Any, io: CIO, args: Any) -> None:
    fmt = format_find(format_output, args.format)(io)
    for event in io.rx():
        fmt.write_event(event)


def cmd_replay(generated: Any, io: CIO, args: Any) -> None:
    fmt = format_find(format_input, args.format)(io)
    offset = None  # not known until the first event
    for event in fmt.events(args.infile):
        now = time.monotonic()
        (t_mono, t_utc, channel, data) = event
        if args.channel is None or channel in args.channel:
            if not args.full_speed:
                if offset is None:
                    offset = now - t_mono
                t = t_mono + offset
                delta = t - now
                if delta > 0:
                    time.sleep(delta)
            io.tx(event)


def cmd_custom(generated: Any, io: CIO, args: Any) -> None:
    # import custom script
    filename = args.script
    p = os.path.dirname(os.path.abspath(filename))
    sys.path.insert(0, p)
    with open(filename, 'r') as f:
        code = compile(f.read(), filename, 'exec')
    run_globals = {
        '__name__': None,
        '__file__': None,
        '__loader__': None,
        '__package__': None,
    }
    exec(code, run_globals)
    sys.path.pop(0)

    # Create and call user function with the following arguments
    #   - asterix base module (already imported)
    #   - asterix generated module (if imported, otherwise None)
    #   - IO instance for standard input/output
    #   - all command line arguments
    f = run_globals[args.call]  # type: ignore
    gen2 = None if fast else generated
    f(base, gen2, io, args)  # type: ignore


def check_ttl(arg: int) -> int:
    ttlInterval = (1, 255)
    try:
        val = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError('must be integer')
    try:
        assert val >= ttlInterval[0]
        assert val <= ttlInterval[1]
    except AssertionError:
        raise argparse.ArgumentTypeError('must be in interval [{},{}]'.format(
            ttlInterval[0], ttlInterval[1]))
    return val


def check_min(arg: int) -> Any:
    @no_type_check
    def f(x: int) -> int:
        try:
            x = int(x)
        except ValueError:
            raise argparse.ArgumentTypeError('must be integer')
        if x < arg:
            raise argparse.ArgumentTypeError('range error')
        return x
    return f


def main() -> None:

    parser = argparse.ArgumentParser(description='Asterix data processor.')

    libasterix_version = importlib.metadata.version('libasterix')

    parser.add_argument(
        '--fast-startup',
        action='store_true',
        help='Reduce features for faster startup time, this argument must be set first')

    parser.add_argument(
        '--version-tool',
        action='version',
        version=__version__,
        help='show tool version number and exit')

    parser.add_argument(
        '--version-lib',
        action='version',
        version=libasterix_version,
        help='show lib version number and exit')

    parser.add_argument('--version', action='version',
                        version='%(prog)s {}, libasterix {}'.format(
                            __version__,
                            libasterix_version),
                        help='show program version number and exit')

    if not fast:
        parser.add_argument(
            '--multi-uap-max-parsing-depth',
            type=int,
            default=None,
            help='Limit max parsing depth when parsing multi UAP category')

        parser.add_argument(
            '--custom-generated',
            metavar='FILE',
            help='Use custom asterix specification generated file in place of default generated.py from the libasterix library')

        parser.add_argument(
            '--empty-selection',
            action='store_true',
            help='Use empty initial cat/ref selection instead of latest editions')

        parser.add_argument('--cat', nargs=2, metavar=('CAT', 'EDITION'),
                            action='append',
                            help='Explicit category selection')

        parser.add_argument('--ref', nargs=2, metavar=('CAT', 'EDITION'),
                            action='append',
                            help='Explicit expansion selection')

        parser.add_argument('--expand', nargs=2, metavar=('CAT', 'ITEM-NAME'),
                            action='append',
                            help='Expand CAT/ITEM-NAME with REF expansion')

    parser.add_argument('--ttl', dest='ttl',
                        type=check_ttl, default=32,  # type: ignore
                        help='Time to live for outgoing multicast traffic, default: %(default)s')

    parser.add_argument(
        '--local-ip',
        dest='local_ip',
        type=str,
        help='Optional IP address of local interface to use for multicast.')

    parser.add_argument(
        '--simple-input',
        action='store_true',
        help='Data input flow is in simple form, without meta information')

    parser.add_argument(
        '--simple-output',
        action='store_true',
        help='Data output flow is in simple form, without meta information')

    parser.add_argument(
        '-s',
        '--simple',
        action='store_true',
        help='Data input/output flow is in simple form, without meta information')

    parser.add_argument('--no-flush', action='store_true',
                        help='Do not flush output on each event')

    subparsers = parser.add_subparsers(required=True, help='sub-commands')

    if not fast:
        # 'manifest' command
        parser_manifest = subparsers.add_parser(
            'manifest', help='show available categories')
        parser_manifest.set_defaults(func=cmd_show_manifest)
        parser_manifest.add_argument('--latest', action='store_true',
                                     help='show latest editions only')

        # 'random' command
        parser_random = subparsers.add_parser(
            'random', help='asterix sample generator')
        parser_random.set_defaults(func=cmd_gen_random)
        parser_random.add_argument(
            '--sleep',
            type=float,
            help="Sleep 't' seconds between random samples")
        parser_random.add_argument('--seed', type=int,
                                   help='Randomm generator seed value')
        parser_random.add_argument(
            '--populate-all-items',
            action='store_true',
            help='Populate all defined items instead of random selection')
        parser_random.add_argument(
            '--include-multi-uap',
            action='store_true',
            help='Include multiple UAP categories. Generated record might not \
                    be correct according to the record selector item value.')
        parser_random.add_argument(
            '--channel',
            metavar='STR',
            action='append',
            default=[],
            help='Channel name (can be specified multiple times)')
        parser_random.add_argument(
            '--max-datablocks',
            default=5,
            type=check_min(1),
            help='Max number of datablocks per datagram, default: %(default)s')
        parser_random.add_argument(
            '--max-records',
            default=5,
            type=check_min(1),
            help='Max number of records per datablock, default: %(default)s')
        parser_random.add_argument(
            '--error-bit-flip',
            required=False,
            type=check_min(1),
            metavar='N',
            help='Random bit flip in every N-th byte')

        # 'decode' command
        parser_decode = subparsers.add_parser('decode', help='asterix decoder')
        parser_decode.set_defaults(func=cmd_asterix_decoder)
        parser_decode.add_argument(
            '--truncate',
            type=int,
            metavar='N',
            default=0,
            help='truncate long data lines to N characters or 0 for none, \
                default: %(default)s')
        parser_decode.add_argument(
            '--check-spare-bits',
            action='store_true',
            help='Check spare bits for zero value')
        parser_decode.add_argument(
            '--stop-on-error',
            action='store_true',
            help='exit on first parsing error')
        parser_decode.add_argument(
            '-l',
            '--parsing-level',
            type=int,
            metavar='N',
            default=0,
            help='limit parsing depth, 0 for none, default: %(default)s')

        # 'inspect' command
        parser_inspect = subparsers.add_parser(
            'inspect', help='report asterix parsing status per category/edition')
        parser_inspect.set_defaults(func=cmd_inspect)

    # 'from-udp' command
    parser_from_udp = subparsers.add_parser(
        'from-udp', help='UDP datagram receiver')
    parser_from_udp.set_defaults(func=cmd_from_udp)
    parser_from_udp.add_argument(
        '--unicast',
        action='append',
        help='Unicast UDP input',
        default=[],
        nargs=3,
        metavar=(
            'channel',
            'ip',
            'port'))
    parser_from_udp.add_argument(
        '--multicast',
        action='append',
        help='Multicast UDP input',
        default=[],
        nargs=3,
        metavar=(
            'channel',
            'mcast-ip',
            'port'))

    # 'to-udp' command
    parser_to_udp = subparsers.add_parser(
        'to-udp', help='UDP datagram transmitter')
    parser_to_udp.set_defaults(func=cmd_to_udp)
    parser_to_udp.add_argument(
        '--unicast',
        action='append',
        help='Unicast UDP output, use channel "*" for any channel',
        default=[],
        nargs=3,
        metavar=(
            'channel',
            'ip',
            'port'))
    parser_to_udp.add_argument(
        '--multicast',
        action='append',
        help='Multicast UDP output, use channel "*" for any channel',
        default=[],
        nargs=3,
        metavar=(
            'channel',
            'mcast-ip',
            'port'))

    # 'record' command
    parser_record = subparsers.add_parser('record',
                                          help='data recorder')
    parser_record.set_defaults(func=cmd_record)
    parser_record.add_argument('--format',
                               choices=[fmt.name for fmt in format_output],
                               default=format_output[0].name,
                               help='data format, default: %(default)s')

    # 'replay' command
    parser_replay = subparsers.add_parser('replay',
                                          help='data replay from recording')
    parser_replay.set_defaults(func=cmd_replay)
    parser_replay.add_argument('--channel', action='append', metavar='STR',
                               help='Channel to process')
    parser_replay.add_argument('--format',
                               choices=[fmt.name for fmt in format_input],
                               default=format_input[0].name,
                               help='data format, default: %(default)s')
    parser_replay.add_argument('--full-speed', action='store_true',
                               help='Replay at full speed')
    parser_replay.add_argument('infile', nargs='?')

    # 'custom' command
    parser_custom = subparsers.add_parser('custom',
                                          help='run custom python script')
    parser_custom.set_defaults(func=cmd_custom)
    parser_custom.add_argument('--script', help='File to import',
                               required=True, metavar='filename')
    parser_custom.add_argument('--call', help='Function to call',
                               required=True, metavar='callable')
    parser_custom.add_argument('--args', help='Additional arguments (string)',
                               metavar='args')

    # Empty argument raises TypeError on some old pip/python versions.
    try:
        args = parser.parse_args()
    except TypeError:
        print("Arguments are required, try '--help'.")
        sys.exit(0)

    io = CIO(
        args.simple or args.simple_input,
        args.simple or args.simple_output,
        not args.no_flush)

    generated = generated_orig
    if not fast:
        if args.custom_generated is not None:
            generated = import_from_path(
                'asterix.generated', args.custom_generated)

    try:
        args.func(generated, io, args)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    main()
