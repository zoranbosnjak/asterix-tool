#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Asterix data processing tool

import argparse
import fileinput
import random
import os
import sys
import time
import datetime
import socket
import selectors
import uuid
import json
import locale

import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

from scapy.all import rdpcap, IP, UDP

import asterix as ast
from asterix import *

__version__ = "0.9.0"

def output(*args):
    """Like 'print', but handle broken pipe exception and flush."""
    try:
        print(*args, flush=True)
    except BrokenPipeError:
        sys.stdout = None
        sys.exit(0)

class PCG32:
    """Simple random number generator,
    based on https://www.pcg-random.org/download.html"""

    top64 : int = pow(2, 64)
    top32 : int = pow(2, 32)

    def mod64(self, val : int) -> int: return val % self.__class__.top64
    def mod32(self, val : int) -> int: return val % self.__class__.top32

    def __init__(self, state : int = 0, inc : int = 0) -> None:
        self.state = self.mod64(state)
        self.inc = self.mod64(inc)

    def next(self) -> int:
        """Generate next 32-bit random value."""
        oldstate = self.state
        self.state = self.mod64(oldstate * 6364136223846793005 + (self.inc | 1))
        xorshifted = self.mod32(((oldstate >> 18) ^ oldstate) >> 27)
        rot = oldstate >> 59
        return self.mod32((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))

    def choose(self, lst : List[Any]) -> Any:
        """Choose list element."""
        ix = self.next() % len(lst)
        return lst[ix]

    def bool(self):
        """Generate random bool value."""
        return bool(self.next() % 2)

    def bigint(self, bitsize):
        """Generate random number of arbitrary bitsize."""
        val = self.next()
        size = 32
        while True:
            if size >= bitsize:
                return val % pow(2,bitsize)
            size += 32
            val = val*pow(2,32) + self.next()

class Fmt:
    name : str

    def __init__(self, channel=None, sender=None):
        pass

class FmtSimple(Fmt):
    """Simple line oriented data format: {timestamp-iso} {hex data} {time-mono-ns}"""
    name = 'simple'

    def write_event(self, t_mono, t_utc, data):
        t_mono_ns = round(t_mono*1000*1000*1000)
        output('{} {} {}'.format(t_utc.isoformat(), data, t_mono_ns))

    def events(self, infile):
        for line in fileinput.input(infile or '-'):
            (t, data, t_mono_ns) = line.split()
            t_utc = datetime.datetime.fromisoformat(t)
            t_mono = int(t_mono_ns)/(1000*1000*1000)
            yield(t_mono, t_utc, data)

class FmtVcr(Fmt):
    """JSON based time format from the 'vcr' recording/replay project."""
    name = 'vcr'
    period = 0x100000000
    time_format = '%Y-%m-%dT%H:%M:%S.%fZ'

    def __init__(self, channel=None, sender=None):
        self.channel = channel
        self.sender = sender
        self.session = str(uuid.uuid4())[0:8]
        self.track = str(uuid.uuid4())[0:8]
        self.sequence = 0

    def write_event(self, t_mono, t_utc, data):
        rec = {
            "channel": self.channel,
            "tMono": round(t_mono*1000*1000*1000),
            "tUtc": t_utc.strftime(self.__class__.time_format),
            "session": self.session,
            "track": self.track,
            "sequence": self.sequence,
            "value": {
                "data": data,
                "sender": self.sender,
                },
        }
        self.sequence = (self.sequence + 1) % self.__class__.period
        s = json.dumps(rec)
        output(s)

    def events(self, infile):
        for line in fileinput.input(infile or '-'):
            o = json.loads(line)
            if self.channel is not None:
                if o['channel'] != self.channel:
                    continue
            t_mono = o['tMono']/(1000*1000*1000)
            # datetime does not support nanoseconds, round to microseconds
            t_utc = o['tUtc'].split('.')
            subseconds = float('0.' + t_utc[-1][:-1])
            microseconds = round(subseconds*1000*1000)
            t_utc = ''.join(t_utc[:-1]) + '.{:06d}Z'.format(microseconds)
            t_utc = datetime.datetime.strptime(t_utc, self.__class__.time_format)
            t_utc = t_utc.replace(tzinfo=datetime.timezone.utc)
            data = o['value']['data']
            yield (t_mono, t_utc, data)

class FmtPcap(Fmt):
    """Wireshark/tcpdump file format."""
    name = 'pcap'

    def events(self, infile):
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
            yield (t_mono, t_utc, data.hex())

class FmtBonita(Fmt):
    """Simple line oriented data format: {timestamp.seconds} : {hex data}"""
    name = 'bonita'

    def events(self, infile):
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
                try: locale.setlocale(locale.LC_ALL, loc)
                except: pass
                continue
            t, data = line.split(':')
            t_mono = float(t.strip())
            if first_packet_time is None:
                first_packet_time = t_mono
            delta = t_mono - first_packet_time
            t_utc = None
            if start_time is not None:
                t_utc = start_time + datetime.timedelta(seconds=delta)
            data = data.strip()
            yield (t_mono, t_utc, data)

format_input  = [FmtSimple, FmtVcr, FmtPcap, FmtBonita]
format_output = [FmtSimple, FmtVcr]

def format_find(lst, name):
    for i in lst:
        if i.name == name:
            return i
    return None

def string_to_edition(ed):
    """Convert edition string to a tuple, for example "1.2" -> (1,2)"""
    a,b = ed.split('.')
    return (int(a), int(b))

def cmd_show_manifest(args):

    def fmt(what, n, ed): # type: ignore
        return '{} {}, edition {}'.format(what, str(n).zfill(3), ed)

    def loop(arg, what):
        for n in sorted(manifest[arg]):
            d = manifest[arg][n]
            for ed in sorted(d.keys(), key=string_to_edition, reverse=True):
                output(fmt(what, n, ed))
                if args.latest: break
    loop('CATS', 'cat')
    loop('REFS', 'ref')

def get_selection(empty, explicit_cats, explicit_refs):
    """Get category selection."""

    def get_latest(lst):
        return sorted(lst, key=lambda pair: string_to_edition(pair[0]), reverse=True)[0]

    # get latest
    cats = {cat: get_latest(manifest['CATS'][cat].items())[1] for cat in manifest['CATS'].keys()}
    refs = {cat: get_latest(manifest['REFS'][cat].items())[1] for cat in manifest['REFS'].keys()}

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

def get_expansions(selection, expansions):
    result = []
    for (cat, name) in expansions:
        cat = int(cat)
        assert cat in selection['REFS'].keys(), 'REF not defined'
        spec = selection['CATS'][cat]
        subitem = spec.variation.spec(name)
        assert issubclass(subitem, Explicit)
        result.append((cat, name))
    return result

def get_parsing_options(args):
    parsing_opt = ParsingOptions.default()
    parsing_opt.no_check_spare = bool(args.no_check_spare)
    return parsing_opt

class AsterixSamples:
    """Asterix sample generator."""
    def __init__(self, gen, sel, exp, populate_all_items):
        self.gen = gen
        self.expand = set(exp)
        self.populate_all_items = populate_all_items

        # for some specs it is not possible to generate valid record,
        # without knowing the profile, so skip those
        self.valid_specs = {}
        for (cat, spec) in sel['CATS'].items():
            if hasattr(spec, 'make_record') or hasattr(spec, 'is_valid'):
                self.valid_specs[cat] = spec
        assert self.valid_specs # non-empty list is required
        self.refs = sel['REFS']

    def __iter__(self):
        return self

    def random_record(self, cat, cls):
        gen = self.gen

        def random_var(var, name=None):

            if issubclass(var, Element):
                return gen.bigint(var.bit_size)

            if issubclass(var, Group):
                return gen.bigint(var.bit_size)

            if issubclass(var, Extended):
                values = [gen.bigint(i) for i in var.groups_bit_sizes]
                values = values[0:1+(gen.next() % len(values))]
                return tuple(values)

            if issubclass(var, Repetitive):
                n = 1 + (gen.next() % 10)
                return [gen.bigint(var.variation_bit_size) for i in range(n)]

            if issubclass(var, Explicit):
                this_item = (cat, name)
                if not this_item in self.expand:
                    return None
                sub = self.refs[cat].variation
                val = random_var(sub)
                return sub(val).unparse_bits().to_bytes() or None # avoid empty

            if issubclass(var, Compound):
                d = {}
                for (name, (_title, sub, _fspec)) in var.subitems_dict.items():
                    populate_this_item = self.populate_all_items or gen.bool()
                    if populate_this_item:
                        x = random_var(sub, name)
                        if not x is None:
                            d[name] = x
                return d or None # turn {} into None, to skip this subitem
            raise Exception('internal error, unexpected variation', var)

        while True:

            if hasattr(cls, 'make_record'):
                var = cls.variation
                rec = cls.make_record(random_var(var))
            else:
                uap = gen.choose(list(cls.uaps.keys()))
                var = cls.uaps[uap]
                rec = cls.make_record_unsafe(uap, random_var(var))
                if not cls.is_valid(rec):
                    continue

            return rec

    def __next__(self):
        cat = self.gen.choose(list(self.valid_specs.keys()))
        cls = self.valid_specs[cat]
        rec = self.random_record(cat, cls)
        db = cls.make_datablock([rec])
        return db.unparse()

def cmd_gen_random(args):
    """Generate random samples."""
    sel = get_selection(args.empty_selection, args.cat or [], args.ref or [])
    exp = get_expansions(sel, args.expand or [])
    populate_all_items = args.populate_all_items
    seed = args.seed
    if seed is None:
        seed = random.randint(0,pow(2,64)-1)
    gen = PCG32(seed)
    for sample in AsterixSamples(gen, sel, exp, populate_all_items):
        output(sample.hex())
        if args.sleep is not None:
            time.sleep(args.sleep)

def cmd_asterix_decoder(args):
    parsing_opt = get_parsing_options(args)
    sel = get_selection(args.empty_selection, args.cat or [], args.ref or [])
    exp = get_expansions(sel, args.expand or [])
    if args.truncate:
        def smax(n, s):
            return s if (len(s) <= n) else (s[0:n] + '|')
        truncate = lambda s: output(smax(args.truncate, s))
    else:
        truncate = lambda s: output(s)

    def too_deep(i):
        """Parsing level check."""
        max_level = args.parsing_level
        if max_level <= 0:
            return False
        return i > max_level

    def handle_variation(cat, i, var, path):
        if too_deep(i): return
        cls = var.__class__

        def path_line(name, title, bits):
            truncate('{}{}: "{}", {}, len={} bits, bin={}'.format('  '*i, path+[name], title, sub.__class__.variation, len(bits), str(bits)))

        if isinstance(var, Element):
            x = var.to_uinteger()
            if hasattr(var, 'table_value'):
                tv = var.table_value
                if tv is None:
                    tv = '(undefined value)'
                truncate('{}value: {} -> {}'.format('  '*i, x, tv))
            elif hasattr(var, 'to_string'):
                truncate('{}value: {} -> {}'.format('  '*i, x, repr(var.to_string())))
            elif hasattr(var, 'to_quantity'):
                truncate('{}value: {} -> {} {}'.format('  '*i, x, var.to_quantity(), var.__class__.quantity.unit))
            else:
                truncate('{}value: {} = {} = {}'.format('  '*i, x, hex(x), oct(x)))

        elif isinstance(var, Group):
            for j in cls.subitems_list:
                if type(j) is tuple:
                    name = j[0]
                    title = cls.subitems_dict[name][0]
                    sub = var.get_item(name)
                    bits = sub.unparse_bits()
                    path_line(name, title, bits)
                    handle_variation(cat, i+1, sub, path+[name])
                else:
                    truncate('{}Spare len={} bits'.format('  '*i, j.bit_size))

        elif isinstance(var, Extended):
            for j in cls.subitems_list:
                for k in j:
                    if type(k) is tuple:
                        name = k[0]
                        title = cls.subitems_dict[name][0]
                        sub = var.get_item(name)
                        if sub is None:
                            continue
                        bits = sub.unparse_bits()
                        path_line(name, title, bits)
                        handle_variation(cat, i+1, sub, path+[name])
                    else:
                        truncate('{}Spare len={} bits'.format('  '*i, k.bit_size))

        elif isinstance(var, Repetitive):
            for cnt, sub in enumerate(var):
                truncate('{}subitem {}'.format('  '*i, cnt))
                handle_variation(cat, i+1, sub, path+[cnt])

        elif isinstance(var, Explicit):
            this_item = (cat, path[0])
            if not this_item in exp:
                return
            sub = sel['REFS'][cat].variation
            bits = Bits.from_bytes(var.raw)
            try:
                (val, b) = sub.parse_bits(bits, parsing_opt)
                if len(b):
                    raise AsterixError('Unexpected remaining bits in explicit item')
                handle_variation(cat, i, val, path)
            except AsterixError as e:
                truncate('Error!', e)
                truncate('Unable to parse explicit subitem:', s.hex())
                if args.stop_on_error:
                    sys.exit(1)

        elif isinstance(var, Compound):
            for j in cls.subitems_list:
                if j is None:
                    continue
                name = j[0]
                title = cls.subitems_dict[name][0]
                sub = var.get_item(name)
                if sub is None:
                    continue
                bits = sub.unparse_bits()
                path_line(name, title, bits)
                handle_variation(cat, i+1, sub, path+[name])
        else:
            raise Exception('internal error, unexpected variation', var)

    def handle_record(cat, i, rec):
        if too_deep(i): return
        raw = rec.unparse_bits().to_bytes()
        truncate('{}record: len={} bytes, hex={}'.format('  '*i, len(raw), raw.hex()))
        handle_variation(cat, i+1, rec, [])

    def handle_datablock(i, db):
        if too_deep(i): return
        cat = db.category
        truncate('{}datablock: cat={}, len={} bytes, records={}'.format('  '*i, cat, db.length, db.raw_records.hex()))
        spec = sel['CATS'].get(cat)
        if spec is None:
            return
        try:
            db = spec.parse(db, parsing_opt)
            for rec in db.records:
                handle_record(cat, i+1, rec)
        except AsterixError as e:
            truncate('Error! {}'.format(e))
            truncate('Unable to parse datablock: {}'.format(db.unparse().hex()))
            if args.stop_on_error:
                sys.exit(1)

    def handle_datagram(i, s):
        if too_deep(i): return
        truncate('{}datagram: len={} bytes, hex={}'.format('  '*i, len(s), s.hex()))
        try:
            dbs = RawDatablock.parse(s)
        except AsterixError as e:
            truncate('Error! {}'.format(e))
            truncate('Unable to parse datagram: {}'.format(s.hex()))
            if args.stop_on_error:
                sys.exit(1)
            return
        for db in dbs:
            handle_datablock(i+1, db)

    def handle_event(t, line):
        s = bytes.fromhex(line)
        truncate('timestamp: {}'.format(t if t is not None else '<unknown>'))
        handle_datagram(1, s)

    if args.file is None:
        for line in fileinput.input('-'):
            t = datetime.datetime.now(tz=datetime.timezone.utc)
            handle_event(t, line)
    else:
        fmt = format_find(format_input, args.format)(args.channel)
        for event in fmt.events(args.file):
            (t_mono, t_utc, data) = event
            handle_event(t_utc, data)

def cmd_from_udp(args):
    sel = selectors.DefaultSelector()
    for (ip, port) in args.unicast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, int(port)))
        sock.setblocking(False)
        sel.register(sock, selectors.EVENT_READ)
    for (mcast, port, local_ip) in args.multicast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((mcast, int(port)))
        mreq = socket.inet_aton(mcast) + socket.inet_aton(local_ip)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setblocking(False)
        sel.register(sock, selectors.EVENT_READ)

    # processing loop
    while True:
        select_events = sel.select()
        for key, mask in select_events:
            sock = key.fileobj
            (s, addr) = sock.recvfrom(pow(2,16))
            output(s.hex())

def check_ttl(arg):
    ttlInterval = (1, 255)
    try:
        val = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError('must be integer')
    try:
        assert val >= ttlInterval[0]
        assert val <= ttlInterval[1]
    except AssertionError:
        raise argparse.ArgumentTypeError('must be in interval [{},{}]'.format(ttlInterval[0], ttlInterval[1]))
    return val

def cmd_to_udp(args):
    sockets = []
    for (ip, port) in args.unicast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockets.append((sock, ip, int(port)))
    for (mcast, port, local_ip) in args.multicast:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, args.ttl)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip))
        sockets.append((sock, mcast, int(port)))

    # processing loop
    for line in fileinput.input('-'):
        s = bytes.fromhex(line)
        for (sock, ip, port) in sockets:
            sock.sendto(s, (ip, port))

def cmd_inspect(args):
    hex_errors = 0
    raw_datablock_errors = 0
    unknown_categories = set()
    processed_categories = set()
    parse_errors = dict()
    parsing_opt = get_parsing_options(args)

    def handle_event(line):
        nonlocal hex_errors, raw_datablock_errors, unknown_categories, processed_categories, parse_errors
        try:
            s = bytes.fromhex(line)
        except ValueError:
            hex_errors += 1
            return
        try:
            raw_datablocks = ast.RawDatablock.parse(s)
        except AsterixError:
            raw_datablocks = []
            raw_datablock_errors += 1
        for raw_db in raw_datablocks:
            cat = raw_db.category
            editions = ast.manifest['CATS'].get(cat)
            if editions is None:
                unknown_categories.add(cat)
            processed_categories.add(cat)
            for ed in editions:
                Spec = ast.manifest['CATS'][cat][ed]
                try:
                    db = Spec.parse(raw_db, parsing_opt)
                except AsterixError:
                    problems = parse_errors.get(cat, set())
                    problems.add(ed)
                    parse_errors[cat] = problems

    try:
        if args.file is None:
            for line in fileinput.input('-'):
                handle_event(line)
        else:
            fmt = format_find(format_input, args.format)(args.channel)
            for event in fmt.events(args.file):
                (t_mono, t_utc, data) = event
                handle_event(data)
    except KeyboardInterrupt:
        pass

    print('done...')
    print('hex errors: {}'.format(hex_errors))
    print('datablock erros: {}'.format(raw_datablock_errors))
    print('unknown categories: {}'.format(sorted(unknown_categories)))
    print('success category/edition:')
    for cat in sorted(processed_categories):
        editions = set(ast.manifest['CATS'][cat].keys())
        problems = parse_errors.get(cat, set())
        editions.difference_update(problems)
        print('{} -> {}'.format(cat, sorted(editions, key=string_to_edition)))
    print('problems category/edition:')
    for cat in sorted(processed_categories):
        problems = parse_errors.get(cat)
        if not problems:
            continue
        print('{} -> {}'.format(cat, sorted(problems, key=string_to_edition)))

def cmd_record(args):
    fmt = format_find(format_output, args.format)(args.channel, args.sender)
    for line in fileinput.input('-'):
        t_mono = time.monotonic()
        t_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        fmt.write_event(t_mono, t_utc, line.strip())

def cmd_replay(args):
    fmt = format_find(format_input, args.format)(args.channel)
    offset = None # not known until the first event
    for event in fmt.events(args.infile):
        now = time.monotonic()
        (t_mono, t_utc, data) = event
        if not args.full_speed:
            if offset is None:
                offset = now - t_mono
            t = t_mono + offset
            delta = t - now
            if delta > 0:
                time.sleep(delta)
        output(data)

def cmd_custom(args):
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

    # Call function with the following arguments
    #   - asterix module (already imported)
    #   - file input object (rx from stdin)
    #   - output function (tx to stdout)
    #   Custom modul can process lines, for example: for line in rx: tx(line)
    f = run_globals[args.call]
    f(ast, fileinput.input('-'), output, args.args)

def main():

    parser = argparse.ArgumentParser(description='Asterix data processor.')

    parser.add_argument('--version', action='version',
        version='%(prog)s {}, asterix-lib {}'.format(__version__, ast.VERSION),
        help='show the version number and exit')

    parser.add_argument('--empty-selection', action='store_true',
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

    parser.add_argument('--no-check-spare',
        action='store_true',
        help='do not check spare bits for zero value when parsing')

    subparsers = parser.add_subparsers(required=True, help='sub-commands')

    # 'manifest' command
    parser_manifest = subparsers.add_parser('manifest', help='show available categories')
    parser_manifest.set_defaults(func=cmd_show_manifest)
    parser_manifest.add_argument('--latest', action='store_true',
        help='show latest editions only')

    # 'random' command
    parser_random = subparsers.add_parser('random', help='asterix sample generator')
    parser_random.set_defaults(func=cmd_gen_random)
    parser_random.add_argument('--sleep', type=float,
        help="sleep 't' seconds between random samples")
    parser_random.add_argument('--seed', type=int,
        help='randomm generator seed value')
    parser_random.add_argument('--populate-all-items', action='store_true',
        help='populate all defined items instead of random selection')

    # 'decode' command
    parser_decode = subparsers.add_parser('decode', help='asterix decoder')
    parser_decode.set_defaults(func=cmd_asterix_decoder)
    parser_decode.add_argument('--truncate', type=int,
        metavar='N', default=0,
        help='truncate long data lines to N characters or 0 for none, default: %(default)s')
    parser_decode.add_argument('--stop-on-error',
        action='store_true',
        help='exit on first parsing error')
    parser_decode.add_argument('--parsing-level', type=int,
        metavar='N', default=0,
        help='limit parsing depth, 0 for none, default: %(default)s')
    parser_decode.add_argument('--file', metavar='filename',
        help='use file input instead of realtime over STDIN')
    parser_decode.add_argument('--format',
        choices=[fmt.name for fmt in format_input],
        default=format_input[0].name,
        help='file data format, default: %(default)s')
    parser_decode.add_argument('--channel', metavar='STR',
        help='Set channel name (not supported on all formats)')

    # 'from-udp' command
    parser_from_udp = subparsers.add_parser('from-udp', help='UDP datagram receiver')
    parser_from_udp.set_defaults(func=cmd_from_udp)
    parser_from_udp.add_argument('--unicast', action='append', help='Unicast UDP input',
        default=[], nargs=2, metavar=('ip', 'port'))
    parser_from_udp.add_argument('--multicast', action='append', help='Multicast UDP input',
        default=[], nargs=3, metavar=('mcast-ip', 'port', 'local-ip'))

    # TTL argument
    parser.add_argument('--multicast-ttl', dest='ttl', type=check_ttl, default=32,
        help='Time to live for outgoing multicast traffic, default: %(default)s')

    # 'to-udp' command
    parser_to_udp = subparsers.add_parser('to-udp', help='UDP datagram transmitter')
    parser_to_udp.set_defaults(func=cmd_to_udp)
    parser_to_udp.add_argument('--unicast', action='append', help='Unicast UDP output',
        default=[], nargs=2, metavar=('ip', 'port'))
    parser_to_udp.add_argument('--multicast', action='append', help='Multicast UDP output',
        default=[], nargs=3, metavar=('mcast-ip', 'port', 'local-ip'))

    # 'inspect' command
    parser_inspect = subparsers.add_parser('inspect',
        help='report asterix parsing status per category/edition')
    parser_inspect.set_defaults(func=cmd_inspect)
    parser_inspect.add_argument('--file', metavar='filename',
        help='use file input instead of realtime over STDIN')
    parser_inspect.add_argument('--format',
        choices=[fmt.name for fmt in format_input],
        default=format_input[0].name,
        help='file data format, default: %(default)s')
    parser_inspect.add_argument('--channel', metavar='STR',
        help='Set channel name (not supported on all formats)')

    # 'record' command
    parser_record = subparsers.add_parser('record',
        help='data recorder, augment data with timestamp')
    parser_record.set_defaults(func=cmd_record)
    parser_record.add_argument('--channel', metavar='STR',
        help='Set channel name (not supported on all formats)')
    parser_record.add_argument('--sender', help='Set sender attribute',
        nargs=2, metavar=('ip', 'port'))
    parser_record.add_argument('--format',
        choices=[fmt.name for fmt in format_output],
        default=format_output[0].name,
        help='data format, default: %(default)s')

    # 'replay' command
    parser_replay = subparsers.add_parser('replay',
        help='data replay from recording')
    parser_replay.set_defaults(func=cmd_replay)
    parser_replay.add_argument('--channel', metavar='STR',
        help='Set channel name (not supported on all formats)')
    parser_replay.add_argument('--format',
        choices=[fmt.name for fmt in format_input],
        default=format_input[0].name,
        help='data format, default: %(default)s')
    parser_replay.add_argument('--full-speed', action='store_true',
        help='Replay at full speed')
    parser_replay.add_argument('infile', nargs='?')

    # 'custom' command
    parser_custom = subparsers.add_parser('custom', help='run custom python script')
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
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == '__main__':
    main()

