#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Asterix data processing tool

import argparse
import fileinput
import random
import sys
import datetime
import socket
import selectors

from asterix import *

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

def string_to_edition(ed):
    """Convert edition string to a tuple, for example "1.2" -> (1,2)"""
    a,b = ed.split('.')
    return (int(a), int(b))

def show_manifest(args):

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
                for (name, (sub, _fspec)) in var.subitems_dict.items():
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

def gen_random(args):
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

def asterix_decoder(args):
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
                    sub = var.get_item(name)
                    bits = sub.unparse_bits()
                    truncate('{}{}: {}, len={} bits, bin={}'.format('  '*i, path+[name], sub.__class__.variation, len(bits), str(bits)))
                    handle_variation(cat, i+1, sub, path+[name])
                else:
                    truncate('{}Spare len={} bits'.format('  '*i, j.bit_size))

        elif isinstance(var, Extended):
            for j in cls.subitems_list:
                for k in j:
                    if type(k) is tuple:
                        name = k[0]
                        sub = var.get_item(name)
                        if sub is None:
                            continue
                        bits = sub.unparse_bits()
                        truncate('{}{}: {}, len={} bits, bin={}'.format('  '*i, path+[name], sub.__class__.variation, len(bits), str(bits)))
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
                (val, b) = sub.parse_bits(bits)
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
                sub = var.get_item(name)
                if sub is None:
                    continue
                bits = sub.unparse_bits()
                truncate('{}{}: {}, len={} bits, bin={}'.format('  '*i, path+[name], sub.__class__.variation, len(bits), str(bits)))
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
            db = spec.parse(db)
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

    for line in fileinput.input('-'):
        t = datetime.datetime.now()
        s = bytes.fromhex(line)
        truncate('timestamp: {}'.format(t if t is not None else '<unknown>'))
        handle_datagram(1, s)

def from_udp(args):
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
        events = sel.select()
        for key, mask in events:
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

def to_udp(args):
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

def main():

    parser = argparse.ArgumentParser(description='Asterix data processor.')

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

    subparsers = parser.add_subparsers(required=True, help='sub-commands')

    # 'manifest' command
    parser_manifest = subparsers.add_parser('manifest', help='show available categories')
    parser_manifest.set_defaults(func=show_manifest)
    parser_manifest.add_argument('--latest', action='store_true',
        help='show latest editions only')

    # 'random' command
    parser_random = subparsers.add_parser('random', help='asterix sample generator')
    parser_random.set_defaults(func=gen_random)
    parser_random.add_argument('--seed', type=int,
        help='randomm generator seed value')
    parser_random.add_argument('--populate-all-items', action='store_true',
        help='populate all defined items instead of random selection')

    # 'decode' command
    parser_decode = subparsers.add_parser('decode', help='asterix decoder')
    parser_decode.set_defaults(func=asterix_decoder)
    parser_decode.add_argument('--truncate', type=int,
        metavar='N', default=0,
        help='truncate long data lines to N characters or 0 for none, default: %(default)s')
    parser_decode.add_argument('--stop-on-error',
        action='store_true',
        help='exit on first parsing error')
    parser_decode.add_argument('--parsing-level', type=int,
        metavar='N', default=0,
        help='limit parsing depth, 0 for none, default: %(default)s')

    # 'from-udp' command
    parser_from_udp = subparsers.add_parser('from-udp', help='UDP datagram receiver')
    parser_from_udp.set_defaults(func=from_udp)
    parser_from_udp.add_argument('--unicast', action='append', help='Unicast UDP input',
        default=[], nargs=2, metavar=('ip', 'port'))
    parser_from_udp.add_argument('--multicast', action='append', help='Multicast UDP input',
        default=[], nargs=3, metavar=('mcast-ip', 'port', 'local-ip'))

    # TTL argument
    parser.add_argument('--multicast-ttl', dest='ttl', type=check_ttl, default=32,
        help='Time to live for outgoing multicast traffic, default: %(default)s')

    # 'to-udp' command
    parser_to_udp = subparsers.add_parser('to-udp', help='UDP datagram transmitter')
    parser_to_udp.set_defaults(func=to_udp)
    parser_to_udp.add_argument('--unicast', action='append', help='Unicast UDP output',
        default=[], nargs=2, metavar=('ip', 'port'))
    parser_to_udp.add_argument('--multicast', action='append', help='Multicast UDP output',
        default=[], nargs=3, metavar=('mcast-ip', 'port', 'local-ip'))

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == '__main__':
    main()

