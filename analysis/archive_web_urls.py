#!/usr/bin/env python3
"""Save public HTTP pages referenced by recorded tool replies after evaluation."""

import argparse
import csv
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urljoin, urlsplit, urlunsplit

URL_RE = re.compile(r"""https?://[^\s)\]}>,<"']+""")


def fetch(url, timeout, max_bytes):
    for _ in range(6):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username is not None:
            raise ValueError('expected a public HTTP(S) URL')
        host = parsed.hostname.encode('idna').decode('ascii')
        secure = parsed.scheme == 'https'
        port = parsed.port or (443 if secure else 80)
        addresses = [entry[4][0] for entry in socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM)]
        if not addresses or any(not ipaddress.ip_address(ip).is_global
                                or ipaddress.ip_address(ip).is_multicast for ip in addresses):
            raise ValueError('non-public address')
        # Connect to a validated IP. Preserve the hostname for the Host header and TLS.
        with closing(http.client.HTTPConnection(host, port, timeout=timeout)) as connection:
            for address in addresses:
                try:
                    connection.sock = socket.create_connection((address, port), timeout)
                    break
                except OSError:
                    if address == addresses[-1]:
                        raise
            if secure:
                connection.sock = ssl.create_default_context().wrap_socket(
                    connection.sock, server_hostname=host)
            target = urlunsplit(('', '', parsed.path or '/', parsed.query, ''))
            connection.request('GET', target, headers={'User-Agent': 'GTA-web-archive/1.0'})
            with connection.getresponse() as response:
                if response.status in (301, 302, 303, 307, 308) and response.getheader('Location'):
                    url = urljoin(url, response.getheader('Location'))
                    continue
                if not 200 <= response.status < 300:
                    raise ValueError('HTTP {}'.format(response.status))
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise ValueError('response exceeds --max-bytes')
                return url, response.getheader('Content-Type'), body
    raise ValueError('too many redirects')


def atomic_write(path, body):
    tmp = NamedTemporaryFile(dir=path.parent, delete=False)
    name = Path(tmp.name)
    try:
        with tmp:
            tmp.write(body)
        os.replace(name, path)
    finally:
        if name.exists():
            name.unlink()


def archive(urls, output, timeout, max_bytes):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    entries = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    for url in sorted(set(urls)):
        name = hashlib.sha256(url.encode()).hexdigest() + '.body'
        target = output / name
        old = entries.get(url, {})
        if (old.get('status') == 'saved' and target.is_file() and
                hashlib.sha256(target.read_bytes()).hexdigest() == old.get('sha256')):
            continue
        item = {'url': url, 'archived_after_evaluation': True,
                'archived_at_utc': datetime.now(timezone.utc).isoformat()}
        try:
            final_url, mime, body = fetch(url, timeout, max_bytes)
            atomic_write(target, body)
            item.update(status='saved', path=name, final_url=final_url,
                        content_type=mime, sha256=hashlib.sha256(body).hexdigest())
        except (OSError, ValueError, http.client.HTTPException) as exc:
            item.update(status='failed', error=str(exc))
        entries[url] = item
        atomic_write(manifest_path, json.dumps(entries, ensure_ascii=False, indent=2).encode())
    if not manifest_path.exists():
        atomic_write(manifest_path, b'{}')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--max-bytes', type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_bytes < 1:
        parser.error('--timeout and --max-bytes must be positive')
    with args.input.open(encoding='utf-8', newline='') as stream:
        urls = {url for row in csv.DictReader(stream)
                for url in URL_RE.findall(row['search_response'])}
    entries = archive(urls, args.output_dir, args.timeout, args.max_bytes)
    print('Saved {} of {} requested URLs'.format(
        sum(entries[url]['status'] == 'saved' for url in urls), len(urls)))


if __name__ == '__main__':
    main()
