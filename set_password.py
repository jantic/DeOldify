#!/usr/bin/env python3

import sys

from notebook.auth import passwd


def run():
    args = sys.argv[1:]

    if not args:
        print('Error: Missing password.', file=sys.stderr)
        return

    password = args[0]

    if not password:
        print('Error: Empty password.', file=sys.stderr)
    else:
        encoded = passwd(password)
        print(encoded)


if __name__ == '__main__':
    run()
