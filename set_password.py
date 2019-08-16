#!/usr/bin/env python3

import sys

from notebook.auth import passwd


def run():
    password = sys.argv[1:].pop()

    if not password:
        print('Error: Missing or empty password.', file=sys.stderr)
    else:
        encoded = passwd(password)
        print(encoded)


if __name__ == '__main__':
    run()
