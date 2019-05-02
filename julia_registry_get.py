# coding: utf-8

"""Fetches the latest Julia registry packages and Julia language
dependencies, and packages them nicely for ingress.

"""

import logging
import os
import shutil
import subprocess
import sys

from typing import Tuple, Callable, Type, Any

import toml

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(threadName)s:%(message)s')

# Path to store package registry and the Git executable
PACKAGES_PATH = 'K:\\julia_packages2'
GIT = 'K:\\Git\\bin\\git.exe'


def run(*args, **kwargs):

    """Runs a command using subprocess."""

    logging.info('Running %s',
                 ' '.join(list(args) + [key + '=' + val
                                        for key, val in kwargs.items()]))
    return subprocess.run(args, **kwargs)


def onerror(func: Callable, path: str, exc_info: Tuple[Type, Exception, Any]):

    """Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file) it attempts
    to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``

    """

    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def main():

    """The main function."""

    # Set the URL to the Julia default package registry and clone/fetch the
    # latest registry
    logging.info('Cloning package registry')
    reg_url = 'https://github.com/JuliaRegistries/General.git'
    reg_path = os.path.join(PACKAGES_PATH, 'General')
    if os.path.exists(reg_path):
        run(GIT, 'pull', cwd=reg_path)
    else:
        run(GIT, 'clone', reg_url, reg_path)

    # Parse the registry TOML file into a really big dictionary
    logging.info('Parsing registry file')
    with open(os.path.join(reg_path, 'Registry.toml')) as infile:
        registry_data = toml.load(infile)

    # Walk through the TOML dictionary, grabbing the package names and their
    # Git repo URLs
    logging.info('Gathering package names')
    pkg_download = {}
    for pkg in sorted(registry_data['packages'].values(),
                      key=lambda x: x['name']):
        toml_file = os.path.join(reg_path, pkg['path'], 'Package.toml')
        with open(toml_file) as infile:
            pkg_toml = toml.load(infile)
        pkg_repo = pkg_toml['repo'].replace('git://', 'https://')
        pkg_download[pkg_toml['name']] = pkg_repo

    # Loop over the package URLs, cloning or fetching each one to ensure
    # we have the latest and the clone is good
    logging.info('Cloning or fetching packages')
    for pkg in pkg_download.values():
        pkg_repo_base = os.path.basename(pkg)
        pkg_path = os.path.join(PACKAGES_PATH, pkg_repo_base)
        if os.path.exists(pkg_path):
            run(GIT, 'fetch', cwd=pkg_path)
        else:
            run(GIT, 'clone', '--mirror', pkg_repo, pkg_path, input=b'\n\n')

    # The Julia language repo needs to download some dependencies when it's
    # built. This is normally done using "make -d deps getall", but we don't
    # have GNU make on the IAS. So we have to walk the Makefiles ourselves
    # and manuall download the dependencies so we can ingress them, too!

    # Clone Julia from the bare repo
    logging.info('Cloning non-bare Julia language repo')
    julia_repo = os.path.join(PACKAGES_PATH, 'julia.git')
    julia_path = os.path.join(PACKAGES_PATH, 'julia')
    if os.path.exists(julia_path):
        shutil.rmtree(julia_path, onerror=onerror)
    run(GIT, 'clone', julia_repo, julia_path)

    # Run GNU make to get the dependencies
    # logging.info('Downloading Julia dependencies')
    # ret = run('make', 'getall', cwd=os.path.join(julia_path, 'deps'))
    # if ret != 0:
    #     raise RuntimeError("Getting Julia dependencies failed!")


    # Zip up the directory to transfer to NNPP
    # logging.info('Packaging Julia installation tarball')
    # topdir = os.path.realpath(os.path.dirname(PACKAGES_PATH))
    # basename = os.path.basename(PACKAGES_PATH)
    # run('tar', 'czf', basename + '.tar.gz', basename, cwd=topdir)


if __name__ == '__main__':
    sys.exit(main())
