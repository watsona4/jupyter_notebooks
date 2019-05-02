# coding: utf-8

import asyncio
import logging
import os
import re
import shutil
import subprocess
import textwrap
import toml

from typing import Tuple, Callable, Type, Any, List, Dict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(threadName)s:%(message)s')

# Path to store package registry and the Git executable
PACKAGES_PATH = 'K:\\julia_packages2'
GIT = 'K:\\Git\\bin\\git.exe'

# The maximum number of concurrent workers
NUM_WORKERS = 10


async def run(*args, **kwargs):

    """Runs a command using subprocess under asyncio."""

    logging.info('Running %s',
                 ' '.join(list(args) + [key + '=' + val
                                        for key, val in kwargs.items()]))
    process = await asyncio.create_subprocess_exec(*args, **kwargs)
    return await process.wait()


async def clone_mirror(pkg_repo: str, packages_path: str = PACKAGES_PATH,
                       redo: bool = False) -> Tuple[str, int]:

    """Performs a mirror Git clone of a repository.

    Args:
        pkg_repo: The URL of the Git repository to clone.
        packages_path: The path to the local directory to clone into.
        redo: If a clone exists, delete it and re-clone, otherwise do nothing.

    Returns:
        pkg_repo: The input `pkg_repo` parameter.
        ret: The return code from Git, or ``None`` if `redo` was ``False`` and
            the clone existed.

    """

    pkg_repo_base = os.path.basename(pkg_repo)
    pkg_path = os.path.join(packages_path, pkg_repo_base)

    ret = None

    if redo:
        try:
            shutil.rmtree(pkg_path)
        except FileNotFoundError:
            pass

    if not os.path.exists(pkg_path):
        logging.info('Cloning %s from %s', pkg_repo_base, pkg_repo)
        ret = await run(GIT, 'clone', '--mirror', pkg_repo, pkg_path)

    return pkg_repo, ret


async def worker(func: Callable, queue: asyncio.Queue, results: List[Any]):

    """Runs a function on data in the provided queue, putting the function's
    result in the results list.

    Args:
      func: the function to run
      queue: the ``asyncio.Queue``
      results: the list of function results

    """

    while True:
        data = await queue.get()
        result = await func(data)
        queue.task_done()
        results.append(result)


async def run_queue(func: Callable, packages: Dict[str, str],
                    num_workers: int = NUM_WORKERS) -> Dict[str, int]:

    """Runs the function on the packages asynchronously using an ``asyncio.Queue``.

    Args:
      func: the function to run
      packages: a dictionary containing the package name and its URL
      num_workers: the maximum number of concurrent tasks

    Returns:
      A dictionary containing the package name and the return code from Git.

    """

    queue = asyncio.Queue()
    for pkg in packages.values():
        queue.put_nowait(pkg)

    results = list()

    tasks = [asyncio.ensure_future(worker(func, queue, results))
             for _ in range(num_workers)]

    await queue.join()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    return dict(results)


async def fetch(pkg_repo: str,
                packages_path: str = PACKAGES_PATH) -> Tuple[str, int]:

    """Performs a fetch in a local Git repository, whose location is
    determined from the remote URL.

    Args:
        pkg_repo: The URL of the remote Git repository.
        packages_path: The path to the local directory where the clone
          is located.

    Returns:
        pkg_repo: The input `pkg_repo` parameter.
        ret: The return code from Git, or ``None`` if the clone doesn't
          exist.

    """

    pkg_repo_base = os.path.basename(pkg_repo)
    pkg_path = os.path.join(packages_path, pkg_repo_base)

    ret = None

    if os.path.exists(pkg_path):
        logging.info('Fetching in %s', pkg_path)
        ret = await run(GIT, 'fetch', cwd=pkg_path)

    return pkg_repo, ret


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


async def main():

    # Set the URL to the Julia default package registry and clone/fetch the
    # latest registry
    logging.info('Cloning package registry')
    reg_url = 'https://github.com/JuliaRegistries/General.git'
    reg_path = os.path.join(PACKAGES_PATH, 'General')
    if os.path.exists(reg_path):
        await run(GIT, 'pull', cwd=reg_path)
    else:
        await run(GIT, 'clone', reg_url, reg_path)

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

    # Async loop over the package URLs, cloning each one and storing the
    # return codes in a dictionary
    logging.info('Cloning packages')
    results = await run_queue(clone_mirror, pkg_download)

    # Async loop over the package URLs, fetching on each one to ensure we
    # have the latest and the clone is good
    logging.info('Fetching latest package updates')
    results = await run_queue(fetch, pkg_download)

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
    await run(GIT, 'clone', julia_repo, julia_path)

    # Run GNU make to get the dependencies
    # logging.info('Downloading Julia dependencies')
    # pwd = os.getcwd()
    # os.chdir(os.path.join(julia_path, 'deps'))
    # ret = os.system(f"make getall")
    # os.chdir(pwd)
    # if ret != 0:
    #     raise RuntimeError("Getting Julia dependencies failed!")


    # Zip up the directory to transfer to NNPP
    # logging.info('Packaging Julia installation tarball')
    # topdir = os.path.realpath(os.path.dirname(PACKAGES_PATH))
    # loop.run_until_complete(run('tar', 'czf', 'julia_packages.tar.gz',
    #                             'julia_packages', cwd=topdir))


if __name__ == '__main__':
    asyncio.set_event_loop(asyncio.ProactorEventLoop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
