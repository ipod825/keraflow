import os
import tarfile
import urllib
import shutil
from six.moves.urllib.error import URLError, HTTPError

from tqdm import tqdm

from ..konfig import keraflow_dir


def download(url, untar_name=None):
    datadir = os.path.join(keraflow_dir, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fpath = os.path.join(datadir, os.path.basename(url))

    def tqdm_hook(t):
        last_b = [0]

        def inner(transfered_blocks=1, block_size=1, total_size=None):
            if total_size is not None:
                t.total = total_size
            t.update((transfered_blocks - last_b[0]) * block_size)
            last_b[0] = transfered_blocks
        return inner

    def rollback_rm(path):
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)

    if not os.path.exists(fpath):
        print('Downloading {} ...'.format(fpath))

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                with tqdm(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(fpath)) as t:
                    urllib.urlretrieve(url, filename=fpath, reporthook=tqdm_hook(t), data=None)
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(url, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            rollback_rm(fpath)
            raise

    if untar_name is not None:
        datadir = os.path.dirname(fpath)
        untar_path = os.path.join(datadir, untar_name)
        if not os.path.exists(untar_path):
            print('Untaring file...')
            try:
                tfile = tarfile.open(fpath, 'r:gz')
                tfile.extractall(path=datadir)
                tfile.close()
            except (Exception, KeyboardInterrupt) as e:
                rollback_rm(untar_path)
        return untar_path

    return fpath
