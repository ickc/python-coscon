import gzip
import json
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Union

import toml
import yaml
import yamlloader

logger = getLogger('coscon')

H5_CREATE_KW = {
    'compression': 'gzip',
    # shuffle minimize the output size
    'shuffle': True,
    # checksum for data integrity
    'fletcher32': True,
    # turn off track_times so that identical output gives the same md5sum
    'track_times': False
}


def dumper(
    data: dict,
    path: Union[Path, str],
    overwrite: bool = False,
    compress: bool = None
):
    """Write dict to a TOML/YAML/JSON file.

    Dump data to a TOML/YAML/JSON format file, optionally compressing the contents
    with gzip and optionally overwriting the file.

    Args:
        data (dict): The dictionary to be written.
        path (str): The file to write. Valid extensions are .toml, .json, .yaml with optional extension .gz.
        overwrite (bool): If True, overwrite the file if it exists.
            If False, then existing files will cause an exception.
        compress (bool): If True, compress the data with gzip on write. If None, dispatch
            by its extension.

    Returns:
        None

    """
    path = Path(path)

    if path.exists():
        if overwrite:
            if path.is_file():
                path.unlink()
            else:
                raise FileExistsError(f'{path} exists and is not a file, abort overwriting...')
        else:
            raise FileExistsError(f"{path} exists.  Consider using `overwrite` option.")

    ext = path.suffix.lower()
    if compress is None:
        if ext == '.gz':
            logger.info(f'Setting compression to on according to extension {ext}')
            compress = True
    elif compress is True:
        if ext != '.gz':
            raise ValueError(f'Compression is on but the extension {ext} is not .gz. Consider changing that to .gz.')

    if compress:
        # ext must be .gz now
        exts = path.suffixes
        if len(exts) != 2:
            raise ValueError(f'Entension {exts} not understood. Expect for example .toml.gz')
        ext = exts[0].lower()

    dumper_dict = {
        '.json': partial(json.dumps, indent=4, sort_keys=True),
        '.toml': partial(toml.dumps, encoder=toml.TomlNumpyEncoder()),
        '.yaml': yaml.dump if yamlloader is None else partial(yaml.dump, Dumper=yamlloader.ordereddict.CDumper),
    }

    try:
        data_str = dumper_dict[ext](data)
    except AttributeError:
        raise ValueError('Do not understand extension {ext}. Consider choosing .json, .toml, or .yaml.')

    if compress:
        with gzip.open(path, "wb") as f:
            f.write(data_str.encode())
    else:
        with open(path, "w") as f:
            f.write(data_str)
    return


def loader(path: Union[Path, str]) -> dict:
    """Read data from a TOML/YAML/JSON file.

    The file can either be regular text or a gzipped version of a TOML
    file.

    Args:
        path (str): The file to read.

    Returns:
        data (dict): The data read.

    """
    path = Path(path)

    ext = path.suffix.lower()

    if ext == '.gz':
        logger.info(f'Extension is .gz, assume gzip compression is used.')
        compress = True
        exts = path.suffixes
        if len(exts) != 2:
            raise ValueError(f'Entension {exts} not understood. Expect for example .toml.gz')
        ext = exts[0].lower()
    else:
        compress = False

    if compress:
        with gzip.open(path, "rb") as f:
            data_str = f.read().decode()
    else:
        with open(path, "r") as f:
            data_str = f.read()

    loader_dict = {
        '.json': json.loads,
        '.toml': toml.loads,
        '.yaml': yaml.load if yamlloader is None else partial(yaml.load, Loader=yamlloader.ordereddict.CLoader),
    }

    try:
        data = loader_dict[ext](data_str)
    except AttributeError:
        raise ValueError('Do not understand extension {ext}. Consider choosing .json, .toml, or .yaml.')

    return data
