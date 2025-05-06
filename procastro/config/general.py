import re
from pathlib import Path
from shutil import copy2

import platformdirs as pd
import toml
from procastro.config.definitions import AppName, AppAuthor

__all__ = ['config_user', 'config_save', 'config_update_file']

config_dir = pd.user_config_dir(AppName, AppAuthor)
cache_dir = pd.user_cache_dir(AppName, AppAuthor)

def config_user(section: str,
                default: str = None,
                read_default: bool = False,
                ):
    """
    Returns in a TOML-style dictionary with variables taken from the configuration file. The first time
    a section is run it will take values from the default/ directory... afterward it will copy the
    config dir to the user directory.

    Parameters
    ----------
    section: str
    Name of the section config to load

    read_default: bool
    whether to return the default configuration file or the user's

    default: str
    if given, then uses this file as default value instead, it will be located in the user directory
    """
    user_dir = Path(config_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    user_file = user_dir.joinpath(f"{section}.toml")

    # if the user file does not exist, then read from default add cache's directory and save
    if not user_file.exists() or read_default:
        if default is None:
            default = Path(__file__).parent.joinpath("..", "defaults", f"{section}.toml")
        else:
            default = user_dir / default

        config_content = ""
        if default.exists():
            config_content = default.read_text(encoding='utf-8')

        # copy files that should go into config directory
        replace_pattern = re.compile(r'"%CONFIG_DIR%/(.+?)"')
        changes = []
        for filename in re.findall(replace_pattern, config_content):
            orig = Path(__file__).parent.joinpath('../defaults') / filename
            target = user_dir / filename
            print(f"copying {orig} to {target}")
            copy2(orig, target)
            config_content = re.sub(f'"%CONFIG_DIR%/{filename}"',
                                    f'"{str(target).encode('unicode_escape').decode().encode('unicode_escape').decode()}"',
                                    config_content,
                                    count=1)
            print(config_content)

        # add cache directory
        config = toml.loads(config_content)
        config['cache_dir'] = str(Path(cache_dir).joinpath(section))

        toml.dump(config, open(user_file, 'w', encoding='utf-8'))

        if read_default:
            return config

    config =  toml.loads(user_file.read_text(encoding='utf-8'))
    if 'cache_dir' in config:
        Path(config['cache_dir']).mkdir(parents=True, exist_ok=True)

    return config


def config_save(section: str,
                config: dict,
                ):
    section = f"{section}.toml"
    user_file = Path(config_dir).joinpath(section)

    return toml.dump(config, open(user_file, 'w', encoding='utf-8'))


def config_update_file(section: str,
                       key: str,
                       value: any,
                       ) -> dict:
    """
    update configuration file

    Parameters
    ----------
    section: str
    Section to update

    key: str
    key to update.  Use period (as in subsection.key) to update a value in a subsection

    value: any
    new value of key

    Returns
    -------
    dict
    updated configuration dictionary
    """

    orig_config = config_user(section)

    config = orig_config
    subsections = key.split(".")
    for subsection in subsections[:-1]:
        config = config[subsection]
    config[subsections[-1]] = value

    config_save(section, orig_config)

    return config

