import re
from pathlib import Path
from shutil import copy2

import platformdirs as pdir
import tomllib as toml
import tomli_w as tomlw
from procastro.config.definitions import AppName, AppAuthor

__all__ = ['config_user', 'config_save', 'config_update_file']

config_dir = pdir.user_config_dir(AppName, AppAuthor)
cache_dir = pdir.user_cache_dir(AppName, AppAuthor)


def config_user(section: str,
                default: str = None,
                read_default: bool = False,
                ):
    """
    Returns in a TOML-style dictionary with variables taken from the configuration file. The first time
    a section is run, it will take values from the default/ directory and copy them to the user directory.
    Afterward, it will use the config from the user directory unless forced to read_default.

    Parameters
    ----------
    section: str
    Name of the section config to load

    read_default: bool
    whether to return the default configuration file or the user's

    default: str
    if given, then it uses this file as the default value instead, it will be located in the user directory
    """
    user_dir = Path(config_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    user_file = user_dir.joinpath(f"{section}.toml")

    # if the user file does not exist, then read from default, add cache's directory and save
    if not user_file.exists() or read_default:
        if default is None:
            default = Path(__file__).parent.joinpath("..", "defaults", f"{section}.toml")
        else:
            default = user_dir / default

        config_content = ""
        if default.exists():
            config_content = default.read_text(encoding='utf-8')

        # Any file identified by %CONFIG_DIR%/file will be copied verbatim into user config dir
        # copy files that should go into the config directory
        replace_pattern = re.compile(r'"%CONFIG_DIR%/(.+?)"')
        for filename in re.findall(replace_pattern, config_content):
            orig = Path(__file__).parent.joinpath('../defaults') / filename
            target = user_dir / filename
            copy2(orig, target)
            config_content = re.sub(f'"%CONFIG_DIR%/{filename}"',
                                    f'"{str(target).encode('unicode_escape').decode().encode('unicode_escape').decode()}"',
                                    config_content,
                                    count=1)

        # cache directory will be set according to platform-specific location
        config_content = config_content.replace(r'%CACHE_DIR%',
                                                str(Path(cache_dir).joinpath(section)))

        # other directories will just use their name
        for dir_type in re.findall(re.compile(r'%([A-Z].+)_DIR%'), config_content):
            target_dir = user_dir / dir_type.lower()
            if target_dir.exists():
                if not target_dir.is_dir():
                    raise NotADirectoryError(f"{target_dir} exists, but is not a directory. Error "
                                             f"configuring section '{section}'")
            else:
                target_dir.mkdir(parents=True, exist_ok=True)

            config_content = config_content.replace(f'%{dir_type}_DIR%', str(target_dir))

        # save update config file into user directory
        config = toml.loads(config_content)
        config_save(section, config)

        if read_default:
            return config

    config =  toml.loads(user_file.read_text(encoding='utf-8'))
    if 'cache_dir' in config:
        Path(config['cache_dir']).mkdir(parents=True, exist_ok=True)

    return config


def config_save(section: str,
                config: dict,
                ):
    """Save the `config` TOML-style dictionary into the appropriate config file. This alternative overwrites the
     whole configuration file. If you want to modify a single key-value use `update_config` instead.

    Parameters
    ----------
    section
      name of the config section
    config
       TOML-style dictionary with the configuration parametrers
    """

    section = f"{section}.toml"
    user_file = Path(config_dir).joinpath(section)

    return tomlw.dump(config, open(user_file, 'wb'))


def config_update_file(section: str,
                       key: str,
                       value: any,
                       ) -> dict:
    """
    update one key-value pair in a configuration file

    Parameters
    ----------
    section: str
    Section to update

    key: str
    key to update.  Use period (as in subsection.key) to update a value in a subsection

    value: any
    new value of a key

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

