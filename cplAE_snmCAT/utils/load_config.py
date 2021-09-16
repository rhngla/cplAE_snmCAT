def load_config(verbose=False):
    """Loads dictionary with path names and any other variables set through config.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """
    import toml
    from pathlib import Path
    package_dir = Path(__file__).parent.parent.parent.absolute()
    config_file = package_dir / "config.toml"
    if verbose:
        print(f'package dir: {package_dir}')
        print(f'config file: {config_file}')
    config = toml.load(config_file)
    config.update({'package_dir':package_dir,'config_file':config_file})
    for key in config:
        if Path(config[key]).exists(): 
            config[key] = Path(config[key])
    return config