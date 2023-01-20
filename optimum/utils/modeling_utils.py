def recurse_setattr(module, name, value):
    r"""
    A wrapper function to recursively set attributes to a module.
    """
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)
