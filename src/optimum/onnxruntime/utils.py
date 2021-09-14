

def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)

