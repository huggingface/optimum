from dataclasses import fields


def generate_doc_dataclass(cls) -> str:
    """Class decorator for generate the documentation for dataclass."""
    doc = "\f\nAttributes:\n"
    for attribute in fields(cls):
        doc += f"   {attribute.name}"  # attribute name

        # whether optional
        attribute_type = str(attribute.type)
        if attribute_type.startswith("typing.Optional"):
            optional = True
            type_display = attribute_type[attribute_type.find("[") + 1 : -1]
            type_display = type_display.split(".")[-1]
        else:
            optional = False

            if attribute_type.startswith("typing"):
                type_display = attribute_type.split(".")[-1]
            else:
                type_display = attribute.type.__name__

        if optional:
            doc += f" (`{type_display}`, *optional*): "
        else:
            doc += f" (`{type_display}`): "

        doc += f"{attribute.metadata['description']}\n"  # argument description
    cls.__doc__ = (cls.__doc__ if cls.__doc__ is not None else "") + "\n\n" + "".join(doc)
    return cls


def add_dynamic_docstring(
    *docstr,
    text,
    dynamic_elements,
):
    def docstring_decorator(fn):
        func_doc = (fn.__doc__ or "") + "".join(docstr)
        fn.__doc__ = func_doc + text.format(**dynamic_elements)
        return fn

    return docstring_decorator
