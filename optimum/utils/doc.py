def generate_doc_basemodel(cls) -> str:
    """Class decorator for pydantic's BaseModel to generate the documentation."""
    doc = "\f\nAttributes:\n"
    for name, field in cls.__fields__.items():
        field_info = field.field_info
        doc += f"   {name}"  # argument name

        # argument type and whether optional
        type_display = str(field._type_display())
        optional = True if type_display.startswith("Optional") else False
        if optional:
            type_str = type_display[type_display.find("[") + 1 : -1]
            doc += f" (`{type_str}`, *optional*): "
        else:
            type_str = type_display
            doc += f" (`{type_str}`): "

        doc += f"{field_info.description}\n"  # argument description
    cls.__doc__ = (cls.__doc__ if cls.__doc__ is not None else "") + "\n\n" + "".join(doc)
    return cls
