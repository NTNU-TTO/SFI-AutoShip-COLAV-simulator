"""
    config_parsing.py

    Summary:
        Contains functionality for reading and validating configuration files from schemas.

    Author: Trym Tengesdal
"""
from pathlib import Path
from typing import Any, List, Optional

import colav_simulator.common.file_utils as futils
import dacite
from cerberus import Validator


def extract(data_class: Any, config_file: Path, config_schema: Path, converter: Optional[dict] = None, **kwargs) -> Any:
    """Extracts configuration settings from a configuration file, and converts them to a dataclass.
    Validation is performed using the input schema.

    Args:
        data_class (Any): Dataclass to convert the settings to.
        config_file (Path): Path to the configuration file.
        config_schema (Path): Path to the schema used for config validation.
        converter (Optional[dict]): Dictionary specifying data types to convert to in the settings to the dataclass. Defaults to None.

    Returns:
        Any: Configuration settings as a dataclass.
    """

    schema = futils.read_yaml_into_dict(config_schema)

    settings = parse(config_file, schema)
    settings = override(settings, schema, **kwargs)

    settings = convert_settings_dict_to_dataclass(data_class, settings, converter)

    return settings


def convert_settings_dict_to_dataclass(data_class, config_dict: dict, converter: Optional[dict] = None) -> Any:
    """Converts a settings dictionary to a dataclass.

    Args:
        dataclass (Any): Data class to convert to.

    Returns:
        Any: The dataclass.
    """
    if converter is not None:
        settings = dacite.from_dict(data_class=data_class, data=config_dict, config=dacite.Config(type_hooks=converter))
    else:
        settings = dacite.from_dict(data_class=data_class, data=config_dict)
    return settings


def validate(settings: dict, schema: dict) -> None:
    """Validates the settings against the schema.

    Args:
        settings (dict): Configuration settings to validate.

    Raises:
        ValueError: On empty settings/schema or invalid settings.
    """
    if not settings:
        raise ValueError("Empty settings!")

    if not schema:
        raise ValueError("Empty schema!")

    validator = Validator(schema)

    if not validator.validate(settings):
        raise ValueError(f"Cerberus validation Error: {validator.errors}")


def extract_valid_sections(schema: dict) -> List[str]:
    """Extracts the valid main sections from the schema.

    Args:
        schema (dict): Configuration schema.

    Raises:
        ValueError: On empty schema.

    Returns:
        List[str]: List of valid main sections as strings.
    """
    if schema is None:
        raise ValueError("No configuration schema provided!")

    sections = []
    for section in schema.keys():
        sections.append(section)

    return sections


def parse(file_name: Path, schema: dict) -> dict:
    """Parses a configuration file into a dictionary, and validates the settings.

    Args:
        file_name (Path): Path to the configuration file.
        schema (dict): Configuration schema to validate the settings against.

    Returns:
        dict: Configuration settings.
    """
    settings = futils.read_yaml_into_dict(file_name)

    validate(settings, schema)

    return settings


def override(settings: dict, schema: dict, **kwargs) -> dict:
    """Overrides settings with keyword arguments, and validates the new values.

    Args:
        settings (dict): Configuration settings to override.
        schema (dict): Configuration schema to validate the new settings against.

    Raises:
        ValueError: On empty keyword arguments or invalid settings.

    Returns:
        dict: The new settings.
    """
    if not kwargs:
        return settings

    new_settings = settings
    for key, value in kwargs.items():
        new_settings[key] = value

    validate(settings, schema)

    return new_settings
