"""Initialize the prompt templates package."""

from .text_template import TextInstructionTemplate
from .image_template import ImageInstructionTemplate
from .uml_template import UMLInstructionTemplate
from .general_template import GeneralInstructionTemplate

__all__ = [
    'TextInstructionTemplate',
    'ImageInstructionTemplate',
    'UMLInstructionTemplate',
    'GeneralInstructionTemplate',
]