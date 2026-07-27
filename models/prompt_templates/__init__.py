"""Initialize the prompt templates package."""

from .general_template import GeneralInstructionTemplate
from .image_template import ImageInstructionTemplate
from .text_template import TextInstructionTemplate
from .uml_template import UMLInstructionTemplate

__all__ = [
    'TextInstructionTemplate',
    'ImageInstructionTemplate',
    'UMLInstructionTemplate',
    'GeneralInstructionTemplate',
]