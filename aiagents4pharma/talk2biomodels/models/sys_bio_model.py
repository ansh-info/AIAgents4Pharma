'''
An abstract base class for BioModels in the BioModels repository.
'''

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field, model_validator

class SysBioModel(ABC, BaseModel):
    """
    Abstract base class for BioModels in the BioModels repository.
    This class serves as a general structure for models, allowing
    different mathematical approaches to be implemented in subclasses.
    """
    model_id: Optional[int] = Field(None, description="BioModel ID of the model")
    sbml_file_path: Optional[str] = Field(None, description="Path to an SBML file")
    name: Optional[str] = Field(..., description="Name of the model")
    description: Optional[str] = Field("", description="Description of the model")

    @model_validator(mode="after")
    def check_model_id_or_sbml_file_path(self):
        """
        Validate that either model_id or sbml_file_path is provided.
        """
        if not self.model_id and not self.sbml_file_path:
            raise ValueError("Either model_id or sbml_file_path must be provided.")
        return self

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Union[str, int]]:
        """
        Abstract method to retrieve metadata of the model.
        This method should return a dictionary containing model metadata.

        Returns:
            dict: Dictionary with model metadata
        """

    @abstractmethod
    def simulate(self,
                 parameters: Dict[str, Union[float, int]],
                 duration: Union[int, float]) -> List[float]:
        """
        Abstract method to run a simulation of the model.
        This method should be implemented to simulate model 
        behavior based on the provided parameters.

        Args:
            parameters: Dictionary of parameter values.
            duration: Duration of the simulation.

        Returns:
            list: List of simulation results.
        """