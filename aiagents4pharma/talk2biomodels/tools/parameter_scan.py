#!/usr/bin/env python3

"""
Tool for parameter scan.
"""

import logging
from dataclasses import dataclass
from typing import Type, Union, List, Annotated
import pandas as pd
import basico
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from .load_biomodel import ModelData, load_biomodel

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeData:
    """
    Dataclass for storing the time data.
    """
    duration: Union[int, float] = 100
    interval: Union[int, float] = 10

@dataclass
class SpeciesData:
    """
    Dataclass for storing the species data.
    """
    species_name: List[str] = Field(description="species name", default=[])
    species_concentration: List[Union[int, float]] = Field(
        description="initial species concentration",
        default=[])

@dataclass
class TimeSpeciesNameConcentration:
    """
    Dataclass for storing the time, species name, and concentration data.
    """
    time: Union[int, float] = Field(description="time point where the event occurs")
    species_name: str = Field(description="species name")
    species_concentration: Union[int, float] = Field(
        description="species concentration at the time point")

@dataclass
class ReocurringData:
    """
    Dataclass for species that reoccur. In other words, the concentration
    of the species resets to a certain value after a certain time interval.
    """
    data: List[TimeSpeciesNameConcentration] = Field(
        description="time, name, and concentration data of species that reoccur",
        default=[])

@dataclass
class ParameterScanData(BaseModel):
    """
    Dataclass for storing the parameter scan data.
    """
    species_names: List[str] = Field(description="species names to scan",
                              default=[])
    parameter_name: str = Field(description="Parameter name to scan",
                                 default_factory=None)
    parameter_values: List[Union[int, float]] = Field(
        description="Parameter values to scan",
        default_factory=None)

@dataclass
class ArgumentData:
    """
    Dataclass for storing the argument data.
    """
    time_data: TimeData = Field(description="time data", default=None)
    species_data: SpeciesData = Field(
        description="species name and initial concentration data",
        default=None)
    reocurring_data: ReocurringData = Field(
        description="""Concentration and time data of species that reoccur
            For example, a species whose concentration resets to a certain value
            after a certain time interval""")
    parameter_scan_data: ParameterScanData = Field(
        description="parameter scan data",
        default=None)
    scan_name: str = Field(
        description="""An AI assigned `_` separated name of
        the parameter scan experiment based on human query""")

def add_rec_events(model_object, reocurring_data):
    """
    Add reocurring events to the model.
    """
    for row in reocurring_data.data:
        tp, sn, sc = row.time, row.species_name, row.species_concentration
        basico.add_event(f'{sn}_{tp}',
                            f'Time > {tp}',
                            [[sn, str(sc)]],
                            model=model_object.copasi_model)

def make_list_dic_scanned_data(dic_param_scan, arg_data, sys_bio_model, tool_call_id):
    """
    Prepare the list dictionary of scanned data
    that will be passed to the state of the graph.

    Args:
        dic_param_scan: Dictionary of parameter scan results.
        arg_data: The argument data.
        sys_bio_model: The model data.
        tool_call_id: The tool call ID.

    Returns:
        list: List of dictionary of scanned data.
    """
    list_dic_scanned_data = []
    for species_name, df_param_scan in dic_param_scan.items():
        logger.log(logging.INFO, "Parameter scan results for %s with shape %s",
                    species_name,
                    df_param_scan.shape)
        # Prepare the list dictionary of scanned data
        # that will be passed to the state of the graph
        list_dic_scanned_data.append({
            'name': arg_data.scan_name+':'+species_name,
            'source': sys_bio_model.biomodel_id if sys_bio_model.biomodel_id else 'upload',
            'tool_call_id': tool_call_id,
            'data': df_param_scan.to_dict()
        })
    return list_dic_scanned_data
def run_parameter_scan(model_object, arg_data, dic_species_data, duration, interval) -> dict:
    """
    Run parameter scan on the model.

    Args:
        model_object: The model object.
        arg_data: The argument data.
        dic_species_data: Dictionary of species data.
        duration: Duration of the simulation.
        interval: Interval between time points in the simulation.

    Returns:
        dict: Dictionary of parameter scan results. Each key is a species name
        and each value is a DataFrame containing the results of the parameter scan.
    """
    # Extract all parameter names from the model and verify if the given parameter name is valid
    df_all_parameters = basico.model_info.get_parameters(model=model_object.copasi_model)
    all_parameters = df_all_parameters.index.tolist()
    if arg_data.parameter_scan_data.parameter_name not in all_parameters:
        logger.error(
            "Invalid parameter name: %s", arg_data.parameter_scan_data.parameter_name)
        raise ValueError(
            f"Invalid parameter name: {arg_data.parameter_scan_data.parameter_name}")
    # Extract all species name from the model and verify if the given species name is valid
    df_all_species = basico.model_info.get_species(model=model_object.copasi_model)
    all_species = df_all_species['display_name'].tolist()
    # Dictionary to store the parameter scan results
    dic_param_scan_results = {}
    for species_name in arg_data.parameter_scan_data.species_names:
        if species_name not in all_species:
            logger.error("Invalid species name: %s", species_name)
            raise ValueError(f"Invalid species name: {species_name}")
        # Update the fixed model species and parameters
        # These are the initial conditions of the model
        # set by the user
        model_object.update_parameters(dic_species_data)
        # Initialize empty DataFrame to store results
        # of the parameter scan
        df_param_scan = pd.DataFrame()
        for param_value in arg_data.parameter_scan_data.parameter_values:
            # Update the parameter value in the model
            model_object.update_parameters(
                {arg_data.parameter_scan_data.parameter_name: param_value})
            # Simulate the model
            model_object.simulate(duration=duration, interval=interval)
            # If the column name 'Time' is not present in the results DataFrame
            if 'Time' not in df_param_scan.columns:
                df_param_scan['Time'] = model_object.simulation_results['Time']
            # Add the simulation results to the results DataFrame
            col_name = f"{arg_data.parameter_scan_data.parameter_name}_{param_value}"
            df_param_scan[col_name] = model_object.simulation_results[species_name]

        logger.log(logging.INFO, "Parameter scan results with shape %s", df_param_scan.shape)
        # Add the results of the parameter scan to the dictionary
        dic_param_scan_results[species_name] = df_param_scan
    # return df_param_scan
    return dic_param_scan_results

class ParameterScanInput(BaseModel):
    """
    Input schema for the ParameterScan tool.
    """
    sys_bio_model: ModelData = Field(description="model data",
                                     default=None)
    arg_data: ArgumentData = Field(description=
                                   """time, species, and reocurring data
                                   as well as the parameter scan name and
                                   data""",
                                   default=None)
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ParameterScanTool(BaseTool):
    """
    Tool for parameter scan.
    """
    name: str = "parameter_scan"
    description: str = """A tool to perform parameter scan
        of a list of parameter values for a given species."""
    args_schema: Type[BaseModel] = ParameterScanInput

    def _run(self,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        sys_bio_model: ModelData = None,
        arg_data: ArgumentData = None
    ) -> Command:
        """
        Run the tool.

        Args:
            tool_call_id (str): The tool call ID. This is injected by the system.
            state (dict): The state of the tool.
            sys_bio_model (ModelData): The model data.
            arg_data (ArgumentData): The argument data.

        Returns:
            Command: The updated state of the tool.
        """
        logger.log(logging.INFO, "Calling parameter_scan tool %s, %s",
                   sys_bio_model, arg_data)
        sbml_file_path = state['sbml_file_path'][-1] if len(state['sbml_file_path']) > 0 else None
        model_object = load_biomodel(sys_bio_model,
                                  sbml_file_path=sbml_file_path)
        # Prepare the dictionary of species data
        # that will be passed to the simulate method
        # of the BasicoModel class
        duration = 100.0
        interval = 10
        dic_species_data = {}
        if arg_data:
            # Prepare the dictionary of species data
            if arg_data.species_data is not None:
                dic_species_data = dict(zip(arg_data.species_data.species_name,
                                            arg_data.species_data.species_concentration))
            # Add reocurring events (if any) to the model
            if arg_data.reocurring_data is not None:
                add_rec_events(model_object, arg_data.reocurring_data)
            # Set the duration and interval
            if arg_data.time_data is not None:
                duration = arg_data.time_data.duration
                interval = arg_data.time_data.interval

        # Run the parameter scan
        dic_param_scan = run_parameter_scan(model_object,
                                           arg_data,
                                           dic_species_data,
                                           duration,
                                           interval)

        logger.log(logging.INFO, "Parameter scan results ready")
        # Prepare the list dictionary of scanned data
        list_dic_scanned_data = make_list_dic_scanned_data(dic_param_scan,
                                                           arg_data,
                                                           sys_bio_model,
                                                           tool_call_id)
        # Prepare the dictionary of updated state for the model
        dic_updated_state_for_model = {}
        for key, value in {
            "model_id": [sys_bio_model.biomodel_id],
            "sbml_file_path": [sbml_file_path],
            "dic_scanned_data": list_dic_scanned_data,
            }.items():
            if value:
                dic_updated_state_for_model[key] = value
        # Return the updated state
        return Command(
                update=dic_updated_state_for_model|{
                # update the message history
                "messages": [
                    ToolMessage(
                        content=f"Parameter scan results of {arg_data.scan_name}",
                        tool_call_id=tool_call_id
                        )
                    ],
                }
            )