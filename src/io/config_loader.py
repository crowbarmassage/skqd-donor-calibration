"""
Configuration loader for SKQD donor calibration.

Loads experiment configurations and validates against global contracts.
Prints contract values at runtime start as required by Step 0.2.
"""

import json
from pathlib import Path
from typing import Any


# Project root relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
METADATA_DIR = PROJECT_ROOT / "results" / "metadata"
GLOBAL_CONTRACT_PATH = METADATA_DIR / "global_contract.json"


class ContractViolationError(Exception):
    """Raised when a configuration violates global contracts."""
    pass


def load_global_contract() -> dict:
    """Load the frozen global contract specification."""
    if not GLOBAL_CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Global contract not found at {GLOBAL_CONTRACT_PATH}. "
            "Run Step 0.2 to create it."
        )
    with open(GLOBAL_CONTRACT_PATH, "r") as f:
        return json.load(f)


def load_config(config_name: str) -> dict:
    """
    Load an experiment configuration by name.

    Args:
        config_name: Name of config file (with or without .json extension)

    Returns:
        Configuration dictionary
    """
    if not config_name.endswith(".json"):
        config_name = f"{config_name}.json"

    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def validate_config_against_contract(config: dict, contract: dict) -> None:
    """
    Validate that a configuration respects global contracts.

    Args:
        config: Experiment configuration dictionary
        contract: Global contract dictionary

    Raises:
        ContractViolationError: If any contract is violated
    """
    numerical = contract["numerical_contracts"]

    # Check residual tolerance
    config_tolerance = config.get("krylov", {}).get("residual_tolerance")
    contract_tolerance = numerical["residual_tolerance"]
    if config_tolerance != contract_tolerance:
        raise ContractViolationError(
            f"Residual tolerance mismatch: config has {config_tolerance}, "
            f"contract requires {contract_tolerance}"
        )

    # Check basis type
    config_basis = config.get("active_space", {}).get("basis")
    contract_basis = numerical["basis_type"]
    if config_basis != contract_basis:
        raise ContractViolationError(
            f"Basis type mismatch: config has '{config_basis}', "
            f"contract requires '{contract_basis}'"
        )

    # Check active space qubit counts
    active_spaces = contract["active_spaces"]
    config_space_type = config.get("active_space", {}).get("type")
    config_qubits = config.get("active_space", {}).get("num_qubits")

    if config_space_type in active_spaces:
        expected_qubits = active_spaces[config_space_type]["num_qubits"]
        if config_qubits != expected_qubits:
            raise ContractViolationError(
                f"Qubit count mismatch for '{config_space_type}' space: "
                f"config has {config_qubits}, contract requires {expected_qubits}"
            )


def print_contract_summary(contract: dict) -> None:
    """Print global contract values at runtime start (Step 0.2 requirement)."""
    print("=" * 60)
    print("SKQD DONOR CALIBRATION - GLOBAL CONTRACT")
    print("=" * 60)

    numerical = contract["numerical_contracts"]

    print(f"\nResidual Tolerance: {numerical['residual_tolerance']}")
    print(f"  -> {numerical['residual_tolerance_description']}")

    print(f"\nBasis Type: {numerical['basis_type']}")
    print(f"  -> {numerical['basis_description']}")
    print(f"  -> Eigenbasis allowed: {numerical['eigenbasis_allowed']}")

    print(f"\nLog Residuals Every Iteration: {numerical['log_residuals_every_iteration']}")

    print("\nPhysical Parameters:")
    for material, params in contract["physical_parameters"]["valley_orbit_splitting"].items():
        print(f"  {material}: {params['value']} {params['unit']}")

    print("\nActive Spaces:")
    for space_type, info in contract["active_spaces"].items():
        print(f"  {space_type}: {info['num_qubits']} qubits ({info['description']})")

    print("\nExpected Orderings:")
    for metric, ordering in contract["expected_orderings"].items():
        print(f"  {metric}: {ordering}")

    print("=" * 60)


def load_and_validate_config(config_name: str, verbose: bool = True) -> dict:
    """
    Load configuration and validate against global contracts.

    This is the main entry point for loading configurations.

    Args:
        config_name: Name of config file
        verbose: If True, print contract summary

    Returns:
        Validated configuration dictionary
    """
    # Load global contract
    contract = load_global_contract()

    # Print contract summary if verbose
    if verbose:
        print_contract_summary(contract)

    # Load and validate config
    config = load_config(config_name)
    validate_config_against_contract(config, contract)

    if verbose:
        print(f"\nLoaded configuration: {config_name}")
        print(f"  System: {config.get('system')}")
        print(f"  Active space: {config.get('active_space', {}).get('type')}")
        print(f"  Qubits: {config.get('active_space', {}).get('num_qubits')}")
        print("  Contract validation: PASSED")
        print()

    return config


def list_available_configs() -> list[str]:
    """List all available configuration files."""
    return [f.stem for f in CONFIGS_DIR.glob("*.json")]


if __name__ == "__main__":
    # Test the loader
    import sys

    print("Available configurations:", list_available_configs())
    print()

    # Load each config and validate
    for config_name in list_available_configs():
        try:
            config = load_and_validate_config(config_name, verbose=True)
        except (ContractViolationError, FileNotFoundError) as e:
            print(f"ERROR loading {config_name}: {e}")
            sys.exit(1)
