import yaml
from rich.table import Table
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.pretty import Pretty
import psutil

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def create_config_tree(config_dict, path="Configuration"):
    tree = Tree(f"[bold blue]{path}" if path is not None else "")
    for key, value in config_dict.items():
        if isinstance(value, dict):
            subtree = create_config_tree(value, key)
            tree.add(subtree)
        else:
            tree.add(f"[green]{key}: [yellow]{value}")
    return tree

def display_config(config, title="[bold magenta]Experiment Configuration", border_style="bright_magenta", width=70):
    console = Console()
    config_tree = create_config_tree(config)
    panel = Panel(config_tree, title=f"[bold magenta]{title}", border_style=border_style, width=width)
    console.print(panel)

def initialize_component(component_info, module, additional_args=None):
    """General function to initialize a component based on its configuration."""
    component_type = component_info.get('type')
    if component_type is None:
        raise ValueError("Component type is not specified in the configuration.")

    if hasattr(module, component_type):
        component_class = getattr(module, component_type)
        args = component_info.get('args', {})
        if additional_args:
            args.update(additional_args)
        return component_class(**args)
    else:
        raise ValueError(f"Component type '{component_type}' not found in the specified module.")

def check_memory(threshold=0.8):
    """檢查memory使用情況 如果超過閾值 返回True"""
    memory = psutil.virtual_memory()
    return memory.used / memory.total > threshold