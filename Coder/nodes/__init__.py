# oder/nodes/__init__.py
"""
Gói quản lý các Graph Nodes tuần tự dành cho LangGraph Production Coder.
Sử dụng Facade Pattern để duy trì tương thích ngược với main.py.
"""
from .triage import detect_and_triage_node, triage_node
from .debugger import isolated_debugger_node
from .executor import executor_node
from .doubt import doubt_reviewer_node, doubt_gate_node
from .replanner import replanner_node, replanner_interrupt_node
from .compression import context_compressor_node, fluxmem_distillation_node
from .synthesis import synthesis_node
from .commit import commit_node
from .human import human_interaction_gate_node