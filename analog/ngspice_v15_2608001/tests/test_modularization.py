"""Regression tests for the v14 package and controller boundaries."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import symtable
import unittest
from pathlib import Path

import ngspice_channel_sweeper as compatibility_module
from ngspice_channel_sweeper import SweeperGUI
from sweeper.gui.ac_tab import AcTabController
from sweeper.gui.common import CommonController
from sweeper.gui.dc_tab import DcTabController
from sweeper.gui.detector_tab import DetectorTabController
from sweeper.gui.pwl_tab import PwlTabController
from sweeper.gui.state import AppState
from sweeper.gui.transient_tab import TransientTabController


class ModularizationTests(unittest.TestCase):
    """Keep the thin entry point, composed controllers, and shared state explicit."""

    def test_compatibility_entry_point_stays_thin(self) -> None:
        """The legacy import/CLI file must not grow back into a monolithic GUI."""

        entry = Path(compatibility_module.__file__).resolve()
        self.assertLess(len(entry.read_text(encoding="utf-8").splitlines()), 150)
        self.assertTrue((entry.parent / "sweeper" / "gui" / "app.py").is_file())
        self.assertTrue(
            (entry.parent / "sweeper" / "gui" / "transient_tab.py").is_file()
        )

    def test_tab_methods_are_implemented_by_composed_controllers(self) -> None:
        """Public GUI methods should resolve to their owning controller modules."""

        expected = {
            "_build_dc_tab": "dc_tab.py",
            "_build_ac_tab": "ac_tab.py",
            "_build_detector_tab": "detector_tab.py",
            "_build_transient_tab": "transient_tab.py",
            "_build_converter_tab": "pwl_tab.py",
            "_poll_events": "common.py",
        }
        for method_name, filename in expected.items():
            method = getattr(SweeperGUI, method_name)
            self.assertEqual(Path(inspect.getsourcefile(method) or "").name, filename)

    def test_controller_method_names_do_not_overlap(self) -> None:
        """A workflow method must have one unambiguous controller owner."""

        controller_types = (
            CommonController,
            DcTabController,
            AcTabController,
            DetectorTabController,
            TransientTabController,
            PwlTabController,
        )
        owners: dict[str, type[object]] = {}
        for controller_type in controller_types:
            methods = {
                name
                for name, value in controller_type.__dict__.items()
                if callable(value) and name.startswith("_")
            }
            for name in methods:
                self.assertNotIn(name, owners)
                owners[name] = controller_type

    def test_state_descriptors_support_headless_instances(self) -> None:
        """Shared fields should live in AppState even when Tk was not initialized."""

        gui = SweeperGUI.__new__(SweeperGUI)
        gui.component_revision = 7
        gui.channels = []
        self.assertIsInstance(gui.state, AppState)
        self.assertEqual(gui.state.component_revision, 7)
        self.assertIs(gui.channels, gui.state.channels)

    def test_controller_honors_application_method_override(self) -> None:
        """Tests and extensions may replace one instance method without subclassing."""

        gui = SweeperGUI.__new__(SweeperGUI)
        marker = object()
        replacement = lambda _result: marker
        gui._render_transient_result = replacement
        controller = TransientTabController(gui)
        self.assertIs(controller._render_transient_result, replacement)

    def test_component_revision_is_advanced_by_all_update_paths(self) -> None:
        """CSV reload, manual commit, and margin retune must invalidate shared state."""

        reload_source = inspect.getsource(CommonController._reload_channels)
        commit_source = inspect.getsource(DcTabController._commit_component_editor)
        retune_source = inspect.getsource(DcTabController._prepare_margin_retune)
        self.assertIn("component_revision += 1", reload_source)
        self.assertIn("component_revision += 1", commit_source)
        self.assertIn("component_revision += 1", retune_source)

    def test_v13_public_symbols_remain_importable(self) -> None:
        """Existing scripts may keep importing the established public names."""

        expected = (
            "Channel",
            "SweepSettings",
            "TransientSettings",
            "Simulator",
            "TransientSimulator",
            "SweeperGUI",
            "load_channels",
            "make_analysis_netlist",
            "read_ascii_rawfile",
            "calculate_ac_metrics",
        )
        for name in expected:
            self.assertTrue(hasattr(compatibility_module, name), name)

    def test_production_modules_have_no_accidental_unused_imports(self) -> None:
        """Only package facades may retain imports solely for public re-export."""

        project_root = Path(__file__).parents[1]
        failures: list[str] = []
        for path in sorted((project_root / "sweeper").rglob("*.py")):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: list[tuple[str, int]] = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported.extend(
                        (alias.asname or alias.name.split(".")[0], node.lineno)
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                    imported.extend(
                        (alias.asname or alias.name, node.lineno)
                        for alias in node.names
                        if alias.name != "*"
                    )
            loaded = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
            }
            failures.extend(
                f"{path.relative_to(project_root)}:{line}:{name}"
                for name, line in imported
                if name not in loaded
            )
        self.assertEqual(failures, [])

    def test_all_package_global_references_resolve(self) -> None:
        """Static global references must exist after importing each package module."""

        project_root = Path(__file__).parents[1]
        failures: list[str] = []
        for path in sorted((project_root / "sweeper").rglob("*.py")):
            relative = path.relative_to(project_root).with_suffix("")
            module_name = ".".join(relative.parts)
            if module_name.endswith(".__init__"):
                module_name = module_name.rsplit(".", 1)[0]
            module = importlib.import_module(module_name)
            root_scope = symtable.symtable(
                path.read_text(encoding="utf-8"),
                str(path),
                "exec",
            )
            referenced: set[str] = set()
            pending = [root_scope]
            while pending:
                scope = pending.pop()
                referenced.update(
                    symbol.get_name()
                    for symbol in scope.get_symbols()
                    if symbol.is_referenced() and symbol.is_global()
                )
                pending.extend(scope.get_children())
            failures.extend(
                f"{path.relative_to(project_root)}:{name}"
                for name in sorted(referenced)
                if name not in module.__dict__
                and not hasattr(builtins, name)
            )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
