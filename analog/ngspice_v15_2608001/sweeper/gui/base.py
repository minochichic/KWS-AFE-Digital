"""Controller delegation primitives used by the Tk application."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_PLOT_FONTS_CONFIGURED = False
# Ordered by preference; Malgun Gothic ships with Korean Windows.
_KOREAN_PLOT_FONTS = (
    "Malgun Gothic",
    "NanumGothic",
    "AppleGothic",
    "Gulim",
    "Dotum",
)


def configure_plot_fonts() -> None:
    """Make Matplotlib able to draw the Korean text used in plot labels.

    The default DejaVu Sans has no Hangul glyphs, so titles such as
    ``절대전압`` render as empty boxes.  The first available Korean-capable
    system font is prepended to the sans-serif list, keeping the previous
    entries as fallbacks.  Called lazily so importing the package stays cheap.
    """

    global _PLOT_FONTS_CONFIGURED
    if _PLOT_FONTS_CONFIGURED:
        return
    _PLOT_FONTS_CONFIGURED = True
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in _KOREAN_PLOT_FONTS:
        if candidate not in available:
            continue
        existing = [
            name
            for name in matplotlib.rcParams["font.sans-serif"]
            if name != candidate
        ]
        matplotlib.rcParams["font.sans-serif"] = [candidate, *existing]
        # Korean fonts usually lack U+2212, which would break negative ticks.
        matplotlib.rcParams["axes.unicode_minus"] = False
        return


class AppController:
    """Proxy controller state and cross-controller calls through the application."""

    __slots__ = ("app",)

    def __init__(self, app: object) -> None:
        """Bind this controller to one application instance."""

        object.__setattr__(self, "app", app)

    def __getattribute__(self, name: str) -> Any:
        """Honor an app-instance override before using a controller method."""

        if name in {"app", "__class__", "__slots__", "__dict__"}:
            return object.__getattribute__(self, name)
        app = object.__getattribute__(self, "app")
        if name in app.__dict__:
            return app.__dict__[name]
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        """Read shared widgets, state, or another controller method from the app."""

        return getattr(self.app, name)

    def __setattr__(self, name: str, value: object) -> None:
        """Write controller-created widgets and state onto the application."""

        if name == "app":
            object.__setattr__(self, name, value)
            return
        setattr(self.app, name, value)


class ControllerMethod:
    """Descriptor exposing a composed controller method on ``SweeperGUI``."""

    def __init__(
        self,
        controller_attribute: str,
        controller_type: type[AppController],
        method_name: str,
    ) -> None:
        """Store enough metadata to resolve or lazily create the controller."""

        self.controller_attribute = controller_attribute
        self.controller_type = controller_type
        self.method_name = method_name

    def __get__(
        self,
        instance: object | None,
        owner: type[object],
    ) -> Callable[..., Any]:
        """Return the implementation for class inspection or a bound controller call."""

        if instance is None:
            return getattr(self.controller_type, self.method_name)
        controller = instance.__dict__.get(self.controller_attribute)
        if controller is None:
            controller = self.controller_type(instance)
            instance.__dict__[self.controller_attribute] = controller
        return getattr(controller, self.method_name)
