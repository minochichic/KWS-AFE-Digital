"""Matplotlib line-following cursors and AC metric overlays."""

from __future__ import annotations

import bisect
import math
from typing import Callable, Iterable, Mapping, Sequence

from ..models import AcMetrics


class InteractivePlotCursor:
    """Button-armed line-following inspector for one Matplotlib axes.

    Ordinary mouse movement remains inert.  After ``arm()`` a temporary
    vertical guide, point marker, and value label follow the closest plotted
    sample.  A left click converts that preview into one persistent cursor;
    repeated arming therefore leaves multiple independent markers.

    The temporary artists are animated and use Matplotlib blitting when the
    backend supports it.  This avoids redrawing hundreds of thousands of
    transient samples for every mouse-motion event.
    """

    def __init__(
        self,
        canvas: object,
        axes: object,
        lines: Sequence[object],
        *,
        x_name: str = "x",
        x_unit: str = "",
        y_name: str = "y",
        y_unit: str = "",
        interaction_blocker: Callable[[], bool] | None = None,
    ) -> None:
        """Attach reusable preview and fixed-cursor artists to one axes."""

        self.canvas = canvas
        self.axes = axes
        self.lines = tuple(lines)
        self.x_name = x_name
        self.x_unit = x_unit
        self.y_name = y_name
        self.y_unit = y_unit
        self.interaction_blocker = interaction_blocker
        self.armed = False
        self._disposed = False
        self._background = None
        self.artists: list[tuple[object, object, object]] = []
        self.preview_vertical = axes.axvline(
            0.0,
            color="#ef6c00",
            linewidth=1.0,
            linestyle="--",
            alpha=0.95,
            visible=False,
            animated=True,
            zorder=20,
        )
        self.preview_marker = axes.plot(
            [],
            [],
            marker="o",
            markersize=5,
            markerfacecolor="#fff3e0",
            markeredgecolor="#ef6c00",
            linestyle="none",
            visible=False,
            animated=True,
            zorder=21,
        )[0]
        self.preview_annotation = axes.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fff3e0", "ec": "#ef6c00"},
            arrowprops={"arrowstyle": "->", "color": "#ef6c00"},
            fontsize=9,
            fontfamily="DejaVu Sans",
            visible=False,
            animated=True,
            zorder=22,
        )
        self._preview_artists = (
            self.preview_vertical,
            self.preview_marker,
            self.preview_annotation,
        )
        self._connection_ids = (
            canvas.mpl_connect("button_press_event", self._on_click),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("figure_leave_event", self._on_leave),
            canvas.mpl_connect("draw_event", self._on_draw),
        )

    def arm(self) -> None:
        """Enable line-following preview until the next left click."""

        if self._disposed:
            return
        self.armed = True

    def cancel(self) -> None:
        """Cancel preview mode without changing existing fixed cursors."""

        self.armed = False
        self._hide_preview()

    def _toolbar_busy(self) -> bool:
        """Return whether another interaction mode currently owns the mouse."""

        toolbar = getattr(self.canvas, "toolbar", None)
        blocked = (
            self.interaction_blocker is not None
            and bool(self.interaction_blocker())
        )
        return bool(getattr(toolbar, "mode", "")) or blocked

    def _nearest(self, event: object) -> tuple[object, float, float] | None:
        """Find the plotted sample nearest to the event in display coordinates."""

        event_pixel_x = getattr(event, "x", None)
        event_pixel_y = getattr(event, "y", None)
        if event_pixel_x is None or event_pixel_y is None:
            return None
        try:
            event_pixel_x = float(event_pixel_x)
            event_pixel_y = float(event_pixel_y)
        except (TypeError, ValueError, OverflowError):
            return None
        best: tuple[float, object, float, float] | None = None
        for line in self.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            count = min(len(x_data), len(y_data))
            if count == 0:
                continue
            indices: Iterable[int] = range(count)
            event_x = getattr(event, "xdata", None)
            if (
                count > 32
                and event_x is not None
                and math.isfinite(float(event_x))
            ):
                try:
                    first_x = float(x_data[0])
                    last_x = float(x_data[count - 1])
                    if first_x <= last_x:
                        position = bisect.bisect_left(
                            x_data, float(event_x), 0, count
                        )
                        indices = range(
                            max(0, position - 4),
                            min(count, position + 5),
                        )
                except (TypeError, ValueError, OverflowError):
                    indices = range(count)
            for index in indices:
                x_value = x_data[index]
                y_value = y_data[index]
                try:
                    x_float = float(x_value)
                    y_float = float(y_value)
                    if not math.isfinite(x_float) or not math.isfinite(y_float):
                        continue
                    pixel_x, pixel_y = self.axes.transData.transform(
                        (x_float, y_float)
                    )
                except (TypeError, ValueError, OverflowError):
                    continue
                distance = (
                    (pixel_x - event_pixel_x) ** 2
                    + (pixel_y - event_pixel_y) ** 2
                )
                if best is None or distance < best[0]:
                    best = (distance, line, x_float, y_float)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _annotation_text(
        self,
        line: object,
        x_value: float,
        y_value: float,
        state: str,
    ) -> str:
        """Format one preview or fixed-cursor label with configured units."""

        label = line.get_label()
        if not label or label.startswith("_"):
            label = "data"
        x_suffix = f" {self.x_unit}" if self.x_unit else ""
        y_suffix = f" {self.y_unit}" if self.y_unit else ""
        return (
            f"{label}\n"
            f"{self.x_name} = {x_value:.7g}{x_suffix}\n"
            f"{self.y_name} = {y_value:.7g}{y_suffix}\n"
            f"[{state}]"
        )

    def _preview_is_visible(self) -> bool:
        """Return whether the temporary guide is currently on the axes."""

        return bool(self.preview_vertical.get_visible())

    def _redraw_preview(self) -> None:
        """Blit only animated preview artists, falling back to an idle redraw."""

        if self._disposed:
            return
        if getattr(self.canvas, "supports_blit", False) and self._background is not None:
            try:
                self.canvas.restore_region(self._background)
                for artist in self._preview_artists:
                    if artist.get_visible():
                        self.axes.draw_artist(artist)
                self.canvas.blit(self.axes.bbox)
                return
            except (AttributeError, RuntimeError, ValueError):
                self._background = None
        self.canvas.draw_idle()

    def _show_preview(
        self,
        line: object,
        x_value: float,
        y_value: float,
    ) -> None:
        """Move the temporary vertical guide and value marker to one sample."""

        self.preview_vertical.set_xdata((x_value, x_value))
        self.preview_marker.set_data((x_value,), (y_value,))
        self.preview_annotation.xy = (x_value, y_value)
        self.preview_annotation.set_text(
            self._annotation_text(line, x_value, y_value, "CLICK TO LOCK")
        )
        for artist in self._preview_artists:
            artist.set_visible(True)
        self._redraw_preview()

    def _hide_preview(self, *, redraw: bool = True) -> None:
        """Hide the temporary guide and optionally repaint its previous area."""

        was_visible = self._preview_is_visible()
        for artist in self._preview_artists:
            artist.set_visible(False)
        if was_visible and redraw:
            self._redraw_preview()

    def _on_draw(self, _event: object) -> None:
        """Refresh the clean axes background used for fast preview blitting."""

        if self._disposed or not getattr(self.canvas, "supports_blit", False):
            return
        try:
            self._background = self.canvas.copy_from_bbox(self.axes.bbox)
            if self._preview_is_visible():
                for artist in self._preview_artists:
                    self.axes.draw_artist(artist)
                self.canvas.blit(self.axes.bbox)
        except (AttributeError, RuntimeError, ValueError):
            self._background = None

    def _on_motion(self, event: object) -> None:
        """Follow the nearest line only while the user has armed the cursor."""

        if not self.armed or self._disposed:
            return
        if self._toolbar_busy() or getattr(event, "inaxes", None) is not self.axes:
            self._hide_preview()
            return
        nearest = self._nearest(event)
        if nearest is None:
            self._hide_preview()
            return
        self._show_preview(*nearest)

    def _on_leave(self, _event: object) -> None:
        """Remove the temporary guide when the pointer leaves the figure."""

        if self.armed:
            self._hide_preview()

    def _show(self, line: object, x_value: float, y_value: float) -> None:
        """Create one fixed vertical guide, point marker, and annotation."""

        vertical = self.axes.axvline(
            x_value,
            color="#d32f2f",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
        )
        marker = self.axes.plot(
            (x_value,),
            (y_value,),
            marker="o",
            markersize=5,
            markerfacecolor="#ffebee",
            markeredgecolor="#d32f2f",
            linestyle="none",
        )[0]
        annotation = self.axes.annotate(
            "",
            xy=(x_value, y_value),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fff9c4", "ec": "#555"},
            arrowprops={"arrowstyle": "->", "color": "#555"},
            fontsize=9,
            fontfamily="DejaVu Sans",
        )
        annotation.set_text(
            self._annotation_text(line, x_value, y_value, "LOCKED")
        )
        self.artists.append((vertical, marker, annotation))
        self.canvas.draw_idle()

    def remove_last(self) -> None:
        """Remove the most recently added fixed cursor from this axes."""

        if not self.artists:
            return
        for artist in self.artists.pop():
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self.canvas.draw_idle()

    def clear(self) -> None:
        """Cancel placement and delete every fixed cursor on this axes."""

        self.cancel()
        while self.artists:
            for artist in self.artists.pop():
                try:
                    artist.remove()
                except (AttributeError, ValueError):
                    pass
        self.canvas.draw_idle()

    def _on_click(self, event: object) -> None:
        """Lock the line-following preview after one explicitly armed click."""

        if (
            not self.armed
            or self._disposed
            or getattr(event, "button", None) != 1
        ):
            return
        self.armed = False
        self._hide_preview(redraw=False)
        if (
            self._toolbar_busy()
            or getattr(event, "inaxes", None) is not self.axes
        ):
            self._redraw_preview()
            return
        nearest = self._nearest(event)
        if nearest is not None:
            self._show(*nearest)
        else:
            self._redraw_preview()

    def dispose(self) -> None:
        """Disconnect callbacks and remove artists before their widget is rebuilt."""

        if self._disposed:
            return
        self._disposed = True
        self.armed = False
        for connection_id in self._connection_ids:
            try:
                self.canvas.mpl_disconnect(connection_id)
            except (AttributeError, RuntimeError):
                pass
        for artist_group in (*self.artists, self._preview_artists):
            for artist in artist_group:
                try:
                    artist.remove()
                except (AttributeError, ValueError):
                    pass
        self.artists.clear()
        self._background = None

class MetricSelectionOverlay:
    """Persistent peak cursors controlled by the 16-channel metrics list."""

    def __init__(
        self,
        canvas: object,
        axes: object,
        entries: Mapping[int, tuple[object, AcMetrics, float]],
        *,
        gain_unit: str = "dB",
    ) -> None:
        """Keep the plotted line, metric, and displayed peak Y for each channel."""

        if gain_unit not in {"dB", "Linear"}:
            raise ValueError(f"지원하지 않는 Gain 단위: {gain_unit!r}")
        self.canvas = canvas
        self.axes = axes
        self.entries = dict(entries)
        self.gain_unit = gain_unit
        self.artists: list[object] = []

    def clear(self, redraw: bool = True) -> None:
        """Remove every list-controlled cursor and optional redraw the canvas."""

        for artist in self.artists:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self.artists.clear()
        if redraw:
            self.canvas.draw_idle()

    def show_channels(self, channel_numbers: Sequence[int]) -> None:
        """Draw one color-matched f0/gain/Q cursor for each selected list row."""

        self.clear(redraw=False)
        visible = [
            (channel, self.entries[channel])
            for channel in channel_numbers
            if channel in self.entries
        ]
        for index, (channel, (line, metric, marker_y)) in enumerate(visible):
            color = line.get_color()
            vertical = self.axes.axvline(
                metric.center_hz,
                color=color,
                linewidth=1.2,
                linestyle=":",
                alpha=0.9,
            )
            point = self.axes.plot(
                [metric.center_hz],
                [marker_y],
                marker="o",
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=2,
                color=color,
                label="_nolegend_",
                zorder=6,
            )[0]
            q_text = "—" if metric.q is None else f"{metric.q:.5g}"
            gain_value = metric.peak_db if self.gain_unit == "dB" else marker_y
            annotation = self.axes.annotate(
                (
                    f"ch{channel:02d}\n"
                    f"f0={metric.center_hz:.7g} Hz\n"
                    f"gain={gain_value:.5g} {self.gain_unit}\n"
                    f"Q={q_text}"
                ),
                xy=(metric.center_hz, marker_y),
                xytext=(10, 12 + 42 * (index % 4)),
                textcoords="offset points",
                color=color,
                fontsize=8,
                fontfamily="DejaVu Sans",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "fc": "white",
                    "ec": color,
                    "alpha": 0.92,
                },
                arrowprops={"arrowstyle": "->", "color": color},
                zorder=7,
            )
            self.artists.extend((vertical, point, annotation))
        self.canvas.draw_idle()
