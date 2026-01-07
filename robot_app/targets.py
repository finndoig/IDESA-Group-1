"""
targets.py
Manages:
- Home position (world coordinates)
- List of clicked targets (world coordinates)
- "Click mode" so a click can mean "set home" or "add target"
"""

from dataclasses import dataclass, field
from enum import Enum, auto

# define click modes - target adding or home setting
class ClickMode(Enum):
    ADD_TARGET = auto()
    SET_HOME = auto()


@dataclass
class TargetManager:
    # Home position (world coords in metres)
    home_world: tuple[float, float] | None = None

    # Targets as a list of world points (metres)
    targets_world: list[tuple[float, float]] = field(default_factory=list)

    # What a click means right now
    click_mode: ClickMode = ClickMode.ADD_TARGET

    # If user clicks while homography is unavailable, store it here and resolve later
    pending_click_px: tuple[float, float] | None = None
    pending_click_mode: ClickMode | None = None

    # function to set click mode
    def set_click_mode(self, mode: ClickMode) -> None:
        self.click_mode = mode

    # function to set home world coordinates
    def set_home_world(self, xy: tuple[float, float]) -> None:
        self.home_world = xy

    # function to add target world coordinates
    def add_target_world(self, xy: tuple[float, float]) -> None:
        self.targets_world.append(xy)

    # function to clear all targets
    def clear_targets(self) -> None:
        self.targets_world.clear()

    # function to undo last target
    def undo_last_target(self) -> None:
        if self.targets_world:
            self.targets_world.pop()

    # function to get next target
    def next_target(self) -> tuple[float, float] | None:
        """Returns the current 'next' target without removing it."""
        return self.targets_world[0] if self.targets_world else None

    # function to pop next target to remove it from the list
    def pop_next_target(self) -> tuple[float, float] | None:
        """Removes and returns the next target."""
        return self.targets_world.pop(0) if self.targets_world else None

    # function to set pending click when homography is unavailable
    def set_pending_click(self, px_xy: tuple[float, float], mode: ClickMode) -> None:
        """Store a click until homography becomes available."""
        self.pending_click_px = px_xy
        self.pending_click_mode = mode

    # function to clear pending click
    def clear_pending_click(self) -> None:
        self.pending_click_px = None
        self.pending_click_mode = None
