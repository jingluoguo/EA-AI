from __future__ import annotations

from dataclasses import dataclass

from ..structure import Entity, Frame, State


@dataclass(frozen=True)
class WorkingMemory:
    focus_entities: tuple[Entity, ...] = ()
    recent_frames: tuple[Frame, ...] = ()
    current_states: tuple[State, ...] = ()

    def with_focus(self, *entities: Entity) -> "WorkingMemory":
        return WorkingMemory(
            focus_entities=(*self.focus_entities, *entities),
            recent_frames=self.recent_frames,
            current_states=self.current_states,
        )

    def with_frame(self, frame: Frame) -> "WorkingMemory":
        return WorkingMemory(
            focus_entities=self.focus_entities,
            recent_frames=(*self.recent_frames, frame),
            current_states=self.current_states,
        )

    def with_states(self, *states: State) -> "WorkingMemory":
        return WorkingMemory(
            focus_entities=self.focus_entities,
            recent_frames=self.recent_frames,
            current_states=states,
        )
