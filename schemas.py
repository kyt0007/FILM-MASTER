"""Pydantic schemas shared across all film_generator_agent tools."""

from typing import List, Dict, Optional
from pydantic import BaseModel


class FilmConcept(BaseModel):
    title: str
    genre: str
    visual_style: str          # e.g. "anime", "photorealistic", "noir"
    mood: str
    setting: str
    act_1: str
    act_2: str
    act_3: str
    sound: str                 # recommended music / soundscape


class Dialogue(BaseModel):
    character: str             # character name, or "NARRATOR"
    line: str


class SceneScript(BaseModel):
    scene_number: int
    title: str
    plot: str
    visual_description: str    # full cinematic description
    dialogue: List[Dialogue]
    emotional_tone: str
    cinematography: str        # camera techniques


class Script(BaseModel):
    scenes: Dict[str, SceneScript]     # key = str(scene_number)
    characters: List["Character"]


class Character(BaseModel):
    name: str                  # underscores for multi-word, e.g. "Dark_Knight"
    description: str           # visual description: age, appearance, outfit


class CharacterInShot(BaseModel):
    name: str
    visual_description: str    # outfit/action/expression/position in this shot


class ShotDesign(BaseModel):
    scene_number: int
    involving_characters: Optional[List[CharacterInShot]] = []
    visual_description: str    # 30+ words
    coarse_action: str         # < 20 words, no names, describe actions only
    emotion: str
    shot_type: str             # e.g. "medium close-up, gradual"
    camera_movement: str       # e.g. "slow pan left", "dolly in", "static"
    dialogue: Optional[List[Dialogue]] = []


class ShotList(BaseModel):
    shots: Dict[str, ShotDesign]       # key = shot_id e.g. "1", "2"
