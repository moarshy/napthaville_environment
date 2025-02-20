from pydantic import BaseModel, Field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum

class TileDetails(BaseModel):
    """Schema for individual tile details"""
    world: str
    sector: str = ""
    arena: str = ""
    game_object: str = ""
    spawning_location: str = ""
    collision: bool = False
    events: Set[Tuple[str, Optional[str], Optional[str], Optional[str]]] = Field(default_factory=set)

class MazeConfig(BaseModel):
    """Schema for maze configuration"""
    world_name: str
    maze_width: int
    maze_height: int
    sq_tile_size: int
    special_constraint: str
    env_matrix_path: str

class MazeState(BaseModel):
    """Schema for complete maze state"""
    maze_name: str
    collision_maze: List[List[str]]
    tiles: List[List[TileDetails]]
    address_tiles: Dict[str, Set[Tuple[int, int]]]

class TileLocation(BaseModel):
    """Schema for tile coordinates"""
    x: int
    y: int

class PixelCoordinate(BaseModel):
    """Schema for pixel coordinates"""
    x: float
    y: float

class AddressTile(BaseModel):
    """Schema for address to tile mapping"""
    maze_name: str
    address: str
    coordinates: Set[Tuple[int, int]]

class TileLevel(str, Enum):
    """Enum for possible tile levels"""
    WORLD = "world"
    SECTOR = "sector"
    ARENA = "arena"
    GAME_OBJECT = "game_object"

class TilePath(BaseModel):
    """Schema for tile path result"""
    path: str

class VisionRadius(BaseModel):
   radius: int = Field(ge=0)  # Must be non-negative

class NearbyTiles(BaseModel):
   tiles: List[TileLocation]

class EventTuple(BaseModel):
    """Helper schema for event tuples"""
    name: str
    param1: Optional[str] = None
    param2: Optional[str] = None
    param3: Optional[str] = None

    def to_tuple(self) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        return (self.name, self.param1, self.param2, self.param3)

    @classmethod
    def from_tuple(cls, t: Tuple) -> 'EventTuple':
        return cls(name=t[0], param1=t[1], param2=t[2], param3=t[3])

class AvailableFunctions(str, Enum):
    """Enum for available functions"""
    INIT = "init"
    TURN_COORDINATE_TO_TILE = "turn_coordinate_to_tile"
    ACCESS_TILE = "access_tile"
    GET_TILE_PATH = "get_tile_path"
    GET_NEARBY_TILES = "get_nearby_tiles"
    ADD_EVENT_FROM_TILE = "add_event_from_tile"
    TURN_EVENT_FROM_TILE_IDLE = "turn_event_from_tile_idle"
    REMOVE_EVENT_FROM_TILE = "remove_event_from_tile"
    REMOVE_SUBJECT_EVENTS_FROM_TILE = "remove_subject_events_from_tile"

class InputSchema(BaseModel):
    function_name: AvailableFunctions
    function_input_data: Optional[Dict[str, Any]] = None


"""
# Example input schema for turn_coordinate_to_tile
run_inputs = {
    "function_name": "turn_coordinate_to_tile",
    "function_input_data": {
        "px_coordinate": {
            "x": 1600,
            "y": 384
        }
    }
}

# Example input schema for access_tile
run_inputs = {
    "function_name": "access_tile",
    "function_input_data": {
        "tile": {
            "x": 58,
            "y": 9
        }
    }
}

# Example input schema for get_tile_path
run_inputs = {
    "function_name": "get_tile_path",
    "function_input_data": {
        "tile": {
            "x": 58,
            "y": 9
        },
        "level": "arena"
    }
}

# Example input schema for get_nearby_tiles
run_inputs = {
    "function_name": "get_nearby_tiles",
    "function_input_data": {
        "tile": {
            "x": 58,
            "y": 9
        },
        "vision_r": 2
    }
}

# Example input schema for add_event_from_tile
run_inputs = {
    "function_name": "add_event_from_tile",
    "function_input_data": {
        "tile": {
            "x": 58,
            "y": 9
        },
        "curr_event": ("test_event", None, None, None)
    }
}

# Example input schema for turn_event_from_tile_idle
run_inputs = {
    "function_name": "turn_event_from_tile_idle",
    "function_input_data": {
        "tile": {
            "x": 58,
            "y": 9
        },
        "curr_event": ("test_event", None, None, None)
    }
}

# Example input schema for remove_event_from_tile
run_inputs = {
    "function_name": "remove_event_from_tile",
    "function_input_data": {
        "tile": {
            "x": 58,
            "y": 9
        },
        "curr_event": ("test_event", None, None, None)
    }
}

# Example input schema for remove_subject_events_from_tile
run_inputs = {
    "function_name": "remove_subject_events_from_tile",
    "function_input_data": {
        "tile": {
            "x": 58,
            "y": 9
        },
        "subject": "test_subject"
    }
}
"""