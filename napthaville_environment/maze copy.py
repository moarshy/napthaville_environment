# maze.py

import json
import numpy
import datetime
import pickle
import time
import math
import sqlite3
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from napthaville_environment.utils import read_file_to_list
from napthaville_environment.schemas import (
    MazeConfig, 
    MazeState, 
    TileDetails, 
    TileLocation, 
    PixelCoordinate, 
    TileLevel,
    TilePath,
    VisionRadius,
    NearbyTiles
)
from naptha_sdk.storage.schemas import (
    StorageType,
    StorageLocation,
    StorageObject,
    CreateStorageRequest,
    ReadStorageRequest,
    UpdateStorageRequest,
    DeleteStorageRequest,
    DatabaseReadOptions
)

file_dir = Path(__file__).parent

class Maze:
    def __init__(self, config: Dict):
        self.config = config
        self.maze_name = config["world_name"]    
        self.maze_width = config["maze_width"]
        self.maze_height = config["maze_height"]
        self.sq_tile_size = config["sq_tile_size"]
        self.special_constraint = config["special_constraint"]
        self.env_matrix_path = config["env_matrix_path"]
        self.db_path = file_dir / "data" / "maze.db"
        
        # Initialize empty placeholders
        self.collision_maze = []
        self.tiles = []
        self.address_tiles = dict()

        # Check if database exists and is initialized
        if not self.db_path.exists() or not self.is_db_initialized():
            self.env_matrix = self.load_env_matrix()
            # Load maze and save to DB
            self.load_env_matrix()
        else:
            # Just load from existing DB
            self.collision_maze = self.load_collision_maze()
            self.tiles = self.load_tiles()
            self.address_tiles = self.load_address_tiles()

    def is_db_initialized(self) -> bool:
        """Check if database exists and has required tables with data"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if tables exist and have data for this maze
            c.execute('''SELECT COUNT(*) FROM tiles WHERE maze_name = ?''', 
                    (self.maze_name,))
            tiles_count = c.fetchone()[0]
            
            c.execute('''SELECT COUNT(*) FROM collision_maze WHERE maze_name = ?''', 
                    (self.maze_name,))
            collision_count = c.fetchone()[0]
            
            c.execute('''SELECT COUNT(*) FROM address_tiles WHERE maze_name = ?''', 
                    (self.maze_name,))
            address_count = c.fetchone()[0]
            
            conn.close()
            
            # Check if we have the expected number of records
            return (tiles_count == self.maze_width * self.maze_height and 
                    collision_count == self.maze_width * self.maze_height and 
                    address_count > 0)  # At least some address mappings exist
            
        except sqlite3.Error:
            return False

    def load_env_matrix(self):
        env_matrix_path = file_dir / self.env_matrix_path
        blocks_folder = env_matrix_path / "special_blocks"
        
        # Load blocks
        _wb = blocks_folder / "world_blocks.csv"
        wb_rows = read_file_to_list(_wb, header=False)
        wb = wb_rows[0][-1]
   
        _sb = blocks_folder / "sector_blocks.csv" 
        sb_rows = read_file_to_list(_sb, header=False)
        sb_dict = dict()
        for i in sb_rows: sb_dict[i[0]] = i[-1]
    
        _ab = blocks_folder / "arena_blocks.csv"
        ab_rows = read_file_to_list(_ab, header=False)
        ab_dict = dict()
        for i in ab_rows: ab_dict[i[0]] = i[-1]
    
        _gob = blocks_folder / "game_object_blocks.csv"
        gob_rows = read_file_to_list(_gob, header=False)
        gob_dict = dict()
        for i in gob_rows: gob_dict[i[0]] = i[-1]
    
        _slb = blocks_folder / "spawning_location_blocks.csv"
        slb_rows = read_file_to_list(_slb, header=False)
        slb_dict = dict()
        for i in slb_rows: slb_dict[i[0]] = i[-1]

        # Load mazes
        maze_folder = env_matrix_path / "maze"

        _cm = maze_folder / "collision_maze.csv"
        collision_maze_raw = read_file_to_list(_cm, header=False)[0]
        _sm = maze_folder / "sector_maze.csv"
        sector_maze_raw = read_file_to_list(_sm, header=False)[0]
        _am = maze_folder / "arena_maze.csv"
        arena_maze_raw = read_file_to_list(_am, header=False)[0]
        _gom = maze_folder / "game_object_maze.csv"
        game_object_maze_raw = read_file_to_list(_gom, header=False)[0]
        _slm = maze_folder / "spawning_location_maze.csv"
        spawning_location_maze_raw = read_file_to_list(_slm, header=False)[0]

        # Convert 1D to 2D
        self.collision_maze = []
        sector_maze = []
        arena_maze = []
        game_object_maze = []
        spawning_location_maze = []
        for i in range(0, len(collision_maze_raw), self.maze_width): 
            tw = self.maze_width
            self.collision_maze += [collision_maze_raw[i:i+tw]]
            sector_maze += [sector_maze_raw[i:i+tw]]
            arena_maze += [arena_maze_raw[i:i+tw]]
            game_object_maze += [game_object_maze_raw[i:i+tw]]
            spawning_location_maze += [spawning_location_maze_raw[i:i+tw]]

        # Initialize tiles
        self.tiles = []
        for i in range(self.maze_height): 
            row = []
            for j in range(self.maze_width):
                tile_details = dict()
                tile_details["world"] = wb
                
                tile_details["sector"] = ""
                if sector_maze[i][j] in sb_dict: 
                    tile_details["sector"] = sb_dict[sector_maze[i][j]]
                
                tile_details["arena"] = ""
                if arena_maze[i][j] in ab_dict: 
                    tile_details["arena"] = ab_dict[arena_maze[i][j]]
                
                tile_details["game_object"] = ""
                if game_object_maze[i][j] in gob_dict: 
                    tile_details["game_object"] = gob_dict[game_object_maze[i][j]]
                
                tile_details["spawning_location"] = ""
                if spawning_location_maze[i][j] in slb_dict: 
                    tile_details["spawning_location"] = slb_dict[spawning_location_maze[i][j]]
                
                tile_details["collision"] = False
                if self.collision_maze[i][j] != "0": 
                    tile_details["collision"] = True

                tile_details["events"] = set()
                
                row += [tile_details]
            self.tiles += [row]

        # Initialize game object events
        for i in range(self.maze_height):
            for j in range(self.maze_width): 
                if self.tiles[i][j]["game_object"]:
                    object_name = ":".join([self.tiles[i][j]["world"], 
                                        self.tiles[i][j]["sector"], 
                                        self.tiles[i][j]["arena"], 
                                        self.tiles[i][j]["game_object"]])
                    go_event = (object_name, None, None, None)
                    self.tiles[i][j]["events"].add(go_event)

        # Initialize address tiles
        self.address_tiles = dict()
        for i in range(self.maze_height):
            for j in range(self.maze_width): 
                addresses = []
                if self.tiles[i][j]["sector"]: 
                    add = f'{self.tiles[i][j]["world"]}:'
                    add += f'{self.tiles[i][j]["sector"]}'
                    addresses += [add]
                if self.tiles[i][j]["arena"]: 
                    add = f'{self.tiles[i][j]["world"]}:'
                    add += f'{self.tiles[i][j]["sector"]}:'
                    add += f'{self.tiles[i][j]["arena"]}'
                    addresses += [add]
                if self.tiles[i][j]["game_object"]: 
                    add = f'{self.tiles[i][j]["world"]}:'
                    add += f'{self.tiles[i][j]["sector"]}:'
                    add += f'{self.tiles[i][j]["arena"]}:'
                    add += f'{self.tiles[i][j]["game_object"]}'
                    addresses += [add]
                if self.tiles[i][j]["spawning_location"]: 
                    add = f'<spawn_loc>{self.tiles[i][j]["spawning_location"]}'
                    addresses += [add]

                for add in addresses: 
                    if add in self.address_tiles: 
                        self.address_tiles[add].add((j, i))
                    else: 
                        self.address_tiles[add] = set([(j, i)])

        # Now save everything to SQLite with proper types
        self.save_to_db()

    def turn_coordinate_to_tile(self, px_coordinate: PixelCoordinate) -> TileLocation:
        """
        Turns a pixel coordinate to a tile coordinate.

        INPUT
            px_coordinate: PixelCoordinate(x=1600, y=384)
        OUTPUT
            TileLocation(x=50, y=12)
        """
        x = math.ceil(px_coordinate.x/self.sq_tile_size)
        y = math.ceil(px_coordinate.y/self.sq_tile_size)
        return TileLocation(x=x, y=y)

    def access_tile(self, tile: TileLocation) -> Optional[TileDetails]:
        """
        Returns the tiles details from the database for the designated location.

        INPUT
            tile: TileLocation(x=58, y=9)
        OUTPUT
            TileDetails object containing tile information or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''SELECT world, sector, arena, game_object, spawning_location, 
                    collision, events FROM tiles 
                    WHERE maze_name = ? AND x = ? AND y = ?''', 
                (self.maze_name, tile.x, tile.y))
        
        row = c.fetchone()
        if row:
            tile_details = TileDetails(
                world=row[0],
                sector=row[1],
                arena=row[2],
                game_object=row[3],
                spawning_location=row[4],
                collision=bool(row[5]),
                events=set(tuple(e) for e in json.loads(row[6]))
            )
        else:
            tile_details = None

        conn.close()
        return tile_details

    def get_tile_path(self, tile: TileLocation, level: TileLevel) -> TilePath:
        """
        Get the tile string address given its coordinate and level.

        INPUT:
            tile: TileLocation(x=58, y=9)
            level: TileLevel.ARENA
        OUTPUT:
            TilePath containing address string
        EXAMPLE:
            get_tile_path(TileLocation(x=58, y=9), TileLevel.ARENA)
            Returns: TilePath(path="double studio:double studio:bedroom 2")
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''SELECT world, sector, arena, game_object 
                    FROM tiles 
                    WHERE maze_name = ? AND x = ? AND y = ?''',
                (self.maze_name, tile.x, tile.y))
        
        row = c.fetchone()
        if not row:
            conn.close()
            return None

        world, sector, arena, game_object = row
        
        path = world
        if level == TileLevel.WORLD:
            conn.close()
            return TilePath(path=path)
        
        path += f":{sector}"
        if level == TileLevel.SECTOR:
            conn.close()
            return TilePath(path=path)
        
        path += f":{arena}"
        if level == TileLevel.ARENA:
            conn.close()
            return TilePath(path=path)
        
        path += f":{game_object}"
        conn.close()
        return TilePath(path=path)

    def get_nearby_tiles(self, tile: TileLocation, vision_r: VisionRadius) -> NearbyTiles:
        """
        Get tiles within a square boundary around the specified tile.

        INPUT:
            tile: TileLocation(x=10, y=10)
            vision_r: VisionRadius(radius=2)
        OUTPUT:
            NearbyTiles containing list of TileLocation within range
            
        Visual example for radius 2:
        x x x x x 
        x x x x x
        x x P x x 
        x x x x x
        x x x x x
        """
        left_end = 0
        if tile.x - vision_r.radius > left_end:
            left_end = tile.x - vision_r.radius

        right_end = self.maze_width - 1
        if tile.x + vision_r.radius + 1 < right_end:
            right_end = tile.x + vision_r.radius + 1

        bottom_end = self.maze_height - 1 
        if tile.y + vision_r.radius + 1 < bottom_end:
            bottom_end = tile.y + vision_r.radius + 1

        top_end = 0
        if tile.y - vision_r.radius > top_end:
            top_end = tile.y - vision_r.radius

        nearby_tiles = []
        for i in range(left_end, right_end):
            for j in range(top_end, bottom_end):
                nearby_tiles.append(TileLocation(x=i, y=j))

        return NearbyTiles(tiles=nearby_tiles)

    def add_event_from_tile(self, curr_event: Tuple[str, Optional[str], Optional[str]], tile: TileLocation) -> None:
        """
        Add an event triple to a tile.  

        INPUT: 
            curr_event: Tuple of (str, Optional[str], Optional[str])
                e.g., ('double studio:double studio:bedroom 2:bed', None, None)
            tile: TileLocation(x=58, y=9)
        """
        # 1. Get current tile data
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT events FROM tiles 
                    WHERE maze_name = ? AND x = ? AND y = ?''',
                (self.maze_name, tile.x, tile.y))
        
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        
        # 2. Update events set
        current_events = set(tuple(e) for e in json.loads(row[0]))
        current_events.add(curr_event)
        
        # 3. Save updated events back to database
        c.execute('''UPDATE tiles 
                    SET events = ?
                    WHERE maze_name = ? AND x = ? AND y = ?''',
                (json.dumps(list(current_events)), self.maze_name, tile.x, tile.y))
        
        conn.commit()
        conn.close()

    def remove_event_from_tile(self, curr_event: Tuple[str, Optional[str], Optional[str]], tile: TileLocation) -> None:
        """
        Remove an event triple from a tile.  

        INPUT: 
            curr_event: Tuple of (str, Optional[str], Optional[str])
                e.g., ('double studio:double studio:bedroom 2:bed', None, None)
            tile: TileLocation(x=58, y=9)
        """
        # 1. Get current tile data
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT events FROM tiles 
                        WHERE maze_name = ? AND x = ? AND y = ?''',
                    (self.maze_name, tile.x, tile.y))
        
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        
        # 2. Update events set by removing matching event
        current_events = set(tuple(e) for e in json.loads(row[0]))
        current_events.discard(curr_event)  # Discard safely removes if exists
        
        # 3. Save updated events back to database
        c.execute('''UPDATE tiles 
                        SET events = ?
                        WHERE maze_name = ? AND x = ? AND y = ?''',
                    (json.dumps(list(current_events)), self.maze_name, tile.x, tile.y))
        
        conn.commit()
        conn.close()

    def turn_event_from_tile_idle(self, curr_event: Tuple[str, Optional[str], Optional[str]], tile: TileLocation) -> None:
        """
        Convert an event to idle state (all parameters set to None) for a tile.

        INPUT:
            curr_event: Tuple of (str, Optional[str], Optional[str])
                e.g., ('double studio:double studio:bedroom 2:bed', 'param1', 'param2')
            tile: TileLocation(x=58, y=9)
        """
        # 1. Get current tile data
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT events FROM tiles 
                        WHERE maze_name = ? AND x = ? AND y = ?''',
                    (self.maze_name, tile.x, tile.y))
        
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        
        # 2. Update events set - remove old event and add idle version
        current_events = set(tuple(e) for e in json.loads(row[0]))
        if curr_event in current_events:
            current_events.remove(curr_event)
            new_event = (curr_event[0], None, None, None)
            current_events.add(new_event)
        
        # 3. Save updated events back to database
        c.execute('''UPDATE tiles 
                        SET events = ?
                        WHERE maze_name = ? AND x = ? AND y = ?''',
                    (json.dumps(list(current_events)), self.maze_name, tile.x, tile.y))
        
        conn.commit()
        conn.close()

    def remove_subject_events_from_tile(self, subject: str, tile: TileLocation) -> None:
        """
        Remove all events with matching subject from a tile.

        INPUT:
            subject: str, e.g. "Isabella Rodriguez" 
            tile: TileLocation(x=58, y=9)
        """
        # 1. Get current tile data
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT events FROM tiles 
                        WHERE maze_name = ? AND x = ? AND y = ?''',
                    (self.maze_name, tile.x, tile.y))
        
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        
        # 2. Remove events matching subject
        current_events = set(tuple(e) for e in json.loads(row[0]))
        updated_events = {event for event in current_events if event[0] != subject}
        
        # 3. Save updated events back to database
        c.execute('''UPDATE tiles 
                        SET events = ?
                        WHERE maze_name = ? AND x = ? AND y = ?''',
                    (json.dumps(list(updated_events)), self.maze_name, tile.x, tile.y))
        
        conn.commit()
        conn.close()

    def save_to_db(self):
        """Save everything to SQLite with proper types"""
        conn = sqlite3.connect(file_dir / "data" / "maze.db")
        c = conn.cursor()

        # Create tables with proper column types
        c.execute('''CREATE TABLE IF NOT EXISTS collision_maze (
            maze_name TEXT NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (maze_name, x, y)
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS tiles (
            maze_name TEXT NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            world TEXT NOT NULL,
            sector TEXT NOT NULL,
            arena TEXT NOT NULL,
            game_object TEXT NOT NULL,
            spawning_location TEXT NOT NULL,
            collision INTEGER NOT NULL,  -- SQLite doesn't have boolean, use INTEGER
            events TEXT NOT NULL,        -- JSON string of events
            PRIMARY KEY (maze_name, x, y)
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS address_tiles (
            maze_name TEXT NOT NULL,
            address TEXT NOT NULL,
            coordinates TEXT NOT NULL,   -- JSON string of coordinate tuples
            PRIMARY KEY (maze_name, address)
        )''')

        # Save collision_maze
        for i in range(len(self.collision_maze)):
            for j in range(len(self.collision_maze[i])):
                c.execute('INSERT OR REPLACE INTO collision_maze VALUES (?, ?, ?, ?)',
                        (self.maze_name, j, i, self.collision_maze[i][j]))

        # Save tiles
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[i])):
                tile = self.tiles[i][j]
                events_json = json.dumps(list(tile["events"]))  # Convert set to list for JSON
                c.execute('INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (self.maze_name, j, i, tile["world"], tile["sector"],
                        tile["arena"], tile["game_object"], tile["spawning_location"],
                        1 if tile["collision"] else 0, events_json))

        # Save address_tiles
        for address, coordinates in self.address_tiles.items():
            coordinates_json = json.dumps(list(coordinates))  # Convert set to list for JSON
            c.execute('INSERT OR REPLACE INTO address_tiles VALUES (?, ?, ?)',
                    (self.maze_name, address, coordinates_json))

        conn.commit()
        conn.close()

    def load_collision_maze(self) -> List[List[str]]:
        """Load collision maze from SQLite"""
        conn = sqlite3.connect(file_dir / "data" / "maze.db")
        c = conn.cursor()
        
        # Initialize 2D list
        collision_maze = [[None] * self.maze_width for _ in range(self.maze_height)]
        
        # Load data
        c.execute('SELECT x, y, value FROM collision_maze WHERE maze_name = ?', 
                (self.maze_name,))
        for x, y, value in c.fetchall():
            collision_maze[y][x] = value
        
        conn.close()
        return collision_maze

    def load_tiles(self) -> List[List[Dict]]:
        """Load tiles from SQLite"""
        conn = sqlite3.connect(file_dir / "data" / "maze.db")
        c = conn.cursor()
        
        # Initialize 2D list
        tiles = [[None] * self.maze_width for _ in range(self.maze_height)]
        
        # Load data
        c.execute('''SELECT x, y, world, sector, arena, game_object, 
                    spawning_location, collision, events 
                    FROM tiles WHERE maze_name = ?''', (self.maze_name,))
        
        for x, y, world, sector, arena, game_object, spawning_location, collision, events in c.fetchall():
            tiles[y][x] = {
                "world": world,
                "sector": sector,
                "arena": arena,
                "game_object": game_object,
                "spawning_location": spawning_location,
                "collision": bool(collision),
                "events": set(tuple(e) for e in json.loads(events))
            }
        
        conn.close()
        return tiles

    def load_address_tiles(self) -> Dict[str, Set[Tuple[int, int]]]:
        """Load address tiles from SQLite"""
        conn = sqlite3.connect(file_dir / "data" / "maze.db")
        c = conn.cursor()
        
        address_tiles = {}
        
        # Load data
        c.execute('SELECT address, coordinates FROM address_tiles WHERE maze_name = ?', 
                (self.maze_name,))
        
        for address, coordinates in c.fetchall():
            address_tiles[address] = set(tuple(coord) for coord in json.loads(coordinates))
        
        conn.close()
        return address_tiles
    
if __name__ == "__main__":
    config = json.load(open(file_dir / "configs" / "deployment.json"))
    maze = Maze(config[0]["config"])