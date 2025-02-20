import json
import math
import logging
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
    ListStorageRequest,
    DatabaseReadOptions
)
from naptha_sdk.storage.storage_client import StorageClient

file_dir = Path(__file__).parent
logger = logging.getLogger(__name__)

class Maze:
    @classmethod
    async def create(cls, config: Dict, storage_client: StorageClient) -> "Maze":
        """Create a new Maze instance asynchronously"""
        maze = cls.__new__(cls)
        await maze.__ainit__(config, storage_client)
        return maze
    
    async def __ainit__(self, config: Dict, storage_client: StorageClient):
        """Async initialization method"""
        self.config = config
        self.storage_client = storage_client
        
        # Basic maze config
        self.maze_name = config["world_name"]    
        self.maze_width = config["maze_width"]
        self.maze_height = config["maze_height"]
        self.sq_tile_size = config["sq_tile_size"]
        self.special_constraint = config["special_constraint"]
        self.env_matrix_path = config["env_matrix_path"]
        
        # Initialize empty placeholders
        self.collision_maze = []
        self.tiles = []
        self.address_tiles = dict()

        # Check if it has been initialized
        is_init = await self.is_initialized()
        logger.info(f"Maze {self.maze_name} is initialized: {is_init}")
        if not is_init:
            await self.load_env_matrix()
        else:
            # Load from existing DB
            await self.load_data_from_db()

    def __init__(self):
        """
        Prevent direct instantiation, use create() instead
        """
        raise RuntimeError("Please use Maze.create() to instantiate")

    async def is_initialized(self) -> bool:
        """
        Check if database table exists and has required data types
        """
        try:
            check_request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                options=DatabaseReadOptions(
                    columns=["type"],
                    conditions=[]
                ).model_dump()
            )

            try:
                check_result = await self.storage_client.execute(check_request)
                return True
            except Exception as e:
                logger.error(f"Table check failed: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Error checking database initialization: {str(e)}")
            return False
        
    async def load_data_from_db(self):
        """Load data from database"""
        try:
            self.collision_maze = await self.load_collision_maze()
            self.tiles = await self.load_tiles()
            self.address_tiles = await self.load_address_tiles()
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            raise

    async def load_collision_maze(self) -> List[List[str]]:
        """Load collision maze from database"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                options={"conditions": [{"type": "collision_maze"}]}
            )
            result = await self.storage_client.execute(request)
            
            # Parse the JSONB data
            data = json.loads(result.data[0]["data"])
            return data["collision_matrix"]

        except Exception as e:
            logger.error(f"Error loading collision maze: {str(e)}")
            raise

    async def load_tiles(self) -> List[List[Dict]]:
        """Load tiles from database"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                options={"conditions": [{"type": "tiles"}]}
            )
            result = await self.storage_client.execute(request)
            
            # Parse the JSONB data
            data = json.loads(result.data[0]["data"])
            tiles_matrix = data["tiles_matrix"]

            # Convert event lists back to sets
            for i in range(len(tiles_matrix)):
                for j in range(len(tiles_matrix[i])):
                    if tiles_matrix[i][j]["events"]:
                        tiles_matrix[i][j]["events"] = set(tuple(e) for e in tiles_matrix[i][j]["events"])
            
            return tiles_matrix

        except Exception as e:
            logger.error(f"Error loading tiles: {str(e)}")
            raise

    async def load_address_tiles(self) -> Dict[str, Set[Tuple[int, int]]]:
        """Load address tiles from database"""
        try:
            request = ReadStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                options={"conditions": [{"type": "address_tiles"}]}
            )
            result = await self.storage_client.execute(request)
            
            # Parse the JSONB data
            data = json.loads(result.data[0]["data"])
            address_mappings = data["address_mappings"]

            # Convert lists back to sets of tuples
            return {addr: set(tuple(coord) for coord in coords) for addr, coords in address_mappings.items()}

        except Exception as e:
            logger.error(f"Error loading address tiles: {str(e)}")
            raise

    async def save_to_db(self):
        """Save maze data to database using storage client"""
        try:
            storage_schema = self.config["storage_config"]["storage_schema"]

            # Create napthaville_environment table
            table_name = "napthaville_environment"
            schema_def = storage_schema["napthaville_environment"]

            create_table_request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=table_name,
                data={"schema": schema_def}
            )

            try:
                await self.storage_client.execute(create_table_request)
                logger.info(f"Created table {table_name}")
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {str(e)}")

            # Save tiles
            full_tiles_data = {
                "maze_name": self.maze_name,
                "tiles_matrix": [[{
                    "world": tile["world"],
                    "sector": tile["sector"],
                    "arena": tile["arena"],
                    "game_object": tile["game_object"],
                    "spawning_location": tile["spawning_location"],
                    "collision": tile["collision"],
                    "events": []  # Start with empty list instead of set
                } for tile in row] for row in self.tiles]
            }

            tiles_create = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                data={"data": {
                    "type": "tiles",
                    "data": json.dumps(full_tiles_data)
                }}
            )

            try:
                await self.storage_client.execute(tiles_create)
                logger.info("Full tiles data inserted successfully")
            except Exception as e:
                logger.error(f"Error inserting tiles data: {str(e)}")
            
            # Save address tiles
            # Convert sets to lists for JSON serialization
            address_data = {
                "maze_name": self.maze_name,
                "address_mappings": {
                    addr: list(coords) for addr, coords in self.address_tiles.items()
                }
            }

            address_create = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                data={"data": {
                    "type": "address_tiles",
                    "data": json.dumps(address_data)
                }}
            )

            try:
                await self.storage_client.execute(address_create)
                logger.info("Address tiles data inserted successfully")
            except Exception as e:
                logger.error(f"Error inserting address tiles data: {str(e)}")

            # Collision maze data
            collision_data = {
                "maze_name": self.maze_name,
                "collision_matrix": self.collision_maze
            }

            collision_create = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                data={"data": {
                    "type": "collision_maze",
                    "data": json.dumps(collision_data)
                }}
            )

            try:
                await self.storage_client.execute(collision_create)
                logger.info("Collision maze data inserted successfully")
            except Exception as e:
                logger.error(f"Error inserting collision maze data: {str(e)}")
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            raise

    async def load_env_matrix(self):
        """
        Load environment matrix from files and save to database
        Loads block definitions and maze layouts, then initializes tiles and address mappings
        """
        try:
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

            # Now save everything to database
            await self.save_to_db()
            
        except Exception as e:
            logger.error(f"Error loading environment matrix: {str(e)}")
            raise

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

    async def access_tile(self, tile: TileLocation) -> Optional[TileDetails]:
        """
        Returns the tiles details from the database for the designated location.

        INPUT
            tile: TileLocation(x=58, y=9)
        OUTPUT
            TileDetails object containing tile information or None if not found
        """
        try:
            tile_details = self.tiles[tile.y][tile.x]
            if tile_details:
                return TileDetails(
                    world=tile_details["world"],
                    sector=tile_details["sector"],
                    arena=tile_details["arena"],
                    game_object=tile_details["game_object"],
                    spawning_location=tile_details["spawning_location"],
                    collision=tile_details["collision"],
                    events=tile_details["events"]
                )
            return None
        except Exception as e:
            logger.error(f"Error accessing tile at ({tile.x}, {tile.y}): {str(e)}")
            raise

    async def get_tile_path(self, tile: TileLocation, level: TileLevel) -> TilePath:
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
        try:
            tile_details = await self.access_tile(tile)
            if not tile_details:
                return None

            path = tile_details.world
            if level == TileLevel.WORLD:
                return TilePath(path=path)
            
            path += f":{tile_details.sector}"
            if level == TileLevel.SECTOR:
                return TilePath(path=path)
            
            path += f":{tile_details.arena}"
            if level == TileLevel.ARENA:
                return TilePath(path=path)
            
            path += f":{tile_details.game_object}"
            return TilePath(path=path)

        except Exception as e:
            logger.error(f"Error getting tile path at ({tile.x}, {tile.y}): {str(e)}")
            raise

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
        try:
            left_end = max(0, tile.x - vision_r.radius)
            right_end = min(self.maze_width - 1, tile.x + vision_r.radius + 1)
            top_end = max(0, tile.y - vision_r.radius)
            bottom_end = min(self.maze_height - 1, tile.y + vision_r.radius + 1)

            nearby_tiles = []
            for y in range(top_end, bottom_end):
                for x in range(left_end, right_end):
                    nearby_tiles.append(TileLocation(x=x, y=y))

            return NearbyTiles(tiles=nearby_tiles)

        except Exception as e:
            logger.error(f"Error getting nearby tiles for ({tile.x}, {tile.y}): {str(e)}")
            raise

    async def add_event_from_tile(self, curr_event: Tuple[str, Optional[str], Optional[str], Optional[str]], tile: TileLocation) -> None:
        """
        Add an event triple to a tile.  

        INPUT: 
            curr_event: Tuple of (str, Optional[str], Optional[str])
                e.g., ('double studio:double studio:bedroom 2:bed', None, None)
            tile: TileLocation(x=58, y=9)
        """
        try:
            events = self.tiles[tile.y][tile.x]["events"]
            events = set(tuple(e) for e in events)
            events.add(curr_event)
            self.tiles[tile.y][tile.x]["events"] = list(events)

            tiles_matrix = {
                "tiles_matrix": [[{
                    "world": tile["world"],
                    "sector": tile["sector"],
                    "arena": tile["arena"],
                    "game_object": tile["game_object"],
                    "spawning_location": tile["spawning_location"],
                    "collision": tile["collision"],
                    "events": list(events)
                } for tile in row] for row in self.tiles]
            }

            # Update tile details
            update_request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                data={
                    "data": {
                        "data": json.dumps(tiles_matrix)
                    }
                },
                options={
                    "condition": {"type": "tiles"}
                }
            )

            await self.storage_client.execute(update_request)

            return {
                "success": True,
                "message": f"Event added to tile ({tile.x}, {tile.y})"
            }

        except Exception as e:
            logger.error(f"Error adding event to tile ({tile.x}, {tile.y}): {str(e)}")
            raise

    async def remove_event_from_tile(self, curr_event: Tuple[str, Optional[str], Optional[str], Optional[str]], tile: TileLocation) -> None:
        """
        Remove an event triple from a tile.  

        INPUT: 
            curr_event: Tuple of (str, Optional[str], Optional[str])
                e.g., ('double studio:double studio:bedroom 2:bed', None, None)
            tile: TileLocation(x=58, y=9)
        """
        try:
            events = self.tiles[tile.y][tile.x]["events"]
            events = set(tuple(e) for e in events)
            events.discard(curr_event)

            # Update tile details
            self.tiles[tile.y][tile.x]["events"] = list(events)

            tiles_matrix = {
                "tiles_matrix": [[{
                    "world": tile["world"],
                    "sector": tile["sector"],
                    "arena": tile["arena"],
                    "game_object": tile["game_object"],
                    "spawning_location": tile["spawning_location"],
                    "collision": tile["collision"],
                    "events": list(events)
                } for tile in row] for row in self.tiles]
            }

            update_request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                data={
                    "data": {
                        "data": json.dumps(tiles_matrix)
                    }
                },
                options={
                    "condition": {"type": "tiles"}
                }
            )
            await self.storage_client.execute(update_request)

        except Exception as e:
            logger.error(f"Error removing event from tile ({tile.x}, {tile.y}): {str(e)}")
            raise

    async def turn_event_from_tile_idle(self, curr_event: Tuple[str, Optional[str], Optional[str], Optional[str]], tile: TileLocation) -> None:
        """
        Convert an event to idle state (all parameters set to None) for a tile.

        INPUT:
            curr_event: Tuple of (str, Optional[str], Optional[str], Optional[str])
                e.g., ('double studio:double studio:bedroom 2:bed', 'param1', 'param2')
            tile: TileLocation(x=58, y=9)
        """
        try:
            events = self.tiles[tile.y][tile.x]["events"]
            events = set(tuple(e) for e in events)
            if curr_event in events:
                events.remove(curr_event)
                new_event = (curr_event[0], None, None, None)
                events.add(new_event)
            
            # Update tile details
            self.tiles[tile.y][tile.x]["events"] = list(events)

            tiles_matrix = {
                "tiles_matrix": [[{
                    "world": tile["world"],
                    "sector": tile["sector"],
                    "arena": tile["arena"],
                    "game_object": tile["game_object"],
                    "spawning_location": tile["spawning_location"],
                    "collision": tile["collision"],
                    "events": list(events)
                } for tile in row] for row in self.tiles]
            }
            
            update_request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                data={
                    "data": {
                        "data": json.dumps(tiles_matrix)
                    }
                },
                options={
                    "condition": {"type": "tiles"}
                }
            )
            await self.storage_client.execute(update_request)

        except Exception as e:
            logger.error(f"Error turning event to idle state for tile ({tile.x}, {tile.y}): {str(e)}")
            raise

    async def remove_subject_events_from_tile(self, subject: str, tile: TileLocation) -> None:
        """
        Remove all events with matching subject from a tile.

        INPUT:
            subject: str, e.g. "Isabella Rodriguez" 
            tile: TileLocation(x=58, y=9)
        """
        try:
            events = self.tiles[tile.y][tile.x]["events"]
            events = set(tuple(e) for e in events)
            updated_events = {event for event in events if event[0] != subject}
            
            # Update tile details
            self.tiles[tile.y][tile.x]["events"] = list(updated_events)

            tiles_matrix = {
                "tiles_matrix": [[{
                    "world": tile["world"],
                    "sector": tile["sector"],
                    "arena": tile["arena"],
                    "game_object": tile["game_object"],
                    "spawning_location": tile["spawning_location"],
                    "collision": tile["collision"],
                    "events": list(updated_events)
                } for tile in row] for row in self.tiles]
            }
            
            update_request = UpdateStorageRequest(
                storage_type=StorageType.DATABASE,
                path="napthaville_environment",
                data={
                    "data": {
                        "data": json.dumps(tiles_matrix)
                    }
                },
                options={
                    "condition": {"type": "tiles"}
                }
            )
            await self.storage_client.execute(update_request)

        except Exception as e:
            logger.error(f"Error removing subject events from tile ({tile.x}, {tile.y}): {str(e)}")
            raise