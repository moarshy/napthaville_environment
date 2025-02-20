# run.py

from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
from napthaville_environment.maze import Maze
from napthaville_environment.schemas import (
    TileLocation,
    PixelCoordinate,
    TileLevel,
    VisionRadius,
    MazeConfig,
    InputSchema
)

logger = logging.getLogger(__name__)

class NapthavilleEnvironment:
    def __init__(self, deployment: Dict):
        self.deployment = deployment
        self.config = MazeConfig(**deployment["config"])
        self.maze = Maze(deployment["config"])

    def turn_coordinate_to_tile(self, inputs: Dict) -> Dict:
        pixel_coord = PixelCoordinate(**inputs["px_coordinate"])
        result = self.maze.turn_coordinate_to_tile(pixel_coord)
        return result.model_dump()

    def access_tile(self, inputs: Dict) -> Dict:
        """Access tile data including events"""
        tile = TileLocation(**inputs["tile"])
        result = self.maze.access_tile(tile)
        if result:
            result_dict = result.model_dump()
            result_dict["events"] = [list(event) for event in result_dict["events"]]
            return result_dict
        return None

    def get_tile_path(self, inputs: Dict) -> Dict:
        tile = TileLocation(**inputs["tile"])
        level = TileLevel(inputs["level"])
        result = self.maze.get_tile_path(tile, level)
        return result.model_dump() if result else None

    def get_nearby_tiles(self, inputs: Dict) -> Dict:
        tile = TileLocation(**inputs["tile"])
        vision_r = VisionRadius(radius=inputs["vision_r"])
        result = self.maze.get_nearby_tiles(tile, vision_r)
        return {"tiles": [t.model_dump() for t in result.tiles]}

    def add_event_from_tile(self, inputs: Dict) -> Dict:
        """Add an event to a tile"""
        try:
            tile = TileLocation(**inputs["tile"])
            event_tuple = tuple(inputs["curr_event"])  # Direct conversion
            
            self.maze.add_event_from_tile(event_tuple, tile)
            tile_after = self.maze.access_tile(tile)
            
            return {
                "success": True,
                "events": [list(e) for e in tile_after.events]
            }
        except Exception as e:
            logger.error(f"Error in add_event_from_tile: {e}")
            return {"success": False, "error": str(e)}

    def remove_event_from_tile(self, inputs: Dict) -> Dict:
        """Remove an event from a tile"""
        try:
            tile = TileLocation(**inputs["tile"])
            event = tuple(inputs["curr_event"])
            
            self.maze.remove_event_from_tile(event, tile)
            result = self.maze.access_tile(tile)
            
            return {
                "success": True,
                "events": [list(e) for e in result.events]
            }
        except Exception as e:
            logger.error(f"Error in remove_event_from_tile: {e}")
            return {"success": False, "error": str(e)}

    def turn_event_from_tile_idle(self, inputs: Dict) -> Dict:
        """Convert an event to idle state"""
        try:
            tile = TileLocation(**inputs["tile"])
            self.maze.turn_event_from_tile_idle(inputs["curr_event"], tile)
            tile_after = self.maze.access_tile(tile)
            return {
                "success": True,
                "events": [list(e) for e in tile_after.events]
            }
        except Exception as e:
            logger.error(f"Error in turn_event_from_tile_idle: {e}")
            return {"success": False, "error": str(e)}

    def remove_subject_events_from_tile(self, inputs: Dict) -> Dict:
        try:
            tile = TileLocation(**inputs["tile"])
            subject = str(inputs["subject"])
            
            self.maze.remove_subject_events_from_tile(subject, tile)
            result = self.maze.access_tile(tile)
            
            return {
                "success": True,
                "events": [list(e) for e in result.events]
            }
        except Exception as e:
            logger.error(f"Error in remove_subject_events_from_tile: {e}")
            return {"success": False, "error": str(e)}

async def run(module_run: Dict, *args, **kwargs) -> Dict:
    """Main entry point for the environment module"""
    try:
        module_run_input = InputSchema(**module_run["inputs"])
        napthaville_env = NapthavilleEnvironment(module_run["deployment"])
        method = getattr(napthaville_env, module_run_input.function_name, None)
        if not method:
            raise ValueError(f"Unknown function: {module_run_input.function_name}")
        return method(module_run_input.function_input_data)
    except Exception as e:
        logger.error(f"Error in run: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    file_dir = Path(__file__).parent
    deployment = json.load(open(file_dir / "configs" / "deployment.json"))

    async def test_all():
        """Run all tests matching the successful direct testing approach"""
        test_tile = {"x": 58, "y": 9}
        tests = [
            # Test 1: Coordinate to tile conversion
            {
                "name": "coordinate_conversion",
                "inputs": {
                    "function_name": "turn_coordinate_to_tile",
                    "function_input_data": {
                        "px_coordinate": {"x": 1600, "y": 384}
                    }
                }
            },
            
            # Test 2: Access tile details
            {
                "name": "access_tile",
                "inputs": {
                    "function_name": "access_tile",
                    "function_input_data": {
                        "tile": test_tile
                    }
                }
            },
            
            # Test 3: Get tile path
            {
                "name": "get_tile_path",
                "inputs": {
                    "function_name": "get_tile_path",
                    "function_input_data": {
                        "tile": test_tile,
                        "level": "arena"
                    }
                }
            },
            
            # Test 4: Get nearby tiles
            {
                "name": "get_nearby_tiles",
                "inputs": {
                    "function_name": "get_nearby_tiles",
                    "function_input_data": {
                        "tile": test_tile,
                        "vision_r": 2
                    }
                }
            },
            
            # Test 5a: Add event
            {
                "name": "add_event",
                "inputs": {
                    "function_name": "add_event_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ("test_event", None, None, None)
                    }
                }
            },

            # test 5a.1: access tile after adding event
            {
                "name": "access_tile_after_adding_event",
                "inputs": {
                    "function_name": "access_tile",
                    "function_input_data": {
                        "tile": test_tile
                    }
                }
            },
            
            # Test 5b: Make event idle
            {
                "name": "make_event_idle",
                "inputs": {
                    "function_name": "turn_event_from_tile_idle",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ("test_event", None, None, None)
                    }
                }
            },
            
            # Test 5c: Remove event
            {
                "name": "remove_event",
                "inputs": {
                    "function_name": "remove_event_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ("test_event", None, None, None)
                    }
                }
            },
            
            # Test 6a: Add subject event
            {
                "name": "add_subject_event",
                "inputs": {
                    "function_name": "add_event_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ("test_subject", None, None, None)
                    }
                }
            },
            
            # Test 6b: Remove subject events
            {
                "name": "remove_subject_events",
                "inputs": {
                    "function_name": "remove_subject_events_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "subject": "test_subject"
                    }
                }
            }
        ]

        results = {}
        for test in tests:
            logger.info(f"\nRunning test: {test['name']}")
            module_run = {"inputs": test["inputs"], "deployment": deployment[0]}
            result = await run(module_run)            
            # Log appropriate details based on test type
            if test["name"] == "coordinate_conversion":
                logger.info(f"Pixel (1600, 384) -> Tile {result}")
            elif test["name"] == "access_tile":
                logger.info(f"Tile {test_tile} details: {result}")
            elif test["name"] == "get_tile_path":
                logger.info(f"Path for tile {test_tile} at arena level: {result}")
            elif test["name"] == "get_nearby_tiles":
                logger.info(f"Nearby tiles count: {len(result['tiles'])}")
                logger.info(f"First few nearby tiles: {result['tiles'][:5]}")
            elif "event" in test["name"]:
                logger.info(f"Events after {test['name']}: {result.get('events', [])}")
            
            results[test["name"]] = result

        # Test 7: Load functions
        logger.info("\nTesting load functions")
        env = NapthavilleEnvironment(deployment[0])
        collision_maze = env.maze.load_collision_maze()
        tiles = env.maze.load_tiles()
        address_tiles = env.maze.load_address_tiles()
        
        logger.info(f"Loaded collision maze size: {len(collision_maze)} x {len(collision_maze[0])}")
        logger.info(f"Loaded tiles size: {len(tiles)} x {len(tiles[0])}")
        logger.info(f"Loaded address tiles count: {len(address_tiles)}")

        return results

    asyncio.run(test_all())