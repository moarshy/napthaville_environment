from pathlib import Path
import json
import logging
import asyncio
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
from naptha_sdk.storage.storage_client import StorageClient
from naptha_sdk.schemas import NodeConfigUser

logger = logging.getLogger(__name__)

class NapthavilleEnvironment:
    @classmethod
    async def create(cls, deployment: Dict) -> "NapthavilleEnvironment":
        """Create a new NapthavilleEnvironment instance asynchronously"""
        env = cls.__new__(cls)
        await env.__ainit__(deployment)
        return env
    
    async def __ainit__(self, deployment: Dict):
        """Async initialization method"""
        self.deployment = deployment
        self.config = MazeConfig(**deployment["config"])
        if not isinstance(deployment["node"], NodeConfigUser):
            node_config = NodeConfigUser(
                ip=deployment["node"]["ip"],
                user_communication_port=deployment["node"]["user_communication_port"],
                user_communication_protocol=deployment["node"]["user_communication_protocol"]
            )
        else:
            node_config = deployment["node"]
        self.storage_client = StorageClient(node_config)
        self.maze = await Maze.create(deployment["config"], self.storage_client)

    def __init__(self):
        """Prevent direct instantiation, use create() instead"""
        raise RuntimeError("Please use NapthavilleEnvironment.create() to instantiate")

    def turn_coordinate_to_tile(self, inputs: Dict) -> Dict:
        """Convert pixel coordinates to tile location"""
        pixel_coord = PixelCoordinate(**inputs["px_coordinate"])
        result = self.maze.turn_coordinate_to_tile(pixel_coord)
        return result.model_dump()

    async def access_tile(self, inputs: Dict) -> Dict:
        """Access tile data including events"""
        tile = TileLocation(**inputs["tile"])
        result = await self.maze.access_tile(tile)
        if result:
            result_dict = result.model_dump()
            result_dict["events"] = [list(event) for event in result_dict["events"]]
            return result_dict
        return None

    async def get_tile_path(self, inputs: Dict) -> Dict:
        """Get tile path for given level"""
        tile = TileLocation(**inputs["tile"])
        level = TileLevel(inputs["level"])
        result = await self.maze.get_tile_path(tile, level)
        return result.model_dump() if result else None

    def get_nearby_tiles(self, inputs: Dict) -> Dict:
        """Get nearby tiles within vision radius"""
        tile = TileLocation(**inputs["tile"])
        vision_r = VisionRadius(radius=inputs["vision_r"])
        result = self.maze.get_nearby_tiles(tile, vision_r)
        return {"tiles": [t.model_dump() for t in result.tiles]}

    async def add_event_from_tile(self, inputs: Dict) -> Dict:
        """Add an event to a tile"""
        try:
            tile = TileLocation(**inputs["tile"])
            event_tuple = tuple(inputs["curr_event"])
            
            await self.maze.add_event_from_tile(event_tuple, tile)
            tile_after = await self.maze.access_tile(tile)
            
            return {
                "success": True,
                "events": [list(e) for e in tile_after.events]
            }
        except Exception as e:
            logger.error(f"Error in add_event_from_tile: {e}")
            return {"success": False, "error": str(e)}

    async def remove_event_from_tile(self, inputs: Dict) -> Dict:
        """Remove an event from a tile"""
        try:
            tile = TileLocation(**inputs["tile"])
            event = tuple(inputs["curr_event"])
            
            await self.maze.remove_event_from_tile(event, tile)
            result = await self.maze.access_tile(tile)
            
            return {
                "success": True,
                "events": [list(e) for e in result.events]
            }
        except Exception as e:
            logger.error(f"Error in remove_event_from_tile: {e}")
            return {"success": False, "error": str(e)}

    async def turn_event_from_tile_idle(self, inputs: Dict) -> Dict:
        """Convert an event to idle state"""
        try:
            tile = TileLocation(**inputs["tile"])
            event = tuple(inputs["curr_event"])
            
            await self.maze.turn_event_from_tile_idle(event, tile)
            tile_after = await self.maze.access_tile(tile)
            return {
                "success": True,
                "events": [list(e) for e in tile_after.events]
            }
        except Exception as e:
            logger.error(f"Error in turn_event_from_tile_idle: {e}")
            return {"success": False, "error": str(e)}

    async def remove_subject_events_from_tile(self, inputs: Dict) -> Dict:
        """Remove all events for a subject from a tile"""
        try:
            tile = TileLocation(**inputs["tile"])
            subject = str(inputs["subject"])
            
            await self.maze.remove_subject_events_from_tile(subject, tile)
            result = await self.maze.access_tile(tile)
            
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
        env = await NapthavilleEnvironment.create(module_run["deployment"])
        method = getattr(env, module_run_input.function_name, None)
        
        if not method:
            raise ValueError(f"Unknown function: {module_run_input.function_name}")
        
        if asyncio.iscoroutinefunction(method):
            return await method(module_run_input.function_input_data)
        return method(module_run_input.function_input_data)
    except Exception as e:
        logger.error(f"Error in run: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    file_dir = Path(__file__).parent
    deployment = json.load(open(file_dir / "configs" / "deployment.json"))
    deployment = deployment[0]
    deployment["node"] = {
        "ip": "localhost",
        "user_communication_port": 7001,
        "user_communication_protocol": "http"
    }
    
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
            
            # Test 5: Event operations
            {
                "name": "add_event",
                "inputs": {
                    "function_name": "add_event_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ["test_event", "param1", "param2", "param3"]
                    }
                }
            },
            
            {
                "name": "make_event_idle",
                "inputs": {
                    "function_name": "turn_event_from_tile_idle",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ["test_event", "param1", "param2", "param3"]
                    }
                }
            },
            
            {
                "name": "remove_event",
                "inputs": {
                    "function_name": "remove_event_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ["test_event", None, None, None]
                    }
                }
            },
            
            # Test 6: Subject events
            {
                "name": "add_subject_event",
                "inputs": {
                    "function_name": "add_event_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "curr_event": ["Mr. X", "param1", "param2", "param3"]
                    }
                }
            },
            
            {
                "name": "remove_subject_events",
                "inputs": {
                    "function_name": "remove_subject_events_from_tile",
                    "function_input_data": {
                        "tile": test_tile,
                        "subject": "Mr. X"
                    }
                }
            }
        ]

        results = {}
        for test in tests:
            logger.info(f"\nRunning test: {test['name']}")
            module_run = {"inputs": test["inputs"], "deployment": deployment}
            result = await run(module_run)
            
            logger.info(f"Result for {test['name']}: {result}")
            results[test['name']] = result

        return results

    asyncio.run(test_all())