"""JSON schemas for track logging types."""

import json

LOG_SCHEMA = {
    "title": "foxglove.Log",
    "description": "A log message",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "description": "Timestamp of the log message",
            "properties": {
                "sec": {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
            },
            "required": ["sec", "nsec"],
        },
        "level": {
            "type": "integer",
            "title": "foxglove.LogLevel",
            "description": "Log level (1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR, 5=FATAL)",
            "enum": [0, 1, 2, 3, 4, 5],
        },
        "message": {"type": "string", "description": "Log message"},
        "name": {"type": "string", "description": "Process or node name"},
        "file": {"type": "string", "description": "Filename"},
        "line": {"type": "integer", "minimum": 0, "description": "Line number in the file"},
    },
    "required": ["timestamp", "level", "message"],
}

COMPRESSED_IMAGE_SCHEMA = {
    "title": "foxglove.CompressedImage",
    "description": "A compressed image",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "description": "Timestamp of the image",
            "properties": {
                "sec": {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
            },
            "required": ["sec", "nsec"],
        },
        "frame_id": {
            "type": "string",
            "description": "Frame of reference for the image",
        },
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Compressed image data (base64 encoded)",
        },
        "format": {
            "type": "string",
            "description": "Image format (e.g., 'jpeg', 'png', 'webp')",
        },
    },
    "required": ["timestamp", "data", "format"],
}

POINT_CLOUD_SCHEMA = {
    "title": "foxglove.PointCloud",
    "description": "A collection of N-dimensional points",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "description": "Timestamp of the point cloud",
            "properties": {
                "sec": {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
            },
            "required": ["sec", "nsec"],
        },
        "frame_id": {
            "type": "string",
            "description": "Frame of reference",
        },
        "pose": {
            "type": "object",
            "title": "foxglove.Pose",
            "description": "Origin of the point cloud relative to the frame of reference",
            "properties": {
                "position": {
                    "type": "object",
                    "title": "foxglove.Vector3",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                    "required": ["x", "y", "z"],
                },
                "orientation": {
                    "type": "object",
                    "title": "foxglove.Quaternion",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "w": {"type": "number"},
                    },
                    "required": ["x", "y", "z", "w"],
                },
            },
            "required": ["position", "orientation"],
        },
        "point_stride": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of bytes between points in the data",
        },
        "fields": {
            "type": "array",
            "description": "Fields in the data buffer",
            "items": {
                "type": "object",
                "title": "foxglove.PackedElementField",
                "properties": {
                    "name": {"type": "string", "description": "Field name"},
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Byte offset from start of point",
                    },
                    "type": {
                        "type": "integer",
                        "description": "Numeric type (1=UINT8, 2=INT8, 3=UINT16, 4=INT16, 5=UINT32, 6=INT32, 7=FLOAT32, 8=FLOAT64)",
                        "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    },
                },
                "required": ["name", "offset", "type"],
            },
        },
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Point data (base64 encoded)",
        },
    },
    "required": ["timestamp", "frame_id", "pose", "point_stride", "fields", "data"],
}


def get_schema_json(schema: dict) -> bytes:
    """Convert schema dict to JSON bytes."""
    return json.dumps(schema).encode("utf-8")
