"""
cv_flow — Topic-based DAM (Direct Access Memory) computer vision pipeline framework.

Quick start
-----------
    import cv_flow

    cv_flow.load_topics("topics/")

    class MyNode(cv_flow.Node):
        def initialize(self):
            self.pub = self.advertise("camera_frame")
        def spin_once(self):
            ...

    executor = cv_flow.Executor([MyNode()])
    executor.spin()
"""
from __future__ import annotations

__version__ = "0.3.0"

from cv_flow.node import Node
from cv_flow.executor import Executor
from cv_flow.topic.topic import (
    Topic, get_topic, list_topics, clear_topics, load_topics,
)
from cv_flow.topic.types import TopicDef, PortDef, FieldDef, DTYPE_MAP
from cv_flow.topic.parser import parse_topic_file, load_topics_dir, ParseError

__all__ = [
    "__version__",
    "Node", "Executor",
    "Topic", "get_topic", "list_topics", "clear_topics", "load_topics",
    "TopicDef", "PortDef", "FieldDef", "DTYPE_MAP",
    "parse_topic_file", "load_topics_dir", "ParseError",
]
