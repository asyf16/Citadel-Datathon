import re

LAUNCH_PATTERNS = [
        r"\blaunch\b", r"\bintroducing\b", r"\bannounce\b", r"\brelease\b",
        r"\bnow available\b", r"\blaunching\b", r"\bintroduce\b"
    ]
launch_re = re.compile("|".join(LAUNCH_PATTERNS), re.IGNORECASE)