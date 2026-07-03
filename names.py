"""Human-friendly session names — adjective-adjective-noun slugs.

``cursor_create_session`` mints one (e.g. ``playful-space-bunny``) as THE
handle callers pass to every other tool. The cursor ACP session UUID stays
recorded on the handle entry as an alias (``handles.resolve``), so a UUID
still resolves anywhere a name is accepted.

Vocabulary: 50 mood adjectives x 50 modifier adjectives x 100 creatures =
250,000 combinations. Collision policy: random draws checked against the
caller-supplied ``taken`` predicate (the handle table), then a numeric
suffix as the can't-fail fallback.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

# 50 mood adjectives.
MOODS = (
    "playful", "mellow", "brave", "calm", "clever",
    "daring", "eager", "gentle", "happy", "jolly",
    "keen", "lively", "merry", "noble", "proud",
    "quick", "quiet", "rapid", "sunny", "swift",
    "tidy", "witty", "zesty", "bold", "bright",
    "breezy", "cheery", "cozy", "crisp", "dandy",
    "dapper", "dreamy", "fearless", "frisky", "groovy",
    "humble", "jaunty", "lucky", "nimble", "peppy",
    "perky", "plucky", "rosy", "sleek", "snappy",
    "spry", "sturdy", "zippy", "chipper", "spunky",
)

# 50 modifier adjectives (colors / materials / places / elements).
MODIFIERS = (
    "space", "copper", "amber", "azure", "cedar",
    "coral", "cosmic", "crimson", "crystal", "desert",
    "dusty", "ebony", "emerald", "foggy", "forest",
    "frosty", "golden", "granite", "hazel", "honey",
    "indigo", "iron", "ivory", "jade", "lunar",
    "maple", "marble", "meadow", "midnight", "misty",
    "mossy", "neon", "ocean", "olive", "onyx",
    "opal", "pearl", "pine", "plasma", "prairie",
    "quartz", "river", "ruby", "rustic", "sandy",
    "scarlet", "silver", "solar", "velvet", "winter",
)

# 100 creatures.
CREATURES = (
    "bunny", "otter", "fox", "owl", "wolf",
    "bear", "hawk", "lynx", "mole", "newt",
    "crab", "dove", "duck", "deer", "elk",
    "finch", "frog", "gecko", "goose", "heron",
    "ibis", "koala", "lemur", "llama", "marmot",
    "mink", "moose", "moth", "mouse", "orca",
    "panda", "pika", "pony", "puffin", "quail",
    "rabbit", "raven", "robin", "seal", "shrew",
    "skunk", "sloth", "snail", "sparrow", "squid",
    "stork", "swan", "tapir", "toad", "trout",
    "turtle", "viper", "vole", "walrus", "weasel",
    "whale", "wren", "yak", "zebra", "badger",
    "beaver", "bison", "bobcat", "camel", "cheetah",
    "chipmunk", "cobra", "condor", "cougar", "coyote",
    "crane", "dingo", "donkey", "eagle", "falcon",
    "ferret", "gazelle", "gibbon", "hamster", "hedgehog",
    "jackal", "jaguar", "kestrel", "kitten", "leopard",
    "lizard", "lobster", "magpie", "meerkat", "ocelot",
    "octopus", "osprey", "oriole", "panther", "parrot",
    "pelican", "penguin", "pigeon", "salmon", "gopher",
)

# Random draws before falling back to a numeric suffix. With 250k combos
# even a heavily-populated handle table (cap 500) collides ~0.2% per draw,
# so 64 draws makes the fallback effectively unreachable — but it exists,
# so name generation can never loop forever or fail.
MAX_RANDOM_ATTEMPTS = 64


def generate(
    taken: Callable[[str], bool],
    rng: Optional[random.Random] = None,
) -> str:
    """A fresh slug that ``taken`` does not already claim."""
    pick = (rng or random).choice
    name = ""
    for _ in range(MAX_RANDOM_ATTEMPTS):
        name = f"{pick(MOODS)}-{pick(MODIFIERS)}-{pick(CREATURES)}"
        if not taken(name):
            return name
    suffix = 2
    while taken(f"{name}-{suffix}"):
        suffix += 1
    return f"{name}-{suffix}"
