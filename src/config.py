import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Paths:
    root: Path = Path(__file__).parent
    data: Path = root / "data"
    book: Path = (
        data
        / "Marcus_Aurelius_Antoninus_-_His_Meditations_concerning_himselfe/index.html"
    )
    manual: Path = (
        data
        / "c2b0c2b0c2b0-cessna_182s_1997on_mm_182smm.pdf"
    )


openai_api_key = os.getenv("OPENAI_API_KEY")
