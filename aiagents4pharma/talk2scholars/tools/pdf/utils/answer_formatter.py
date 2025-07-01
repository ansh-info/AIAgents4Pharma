"""
Format the final answer text with source attributions and hardware info.
"""

import logging
from typing import Any, Dict, List


from .generate_answer import generate_answer


logger = logging.getLogger(__name__)


def format_answer(
    question: str,
    chunks: List[Any],
    llm: Any,
    articles: Dict[str, Any],
    config: Any,
    call_id: str,
    has_gpu: bool,
) -> str:
    """Generate the final answer text with source attributions and hardware info."""
    result = generate_answer(question, chunks, llm, config)
    answer = result.get("output_text", "No answer generated.")

    # Get unique paper titles for source attribution
    titles: Dict[str, str] = {}
    for pid in result.get("papers_used", []):
        if pid in articles:
            titles[pid] = articles[pid].get("Title", "Unknown paper")

    # Format sources
    if titles:
        srcs = "\n\nSources:\n" + "\n".join(f"- {t}" for t in titles.values())
    else:
        srcs = ""

    # Log final statistics with hardware info
    hardware_info = "GPU-accelerated" if has_gpu else "CPU-processed"
    logger.info(
        "%s: Generated answer using %d chunks from %d papers (%s)",
        call_id,
        len(chunks),
        len(titles),
        hardware_info,
    )

    # Add subtle hardware info to logs but not to user output
    logger.debug(
        "%s: Answer generation completed with %s processing",
        call_id,
        hardware_info,
    )

    return f"{answer}{srcs}"
