"""
AthleteOS - RAG integration layer (Step 5).

Connects retriever.py (Step 3) to the LLM provider: turns an athlete profile
into a short search query, retrieves the top-k most relevant knowledge_base
chunks, and formats them into a "context block" the provider can inject into
its prompt. Kept as its own module (not inside llm_provider.py) so retrieval
logic stays separate from the code that talks to the model - the mock
provider and the rest of the app never import this file.

Also owns the logging for this stage, so a developer can see:
  - the query sent to the retriever
  - which chunks came back (and how relevant they were)
  - the length of the final prompt after the context block is injected
"""

import logging

from regional_foods import REGIONS, detect_region
from retriever import retrieve_relevant_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("athleteos.rag")

TOP_K = 3


def _developmental_stage_phrase(age):
    """
    Turn age into a short phrase that biases retrieval toward age-appropriate
    knowledge_base content (e.g. youth_athlete_nutrition.md) for a minor
    athlete. Returns "" for an adult (age is None or >= 18) so the query -
    and therefore retrieval - for the common adult case is unchanged.

    Audit finding this fixes: build_retrieval_query() previously never
    referenced age at all, so a 10-year-old and a 25-year-old with an
    otherwise identical profile produced the EXACT SAME retrieval query and
    therefore the exact same retrieved chunks - the youth knowledge-base
    document was never actually surfaced for a real profile, even though it
    existed and was retrievable by a hand-written query that mentioned age.
    """
    if age is None or age >= 18:
        return ""
    if age < 13:
        return f"{age} year old child athlete"
    return f"{age} year old adolescent athlete"


# Human-readable zone labels, phrased the way the knowledge base itself
# refers to each zone ("## South India", "a south Indian athlete's plate"),
# so the query wording actually matches the documents it should surface.
REGION_LABELS = {
    "north": "North Indian",
    "south": "South Indian",
    "east": "East Indian",
    "west": "West Indian",
    "central": "Central Indian",
    "northeast": "Northeast Indian",
}


def _matches_a_known_zone(region_text):
    """
    True only if the region text actually matches one of the six zones'
    keywords. detect_region() falls back to "north" for anything it doesn't
    recognise, which is the right default for PICKING FOOD but the wrong
    thing to assert in a retrieval query - telling the retriever that an
    athlete in London is "North Indian" would actively mislead it.
    """
    text = " " + region_text.lower()
    return any(k in text for data in REGIONS.values() for k in data["keywords"])


def _region_phrase(profile):
    """
    Turn the athlete's region into a phrase that biases retrieval toward
    region-appropriate knowledge (e.g. indian_regional_eating_patterns.md,
    and the regional availability table in indian_protein_sources.md).
    Returns "" when no region is given, so those profiles are unchanged.

    Same pattern, and the same underlying bug, as
    _developmental_stage_phrase() above: build_retrieval_query() previously
    never referenced region at all, so a Chennai athlete and a Lucknow
    athlete produced the EXACT SAME retrieval query and therefore the exact
    same chunks - the regional knowledge-base documents could never surface
    through the real pipeline, only through a hand-written query that
    happened to name a region.

    The raw region text is always included (it is what the athlete actually
    typed, and the documents do name states). The zone label is added only
    when the text genuinely matches a known zone - see
    _matches_a_known_zone().
    """
    region = (profile.get("region") or "").strip()
    if not region:
        return ""
    if not _matches_a_known_zone(region):
        return region
    label = REGION_LABELS.get(detect_region(profile))
    return f"{region} - {label} regional food and eating patterns" if label else region


def build_retrieval_query(profile):
    """
    Turn the athlete profile into a short natural-language query for the
    retriever - the same kind of sentence a person might type in, built from
    the fields most likely to match a nutrition knowledge_base chunk.
    """
    parts = [
        profile.get("goal", ""),
        f"{profile.get('diet', '')} footballer",
        f"training {profile.get('training_days_per_week', '')} days/week, "
        f"{profile.get('training_intensity', '')} intensity",
        _developmental_stage_phrase(profile.get("age")),
        _region_phrase(profile),
    ]
    if profile.get("notes"):
        parts.append(profile["notes"])
    return " - ".join(p for p in parts if p)


# How many chunks each focused top-up search may contribute.
#
# Region needs two, not one: the regional document's own introduction
# outranks the individual region sections for a region query, so a single
# slot gets spent on the intro and the actual section is never reached.
# Measured across all six zones, the correct section is always within the
# region query's top two.
#
# Age needs only one: the developmental-stage phrase on its own retrieves
# youth_athlete_nutrition.md at rank 1 for every age from 8 to 17.
REGION_TOP_UP = 2
YOUTH_TOP_UP = 1


def _top_up(chunks, focused_query, limit, label):
    """
    Append up to `limit` chunks found by a short, single-topic query,
    skipping anything already retrieved. Mutates and returns `chunks`.

    Why a second (and third) search instead of just a longer main query: a
    profile query is mostly training vocabulary ("Improve football
    performance - Vegetarian footballer - training 5 days/week, High
    intensity - 10 year old child athlete - Chennai, Tamil Nadu - South
    Indian regional food..."). Retrieval embeds that whole string as ONE
    vector, so short tails like the region or the developmental stage are
    averaged away and generic athlete-nutrition chunks win. Measured, the
    per-region sections did not reach even the FAISS top 10, and for a
    10-year-old whose goal read "Improve football performance" the youth
    document did not appear at all. Each phrase searched ON ITS OWN
    retrieves the right material immediately.

    This reuses the existing retriever - same model, same index, same
    retrieve_relevant_chunks() - rather than changing how retrieval works.
    An empty focused_query (adult athlete, or no region given) contributes
    nothing, so those profiles behave exactly as before.
    """
    if not focused_query or limit <= 0:
        return chunks

    already_have = {c["id"] for c in chunks}
    added = 0
    # Ask for enough candidates that `limit` survive de-duplication.
    for chunk in retrieve_relevant_chunks(focused_query, top_k=limit + len(chunks)):
        if added >= limit:
            break
        if chunk["id"] not in already_have:
            chunks.append(chunk)
            already_have.add(chunk["id"])
            added += 1
            logger.info("%s top-up added chunk: %s", label, chunk["id"])

    return chunks


def retrieve_profile_chunks(profile, query, top_k=TOP_K):
    """
    Retrieve context for one athlete: the normal profile-driven search, plus
    small focused top-ups for the two things the main query reliably loses -
    the athlete's developmental stage and their region. See _top_up().

    The youth top-up runs first: for a minor, age-appropriate guidance is
    the more safety-relevant of the two.
    """
    chunks = retrieve_relevant_chunks(query, top_k=top_k)
    chunks = _top_up(chunks, _developmental_stage_phrase(profile.get("age")),
                     YOUTH_TOP_UP, "youth")
    chunks = _top_up(chunks, _region_phrase(profile), REGION_TOP_UP, "region")
    return chunks


def format_context_block(chunks):
    """Turn a list of retrieved chunks into one text block, labeled by source."""
    if not chunks:
        return ""
    sections = [f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in chunks]
    return "\n\n".join(sections)


def get_context_block(profile, top_k=TOP_K):
    """
    Retrieve the most relevant knowledge_base chunks for this athlete profile
    and return them as one formatted context block. Logs the query and each
    retrieved chunk along the way.

    Delegates to retrieve_profile_chunks() - the canonical profile-aware
    retrieval path - so this returns the SAME context the orchestrator would
    build for the same athlete. It previously called retrieve_relevant_chunks()
    directly and therefore silently skipped the age and region top-ups: for a
    10-year-old in Chennai it returned neither the youth document nor the South
    India section, while the orchestrator path returned both. Only
    claude_provider()'s standalone mode (agent_context=None) and the CLI reach
    this function, which is exactly why the divergence went unnoticed.
    """
    query = build_retrieval_query(profile)
    logger.info("RAG query: %s", query)

    chunks = retrieve_profile_chunks(profile, query, top_k=top_k)
    for rank, chunk in enumerate(chunks, start=1):
        logger.info(
            "  retrieved #%d: %s (similarity=%.3f)",
            rank, chunk["id"], chunk["similarity"],
        )

    return format_context_block(chunks)
