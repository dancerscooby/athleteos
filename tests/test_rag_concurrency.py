"""
AthleteOS - RAG concurrency/thread-safety tests (production-safety fix).

FastAPI runs sync `def` endpoints (api.py's create_meal_plan/chat) in a
shared thread pool, so concurrent HTTP requests become concurrent Python
threads inside ONE process. This file reproduces, against the REAL
retriever/reranker pipeline (no mocking of the ONNX models - a mock would
defeat the purpose of testing actual thread safety), the two problems the
concurrency investigation found and the two locks that fix them:

  1. retriever._load_lock / reranker._load_lock - without these, concurrent
     FIRST requests on a fresh process could each construct their own
     separate ~300MB model instance before any of them finished (measured:
     4 racing threads => 4 separate constructions). Fixed with
     double-checked locking - see retriever.py's module docstring.

  2. retriever._retrieval_lock - without this, concurrent requests on an
     ALREADY-warm, correctly-shared model still cost real additional
     memory per simultaneous request (measured: peak RSS crossed Render's
     512MB limit at just 3 concurrent requests). Fixed by serializing
     retrieve_relevant_chunks() itself - concurrent callers wait their
     turn instead of running simultaneously.

Every test here restores retriever's and reranker's module-level state in
tearDown, since these are process-global singletons and this test file
runs inside the same process as every other test file.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import mock

import reranker
import retriever


def _save_module_state():
    return {
        "retriever._model": retriever._model,
        "retriever._index": retriever._index,
        "retriever._records": retriever._records,
        "reranker._reranker": reranker._reranker,
    }


def _restore_module_state(saved):
    retriever._model = saved["retriever._model"]
    retriever._index = saved["retriever._index"]
    retriever._records = saved["retriever._records"]
    reranker._reranker = saved["reranker._reranker"]


class RAGStateIsolatedTestCase(unittest.TestCase):
    """Saves/restores retriever.py's and reranker.py's module-level
    singleton state, so tests that force a cold (None) state don't leave
    other test files' RAG calls needing to reload the models."""

    def setUp(self):
        self._saved = _save_module_state()

    def tearDown(self):
        _restore_module_state(self._saved)


# ----------------------------------------------------------------------
# 1. Concurrent cold-start initialization -> exactly one instance each
# ----------------------------------------------------------------------

class TestConcurrentColdStartIsThreadSafe(RAGStateIsolatedTestCase):
    def test_concurrent_first_requests_construct_the_model_exactly_once(self):
        """N threads all reaching retrieve_relevant_chunks() for the very
        first time (a freshly-started process) must construct exactly ONE
        embedding model and ONE reranker, never one per racing thread."""
        retriever._model = None
        retriever._index = None
        retriever._records = None
        reranker._reranker = None

        real_text_embedding = retriever.TextEmbedding
        real_cross_encoder = reranker.TextCrossEncoder
        embedding_constructions = []
        reranker_constructions = []
        construction_lock = threading.Lock()

        def counting_text_embedding(*a, **kw):
            with construction_lock:
                embedding_constructions.append(1)
            return real_text_embedding(*a, **kw)

        def counting_cross_encoder(*a, **kw):
            with construction_lock:
                reranker_constructions.append(1)
            return real_cross_encoder(*a, **kw)

        n = 4
        barrier = threading.Barrier(n)

        def worker(query):
            barrier.wait()  # force every thread to call at the same instant
            return retriever.retrieve_relevant_chunks(query, top_k=3)

        with mock.patch.object(retriever, "TextEmbedding", side_effect=counting_text_embedding), \
             mock.patch.object(reranker, "TextCrossEncoder", side_effect=counting_cross_encoder):
            with ThreadPoolExecutor(max_workers=n) as ex:
                futures = [ex.submit(worker, f"query {i}") for i in range(n)]
                results = [f.result() for f in futures]

        self.assertEqual(len(embedding_constructions), 1,
                          "the embedding model must be constructed exactly once, not once per racing thread")
        self.assertEqual(len(reranker_constructions), 1,
                          "the reranker must be constructed exactly once, not once per racing thread")
        # Every racing thread must still get a real, usable result back.
        self.assertTrue(all(r for r in results))

    def test_all_threads_end_up_sharing_the_same_singleton(self):
        """After the race settles, every thread's view of the module-level
        singletons must point at the SAME object - no thread left holding
        a stale duplicate."""
        retriever._model = None
        retriever._index = None
        retriever._records = None
        reranker._reranker = None

        n = 4
        barrier = threading.Barrier(n)
        seen_model_ids = []
        seen_reranker_ids = []
        ids_lock = threading.Lock()

        def worker(query):
            barrier.wait()
            retriever.retrieve_relevant_chunks(query, top_k=3)
            with ids_lock:
                seen_model_ids.append(id(retriever._model))
                seen_reranker_ids.append(id(reranker._reranker))

        with ThreadPoolExecutor(max_workers=n) as ex:
            list(as_completed([ex.submit(worker, f"query {i}") for i in range(n)]))

        self.assertEqual(len(set(seen_model_ids)), 1)
        self.assertEqual(len(set(seen_reranker_ids)), 1)


# ----------------------------------------------------------------------
# 2. Concurrent RAG requests are serialized
# ----------------------------------------------------------------------

class TestConcurrentRequestsAreSerialized(RAGStateIsolatedTestCase):
    def test_retrieval_never_runs_on_more_than_one_thread_at_once(self):
        """With the retrieval lock in place, no two threads may be inside
        retrieve_relevant_chunks()'s critical section at the same time.
        Wraps the real rerank() with a small delay + an in-flight counter,
        so a broken lock would reliably be caught (a fast, undelayed call
        could coincidentally never overlap even if unserialized)."""
        # Ensure the models are already warm before measuring serialization,
        # matching the production scenario this fix targets (concurrent
        # requests on an already-loaded process).
        retriever.retrieve_relevant_chunks("warm-up", top_k=1)

        real_rerank = retriever.rerank
        in_flight = {"count": 0}
        max_observed = {"value": 0}
        state_lock = threading.Lock()

        def delayed_rerank(query, candidates, top_k=3):
            with state_lock:
                in_flight["count"] += 1
                max_observed["value"] = max(max_observed["value"], in_flight["count"])
            time.sleep(0.05)
            try:
                return real_rerank(query, candidates, top_k=top_k)
            finally:
                with state_lock:
                    in_flight["count"] -= 1

        n = 5
        with mock.patch.object(retriever, "rerank", side_effect=delayed_rerank):
            with ThreadPoolExecutor(max_workers=n) as ex:
                futures = [ex.submit(retriever.retrieve_relevant_chunks, f"query {i}", 3) for i in range(n)]
                results = [f.result() for f in futures]

        self.assertEqual(
            max_observed["value"], 1,
            "more than one thread was inside the retrieval critical section "
            "at the same time - the retrieval lock did not serialize them",
        )
        self.assertTrue(all(r for r in results))


# ----------------------------------------------------------------------
# 3. Sequential behaviour is unchanged
# ----------------------------------------------------------------------

class TestSequentialBehaviourUnchanged(RAGStateIsolatedTestCase):
    def test_repeated_sequential_calls_still_work_and_reuse_the_singleton(self):
        query = "What should I eat before a football match?"
        first = retriever.retrieve_relevant_chunks(query, top_k=3)
        model_after_first = id(retriever._model)
        reranker_after_first = id(reranker._reranker)

        second = retriever.retrieve_relevant_chunks(query, top_k=3)

        self.assertEqual(first, second, "the same query must still return the same result deterministically")
        self.assertEqual(id(retriever._model), model_after_first)
        self.assertEqual(id(reranker._reranker), reranker_after_first)

    def test_a_single_caller_is_not_blocked_by_its_own_lock(self):
        """A single, non-concurrent caller must never deadlock against its
        own retrieval lock (retrieve_relevant_chunks calls search_faiss_only
        and rerank sequentially within the same lock acquisition, not
        re-entrantly)."""
        result = retriever.retrieve_relevant_chunks("hydration during training", top_k=2)
        self.assertTrue(result)


# ----------------------------------------------------------------------
# 4. Athlete isolation remains intact under concurrency
# ----------------------------------------------------------------------

import tempfile
from pathlib import Path

import memory_integration as mi
import memory_manager as mm
import orchestrator as orch

BASE_PROFILE = dict(
    age=22, height_cm=178, weight_kg=72, sex="Male",
    goal="Improve football performance",
    diet="Vegetarian", training_days_per_week=5, training_intensity="High",
    allergies="none", budget_per_day=None, food_preferences=None,
    foods_to_avoid=None, notes=None, available_ingredients=None,
)


class MemoryIsolatedTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        self._orig = {n: getattr(mm, n) for n in
                      ["MEMORY_DIR", "PROFILES_FILE", "PREFERENCES_FILE", "LATEST_PLAN_FILE",
                       "RECIPES_FILE", "EPISODES_DIR", "CACHE_DIR", "UPDATABLE_CATEGORIES"]}
        mm.MEMORY_DIR = tmp
        mm.PROFILES_FILE = tmp / "profiles.json"
        mm.PREFERENCES_FILE = tmp / "preferences.json"
        mm.LATEST_PLAN_FILE = tmp / "latest_plans.json"
        mm.RECIPES_FILE = tmp / "recipes.json"
        mm.EPISODES_DIR = tmp / "episodes"
        mm.CACHE_DIR = tmp / "cache"
        mm.UPDATABLE_CATEGORIES = {"profile": mm.PROFILES_FILE,
                                   "preference": mm.PREFERENCES_FILE,
                                   "latest_plan": mm.LATEST_PLAN_FILE,
                                   "recipes": mm.RECIPES_FILE}

    def tearDown(self):
        for name, value in self._orig.items():
            setattr(mm, name, value)
        self._tmpdir.cleanup()


class TestAthleteIsolationUnderConcurrency(MemoryIsolatedTestCase):
    def test_two_athletes_generated_concurrently_never_mix_regions(self):
        """Two different athletes (different regions/athlete_ids) run
        through the full orchestrator - including the now-serialized RAG
        path - AT THE SAME TIME. Serialization must only delay execution,
        never share or mix up one request's region-specific data with
        another's."""
        from regional_foods import REGIONS

        south_profile = dict(BASE_PROFILE, region="Chennai, Tamil Nadu")
        north_profile = dict(BASE_PROFILE, region="Lucknow, Uttar Pradesh")

        barrier = threading.Barrier(2)

        def run(profile, athlete_id):
            barrier.wait()
            trace = orch.run_with_trace(profile, athlete_id=athlete_id)
            # Mirrors what api.py's create_meal_plan actually does after
            # run_with_trace - persisting is the API layer's job, not
            # the orchestrator's, so it's done explicitly here too.
            mi.update_memory_from_profile(profile, athlete_id)
            return trace

        with ThreadPoolExecutor(max_workers=2) as ex:
            south_future = ex.submit(run, south_profile, "south-athlete")
            north_future = ex.submit(run, north_profile, "north-athlete")
            south_trace = south_future.result()
            north_trace = north_future.result()

        south_items = {ing["item"] for m in south_trace["plan"]["meals"] for ing in m["ingredients"]}
        north_items = {ing["item"] for m in north_trace["plan"]["meals"] for ing in m["ingredients"]}

        south_pool = set(REGIONS["south"]["veg_protein"]) | set(REGIONS["south"]["carbs"])
        north_pool = set(REGIONS["north"]["veg_protein"]) | set(REGIONS["north"]["carbs"])

        # Each athlete's plan must draw from their OWN region's pool, not
        # the other's - proving concurrent execution never crossed wires.
        self.assertTrue(south_items & south_pool, "south athlete's plan lost its regional foods")
        self.assertTrue(north_items & north_pool, "north athlete's plan lost its regional foods")

        # Memory stayed isolated per athlete_id too (existing guarantee,
        # reconfirmed here specifically under concurrent execution).
        self.assertIsNotNone(mm.get("profile", "south-athlete"))
        self.assertIsNotNone(mm.get("profile", "north-athlete"))
        self.assertEqual(mm.get("profile", "south-athlete")["region"], "Chennai, Tamil Nadu")
        self.assertEqual(mm.get("profile", "north-athlete")["region"], "Lucknow, Uttar Pradesh")


if __name__ == "__main__":
    unittest.main(verbosity=2)
