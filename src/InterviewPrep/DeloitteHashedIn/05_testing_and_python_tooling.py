"""
===================================================================================
PHASE 5 — TESTING (pytest) & PYTHON TOOLING (HashedIn by Deloitte — Lead)
===================================================================================

WHY THIS PHASE:
    The JD explicitly demands: "Strong proficiency with Python testing
    frameworks like pytest, with a focus on writing comprehensive unit,
    functional, and integration tests" AND "Solid understanding of Python
    packaging, dependency management, and virtual environments... Poetry, uv,
    pip, and virtualenv/venv." For a LEAD, you OWN the testing strategy and the
    project's tooling/standards — not just write tests.

DEPTH LEVEL: Technical Lead. Strategy + craft + how you'd enforce it on a team.

SECTIONS:
    1.  The Testing Strategy (pyramid, what/why, lead's view)
    2.  pytest Fundamentals (test discovery, assertions, structure)
    3.  Fixtures (the heart of pytest) + conftest + scopes
    4.  Parametrization (table-driven tests)
    5.  Mocking & Patching (isolate the unit)
    6.  Testing Async Code + FastAPI (ties to Phases 1-2)
    7.  Testing AI/LLM Code (the hard part — non-determinism)
    8.  Coverage, CI, and Test Quality
    9.  Python Packaging & Dependency Management (pip/venv/Poetry/uv)
    10. Project Structure & Environments
    11. Interview Q&A (lead-level)
    12. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: THE TESTING STRATEGY (pyramid, what/why, lead's view)
# =================================================================================
'''
A lead OWNS the testing strategy. Don't just "write tests" — have a philosophy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE TEST PYRAMID (the JD names unit, functional, integration):

         /\        E2E / UI tests        — FEW: slow, brittle, full system
        /  \       (the whole app, real deps)
       /----\      Integration tests     — SOME: components together
      /      \     (endpoint + real DB, service + service)
     /--------\    Unit tests            — MANY: fast, isolated, one unit
    /__________\   (one function/class, deps mocked)

    PRINCIPLE: MANY fast unit tests at the base, FEWER integration tests,
    FEWEST slow E2E tests. Inverted pyramid (mostly E2E) = slow, flaky suite.

    THE THREE LEVELS (JD's exact words):
    - UNIT: one function/class in ISOLATION; dependencies mocked. Milliseconds.
    - INTEGRATION: multiple components TOGETHER (endpoint + DB, two services).
      Catches wiring/contract bugs unit tests miss.
    - FUNCTIONAL / E2E: a full user-facing flow end-to-end. Validates the
      system does what the user needs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY WE TEST (the lead framing — it's not bureaucracy):
    - CONFIDENCE to change/refactor without fear (the biggest value).
    - CATCH regressions early (cheaper than in production).
    - DOCUMENTATION: tests show how code is meant to be used.
    - DESIGN pressure: hard-to-test code is usually badly-designed code
      (tight coupling) — tests push you toward good design.

WHAT TO TEST (and what NOT to):
    - Test BEHAVIOR, not implementation (so refactors don't break tests).
    - Test the PUBLIC interface + edge cases + error paths, not private internals.
    - Prioritize: critical business logic, complex logic, bug-prone areas,
      anything with money/security/data integrity.
    - Don't chase 100% coverage on trivial getters; don't test the framework/libs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TDD (be ready to discuss):
    Test-Driven Development: Red (write a failing test) -> Green (minimal code
    to pass) -> Refactor. Pros: design-first, high coverage, confidence. I use
    it for well-specified logic; for exploratory/uncertain work I prototype
    then add tests. Pragmatic, not dogmatic.

INTERVIEW ANSWER:
    "My strategy follows the test pyramid — many fast, isolated unit tests at
    the base, fewer integration tests that wire components together like an
    endpoint with a real test DB, and a few end-to-end tests for critical user
    flows. I test behavior, not implementation, so refactoring doesn't break
    tests, and I focus on the public interface, edge cases, and error paths.
    The real value is confidence to change code safely and catching regressions
    early. I use TDD for well-specified logic but prototype-then-test for
    exploratory work — pragmatic over dogmatic. As a lead I enforce this with
    coverage gates and CI so it's a team standard, not optional."
'''


# =================================================================================
# SECTION 2: pytest FUNDAMENTALS (discovery, assertions, structure)
# =================================================================================
'''
Why pytest is the standard, and the mechanics you must know.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY pytest (over unittest):
    - Plain 'assert' (with rich introspection on failure) — no self.assertEqual.
    - Powerful FIXTURES (dependency injection for tests) — Section 3.
    - PARAMETRIZE for table-driven tests — Section 4.
    - Huge plugin ecosystem (pytest-cov, pytest-asyncio, pytest-mock, xdist).
    - Less boilerplate; functions, not mandatory classes.

TEST DISCOVERY (conventions):
    - Files: test_*.py or *_test.py.
    - Functions: test_*. Classes: Test* (no __init__).
    - Run: `pytest` (auto-discovers), `pytest -v`, `pytest path::test_name`,
      `pytest -k "keyword"`, `pytest -m marker`.

ANATOMY (Arrange-Act-Assert):
    def test_discount_applies_correctly():
        # Arrange
        cart = Cart(items=[Item(price=100)])
        # Act
        total = cart.total_with_discount(0.10)
        # Assert
        assert total == 90

    AAA structure keeps tests readable: set up, do the thing, check the result.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ASSERTIONS:
    assert result == expected
    assert "error" in response.text
    assert obj.status is None          # identity for None
    import pytest
    with pytest.raises(ValueError, match="invalid"):   # assert it raises
        parse(bad_input)
    assert value == pytest.approx(0.3)  # float comparison (the 0.1+0.2 trap!)

GOOD TEST PRINCIPLES (FIRST):
    Fast, Independent (no shared state/order dependence), Repeatable
    (deterministic — no real time/network/random), Self-validating (pass/fail,
    no manual checking), Timely (written with the code).
    + One logical assertion/concept per test; descriptive test names that say
      WHAT and the EXPECTED outcome.

INTERVIEW POINT:
    "pytest is the standard because of plain-assert introspection, powerful
    fixtures, parametrization, and a rich plugin ecosystem with far less
    boilerplate than unittest. I structure tests Arrange-Act-Assert with
    descriptive names, use pytest.raises for error paths and pytest.approx for
    floats, and follow FIRST — fast, independent, repeatable, self-validating,
    timely. Tests must be deterministic, so no real time, network, or
    randomness without control."
'''


# =================================================================================
# SECTION 3: FIXTURES (the heart of pytest) + conftest + scopes
# =================================================================================
'''
Fixtures are pytest's killer feature and a guaranteed deep-dive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT A FIXTURE IS:
    A reusable SETUP (and teardown) function that pytest INJECTS into a test by
    name. It's dependency injection for tests — the test declares what it needs
    as an argument; pytest provides it.

    import pytest

    @pytest.fixture
    def sample_cart():
        return Cart(items=[Item(price=100), Item(price=50)])

    def test_total(sample_cart):          # pytest injects sample_cart
        assert sample_cart.total() == 150

SETUP + TEARDOWN with yield:
    @pytest.fixture
    def db_session():
        session = create_test_session()   # setup
        yield session                     # provide to the test
        session.rollback(); session.close()  # teardown (runs after test)

    Code before yield = setup; after yield = cleanup (runs even if test fails).
    (Same pattern as FastAPI's yield dependencies — Phase 1.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIXTURE SCOPES (control how often setup runs — perf vs isolation):
    - function (default): runs for EVERY test. Max isolation.
    - class: once per test class.
    - module: once per test file.
    - session: once for the entire test run.
    Use a wider scope for expensive setup (a DB container, a test app) BUT
    beware shared mutable state leaking between tests. Trade-off: speed vs
    isolation. Expensive + read-only -> session; stateful -> function.

conftest.py (shared fixtures):
    Fixtures defined in conftest.py are auto-available to all tests in that
    directory tree — NO import needed. The place for shared fixtures (test
    client, db session, factories). Hierarchical: a conftest per folder.

FIXTURE COMPOSITION:
    Fixtures can use other fixtures (just declare them as args), so you build
    layered setup: db_session -> seeded_db -> test_client(uses seeded_db).

AUTOUSE + FACTORY fixtures:
    @pytest.fixture(autouse=True) — runs for every test without being requested
    (e.g., reset a cache). Factory fixtures return a FUNCTION so tests can
    create customized objects: def make_user(**kw): ...

INTERVIEW ANSWER:
    "Fixtures are pytest's dependency injection for tests — a test declares what
    it needs as an argument and pytest provides the set-up object, with a
    yield-based teardown that runs even on failure, just like FastAPI's yield
    dependencies. Scopes control how often setup runs: function-scope for
    isolation, session-scope for expensive read-only resources like a test
    database, trading isolation for speed. Shared fixtures go in conftest.py so
    they're auto-available without imports, and fixtures compose — a seeded-db
    fixture builds on a session fixture, and a test-client fixture builds on
    that. Factory fixtures return a function so each test can build customized
    data."
'''


# =================================================================================
# SECTION 4: PARAMETRIZATION (table-driven tests)
# =================================================================================
'''
Test many inputs with one test function — DRY testing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PATTERN:
    @pytest.mark.parametrize("value, expected", [
        (2, 4),
        (3, 9),
        (0, 0),
        (-2, 4),
    ])
    def test_square(value, expected):
        assert square(value) == expected
    -> pytest runs this as FOUR separate tests, each reported individually.
       One fails, you see exactly which input.

WHY IT MATTERS:
    - Covers many cases (happy path, edges, errors) without copy-paste.
    - Each case is a distinct test (clear failure reporting).
    - Encourages thinking in input/output tables -> better edge coverage.

EDGE CASES TO ALWAYS PARAMETRIZE:
    Empty, zero, negative, max/boundary, None, duplicates, very large, invalid
    types -> the bug-prone inputs.

ADVANCED:
    - pytest.param(..., id="descriptive-name") to name a case.
    - pytest.param(..., marks=pytest.mark.xfail) for known-failing cases.
    - Stack two @parametrize decorators for a cartesian product.
    - Parametrize FIXTURES (params=[...]) to run a whole suite against multiple
      backends/configs.

INTERVIEW POINT:
    "I use parametrize for table-driven tests — one function, many input/output
    cases — so each case reports as its own test and I get clear failure
    isolation. It pushes me to enumerate edge cases like empty, zero, negative,
    boundary, and None instead of copy-pasting test bodies. I can name cases
    with pytest.param ids and even parametrize fixtures to run the same suite
    against multiple backends."
'''


# =================================================================================
# SECTION 5: MOCKING & PATCHING (isolate the unit)
# =================================================================================
'''
To unit-test in isolation, replace external dependencies (DB, API, LLM, time,
randomness) with controllable fakes. A guaranteed interview topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY MOCK:
    A UNIT test must test YOUR code, not the network/DB/LLM. Mocking removes
    slow, flaky, costly, or non-deterministic dependencies so the test is fast,
    reliable, and focused. (Also lets you simulate errors easily.)

THE TYPES OF TEST DOUBLES (know the vocabulary):
    - STUB: returns canned data ("when called, return this").
    - MOCK: a stub that ALSO records calls so you can ASSERT it was called
      (with what args, how many times).
    - FAKE: a working lightweight implementation (in-memory DB).
    - SPY: wraps a real object but records interactions.
    - DUMMY: a placeholder passed but not used.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

unittest.mock / pytest-mock:
    from unittest.mock import MagicMock, patch

    # Replace a dependency object:
    def test_notify(mocker):                      # pytest-mock's 'mocker'
        email = mocker.Mock()
        service = Notifier(email_client=email)
        service.notify("hi")
        email.send.assert_called_once_with("hi")  # assert interaction

    # Patch where it's USED (critical gotcha):
    @patch("myapp.orders.payment_gateway.charge")
    def test_checkout(mock_charge):
        mock_charge.return_value = {"status": "ok"}
        ...
        mock_charge.assert_called_once()

THE #1 MOCKING GOTCHA — "patch where it's used, not where it's defined":
    If module 'orders' does `from payments import charge`, you patch
    "orders.charge" (where orders looked it up), NOT "payments.charge".
    Getting this wrong = the mock silently doesn't apply.

MORE MOCK MECHANICS:
    - return_value: what the mock returns when called.
    - side_effect: a function, exception, or list — raise errors or return a
      sequence (mock.side_effect = TimeoutError to test error handling).
    - assert_called_once(), assert_called_with(args), call_count, call_args.
    - AsyncMock for async functions (Section 6).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN NOT TO OVER-MOCK (the senior nuance):
    Over-mocking couples tests to implementation -> brittle, and you stop
    testing real behavior. Prefer FAKES (in-memory) or real test instances for
    integration-level confidence. Mock only the EXPENSIVE/EXTERNAL boundaries
    (third-party APIs, LLMs, payment, email). "Mock at the edges, use real
    objects within."

INTERVIEW ANSWER:
    "To unit-test in isolation I replace external dependencies — DB, third-party
    APIs, the LLM, time, randomness — with test doubles so the test is fast,
    deterministic, and focused on my code. I use unittest.mock or pytest-mock's
    mocker, setting return_value for canned responses and side_effect to
    simulate errors, and I assert interactions with assert_called_once_with.
    The classic gotcha is patching where the name is USED, not where it's
    defined. But I'm careful not to over-mock — that couples tests to
    implementation; I mock at the external edges and use real or in-memory
    fakes within, which gives more confidence."
'''


# =================================================================================
# SECTION 6: TESTING ASYNC CODE + FastAPI (ties to Phases 1-2)
# =================================================================================
'''
The JD's stack is async FastAPI — you must test it. Connects Phases 1, 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TESTING ASYNC FUNCTIONS:
    import pytest

    @pytest.mark.asyncio                  # from pytest-asyncio
    async def test_fetch():
        result = await fetch_data()
        assert result == expected

    - pytest-asyncio (or anyio) lets pytest run async test functions.
    - Mock async deps with AsyncMock (a regular Mock isn't awaitable):
        mocker.patch("module.client.get", new_callable=AsyncMock,
                     return_value={"ok": True})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TESTING FastAPI ENDPOINTS (Phase 1):
    from fastapi.testclient import TestClient
    from myapp import app
    client = TestClient(app)              # sync client, drives the async app

    def test_create_item():
        resp = client.post("/items", json={"name": "Widget", "price": 9.99})
        assert resp.status_code == 201
        assert resp.json()["name"] == "Widget"

    Async client (for true async tests): httpx.AsyncClient(app=app).

THE KILLER FEATURE — dependency_overrides (Phase 1 + Phase 5 meet):
    app.dependency_overrides[get_db] = lambda: fake_test_session
    -> Inject a fake DB / mocked LLM client into the app WITHOUT touching route
       code. This is WHY dependency injection matters for testability. Clear it
       in teardown.

LEVELS FOR A FastAPI APP:
    - UNIT: test service/logic functions with mocked deps.
    - INTEGRATION: TestClient + a real test DB (transaction rolled back per test
      via a fixture) — tests routing + validation + DB together.
    - Use a fixture for the client + a fixture for the test DB; override the DB
      dependency to point at the test DB.

INTERVIEW ANSWER:
    "For async code I use pytest-asyncio so pytest can run async test functions,
    and AsyncMock for async dependencies since a normal Mock isn't awaitable.
    For FastAPI I use the TestClient — or httpx AsyncClient for true async — to
    hit endpoints and assert status and body. The key enabler is
    dependency_overrides: I swap the real DB or LLM client for a fake or test
    instance without changing route code, which is exactly why I lean on
    dependency injection. Integration tests run against a real test DB with each
    test wrapped in a transaction that rolls back, so they're isolated and
    repeatable."
'''


# =================================================================================
# SECTION 7: TESTING AI / LLM CODE (the hard part — non-determinism)
# =================================================================================
'''
Your differentiator. LLMs are non-deterministic — standard assertions don't
work. The JD is a GenAI role; this WILL come up.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CORE PROBLEM:
    assert llm_answer == "exact string" FAILS — the model phrases differently
    each time. You can't unit-test an LLM's output with equality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE STRATEGY — SEPARATE DETERMINISTIC CODE FROM THE LLM:

    1. UNIT-TEST your DETERMINISTIC code by MOCKING the LLM:
       - Chunking, parsing, routing logic, prompt construction, retrieval
         filtering, response handling — these are normal code. Mock the LLM
         call (return canned output) and test YOUR logic deterministically.
       - Test that you BUILD the right prompt, PARSE the response correctly,
         HANDLE errors/timeouts, ROUTE to the right tool, ENFORCE the gate.
       - This is 80% of your code and is fully unit-testable.

    2. EVALUATE the LLM OUTPUT separately (not equality — quality metrics):
       - This is EVAL, not unit testing. Golden dataset + metrics.
       - Semantic similarity (embedding/BERTScore) instead of exact match.
       - LLM-as-judge for quality (faithfulness, relevance) — RAGAS.
       - Assert it CONTAINS key facts / matches a schema, not exact wording.
       - Run as a separate eval suite (slower, may cost tokens), not in fast CI.

    3. TEST STRUCTURE not prose: if you use structured output (Pydantic), you
       CAN assert the schema/fields deterministically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRACTICAL TECHNIQUES:
    - Set temperature=0 for more deterministic outputs in tests (still not
      guaranteed identical, but more stable).
    - Mock the LLM client (AsyncMock) returning fixed responses for unit tests.
    - Record/replay (VCR-style): record real LLM responses once, replay in tests
      — fast, free, deterministic, but can go stale.
    - Contract-test the LLM I/O: assert the output PARSES into your schema.
    - Eval gates in CI for prompt/model changes (regression on a golden set) —
      Lessons 17/21/25.

INTERVIEW ANSWER:
    "LLM output is non-deterministic, so equality assertions don't work. My
    approach is to separate the deterministic code from the model. Eighty
    percent of an AI app — chunking, prompt construction, response parsing,
    routing, the retrieval gate, error handling — is normal code, so I mock the
    LLM with a canned response and unit-test that logic deterministically:
    does it build the right prompt, parse the response, handle a timeout, route
    correctly. The model's actual output I evaluate separately as quality, not
    equality — a golden dataset with semantic similarity, schema/contains
    checks, and LLM-as-judge metrics like faithfulness via RAGAS, run as a
    separate eval suite outside fast CI. If I use structured output I can assert
    the schema deterministically, and I set temperature to zero in tests for
    stability. For prompt or model changes I gate on a regression eval set."
'''


# =================================================================================
# SECTION 8: COVERAGE, CI, and TEST QUALITY
# =================================================================================
'''
A lead drives quality systemically, not test-by-test.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COVERAGE (pytest-cov):
    pytest --cov=myapp --cov-report=term-missing
    - LINE coverage: % of lines executed by tests.
    - BRANCH coverage: % of decision branches taken (stronger — catches the
      untested else path). Prefer branch coverage.

    THE COVERAGE TRAP (say this — it's a senior signal):
    "Coverage measures what's EXECUTED, not what's VERIFIED. You can have 100%
    coverage with zero assertions. High coverage is necessary, not sufficient.
    I set a reasonable gate (e.g., 80%) to catch untested code, but I care
    about whether tests actually ASSERT meaningful behavior, not the number.
    100% is usually not worth chasing — diminishing returns on trivial code."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CI INTEGRATION (the gate):
    On every PR (GitHub Actions / GitLab CI):
    - Run linters (ruff/flake8) + formatter check (black) + type check (mypy).
    - Run the test suite; fail the PR if any test fails.
    - Enforce a coverage threshold.
    - (For AI) optionally run an eval suite on prompt/model changes.
    -> No code merges without passing. Standards enforced automatically, not by
       nagging in reviews.

    PARALLELIZE the suite (pytest-xdist: `pytest -n auto`) to keep CI fast.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FLAKY TESTS (a real lead concern):
    Tests that pass/fail non-deterministically destroy trust in the suite.
    Causes: real time/dates, real network, randomness, test-order dependence,
    shared state, async race conditions. FIX: control time (freezegun), mock
    network, seed randomness, isolate state (function-scope fixtures), make
    tests order-independent. "A flaky test is worse than no test — it trains
    the team to ignore failures."

OTHER QUALITY SIGNALS:
    - Mutation testing (mutmut) — checks if tests actually CATCH bugs by
      mutating code and seeing if tests fail. The real measure of test quality.
    - Fast suite -> people run it. Slow suite -> people skip it.

INTERVIEW ANSWER:
    "I use pytest-cov with branch coverage and a sensible gate like 80% in CI,
    but I'm clear that coverage measures execution, not verification — you can
    hit 100% with no assertions, so I care that tests assert meaningful
    behavior, not the number. In CI on every PR I run linters, formatter and
    type checks, the test suite, and the coverage gate, so nothing merges
    without passing — standards are automated, not nagged. I keep the suite
    fast with pytest-xdist so people actually run it, and I treat flaky tests
    as critical to fix because a flaky suite trains the team to ignore failures.
    For measuring test quality itself, mutation testing checks whether tests
    actually catch bugs."
'''


# =================================================================================
# SECTION 9: PYTHON PACKAGING & DEPENDENCY MANAGEMENT (pip/venv/Poetry/uv)
# =================================================================================
'''
The JD names pip, venv/virtualenv, Poetry, AND uv explicitly. Know the
landscape, the problems each solves, and the trade-offs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PROBLEMS THESE TOOLS SOLVE:
    1. ISOLATION — different projects need different package versions; a global
       install causes conflicts. -> VIRTUAL ENVIRONMENTS.
    2. REPRODUCIBILITY — "works on my machine" — everyone needs the SAME
       versions. -> LOCK FILES (exact pinned versions + hashes).
    3. DEPENDENCY RESOLUTION — find a set of versions that satisfy all
       constraints (and sub-dependencies). -> RESOLVERS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE TOOLS:

    venv / virtualenv — ISOLATION only:
        python -m venv .venv ; source .venv/bin/activate (or .venv\Scripts\activate)
        Creates an isolated interpreter + site-packages. venv is stdlib;
        virtualenv is the older faster third-party one. Solves isolation, NOT
        locking/resolution.

    pip — the INSTALLER:
        pip install x ; pip freeze > requirements.txt
        Installs packages. requirements.txt is a flat list — NOT a true lock
        file (pip freeze pins versions but not hashes, and mixes direct +
        transitive deps). Classic, universal, but manual for reproducibility.
        (pip-tools improves this: requirements.in -> compiled requirements.txt.)

    Poetry — all-in-one project + dependency manager:
        pyproject.toml (declares deps) + poetry.lock (exact, hashed, resolved).
        Manages venv, dependency resolution, lock file, building, publishing.
        Separates direct vs transitive; dev vs prod groups. The mature standard
        for serious projects. Slower resolver historically.

    uv — the modern, FAST tool (Rust, by Astral):
        A drop-in, extremely fast replacement that does pip + venv + resolution
        + locking (uv.lock) — often 10-100x faster than pip/Poetry. Rising
        rapidly as the new standard. (Your own project uses uv.lock.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE LANDSCAPE TABLE:
    TOOL         ISOLATION  INSTALL  RESOLVE  LOCK     SPEED   NOTE
    venv         yes        no       no       no       -       stdlib isolation
    pip          no         yes      basic    weak*    medium  universal installer
    pip-tools    no         via pip  yes      yes      medium  adds real locking
    Poetry       yes        yes      yes      yes      slower  mature all-in-one
    uv           yes        yes      yes      yes      fastest modern, Rust

    *requirements.txt isn't a true lock file (no hashes, mixes direct/transitive).

LOCK FILE — WHY IT MATTERS (the senior point):
    A lock file pins EXACT versions (+ hashes) of ALL deps including transitive
    ones, so every machine/CI/prod installs the IDENTICAL set -> reproducible
    builds + security (hash verification). pyproject.toml = what you WANT;
    lock file = what you GOT. Commit the lock file.

INTERVIEW ANSWER:
    "These tools solve three problems: isolation, reproducibility, and
    dependency resolution. venv gives isolation — an isolated interpreter per
    project. pip installs packages, but requirements.txt isn't a true lock file
    since it lacks hashes and mixes direct and transitive deps, so I'd use
    pip-tools or better. Poetry is the mature all-in-one — pyproject.toml for
    declared deps plus a real poetry.lock with exact, hashed, resolved
    versions, managing the venv, dev/prod groups, and building. uv is the
    modern Rust-based tool that does all of that 10 to 100 times faster and is
    becoming the standard — my own project uses uv. The key concept is the lock
    file: pyproject.toml is what you want, the lock file is what you got, pinned
    exactly so every environment is reproducible. I always commit it."
'''


# =================================================================================
# SECTION 10: PROJECT STRUCTURE & ENVIRONMENTS
# =================================================================================
'''
A lead sets up the project for maintainability. Round out the tooling picture.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A CLEAN PROJECT LAYOUT (src layout — the recommended one):
    myproject/
        pyproject.toml          # deps, build config, tool config (one place)
        uv.lock / poetry.lock   # pinned, committed
        README.md
        .gitignore
        src/
            myapp/
                __init__.py
                main.py
                api/            # FastAPI routers
                services/       # business logic
                models/         # Pydantic + DB models
                core/           # config, settings
        tests/
            conftest.py         # shared fixtures
            unit/
            integration/

    "src layout" (package under src/) prevents accidentally importing the local
    package instead of the installed one — catches packaging bugs early.

pyproject.toml — THE MODERN STANDARD:
    One file for build system, dependencies, AND tool config (ruff, black,
    mypy, pytest, coverage). Replaces the old scatter of setup.py + setup.cfg +
    requirements.txt + per-tool config files. PEP 517/518/621.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE QUALITY TOOLCHAIN (what a lead standardizes on a team):
    - FORMATTER: black (or ruff format) — auto-format, no style debates.
    - LINTER: ruff (fast, replaces flake8 + isort + more) or flake8.
    - TYPE CHECKER: mypy (or pyright) — static type checking on type hints.
    - PRE-COMMIT hooks: run format/lint/type before every commit locally.
    - CI: re-run all of the above + tests as gates.
    "Automate style and lint so humans review logic, not formatting."

ENVIRONMENT MANAGEMENT (config, not packages):
    - 12-factor: config via ENVIRONMENT VARIABLES, not hardcoded.
    - .env files for local dev (never committed — secrets); pydantic-settings
      to load + validate (Phase 1).
    - Separate config per environment (dev/staging/prod); secrets from a vault
      in prod, never in code or the repo.
    - Python version pinning (.python-version / pyproject `requires-python`).

INTERVIEW POINT:
    "I use a src layout so tests import the installed package, not the local
    folder, which catches packaging bugs. pyproject.toml centralizes
    dependencies and all tool config in one file — the modern PEP standard. On
    a team I standardize the toolchain: black or ruff for formatting, ruff for
    linting, mypy for type checking, pre-commit hooks locally and the same
    checks as CI gates, so style is automated and reviews focus on logic. Config
    is twelve-factor — environment variables loaded and validated with
    pydantic-settings, .env for local, a vault for prod secrets, never in the
    repo."
'''


# =================================================================================
# SECTION 11: INTERVIEW Q&A (lead-level)
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. Unit vs integration vs functional test?
A:  "Unit = one unit in isolation, deps mocked, fast. Integration = components
    together, e.g. endpoint + real DB. Functional/E2E = a full user flow.
    Pyramid: many unit, fewer integration, fewest E2E."

Q2. What's a fixture?
A:  "Reusable setup pytest injects into a test by name — DI for tests. yield
    gives teardown that runs even on failure. Scopes (function/session) trade
    isolation for speed; shared ones live in conftest.py."

Q3. How do you test code that calls an external API/DB?
A:  "Mock the dependency for unit tests — return canned data with return_value,
    simulate errors with side_effect, assert interactions. Patch where it's
    used, not where it's defined. Integration tests use a real test DB."

Q4. How do you test async FastAPI code?
A:  "pytest-asyncio for async tests, AsyncMock for async deps, TestClient or
    httpx AsyncClient for endpoints, and dependency_overrides to inject a test
    DB or mock without changing route code."

Q5. How do you test non-deterministic LLM output?
A:  "Separate deterministic code from the model — mock the LLM and unit-test
    prompt building, parsing, routing, error handling. Evaluate the model
    output separately as quality, not equality: golden set, semantic
    similarity, schema/contains checks, LLM-as-judge via RAGAS."

Q6. Is 100% coverage the goal?
A:  "No — coverage measures execution, not verification; you can hit 100% with
    no assertions. I gate around 80% to catch untested code but care that tests
    assert meaningful behavior. Mutation testing actually measures test quality."

Q7. What's the patch-where-it's-used gotcha?
A:  "You patch the name in the module that LOOKED IT UP, not where it's defined.
    If orders.py does 'from payments import charge', patch 'orders.charge'."

Q8. requirements.txt vs Poetry vs uv?
A:  "requirements.txt + pip is simple but not a true lock file. Poetry is the
    mature all-in-one with a real lock file and dependency groups. uv does the
    same far faster in Rust and is becoming the standard. The point is a
    committed lock file for reproducible builds."

Q9. Why a lock file?
A:  "It pins exact versions and hashes of all deps including transitive, so
    every environment installs identically — reproducible builds plus hash
    security. pyproject.toml is what you want; the lock is what you got."

Q10. How do you keep a test suite trustworthy?
A:  "Fast (xdist), deterministic (no real time/network/randomness), isolated
    (function-scope fixtures), and zero tolerance for flaky tests — a flaky
    test trains the team to ignore failures."

Q11. How do you enforce quality on a team?
A:  "Automate: ruff/black/mypy + pre-commit hooks + CI gates (tests pass +
    coverage threshold). Nothing merges without passing, so reviews focus on
    logic and design, not style."
'''


# =================================================================================
# SECTION 12: GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. TEST PYRAMID: many unit, fewer integration, fewest E2E. Inverted = slow/flaky.
2. TEST BEHAVIOR, NOT IMPLEMENTATION — so refactors don't break tests.
3. FIXTURES = DI FOR TESTS; yield for teardown; scope = isolation vs speed;
   conftest for shared. The heart of pytest.
4. PARAMETRIZE for table-driven edge-case coverage with clear failure isolation.
5. MOCK AT THE EDGES, NOT EVERYWHERE — patch where USED; don't over-mock.
6. ASYNC/FASTAPI: pytest-asyncio + AsyncMock + TestClient + dependency_overrides.
7. LLM CODE: unit-test the deterministic 80% (mock the LLM); EVALUATE output
   as quality, not equality (golden set + RAGAS), in a separate suite.
8. COVERAGE MEASURES EXECUTION, NOT VERIFICATION — gate ~80%, care about asserts.
9. FLAKY TESTS ARE WORSE THAN NO TESTS — deterministic + isolated + fast.
10. TOOLING: venv (isolation), pip (install), Poetry/uv (resolve+lock); COMMIT
    THE LOCK FILE for reproducibility; pyproject.toml centralizes everything;
    automate format/lint/type in pre-commit + CI.

ONE-LINE CHEAT SHEET:
    Pyramid: unit(many)/integration(some)/E2E(few); test behavior not impl.
    pytest: assert + fixtures(yield, scope, conftest) + parametrize + raises/approx.
    Mock: return_value/side_effect/assert_called; patch where USED; mock edges only.
    Async/FastAPI: pytest-asyncio + AsyncMock + TestClient + dependency_overrides.
    LLM: mock LLM -> unit-test logic; eval output (semantic/judge/RAGAS) separately.
    Coverage: branch, ~80% gate, executes != verifies; kill flaky tests.
    Tooling: venv+pip | Poetry | uv(fast); lock file = reproducible; pyproject.toml.
    Enforce: ruff+black+mypy + pre-commit + CI gates.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 5 — TESTING (pytest) & PYTHON TOOLING")
    print("=" * 70)
    print()
    print("TEST PYRAMID: unit(MANY) -> integration(SOME) -> E2E(FEW)")
    print("  Test BEHAVIOR not implementation. FIRST: fast/independent/")
    print("  repeatable/self-validating/timely.")
    print()
    print("pytest CORE:")
    print("  fixtures (DI for tests; yield=teardown; scope; conftest)")
    print("  parametrize (table-driven) | pytest.raises | pytest.approx")
    print("  mock: return_value/side_effect/assert_called; PATCH WHERE USED")
    print()
    print("ASYNC/FASTAPI: pytest-asyncio + AsyncMock + TestClient +")
    print("               dependency_overrides (inject test DB/mock)")
    print()
    print("LLM TESTING: mock the LLM -> unit-test deterministic logic;")
    print("  evaluate output as QUALITY not equality (golden set + RAGAS).")
    print()
    print("COVERAGE: executes != verifies; gate ~80%; kill flaky tests.")
    print()
    print("TOOLING: venv(isolate) | pip(install) | Poetry/uv(resolve+LOCK)")
    print("  Commit the lock file = reproducible builds. pyproject.toml = one config.")
    print("  Automate: ruff + black + mypy + pre-commit + CI gates.")
    print()
    print("=" * 70)
    print("Remaining (low priority): Phase 9 (Cloud/DevOps), Phase 10 (Mock).")
    print("=" * 70)
