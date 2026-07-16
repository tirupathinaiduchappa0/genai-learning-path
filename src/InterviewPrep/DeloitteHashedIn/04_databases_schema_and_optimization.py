"""
===================================================================================
PHASE 4 — DATABASES: SCHEMA DESIGN & OPTIMIZATION (HashedIn by Deloitte)
===================================================================================

WHY THIS IS PHASE 4:
    The JD demands: "Advanced skills in database schema design and optimization,
    including relational (PostgreSQL, MySQL) and NoSQL databases, with a focus
    on scalability, normalization, and performance" and "reading and writing
    SQL queries, schema design, and database optimization."

    For a LEAD, DB design is foundational — you OWN the schema that the whole
    system depends on. A bad schema or a missing index can sink performance no
    matter how good the code is.

DEPTH LEVEL: Senior/Lead. Design + optimization + trade-offs.

SECTIONS:
    1.  Relational vs NoSQL — When to Use Which
    2.  Schema Design & Normalization (1NF-3NF, when to denormalize)
    3.  Indexing — The #1 Performance Lever
    4.  Query Optimization & EXPLAIN
    5.  The N+1 Problem (ORM killer)
    6.  Transactions & ACID, Isolation Levels
    7.  Connection Pooling & Scaling Patterns
    8.  SQL You Must Be Able to Write (JOINs, aggregates, window functions)
    9.  NoSQL Deep Dive (document, key-value, the CAP theorem)
    10. ORM vs Raw SQL (SQLAlchemy) + Migrations
    11. Databases for AI (pgvector, vector DBs)
    12. Interview Q&A (senior-level)
    13. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: RELATIONAL vs NoSQL — When to Use Which
# =================================================================================
'''
A lead must justify the database choice with trade-offs — the JD's core theme.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELATIONAL (SQL — PostgreSQL, MySQL):
    Data in TABLES with rows/columns, fixed SCHEMA, related by FOREIGN KEYS.
    Strengths: ACID transactions, strong consistency, complex JOINs/queries,
               data integrity (constraints), mature tooling.
    Weakness: rigid schema (migrations to change), horizontal scaling harder.
    Use for: structured data with relationships, financial/transactional
             systems, anything needing strong consistency. (The default choice.)

NoSQL (4 families):
    1. DOCUMENT (MongoDB) — JSON-like documents, flexible schema.
       Use: content, catalogs, semi-structured data, rapid iteration.
    2. KEY-VALUE (Redis, DynamoDB) — simple key->value, blazing fast.
       Use: caching, sessions, leaderboards, rate limiting.
    3. WIDE-COLUMN (Cassandra, HBase) — rows with dynamic columns, write-heavy.
       Use: time-series, huge write throughput, IoT, logging.
    4. GRAPH (Neo4j) — nodes + edges.
       Use: relationships-first data — social, recommendations, knowledge graphs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DECISION (the senior framing):
    SQL when: relationships matter, you need ACID/consistency, complex queries,
              structured data. (Most business apps.)
    NoSQL when: massive scale/throughput, flexible/evolving schema,
                denormalized access patterns, specific shapes (cache, graph,
                time-series).

    REALITY: most systems are POLYGLOT — PostgreSQL as the source of truth +
    Redis for caching + a vector DB for embeddings. Pick per workload, not
    dogmatically.

    THE TRAP TO AVOID: "NoSQL scales, SQL doesn't" is wrong. PostgreSQL scales
    massively (read replicas, partitioning, sharding). NoSQL trades
    consistency/joins for scale and flexibility — it's a trade-off, not "better."

INTERVIEW ANSWER:
    "I default to a relational database like PostgreSQL because most business
    data has relationships and benefits from ACID guarantees, constraints, and
    complex queries. I reach for NoSQL for specific shapes — document stores
    like MongoDB for flexible semi-structured data, key-value like Redis for
    caching and sessions, wide-column like Cassandra for write-heavy
    time-series, graph like Neo4j for relationship-first data. In practice
    systems are polyglot: Postgres as the source of truth, Redis for cache, a
    vector DB for embeddings. The choice is per workload and access pattern,
    weighing consistency and query needs against scale and schema flexibility."
'''


# =================================================================================
# SECTION 2: SCHEMA DESIGN & NORMALIZATION (1NF-3NF, when to denormalize)
# =================================================================================
'''
The JD explicitly says "normalization." Know the forms AND when to break them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NORMALIZATION = organizing data to REDUCE REDUNDANCY and prevent anomalies
(update/insert/delete anomalies). The normal forms:

    1NF (First Normal Form):
        - Atomic values (no lists/repeating groups in a column).
        - Each row unique (a primary key).
        BAD: a "phones" column with "555-1, 555-2". GOOD: a separate phones row each.

    2NF (Second Normal Form):
        - Is in 1NF AND every non-key column depends on the WHOLE primary key
          (no partial dependency on part of a composite key).
        Fix: split out columns that depend on only part of a composite key.

    3NF (Third Normal Form):
        - Is in 2NF AND no TRANSITIVE dependency (non-key columns depend ONLY
          on the key, not on other non-key columns).
        BAD: orders table with (customer_id, customer_city) — city depends on
        customer, not the order. GOOD: city lives in the customers table.

    (BCNF, 4NF exist but 3NF is the practical target for most designs.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO DENORMALIZE (the senior nuance):
    Normalization reduces redundancy but adds JOINs (which cost performance).
    DENORMALIZE deliberately for read performance when:
    - Read-heavy workloads where JOINs are too slow.
    - Reporting/analytics (star schema in data warehouses).
    - You can tolerate some redundancy + the update cost of keeping copies sync.

    RULE: "Normalize until it hurts, denormalize until it works."
    Normalize first (correctness, integrity), denormalize selectively with
    evidence (measured slow queries), not prematurely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELATIONSHIPS & KEYS:
    - PRIMARY KEY: unique row identifier (prefer surrogate keys — auto int/UUID).
    - FOREIGN KEY: enforces referential integrity between tables.
    - One-to-many: FK on the "many" side (orders.customer_id -> customers.id).
    - Many-to-many: a JUNCTION/JOIN table (student_courses with student_id +
      course_id).
    - One-to-one: FK with a unique constraint.

INTERVIEW ANSWER:
    "Normalization organizes data to remove redundancy and prevent update
    anomalies. 1NF is atomic values with a key; 2NF removes partial
    dependencies on a composite key; 3NF removes transitive dependencies so
    non-key columns depend only on the key. I design to 3NF for integrity, but
    I denormalize deliberately for read-heavy paths or reporting where joins
    are too slow — 'normalize until it hurts, denormalize until it works,'
    and only with measured evidence, never prematurely. Relationships use
    foreign keys: FK on the many-side for one-to-many, a junction table for
    many-to-many."
'''


# =================================================================================
# SECTION 3: INDEXING — The #1 Performance Lever
# =================================================================================
'''
Indexing is the highest-impact optimization and a guaranteed deep-dive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT AN INDEX IS:
    A separate data structure (usually a B-TREE) that lets the DB find rows
    WITHOUT scanning the whole table. Like a book's index — jump to the page
    instead of reading every page.

    Without index: full table scan = O(n).
    With B-tree index: O(log n) lookup.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT TO INDEX:
    - PRIMARY KEYS (auto-indexed).
    - FOREIGN KEYS (often NOT auto-indexed — index them; JOINs use them).
    - Columns in WHERE, JOIN, ORDER BY, GROUP BY clauses.
    - Columns with high SELECTIVITY (many distinct values — email, not gender).

WHAT NOT TO INDEX (the cost):
    Indexes speed READS but SLOW WRITES (every insert/update must update the
    index) and consume storage. Don't index:
    - Low-selectivity columns (boolean, status with 2 values) — scan is cheaper.
    - Columns rarely queried.
    - Over-indexing: too many indexes degrade write performance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INDEX TYPES:
    - B-TREE (default): range queries, equality, sorting. The workhorse.
    - HASH: equality only (= ), faster for exact match, no ranges.
    - COMPOSITE (multi-column): index on (a, b, c). LEFTMOST PREFIX rule —
      it helps queries on a, (a,b), (a,b,c) but NOT on b alone or c alone.
      Order columns by selectivity / query pattern.
    - PARTIAL: index only rows matching a condition (WHERE active=true) — smaller.
    - UNIQUE: enforces uniqueness + speeds lookups.
    - GIN/GiST (Postgres): full-text search, JSONB, arrays, and pgvector.
    - COVERING index: includes all columns a query needs, so the DB never
      touches the table ("index-only scan").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE COMPOSITE INDEX LEFTMOST-PREFIX RULE (commonly tested):
    Index on (last_name, first_name):
    - WHERE last_name = 'X'                 -> USES index
    - WHERE last_name = 'X' AND first_name='Y' -> USES index
    - WHERE first_name = 'Y'                -> does NOT use index (not leftmost)

INTERVIEW ANSWER:
    "An index is usually a B-tree that turns a full table scan into a
    logarithmic lookup. I index primary keys, foreign keys, and columns used
    in WHERE, JOIN, ORDER BY, and GROUP BY, favoring high-selectivity columns.
    But indexes aren't free — they speed reads and slow writes and cost
    storage, so I don't index low-selectivity columns or over-index. For
    multi-column queries I use composite indexes and respect the leftmost-prefix
    rule — an index on (last_name, first_name) helps queries on last_name but
    not first_name alone. In Postgres I use GIN indexes for JSONB and full-text,
    and partial or covering indexes to optimize specific hot queries."
'''


# =================================================================================
# SECTION 4: QUERY OPTIMIZATION & EXPLAIN
# =================================================================================
'''
"A query is slow — how do you fix it?" — a guaranteed lead question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DIAGNOSTIC TOOL — EXPLAIN / EXPLAIN ANALYZE:
    EXPLAIN ANALYZE SELECT ... shows the QUERY PLAN: how the DB will execute
    it, what it actually did, timing, and rows processed.

    What to look for:
    - "Seq Scan" (full table scan) on a big table -> likely needs an index.
    - "Index Scan" / "Index Only Scan" -> good, using an index.
    - High "rows" estimates vs actual -> stale statistics (run ANALYZE).
    - Expensive JOINs (nested loop on big tables) -> indexing / rewrite.
    - Sort/hash spilling to disk -> memory tuning or better index.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTIMIZATION CHECKLIST (in order):
    1. ADD/FIX INDEXES on WHERE/JOIN/ORDER BY columns (biggest win usually).
    2. SELECT ONLY NEEDED COLUMNS — avoid SELECT * (less I/O, enables covering
       indexes).
    3. AVOID N+1 (Section 5) — batch/JOIN instead of per-row queries.
    4. LIMIT result sets — paginate; never fetch unbounded rows.
    5. REWRITE the query — avoid functions on indexed columns in WHERE
       (WHERE LOWER(email)=... defeats the index unless you have a functional
       index), avoid leading-wildcard LIKE '%x' (can't use a normal index).
    6. DENORMALIZE or use a MATERIALIZED VIEW for expensive repeated aggregates.
    7. UPDATE STATISTICS (ANALYZE) so the planner makes good choices.
    8. PARTITION large tables (by date/range) so queries touch fewer rows.
    9. CACHE hot results (Redis) to skip the DB entirely.
    10. CONNECTION POOLING (Section 7) — reduce connection overhead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON QUERY ANTI-PATTERNS:
    - SELECT *  -> fetch only needed columns.
    - Function on indexed column in WHERE -> defeats the index.
    - LIKE '%term'  -> leading wildcard can't use a B-tree index.
    - OR across different columns -> sometimes slower than UNION.
    - Implicit type casts in WHERE -> can skip the index.
    - Fetching all rows to filter in app code -> filter in SQL.

INTERVIEW ANSWER:
    "First I run EXPLAIN ANALYZE to see the actual query plan. A sequential
    scan on a large table in a WHERE or JOIN is the usual smoking gun — that
    means a missing index. I check that WHERE, JOIN, and ORDER BY columns are
    indexed, select only needed columns instead of SELECT *, eliminate N+1
    queries, and make sure I'm not applying a function to an indexed column or
    using a leading-wildcard LIKE, both of which defeat indexes. For repeated
    expensive aggregates I use a materialized view, partition very large
    tables, and cache hot results in Redis. And I keep table statistics fresh
    so the planner chooses well."
'''


# =================================================================================
# SECTION 5: THE N+1 PROBLEM (ORM killer)
# =================================================================================
'''
The most common ORM performance bug. Naming and fixing it = instant credibility.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IT IS:
    You fetch N parent rows, then run 1 MORE query PER parent to get its
    related data -> 1 + N queries instead of 1 or 2.

    BAD (N+1):
        authors = session.query(Author).all()       # 1 query
        for author in authors:                        # N authors
            print(author.books)                       # 1 query EACH = N queries
        Total: 1 + N queries. For 1000 authors = 1001 queries. Disaster.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE FIX — EAGER LOADING (fetch related data up front):
    SQLAlchemy:
        # JOIN-based eager load (one query with a JOIN):
        authors = session.query(Author).options(joinedload(Author.books)).all()
        # OR a second batched query (IN clause) — often better for one-to-many:
        authors = session.query(Author).options(selectinload(Author.books)).all()
    -> 1 or 2 queries total instead of 1 + N.

    Django ORM equivalents:
        select_related()  -> JOIN (for foreign-key / to-one).
        prefetch_related() -> separate batched query (for many-to-many / reverse FK).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY IT HAPPENS:
    Lazy loading (the ORM default) fetches related objects only when accessed —
    convenient but triggers a query each time inside a loop. The fix is to tell
    the ORM to eager-load when you know you'll need the related data.

HOW TO DETECT:
    - Query logging / SQL echo shows a burst of near-identical queries.
    - APM/tracing (e.g., a request firing 500 DB queries).
    - django-debug-toolbar, SQLAlchemy echo=True.

INTERVIEW ANSWER:
    "The N+1 problem is when you fetch N parent rows and then run one query per
    parent for its related data — 1 + N queries. It usually comes from lazy
    loading inside a loop. The fix is eager loading: in SQLAlchemy, joinedload
    for a JOIN or selectinload for a batched IN query; in Django, select_related
    for to-one and prefetch_related for to-many. I detect it with SQL logging
    or tracing showing a burst of repetitive queries, and it's one of the first
    things I check when an endpoint is slow."
'''


# =================================================================================
# SECTION 6: TRANSACTIONS & ACID, Isolation Levels
# =================================================================================
'''
Critical for data integrity — and a favorite deep-dive for senior roles.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACID (what a transaction guarantees):
    ATOMICITY    — all-or-nothing. If any step fails, the whole transaction
                   rolls back. (Transfer money: debit + credit both or neither.)
    CONSISTENCY  — the DB moves from one valid state to another (constraints hold).
    ISOLATION    — concurrent transactions don't corrupt each other (see levels).
    DURABILITY   — once committed, it survives crashes (written to disk/WAL).

    A TRANSACTION groups multiple statements so they succeed or fail together:
        BEGIN; UPDATE ... ; UPDATE ... ; COMMIT;   (or ROLLBACK on error)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCURRENCY PROBLEMS (what isolation prevents):
    - DIRTY READ: read uncommitted data from another transaction.
    - NON-REPEATABLE READ: same row read twice gives different values (another
      txn updated + committed in between).
    - PHANTOM READ: same query returns different ROWS (another txn inserted).

ISOLATION LEVELS (weakest -> strongest; more isolation = less concurrency):
    READ UNCOMMITTED — allows dirty reads (rarely used).
    READ COMMITTED   — no dirty reads (Postgres default).
    REPEATABLE READ  — no dirty or non-repeatable reads (MySQL/InnoDB default).
    SERIALIZABLE     — full isolation, as if transactions ran one at a time;
                       safest, slowest, can cause serialization failures/retries.

    Trade-off: higher isolation = stronger correctness but more locking/
    contention and lower throughput. Pick the lowest level that's correct for
    your use case.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOCKING & DEADLOCKS:
    - Optimistic locking: assume no conflict; check a version column on update
      (good for low-contention; retry on conflict).
    - Pessimistic locking: SELECT ... FOR UPDATE locks rows (high-contention).
    - DEADLOCK: two txns each wait on a lock the other holds. DBs detect and
      kill one. Avoid by acquiring locks in a CONSISTENT ORDER, keeping txns short.

INTERVIEW ANSWER:
    "A transaction gives ACID: atomicity (all-or-nothing with rollback),
    consistency (constraints stay valid), isolation (concurrent transactions
    don't corrupt each other), and durability (committed data survives a
    crash via the write-ahead log). Isolation levels trade correctness for
    concurrency: read committed is the Postgres default and prevents dirty
    reads; repeatable read also prevents non-repeatable reads; serializable is
    fully isolated but slowest. I pick the lowest level that's correct. For
    contention I use optimistic locking with a version column when conflicts
    are rare, pessimistic SELECT FOR UPDATE when they're frequent, and I
    prevent deadlocks by acquiring locks in a consistent order and keeping
    transactions short."
'''


# =================================================================================
# SECTION 7: CONNECTION POOLING & SCALING PATTERNS
# =================================================================================
'''
The JD stresses "scalability." Know how DBs scale and why pooling matters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONNECTION POOLING:
    Opening a DB connection is EXPENSIVE (TCP + auth + session setup). A pool
    keeps a set of open connections and REUSES them across requests.
    - Without pooling: every request opens/closes a connection -> slow, and a
      traffic spike exhausts the DB's connection limit -> errors.
    - With pooling: requests borrow a connection, return it. Far faster, bounded.
    Tools: SQLAlchemy's built-in pool, PgBouncer (external pooler for Postgres).
    Tune pool_size + max_overflow to your DB's max_connections and worker count.

    SENIOR POINT: with many app workers (Phase 1), total connections =
    workers * pool_size. Must stay under the DB's max_connections — a common
    production outage. PgBouncer multiplexes to solve this at scale.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCALING A RELATIONAL DATABASE:
    1. VERTICAL (scale up): bigger machine (CPU/RAM). Simple, has a ceiling.
    2. READ REPLICAS: replicate writes to read-only copies; route reads to
       replicas, writes to primary. Scales READS. (Replication lag = eventual
       consistency on replicas.)
    3. CACHING: Redis in front of the DB for hot reads. Huge relief.
    4. PARTITIONING (within one DB): split a big table by range/list/hash
       (e.g., by month) so queries scan less.
    5. SHARDING (across DBs): split data across multiple DB servers by a shard
       key (e.g., user_id). Scales WRITES + storage. Cost: complex, cross-shard
       joins/transactions are hard. Last resort.
    6. CQRS: separate read and write models/stores for very high scale.

    Order of escalation: optimize queries+indexes -> cache -> read replicas ->
    partition -> shard. Shard only when you truly must.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "Connection pooling reuses a bounded set of open connections instead of
    opening one per request — opening connections is expensive and a spike can
    exhaust the DB's connection limit. With many app workers I watch that
    workers times pool_size stays under max_connections, and at scale I use
    PgBouncer to multiplex. To scale the database I escalate in order: first
    optimize queries and indexes, then cache hot reads in Redis, then add read
    replicas and route reads to them, then partition large tables, and only
    shard across servers as a last resort because cross-shard joins and
    transactions get hard. Replicas introduce replication lag, so I keep
    read-after-write on the primary when consistency matters."
'''


# =================================================================================
# SECTION 8: SQL YOU MUST BE ABLE TO WRITE
# =================================================================================
'''
The JD: "reading and writing SQL queries." Be ready to write these live.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JOINS (know all types):
    INNER JOIN  — only matching rows in both tables.
    LEFT JOIN   — all left rows + matched right (NULLs where no match).
    RIGHT JOIN  — all right rows + matched left.
    FULL OUTER  — all rows from both, NULLs where no match.
    CROSS JOIN  — cartesian product (every combo).
    SELF JOIN   — table joined to itself (e.g., employee -> manager).

    SELECT u.name, o.total
    FROM users u
    LEFT JOIN orders o ON o.user_id = u.id;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGGREGATION + GROUP BY + HAVING:
    SELECT department, COUNT(*) AS cnt, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department
    HAVING COUNT(*) > 5            -- HAVING filters GROUPS (WHERE filters ROWS)
    ORDER BY avg_sal DESC;

    KEY: WHERE filters rows BEFORE grouping; HAVING filters AFTER grouping.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUBQUERIES:
    SELECT name FROM employees
    WHERE salary > (SELECT AVG(salary) FROM employees);  -- scalar subquery

    -- correlated subquery (runs per outer row — can be slow):
    SELECT e.name FROM employees e
    WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE dept = e.dept);

WINDOW FUNCTIONS (senior-level — common screening question):
    -- rank salaries WITHIN each department without collapsing rows:
    SELECT name, department, salary,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
    FROM employees;
    -- ROW_NUMBER, RANK, DENSE_RANK, LAG/LEAD, SUM() OVER (running totals).

    THE "Nth HIGHEST" CLASSIC (2nd highest salary):
    SELECT DISTINCT salary FROM employees ORDER BY salary DESC
    OFFSET 1 LIMIT 1;
    -- or with DENSE_RANK() = 2 in a subquery.

CTEs (readable complex queries):
    WITH high_earners AS (
        SELECT * FROM employees WHERE salary > 100000
    )
    SELECT department, COUNT(*) FROM high_earners GROUP BY department;

INTERVIEW POINT:
    "I'm comfortable with all join types, GROUP BY with HAVING — remembering
    WHERE filters rows before grouping and HAVING filters groups after —
    subqueries including correlated ones, and window functions like
    RANK/ROW_NUMBER with PARTITION BY for things like ranking within a group or
    finding the Nth highest value. I use CTEs to keep complex queries readable."
'''


# =================================================================================
# SECTION 9: NoSQL DEEP DIVE (document, key-value, the CAP theorem)
# =================================================================================
'''
The JD names NoSQL. Know the families, modeling, and CAP.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA MODELING IN NoSQL (vs relational):
    Relational: NORMALIZE, model around the DATA, join at query time.
    NoSQL (esp. document): DENORMALIZE, model around the QUERY/ACCESS PATTERN,
    EMBED related data so a read is one lookup (no joins).

    Example (MongoDB): embed a customer's recent orders inside the customer
    document if you always read them together -> one fast read. Reference them
    (store IDs) if they're large or accessed independently.

    RULE: in NoSQL you design for HOW you'll QUERY, not for normalization.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EMBEDDING vs REFERENCING (MongoDB design decision):
    EMBED when: data is read together, bounded size, 1-to-few. (Fast reads.)
    REFERENCE when: data is large, unbounded (1-to-many/many), or shared/updated
    independently. (Avoids huge documents + duplicate-update pain.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CAP THEOREM (distributed systems — senior must know):
    In a distributed datastore, during a network PARTITION you can have only
    TWO of: Consistency, Availability, Partition-tolerance. Since partitions
    WILL happen, you really choose between C and A during a partition:
    - CP (consistency): reject requests rather than serve stale data
      (e.g., HBase, MongoDB in certain configs).
    - AP (availability): stay up, may serve stale data, reconcile later
      (e.g., Cassandra, DynamoDB).
    Relational single-node DBs sidestep CAP (not distributed) but offer ACID.

    BASE (NoSQL philosophy) vs ACID (SQL): Basically Available, Soft state,
    Eventual consistency — trades strong consistency for availability/scale.

INTERVIEW ANSWER:
    "In NoSQL I model around access patterns, not normalization — I denormalize
    and embed related data so a common read is a single lookup with no joins.
    In MongoDB I embed when data is read together and bounded, and reference
    when it's large, unbounded, or updated independently. The CAP theorem says
    during a network partition you trade consistency for availability — CP
    stores reject to stay consistent, AP stores like Cassandra stay available
    and reconcile later. That's the BASE model — eventual consistency — versus
    SQL's ACID. I choose based on whether the use case can tolerate eventual
    consistency."
'''


# =================================================================================
# SECTION 10: ORM vs RAW SQL (SQLAlchemy) + Migrations
# =================================================================================
'''
The JD mentions "database interaction tools." Know the ORM trade-offs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORM (Object-Relational Mapper — SQLAlchemy, Django ORM):
    Maps tables to Python classes, rows to objects. You write Python, it
    generates SQL.
    PROS: productivity, type safety, DB-agnostic, prevents SQL injection
          (parameterized), handles relationships, migrations.
    CONS: can generate inefficient SQL, hides what's happening (N+1!), a
          leaky abstraction for complex queries, learning curve.

RAW SQL:
    PROS: full control, optimal performance, complex queries/analytics, you see
          exactly what runs.
    CONS: manual, DB-specific, must guard against injection (parameterize!).

THE PRAGMATIC ANSWER (what leads actually do):
    Use the ORM for 90% — CRUD, relationships, productivity. Drop to raw SQL
    (or SQLAlchemy Core) for the 10% — complex reporting queries, performance-
    critical paths, bulk operations. SQLAlchemy lets you mix both. Never
    string-concatenate user input into SQL — always parameterize.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIGRATIONS (schema versioning — critical for teams):
    Tools: Alembic (SQLAlchemy), Django migrations.
    - Each schema change = a versioned migration file (like Git for the schema).
    - Apply (upgrade) / revert (downgrade); run in CI/CD before deploy.
    - NEVER hand-edit production schemas; always go through migrations for a
      reproducible, reviewable, rollbackable history. (Same discipline as
      prompt versioning in your Lesson 25 — versioned, tested, reversible.)
    - Care with destructive migrations (dropping columns) + zero-downtime
      deploys (expand-then-contract pattern).

SQLALCHEMY ASYNC (ties to Phase 1/2):
    Use the async engine (create_async_engine) + AsyncSession with an async
    driver (asyncpg) in FastAPI, so DB calls don't block the event loop.

INTERVIEW ANSWER:
    "I use an ORM like SQLAlchemy for most work — it's productive, handles
    relationships, and parameterizes queries so it prevents SQL injection. But
    ORMs can hide inefficiency, like N+1, and produce poor SQL for complex
    cases, so I drop to raw SQL or SQLAlchemy Core for reporting queries,
    performance-critical paths, and bulk operations. Schema changes always go
    through versioned migrations with Alembic — reviewable, reproducible, and
    rollbackable, never hand-edited in production, with care for zero-downtime
    using expand-then-contract. In FastAPI I use the async engine with asyncpg
    so DB calls don't block the loop."
'''


# =================================================================================
# SECTION 11: DATABASES FOR AI (pgvector, vector DBs)
# =================================================================================
'''
Connect DB knowledge to the GenAI use case — the JD lists pgvector explicitly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VECTOR STORAGE OPTIONS:
    - pgvector: a PostgreSQL EXTENSION that adds a 'vector' column type +
      similarity search. Huge advantage: keep embeddings ALONGSIDE relational
      data in ONE database — join metadata with vectors, transactional, no
      extra infra. Great up to a few million vectors.
    - Dedicated vector DBs: Pinecone (managed), Weaviate, Qdrant, Milvus,
      ChromaDB (the JD names several). Purpose-built ANN indexes, scale to
      billions, metadata filtering.

    DECISION: pgvector when you already use Postgres and scale is moderate —
    one less system to run. Dedicated vector DB at large scale or when you need
    advanced ANN features.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VECTOR INDEXES (ANN — approximate nearest neighbor):
    Exact search is O(n) — too slow at scale. ANN indexes trade a little
    accuracy for huge speed:
    - HNSW (Hierarchical Navigable Small World): graph-based, fast + accurate,
      higher memory. The common default (pgvector supports it).
    - IVFFlat: cluster-based, less memory, needs tuning (lists/probes).
    Metric: cosine similarity (normalized text embeddings), or L2/dot product.

    -- pgvector example:
    CREATE TABLE docs (id bigserial, content text, embedding vector(1536));
    CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);
    SELECT content FROM docs ORDER BY embedding <=> :query_vec LIMIT 5;
    -- <=> is the cosine-distance operator; ORDER BY it for nearest neighbors.

INTERVIEW ANSWER:
    "For AI workloads I often use pgvector — a Postgres extension that adds a
    vector column and similarity search — because it keeps embeddings next to
    relational metadata in one database, so I can filter by metadata and join,
    all transactionally, without running a separate system. It scales well to a
    few million vectors with an HNSW index. Beyond that, or when I need
    advanced ANN features, I move to a dedicated vector DB like Qdrant or
    Pinecone. Either way the search uses an approximate-nearest-neighbor index —
    usually HNSW — with cosine similarity for normalized text embeddings,
    trading a little recall for big speed gains."
'''


# =================================================================================
# SECTION 12: INTERVIEW Q&A (senior-level)
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. SQL vs NoSQL — when?
A:  "SQL for relationships, ACID, complex queries — most business data. NoSQL
    for scale, flexible schema, or specific shapes: document, key-value, graph,
    time-series. Most systems are polyglot."

Q2. What is normalization? 3NF?
A:  "Organizing data to remove redundancy and anomalies. 3NF = no transitive
    dependencies; non-key columns depend only on the key. I denormalize
    deliberately for read performance with evidence."

Q3. How does an index work and when do you add one?
A:  "A B-tree that turns a scan into a log-n lookup. Index WHERE/JOIN/ORDER BY
    columns and FKs, favoring high selectivity. Indexes slow writes, so don't
    over-index or index low-selectivity columns."

Q4. A query is slow — how do you debug?
A:  "EXPLAIN ANALYZE. Look for sequential scans on big tables, fix missing
    indexes, avoid SELECT * and functions on indexed columns, eliminate N+1,
    and cache or materialize expensive aggregates."

Q5. What is the N+1 problem?
A:  "1 query for parents + 1 per parent for related data. Fix with eager
    loading — joinedload/selectinload in SQLAlchemy, select_related/
    prefetch_related in Django."

Q6. Explain ACID and isolation levels.
A:  "Atomicity, consistency, isolation, durability. Levels trade correctness
    for concurrency: read committed (Postgres default), repeatable read,
    serializable. Pick the lowest that's correct."

Q7. Composite index — leftmost prefix?
A:  "An index on (a,b) helps queries on a or (a,b) but not b alone — it must
    use the leftmost column. Order columns by query pattern."

Q8. How do you scale a relational DB?
A:  "Optimize queries+indexes -> cache (Redis) -> read replicas -> partition
    -> shard. Shard only as a last resort due to cross-shard complexity."

Q9. Why connection pooling?
A:  "Connections are expensive; pooling reuses a bounded set. With many
    workers I keep workers*pool_size under max_connections, using PgBouncer at
    scale."

Q10. ORM vs raw SQL?
A:  "ORM for productivity and safety on 90%; raw SQL for complex reporting and
    performance-critical paths. Always parameterize. Schema via versioned
    migrations (Alembic)."

Q11. 2nd highest salary in SQL?
A:  "SELECT DISTINCT salary ORDER BY salary DESC OFFSET 1 LIMIT 1, or
    DENSE_RANK() OVER (ORDER BY salary DESC) = 2 in a subquery."

Q12. How do you store embeddings for RAG?
A:  "pgvector if I'm on Postgres and scale is moderate — embeddings beside
    metadata, HNSW index, cosine distance. A dedicated vector DB like Qdrant
    or Pinecone at large scale."

Q13. WHERE vs HAVING?
A:  "WHERE filters rows before grouping; HAVING filters groups after
    aggregation."

Q14. How do you prevent SQL injection?
A:  "Parameterized queries / ORM bind parameters — never string-concatenate
    user input. Validate input at the boundary too."
'''


# =================================================================================
# SECTION 13: GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — SQL IS THE DEFAULT; NoSQL FOR SPECIFIC SHAPES.
    "NoSQL scales, SQL doesn't" is wrong. It's a trade-off: consistency/joins
    vs scale/flexibility. Most systems are polyglot.

GOLDEN LESSON 2 — NORMALIZE FOR INTEGRITY, DENORMALIZE FOR SPEED (WITH EVIDENCE).
    Design to 3NF, then denormalize measured hot paths. Never prematurely.

GOLDEN LESSON 3 — INDEXING IS THE #1 PERFORMANCE LEVER.
    Index WHERE/JOIN/ORDER BY + FKs, high selectivity. But indexes slow writes —
    don't over-index. Know the composite leftmost-prefix rule.

GOLDEN LESSON 4 — EXPLAIN ANALYZE IS YOUR FIRST MOVE ON A SLOW QUERY.
    Seq scan on a big table = missing index. Measure, don't guess.

GOLDEN LESSON 5 — N+1 IS THE ORM KILLER.
    Eager-load (joinedload/selectinload, select_related/prefetch_related).
    Name it and fix it = instant credibility.

GOLDEN LESSON 6 — ACID + ISOLATION = PICK THE LOWEST CORRECT LEVEL.
    Higher isolation = more correctness, less concurrency. Know the anomalies
    each level prevents.

GOLDEN LESSON 7 — SCALE IN ORDER: OPTIMIZE -> CACHE -> REPLICAS -> PARTITION -> SHARD.
    Shard last. Pooling keeps connections bounded under many workers.

GOLDEN LESSON 8 — ORM FOR 90%, RAW SQL FOR THE HARD 10%, MIGRATIONS ALWAYS.
    Parameterize everything. Schema changes are versioned and rollbackable.

GOLDEN LESSON 9 — NoSQL MODELS AROUND THE QUERY; CAP IS A TRADE-OFF.
    Embed vs reference by access pattern. CP vs AP during a partition.

GOLDEN LESSON 10 — FOR AI: pgvector BESIDE YOUR DATA, OR A DEDICATED VECTOR DB.
    HNSW index, cosine distance. Ties DB skills to the JD's RAG focus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET:
    SQL default (ACID, joins); NoSQL for scale/shape (document/KV/wide/graph).
    Normalize to 3NF; denormalize hot paths with evidence.
    Index WHERE/JOIN/ORDER BY + FKs; composite = leftmost prefix; don't over-index.
    Slow query -> EXPLAIN ANALYZE -> kill seq scans / N+1 / SELECT *.
    ACID + isolation (read committed -> repeatable read -> serializable).
    Scale: optimize -> cache -> replicas -> partition -> shard (last). Pool connections.
    ORM 90% + raw SQL 10%; always parameterize; migrations via Alembic.
    AI: pgvector (with metadata) or Qdrant/Pinecone; HNSW + cosine.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4 — DATABASES: SCHEMA DESIGN & OPTIMIZATION")
    print("=" * 70)
    print()
    print("SQL vs NoSQL: SQL default (ACID/joins) | NoSQL for scale/shape")
    print("NORMALIZE to 3NF for integrity; denormalize hot paths WITH evidence")
    print()
    print("INDEXING = #1 lever: WHERE/JOIN/ORDER BY + FKs; composite leftmost-prefix")
    print("  (indexes speed reads, slow writes -> don't over-index)")
    print()
    print("SLOW QUERY -> EXPLAIN ANALYZE -> seq scan = missing index")
    print("N+1 PROBLEM -> eager load (joinedload/selectinload, select_related)")
    print()
    print("ACID + isolation: read committed -> repeatable read -> serializable")
    print("SCALE ORDER: optimize -> cache -> replicas -> partition -> shard(last)")
    print("  + connection pooling (workers * pool_size < max_connections)")
    print()
    print("ORM 90% + raw SQL 10%; always parameterize; Alembic migrations")
    print("AI: pgvector beside metadata, or Qdrant/Pinecone; HNSW + cosine")
    print()
    print("=" * 70)
    print("PRIMARY-FOCUS BLOCK COMPLETE (Phases 1-4).")
    print("Next: Phase 6 — Multi-Agent Systems & Frameworks (medium priority)")
    print("=" * 70)
