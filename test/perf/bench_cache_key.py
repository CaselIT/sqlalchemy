"""Standalone cache key generation benchmark.

Self contained so that it can be run against any checkout to compare
cache key generation across branches:

.. sourcecode:: text

    python test/perf/bench_cache_key.py <label> [outfile.json]

Prints microseconds per ``_generate_cache_key()`` call for each statement,
best of seven runs.

Note this deliberately does not use test/perf/compiled_extensions/cache_key.py,
which compares implementations within a single process rather than across
checkouts (and which currently references a few APIs that have since been
renamed).

"""

from __future__ import annotations

import json
import sys
import timeit
from types import SimpleNamespace

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.oracle.base import OracleDialect
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.engine import ObjectKind
from sqlalchemy.engine import ObjectScope


def setup_objects():
    metadata = sa.MetaData()
    parent = sa.Table(
        "parent",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("data", sa.String(20)),
    )
    child = sa.Table(
        "child",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("data", sa.String(20)),
        sa.Column(
            "parent_id", sa.Integer, sa.ForeignKey("parent.id"), nullable=False
        ),
    )

    class Parent:
        pass

    class Child:
        pass

    registry = orm.registry()
    registry.map_imperatively(
        Parent,
        parent,
        properties={"children": orm.relationship(Child, backref="parent")},
    )
    registry.map_imperatively(Child, child)

    many_types = sa.Table(
        "large",
        metadata,
        sa.Column("col_ARRAY", sa.ARRAY(sa.Integer)),
        *[
            sa.Column("col_%s" % name, getattr(sa, name))
            for name in (
                "BIGINT",
                "BigInteger",
                "BINARY",
                "BLOB",
                "BOOLEAN",
                "Boolean",
                "CHAR",
                "CLOB",
                "DATE",
                "Date",
                "DATETIME",
                "DateTime",
                "DECIMAL",
                "DOUBLE",
                "Double",
                "DOUBLE_PRECISION",
                "FLOAT",
                "Float",
                "INT",
                "INTEGER",
                "Integer",
                "Interval",
                "JSON",
                "LargeBinary",
                "NCHAR",
                "NUMERIC",
                "Numeric",
                "NVARCHAR",
                "PickleType",
                "REAL",
                "SMALLINT",
                "SmallInteger",
                "String",
                "TEXT",
                "Text",
                "TIME",
                "Time",
                "TIMESTAMP",
                "Unicode",
                "UnicodeText",
                "UUID",
                "Uuid",
                "VARBINARY",
                "VARCHAR",
            )
        ],
    )

    registry.configure()

    return SimpleNamespace(**locals())


def setup_statements():
    setup = setup_objects()

    stmts = {}

    stmts["core_small"] = sa.select(setup.parent).where(
        setup.parent.c.id == 42
    )
    stmts["orm_select"] = (
        sa.select(setup.Parent)
        .order_by(setup.Parent.id)
        .where(setup.Parent.data.like("cat"))
    )
    stmts["orm_join"] = (
        sa.select(setup.Parent.id, setup.Child.id)
        .select_from(
            orm.join(setup.Parent, setup.Child, setup.Parent.children)
        )
        .where(setup.Child.id == 5)
    )
    stmts["many_types"] = sa.select(setup.many_types).where(
        setup.many_types.c.col_Boolean
    )
    stmts["core_insert"] = setup.parent.insert().values(
        id=1, data=sa.bindparam("d")
    )
    stmts["core_update"] = (
        setup.parent.update().where(setup.parent.c.id == 5).values(data="x")
    )
    stmts["core_union"] = sa.union(
        sa.select(setup.parent.c.id).where(setup.parent.c.data == "a"),
        sa.select(setup.child.c.id).where(setup.child.c.data == "b"),
    )
    stmts["deep_and"] = sa.select(setup.parent).where(
        sa.and_(
            *[setup.parent.c.data == str(i) for i in range(20)],
        )
    )

    oracle = OracleDialect()
    oracle.server_version_info = (21, 0, 0)
    stmts["oracle_all_objects"] = oracle._all_objects_query(
        "scott", ObjectScope.DEFAULT, ObjectKind.ANY, False, False
    )
    stmts["oracle_column"] = oracle._column_query("scott")
    stmts["oracle_index"] = oracle._index_query("scott")
    stmts["oracle_constraint"] = oracle._constraint_query("scott")

    pg = PGDialect()
    pg.server_version_info = (16, 0, 0)
    stmts["pg_columns"] = pg._columns_query(
        "scott", False, ObjectScope.DEFAULT, ObjectKind.ANY
    )
    stmts["pg_index"] = pg._index_query
    stmts["pg_constraint"] = pg._constraint_query
    stmts["pg_fk"] = pg._foreing_key_query(
        "scott", False, ObjectScope.DEFAULT, ObjectKind.ANY
    )
    stmts["pg_enum"] = pg._enum_query("scott")

    return stmts


def bench(label, outfile=None):
    from sqlalchemy.sql import _util_cy

    stmts = setup_statements()
    results = {}
    for name, stmt in stmts.items():
        gen = stmt._generate_cache_key
        assert gen() is not None, name
        # calibrate: aim for ~0.25s per repeat
        number = 100
        while True:
            t = timeit.timeit(gen, number=number)
            if t > 0.15:
                break
            number *= 4
        times = [timeit.timeit(gen, number=number) for _ in range(7)]
        times.remove(min(times))
        times.remove(max(times))
        per_call = sum(times) / len(times) / number
        results[name] = per_call * 1e6  # microseconds

    out = {
        "label": label,
        "compiled": _util_cy._is_compiled(),
        "results": results,
    }
    print(json.dumps(out, indent=2))
    if outfile:
        with open(outfile, "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    bench(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
