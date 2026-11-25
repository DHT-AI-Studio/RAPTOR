"""Shared paging for the admin list routes -- GD-7's audit log
(access.py) and V05-10's job list (jobs.py): the `from`/`to` check, the
WHERE builder and the count + page query, written once.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException

MAX_PAGE_SIZE = 1000


def check_window(from_: datetime | None, to: datetime | None) -> None:
    if from_ and to and from_ > to:
        raise HTTPException(status_code=422, detail="`from` must not be after `to`")


def where_clause(conditions: list[tuple[str, object]]) -> tuple[str, list]:
    """WHERE clause + args from (sql, value) pairs, skipping None values.
    Each sql is a fixed literal with one `{n}` for its placeholder number --
    never caller input; every value is parameterized."""
    clauses, args = [], []
    for sql, value in conditions:
        if value is None:
            continue
        args.append(value)
        clauses.append(sql.format(n=len(args)))
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), args


def page_args(page: int, page_size: int) -> tuple[int, int]:
    return page_size, (page - 1) * page_size


async def fetch_page(
    conn, *, table: str, columns: str, where: str, args: list, order_by: str, page: int, page_size: int,
) -> tuple[int, list]:
    """(total matching rows, this page's rows). table/columns/order_by are
    the caller's fixed literals; where comes from where_clause()."""
    limit, offset = page_args(page, page_size)
    total = await conn.fetchval(f"SELECT count(*) FROM {table} {where}", *args)  # nosec B608 -- table and where_clause() output are fixed literals, every value parameterized
    rows = await conn.fetch(
        f"SELECT {columns} FROM {table} {where} ORDER BY {order_by} "
        f"LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}",  # nosec B608 -- same where; limit/offset are Query(ge=, le=)-validated ints
        *args, limit, offset,
    )
    return total, rows
