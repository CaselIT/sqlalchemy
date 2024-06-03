from __future__ import annotations

from enum import Enum
import operator
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .row import Row
from .row import RowMapping
from .. import exc
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import TupleAny
from ..util.typing import Unpack


if TYPE_CHECKING:
    from .result import Result
    from .result import ResultMetaData

try:
    # NOTE: the cython compiler needs this "import cython" in the file, it
    # can't be only "from sqlalchemy.util import cython" with the fallback
    # in that module
    import cython
except ModuleNotFoundError:
    from sqlalchemy.util import cython


_RowData = Union[Row[Unpack[TupleAny]], RowMapping, Any]
"""A generic form of "row" that accommodates for the different kinds of
"rows" that different result objects return, including row, row mapping, and
scalar values"""
_R = TypeVar("_R", bound=_RowData)
_InterimRowType = Union[_R, TupleAny]
"""a catchall "anything" kind of return type that can be applied
across all the result types

"""

_UniqueFilterType = Callable[[Any], Any]
_UniqueFilterStateType = Tuple[Set[Any], Optional[_UniqueFilterType]]


class _NoRow(Enum):
    """a symbol that indicates to internal Result methods that
    "no row is returned".  We can't use None for those cases where a scalar
    filter is applied to rows.
    """

    _NO_ROW = 0


_NO_ROW = _NoRow._NO_ROW

_FLAG_SIMPLE = cython.declare(cython.int, 0)
_FLAG_SCALAR_TO_TUPLE = cython.declare(cython.int, 1)
_FLAG_TUPLE_FILTER = cython.declare(cython.int, 2)


class BaseResultInternal(Generic[_R]):
    __slots__ = ()

    _real_result: Optional[Result[Unpack[TupleAny]]] = None
    _generate_rows: bool = True
    _row_logging_fn: Optional[Callable[[Any], Any]]

    _unique_filter_state: Optional[_UniqueFilterStateType] = None
    # TODO remove _post_creational_filter since it's only used by scalar
    _post_creational_filter: Optional[Callable[[Any], Any]] = None
    _is_cursor = False

    _metadata: ResultMetaData

    _source_supports_scalars: bool

    def _fetchiter_impl(
        self,
    ) -> Iterator[_InterimRowType[Row[Unpack[TupleAny]]]]:
        raise NotImplementedError()

    def _fetchone_impl(
        self, hard_close: bool = False
    ) -> Optional[_InterimRowType[Row[Unpack[TupleAny]]]]:
        raise NotImplementedError()

    def _fetchmany_impl(
        self, size: Optional[int] = None
    ) -> List[_InterimRowType[Row[Unpack[TupleAny]]]]:
        raise NotImplementedError()

    def _fetchall_impl(
        self,
    ) -> List[_InterimRowType[Row[Unpack[TupleAny]]]]:
        raise NotImplementedError()

    def _soft_close(self, hard: bool = False) -> None:
        raise NotImplementedError()

    def _get_real_result(self) -> Result[Unpack[TupleAny]]:
        if self._real_result is not None:
            return self._real_result
        else:
            return self  # type: ignore

    @HasMemoized_ro_memoized_attribute
    def _row_getter(
        self,
    ) -> Tuple[Optional[Callable[..., _R]], Optional[Callable[..., list[_R]]]]:
        real_result = self._get_real_result()

        metadata = self._metadata
        tuple_filters = metadata._tuplefilter
        flag: cython.int = _FLAG_SIMPLE

        if real_result._source_supports_scalars:
            if not self._generate_rows:
                return None, None
            else:
                flag = _FLAG_SCALAR_TO_TUPLE
        elif tuple_filters is not None:
            flag = _FLAG_TUPLE_FILTER

        if metadata._effective_processors is not None:
            ep = metadata._effective_processors
            if flag == _FLAG_TUPLE_FILTER:
                ep = tuple_filters(ep)

            processors: tuple = tuple(ep)
        else:
            processors = ()

        proc_size: cython.Py_ssize_t = len(processors)
        _log_row = real_result._row_logging_fn

        key_to_index = metadata._key_to_index
        _Row = Row

        if flag == _FLAG_SIMPLE and proc_size == 0 and _log_row is None:
            # just build the rows

            # TODO: try partial
            def single_row_simple(input_row, /) -> Row:
                return _Row(metadata, None, key_to_index, input_row)

            # TODO: list comprehension
            def many_rows_simple(rows: list, /) -> list:
                for i in range(len(rows)):
                    rows[i] = _Row(metadata, None, key_to_index, rows[i])
                return rows

            return single_row_simple, many_rows_simple

        first_row: cython.bint = True

        def single_row(input_row, /) -> Row:
            nonlocal first_row

            if flag == _FLAG_SCALAR_TO_TUPLE:
                input_row = (input_row,)
            elif flag == _FLAG_TUPLE_FILTER:
                input_row = tuple_filters(input_row)

            if proc_size != 0:
                if first_row:
                    first_row = False
                    assert len(input_row) == proc_size
                input_row = _apply_processors(processors, proc_size, input_row)

            row = _Row(metadata, None, key_to_index, input_row)
            if _log_row is not None:
                row = _log_row(row)
            return row

        def many_rows(rows: list, /) -> list:
            for i in range(len(rows)):
                rows[i] = single_row(rows[i])
            return rows

        return single_row, many_rows

    @HasMemoized_ro_memoized_attribute
    def _iterator_getter(self) -> Callable[[], Iterator[_R]]:
        make_row = self._row_getter[0]

        post_creational_filter = self._post_creational_filter

        if self._unique_filter_state:
            uniques: set  # TODO: is type def needed?
            uniques, strategy = self._unique_strategy

            def iterrows() -> Iterator[_R]:
                row: _InterimRowType[Any]
                for raw_row in self._fetchiter_impl():
                    row = (
                        make_row(raw_row) if make_row is not None else raw_row
                    )
                    hashed = strategy(row) if strategy is not None else row
                    if hashed in uniques:
                        continue
                    uniques.add(hashed)
                    if post_creational_filter is not None:  # TODO: remove
                        row = post_creational_filter(row)
                    yield row  # type: ignore

        else:

            def iterrows() -> Iterator[_R]:
                row: _InterimRowType[Any]
                for raw_row in self._fetchiter_impl():
                    row = (
                        make_row(raw_row) if make_row is not None else raw_row
                    )
                    if post_creational_filter is not None:  # TODO: remove
                        row = post_creational_filter(row)
                    yield row  # type: ignore

        return iterrows

    def _raw_all_rows(self) -> List[_R]:
        make_rows = self._row_getter[1]
        assert make_rows is not None
        return make_rows(self._fetchall_impl())

    def _allrows(self) -> List[_R]:
        post_creational_filter = self._post_creational_filter

        make_rows = self._row_getter[1]

        rows = self._fetchall_impl()
        made_rows: List[_InterimRowType[_R]]
        if make_rows is not None:
            made_rows = make_rows(rows)
        else:
            made_rows = rows  # type: ignore

        interim_rows: List[_R]

        if self._unique_filter_state:
            uniques: set  # TODO: is type def needed?
            uniques, strategy = self._unique_strategy
            interim_rows = _apply_unique_strategy(
                made_rows, [], uniques, strategy
            )
        else:
            interim_rows = made_rows  # type: ignore

        if post_creational_filter is not None:  # TODO: remove
            interim_rows = [
                post_creational_filter(row) for row in interim_rows
            ]
        return interim_rows

    @HasMemoized_ro_memoized_attribute
    def _onerow_getter(
        self,
    ) -> Callable[[], Union[Literal[_NO_ROW], _R]]:
        make_row = self._row_getter[0]

        post_creational_filter = self._post_creational_filter

        if self._unique_filter_state:
            uniques: set
            uniques, strategy = self._unique_strategy

            def onerow() -> Union[Literal[_NO_ROW], _R]:
                while True:
                    row = self._fetchone_impl()
                    if row is None:
                        return _NO_ROW
                    else:
                        obj: _InterimRowType[Any] = (
                            make_row(row) if make_row is not None else row
                        )
                        hashed = strategy(obj) if strategy is not None else obj
                        if hashed in uniques:
                            continue
                        uniques.add(hashed)
                        if post_creational_filter is not None:
                            obj = post_creational_filter(obj)
                        return obj  # type: ignore

        else:

            def onerow() -> Union[Literal[_NO_ROW], _R]:
                row = self._fetchone_impl()
                if row is None:
                    return _NO_ROW
                else:
                    interim_row: _InterimRowType[Any] = (
                        make_row(row) if make_row is not None else row
                    )
                    if post_creational_filter is not None:
                        interim_row = post_creational_filter(interim_row)
                    return interim_row  # type: ignore

        return onerow

    @HasMemoized_ro_memoized_attribute
    def _manyrow_getter(self) -> Callable[[Optional[int]], List[_R]]:
        make_rows = self._row_getter[1]
        real_result = self._get_real_result()

        post_creational_filter = self._post_creational_filter

        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            def manyrows(num: Optional[int], /) -> List[_R]:
                collect: List[_R] = []

                _manyrows = self._fetchmany_impl

                if num is None:
                    # if None is passed, we don't know the default
                    # manyrows number, DBAPI has this as cursor.arraysize
                    # different DBAPIs / fetch strategies may be different.
                    # do a fetch to find what the number is.  if there are
                    # only fewer rows left, then it doesn't matter.
                    if real_result._yield_per:
                        num_required = num = real_result._yield_per
                    else:
                        rows = _manyrows(num)
                        num = len(rows)
                        made_rows = (
                            rows if make_rows is None else make_rows(rows)
                        )
                        _apply_unique_strategy(
                            made_rows, collect, uniques, strategy
                        )
                        num_required = num - len(collect)
                else:
                    num_required = num

                assert num is not None

                while num_required:
                    rows = _manyrows(num_required)
                    if not rows:
                        break

                    made_rows = rows if make_rows is None else make_rows(rows)
                    _apply_unique_strategy(
                        made_rows, collect, uniques, strategy
                    )
                    num_required = num - len(collect)

                if post_creational_filter is not None:
                    collect = [post_creational_filter(row) for row in collect]
                return collect

        else:

            def manyrows(num: Optional[int], /) -> List[_R]:
                if num is None:
                    num = real_result._yield_per

                rows: List[_InterimRowType[Any]] = self._fetchmany_impl(num)
                if make_rows is not None:
                    rows = make_rows(rows)
                if post_creational_filter is not None:
                    rows = [post_creational_filter(row) for row in rows]
                return rows  # type: ignore

        return manyrows

    @overload
    def _only_one_row(
        self,
        raise_for_second_row: bool,
        raise_for_none: Literal[True],
        scalar: bool,
    ) -> _R: ...

    @overload
    def _only_one_row(
        self,
        raise_for_second_row: bool,
        raise_for_none: bool,
        scalar: bool,
    ) -> Optional[_R]: ...

    def _only_one_row(
        self,
        raise_for_second_row: bool,
        raise_for_none: bool,
        scalar: bool,
    ) -> Optional[_R]:
        onerow = self._fetchone_impl

        row: Optional[_InterimRowType[Any]] = onerow(hard_close=True)
        if row is None:
            if raise_for_none:
                raise exc.NoResultFound(
                    "No row was found when one was required"
                )
            else:
                return None

        if scalar and self._source_supports_scalars:
            self._generate_rows = False
            make_row = None
        else:
            make_row = self._row_getter[0]

        try:
            row = make_row(row) if make_row is not None else row
        except:
            self._soft_close(hard=True)
            raise

        if raise_for_second_row:
            if self._unique_filter_state:
                # for no second row but uniqueness, need to essentially
                # consume the entire result :(
                uniques, strategy = self._unique_strategy

                existing_row_hash = strategy(row) if strategy else row

                while True:
                    next_row: Any = onerow(hard_close=True)
                    if next_row is None:
                        next_row = _NO_ROW
                        break

                    try:
                        next_row = (
                            make_row(next_row)
                            if make_row is not None
                            else next_row
                        )

                        if strategy is not None:
                            assert next_row is not _NO_ROW
                            if existing_row_hash == strategy(next_row):
                                continue
                        elif row == next_row:
                            continue
                        # here, we have a row and it's different
                        break
                    except:
                        self._soft_close(hard=True)
                        raise
            else:
                next_row = onerow(hard_close=True)
                if next_row is None:
                    next_row = _NO_ROW

            if next_row is not _NO_ROW:
                self._soft_close(hard=True)
                raise exc.MultipleResultsFound(
                    "Multiple rows were found when exactly one was required"
                    if raise_for_none
                    else "Multiple rows were found when one or none "
                    "was required"
                )
        else:
            next_row = _NO_ROW
            # if we checked for second row then that would have
            # closed us :)
            self._soft_close(hard=True)

        if not scalar:
            post_creational_filter = self._post_creational_filter
            if post_creational_filter is not None:
                row = post_creational_filter(row)

        if scalar and make_row is not None:
            return row[0]  # type: ignore
        else:
            return row  # type: ignore

    def _iter_impl(self) -> Iterator[_R]:
        return self._iterator_getter()

    def _next_impl(self) -> _R:
        row = self._onerow_getter()
        if row is _NO_ROW:
            raise StopIteration()
        else:
            return row

    @HasMemoized_ro_memoized_attribute
    def _unique_strategy(self) -> _UniqueFilterStateType:
        assert self._unique_filter_state is not None
        uniques, strategy = self._unique_filter_state

        if not strategy and self._metadata._unique_filters:
            real_result = self._get_real_result()
            if (
                real_result._source_supports_scalars
                and not self._generate_rows
            ):
                strategy = self._metadata._unique_filters[0]
            else:
                filters = self._metadata._unique_filters
                if self._metadata._tuplefilter is not None:
                    filters = self._metadata._tuplefilter(filters)

                strategy = operator.methodcaller("_filter_on_values", filters)
        return uniques, strategy


@cython.inline
@cython.cfunc
def _apply_processors(
    processors: tuple, proc_size: cython.Py_ssize_t, data: Sequence[Any]
) -> List[Any]:
    res: List[Any] = list(data)
    # TODO: this could be a dict index -> proc / list[tuple[index, proc]] / two list index & proc
    for i in range(proc_size):
        p = processors[i]
        if p is not None:
            res[i] = p(res[i])
    return res


@cython.inline
@cython.cfunc
def _apply_unique_strategy(
    rows: List[Any],
    destination: List[Any],
    uniques: Set[Any],
    strategy: Callable | None,
) -> List[Any]:
    for i in range(len(rows)):
        row = rows[i]
        hashed = strategy(row) if strategy is not None else row
        if hashed in uniques:
            continue
        uniques.add(hashed)
        destination.append(row)
    return destination
