# NOTE: typing implementation is an half assed mess so unit test it.

import typing as T
from sqlalchemy.testing.assertions import eq_
from sqlalchemy.testing.assertions import is_
from sqlalchemy.testing.config import fixture
import typing_extensions as TU
from sqlalchemy.util import py312
from sqlalchemy.testing import fixtures
from sqlalchemy.util import typing as sa_typing

_type_aliases = {}


def _type_to_str_eval(thing):
    if isinstance(thing, type):
        return thing.__name__
    str_thing = str(thing)
    if "typing." in str_thing:
        return str_thing.replace("typing.", "T.")
    assert "<" not in str_thing
    assert eval(str_thing, None, _type_aliases) == thing, thing
    return str_thing


def _or_union_str(types):
    ts = [_type_to_str_eval(t) for t in types]
    return "|".join(ts)


def _eval_or_union(types):
    return eval(_or_union_str(types), None, _type_aliases)


def _type_alias(types):
    if not py312:
        return []
    name = f"ta{len(_type_aliases)}"
    assert name not in _type_aliases
    code = f"type {name} = {_or_union_str(types)}"
    exec(code, None, _type_aliases)
    res = _type_aliases[name]
    globals()[name] = res
    return [res]


def many_unions(t1, t2, *tn):
    types = (t1, t2, *tn)
    unions = [T.Union[*types], _eval_or_union(types)]
    unions += _type_alias(types)
    return unions


def many_optionals(*types):
    optionals = [T.Optional[t] for t in types]
    optionals += [T.Union[t, None] for t in types]
    optionals += [_eval_or_union((t, None)) for t in types]
    for t in types:
        optionals += _type_alias((t, None))
    return optionals


def str_unique(things):
    return {str(t): t for t in things}.values()


class Fixture(fixtures.TestBase):
    @fixture(autouse=True)
    def clear_aliases(self):
        for k in _type_aliases:
            globals().pop(k)
        _type_aliases.clear()


class TestTestingThings(Fixture):
    def test_unions_are_the_same(self):
        # no need to test TU.Union, TU.Optional
        is_(T.Union, TU.Union)
        is_(T.Optional, TU.Optional)

    def test_make_union(self):
        v = int, str
        eq_(T.Union[int, str], T.Union[*(int, str)])
        eq_(T.Union[int, str], T.Union[*v])

    def test_many_unions(self):
        # Union[int, str] is equal to int | str but has different str
        if py312:
            code = """
type ta0 = int | str
o1 = T.Optional[ta0]
o3 = ta0 | None
type ta1 = T.Union[int, str] | None
type ta2 = int | str | None
type ta3 = ta0 | None
type ta4 = float | str | None
"""
            scope = {}
            exec(code, None, scope)
            t_aliases = [[v] for v in scope.values()]
        else:
            t_aliases = [[]] * 7

        unions = many_unions(int, str)
        eq_(
            str(unions),
            str(
                [
                    T.Union[int, str],
                    int | str,
                    *t_aliases[0],
                ]
            ),
        )
        eq_(
            str(many_optionals(*unions)),
            str(
                [
                    T.Optional[T.Union[int, str]],
                    T.Optional[int | str],
                    *t_aliases[1],
                    T.Union[int, str, None],
                    T.Union[int, str, None],
                    *t_aliases[1],
                    T.Union[int, str] | None,
                    int | str | None,
                    *t_aliases[2],
                    *t_aliases[3],
                    *t_aliases[4],
                    *t_aliases[5],
                ]
            ),
        )
        eq_(
            str(many_unions(float, str, None)),
            str(
                [
                    T.Union[float, str, None],
                    float | str | None,
                    *t_aliases[6],
                ]
            ),
        )


class TestTyping(Fixture):
    def test_is_union(self):
        eq_(sa_typing.is_union(str), False)
        eq_(sa_typing.is_union(T.Union[int, str]), True)
        eq_(sa_typing.is_union(T.Optional[int]), True)
        eq_(sa_typing.is_union(str | int), True)
        eq_(sa_typing.is_union(T.Optional[int | str]), True)
        for t in _type_alias([str]):
            eq_(sa_typing.is_union(t), False)
        for t in _type_alias([str, int]):
            eq_(sa_typing.is_union(t), True)

    def test_pep695(self):
        for t in _type_alias([str]):
            eq_(sa_typing.is_pep695(t), True)
        eq_(sa_typing.is_pep695(T.Union[int, str]), False)

    def test_pep695_value(self):
        is_(sa_typing.pep695_value(int), int)
        eq_(sa_typing.pep695_value(T.Union[int, str]), T.Union[int, str])
        for t in _type_alias([str]):
            eq_(sa_typing.pep695_value(t), str)
            for t2 in _type_alias([t]):
                eq_(sa_typing.pep695_value(t2), str)
                eq_(sa_typing.pep695_value(t2, deep=False), t)
            for t2 in _type_alias([t, int, float]):
                eq_(sa_typing.pep695_value(t2), T.Union[str, int, float])
                eq_(
                    sa_typing.pep695_value(t2, deep=False),
                    T.Union[t, int, float],
                )
                for t3 in _type_alias([t2, t, bool]):
                    eq_(
                        sa_typing.pep695_value(t3),
                        T.Union[str, int, float, bool],
                    )
                    eq_(
                        sa_typing.pep695_value(t3, deep=False),
                        T.Union[t2, t, bool],
                    )

    def test_is_optional(self):
        for opt in str_unique(many_unions(int, None)):
            eq_(sa_typing.is_optional(opt), True, opt)
        for opt in str_unique(many_optionals(int, int | None)):
            eq_(sa_typing.is_optional(opt), True, opt)


type a = int | None
type b = str | a
type c = int | None | list[c]
type d = int | None | e
type e = str | d
type z = int | None | z

print(sa_typing.pep695_value(a))
print(sa_typing.pep695_value(b))
print(sa_typing.pep695_value(c))
print(sa_typing.pep695_value(d))
print(sa_typing.pep695_value(e))
print(sa_typing.pep695_value(z))
