#!/usr/bin/env python3
"""Ultimate Tic-Tac-Toe: the whole web app in one Python file.

Run it:
    python ult_ttt.py              # starts the game and opens http://localhost:8080
    python ult_ttt.py --port 9000  # use another port
    python ult_ttt.py --lan        # let other devices on your Wi-Fi join
    python ult_ttt.py --test       # run the built-in checks

Needs Python 3.10 or newer and nothing else: it uses only the standard library.
Saved games live in ult_ttt.db, next to this file.

How the file is laid out:
    1. Rules      the game rules, as pure functions (no web, no database)
    2. Storage    SQLite tables for games and moves
    3. Service    saved games: create, list, rename, delete, play, rematch, review
    4. Web API    JSON over HTTP under /api/v1/games
    5. Web page   the HTML, CSS and JavaScript the browser runs
    6. Self-tests run with --test
    7. main()     command-line options and the server
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import webbrowser
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

# =============================================================================
# 1. Rules
# =============================================================================
#
# Boards and cells are numbered 0-8 in reading order:
#
#     0 | 1 | 2
#     3 | 4 | 5
#     6 | 7 | 8
#
# The same numbers are used for the nine small boards and for the nine cells
# inside each one. That is what makes the "send" rule work: playing cell 2 of
# any board sends the opponent to board 2.
#
# The rules:
#   1. X moves first, anywhere.
#   2. The cell you play in sends your opponent to the matching board.
#   3. Three in a row on a small board wins it; a won board is closed.
#   4. A small board that fills with no winner is a draw and belongs to nobody.
#   5. If you are sent to a closed (won or full) board, play in any open board.
#   6. Three small boards in a row wins the game.
#   7. If nobody can move and nobody has won, the game is a draw.

X, O = "x", "o"
BOARD_SIZE = 9
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))

OPEN, X_WON, O_WON, DRAW = "open", "x_won", "o_won", "draw"
IN_PROGRESS = "in_progress"


class MoveRejected(Exception):
    """The rules refuse a move. ``code`` is what the API sends back."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class GameState:
    cells: tuple[tuple[str | None, ...], ...]  # cells[board][cell] is "x", "o" or None
    board_statuses: tuple[str, ...]  # "open", "x_won", "o_won" or "draw"
    status: str  # "in_progress", "x_won", "o_won" or "draw"
    next_player: str | None  # None once the game is over
    required_board: int | None  # None: any open board (or the game is over)
    move_count: int


def player_for_sequence(sequence: int) -> str:
    """X moves first, so even-numbered moves are X's."""
    return X if sequence % 2 == 0 else O


def initial_state() -> GameState:
    return GameState(
        cells=((None,) * BOARD_SIZE,) * BOARD_SIZE,
        board_statuses=(OPEN,) * BOARD_SIZE,
        status=IN_PROGRESS,
        next_player=X,
        required_board=None,
        move_count=0,
    )


def _line_winner(owners: Sequence[str | None]) -> str | None:
    for a, b, c in LINES:
        if owners[a] is not None and owners[a] == owners[b] == owners[c]:
            return owners[a]
    return None


def _small_board_status(cells: Sequence[str | None]) -> str:
    winner = _line_winner(cells)
    if winner:
        return X_WON if winner == X else O_WON
    return DRAW if all(cells) else OPEN


def _game_status(board_statuses: Sequence[str]) -> str:
    # A drawn board belongs to nobody, so it never completes a line.
    owners = [X if s == X_WON else O if s == O_WON else None for s in board_statuses]
    winner = _line_winner(owners)
    if winner:
        return X_WON if winner == X else O_WON
    return DRAW if OPEN not in board_statuses else IN_PROGRESS


def validate_move(state: GameState, board: int, cell: int) -> None:
    if state.status != IN_PROGRESS:
        raise MoveRejected("game_over", "This game is already over.")
    if state.required_board is not None and board != state.required_board:
        raise MoveRejected(
            "wrong_board", f"This move must be played in board {state.required_board}."
        )
    if state.board_statuses[board] != OPEN:
        raise MoveRejected("board_closed", f"Board {board} is already finished. Choose an open board.")
    if state.cells[board][cell] is not None:
        raise MoveRejected("cell_occupied", f"Cell {cell} of board {board} is already taken.")


def apply_move(state: GameState, board: int, cell: int) -> GameState:
    """The state after this move. Raises MoveRejected if the rules refuse it."""
    validate_move(state, board, cell)
    player = state.next_player
    assert player is not None  # validate_move guarantees the game is in progress

    small = list(state.cells[board])
    small[cell] = player
    cells = list(state.cells)
    cells[board] = tuple(small)
    statuses = list(state.board_statuses)
    statuses[board] = _small_board_status(small)
    status = _game_status(statuses)

    if status == IN_PROGRESS:
        next_player: str | None = O if player == X else X
        required: int | None = cell if statuses[cell] == OPEN else None
    else:
        next_player, required = None, None

    return GameState(
        cells=tuple(cells),
        board_statuses=tuple(statuses),
        status=status,
        next_player=next_player,
        required_board=required,
        move_count=state.move_count + 1,
    )


def replay(moves: Iterable[tuple[int, int]]) -> GameState:
    """Work out a game's state from its (board, cell) moves, in order."""
    state = initial_state()
    for board, cell in moves:
        state = apply_move(state, board, cell)
    return state


# =============================================================================
# 2. Storage
# =============================================================================
#
# Only names and moves are stored. Everything else (whose turn it is, who won,
# the tallies) is worked out from the moves each time, so it can never be out
# of date.

NAME_MAX_LENGTH = 25
DEFAULT_X_NAME, DEFAULT_O_NAME = "Player X", "Player O"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY,
    player_x_name TEXT    NOT NULL CHECK (length(player_x_name) BETWEEN 1 AND {NAME_MAX_LENGTH}),
    player_o_name TEXT    NOT NULL CHECK (length(player_o_name) BETWEEN 1 AND {NAME_MAX_LENGTH}),
    -- Deleting a game only hides it ("soft delete").
    is_archived   INTEGER NOT NULL DEFAULT 0,
    -- Rematches form a series: the id of the series' first game. NULL on the
    -- first game itself, whose own id is then the series id.
    series_id     INTEGER REFERENCES games (id),
    created_at    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS moves (
    id          INTEGER PRIMARY KEY,
    game_id     INTEGER NOT NULL REFERENCES games (id),
    sequence    INTEGER NOT NULL CHECK (sequence >= 0),
    board_index INTEGER NOT NULL CHECK (board_index BETWEEN 0 AND 8),
    cell_index  INTEGER NOT NULL CHECK (cell_index BETWEEN 0 AND 8),
    created_at  TEXT    NOT NULL,
    -- Two moves submitted at once (a double-click): only the first is saved.
    UNIQUE (game_id, sequence),
    UNIQUE (game_id, board_index, cell_index)
);
CREATE INDEX IF NOT EXISTS ix_moves_game_id ON moves (game_id);
CREATE INDEX IF NOT EXISTS ix_games_series_id ON games (series_id);
"""


def utc_now() -> str:
    """ISO 8601 UTC with microseconds. Sorting these strings sorts by time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """One connection for one piece of work: committed if it succeeds,
        rolled back if it fails, and always closed."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


# =============================================================================
# 3. Service
# =============================================================================


class NotFound(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail


class Conflict(Exception):
    """A rule refused the request (HTTP 409)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail


GAME_COLUMNS = """
    g.id, g.player_x_name, g.player_o_name, g.is_archived, g.series_id, g.created_at,
    COALESCE(g.series_id, g.id) AS series_key,
    COALESCE((SELECT MAX(m.created_at) FROM moves m WHERE m.game_id = g.id), g.created_at)
        AS last_played_at
"""

SORTS = {
    "created_at": "g.created_at ASC, g.id ASC",
    "-created_at": "g.created_at DESC, g.id DESC",
    "last_played_at": "last_played_at ASC, g.id ASC",
    "-last_played_at": "last_played_at DESC, g.id DESC",
}


def _moves_for(conn: sqlite3.Connection, game_ids: list[int]) -> dict[int, list[sqlite3.Row]]:
    found: dict[int, list[sqlite3.Row]] = {gid: [] for gid in game_ids}
    if game_ids:
        marks = ",".join("?" * len(game_ids))
        for row in conn.execute(
            f"SELECT * FROM moves WHERE game_id IN ({marks}) ORDER BY game_id, sequence", game_ids
        ):
            found[row["game_id"]].append(row)
    return found


def _pair(game: sqlite3.Row | dict[str, Any]) -> tuple[str, str]:
    """Both names in lower case and a fixed order, so a rematch (names swapped) matches."""
    a, b = sorted((game["player_x_name"].lower(), game["player_o_name"].lower()))
    return a, b


def _tally(game: dict[str, Any], results: list[tuple[dict[str, Any], str]]) -> dict[str, int]:
    """Wins and draws seen from this game's players. Wins are matched by name,
    because letters swap in a rematch."""
    x_name, o_name = game["player_x_name"].lower(), game["player_o_name"].lower()
    tally = {"player_x_wins": 0, "player_o_wins": 0, "draws": 0, "games_played": 0}
    for other, status in results:
        if status == IN_PROGRESS:
            continue
        tally["games_played"] += 1
        if status == DRAW:
            tally["draws"] += 1
            continue
        winner = (other["player_x_name"] if status == X_WON else other["player_o_name"]).lower()
        # If both players share a name, a win counts for both.
        tally["player_x_wins"] += winner == x_name
        tally["player_o_wins"] += winner == o_name
    return tally


class GameService:
    def __init__(self, db: Database) -> None:
        self.db = db

    # --- reading --------------------------------------------------------------

    def _rows(self, conn: sqlite3.Connection, where: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        rows = conn.execute(f"SELECT {GAME_COLUMNS} FROM games g WHERE {where}", params).fetchall()
        return [dict(r) for r in rows]

    def _load(self, conn: sqlite3.Connection, game_id: int) -> dict[str, Any]:
        rows = self._rows(conn, "g.id = ? AND g.is_archived = 0", [game_id])
        if not rows:
            raise NotFound("game_not_found", f"Game {game_id} does not exist.")
        return rows[0]

    def _views(self, conn: sqlite3.Connection, games: list[dict[str, Any]], detail: bool) -> list[dict[str, Any]]:
        """Turn stored games into what the API returns, with one query for all tallies."""
        if not games:
            return []
        # Every game that could count towards these games' tallies.
        keys = sorted({g["series_key"] for g in games})
        pairs = sorted({_pair(g) for g in games})
        conditions = [f"COALESCE(g.series_id, g.id) IN ({','.join('?' * len(keys))})"]
        params: list[Any] = list(keys)
        for a, b in pairs:
            conditions.append(
                "((lower(g.player_x_name) = ? AND lower(g.player_o_name) = ?)"
                " OR (lower(g.player_x_name) = ? AND lower(g.player_o_name) = ?))"
            )
            params += [a, b, b, a]
        candidates = self._rows(conn, f"g.is_archived = 0 AND ({' OR '.join(conditions)})", params)

        moves = _moves_for(conn, sorted({g["id"] for g in games} | {c["id"] for c in candidates}))
        states = {
            gid: replay((m["board_index"], m["cell_index"]) for m in ms) for gid, ms in moves.items()
        }
        results = [(c, states[c["id"]].status) for c in candidates]

        views = []
        for game in games:
            in_series = [(c, s) for c, s in results if c["series_key"] == game["series_key"]]
            between = [(c, s) for c, s in results if _pair(c) == _pair(game)]
            views.append(
                self._view(
                    game,
                    states[game["id"]],
                    moves[game["id"]],
                    {
                        "series": _tally(game, in_series),
                        "head_to_head": _tally(game, between),
                        "games_in_series": len(in_series),
                    },
                    detail,
                )
            )
        return views

    @staticmethod
    def _view(
        game: dict[str, Any],
        state: GameState,
        moves: list[sqlite3.Row],
        record: dict[str, Any],
        detail: bool,
    ) -> dict[str, Any]:
        view: dict[str, Any] = {
            "id": game["id"],
            "player_x_name": game["player_x_name"],
            "player_o_name": game["player_o_name"],
            "status": state.status,
            "next_player": state.next_player,
            "move_count": state.move_count,
            "created_at": game["created_at"],
            "last_played_at": game["last_played_at"],
            "record": record,
        }
        if detail:
            view.update(
                required_board=state.required_board,
                board_statuses=list(state.board_statuses),
                cells=[list(board) for board in state.cells],
                moves=[
                    {
                        "sequence": m["sequence"],
                        "player": player_for_sequence(m["sequence"]),
                        "board_index": m["board_index"],
                        "cell_index": m["cell_index"],
                        "created_at": m["created_at"],
                    }
                    for m in moves[: state.move_count]
                ],
            )
        return view

    def get_game(self, game_id: int, at_move: int | None = None) -> dict[str, Any]:
        """The game now, or as it stood after its first ``at_move`` moves (for review)."""
        with self.db.connect() as conn:
            view = self._views(conn, [self._load(conn, game_id)], detail=True)[0]
            if at_move is None:
                return view
            moves = _moves_for(conn, [game_id])[game_id]
        if at_move > len(moves):
            raise NotFound(
                "move_not_found",
                f"This game has {len(moves)} moves, so there is no move {at_move}.",
            )
        state = replay((m["board_index"], m["cell_index"]) for m in moves[:at_move])
        position = self._view({**view}, state, moves, view["record"], detail=True)
        return position

    def list_games(self, q: str, sort: str, page: int, page_size: int) -> dict[str, Any]:
        where, params = "g.is_archived = 0", []
        if q:
            where += " AND (g.player_x_name LIKE ? OR g.player_o_name LIKE ?)"
            params += [f"%{q}%", f"%{q}%"]
        with self.db.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM games g WHERE {where}", params).fetchone()[0]
            games = self._rows(
                conn,
                f"{where} ORDER BY {SORTS[sort]} LIMIT ? OFFSET ?",
                [*params, page_size, (page - 1) * page_size],
            )
            items = self._views(conn, games, detail=False)
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    # --- writing --------------------------------------------------------------

    def create_game(self, x_name: str, o_name: str, series_id: int | None = None) -> dict[str, Any]:
        with self.db.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO games (player_x_name, player_o_name, series_id, created_at)"
                " VALUES (?, ?, ?, ?)",
                (x_name, o_name, series_id, utc_now()),
            )
            new_id = cursor.lastrowid
        assert new_id is not None
        return self.get_game(new_id)

    def rename(self, game_id: int, names: dict[str, str]) -> dict[str, Any]:
        # Allowed at any point in the game, including after it has ended.
        with self.db.connect() as conn:
            self._load(conn, game_id)
            for field, value in names.items():
                # Only these two column names can reach here (checked in the API).
                assert field in ("player_x_name", "player_o_name")
                conn.execute(f"UPDATE games SET {field} = ? WHERE id = ?", (value, game_id))
        return self.get_game(game_id)

    def archive(self, game_id: int) -> None:
        with self.db.connect() as conn:
            self._load(conn, game_id)
            conn.execute("UPDATE games SET is_archived = 1 WHERE id = ?", (game_id,))

    def make_move(self, game_id: int, board: int, cell: int) -> dict[str, Any]:
        with self.db.connect() as conn:
            self._load(conn, game_id)
            moves = _moves_for(conn, [game_id])[game_id]
            state = replay((m["board_index"], m["cell_index"]) for m in moves)
            try:
                validate_move(state, board, cell)
            except MoveRejected as refused:
                raise Conflict(refused.code, refused.detail) from refused
            try:
                conn.execute(
                    "INSERT INTO moves (game_id, sequence, board_index, cell_index, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (game_id, state.move_count, board, cell, utc_now()),
                )
            except sqlite3.IntegrityError as clash:
                raise Conflict(
                    "stale_game", "This game was changed by another request. Reload it and try again."
                ) from clash
        return self.get_game(game_id)

    def rematch(self, game_id: int) -> dict[str, Any]:
        """The next game in the series, letters swapped: whoever was O now plays X
        and moves first. Only once the game has ended."""
        with self.db.connect() as conn:
            game = self._load(conn, game_id)
            moves = _moves_for(conn, [game_id])[game_id]
        if replay((m["board_index"], m["cell_index"]) for m in moves).status == IN_PROGRESS:
            raise Conflict("game_not_over", "A rematch can only start once this game has ended.")
        return self.create_game(game["player_o_name"], game["player_x_name"], game["series_key"])


# =============================================================================
# 4. Web API
# =============================================================================
#
#   GET    /api/v1/games                  list (q, sort, page, page_size)
#   POST   /api/v1/games                  start a game       -> 201 + Location
#   GET    /api/v1/games/{id}             one game (?at_move=n for review)
#   PATCH  /api/v1/games/{id}             rename one or both players
#   POST   /api/v1/games/{id}/archive     delete (soft)      -> 204
#   POST   /api/v1/games/{id}/moves       play a move        -> 201
#   POST   /api/v1/games/{id}/rematch     rematch, swapped   -> 201
#
# Every error is {"code": "...", "detail": "..."}: 404 not found, 409 a rule
# refused, 422 the request itself was malformed. Every other path returns the
# web page, which works out what to show from the address.


class Invalid(Exception):
    """The request itself is malformed (HTTP 422)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _query_int(query: dict[str, list[str]], name: str, default: int, low: int, high: int | None) -> int:
    raw = query.get(name, [str(default)])[-1]
    try:
        value = int(raw)
    except ValueError:
        raise Invalid(f"{name}: must be a whole number") from None
    if value < low or (high is not None and value > high):
        limit = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise Invalid(f"{name}: must be {limit}")
    return value


def _name(body: dict[str, Any], field: str) -> str:
    value = body[field]
    if not isinstance(value, str):
        raise Invalid(f"{field}: must be text")
    value = value.strip()
    if not 1 <= len(value) <= NAME_MAX_LENGTH:
        raise Invalid(f"{field}: must be 1 to {NAME_MAX_LENGTH} characters")
    return value


def _position(body: dict[str, Any], field: str) -> int:
    value = body.get(field)
    # bool is a kind of int in Python; true/false are not positions.
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 8:
        raise Invalid(f"{field}: must be a whole number from 0 to 8")
    return value


def _only(body: dict[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(body) - allowed)
    if extra:
        raise Invalid(f"{extra[0]}: unknown field")


class Api:
    ROUTES = [
        ("GET", re.compile(r"^/api/v1/games$"), "list_games"),
        ("POST", re.compile(r"^/api/v1/games$"), "create_game"),
        ("GET", re.compile(r"^/api/v1/games/([^/]+)$"), "get_game"),
        ("PATCH", re.compile(r"^/api/v1/games/([^/]+)$"), "rename"),
        ("POST", re.compile(r"^/api/v1/games/([^/]+)/archive$"), "archive"),
        ("POST", re.compile(r"^/api/v1/games/([^/]+)/moves$"), "move"),
        ("POST", re.compile(r"^/api/v1/games/([^/]+)/rematch$"), "rematch"),
    ]

    def __init__(self, service: GameService) -> None:
        self.service = service

    def handle(
        self, method: str, path: str, query: dict[str, list[str]], body: Any
    ) -> tuple[int, Any, dict[str, str]]:
        """Returns (status, JSON body or None, extra headers)."""
        path_matched = False
        for route_method, pattern, action in self.ROUTES:
            match = pattern.match(path)
            if not match:
                continue
            path_matched = True
            if route_method != method:
                continue
            try:
                args = [self._game_id(g) for g in match.groups()]
                return getattr(self, action)(*args, query=query, body=body)
            except Invalid as error:
                return 422, {"code": "validation_error", "detail": error.detail}, {}
            except NotFound as error:
                return 404, {"code": error.code, "detail": error.detail}, {}
            except Conflict as error:
                return 409, {"code": error.code, "detail": error.detail}, {}
        if path_matched:
            return 405, {"code": "method_not_allowed", "detail": "Method Not Allowed"}, {}
        return 404, {"code": "not_found", "detail": "Not Found"}, {}

    @staticmethod
    def _game_id(raw: str) -> int:
        if not raw.isdigit():
            raise Invalid("game_id: must be a whole number")
        return int(raw)

    @staticmethod
    def _body(body: Any) -> dict[str, Any]:
        if body is None:
            return {}
        if not isinstance(body, dict):
            raise Invalid("body: must be a JSON object")
        return body

    @staticmethod
    def _created(game: dict[str, Any]) -> tuple[int, Any, dict[str, str]]:
        return 201, game, {"Location": f"/api/v1/games/{game['id']}"}

    # --- actions ----------------------------------------------------------------

    def list_games(self, *, query: dict[str, list[str]], body: Any) -> tuple[int, Any, dict[str, str]]:
        q = query.get("q", [""])[-1].strip()
        if len(q) > 50:
            raise Invalid("q: must be 50 characters or fewer")
        sort = query.get("sort", ["-last_played_at"])[-1]
        if sort not in SORTS:
            raise Invalid(f"sort: must be one of {', '.join(SORTS)}")
        page = _query_int(query, "page", 1, 1, None)
        page_size = _query_int(query, "page_size", 25, 1, 100)
        return 200, self.service.list_games(q, sort, page, page_size), {}

    def create_game(self, *, query: dict[str, list[str]], body: Any) -> tuple[int, Any, dict[str, str]]:
        data = self._body(body)
        _only(data, {"player_x_name", "player_o_name"})
        x = _name(data, "player_x_name") if "player_x_name" in data else DEFAULT_X_NAME
        o = _name(data, "player_o_name") if "player_o_name" in data else DEFAULT_O_NAME
        return self._created(self.service.create_game(x, o))

    def get_game(self, game_id: int, *, query: dict[str, list[str]], body: Any) -> tuple[int, Any, dict[str, str]]:
        at_move = _query_int(query, "at_move", 0, 0, None) if "at_move" in query else None
        return 200, self.service.get_game(game_id, at_move), {}

    def rename(self, game_id: int, *, query: dict[str, list[str]], body: Any) -> tuple[int, Any, dict[str, str]]:
        data = self._body(body)
        _only(data, {"player_x_name", "player_o_name"})
        names = {field: _name(data, field) for field in ("player_x_name", "player_o_name") if field in data}
        return 200, self.service.rename(game_id, names), {}

    def archive(self, game_id: int, *, query: dict[str, list[str]], body: Any) -> tuple[int, Any, dict[str, str]]:
        self.service.archive(game_id)
        return 204, None, {}

    def move(self, game_id: int, *, query: dict[str, list[str]], body: Any) -> tuple[int, Any, dict[str, str]]:
        data = self._body(body)
        _only(data, {"board_index", "cell_index"})
        board, cell = _position(data, "board_index"), _position(data, "cell_index")
        game = self.service.make_move(game_id, board, cell)
        return 201, game, {"Location": f"/api/v1/games/{game_id}"}

    def rematch(self, game_id: int, *, query: dict[str, list[str]], body: Any) -> tuple[int, Any, dict[str, str]]:
        return self._created(self.service.rematch(game_id))


class RequestHandler(BaseHTTPRequestHandler):
    api: Api  # set by make_server()
    server_version = "UltimateTicTacToe/1.0"

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str]) -> None:
        self.send_response(status)
        if body or status != 204:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _handle(self) -> None:
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            if self.command not in ("GET", "HEAD"):
                self._send(405, b"", "text/plain", {})
                return
            # The page itself; its script shows the right screen for the address.
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8", {})
            return

        body: Any = None
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(422, {"code": "validation_error", "detail": "body: must be valid JSON"}, {})
                return
        status, payload, headers = self.api.handle(
            self.command, url.path, parse_qs(url.query, keep_blank_values=True), body
        )
        self._json(status, payload, headers)

    def _json(self, status: int, payload: Any, headers: dict[str, str]) -> None:
        body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json", headers)

    do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = do_HEAD = _handle

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (name set by the base class)
        if not getattr(self.server, "quiet", False):
            sys.stderr.write(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")


def make_server(db_path: str, host: str, port: int, quiet: bool = False) -> ThreadingHTTPServer:
    handler = type("Handler", (RequestHandler,), {"api": Api(GameService(Database(db_path)))})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.quiet = quiet  # type: ignore[attr-defined]
    return server


# =============================================================================
# 5. Web page
# =============================================================================
#
# The browser runs this. It is one page: the script reads the address
# (/, /rules, /games, /games/12, /games/12?move=5) and draws that screen,
# talking to the API above. The game rules are NOT repeated here; the page
# only switches off cells that obviously cannot be played, and the server
# decides every move.

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ultimate Tic-Tac-Toe</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%231e293b'/%3E%3Cpath d='M8 8l7 7M15 8l-7 7' stroke='%2393c5fd' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='21' cy='21' r='4.5' fill='none' stroke='%23fca5a5' stroke-width='3'/%3E%3C/svg%3E">
<style>
  :root {
    --ink: #0f172a; --muted: #475569; --faint: #64748b; --line: #e2e8f0; --line-strong: #cbd5e1;
    --page: #f8fafc; --card: #ffffff; --dark: #0f172a; --dark-hover: #334155;
    --board: #1e293b; --cell: #ffffff; --cell-off: #f8fafc; --small: #e2e8f0;
    --highlight: #e5b830; --danger: #b91c1c; --danger-bg: #fef2f2;
    /* Player colors: defaults here, replaced by the players' choice. */
    --x: #1d4ed8; --x-soft: #dbeafe; --o: #b91c1c; --o-soft: #fee2e2;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; color: var(--ink); background: var(--page); line-height: 1.5; }
  a { color: inherit; }
  button, input, select { font: inherit; color: inherit; }
  :focus-visible { outline: 3px solid var(--ink); outline-offset: 2px; }
  [hidden] { display: none !important; }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

  header { background: var(--card); border-bottom: 1px solid var(--line); }
  .bar { max-width: 56rem; margin: 0 auto; padding: .75rem 1rem; display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; justify-content: space-between; }
  .brand { font-weight: 700; font-size: 1.125rem; text-decoration: none; }
  nav.main ul { list-style: none; display: flex; gap: .5rem; margin: 0; padding: 0; }
  nav.main a { text-decoration: none; font-weight: 500; padding: .375rem .75rem; border-radius: .375rem; color: #334155; }
  nav.main a:hover { background: var(--line); }
  nav.main a[aria-current="page"] { background: var(--dark); color: #fff; }
  main { max-width: 56rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }

  h1 { font-size: 1.875rem; margin: 0; line-height: 1.2; }
  .stack > * + * { margin-top: 1.5rem; }
  .row { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; }
  .between { justify-content: space-between; }
  .muted { color: var(--muted); }
  .center { text-align: center; }
  .link { font-weight: 500; text-decoration: underline; text-underline-offset: 4px; background: none; border: 0; padding: 0; cursor: pointer; }

  .btn { display: inline-block; border-radius: .375rem; padding: .5rem 1rem; font-weight: 500; border: 1px solid var(--line-strong); background: var(--card); cursor: pointer; text-decoration: none; }
  .btn:hover { background: #f1f5f9; }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn.primary { background: var(--dark); border-color: var(--dark); color: #fff; }
  .btn.primary:hover { background: var(--dark-hover); }
  .btn.danger { color: var(--danger); }
  .btn.danger-solid { background: var(--danger); border-color: var(--danger); color: #fff; }
  .btn.big { width: 100%; padding: 1rem 1.5rem; font-size: 1.125rem; font-weight: 600; border-radius: .5rem; text-align: center; }

  .panel { background: var(--card); border: 1px solid var(--line); border-radius: .5rem; padding: .75rem 1rem; }
  .status { font-size: 1.125rem; font-weight: 500; margin: 0; }
  .alert { border: 1px solid #fca5a5; background: var(--danger-bg); color: #7f1d1d; border-radius: .375rem; padding: 1rem; display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; justify-content: space-between; }
  .alert p { margin: 0; }
  .empty { border: 1px dashed var(--line-strong); border-radius: .5rem; padding: 2rem; text-align: center; color: var(--muted); }

  label { font-size: .875rem; font-weight: 500; color: #334155; }
  .field { display: flex; flex-direction: column; gap: .25rem; flex: 1; min-width: 10rem; }
  .field input, .field select { border: 1px solid var(--line-strong); border-radius: .375rem; padding: .5rem .75rem; background: var(--card); }
  .err { color: #991b1b; font-size: .875rem; margin: 0; }

  /* Home */
  .home { max-width: 28rem; margin: 0 auto; padding-top: 2rem; text-align: center; }
  .home h1 { font-size: 2.25rem; }
  .home ul { list-style: none; padding: 0; margin: 2rem 0 0; display: grid; gap: .75rem; }

  /* Rules */
  .rules { max-width: 42rem; margin: 0 auto; }
  .rules ol { padding-left: 1.5rem; }
  .rules li { margin: .75rem 0; }
  .rules li::marker { font-weight: 700; }

  /* Saved games */
  .games { list-style: none; padding: 0; margin: 0; display: grid; gap: .75rem; }
  .game { background: var(--card); border: 1px solid var(--line); border-radius: .5rem; padding: 1rem; display: flex; flex-wrap: wrap; gap: .75rem; justify-content: space-between; align-items: center; }
  .game h2 { font-size: 1.125rem; margin: 0; }
  .game h2 a { text-decoration: none; }
  .game h2 a:hover { text-decoration: underline; }
  .small-text { font-size: .875rem; color: var(--faint); margin: 0; }
  .record { font-size: .875rem; margin: .25rem 0 0; }
  .record div { display: flex; flex-wrap: wrap; column-gap: .5rem; }
  .record dt { font-weight: 600; }
  .record dd { margin: 0; }

  /* Names and colors */
  .names { display: flex; flex-wrap: wrap; gap: .75rem; }
  .names .note { width: 100%; font-size: .875rem; color: var(--muted); margin: 0; }
  .name-x label { color: var(--x); font-weight: 600; }
  .name-o label { color: var(--o); font-weight: 600; }
  .name-x input { background: var(--x-soft); border-color: color-mix(in srgb, var(--x) 50%, transparent); }
  .name-o input { background: var(--o-soft); border-color: color-mix(in srgb, var(--o) 50%, transparent); }
  .names input[aria-invalid="true"] { border: 2px solid #dc2626; }
  fieldset.colors { border: 0; padding: 0; margin: 0; }
  fieldset.colors legend { font-size: .875rem; font-weight: 500; color: #334155; margin-bottom: .25rem; }
  .swatches { display: flex; flex-wrap: wrap; gap: .375rem; }
  .swatch { position: relative; width: 2rem; height: 2rem; border-radius: 999px; border: 2px solid #fff; box-shadow: 0 0 0 1px var(--line-strong); display: grid; place-items: center; color: #fff; font-weight: 700; cursor: pointer; }
  .swatch.chosen { border-color: var(--ink); }
  .swatch.taken { opacity: .3; cursor: not-allowed; }
  .swatch:has(input:focus-visible) { outline: 3px solid var(--ink); outline-offset: 2px; }

  /* Board */
  .board-wrap { position: relative; max-width: 36rem; margin: 0 auto; }
  .board { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; background: var(--board); padding: .5rem; border-radius: .5rem; }
  .small-board { position: relative; background: var(--small); border-radius: .375rem; padding: .25rem; }
  .small-board.playable { background: var(--highlight); box-shadow: 0 0 0 4px var(--highlight); }
  .cells { display: grid; grid-template-columns: repeat(3, 1fr); gap: .125rem; }
  .cell { aspect-ratio: 1; border: 0; border-radius: .2rem; background: var(--cell-off); font-weight: 700; font-size: clamp(1rem, 4vw, 1.5rem); display: grid; place-items: center; padding: 0; }
  .cell:not(:disabled) { background: var(--cell); cursor: pointer; }
  .cell:not(:disabled):hover { background: #f1f5f9; }
  .cell.x { color: var(--x); } .cell.o { color: var(--o); }
  .cell.last { text-decoration: underline; text-decoration-thickness: 4px; text-underline-offset: 4px; }
  .won { position: absolute; inset: 0; display: grid; place-items: center; border-radius: .375rem; font-weight: 900; font-size: clamp(3rem, 14vw, 6rem); opacity: .9; }
  .won.x { background: var(--x-soft); color: var(--x); }
  .won.o { background: var(--o-soft); color: var(--o); }
  .won.draw { background: rgb(203 213 225 / .85); color: #1e293b; font-size: 1.125rem; }
  .overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; padding: 1rem; background: rgb(15 23 42 / .5); border-radius: .5rem; }
  .card { width: 100%; max-width: 24rem; background: var(--card); border-radius: .5rem; padding: 1.5rem; text-align: center; box-shadow: 0 20px 25px -5px rgb(0 0 0 / .2); }
  .card > * + * { margin-top: 1rem; }
  .card h2 { font-size: 1.5rem; margin: 0; }
  .card .record { background: #f8fafc; border-radius: .375rem; padding: .5rem .75rem; text-align: left; }
  .card .buttons { display: grid; gap: .5rem; }
  .review { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; justify-content: center; }
  .review p { margin: 0; min-width: 8rem; text-align: center; font-weight: 600; }
</style>
</head>
<body>
<header>
  <div class="bar">
    <a class="brand" href="/" data-link>Ultimate Tic-Tac-Toe</a>
    <nav class="main" aria-label="Main">
      <ul>
        <li><a href="/games" data-link>Saved Games</a></li>
        <li><a href="/rules" data-link data-from>Rules</a></li>
      </ul>
    </nav>
  </div>
</header>
<main id="main"></main>
<script>
"use strict";

// ---------------------------------------------------------------- basics

const main = document.getElementById("main");
const POSITIONS = ["top left", "top middle", "top right", "middle left", "center",
  "middle right", "bottom left", "bottom middle", "bottom right"];
const NAME_MAX = 25;
const PAGE_SIZE = 10;

const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const esc = (value) => String(value).replace(/[&<>"']/g, (ch) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const dateFormat = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

// Browser storage can be switched off (private browsing); the app still works.
const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* not remembered */ } },
};

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 204) return null;
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) {
    const good = data && typeof data.code === "string" && typeof data.detail === "string";
    const error = new Error(good ? data.detail : "Something went wrong. Please try again.");
    error.status = response.status;
    error.code = good ? data.code : "unexpected_error";
    throw error;
  }
  return data;
}

const loading = (label) => `<p role="status" class="muted center">${esc(label)}</p>`;

function showError(element, message, retry) {
  element.innerHTML = `<div role="alert" class="alert"><p>${esc(message)}</p>` +
    (retry ? `<button type="button" class="btn">Try again</button>` : "") + `</div>`;
  if (retry) element.querySelector("button").addEventListener("click", retry);
}

// ---------------------------------------------------------------- colors

const PALETTE = [
  { id: "blue", name: "Blue", main: "#1d4ed8", soft: "#dbeafe" },
  { id: "red", name: "Red", main: "#b91c1c", soft: "#fee2e2" },
  { id: "green", name: "Green", main: "#15803d", soft: "#dcfce7" },
  { id: "purple", name: "Purple", main: "#7e22ce", soft: "#f3e8ff" },
  { id: "orange", name: "Orange", main: "#c2410c", soft: "#ffedd5" },
  { id: "teal", name: "Teal", main: "#0f766e", soft: "#ccfbf1" },
  { id: "pink", name: "Pink", main: "#be185d", soft: "#fce7f3" },
  { id: "gray", name: "Gray", main: "#334155", soft: "#e2e8f0" },
];
const COLORS_KEY = "ultimate-ttt:player-colors";
const colorById = (id) => PALETTE.find((c) => c.id === id);

function loadColors() {
  try {
    const saved = JSON.parse(storage.get(COLORS_KEY) || "null");
    if (saved && colorById(saved.x) && colorById(saved.o) && saved.x !== saved.o) return saved;
  } catch { /* use the defaults */ }
  return { x: "blue", o: "red" };
}
let colors = loadColors();

function applyColors() {
  const root = document.documentElement.style;
  for (const player of ["x", "o"]) {
    const color = colorById(colors[player]);
    root.setProperty(`--${player}`, color.main);
    root.setProperty(`--${player}-soft`, color.soft);
  }
}
applyColors();

// ---------------------------------------------------------------- routing

let routeToken = 0;

function navigate(url, { replace = false, state = null } = {}) {
  history[replace ? "replaceState" : "pushState"](state, "", url);
  route();
}

document.addEventListener("click", (event) => {
  const link = event.target.closest("a[data-link]");
  if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  // Links to the rules remember where they came from, for the return button.
  const state = link.hasAttribute("data-from")
    ? { from: link.dataset.from || location.pathname + location.search } : null;
  navigate(link.getAttribute("href"), { state });
});
window.addEventListener("popstate", route);

function route() {
  const token = ++routeToken;
  const path = location.pathname;
  for (const link of document.querySelectorAll("nav.main a")) {
    const here = link.getAttribute("href") === path;
    if (here) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  }
  window.scrollTo(0, 0);
  if (path === "/") return homePage();
  if (path === "/rules") return rulesPage();
  if (path === "/games") return gamesPage(token);
  const match = path.match(/^\/games\/([^/]+)$/);
  if (match) return gamePage(match[1], token);
  return notFoundPage();
}

function setTitle(name) {
  document.title = name ? `${name} · Ultimate Tic-Tac-Toe` : "Ultimate Tic-Tac-Toe";
}

// ---------------------------------------------------------------- words

function playerLabel(game, player) {
  const name = player === "x" ? game.player_x_name : game.player_o_name;
  return `${name} (${player.toUpperCase()})`;
}

function describeStatus(game) {
  if (game.status === "x_won") return `${playerLabel(game, "x")} won`;
  if (game.status === "o_won") return `${playerLabel(game, "o")} won`;
  if (game.status === "draw") return "Draw";
  return game.next_player ? `${playerLabel(game, game.next_player)} to move` : "In progress";
}

function statusBanner(game) {
  if (game.status === "x_won") return `${playerLabel(game, "x")} wins the game!`;
  if (game.status === "o_won") return `${playerLabel(game, "o")} wins the game!`;
  if (game.status === "draw") return "The game is a draw. No moves are left.";
  const where = game.required_board === null
    ? "any open board" : `the ${POSITIONS[game.required_board]} board`;
  return `${playerLabel(game, game.next_player)} to move. Play in ${where}.`;
}

function outcome(game) {
  if (game.status === "x_won") return `${playerLabel(game, "x")} wins!`;
  if (game.status === "o_won") return `${playerLabel(game, "o")} wins!`;
  return "It's a draw!";
}

function describeReviewStep(position, total) {
  const last = position.moves[position.moves.length - 1];
  if (!last) return `Start of the game. ${playerLabel(position, "x")} moves first.`;
  let text = `Move ${position.moves.length} of ${total}: ${playerLabel(position, last.player)} ` +
    `played the ${POSITIONS[last.cell_index]} square of the ${POSITIONS[last.board_index]} board`;
  const closed = position.board_statuses[last.board_index];
  if (closed === `${last.player}_won`) text += ", winning that board";
  else if (closed === "draw") text += ", filling that board with no winner";
  text += ".";
  if (position.status === "x_won" || position.status === "o_won") {
    text += ` ${playerLabel(position, position.status === "x_won" ? "x" : "o")} wins the game.`;
  } else if (position.status === "draw") {
    text += " The game is a draw.";
  }
  return text;
}

function formatTally(tally, xName, oName) {
  return [`${xName} ${plural(tally.player_x_wins, "win")}`,
    `${oName} ${plural(tally.player_o_wins, "win")}`, plural(tally.draws, "draw")].join(" · ");
}

// The two tallies: this rematch series, and every game between these names.
function recordHtml(game) {
  const r = game.record;
  if (r.head_to_head.games_played === 0 && r.games_in_series <= 1) return "";
  return `<section aria-label="Record"><dl class="record">
    <div><dt>This series (${plural(r.games_in_series, "game")}):</dt> <dd>${esc(formatTally(r.series, game.player_x_name, game.player_o_name))}</dd></div>
    <div><dt>All games between them:</dt> <dd>${esc(formatTally(r.head_to_head, game.player_x_name, game.player_o_name))}</dd></div>
  </dl></section>`;
}

function validateName(value) {
  const trimmed = value.trim();
  if (!trimmed) return "Enter a name.";
  if (trimmed.length > NAME_MAX) return `Use ${NAME_MAX} characters or fewer.`;
  return null;
}

// ---------------------------------------------------------------- home

const RULES_KEY = "ultimate-ttt:rules-seen";

function homePage() {
  // The very first visit in this browser shows the rules instead.
  if (storage.get(RULES_KEY) !== "true") {
    storage.set(RULES_KEY, "true");
    return navigate("/rules", { replace: true, state: { from: "/" } });
  }
  setTitle("");
  main.innerHTML = `<div class="home">
    <h1>Ultimate Tic-Tac-Toe</h1>
    <p class="muted">Nine games of tic-tac-toe inside one. Two players, one screen.</p>
    <nav aria-label="Start"><ul>
      <li><button type="button" class="btn big primary" id="new-game">New game</button></li>
      <li><a class="btn big" href="/games" data-link>Saved Games</a></li>
      <li><a class="btn big" href="/rules" data-link data-from="/">Rules</a></li>
    </ul></nav>
    <div id="home-error" style="text-align:left;margin-top:1rem"></div>
  </div>`;
  const button = document.getElementById("new-game");
  button.addEventListener("click", () => startGame(button, document.getElementById("home-error"), "Could not start a new game."));
}

async function startGame(button, errorBox, failure) {
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    const game = await api("POST", "/api/v1/games", {});
    navigate(`/games/${game.id}`);
  } catch (error) {
    button.disabled = false;
    button.textContent = "New game";
    showError(errorBox, `${failure} ${error.message}`);
  }
}

// ---------------------------------------------------------------- rules

function backLink() {
  const from = history.state && typeof history.state.from === "string" ? history.state.from : null;
  if (!from || !from.startsWith("/") || from.startsWith("//")) return { to: "/", label: "Back to home" };
  if (/^\/games\/[^/?#]+/.test(from)) return { to: from, label: "Back to game" };
  if (from.startsWith("/games")) return { to: from, label: "Back to Saved Games" };
  return { to: "/", label: "Back to home" };
}

function rulesPage() {
  setTitle("How to play");
  const back = backLink();
  main.innerHTML = `<article class="rules stack">
    <a class="btn" href="${esc(back.to)}" data-link><span aria-hidden="true">← </span>${esc(back.label)}</a>
    <h1>How to play</h1>
    <p class="muted" style="font-size:1.125rem">Ultimate Tic-Tac-Toe is nine games of tic-tac-toe inside one big one. Two players share this screen and take turns.</p>
    <ol>
      <li><strong>X goes first</strong> and may play in any square of any small board.</li>
      <li><strong>Where you play decides where your opponent plays.</strong> If you play in the top-right square of a small board, your opponent must play next in the top-right small board. The board they must use is highlighted and labelled “play here”.</li>
      <li><strong>Win a small board</strong> by getting three in a row inside it. That board is then yours, and nobody can play in it again.</li>
      <li><strong>A small board that fills up with no winner</strong> is a draw. It belongs to nobody.</li>
      <li><strong>If you are sent to a board that is won or full</strong>, you may play in any open board instead.</li>
      <li><strong>Win the game</strong> by winning three small boards in a row: across, down, or diagonally.</li>
      <li><strong>If no moves are left</strong> and nobody has three boards in a row, the game is a draw.</li>
    </ol>
    <p class="muted">Every move is saved as you play, so you can leave a game and come back to it from <strong>Saved Games</strong>. You can change the players’ names at any time.</p>
    <a class="btn primary" href="${esc(back.to)}" data-link>${esc(back.label)}</a>
  </article>`;
}

function notFoundPage() {
  setTitle("Page not found");
  main.innerHTML = `<div class="stack"><h1>Page not found</h1><p>There is nothing at this address.</p>
    <a class="link" href="/games" data-link>Go to Saved Games</a></div>`;
}

// ---------------------------------------------------------------- saved games

const SORT_OPTIONS = [
  ["-last_played_at", "Last played, newest first"], ["last_played_at", "Last played, oldest first"],
  ["-created_at", "Created, newest first"], ["created_at", "Created, oldest first"],
];

async function gamesPage(token) {
  setTitle("Saved Games");
  const params = new URLSearchParams(location.search);
  const q = params.get("q") || "";
  const sort = SORT_OPTIONS.some(([v]) => v === params.get("sort")) ? params.get("sort") : "-last_played_at";
  const pageNumber = Math.max(1, parseInt(params.get("page") || "1", 10) || 1);

  main.innerHTML = `<div class="stack">
    <div class="row between"><h1>Saved Games</h1>
      <button type="button" class="btn primary" id="new-game">New game</button></div>
    <div id="create-error"></div>
    <div class="row" style="align-items:flex-end">
      <form role="search" id="search" class="row" style="flex:1;align-items:flex-end">
        <div class="field"><label for="game-search">Search player names</label>
          <input id="game-search" name="q" type="search" value="${esc(q)}"></div>
        <button type="submit" class="btn">Search</button>
      </form>
      <div class="field" style="flex:0 1 auto"><label for="game-sort">Sort by</label>
        <select id="game-sort">${SORT_OPTIONS.map(([value, label]) =>
          `<option value="${value}"${value === sort ? " selected" : ""}>${label}</option>`).join("")}</select></div>
    </div>
    <div id="list">${loading("Loading saved games…")}</div>
  </div>`;

  const setParams = (changes) => {
    const next = new URLSearchParams(location.search);
    for (const [key, value] of Object.entries(changes)) {
      if (value === null || value === "") next.delete(key); else next.set(key, value);
    }
    const query = next.toString();
    navigate(`/games${query ? `?${query}` : ""}`);
  };
  const newGame = document.getElementById("new-game");
  newGame.addEventListener("click", () => startGame(newGame, document.getElementById("create-error"), "Could not start a new game."));
  document.getElementById("search").addEventListener("submit", (event) => {
    event.preventDefault();
    setParams({ q: document.getElementById("game-search").value.trim(), page: null });
  });
  document.getElementById("game-sort").addEventListener("change", (event) => setParams({ sort: event.target.value, page: null }));

  const list = document.getElementById("list");
  const load = async () => {
    list.innerHTML = loading("Loading saved games…");
    let data;
    try {
      const query = new URLSearchParams({ sort, page: String(pageNumber), page_size: String(PAGE_SIZE) });
      if (q) query.set("q", q);
      data = await api("GET", `/api/v1/games?${query}`);
    } catch (error) {
      if (token !== routeToken) return;
      return showError(list, `Could not load saved games. ${error.message}`, load);
    }
    if (token !== routeToken) return;
    if (data.total === 0) {
      list.innerHTML = `<p class="empty">${q ? `No saved games match “${esc(q)}”.` : "No saved games yet. Start a new game to play."}</p>`;
      return;
    }
    if (data.items.length === 0) {
      list.innerHTML = `<div class="empty"><p>There are no games on this page.</p>
        <button type="button" class="link" id="first-page">Go to the first page</button></div>`;
      document.getElementById("first-page").addEventListener("click", () => setParams({ page: null }));
      return;
    }
    const pages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
    list.innerHTML = `<ul class="games">${data.items.map(gameItemHtml).join("")}</ul>` +
      (pages > 1 || pageNumber > 1 ? `<nav aria-label="Pages" class="row" style="justify-content:center;margin-top:1rem">
        <button type="button" class="btn" id="prev"${pageNumber <= 1 ? " disabled" : ""}>Previous</button>
        <p style="margin:0">Page ${pageNumber} of ${pages}</p>
        <button type="button" class="btn" id="next"${pageNumber >= pages ? " disabled" : ""}>Next</button></nav>` : "");
    document.getElementById("prev")?.addEventListener("click", () => setParams({ page: pageNumber - 1 === 1 ? null : String(pageNumber - 1) }));
    document.getElementById("next")?.addEventListener("click", () => setParams({ page: String(pageNumber + 1) }));
    list.querySelectorAll("li.game").forEach((item) => wireGameItem(item, data.items.find((g) => g.id === Number(item.dataset.id)), load));
  };
  load();
}

function gameItemHtml(game) {
  const title = `${game.player_x_name} vs ${game.player_o_name}`;
  const finished = game.status !== "in_progress";
  const open = finished
    ? `<a class="btn" href="/games/${game.id}?move=${game.move_count}" data-link aria-label="Review ${esc(title)}">Review</a>`
    : `<a class="btn primary" href="/games/${game.id}" data-link aria-label="Resume ${esc(title)}">Resume</a>`;
  return `<li class="game" data-id="${game.id}">
    <div>
      <h2><a href="/games/${game.id}" data-link>${esc(title)}</a></h2>
      <p style="margin:0">${esc(describeStatus(game))}</p>
      <p class="small-text">Last played ${esc(dateFormat.format(new Date(game.last_played_at)))} · ${plural(game.move_count, "move")}</p>
      ${recordHtml(game)}
    </div>
    <div class="actions row">${open}
      <button type="button" class="btn danger" data-delete aria-label="Delete ${esc(title)}">Delete</button></div>
    <p role="alert" class="err" data-delete-error hidden style="width:100%"></p>
  </li>`;
}

// Deleting asks first, because a deleted game cannot be brought back.
function wireGameItem(item, game, reload) {
  const title = `${game.player_x_name} vs ${game.player_o_name}`;
  const actions = item.querySelector(".actions");
  const original = actions.innerHTML;
  const wire = () => {
    actions.querySelector("[data-delete]").addEventListener("click", () => {
      actions.innerHTML = `<div role="group" aria-label="Confirm deleting ${esc(title)}" class="row">
        <p style="margin:0;font-size:.875rem;font-weight:500">Delete this game? This cannot be undone.</p>
        <button type="button" class="btn danger-solid" data-confirm>Delete game</button>
        <button type="button" class="btn" data-cancel>Cancel</button></div>`;
      actions.querySelector("[data-cancel]").addEventListener("click", () => { actions.innerHTML = original; wire(); });
      actions.querySelector("[data-confirm]").addEventListener("click", async (event) => {
        event.target.disabled = true;
        try {
          await api("POST", `/api/v1/games/${game.id}/archive`);
          reload();
        } catch (error) {
          event.target.disabled = false;
          const box = item.querySelector("[data-delete-error]");
          box.hidden = false;
          box.textContent = `Could not delete this game. ${error.message}`;
        }
      });
    });
  };
  wire();
}

// ---------------------------------------------------------------- one game

function gameNotFound() {
  setTitle("Game not found");
  main.innerHTML = `<div class="stack"><h1>Game not found</h1><p>This game does not exist or has been deleted.</p>
    <a class="link" href="/games" data-link>Back to Saved Games</a></div>`;
}

async function gamePage(rawId, token) {
  const id = Number(rawId);
  if (!Number.isInteger(id) || id < 1) return gameNotFound();
  setTitle("Game");
  main.innerHTML = loading("Loading game…");
  let game;
  try {
    game = await api("GET", `/api/v1/games/${id}`);
  } catch (error) {
    if (token !== routeToken) return;
    if (error.status === 404) return gameNotFound();
    return showError(main, `Could not load this game. ${error.message}`, route);
  }
  if (token !== routeToken) return;
  new GameScreen(game, token).start();
}

class GameScreen {
  constructor(game, token) {
    this.game = game;
    this.token = token;
    this.moving = false;
    this.savingNames = 0;
    this.review = null; // { move, position } while reviewing
    this.reviewToken = 0;
  }

  start() {
    const g = this.game;
    main.innerHTML = `<div class="stack">
      <div class="row between"><h1 id="title"></h1><a class="link" href="/games" data-link>Back to Saved Games</a></div>
      <form class="names" id="names" aria-label="Player names" novalidate>
        ${this.nameField("x")}${this.nameField("o")}
        <p class="note" id="names-note" aria-live="polite">Names are saved automatically.</p>
        <p class="err" id="names-alert" role="alert" hidden style="width:100%"></p>
      </form>
      <div><button type="button" class="btn" id="colors-toggle" aria-expanded="false" aria-controls="colors-panel" style="font-size:.875rem;padding:.375rem .75rem">Change colors</button>
        <div id="colors-panel" class="row" style="gap:1.5rem;margin-top:.75rem" hidden></div></div>
      <div id="status-area"></div>
      <div id="record-area"></div>
      <div id="move-error"></div>
      <div class="board-wrap"><div id="board"></div><div id="overlay"></div></div>
    </div>`;
    document.getElementById("names").addEventListener("submit", (e) => e.preventDefault());
    for (const player of ["x", "o"]) {
      const input = document.getElementById(`name-${player}`);
      input.value = player === "x" ? g.player_x_name : g.player_o_name;
      input.addEventListener("blur", () => this.saveName(player));
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.saveName(player); } });
    }
    const toggle = document.getElementById("colors-toggle");
    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") !== "true";
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = open ? "Hide colors" : "Change colors";
      document.getElementById("colors-panel").hidden = !open;
      if (open) this.renderColors();
    });
    document.getElementById("board").addEventListener("click", (e) => {
      const cell = e.target.closest("button[data-board]");
      if (cell && !cell.disabled) this.play(Number(cell.dataset.board), Number(cell.dataset.cell));
    });
    this.syncReview();
  }

  nameField(player) {
    return `<div class="field name-${player}"><label for="name-${player}">Player ${player.toUpperCase()}</label>
      <input id="name-${player}" type="text" autocomplete="off">
      <p class="err" id="name-${player}-error" hidden></p></div>`;
  }

  // --- drawing ---------------------------------------------------------------

  render() {
    const g = this.game;
    document.getElementById("title").textContent = `${g.player_x_name} vs ${g.player_o_name}`;
    setTitle(`${g.player_x_name} vs ${g.player_o_name}`);
    this.renderStatus();
    document.getElementById("record-area").innerHTML = recordHtml(g)
      ? `<div class="panel">${recordHtml(g)}</div>` : "";
    this.renderBoard();
    this.renderOverlay();
  }

  renderStatus() {
    const area = document.getElementById("status-area");
    if (!this.review) {
      area.innerHTML = `<p role="status" class="panel status">${esc(statusBanner(this.game))}</p>`;
      return;
    }
    const { move } = this.review;
    const total = this.game.move_count;
    const step = (label, name, target, disabled, arrow, after) =>
      `<button type="button" class="btn" aria-label="${name}" data-step="${target}"${disabled ? " disabled" : ""}>` +
      (after ? `${label}<span aria-hidden="true"> ${arrow}</span>` : `<span aria-hidden="true">${arrow} </span>${label}`) + `</button>`;
    area.innerHTML = `<nav aria-label="Review moves" class="panel review">
        ${step("First", "First move", 0, move === 0, "⏮", false)}
        ${step("Previous", "Previous move", move - 1, move === 0, "◀", false)}
        <p>Move ${move} of ${total}</p>
        ${step("Next", "Next move", move + 1, move === total, "▶", true)}
        ${step("Last", "Last move", total, move === total, "⏭", true)}
        <button type="button" class="btn primary" data-done>Done reviewing</button>
      </nav><div id="review-status" style="margin-top:1.5rem"></div>`;
    area.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", () => this.showMove(Number(b.dataset.step))));
    area.querySelector("[data-done]").addEventListener("click", () => this.showMove(null));
    const statusBox = document.getElementById("review-status");
    if (this.review.error) {
      showError(statusBox, `Could not load this move. ${this.review.error}`, () => this.syncReview());
    } else if (this.review.position) {
      statusBox.innerHTML = `<p role="status" class="panel status">${esc(describeReviewStep(this.review.position, total))}</p>`;
    } else {
      statusBox.innerHTML = loading("Loading move…");
    }
  }

  renderBoard() {
    const shown = this.review ? this.review.position : this.game;
    const board = document.getElementById("board");
    if (!shown) return;
    const disabled = Boolean(this.review) || this.moving;
    const last = shown.moves[shown.moves.length - 1];
    let html = `<div class="board">`;
    for (let b = 0; b < 9; b++) {
      const status = shown.board_statuses[b];
      const playable = shown.status === "in_progress" && status === "open" &&
        (shown.required_board === null || shown.required_board === b);
      const describe = status === "x_won" ? "won by X" : status === "o_won" ? "won by O"
        : status === "draw" ? "drawn" : playable ? "play here" : "open";
      const boardName = `${cap(POSITIONS[b])} board`;
      html += `<section class="small-board${playable ? " playable" : ""}" aria-label="${boardName}, ${describe}"><div class="cells">`;
      for (let c = 0; c < 9; c++) {
        const who = shown.cells[b][c];
        const isLast = last && last.board_index === b && last.cell_index === c;
        const canPlay = playable && !who && !disabled;
        const label = `${boardName}, ${POSITIONS[c]}, ${who ? who.toUpperCase() : "empty"}${isLast ? ", last move" : ""}`;
        html += `<button type="button" class="cell${who ? ` ${who}` : ""}${isLast ? " last" : ""}" data-board="${b}" data-cell="${c}" aria-label="${label}"${canPlay ? "" : " disabled"}>${who ? who.toUpperCase() : ""}</button>`;
      }
      html += `</div>`;
      if (status === "x_won" || status === "o_won") {
        const letter = status === "x_won" ? "x" : "o";
        html += `<div class="won ${letter}" aria-hidden="true">${letter.toUpperCase()}</div>`;
      } else if (status === "draw") {
        html += `<div class="won draw" aria-hidden="true">Drawn</div>`;
      }
      html += `</section>`;
    }
    board.innerHTML = html + `</div>`;
  }

  // When a game ends: a card over the board with the result, the tallies,
  // a rematch with letters swapped, and a way into the move-by-move review.
  renderOverlay() {
    const g = this.game;
    const overlay = document.getElementById("overlay");
    if (g.status === "in_progress" || this.review) { overlay.innerHTML = ""; return; }
    overlay.innerHTML = `<div class="overlay"><section class="card" aria-label="Rematch">
      <h2>${esc(outcome(g))}</h2>
      ${recordHtml(g)}
      <p>Play again with letters swapped: <strong>${esc(g.player_o_name)}</strong> plays X and goes first, <strong>${esc(g.player_x_name)}</strong> plays O.</p>
      <div class="buttons">
        <button type="button" class="btn primary" data-rematch style="padding:.625rem 1rem;font-weight:600">Rematch</button>
        <button type="button" class="btn" data-review>Review moves</button>
      </div>
      <div data-rematch-error style="text-align:left"></div>
    </section></div>`;
    overlay.querySelector("[data-review]").addEventListener("click", () => this.showMove(g.move_count));
    const rematch = overlay.querySelector("[data-rematch]");
    rematch.addEventListener("click", async () => {
      rematch.disabled = true;
      rematch.textContent = "Starting rematch…";
      try {
        const next = await api("POST", `/api/v1/games/${g.id}/rematch`);
        navigate(`/games/${next.id}`);
      } catch (error) {
        rematch.disabled = false;
        rematch.textContent = "Rematch";
        showError(overlay.querySelector("[data-rematch-error]"), `Could not start the rematch. ${error.message}`);
      }
    });
  }

  renderColors() {
    const panel = document.getElementById("colors-panel");
    const picker = (player) => {
      const other = player === "x" ? "o" : "x";
      const name = player === "x" ? this.game.player_x_name : this.game.player_o_name;
      return `<fieldset class="colors"><legend>Color for ${esc(name)} (${player.toUpperCase()})</legend><div class="swatches">` +
        PALETTE.map((c) => {
          const taken = colors[other] === c.id;
          const chosen = colors[player] === c.id;
          const label = taken ? `${c.name}, used by the other player` : c.name;
          return `<label class="swatch${chosen ? " chosen" : ""}${taken ? " taken" : ""}" style="background:${c.main}" title="${taken ? `${c.name} (used by the other player)` : c.name}">
            <input class="sr-only" type="radio" name="color-${player}" value="${c.id}" aria-label="${label}"${chosen ? " checked" : ""}${taken ? " disabled" : ""}>
            ${chosen ? '<span aria-hidden="true">✓</span>' : ""}${taken ? '<span aria-hidden="true">✕</span>' : ""}</label>`;
        }).join("") + `</div></fieldset>`;
    };
    panel.innerHTML = picker("x") + picker("o");
    panel.querySelectorAll("input[type=radio]").forEach((input) => input.addEventListener("change", () => {
      const player = input.name.slice(-1);
      const other = player === "x" ? "o" : "x";
      if (colors[other] === input.value) return;
      colors = { ...colors, [player]: input.value };
      storage.set(COLORS_KEY, JSON.stringify(colors));
      applyColors();
      this.renderColors();
      panel.querySelector(`input[name="color-${player}"][value="${input.value}"]`).focus();
    }));
  }

  // --- names: saved as soon as a player leaves the box --------------------------

  async saveName(player) {
    const field = player === "x" ? "player_x_name" : "player_o_name";
    const input = document.getElementById(`name-${player}`);
    const errorBox = document.getElementById(`name-${player}-error`);
    const problem = validateName(input.value);
    errorBox.hidden = !problem;
    errorBox.textContent = problem || "";
    if (problem) {
      input.setAttribute("aria-invalid", "true");
      input.setAttribute("aria-describedby", errorBox.id);
      return;
    }
    input.removeAttribute("aria-invalid");
    input.removeAttribute("aria-describedby");
    const value = input.value.trim();
    if (value === this.game[field]) return;

    this.savingNames++;
    this.updateNamesNote();
    const alertBox = document.getElementById("names-alert");
    try {
      const saved = await api("PATCH", `/api/v1/games/${this.game.id}`, { [field]: value });
      if (this.token !== routeToken) return;
      // Only take the name that was saved: a move may have finished meanwhile.
      this.game = { ...this.game, [field]: saved[field] };
      alertBox.hidden = true;
      this.render();
    } catch (error) {
      alertBox.hidden = false;
      alertBox.textContent = `Could not save the names. ${error.message}`;
    } finally {
      this.savingNames--;
      this.updateNamesNote();
    }
  }

  updateNamesNote() {
    document.getElementById("names-note").textContent =
      this.savingNames > 0 ? "Saving names…" : "Names are saved automatically.";
  }

  // --- playing -----------------------------------------------------------------

  async play(board, cell) {
    if (this.moving) return;
    this.moving = true;
    document.getElementById("move-error").innerHTML = "";
    this.renderBoard();
    try {
      const after = await api("POST", `/api/v1/games/${this.game.id}/moves`, { board_index: board, cell_index: cell });
      if (this.token !== routeToken) return;
      // Take the new board but keep the names on screen: a name save may have
      // finished meanwhile, and this reply could be older than it.
      this.game = { ...after, player_x_name: this.game.player_x_name, player_o_name: this.game.player_o_name };
    } catch (error) {
      if (this.token !== routeToken) return;
      showError(document.getElementById("move-error"), error.message);
      try { this.game = await api("GET", `/api/v1/games/${this.game.id}`); } catch { /* keep what we have */ }
    } finally {
      this.moving = false;
    }
    if (this.token !== routeToken) return;
    this.render();
    // Keyboard players carry on from where the next move must go.
    const next = document.querySelector("#board .small-board.playable .cell:not(:disabled)");
    if (next && document.activeElement === document.body) next.focus();
  }

  // --- reviewing move by move (?move=n in the address) ---------------------------

  showMove(move) {
    const url = move === null ? `/games/${this.game.id}` : `/games/${this.game.id}?move=${move}`;
    history.replaceState(history.state, "", url);
    this.syncReview();
  }

  async syncReview() {
    const raw = new URLSearchParams(location.search).get("move");
    const wanted = raw === null ? NaN : Number(raw);
    if (!Number.isInteger(wanted) || wanted < 0) {
      this.review = null;
      return this.render();
    }
    const total = this.game.move_count;
    const move = Math.min(wanted, total); // past the end: show the last move
    const token = ++this.reviewToken;
    const previous = this.review && this.review.position;
    this.review = { move, position: move === total ? this.game : previous, error: null };
    this.render();
    if (move === total) return;
    try {
      const position = await api("GET", `/api/v1/games/${this.game.id}?at_move=${move}`);
      if (token !== this.reviewToken || this.token !== routeToken) return;
      this.review = { move, position, error: null };
    } catch (error) {
      if (token !== this.reviewToken || this.token !== routeToken) return;
      this.review = { move, position: previous, error: error.message };
    }
    this.render();
  }
}

route();
</script>
</body>
</html>
"""


# =============================================================================
# 6. Self-tests (python ult_ttt.py --test)
# =============================================================================

# Recorded games. Positions are (board, cell).

# X wins with the top row of boards (0, 1, 2); O wins boards 3 and 4 on the way.
X_WINS_TOP_ROW = [
    (0, 3), (3, 0), (0, 4), (4, 0), (0, 5),
    (5, 1),
    (1, 3), (3, 1), (1, 4), (4, 1), (1, 5),
    (5, 2),
    (2, 3), (3, 2), (2, 4), (4, 2), (2, 5),
]  # fmt: skip

# The same game with one opening move, so the roles swap and O wins.
O_WINS = [(8, 0), *X_WINS_TOP_ROW]

# 39 moves. The last move sends O to board 5, which is full with no winner.
SENT_TO_DRAWN_BOARD = [
    (8, 2), (2, 5), (5, 7), (7, 6), (6, 4), (4, 3), (3, 5), (5, 8), (8, 5), (5, 1),
    (1, 5), (5, 6), (6, 7), (7, 8), (8, 0), (0, 1), (1, 6), (6, 1), (1, 8), (8, 6),
    (6, 6), (6, 5), (5, 4), (4, 6), (6, 8), (8, 7), (7, 5), (5, 5), (5, 3), (3, 6),
    (5, 2), (2, 8), (8, 3), (3, 0), (0, 8), (8, 4), (4, 5), (5, 0), (0, 5),
]  # fmt: skip

# 63 moves, still in progress. Diagonal 2-4-6 is X-won, drawn, X-won: not a line.
X_DRAW_X_DIAGONAL = [
    (3, 3), (3, 2), (2, 5), (5, 7), (7, 1), (1, 0), (0, 7), (7, 5), (5, 3), (3, 1),
    (1, 8), (8, 8), (8, 7), (7, 4), (4, 2), (2, 3), (3, 8), (8, 1), (1, 7), (7, 7),
    (7, 6), (6, 0), (0, 1), (1, 2), (2, 8), (8, 5), (5, 0), (0, 3), (3, 0), (0, 4),
    (4, 8), (8, 6), (6, 7), (7, 8), (8, 4), (4, 6), (6, 8), (8, 3), (3, 7), (7, 0),
    (0, 5), (5, 1), (1, 1), (1, 4), (4, 4), (4, 1), (1, 5), (5, 6), (6, 6), (2, 4),
    (4, 7), (4, 5), (5, 8), (8, 2), (2, 7), (4, 0), (0, 2), (2, 2), (2, 6), (0, 0),
    (0, 8), (3, 4), (4, 3),
]  # fmt: skip

# 62 moves: every board finished, no line of boards for anyone.
DRAWN_GAME = [
    (8, 1), (1, 1), (1, 8), (8, 5), (5, 0), (0, 0), (0, 3), (3, 7), (7, 5), (5, 6),
    (6, 0), (0, 4), (4, 7), (7, 3), (3, 6), (6, 2), (2, 3), (3, 4), (4, 4), (4, 6),
    (6, 8), (8, 6), (6, 7), (7, 0), (0, 6), (6, 4), (4, 0), (0, 5), (5, 8), (8, 4),
    (4, 3), (3, 3), (3, 0), (0, 2), (2, 5), (5, 1), (1, 0), (0, 8), (8, 2), (2, 8),
    (8, 7), (7, 6), (6, 3), (3, 1), (1, 7), (4, 1), (1, 3), (6, 5), (5, 5), (5, 4),
    (4, 5), (5, 7), (1, 4), (2, 1), (8, 3), (8, 8), (8, 0), (2, 0), (6, 1), (2, 6),
    (6, 6), (2, 7),
]  # fmt: skip


def run_self_tests() -> bool:
    import tempfile
    import unittest
    import urllib.error
    import urllib.request

    class RulesTests(unittest.TestCase):
        def test_x_moves_first_anywhere(self) -> None:
            state = initial_state()
            self.assertEqual(state.next_player, X)
            self.assertIsNone(state.required_board)

        def test_cell_sends_opponent_to_matching_board(self) -> None:
            state = replay([(4, 7)])
            self.assertEqual(state.next_player, O)
            self.assertEqual(state.required_board, 7)

        def test_wrong_board_refused(self) -> None:
            with self.assertRaises(MoveRejected) as caught:
                replay([(4, 7), (3, 0)])
            self.assertEqual(caught.exception.code, "wrong_board")

        def test_taken_cell_refused(self) -> None:
            with self.assertRaises(MoveRejected) as caught:
                replay([(4, 4), (4, 4)])
            self.assertEqual(caught.exception.code, "cell_occupied")

        def test_three_in_a_row_wins_a_small_board(self) -> None:
            state = replay(X_WINS_TOP_ROW[:5])
            self.assertEqual(state.board_statuses[0], X_WON)

        def test_sent_to_won_board_plays_anywhere(self) -> None:
            # Move 6 is O's (5, 1) ... then X's (1, 3) etc. After X wins board 0,
            # a move into cell 0 sends the opponent to a closed board.
            state = replay(X_WINS_TOP_ROW[:5] + [(5, 0)])
            self.assertIsNone(state.required_board)

        def test_closed_board_refused(self) -> None:
            state = replay(X_WINS_TOP_ROW[:5] + [(5, 0)])
            with self.assertRaises(MoveRejected) as caught:
                apply_move(state, 0, 0)
            self.assertEqual(caught.exception.code, "board_closed")

        def test_x_wins_the_game(self) -> None:
            state = replay(X_WINS_TOP_ROW)
            self.assertEqual(state.status, X_WON)
            self.assertIsNone(state.next_player)
            self.assertEqual(state.board_statuses[3], O_WON)

        def test_o_wins_the_game(self) -> None:
            self.assertEqual(replay(O_WINS).status, O_WON)

        def test_no_moves_after_the_game_ends(self) -> None:
            with self.assertRaises(MoveRejected) as caught:
                apply_move(replay(X_WINS_TOP_ROW), 8, 8)
            self.assertEqual(caught.exception.code, "game_over")

        def test_sent_to_drawn_board_plays_anywhere(self) -> None:
            state = replay(SENT_TO_DRAWN_BOARD)
            self.assertEqual(state.board_statuses[5], DRAW)
            self.assertIsNone(state.required_board)

        def test_drawn_board_is_not_part_of_a_line(self) -> None:
            state = replay(X_DRAW_X_DIAGONAL)
            self.assertEqual(
                (state.board_statuses[2], state.board_statuses[4], state.board_statuses[6]),
                (X_WON, DRAW, X_WON),
            )
            self.assertEqual(state.status, IN_PROGRESS)

        def test_game_with_no_moves_left_is_a_draw(self) -> None:
            state = replay(DRAWN_GAME)
            self.assertEqual(state.status, DRAW)
            self.assertIsNone(state.next_player)

    class ApiTests(unittest.TestCase):
        def setUp(self) -> None:
            self.tmp = tempfile.TemporaryDirectory()
            self.server = make_server(str(Path(self.tmp.name) / "test.db"), "127.0.0.1", 0, quiet=True)
            self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

        def tearDown(self) -> None:
            self.server.shutdown()
            self.server.server_close()
            self.tmp.cleanup()

        def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any, Any]:
            data = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(self.base + path, data=data, method=method)
            if data is not None:
                request.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(request) as response:
                    raw, status, headers = response.read(), response.status, response.headers
            except urllib.error.HTTPError as error:
                raw, status, headers = error.read(), error.code, error.headers
            is_json = (headers.get("Content-Type") or "").startswith("application/json")
            return status, json.loads(raw) if raw and is_json else raw.decode(), headers

        def new_game(self, x: str = "Ann", o: str = "Bob") -> dict[str, Any]:
            status, game, _ = self.call("POST", "/api/v1/games", {"player_x_name": x, "player_o_name": o})
            self.assertEqual(status, 201)
            return game

        def play(self, game_id: int, moves: list[tuple[int, int]]) -> dict[str, Any]:
            game: dict[str, Any] = {}
            for board, cell in moves:
                status, game, _ = self.call(
                    "POST", f"/api/v1/games/{game_id}/moves", {"board_index": board, "cell_index": cell}
                )
                self.assertEqual(status, 201, game)
            return game

        def test_create_game(self) -> None:
            status, game, headers = self.call("POST", "/api/v1/games", {"player_x_name": "  Ann  "})
            self.assertEqual(status, 201)
            self.assertEqual(headers["Location"], f"/api/v1/games/{game['id']}")
            self.assertEqual((game["player_x_name"], game["player_o_name"]), ("Ann", DEFAULT_O_NAME))
            self.assertEqual((game["status"], game["next_player"], game["move_count"]), (IN_PROGRESS, X, 0))
            self.assertEqual(len(game["cells"]), 9)

        def test_bad_names_refused(self) -> None:
            for name in ("   ", "x" * 26, 7, None):
                status, error, _ = self.call("POST", "/api/v1/games", {"player_x_name": name})
                self.assertEqual((status, error["code"]), (422, "validation_error"), name)
            status, error, _ = self.call("POST", "/api/v1/games", {"colour": "red"})
            self.assertEqual(status, 422)

        def test_bad_json_refused(self) -> None:
            request = urllib.request.Request(self.base + "/api/v1/games", data=b"{nope", method="POST")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 422)

        def test_list_search_sort_and_pages(self) -> None:
            for x, o in (("Ann", "Bob"), ("Cleo", "Dan"), ("Eve", "Ann")):
                self.new_game(x, o)
            status, page, _ = self.call("GET", "/api/v1/games?page_size=2")
            self.assertEqual(status, 200)
            self.assertEqual((page["total"], len(page["items"]), page["page"]), (3, 2, 1))
            _, page2, _ = self.call("GET", "/api/v1/games?page_size=2&page=2")
            self.assertEqual(len(page2["items"]), 1)
            _, found, _ = self.call("GET", "/api/v1/games?q=ann")
            self.assertEqual(found["total"], 2)
            _, oldest, _ = self.call("GET", "/api/v1/games?sort=created_at")
            self.assertEqual(oldest["items"][0]["player_x_name"], "Ann")
            for bad in ("sort=name", "page=0", "page_size=101", "page=two"):
                status, _, _ = self.call("GET", f"/api/v1/games?{bad}")
                self.assertEqual(status, 422, bad)

        def test_not_found_and_wrong_method(self) -> None:
            status, error, _ = self.call("GET", "/api/v1/games/999")
            self.assertEqual((status, error["code"]), (404, "game_not_found"))
            status, error, _ = self.call("GET", "/api/v1/nothing")
            self.assertEqual((status, error["code"]), (404, "not_found"))
            status, error, _ = self.call("DELETE", "/api/v1/games/1")
            self.assertEqual((status, error["code"]), (405, "method_not_allowed"))
            status, error, _ = self.call("GET", "/api/v1/games/abc")
            self.assertEqual(status, 422)

        def test_rename_one_player(self) -> None:
            game = self.new_game()
            status, renamed, _ = self.call("PATCH", f"/api/v1/games/{game['id']}", {"player_o_name": "Bea"})
            self.assertEqual(status, 200)
            self.assertEqual((renamed["player_x_name"], renamed["player_o_name"]), ("Ann", "Bea"))
            status, _, _ = self.call("PATCH", f"/api/v1/games/{game['id']}", {"player_o_name": None})
            self.assertEqual(status, 422)

        def test_delete_hides_the_game(self) -> None:
            game = self.new_game()
            status, body, _ = self.call("POST", f"/api/v1/games/{game['id']}/archive")
            self.assertEqual((status, body), (204, ""))
            self.assertEqual(self.call("GET", f"/api/v1/games/{game['id']}")[0], 404)
            self.assertEqual(self.call("GET", "/api/v1/games")[1]["total"], 0)

        def test_moves_and_refusals(self) -> None:
            game = self.new_game()
            path = f"/api/v1/games/{game['id']}/moves"
            status, after, headers = self.call("POST", path, {"board_index": 4, "cell_index": 7})
            self.assertEqual(status, 201)
            self.assertEqual(headers["Location"], f"/api/v1/games/{game['id']}")
            self.assertEqual((after["next_player"], after["required_board"]), (O, 7))
            self.assertEqual(after["moves"][0]["player"], X)
            status, error, _ = self.call("POST", path, {"board_index": 3, "cell_index": 0})
            self.assertEqual((status, error["code"]), (409, "wrong_board"))
            for bad in ({"board_index": 9, "cell_index": 0}, {"board_index": True, "cell_index": 0}, {}):
                self.assertEqual(self.call("POST", path, bad)[0], 422, bad)

        def test_finished_game_and_review(self) -> None:
            game = self.new_game()
            done = self.play(game["id"], X_WINS_TOP_ROW)
            self.assertEqual(done["status"], X_WON)
            status, error, _ = self.call(
                "POST", f"/api/v1/games/{game['id']}/moves", {"board_index": 8, "cell_index": 8}
            )
            self.assertEqual((status, error["code"]), (409, "game_over"))
            _, early, _ = self.call("GET", f"/api/v1/games/{game['id']}?at_move=5")
            self.assertEqual((early["move_count"], early["status"], len(early["moves"])), (5, IN_PROGRESS, 5))
            self.assertEqual(early["board_statuses"][0], X_WON)
            _, start, _ = self.call("GET", f"/api/v1/games/{game['id']}?at_move=0")
            self.assertEqual((start["move_count"], start["next_player"]), (0, X))
            status, error, _ = self.call("GET", f"/api/v1/games/{game['id']}?at_move=99")
            self.assertEqual((status, error["code"]), (404, "move_not_found"))

        def test_rematch_swaps_letters_and_keeps_score(self) -> None:
            game = self.new_game()
            status, error, _ = self.call("POST", f"/api/v1/games/{game['id']}/rematch")
            self.assertEqual((status, error["code"]), (409, "game_not_over"))
            self.play(game["id"], X_WINS_TOP_ROW)  # Ann (X) wins
            status, second, headers = self.call("POST", f"/api/v1/games/{game['id']}/rematch")
            self.assertEqual(status, 201)
            self.assertEqual(headers["Location"], f"/api/v1/games/{second['id']}")
            self.assertEqual((second["player_x_name"], second["player_o_name"]), ("Bob", "Ann"))
            series = second["record"]["series"]
            self.assertEqual((series["player_x_wins"], series["player_o_wins"], series["draws"]), (0, 1, 0))
            self.assertEqual(second["record"]["games_in_series"], 2)

            self.play(second["id"], X_WINS_TOP_ROW)  # Bob (X) wins
            _, third, _ = self.call("POST", f"/api/v1/games/{second['id']}/rematch")
            self.assertEqual((third["player_x_name"], third["player_o_name"]), ("Ann", "Bob"))
            self.assertEqual(third["record"]["series"]["player_x_wins"], 1)
            self.assertEqual(third["record"]["series"]["player_o_wins"], 1)
            self.assertEqual(third["record"]["games_in_series"], 3)

            # A separate game between the same players (any letter case) counts head to head.
            other = self.new_game("bob", "ANN")
            self.play(other["id"], DRAWN_GAME)
            _, now, _ = self.call("GET", f"/api/v1/games/{third['id']}")
            self.assertEqual(now["record"]["head_to_head"]["draws"], 1)
            self.assertEqual(now["record"]["head_to_head"]["games_played"], 3)
            self.assertEqual(now["record"]["series"]["games_played"], 2)

            # Deleted games no longer count.
            self.call("POST", f"/api/v1/games/{game['id']}/archive")
            _, now, _ = self.call("GET", f"/api/v1/games/{third['id']}")
            self.assertEqual(now["record"]["series"]["player_x_wins"], 0)
            self.assertEqual(now["record"]["series"]["player_o_wins"], 1)
            self.assertEqual(now["record"]["games_in_series"], 2)
            _, listed, _ = self.call("GET", "/api/v1/games")
            self.assertTrue(all("record" in item and "cells" not in item for item in listed["items"]))

        def test_any_other_address_is_the_web_page(self) -> None:
            for path in ("/", "/rules", "/games/12?move=3"):
                status, page, headers = self.call("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith("text/html"))
                self.assertIn("Ultimate Tic-Tac-Toe", page)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for case in (RulesTests, ApiTests):
        suite.addTests(loader.loadTestsFromTestCase(case))
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


# =============================================================================
# 7. main()
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Ultimate Tic-Tac-Toe in your browser.")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default this computer only)")
    parser.add_argument("--lan", action="store_true", help="let other devices on your network join")
    parser.add_argument(
        "--db", default=str(Path(__file__).with_name("ult_ttt.db")), help="saved-games file (default ult_ttt.db)"
    )
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    parser.add_argument("--test", action="store_true", help="run the built-in checks and exit")
    args = parser.parse_args(argv)

    if args.test:
        return 0 if run_self_tests() else 1

    host = "0.0.0.0" if args.lan else args.host
    try:
        server = make_server(args.db, host, args.port)
    except OSError as error:
        print(f"Could not start on port {args.port}: {error.strerror}. Try --port 9000.", file=sys.stderr)
        return 1

    url = f"http://localhost:{args.port}/"
    print(f"Ultimate Tic-Tac-Toe is running at {url}")
    if args.lan:
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("192.0.2.1", 80))  # picks the network card; sends nothing
                print(f"Others on your network can open http://{probe.getsockname()[0]}:{args.port}/")
        except OSError:
            print("Others on your network can use this computer's IP address and the same port.")
    print(f"Saved games: {args.db}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
